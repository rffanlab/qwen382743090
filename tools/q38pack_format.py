#!/usr/bin/env python3
"""Q38PACK v1/v2 reader/writer primitives.

v1 stores source GGUF tensor payloads as-is.
v2 keeps the same 256-byte header/directory ABI and uses the directory's
previously-reserved tail for execution-layout metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import mmap
import os
import struct
from pathlib import Path
from typing import BinaryIO, Any

MAGIC = b"Q38PACK\0"
VERSION = 2
MIN_VERSION = 1
HEADER_BYTES = 256
ENTRY_BYTES = 256
DEFAULT_ALIGNMENT = 4096

LAYOUT_GGUF_NATIVE = 0
LAYOUT_SM86_Q5K_SOA = 1

HEADER_STRUCT = struct.Struct("<8sIIIIIIQQQQQQQQQQ")
ENTRY_STRUCT_V1 = struct.Struct("<160sII4Q3QII24x")
ENTRY_STRUCT_V2 = struct.Struct("<160sII4Q3QIIIIQQ")


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & ~(alignment - 1)


@dataclass(slots=True)
class TensorEntry:
    name: str
    ndim: int
    ggml_type: int
    dims: tuple[int, int, int, int]
    data_offset: int
    stored_bytes: int
    source_offset: int
    role: int = 0
    flags: int = 0
    layout: int = LAYOUT_GGUF_NATIVE
    layout_flags: int = 0
    aux0_offset: int = 0
    aux1_offset: int = 0

    def encode(self, version: int = VERSION) -> bytes:
        raw_name = self.name.encode("utf-8")
        if len(raw_name) >= 160:
            raise ValueError(f"tensor name too long for Q38PACK: {self.name!r}")
        if version == 1:
            if self.layout != LAYOUT_GGUF_NATIVE or self.aux0_offset or self.aux1_offset:
                raise ValueError("Q38PACK v1 cannot encode specialized tensor layouts")
            return ENTRY_STRUCT_V1.pack(
                raw_name, self.ndim, self.ggml_type, *self.dims,
                self.data_offset, self.stored_bytes, self.source_offset,
                self.role, self.flags,
            )
        if version == 2:
            return ENTRY_STRUCT_V2.pack(
                raw_name, self.ndim, self.ggml_type, *self.dims,
                self.data_offset, self.stored_bytes, self.source_offset,
                self.role, self.flags, self.layout, self.layout_flags,
                self.aux0_offset, self.aux1_offset,
            )
        raise ValueError(f"unsupported Q38PACK version {version}")

    @classmethod
    def decode(cls, raw: bytes, version: int) -> "TensorEntry":
        if len(raw) != ENTRY_BYTES:
            raise ValueError("bad tensor entry size")
        if version == 1:
            values = ENTRY_STRUCT_V1.unpack(raw)
            name = values[0].split(b"\0", 1)[0].decode("utf-8")
            return cls(
                name=name, ndim=values[1], ggml_type=values[2],
                dims=tuple(values[3:7]), data_offset=values[7],
                stored_bytes=values[8], source_offset=values[9],
                role=values[10], flags=values[11],
            )
        if version == 2:
            values = ENTRY_STRUCT_V2.unpack(raw)
            name = values[0].split(b"\0", 1)[0].decode("utf-8")
            return cls(
                name=name, ndim=values[1], ggml_type=values[2],
                dims=tuple(values[3:7]), data_offset=values[7],
                stored_bytes=values[8], source_offset=values[9],
                role=values[10], flags=values[11],
                layout=values[12], layout_flags=values[13],
                aux0_offset=values[14], aux1_offset=values[15],
            )
        raise ValueError(f"unsupported Q38PACK version {version}")


@dataclass(slots=True)
class PackHeader:
    version: int
    flags: int
    tensor_count: int
    alignment: int
    directory_offset: int
    directory_bytes: int
    manifest_offset: int
    manifest_bytes: int
    raw_gguf_meta_offset: int
    raw_gguf_meta_bytes: int
    data_offset: int
    data_bytes: int
    source_file_size: int
    source_data_offset: int

    def encode(self) -> bytes:
        if not (MIN_VERSION <= self.version <= VERSION):
            raise ValueError(f"unsupported Q38PACK version {self.version}")
        fixed = HEADER_STRUCT.pack(
            MAGIC, self.version, HEADER_BYTES, self.flags, self.tensor_count,
            self.alignment, 0, self.directory_offset, self.directory_bytes,
            self.manifest_offset, self.manifest_bytes,
            self.raw_gguf_meta_offset, self.raw_gguf_meta_bytes,
            self.data_offset, self.data_bytes,
            self.source_file_size, self.source_data_offset,
        )
        return fixed + bytes(HEADER_BYTES - len(fixed))

    @classmethod
    def decode(cls, raw: bytes) -> "PackHeader":
        if len(raw) < HEADER_BYTES:
            raise ValueError("file is too small to be Q38PACK")
        values = HEADER_STRUCT.unpack(raw[: HEADER_STRUCT.size])
        if values[0] != MAGIC:
            raise ValueError("not a Q38PACK file")
        if not (MIN_VERSION <= values[1] <= VERSION) or values[2] != HEADER_BYTES:
            raise ValueError(f"unsupported Q38PACK version/header: {values[1]}/{values[2]}")
        return cls(
            version=values[1], flags=values[3], tensor_count=values[4],
            alignment=values[5], directory_offset=values[7],
            directory_bytes=values[8], manifest_offset=values[9],
            manifest_bytes=values[10], raw_gguf_meta_offset=values[11],
            raw_gguf_meta_bytes=values[12], data_offset=values[13],
            data_bytes=values[14], source_file_size=values[15],
            source_data_offset=values[16],
        )


class PackReader:
    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path)
        self._fp: BinaryIO | None = None
        self._mm: mmap.mmap | None = None
        self.header: PackHeader | None = None
        self.tensors: list[TensorEntry] = []
        self.manifest: dict[str, Any] = {}

    def __enter__(self) -> "PackReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        self.close()
        self._fp = self.path.open("rb")
        self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        self.header = PackHeader.decode(self._mm[:HEADER_BYTES])
        h = self.header
        if h.directory_bytes != h.tensor_count * ENTRY_BYTES:
            raise ValueError("malformed Q38PACK tensor directory")
        size = len(self._mm)
        for off, length, label in (
            (h.directory_offset, h.directory_bytes, "directory"),
            (h.manifest_offset, h.manifest_bytes, "manifest"),
            (h.raw_gguf_meta_offset, h.raw_gguf_meta_bytes, "raw GGUF metadata"),
            (h.data_offset, h.data_bytes, "data"),
        ):
            if off < 0 or length < 0 or off + length > size:
                raise ValueError(f"Q38PACK {label} is out of bounds")

        self.tensors = []
        for i in range(h.tensor_count):
            p = h.directory_offset + i * ENTRY_BYTES
            entry = TensorEntry.decode(self._mm[p : p + ENTRY_BYTES], h.version)
            if entry.data_offset + entry.stored_bytes > size:
                raise ValueError(f"tensor {entry.name!r} is out of bounds")
            if entry.layout == LAYOUT_SM86_Q5K_SOA:
                if not (entry.data_offset <= entry.aux0_offset <= entry.aux1_offset <
                        entry.data_offset + entry.stored_bytes):
                    raise ValueError(f"tensor {entry.name!r} has invalid SM86 plane offsets")
            self.tensors.append(entry)

        raw_manifest = self._mm[h.manifest_offset : h.manifest_offset + h.manifest_bytes]
        self.manifest = json.loads(raw_manifest.decode("utf-8")) if raw_manifest else {}

    def close(self) -> None:
        self.tensors = []
        self.manifest = {}
        self.header = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fp is not None:
            self._fp.close()
            self._fp = None
