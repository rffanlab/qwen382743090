#!/usr/bin/env python3
"""Convert GGUF into Q38PACK v1 or RTX-3090-specialized Q38PACK v2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, BinaryIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q38pack_format import (  # noqa: E402
    DEFAULT_ALIGNMENT,
    ENTRY_BYTES,
    HEADER_BYTES,
    LAYOUT_GGUF_NATIVE,
    LAYOUT_SM86_Q5K_SOA,
    PackHeader,
    TensorEntry,
    align_up,
)

GGML_TYPE_Q5_K = 13
QK_K = 256
Q5_K_BYTES = 176
SM86_META_BYTES = 20
SM86_QH_BYTES = 32
SM86_QS_BYTES = 128

GGUF_TYPES = {
    0: ("uint8", "<B"),
    1: ("int8", "<b"),
    2: ("uint16", "<H"),
    3: ("int16", "<h"),
    4: ("uint32", "<I"),
    5: ("int32", "<i"),
    6: ("float32", "<f"),
    7: ("bool", "<?"),
    8: ("string", None),
    9: ("array", None),
    10: ("uint64", "<Q"),
    11: ("int64", "<q"),
    12: ("float64", "<d"),
}


@dataclass(slots=True)
class GGUFTensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int
    stored_bytes: int = 0


@dataclass(slots=True)
class GGUFInfo:
    version: int
    tensor_count: int
    kv_count: int
    alignment: int
    data_offset: int
    metadata: dict[str, Any]
    tensors: list[GGUFTensor]


class Reader:
    def __init__(self, fp: BinaryIO):
        self.fp = fp

    def read_exact(self, n: int) -> bytes:
        b = self.fp.read(n)
        if len(b) != n:
            raise ValueError("unexpected EOF while reading GGUF")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read_exact(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > (1 << 32):
            raise ValueError(f"unreasonable GGUF string length: {n}")
        return self.read_exact(n).decode("utf-8")

    def value(self, type_id: int, keep_array_values: bool = False) -> Any:
        if type_id not in GGUF_TYPES:
            raise ValueError(f"unknown GGUF value type {type_id}")
        name, fmt = GGUF_TYPES[type_id]
        if fmt:
            return struct.unpack(fmt, self.read_exact(struct.calcsize(fmt)))[0]
        if name == "string":
            return self.string()
        if name == "array":
            elem_type = self.u32()
            count = self.u64()
            if elem_type == 9:
                raise ValueError("nested GGUF arrays are invalid")
            if keep_array_values and count <= 4096:
                return [self.value(elem_type, keep_array_values=True) for _ in range(count)]
            elem_name, elem_fmt = GGUF_TYPES.get(elem_type, (str(elem_type), None))
            if elem_fmt is not None:
                self.fp.seek(struct.calcsize(elem_fmt) * count, os.SEEK_CUR)
            else:
                for _ in range(count):
                    self.value(elem_type, keep_array_values=False)
            return {"array_type": elem_name, "count": count}
        raise AssertionError(name)


def parse_gguf(path: Path) -> GGUFInfo:
    file_size = path.stat().st_size
    with path.open("rb") as fp:
        r = Reader(fp)
        if r.read_exact(4) != b"GGUF":
            raise ValueError("input is not a GGUF file")
        version = r.u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count = r.u64()
        kv_count = r.u64()
        metadata: dict[str, Any] = {}
        alignment = 32
        for _ in range(kv_count):
            key = r.string()
            type_id = r.u32()
            value = r.value(type_id, keep_array_values=False)
            metadata[key] = value
            if key == "general.alignment":
                alignment = int(value)

        tensors: list[GGUFTensor] = []
        for _ in range(tensor_count):
            name = r.string()
            ndim = r.u32()
            if ndim > 4:
                raise ValueError(f"tensor {name!r} has {ndim} dimensions; Q38PACK supports <= 4")
            dims = tuple(r.u64() for _ in range(ndim))
            ggml_type = r.u32()
            offset = r.u64()
            tensors.append(GGUFTensor(name, dims, ggml_type, offset))

        data_offset = align_up(fp.tell(), alignment)

    ordered = sorted(tensors, key=lambda t: t.offset)
    data_bytes = file_size - data_offset
    if data_bytes < 0:
        raise ValueError("GGUF data offset is beyond EOF")
    for i, tensor in enumerate(ordered):
        next_offset = ordered[i + 1].offset if i + 1 < len(ordered) else data_bytes
        if tensor.offset > next_offset or next_offset > data_bytes:
            raise ValueError(f"invalid tensor offset span around {tensor.name!r}")
        tensor.stored_bytes = next_offset - tensor.offset

    return GGUFInfo(
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        alignment=alignment,
        data_offset=data_offset,
        metadata=metadata,
        tensors=tensors,
    )


def tensor_role(name: str) -> int:
    n = name.lower()
    if n.startswith("token_embd") or "embed_tokens" in n:
        return 1
    if n.startswith("output_norm"):
        return 2
    if n == "output.weight" or "lm_head" in n:
        return 3
    if "mtp" in n or "nextn" in n:
        return 8
    if "ssm_" in n or ".ssm" in n or "attn_gate" in n or "gated_delta" in n:
        return 5
    if ".ffn_" in n or "mlp" in n:
        return 6
    if "norm" in n:
        return 7
    if ".attn_" in n:
        return 4
    return 0


def selected_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, dict) and "count" in value:
            out[key] = value
        elif isinstance(value, (str, int, float, bool)):
            out[key] = value
    return out


def copy_exact(src: BinaryIO, dst: BinaryIO, count: int, chunk_size: int = 16 << 20) -> None:
    remaining = count
    while remaining:
        chunk = src.read(min(chunk_size, remaining))
        if not chunk:
            raise ValueError("unexpected EOF while copying tensor payload")
        dst.write(chunk)
        remaining -= len(chunk)


def pad_to(fp: BinaryIO, offset: int) -> None:
    here = fp.tell()
    if here > offset:
        raise ValueError("internal layout overlap")
    if here < offset:
        fp.write(bytes(offset - here))


def q5_block_count(tensor: GGUFTensor) -> int:
    elements = math.prod(tensor.dims)
    if elements % QK_K:
        raise ValueError(f"Q5_K tensor {tensor.name!r} has element count not divisible by 256")
    blocks = elements // QK_K
    expected = blocks * Q5_K_BYTES
    if expected > tensor.stored_bytes:
        raise ValueError(
            f"Q5_K tensor {tensor.name!r} needs {expected} bytes but GGUF span has {tensor.stored_bytes}"
        )
    return blocks


def sm86_q5_layout(start: int, blocks: int) -> tuple[int, int, int, int]:
    meta = start
    qh = align_up(meta + blocks * SM86_META_BYTES, 128)
    qs = align_up(qh + blocks * SM86_QH_BYTES, 128)
    end = qs + blocks * SM86_QS_BYTES
    return meta, qh, qs, end


def require_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except ImportError as exc:
        raise RuntimeError(
            "Q38PACK v2 SM86 Q5_K repack requires NumPy. "
            "Install it with: python3 -m pip install numpy ; "
            "or use --format-version 1."
        ) from exc


def repack_q5_sm86(
    src: BinaryIO,
    dst: BinaryIO,
    source_offset: int,
    blocks: int,
    meta_offset: int,
    qh_offset: int,
    qs_offset: int,
    chunk_blocks: int = 65536,
) -> None:
    np = require_numpy()

    for block0 in range(0, blocks, chunk_blocks):
        n = min(chunk_blocks, blocks - block0)
        src.seek(source_offset + block0 * Q5_K_BYTES)
        raw = src.read(n * Q5_K_BYTES)
        if len(raw) != n * Q5_K_BYTES:
            raise ValueError("unexpected EOF while reading Q5_K tensor")

        a = np.frombuffer(raw, dtype=np.uint8).reshape(n, Q5_K_BYTES)
        meta = np.empty((n, SM86_META_BYTES), dtype=np.uint8)
        meta[:, 0:4] = a[:, 0:4]
        s = a[:, 4:16]

        for g in range(8):
            if g < 4:
                sc = s[:, g] & 63
                mn = s[:, g + 4] & 63
            else:
                sc = (s[:, g + 4] & 15) | ((s[:, g - 4] >> 6) << 4)
                mn = (s[:, g + 4] >> 4) | ((s[:, g] >> 6) << 4)
            meta[:, 4 + 2 * g] = sc
            meta[:, 5 + 2 * g] = mn

        dst.seek(meta_offset + block0 * SM86_META_BYTES)
        dst.write(meta.tobytes(order="C"))
        dst.seek(qh_offset + block0 * SM86_QH_BYTES)
        dst.write(a[:, 16:48].tobytes(order="C"))
        dst.seek(qs_offset + block0 * SM86_QS_BYTES)
        dst.write(a[:, 48:176].tobytes(order="C"))


def convert(
    source: Path,
    destination: Path,
    alignment: int,
    with_sha256: bool,
    pack_version: int = 2,
) -> dict[str, Any]:
    if pack_version not in (1, 2):
        raise ValueError("pack_version must be 1 or 2")
    info = parse_gguf(source)
    if alignment < 256 or alignment & (alignment - 1):
        raise ValueError("--alignment must be a power of two >= 256")

    arch = info.metadata.get("general.architecture")
    specialized_q5 = sum(1 for t in info.tensors if pack_version >= 2 and t.ggml_type == GGML_TYPE_Q5_K)

    manifest: dict[str, Any] = {
        "format": "q38pack",
        "version": pack_version,
        "layout": "sm86-v2" if pack_version >= 2 else "gguf-native-v1",
        "target": {"model": "Qwen3.8-27B", "gpu": "RTX 3090", "sm": 86},
        "source": {
            "filename": source.name,
            "file_size": source.stat().st_size,
            "gguf_version": info.version,
            "gguf_alignment": info.alignment,
            "gguf_data_offset": info.data_offset,
        },
        "model": {
            "architecture": arch,
            "name": info.metadata.get("general.name"),
            "tensor_count": info.tensor_count,
        },
        "specialized_layouts": {
            "sm86_q5k_soa_tensors": specialized_q5,
        },
        "metadata": selected_metadata(info.metadata),
        "notes": [
            "raw_gguf_meta preserves the original GGUF header/metadata/tensor directory.",
            (
                "Q5_K tensors are repacked losslessly to META20/QH32/QS128 SoA for SM86."
                if pack_version >= 2
                else "Tensor payloads are copied byte-for-byte from GGUF in Q38PACK v1."
            ),
        ],
    }

    if arch not in (None, "qwen35", "qwen3.5", "qwen3_5"):
        manifest["warnings"] = [f"unexpected architecture {arch!r}; converter did not reject it"]

    if with_sha256:
        h = hashlib.sha256()
        with source.open("rb") as hash_fp:
            while True:
                chunk = hash_fp.read(16 << 20)
                if not chunk:
                    break
                h.update(chunk)
        manifest["source"]["sha256"] = h.hexdigest()

    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    directory_offset = HEADER_BYTES
    directory_bytes = info.tensor_count * ENTRY_BYTES
    manifest_offset = align_up(directory_offset + directory_bytes, 64)
    raw_meta_offset = align_up(manifest_offset + len(manifest_bytes), 64)
    raw_meta_bytes = info.data_offset
    data_offset = align_up(raw_meta_offset + raw_meta_bytes, alignment)

    output_entries: list[TensorEntry] = []
    cursor = data_offset
    for t in info.tensors:
        cursor = align_up(cursor, alignment)
        dims = tuple(t.dims) + (0,) * (4 - len(t.dims))

        if pack_version >= 2 and t.ggml_type == GGML_TYPE_Q5_K:
            blocks = q5_block_count(t)
            meta, qh, qs, end = sm86_q5_layout(cursor, blocks)
            output_entries.append(
                TensorEntry(
                    name=t.name,
                    ndim=len(t.dims),
                    ggml_type=t.ggml_type,
                    dims=dims,
                    data_offset=meta,
                    stored_bytes=end - meta,
                    source_offset=info.data_offset + t.offset,
                    role=tensor_role(t.name),
                    layout=LAYOUT_SM86_Q5K_SOA,
                    aux0_offset=qh,
                    aux1_offset=qs,
                )
            )
            cursor = end
        else:
            output_entries.append(
                TensorEntry(
                    name=t.name,
                    ndim=len(t.dims),
                    ggml_type=t.ggml_type,
                    dims=dims,
                    data_offset=cursor,
                    stored_bytes=t.stored_bytes,
                    source_offset=info.data_offset + t.offset,
                    role=tensor_role(t.name),
                    layout=LAYOUT_GGUF_NATIVE,
                )
            )
            cursor += t.stored_bytes

    data_end = cursor
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")

    with source.open("rb") as src, tmp.open("wb+") as dst:
        dst.write(bytes(HEADER_BYTES))
        pad_to(dst, directory_offset)
        for entry in output_entries:
            dst.write(entry.encode(pack_version))
        pad_to(dst, manifest_offset)
        dst.write(manifest_bytes)
        pad_to(dst, raw_meta_offset)

        src.seek(0)
        raw_meta = src.read(raw_meta_bytes)
        if len(raw_meta) != raw_meta_bytes:
            raise ValueError("failed to read original GGUF metadata region")
        dst.write(raw_meta)

        q5_done = 0
        for source_tensor, output_entry in zip(info.tensors, output_entries):
            if output_entry.layout == LAYOUT_SM86_Q5K_SOA:
                q5_done += 1
                if q5_done == 1 or q5_done % 32 == 0 or q5_done == specialized_q5:
                    print(
                        f"repacking SM86 Q5_K tensors: {q5_done}/{specialized_q5}",
                        file=sys.stderr,
                        flush=True,
                    )
                blocks = q5_block_count(source_tensor)
                repack_q5_sm86(
                    src,
                    dst,
                    info.data_offset + source_tensor.offset,
                    blocks,
                    output_entry.data_offset,
                    output_entry.aux0_offset,
                    output_entry.aux1_offset,
                )
            else:
                dst.seek(output_entry.data_offset)
                src.seek(info.data_offset + source_tensor.offset)
                copy_exact(src, dst, source_tensor.stored_bytes)

        pad_to(dst, data_end)

        header = PackHeader(
            version=pack_version,
            flags=1,
            tensor_count=info.tensor_count,
            alignment=alignment,
            directory_offset=directory_offset,
            directory_bytes=directory_bytes,
            manifest_offset=manifest_offset,
            manifest_bytes=len(manifest_bytes),
            raw_gguf_meta_offset=raw_meta_offset,
            raw_gguf_meta_bytes=raw_meta_bytes,
            data_offset=data_offset,
            data_bytes=data_end - data_offset,
            source_file_size=source.stat().st_size,
            source_data_offset=info.data_offset,
        )
        dst.seek(0)
        dst.write(header.encode())
        dst.flush()
        os.fsync(dst.fileno())

    os.replace(tmp, destination)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert GGUF to Q38PACK")
    ap.add_argument("source", type=Path, help="input .gguf")
    ap.add_argument("destination", type=Path, help="output .q38pack")
    ap.add_argument("--format-version", type=int, choices=(1, 2), default=2,
                    help="Q38PACK version (default: 2 / SM86-specialized Q5_K)")
    ap.add_argument("--alignment", type=int, default=DEFAULT_ALIGNMENT,
                    help="tensor alignment (default: 4096)")
    ap.add_argument("--sha256", action="store_true",
                    help="also compute the source GGUF SHA-256 (extra full read)")
    args = ap.parse_args()

    manifest = convert(
        args.source, args.destination, args.alignment, args.sha256, args.format_version
    )
    print(f"wrote {args.destination}")
    print(f"q38pack_version: {manifest['version']}")
    print(f"architecture: {manifest['model'].get('architecture')}")
    print(f"tensors: {manifest['model']['tensor_count']}")
    print(
        "sm86_q5k_soa_tensors: "
        f"{manifest['specialized_layouts']['sm86_q5k_soa_tensors']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
