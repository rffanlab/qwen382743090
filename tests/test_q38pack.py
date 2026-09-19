from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from gguf_to_q38pack import convert  # noqa: E402
from q38pack_format import (  # noqa: E402
    LAYOUT_GGUF_NATIVE,
    LAYOUT_SM86_Q5K_SOA,
    PackReader,
    align_up,
)


def gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def common_kv() -> list[tuple[str, int, bytes]]:
    return [
        ("general.architecture", 8, gguf_string("qwen35")),
        ("general.name", 8, gguf_string("tiny-qwen38-test")),
        ("general.alignment", 4, struct.pack("<I", 32)),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 3) + gguf_string("a") + gguf_string("b") + gguf_string("c")),
    ]


def write_gguf(path: Path, tensors: list[tuple[str, tuple[int, ...], int, int]], data: bytes) -> int:
    kv = common_kv()
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, type_id, raw_value in kv:
        header += gguf_string(key)
        header += struct.pack("<I", type_id)
        header += raw_value
    for name, dims, ggml_type, offset in tensors:
        header += gguf_string(name)
        header += struct.pack("<I", len(dims))
        for d in dims:
            header += struct.pack("<Q", d)
        header += struct.pack("<IQ", ggml_type, offset)

    data_offset = align_up(len(header), 32)
    with path.open("wb") as fp:
        fp.write(header)
        fp.write(bytes(data_offset - len(header)))
        fp.write(data)
    return data_offset


def write_tiny_gguf(path: Path) -> tuple[bytes, bytes]:
    tensors = [
        ("token_embd.weight", (4, 2), 0, 0),
        ("blk.0.attn_norm.weight", (4,), 0, 32),
    ]
    t0 = bytes(range(32))
    t1 = bytes(range(32, 64))
    write_gguf(path, tensors, t0 + t1)
    return t0, t1


def write_tiny_q5_gguf(path: Path) -> bytes:
    # One exact Q5_K super-block: 4B dm + 12B scales + 32B qh + 128B qs.
    dm = struct.pack("<ee", 0.5, 0.25)
    scales = bytes((i * 17 + 3) & 0xFF for i in range(12))
    qh = bytes((i * 11 + 5) & 0xFF for i in range(32))
    qs = bytes((i * 7 + 9) & 0xFF for i in range(128))
    block = dm + scales + qh + qs
    assert len(block) == 176
    write_gguf(path, [("blk.0.attn_gate.weight", (256,), 13, 0)], block)
    return block


def scale_min(scales: bytes, g: int) -> tuple[int, int]:
    if g < 4:
        return scales[g] & 63, scales[g + 4] & 63
    return (
        (scales[g + 4] & 0x0F) | ((scales[g - 4] >> 6) << 4),
        (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4),
    )


class Q38PackTest(unittest.TestCase):
    def test_default_v2_preserves_native_tensor_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / "tiny.gguf"
            dest = td / "tiny.q38pack"
            t0, t1 = write_tiny_gguf(source)

            manifest = convert(source, dest, alignment=4096, with_sha256=False)
            self.assertEqual(manifest["version"], 2)
            self.assertEqual(manifest["model"]["architecture"], "qwen35")

            with PackReader(dest) as pack:
                self.assertEqual(pack.header.version, 2)
                self.assertEqual(pack.header.tensor_count, 2)
                self.assertEqual(pack.manifest["model"]["name"], "tiny-qwen38-test")
                self.assertEqual(pack.manifest["metadata"]["tokenizer.ggml.tokens"]["count"], 3)
                self.assertEqual([t.layout for t in pack.tensors], [
                    LAYOUT_GGUF_NATIVE, LAYOUT_GGUF_NATIVE,
                ])
                mm = pack._mm
                self.assertIsNotNone(mm)
                e0, e1 = pack.tensors
                self.assertEqual(bytes(mm[e0.data_offset:e0.data_offset + e0.stored_bytes]), t0)
                self.assertEqual(bytes(mm[e1.data_offset:e1.data_offset + e1.stored_bytes]), t1)

    def test_v1_stays_readable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / "tiny.gguf"
            dest = td / "tiny-v1.q38pack"
            write_tiny_gguf(source)
            convert(source, dest, alignment=4096, with_sha256=False, pack_version=1)
            with PackReader(dest) as pack:
                self.assertEqual(pack.header.version, 1)
                self.assertTrue(all(t.layout == LAYOUT_GGUF_NATIVE for t in pack.tensors))

    def test_v2_q5k_repack_planes_are_lossless(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / "tiny-q5.gguf"
            dest = td / "tiny-q5.q38pack"
            block = write_tiny_q5_gguf(source)

            manifest = convert(source, dest, alignment=4096, with_sha256=False, pack_version=2)
            self.assertEqual(manifest["specialized_layouts"]["sm86_q5k_soa_tensors"], 1)

            with PackReader(dest) as pack:
                self.assertEqual(pack.header.version, 2)
                self.assertEqual(len(pack.tensors), 1)
                t = pack.tensors[0]
                self.assertEqual(t.layout, LAYOUT_SM86_Q5K_SOA)
                mm = pack._mm
                self.assertIsNotNone(mm)

                meta = bytes(mm[t.data_offset:t.data_offset + 20])
                qh = bytes(mm[t.aux0_offset:t.aux0_offset + 32])
                qs = bytes(mm[t.aux1_offset:t.aux1_offset + 128])

                self.assertEqual(meta[:4], block[:4])
                scales = block[4:16]
                expected_meta = bytearray(block[:4])
                for g in range(8):
                    sc, mn = scale_min(scales, g)
                    expected_meta += bytes((sc, mn))
                self.assertEqual(meta, bytes(expected_meta))
                self.assertEqual(qh, block[16:48])
                self.assertEqual(qs, block[48:176])


if __name__ == "__main__":
    unittest.main()
