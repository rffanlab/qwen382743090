from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from gguf_to_q38pack import convert  # noqa: E402
from q38pack_format import PackReader, align_up  # noqa: E402


def gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_tiny_gguf(path: Path) -> tuple[bytes, bytes]:
    kv = [
        ("general.architecture", 8, gguf_string("qwen35")),
        ("general.name", 8, gguf_string("tiny-qwen38-test")),
        ("general.alignment", 4, struct.pack("<I", 32)),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 3) + gguf_string("a") + gguf_string("b") + gguf_string("c")),
    ]
    tensors = [
        ("token_embd.weight", (4, 2), 0, 0),
        ("blk.0.attn_norm.weight", (4,), 0, 32),
    ]

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
    t0 = bytes(range(32))
    t1 = bytes(range(32, 64))
    with path.open("wb") as fp:
        fp.write(header)
        fp.write(bytes(data_offset - len(header)))
        fp.write(t0)
        fp.write(t1)
    return t0, t1


class Q38PackTest(unittest.TestCase):
    def test_convert_preserves_tensor_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / "tiny.gguf"
            dest = td / "tiny.q38pack"
            t0, t1 = write_tiny_gguf(source)

            manifest = convert(source, dest, alignment=4096, with_sha256=False)
            self.assertEqual(manifest["model"]["architecture"], "qwen35")

            with PackReader(dest) as pack:
                self.assertEqual(pack.header.tensor_count, 2)
                self.assertEqual(pack.manifest["model"]["name"], "tiny-qwen38-test")
                self.assertEqual(pack.manifest["metadata"]["tokenizer.ggml.tokens"]["count"], 3)
                self.assertEqual([t.name for t in pack.tensors], [
                    "token_embd.weight",
                    "blk.0.attn_norm.weight",
                ])
                mm = pack._mm
                self.assertIsNotNone(mm)
                e0, e1 = pack.tensors
                self.assertEqual(bytes(mm[e0.data_offset:e0.data_offset + e0.stored_bytes]), t0)
                self.assertEqual(bytes(mm[e1.data_offset:e1.data_offset + e1.stored_bytes]), t1)


if __name__ == "__main__":
    unittest.main()
