#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q38pack_format import PackReader  # noqa: E402


def gib(n: int) -> float:
    return n / 1024 / 1024 / 1024


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect a Q38PACK file")
    ap.add_argument("model", type=Path)
    ap.add_argument("--tensors", action="store_true")
    args = ap.parse_args()

    with PackReader(args.model) as pack:
        h = pack.header
        assert h is not None
        print(f"Q38PACK v1: {args.model}")
        print(f"architecture: {pack.manifest.get('model', {}).get('architecture')}")
        print(f"tensor_count: {h.tensor_count}")
        print(f"alignment: {h.alignment}")
        print(f"tensor_data: {gib(h.data_bytes):.3f} GiB")
        print(f"source_file: {gib(h.source_file_size):.3f} GiB")
        if args.tensors:
            for t in pack.tensors:
                dims = "x".join(str(x) for x in t.dims[:t.ndim])
                print(f"{t.name}\ttype={t.ggml_type}\tdims={dims}\tbytes={t.stored_bytes}\trole={t.role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
