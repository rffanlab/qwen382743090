#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q38pack_format import (  # noqa: E402
    LAYOUT_GGUF_NATIVE,
    LAYOUT_SM86_Q5K_SOA,
    PackReader,
)


def gib(n: int) -> float:
    return n / 1024 / 1024 / 1024


def layout_name(layout: int) -> str:
    return {
        LAYOUT_GGUF_NATIVE: "GGUF_NATIVE",
        LAYOUT_SM86_Q5K_SOA: "SM86_Q5K_SOA",
    }.get(layout, f"UNKNOWN_{layout}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect a Q38PACK file")
    ap.add_argument("model", type=Path)
    ap.add_argument("--tensors", action="store_true")
    args = ap.parse_args()

    with PackReader(args.model) as pack:
        h = pack.header
        assert h is not None
        print(f"Q38PACK v{h.version}: {args.model}")
        print(f"architecture: {pack.manifest.get('model', {}).get('architecture')}")
        print(f"layout: {pack.manifest.get('layout')}")
        print(f"tensor_count: {h.tensor_count}")
        print(f"alignment: {h.alignment}")
        print(f"tensor_data: {gib(h.data_bytes):.3f} GiB")
        print(f"source_file: {gib(h.source_file_size):.3f} GiB")

        layouts: dict[int, list[int]] = {}
        for t in pack.tensors:
            slot = layouts.setdefault(t.layout, [0, 0])
            slot[0] += 1
            slot[1] += t.stored_bytes
        print("execution_layouts:")
        for layout, (count, nbytes) in sorted(layouts.items()):
            print(f"  {layout_name(layout)}: {count} tensors, {gib(nbytes):.3f} GiB")

        if args.tensors:
            for t in pack.tensors:
                dims = "x".join(str(x) for x in t.dims[:t.ndim])
                extra = ""
                if t.layout == LAYOUT_SM86_Q5K_SOA:
                    extra = f"\tqh=0x{t.aux0_offset:x}\tqs=0x{t.aux1_offset:x}"
                print(
                    f"{t.name}\ttype={t.ggml_type}\tdims={dims}"
                    f"\tbytes={t.stored_bytes}\trole={t.role}"
                    f"\tlayout={layout_name(t.layout)}{extra}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
