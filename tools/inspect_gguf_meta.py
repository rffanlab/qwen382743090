#!/usr/bin/env python3
"""Print Qwen3.8/Qwen35 GGUF metadata needed by the native attention runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_to_q38pack import parse_gguf  # noqa: E402


def relevant(key: str) -> bool:
    k = key.lower()
    needles = (
        "architecture",
        "block_count",
        "embedding_length",
        "context_length",
        "attention",
        "rope",
        "head_count",
        "key_length",
        "value_length",
        "layer_norm_rms_epsilon",
    )
    return any(x in k for x in needles)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    args = ap.parse_args()

    info = parse_gguf(args.model, keep_metadata_arrays=True)
    selected = {
        k: v
        for k, v in sorted(info.metadata.items())
        if relevant(k)
    }

    print("Q38RT GGUF attention metadata")
    print(f"file: {args.model}")
    print(f"gguf_version: {info.version}")
    print(f"tensor_count: {info.tensor_count}")
    print(f"kv_count: {info.kv_count}")
    print(json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
