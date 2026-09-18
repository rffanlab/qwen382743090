# qwen382743090 / Q38RT

Q38RT is intentionally narrow: make Qwen3.8-27B extremely fast on one RTX
3090 / SM86 before trying to become a general-purpose runtime.

The planned native path uses the NVIDIA Driver API directly and does not depend
on PyTorch, vLLM, the CUDA Runtime, or cuBLAS. Hot paths will use GA102-specific
PTX/CUBIN, with custom model layout, KV/recurrent-state management, VMM, and
MTP scheduling.

Current milestone: M0. The repository contains a working GGUF -> Q38PACK
converter, a C++ mmap reader/runtime probe, direct libcuda.so.1 probing, an
OpenAI-compatible HTTP shell, tests, and an optional reference backend used as
a correctness oracle.

Native Qwen3.8 decode is not falsely advertised as complete. Until the SM86
kernels pass correctness checks, chat requests in native mode return a
structured native_decode_not_ready error.

## Convert a GGUF

    python3 tools/gguf_to_q38pack.py /models/Qwen3.8-27B-Q4_K_M.gguf /models/Qwen3.8-27B-Q4_K_M.q38pack

    python3 tools/inspect_q38pack.py /models/Qwen3.8-27B-Q4_K_M.q38pack

Q38PACK v1 copies quantized tensor payloads byte-for-byte and preserves the
original GGUF metadata region. See docs/Q38PACK.md.

## Build

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

    ./build/q38-runtime --model /models/Qwen3.8-27B-Q4_K_M.q38pack

No CUDA Toolkit headers are required by the current host runtime; the NVIDIA
driver library is loaded dynamically.

## OpenAI-compatible server

    python3 server/q38_openai.py --model /models/Qwen3.8-27B-Q4_K_M.q38pack --host 0.0.0.0 --port 8000

For protocol plumbing and oracle comparisons, a temporary upstream can be used:

    python3 server/q38_openai.py --model /models/Qwen3.8-27B-Q4_K_M.q38pack --port 8000 --reference-base-url http://127.0.0.1:8080 --reference-model qwen3.8-27b

The reference path is test infrastructure, not the final runtime.

## Tests

    python3 -m unittest discover -s tests -p 'test_*.py' -v

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure

See docs/ARCHITECTURE.md for the native GPU roadmap.
