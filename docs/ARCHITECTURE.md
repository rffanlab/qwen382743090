# Q38RT architecture

## Scope

Q38RT is not intended to be a general LLM runtime. The first supported target
is Qwen3.8-27B on NVIDIA GA102 / RTX 3090 (SM86), single-user agent inference.

The runtime deliberately avoids PyTorch, vLLM, CUDA Runtime, and cuBLAS in the
native path. Host-side GPU access goes through the NVIDIA Driver API loaded from
libcuda.so.1. Hot kernels will be shipped as SM86-specialized PTX/CUBIN.

## Layers

    OpenAI-compatible HTTP
              |
              v
    scheduler / tokenizer / sampler / MTP controller
              |
              v
    Qwen3.8 fixed execution plan
              |
              +---- KV / recurrent-state manager
              |
              +---- SM86 kernel registry
              |
              v
    NVIDIA Driver API (libcuda)
              |
              v
    GA102

## Milestones

### M0 - container and correctness plumbing

- GGUF -> Q38PACK conversion.
- mmap Q38PACK reader in C++.
- lossless source tensor payload preservation.
- OpenAI protocol surface.
- optional reference backend for llama.cpp comparison.

### M1 - native device bring-up

- Driver API context creation without CUDA Runtime.
- VMM allocator using cuMemAddressReserve, cuMemCreate and cuMemMap.
- module loading from PTX/CUBIN.
- host/device tensor placement from Q38PACK.
- FP16/BF16 smoke kernels and deterministic test vectors.

### M2 - Qwen3.8 graph

- RMSNorm.
- Q4_K / Q4_0 dequant + GEMV/GEMM.
- dense FFN.
- full attention.
- Gated DeltaNet recurrent path.
- tokenizer and chat template.
- greedy decode with logit oracle tests.

### M3 - RTX 3090 specialization

- prepacked SM86 tensor layouts.
- fused norm/projection/state kernels.
- persistent-launch experiments and Driver graph baseline.
- Q8/Q4 KV cache.
- long-context VMM pager.

### M4 - effective-token throughput

- Qwen MTP sidecar/integrated MTP support.
- batched multi-token verification.
- adaptive speculative depth.
- prefix reuse and agent-oriented cache persistence.
