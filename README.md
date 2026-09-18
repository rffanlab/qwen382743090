# qwen382743090 / Q38RT

一个非常窄、非常激进的推理项目：**只先把 Qwen3.8-27B 在 RTX 3090 / SM86 上榨到极限**。

目标不是再做一个通用 vLLM。Q38RT 的 Native 路径计划直接使用 NVIDIA Driver API，
不依赖 PyTorch、vLLM、CUDA Runtime 或 cuBLAS；热点计算最终使用针对 GA102 固定
shape 的 PTX/CUBIN，并由我们自己管理模型布局、KV/recurrent state、VMM 和 MTP。

> 当前状态：**M0 已落地**。GGUF → Q38PACK 转换器、C++ mmap runtime/probe、
> libcuda.so.1 Driver API 探测、OpenAI 兼容 Server 外壳、reference oracle
> 都已提供。Native Qwen3.8 decode kernel 还没有伪装成“已完成”；接口会明确返回
> native_decode_not_ready，直到逐层 correctness 验证通过。

## 为什么先做自己的 Q38PACK

GGUF 是优秀的通用交换格式，但 Runtime 不应该每次启动都为“通用”付费。Q38PACK v1
先做两件事：

- 把 tensor directory 固定成 256-byte entry，启动时直接 mmap；
- tensor 默认 4 KiB 对齐，为后续 cuMemMap / VMM 和 SM86 专用 repack 留接口。

v1 **不会重新量化**。GGUF 中的量化 tensor payload 按原字节复制；原始 GGUF metadata
区也完整保留，因此转换本身不应该引入额外模型误差。

格式见 docs/Q38PACK.md。

## 1. GGUF 转 Q38PACK

只需要 Python 3.10+，转换器本身没有第三方依赖。

    python3 tools/gguf_to_q38pack.py /models/Qwen3.8-27B-Q4_K_M.gguf /models/Qwen3.8-27B-Q4_K_M.q38pack

查看结果：

    python3 tools/inspect_q38pack.py /models/Qwen3.8-27B-Q4_K_M.q38pack

需要把所有 tensor 列出来：

    python3 tools/inspect_q38pack.py /models/Qwen3.8-27B-Q4_K_M.q38pack --tensors

转换器默认允许非 Qwen3.8 GGUF 通过但会在 manifest 中写 warning，方便我们拿合成 GGUF
和 oracle 做测试。Native Runtime 后续会做严格 model contract 校验。

## 2. 编译 Runtime

Linux：

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

Runtime host 代码不需要 CUDA Toolkit 头文件；它在运行时 dlopen("libcuda.so.1")。

无 GPU 先检查 pack：

    ./build/q38-runtime --model /models/Qwen3.8-27B-Q4_K_M.q38pack --no-gpu

3090 上检查驱动、显存和 pack：

    ./build/q38-runtime --model /models/Qwen3.8-27B-Q4_K_M.q38pack

真正跑一遍 Driver API + VMM + PTX kernel 硬件烟测：

    ./build/q38-runtime --model /models/Qwen3.8-27B-Q4_K_M.q38pack --gpu-smoke

成功时应看到 `gpu smoke: PASS (Driver API + VMM + PTX kernel)`。这一步会验证 sm_86、GPU VA 预留、物理显存映射、访问权限、PTX JIT、kernel launch 和 H2D/D2H 数据一致性。

## 3. OpenAI 兼容接口

启动：

    python3 server/q38_openai.py --model /models/Qwen3.8-27B-Q4_K_M.q38pack --host 0.0.0.0 --port 8000

目前可直接使用：

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions

Native decode 未完成时，/v1/chat/completions 会返回 OpenAI 风格的 503 error，而不是
生成假结果。

为了现在就能把 Codex/DSH/客户端接线和做 correctness 对比，可以临时把 llama.cpp
等 OpenAI-compatible server 当 oracle：

    python3 server/q38_openai.py --model /models/Qwen3.8-27B-Q4_K_M.q38pack --host 0.0.0.0 --port 8000 --reference-base-url http://127.0.0.1:8080 --reference-model qwen3.8-27b

这条 reference 路径只是测试基础设施，不是最终 Runtime。

## 4. 测试

    python3 -m unittest discover -s tests -p 'test_*.py' -v

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure

## 接下来

M1 的 Driver context、VMM allocation 和 PTX module/launch smoke 已经落地。下一阶段直接进入模型执行：

1. Q38PACK tensor → GPU VA placement；
2. RMSNorm + FP16/BF16 baseline kernel；
3. Q4_K/Q4_0 dequant-GEMV；
4. dense FFN 与 residual；
5. 用 llama.cpp 固定 logits 做逐层 oracle；
6. 再进入 DeltaNet、full attention、Q4 KV 和 MTP。

详细结构见 docs/ARCHITECTURE.md。

## 项目原则

- 先 correctness，后 benchmark。
- 不为了“看起来能跑”输出假 token。
- 每个优化都必须有固定 workload 的前后数据。
- 先只服务 RTX 3090 / SM86 / Qwen3.8-27B；有性能余量以后再谈通用化。
