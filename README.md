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

不需要模型文件，先单独验证 3090 的 Driver API + VMM + PTX 整条底层链路：

    ./build/q38-gpu-smoke

成功时应看到 `smoke: PASS (Driver API + VMM + PTX kernel)`。这会验证 sm_86、GPU VA 预留、物理显存映射、访问权限、PTX JIT、kernel launch 和 H2D/D2H 数据一致性。

继续验证第一段真实模型数学（hidden=5120 的 RMSNorm baseline）：

    ./build/q38-rmsnorm-smoke

它会在 GPU 上执行 `sum(x²) -> rsqrt(mean+eps) -> x*weight`，并和 CPU reference 做逐元素误差检查。

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

## GPU VA placement

转换出 Q38PACK 后，可以按 3090 实测的 2 MiB VMM granularity 查看权重虚拟地址规划：

    ./build/q38-plan --model /models/Qwen3.8-27B.q38pack --page-bytes 2097152

需要逐 tensor 查看 VA offset：

    ./build/q38-plan --model /models/Qwen3.8-27B.q38pack --page-bytes 2097152 --list


## K_P 量化实际类型分布

K_P 不是新的 GGML tensor type，而是按 tensor 重要性混用标准量化类型。更新代码后重新构建，再运行：

    ./build/q38-plan --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack --page-bytes 2097152

输出底部会额外显示类似：

    type[12/Q4_K]: ...
    type[13/Q5_K]: ...
    type[14/Q6_K]: ...

这决定 Native Runtime 需要优先实现哪些标准 quant kernel。

## 真实模型 Q4_K 解量化烟测

该测试不会使用 synthetic 权重。它会从 Q38PACK 中找到第一个 `GGML_TYPE_Q4_K (12)` tensor，取第一个真实 144-byte / 256-weight super-block，在 RTX 3090 上用我们的 PTX 解量化，并和独立 CPU reference 逐元素比较：

    ./build/q38-q4k-smoke --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack

成功时应看到：

    q4_k: PASS

这一步通过后再进入 fused dequant + dot/GEMV，而不是先把整块权重展开成 FP16/FP32。


### Q5_K 主权重烟测

当前实测模型的 Q5_K 占 12.048 GiB，是绝对主体。Q4_K 通过后，直接验证真实 Q5_K super-block：

    ./build/q38-q5k-smoke --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack

该测试同样从 Q38PACK 里取真实 tensor 的第一个 block，GPU PTX 解量化并和独立 CPU reference 逐元素比较。后续 fused dequant+dot/GEMV 会优先以 Q5_K 为第一优化对象。


## Fused Q5_K GEMV baseline

Q5_K real-block correctness 通过后，下一步直接验证 packed weight × activation，不生成中间 FP32/FP16 权重：

    ./build/q38-q5k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack

默认目标是：

    blk.0.attn_gate.weight
    shape [5120,6144]
    GGML_TYPE_Q5_K

当前 baseline 设计为一个 warp 负责一行，直接从 Q5_K super-block 解码并与 F32 activation 相乘，FP32 累加。为了先锁 correctness，行内归约暂时使用 atomic add；下一版会替换为 warp shuffle / shared reduction。

输出包含：

    max_abs_error
    max_rel_error
    kernel_ms
    effective_weight_bandwidth_GBps

这个 GB/s 是 packed Q5_K 权重流量 / kernel 时间，用来建立 3090 decode 的第一条真实 roofline 基线。


### Warp-reduction v2

第一版真实 `blk.0.attn_gate.weight [5120,6144]` 的实测基线为：

    kernel_ms: 0.0963
    effective_weight_bandwidth_GBps: 224.4699

v2 保持相同数学路径和 benchmark，先做三个低风险优化：

- 每个 Q5_K block 的 qh[32] 每 lane 只读一次；
- ql byte 同时服务低/高 nibble，相邻两个 group 复用一次 load；
- 每行 32 次 global atomic 改为 shfl.sync.down warp reduction + lane0 单次 store。

重新运行同一条命令即可直接和 224.4699 GB/s 基线比较：

    ./build/q38-q5k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack


### Lane-0 metadata broadcast v3

第二版实测：

    kernel_ms: 0.0863
    effective_weight_bandwidth_GBps: 250.7441

v3 在保持相同 packed Q5_K × F32 数学路径的前提下进一步减少 warp 内重复工作：

- d/dmin 只由 lane 0 从 global memory 读取，再用 shfl.sync.idx 广播；
- 每个 32-value group 的 6-bit scale/min 只由 lane 0 解码；
- scale/min 解码结果通过 warp shuffle 广播给其余 31 个 lane；
- qh/ql 缓存和 warp reduction 保持 v2 方案。

重新执行相同 GEMV benchmark，可直接和 250.7441 GB/s 对比：

    ./build/q38-q5k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack


### Rejected experiment: lane-0 metadata broadcast

RTX 3090 real measurement showed the v3 metadata-broadcast experiment was a strong regression:

    v2: 0.0863 ms / 250.7441 GB/s
    v3: 0.2576 ms / 83.9549 GB/s

The change was reverted. On GA102, same-address warp loads for the tiny Q5_K metadata are already handled efficiently enough by the memory hierarchy; serializing those loads through lane 0 costs far more than it saves. The retained F32 baseline is v2.

### Q5_K x Q8_K integer-dot experiment

The next path pairs Q5_K weights with the standard GGML Q8_K activation format:

    block_q8_K = float d + int8 qs[256] + int16 bsums[16]
    size = 292 bytes / 256 activations

For the first experiment, activation quantization is performed on the CPU before timing so GEMV kernel throughput can be measured independently. The GPU uses one CTA per output row and eight warps, one warp per 32-value Q5_K group. Each lane multiplies one Q5 integer with one signed Q8 integer; the warp reduces the integer dot and lane 0 applies scale/min correction using Q8_K bsums.

Run both retained F32 and new Q8_K paths back-to-back:

    ./build/q38-q5k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack && ./build/q38-q5k-q8k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack

The Q8_K benchmark prints `activation_quantization_in_timing: no` explicitly. GPU activation quantization will only be integrated after the packed integer dot path proves worthwhile.


### Rejected Q8_K mapping: 8 warps per row

The first Q5_K x Q8_K kernel was numerically correct but slow:

    F32 v2: 0.0864 ms / 250.3869 GB/s
    Q8_K 8-warp: 0.2338 ms / 92.4999 GB/s

The problem was execution geometry, not Q8_K math: 256 threads were launched for every output row, inflating scheduling/register work while each warp only handled one 32-value group.

### DP4A four-row mapping

The replacement keeps one 32-thread warp as the scheduling unit but computes four output rows at once:

    warp
      subgroup lanes  0..7   -> row 0
      subgroup lanes  8..15  -> row 1
      subgroup lanes 16..23  -> row 2
      subgroup lanes 24..31  -> row 3

For each 32-value Q5_K group, every 8-lane subgroup packs four Q5 bytes and loads four signed Q8 bytes per lane. It then uses:

    dp4a.u32.s32

so one lane computes four integer products per instruction. XOR warp shuffles reduce each independent 8-lane subgroup without crossing row boundaries. Subgroup leaders apply Q5 scale/min and Q8_K bsums correction and write one row result directly, with no global atomics.

Run the retained F32 baseline and the new DP4A path back-to-back:

    ./build/q38-q5k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack && ./build/q38-q5k-q8k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack


### Rejected DP4A mapping: one warp across four rows

Real RTX 3090 result:

    F32 v2:        0.0864 ms / 250.3453 GB/s
    Q8_K 4-row:    0.1564 ms / 138.3079 GB/s

The DP4A math remained correct, but splitting one warp across four independent weight rows destroyed the row-local/coalesced weight access that made the F32 baseline relatively fast.

### Row-local DP4A v2

The active Q8_K DP4A kernel now restores one warp per output row while retaining packed integer dot products:

    one warp -> one Q5_K row

    pass 0:
      lanes  0..7  -> group 0
      lanes  8..15 -> group 1
      lanes 16..23 -> group 2
      lanes 24..31 -> group 3

    pass 1:
      lanes  0..7  -> group 4
      lanes  8..15 -> group 5
      lanes 16..23 -> group 6
      lanes 24..31 -> group 7

Each 8-lane subgroup processes four values per lane with `dp4a.u32.s32`. At the end, subgroup leaders (lanes 0/8/16/24) are gathered with register shuffles and lane 0 writes the row result. No global atomics are used.

Run:

    ./build/q38-q5k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack && ./build/q38-q5k-q8k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack
