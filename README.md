# qwen382743090 / Q38RT

一个非常窄、非常激进的推理项目：**只先把 Qwen3.8-27B 在 RTX 3090 / SM86 上榨到极限**。

目标不是再做一个通用 vLLM。Q38RT 的 Native 路径计划直接使用 NVIDIA Driver API，
不依赖 PyTorch、vLLM、CUDA Runtime 或 cuBLAS；热点计算最终使用针对 GA102 固定
shape 的 PTX/CUBIN，并由我们自己管理模型布局、KV/recurrent state、VMM 和 MTP。

> 当前状态：**Q38PACK v2 / SM86 专用执行格式已经开始落地**。GGUF → Q38PACK v2
> 会把 Q5_K 持久化重排成 META/QH/QS SoA；RTX 3090 实测在
> blk.0.attn_gate.weight 上从约 250 GB/s 提升到约 332 GB/s（+32.8%）。
> Native 完整 Qwen3.8 decode 仍在开发中；未完成的路径不会伪装成可用。

## 为什么先做自己的 Q38PACK

GGUF 是优秀的交换格式，但不是 RTX 3090 的最优执行格式。Q38PACK 保持 256-byte tensor directory 和 mmap 启动能力，同时允许每个 tensor 选择自己的执行布局。

当前版本：

- v1：GGUF payload 原样保存；
- v2：Runtime 仍兼容 v1，同时将 Q5_K 持久化为 `SM86_Q5K_SOA`；
- Q5_K 的 5-bit 权重保持紧凑，只把 scale/min 从 12-byte 6-bit metadata 展成 16 bytes；
- 真实 3090 测试：标准 Q5_K 约 250.2 GB/s，SM86 SoA 约 332.3 GB/s original-equivalent，decoded weights 不变。

格式见 `docs/Q38PACK.md`。

## 1. GGUF 转 Q38PACK v2

v2 是默认格式。Q5_K 的大规模重排使用 NumPy 分块向量化：

    python3 -m pip install numpy

转换你的主模型：

    python3 tools/gguf_to_q38pack.py ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

查看布局：

    python3 tools/inspect_q38pack.py ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

    ./build/q38-plan --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack --page-bytes 2097152

持久化 v2 Q5_K benchmark（不会启动时再 repack）：

    ./build/q38-q5k-sm86-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

需要生成旧 v1 时：

    python3 tools/gguf_to_q38pack.py model.gguf model-v1.q38pack --format-version 1

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


### Row-local DP4A result and decision

The row-local DP4A experiment improved substantially over the earlier Q8_K mappings but still lost to the retained F32 path on RTX 3090:

    Q5_K x F32 v2:              0.0862 ms / 250.8580 GB/s
    Q5_K x Q8_K row-local:     0.1061 ms / 203.8657 GB/s

The Q8_K path is numerically correct, but standard GGUF Q5_K packing still requires too much unpack/reformat work in the hot loop. Further thread-mapping tweaks are no longer the primary optimization path.

### SM86 SoA repack prototype

Before changing Q38PACK on disk, the new experiment repacks only the selected Q5_K tensor in host memory. It preserves the compact 5-bit weights and adds only four metadata bytes per 256-weight block:

    source Q5_K block: 176 bytes

    SM86 prototype logical block:
      d/dmin                     4 bytes
      8 x uint16(scale|min)     16 bytes
      qh                        32 bytes
      qs                       128 bytes
      total                    180 bytes

For GPU access the tensor is stored as three structure-of-arrays planes:

    META plane: 20 bytes/block
    QH plane:   32 bytes/block
    QS plane:  128 bytes/block

QH therefore has a natural 32-byte stride and QS a natural 128-byte stride. Scale/min no longer needs the original 6-bit metadata decode in the GEMV loop. The size overhead is approximately 2.27% before tiny plane-alignment padding.

The prototype also decodes original Q5_K and repacked blocks on CPU before GPU execution and requires exact decoded-weight equality.

Build and compare against the retained winner:

    ./build/q38-q5k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack && ./build/q38-q5k-sm86-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.q38pack

The SM86 benchmark reports both:

    original_equiv_weight_bandwidth_GBps
    physical_repacked_bandwidth_GBps

The first is the apples-to-apples metric against the 250.8580 GB/s GGUF-layout baseline. The second shows actual bytes consumed by the repacked layout. Q38PACK v2 will only be made persistent if the original-equivalent throughput improves enough to justify the small size cost.


## 真实 layer 0：RMSNorm → attn_gate

在持久化 Q38PACK v2 上，可以直接跑 Qwen3.8 第 0 层真实执行链的前两步：

    ./build/q38-layer0-gate --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

该测试使用真实模型 tensor：

    blk.0.attn_norm.weight     F32 / GGUF_NATIVE / [5120]
    blk.0.attn_gate.weight     Q5_K / SM86_Q5K_SOA / [5120,6144]

执行顺序与 Qwen35/Qwen3.8 recurrent layer graph 一致：

    synthetic hidden[5120]
        ↓
    RMSNorm(real attn_norm.weight, eps=1e-6)
        ↓
    SM86 Q5_K SoA GEMV(real attn_gate.weight)
        ↓
    gate projection[6144]

CPU reference 使用相同真实 norm/Q5_K 权重验证前 8 个输出行。输出包含：

    rmsnorm_ms
    attn_gate_projection_ms
    attn_gate_original_equiv_GBps
    chain_ms
    chain_equiv_calls_per_second

这个 target 的意义是从“单算子 benchmark”跨到真实 Qwen3.8 layer execution。下一阶段会把 blk.0 的 qkv、beta、alpha、DeltaNet state 与 FFN 继续接入同一个 Layer Executor。


### layer0 tensor map

真实 RTX 3090 的第一条 Qwen3.8 layer0 链已经通过：

    RMSNorm(real blk.0.attn_norm.weight)
      0.0142 ms
        ↓
    SM86 Q5_K GEMV(real blk.0.attn_gate.weight)
      0.0651 ms / 332.3960 GB/s
        ↓
    chain
      0.0794 ms

correctness:

    max_abs_error: 1.942233e-06
    max_rel_error: 1.060515e-05

继续扩展完整 recurrent layer 之前，先用真实 Q38PACK v2 列出 blk.0 的 tensor type/layout：

    ./build/q38-layer0-map --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

它会检查：

    blk.0.attn_norm.weight
    blk.0.attn_qkv.weight
    blk.0.attn_gate.weight
    blk.0.ssm_conv1d.weight
    blk.0.ssm_dt.bias
    blk.0.ssm_a
    blk.0.ssm_beta.weight
    blk.0.ssm_alpha.weight
    blk.0.ssm_norm.weight
    blk.0.ssm_out.weight
    blk.0.attn_post_norm.weight
    blk.0.ffn_gate.weight
    blk.0.ffn_up.weight
    blk.0.ffn_down.weight

并给每个 tensor 标记当前 runtime 支持状态（例如 Q5K_SM86_READY / NEED_Q6K_GEMV / NEED_IQ4_XS_GEMV / NEED_Q8_0_GEMV）。这个映射决定下一阶段 Qwen35LayerExecutor 的 kernel 优先级。


### Q4_K recurrent projections

Layer0 tensor map showed Qwen3.8 recurrent alpha/beta projections are native Q4_K:

    blk.0.ssm_beta.weight   [5120,48]
    blk.0.ssm_alpha.weight  [5120,48]

The native PTX GEMV path can be validated directly:

    ./build/q38-q4k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack --tensor blk.0.ssm_beta.weight

    ./build/q38-q4k-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack --tensor blk.0.ssm_alpha.weight

### IQ4_XS FFN gate bring-up

The layer0 FFN gate is the remaining large unsupported projection format:

    blk.0.ffn_gate.weight  IQ4_XS  [5120,17408]

Before building GEMV, validate a real 136-byte / 256-weight IQ4_XS block against an independent CPU reference:

    ./build/q38-iq4xs-smoke --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

IQ4_XS uses the standard non-linear 16-value codebook and per-32-value signed scale. Once this smoke passes, the next step is the fused IQ4_XS GEMV / possible SM86 repack benchmark.


### IQ4_XS GEMV baseline

Real-block IQ4_XS dequantization passed with zero GPU/CPU error on:

    blk.0.ffn_gate.weight  [5120,17408]

The next benchmark keeps the original GGUF IQ4_XS layout and performs fused dequant + F32 GEMV:

    ./build/q38-iq4xs-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

The kernel uses one warp per output row. Each 136-byte / 256-weight block contains eight 32-value groups. IQ4_XS nibbles are decoded through the standard nonlinear 16-value codebook; the PTX baseline implements the codebook as four packed 32-bit registers instead of a 16-way branch chain.

Decision rule:

    close to or above ~250 GB/s -> keep native IQ4_XS layout
    clearly below ~250 GB/s    -> prototype SM86_IQ4_XS_SOA and persist it in Q38PACK v2 only if benchmarked beneficial


### SM86 IQ4_XS compact repack prototype

The native IQ4_XS GEMV is numerically correct but the real RTX 3090 result on
`blk.0.ffn_gate.weight [5120,17408]` is only:

    kernel_ms: 0.2373
    effective_weight_bandwidth_GBps: 199.5504

That is too slow for a 47.35 MB FFN projection, so the current experiment keeps
the 4-bit nonlinear weights compact and only moves decode metadata out of the hot loop:

    source IQ4_XS block:
      fp16 d                       2 bytes
      packed scale metadata       6 bytes
      qs                         128 bytes
                                 ---
                                 136 bytes

    SM86 logical block:
      fp16 d                       2 bytes
      int8 group_scale[8]          8 bytes
      qs                         128 bytes
                                 ---
                                 138 bytes

GPU storage uses three SoA planes:

    D plane:      2 bytes/block
    SCALE plane:  8 bytes/block
    QS plane:   128 bytes/block

The weight nibbles and nonlinear 16-value codebook semantics are unchanged.
The logical size increase is only about 1.47% before tiny 128-byte plane
alignment padding.

This is still a temporary benchmark-time repack; Q38PACK v2 is not changed yet.

Run native and SM86 versions back-to-back:

    ./build/q38-iq4xs-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack && ./build/q38-iq4xs-sm86-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

The SM86 benchmark first requires exact decoded-weight equality, then reports:

    original_equiv_weight_bandwidth_GBps
    physical_repacked_bandwidth_GBps

Only a meaningful win over the 199.5504 GB/s native baseline will justify
adding a persistent SM86_IQ4_XS_SOA layout to Q38PACK v2.


### IQ4_XS SoA result: rejected

The compact D2/SCALE8/QS128 SoA prototype preserved weights exactly but regressed
the real RTX 3090 FFN-gate benchmark:

    native IQ4_XS:
      0.2372 ms
      199.5919 GB/s

    temporary SM86 SoA:
      0.2534 ms
      186.8606 GB/s original-equivalent
      189.6085 GB/s physical
      +1.471% size

The SoA layout is therefore rejected and is not persisted in Q38PACK v2.
IQ4_XS remains GGUF_NATIVE on disk.

### IQ4_XS x Q8_1 DP4A experiment

The next path follows the same arithmetic strategy used by upstream CUDA
IQ4_XS vector-dot kernels: keep native IQ4_XS weights, quantize the activation
to standard Q8_1 blocks, turn nonlinear IQ4 codebook values into packed signed
bytes, and use `dp4a.s32.s32`.

Q8_1 layout:

    32 activation values / block
    fp16 d
    fp16 s = d * sum(qs)
    int8 qs[32]
    36 bytes total

The first benchmark quantizes activation on CPU before timing so the core GEMV
can be evaluated independently.

Kernel mapping:

    one warp -> one output row

    pass 0:
      lanes  0..7  -> IQ4 group 0
      lanes  8..15 -> IQ4 group 1
      lanes 16..23 -> IQ4 group 2
      lanes 24..31 -> IQ4 group 3

    pass 1:
      lanes  0..7  -> IQ4 group 4
      lanes  8..15 -> IQ4 group 5
      lanes 16..23 -> IQ4 group 6
      lanes 24..31 -> IQ4 group 7

Each subgroup lane processes four weights with one DP4A instruction.

Compare against the native F32 baseline:

    ./build/q38-iq4xs-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack && ./build/q38-iq4xs-q81-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

The Q8_1 benchmark prints `activation_quantization_in_timing: no`. GPU Q8_1
quantization / RMSNorm fusion will only be implemented if the packed integer
GEMV materially beats the 199.6 GB/s F32 baseline.


### Four-warps-per-CTA IQ4_XS experiment

The IQ4_XS x Q8_1 DP4A experiment was numerically correct but slower than the F32 path:

    IQ4_XS x F32:
      0.2373 ms
      199.5655 GB/s

    IQ4_XS x Q8_1 DP4A:
      0.3077 ms
      153.8915 GB/s

Activation quantization was excluded from the Q8_1 timing, so the packed integer-dot path is rejected for this kernel mapping.

The next experiment keeps native IQ4_XS weights, F32 activation, codebook arithmetic and numerical path unchanged. Only launch geometry changes.

SM86 supports at most 16 resident thread blocks per SM but up to 48 resident warps per SM. A one-warp CTA can therefore hit the block-count limit at only 16 resident warps before register/shared-memory limits are considered.

The new mapping is:

    one CTA = 128 threads = four warps
    warp 0 -> output row CTA*4 + 0
    warp 1 -> output row CTA*4 + 1
    warp 2 -> output row CTA*4 + 2
    warp 3 -> output row CTA*4 + 3

Each warp remains row-local, so weight-access coalescing is unchanged from the original kernel.

Run:

    ./build/q38-iq4xs-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

The output reports:

    kernel_mapping: 4warps_per_cta_1row_per_warp


### IQ4_XS occupancy result and PRMT-vectorized native IQ4_XS

Four warps per CTA only moved the real RTX 3090 result from about 199.6 GB/s
to 202.3 GB/s, so block-count-limited occupancy is not the main bottleneck.

The next native-F32 kernel keeps the 4-warp CTA but changes decode granularity:

    warp = one output row
    eight 4-lane subgroups = eight IQ4_XS 32-value groups

For every 256-weight block:

    each lane loads 4 packed IQ4 bytes = 8 weights
    prmt.b32 performs batched nonlinear codebook lookup
    each lane loads 8 F32 activations and accumulates 8 FMAs
    4-lane subgroup reduction forms one 32-value group dot
    only subgroup leader applies d * group_scale
    final reduction combines eight group leaders

This removes duplicate QS byte loads and moves group scale multiplication
outside the per-weight hot path.

Run:

    ./build/q38-iq4xs-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

Expected output labels:

    kernel_mapping: 4warps_per_cta_8x4lane_prmt
    decode: prmt_byte_permute_8weights_per_lane


### IQ4_XS PRMT result: 553 GB/s

The vectorized native IQ4_XS kernel is the retained implementation.

Real RTX 3090 result on `blk.0.ffn_gate.weight [5120,17408]`:

    previous native:
      0.2373 ms
      199.5655 GB/s

    four-warp CTA only:
      0.2340 ms
      202.3305 GB/s

    PRMT-vectorized native IQ4_XS:
      0.0855 ms
      553.5147 GB/s

Correctness:

    max_abs_error: 1.091286e-07
    max_rel_error: 4.312929e-06

The winning mapping keeps GGUF-native IQ4_XS on disk. Eight 4-lane subgroups
cover the eight 32-value groups, each lane decodes eight weights with
`prmt.b32`, and group scale is applied after subgroup reduction.

The rejected SM86 IQ4_XS SoA and Q8_1/DP4A experiments remain documented but
are not part of the runtime path.

### Vectorized Q5_K SoA follow-up

Because Q5_K is the dominant model storage class, the same execution strategy
is now being tested on persistent `SM86_Q5K_SOA`:

    one CTA = four warps
    one warp = one output row
    eight 4-lane subgroups = eight Q5_K groups
    each lane = eight weights
    d/scale and dmin/min are applied after the subgroup reduction

Run:

    ./build/q38-q5k-sm86-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

Expected labels:

    kernel_mapping: 4warps_per_cta_8x4lane_groups
    decode: 8weights_per_lane_group_scale_after_reduction


### Q5_K vectorized result: 565 GB/s

The vectorized persistent SM86 Q5_K path is now the retained implementation.

Real RTX 3090 result on `blk.0.attn_gate.weight [5120,6144]`:

    previous persistent SM86 Q5_K:
      0.0652 ms
      331.6158 GB/s original-equivalent
      339.1525 GB/s physical

    vectorized SM86 Q5_K:
      0.0382 ms
      565.5825 GB/s original-equivalent
      578.4367 GB/s physical

Correctness:

    max_abs_error: 6.473268e-07
    max_rel_error: 2.778529e-05

The winning kernel uses four warps per CTA and eight 4-lane subgroups per warp.
Each lane processes eight Q5 weights. Group scale/min correction is applied
after subgroup reduction instead of per weight.

The real `q38-layer0-gate` chain now also uses this vectorized Q5_K kernel.

Before treating the 565 GB/s figure as model-wide, benchmark the other major
Q5_K shapes:

    ./build/q38-q5k-sm86-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack --tensor blk.0.attn_qkv.weight

    ./build/q38-q5k-sm86-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack --tensor blk.0.ffn_up.weight

    ./build/q38-q5k-sm86-gemv --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack --tensor blk.0.ffn_down.weight

    ./build/q38-layer0-gate --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack


## Layer0 projection dispatcher

The retained projection paths are now selected by one Runtime dispatcher instead
of per-smoke hard-coding:

    F32 + GGUF_NATIVE        -> F32_DIRECT
    Q4_K + GGUF_NATIVE       -> Q4K_NATIVE
    Q5_K + SM86_Q5K_SOA     -> Q5K_SM86_VEC
    IQ4_XS + GGUF_NATIVE     -> IQ4XS_PRMT

Current measured RTX 3090 Q5_K vectorized results across real layer0 shapes:

    blk.0.attn_qkv.weight  [5120,10240]  621.2945 GB/s
    blk.0.attn_gate.weight [5120, 6144]  564.5549 GB/s
    blk.0.ffn_up.weight    [5120,17408]  655.6136 GB/s
    blk.0.ffn_down.weight  [17408,5120]  472.0586 GB/s
    blk.0.ssm_out.weight   [6144, 5120]  515.8004 GB/s

The real RMSNorm -> attn_gate chain now measures:

    rmsnorm_ms:              0.0143
    attn_gate_projection_ms: 0.0381
    chain_ms:                0.0533

The next runtime target shares one normalized hidden vector across all recurrent
input projections:

    RMSNorm(real blk.0.attn_norm.weight)
        |
        +--> attn_qkv  Q5K_SM86_VEC
        +--> attn_gate Q5K_SM86_VEC
        +--> ssm_beta  Q4K_NATIVE
        +--> ssm_alpha Q4K_NATIVE

Run:

    ./build/q38-layer0-projections --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

The benchmark excludes model upload and PTX JIT time, validates each projection
against the CPU reference, and reports both individual kernel time and the
combined projection-pack chain time.


### Layer0 projection pack result

The shared-normalized-hidden runtime path is now validated on the real model:

    rmsnorm_ms:               0.0140
    attn_qkv_ms:              0.0578
    attn_gate_ms:             0.0382
    ssm_beta_ms:              0.0113
    ssm_alpha_ms:             0.0113
    sum_individual_ms:        0.1326
    projection_pack_chain_ms: 0.1465

All four projections pass their independent CPU reference checks. The combined
chain shares one normalized hidden vector and excludes weight upload / PTX JIT.

### Fused Gated DeltaNet decode core

The next native step is the exact single-token recurrent update used by Qwen35 /
Qwen3.8 recurrent layers.

For the 27B model the real recurrent dimensions are:

    state_dim S_v = 128
    q/k heads     = 16
    value heads   = 48

Value head h reuses q/k head h % 16. The recurrent state is stored in the same
transposed-column layout used by llama.cpp's fused CUDA GDN:

    state[head][column][row]
    48 * 128 * 128 F32 ~= 3 MiB per recurrent layer

For scalar GDA gate g and beta b, one decode step is:

    G = exp(g)
    kv[col] = dot(state[:, col], k)
    delta[col] = (v[col] - G * kv[col]) * beta
    new_state[:, col] = G * state[:, col] + k * delta[col]
    out[col] = dot(new_state[:, col], q) / sqrt(128)

The PTX kernel uses four warps per CTA. Each warp owns one state column and each
lane owns four state rows, matching the upstream CUDA execution structure.

Run:

    ./build/q38-gdn-ar-smoke

The smoke validates all 6144 output values and the complete 48*128*128 updated
state against an independent CPU implementation, then reports kernel latency and
state read+write bandwidth. This target is decode-only (n_tokens=1); prefill
chunking is intentionally deferred.


### Real recurrent prep bring-up

The fused Gated DeltaNet autoregressive core is validated on RTX 3090:

    state_dim: 128
    qk_heads: 16
    value_heads: 48
    kernel_ms: 0.0061
    state_read_write_bandwidth_GBps: 1024.7322

Full output and full 48*128*128 state comparisons pass against the CPU oracle.

The next decode-stage bring-up uses the real layer0 F32 tensors:

    blk.0.ssm_conv1d.weight [4,10240]
    blk.0.ssm_dt.bias       [48]
    blk.0.ssm_a             [48]

and validates the exact Qwen35 preprocessing sequence:

    previous conv state[3,10240] + current qkv[10240]
        -> depthwise Conv1D(kernel=4)
        -> SiLU
        -> q[2048] / k[2048] / v[6144] split
        -> per-head q/k L2 normalization
        -> beta = sigmoid(beta_raw)
        -> gate = softplus(alpha_raw + dt_bias) * ssm_a

Run:

    ./build/q38-recurrent-prep --model ~/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.sm86.q38pack

This target uses deterministic synthetic projection outputs and conv state, but
real model conv/dt/a tensors. It validates the complete conv output, normalized
Q/K, beta/gate values and rolled conv state against independent CPU math before
the stage is embedded into the shared layer0 Runtime workspace.
