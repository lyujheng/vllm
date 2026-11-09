# Day 1: 性能分析工具链 — Profiler + Roofline 模型

## 一、vLLM Profiler 配置（profiler.py）

### 两种 profiler 类型

| 类型 | 底层机制 | 产出 | 适用场景 |
|------|---------|------|---------|
| `torch` | PyTorch profiler，Python 层插桩 | Chrome Trace JSON（Perfetto 可视化） | 分析各组件耗时占比、定位热点算子 |
| `cuda` | `cudaProfilerStart/Stop` API，只是开关信号 | 无，需配合 nsys/ncu 外部工具 | 看 GPU 硬件级细节：kernel 时间线、SM 占用率、NCCL 通信 |

本质区别：`torch` 是 Python 视角从上往下看，`cuda` 是 GPU 视角从下往上看。实际优化中两者互补。

### torch profiler 关键参数

| 参数 | 作用 |
|------|------|
| `torch_profiler_with_flops` | 估算每个算子的 FLOPs，用于 Roofline 分析 |
| `torch_profiler_with_stack` | 记录调用栈，方便定位代码位置 |
| `torch_profiler_record_shapes` | 记录 tensor 形状，开销较大 |
| `torch_profiler_with_memory` | 追踪显存变化，排查显存问题 |
| `torch_profiler_dump_cuda_time_total` | 记录 CUDA 端耗时（区别于 CPU launch 耗时） |
| `capture_torch_profiler` | profiling CUDA Graph capture 阶段 |
| `detailed_trace_annotation` | 在 trace 上标注 roofline 指标（sk, sqsq, sqsk） |
| `ignore_frontend` | 只抓后端 EngineCore，避免前端持续录制的开销 |

### Profiling schedule（时间窗口控制）

```
收到 /start_profile
  │
  ├─ delay_iterations    profiler 未启动，零开销
  ├─ wait_iterations     profiler 已创建但关闭，零开销
  ├─ warmup_iterations   profiler 在跑但数据丢弃（去除 JIT 等一次性噪声）
  ├─ active_iterations   真正采集数据（默认 5 步）
  └─ max_iterations      总步数上限
```

## 二、ComponentMetrics 体系（perf.py）

vLLM 自己的组件级耗时拆分，与 profiler 互补。profiler 给实测 kernel 耗时，ComponentMetrics 给理论计算量和访存量。

### 追踪的组件

| 组件类 | 子操作 | 说明 |
|--------|--------|------|
| AttentionMetrics | qkv_proj, attn_qk, attn_av, out_proj | 标准 MHA/GQA |
| MLAAttentionMetrics | q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, attn_qk, attn_av, out_proj | DeepSeek 系列，KV Cache 压缩 |
| FfnMetrics | dense_ffn, routed_ffn, shared_ffn | 含 Dense 和 MoE |
| UnembedMetrics | matmul | LM Head，hidden_size → vocab_size |

### MLA vs 标准 Attention

- 标准 Attention：KV Cache 存 `2 × num_kv_heads × head_dim`
- MLA：KV Cache 存 `kv_lora_rank + qk_rope_head_dim`（压缩后小很多倍）
- MLA 多了 kv_b_proj 解压步骤，tradeoff：存储省 → 并发高 → 吞吐高，但每步多一次 GEMM → 单步延迟略高

### ModelMetrics 聚合

遍历所有组件实例，给 key 加组件前缀合并成统一的 FLOPs / bytes 分解表：
```python
{"attn.qkv_proj": ..., "attn.attn_qk": ..., "ffn.dense_ffn": ..., "unembed.matmul": ...}
```

## 三、SwiGLU FFN 原理

```
输入 x: [T, D]
    ┌────┴────┐
 gate_proj   up_proj      两个独立线性投影 D → DI
    │         │
  SiLU()      │            gate 分支过 SiLU 激活
    └────×────┘            逐元素相乘（不是 concat）
         │
      down_proj            DI → D
         │
     输出: [T, D]
```

- vLLM 实现上 gate 和 up 权重 concat 成 `[2*DI, D]`，一次 GEMM 产出 `[T, 2*DI]`，再由 `SiluAndMul` 融合算子拆分并做 SiLU+逐元素乘
- FLOPs = `2 × D × 3 × DI × T × L`（3 个矩阵乘：gate、up、down）

## 四、Embed vs Unembed

| | Embed（输入侧） | Unembed（输出侧） |
|--|----------------|-------------------|
| 操作 | 查表（gather/index_select） | 矩阵乘 `[T,D]×[D,V]` |
| 计算量 | 几乎为零 | 大词表时不小（V=152064） |
| ComponentMetrics | 不追踪 | 单独追踪 |

## 五、Roofline 模型

### H800 参数

```
FP16 峰值算力 ≈ 990 TFLOPS
HBM 带宽     ≈ 3.35 TB/s
屋脊拐点     = 990 / 3.35 ≈ 295 FLOPs/byte
```

- intensity < 295 → memory-bound → 优化方向：减少访存（FP8 量化）
- intensity > 295 → compute-bound → 优化方向：减少计算量（FP8 计算）

### 单位换算

TFLOPS / TB/s = FLOPs/byte（T = 10^12 约掉）

### GEMM FLOPs 计算

`[M, K] × [K, N]` → FLOPs = 2 × M × K × N（包含乘和加，即 MAC × 2）

### 一层 Transformer 的 GEMM 清单（实际 kernel 粒度）

```
① qkv_proj:   [T, D] × [D, (q+2*kv)*d]      — 1 个 merged GEMM
② flash_attn:  QK^T + softmax + AV            — 1 个融合 kernel
③ out_proj:    [T, q*d] × [q*d, D]            — 1 个 GEMM
④ gate+up:     [T, D] × [D, 2*DI]             — 1 个 merged GEMM
⑤ down_proj:   [T, DI] × [DI, D]              — 1 个 GEMM
```

整个模型 = 28 层 × (①②③④⑤) + ⑥ unembed

### Prefill vs Decode 的 Roofline 差异

核心规律：**intensity 跟 T（当前新 token 数）正相关**

- Prefill（T=1024）：T 大 → FLOPs 大 → intensity 高 → **compute-bound**
- Decode（T=1）：T 小但权重照读 → intensity 极低 → **memory-bound**

这解释了为什么 continuous batching 有效——增大有效 T，把 GPU 从 memory-bound 推向 compute-bound。

### Qwen2.5-7B GQA 参数

```
hidden_size=3584, num_attention_heads(Q)=28, num_kv_heads=4
head_dim=128, intermediate_size=18944, num_hidden_layers=28
q*d = 28*128 = 3584 = hidden_size（设计上保证残差连接维度一致）
GQA 分组：28/4 = 7 个 Q head 共享 1 组 KV head
```

### FlashAttention 融合 kernel 的 Roofline

```
FLOPs = attn_qk + attn_av = 2 × 28 × T × context × 128 × 2
bytes = 读 Q [T×28×128×2] + 读 K [context×4×128×2] + 读 V [context×4×128×2] + 写 O [T×28×128×2]
```

中间的 scores 不经过 HBM（在 SRAM 中完成），这是 FlashAttention 的核心优化。

## 六、Profiling 实验结果

### 实验配置
- Qwen2.5-7B, TP=1, torch profiler
- with_flops=true, with_stack=true, record_shapes=true
- delay=2, warmup=1, active=3

### kernel 分布（3 个 active iteration）

| 组件 | 占比 | 说明 |
|------|------|------|
| GEMM（投影 + FFN） | ~86% | 绝对主导，qkv_proj + out_proj + gate/up/down + unembed |
| FlashAttention | ~6.5% | attn_qk + attn_av 融合，decode 阶段占比小 |
| RMSNorm | ~2.3% | 逐元素操作 |
| KV Cache 写入 | ~1.0% | reshape_and_cache_flash_kernel |
| SiLU_and_mul | ~0.9% | gate/up 的融合激活 |

### 关键观察

1. **count=84 = 28 层 × 3 步**，验证了模型结构
2. **FlashAttention 占比仅 6.5%**：说明采集到的主要是 decode 阶段（T=1 时 attention 计算量极小）
3. **GEMM 占 86%**：decode 阶段每步都要读完整权重，对应 memory-bound 特征
4. **Unembed（count=3 的大 tile GEMM）单独占 6.2%**：词表 152064 维度大，decode 阶段不可忽略
5. **record_shapes=true 导致明显的 profiling overhead**，后续实验可关闭
