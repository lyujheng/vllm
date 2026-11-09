# KV 传输延迟理论计算

## 模型参数（Qwen2.5-7B）

| 参数 | 值 |
|------|-----|
| num_layers | 28 |
| num_kv_heads | 4（GQA） |
| head_dim | 128 |
| dtype | FP16（2 bytes） |
| 模型参数量 | 7B |

## 1. KV Cache 大小计算

**公式：**
```
KV size = num_layers × 2(K和V) × num_kv_heads × head_dim × seq_len × dtype_bytes
```

**代入 Qwen2.5-7B：**
```
KV size = 28 × 2 × 4 × 128 × seq_len × 2 bytes
        = 57344 × seq_len bytes
        ≈ 56 KB × seq_len
```

**不同 seq_len 下的 KV 大小：**

| seq_len | KV size |
|---------|---------|
| 128 | 7 MB |
| 512 | 28 MB |
| 1024 | 56 MB |
| 4096 | 224 MB |
| 32768 | 1.75 GB |
| 131072 (128K) | 7 GB |

## 2. 传输延迟

**公式：**
```
传输延迟 = KV size / 网络带宽
```

**常见网络带宽参考：**
- InfiniBand 400Gbps ≈ 50 GB/s
- InfiniBand 200Gbps ≈ 25 GB/s
- NVLink（H800）≈ 400 GB/s
- PCIe 4.0 x16 ≈ 32 GB/s

**不同网络条件下的传输时间：**

| seq_len | KV size | IB 400Gbps (50GB/s) | IB 200Gbps (25GB/s) | NVLink H800 (400GB/s) |
|---------|---------|----------------------|----------------------|------------------------|
| 128 | 7 MB | 0.14 ms | 0.28 ms | 0.02 ms |
| 512 | 28 MB | 0.56 ms | 1.12 ms | 0.07 ms |
| 1024 | 56 MB | 1.12 ms | 2.24 ms | 0.14 ms |
| 4096 | 224 MB | 4.48 ms | 8.96 ms | 0.56 ms |
| 32768 | 1.75 GB | 35.8 ms | 71.7 ms | 4.48 ms |

## 3. Prefill 计算时间（对比基准）

**Prefill FLOPs 近似公式：**
```
FLOPs ≈ 2 × num_params × seq_len
      = 2 × 7 × 10^9 × seq_len
      = 14 × 10^9 × seq_len
```

> **公式说明：**
> - `num_params`：模型总参数量（Qwen2.5-7B = 7 × 10^9）
> - `2 ×`：每个参数在矩阵乘中做 2 次浮点运算（1 次乘法 + 1 次加法）
> - `× seq_len`：prefill 阶段每个 token 都过一遍所有 Linear 层，seq_len 个 token 就是 seq_len 倍
> - **局限性**：这是粗略估算，只计算了 Linear 层（FFN + QKV projection）的 FLOPs，
>   未包含 Attention 的 QK^T 计算（O(num_layers × seq_len² × head_dim) 部分）。
>   短序列时 Linear 层主导，估算准确；长序列时 Attention 占比增大，实际 prefill 时间会更长。

**H800 FP16 算力：** ~990 TFLOPS，实际利用率约 60%，有效算力 ≈ 594 TFLOPS

> **利用率说明：** 理论峰值算力在实际推理中无法完全达到，受限于显存带宽、
> kernel 启动开销、算子融合程度等因素。60% 是工程经验值，实际视 batch size 和 seq_len 而变。

**传输与计算时间对比（InfiniBand 400Gbps）：**

| seq_len | FLOPs | Prefill 时间 | KV 传输时间 | 传输/计算比 |
|---------|-------|-------------|------------|------------|
| 128 | 1.8 T | 3 ms | 0.14 ms | 4.7% |
| 512 | 7.2 T | 12 ms | 0.56 ms | 4.7% |
| 1024 | 14.4 T | 24 ms | 1.12 ms | 4.7% |
| 4096 | 57.6 T | 97 ms | 4.48 ms | 4.6% |
| 32768 | 460.8 T | 776 ms | 35.8 ms | 4.6% |

## 4. 结论

### 对于 Qwen2.5-7B

- InfiniBand 400Gbps 下，KV 传输时间约为 prefill 计算时间的 **~5%**，传输不是瓶颈
- InfiniBand 200Gbps 下，传输约占 **~10%**，仍然不是主要瓶颈
- NVLink（单机 8 卡 H800 环境），传输时间可以忽略不计

### 传输成为瓶颈的条件

- **模型更大**：Llama-70B 有 80 层、8 KV heads，KV 是 7B 的约 8 倍
- **网络更慢**：RoCE 100Gbps 以下
- **KV head 更多**：非 GQA 模型（MHA），KV head 数等于 attention head 数，KV 成倍增加

### 传输/计算比的规律

传输/计算比 = KV size / prefill FLOPs × 有效算力 / 网络带宽

对于固定模型和硬件配置，这个比值**与 seq_len 无关**（两者都与 seq_len 线性正比）。
瓶颈是否出现取决于模型架构（KV head 数/层数 vs 参数量）和硬件配置（网络带宽 vs 算力）。

### TP 对传输的影响

TP=N 后，每张卡只持有 1/N 的 KV heads，传输量减少为 1/N。
但 prefill 计算量也分摊为 1/N，所以传输/计算比不变。
实际改善在于：每张卡并行传输，总传输时间不变但单卡传输量更小，对网络压力更分散。
