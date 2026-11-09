# Day 6: KV Cache Offloading + 长上下文内存规划

## 1. 问题背景：GPU 显存是 LLM 推理的核心瓶颈

LLM 推理中 GPU 显存被三部分占据：

```
GPU 显存 = 模型权重 + KV Cache + 临时计算缓冲
```

- **模型权重**：固定不变，加载后不再增长
- **KV Cache**：随请求数量和序列长度**动态增长**，是主要的显存压力来源
- **临时缓冲**：activation、中间结果等，相对较小

核心矛盾：**显存有限，但并发请求的 KV Cache 总量可能超出 GPU 显存容量**。

当 GPU 显存不足以容纳所有 running 请求的 KV Cache 时，必须做出选择：
- 方案 A：**抢占（Preemption）** — 把某个请求踢出去，释放其 KV Cache，下次重新 prefill
- 方案 B：**Offloading** — 把某个请求的 KV Cache 搬到 CPU 内存，需要时再搬回来

## 2. 两种策略的对比：Preemption vs Offloading

### 2.1 Preemption（抢占 + 重算）

当显存不足时，scheduler 选择一个 running 请求，释放其所有 KV Cache block，将请求退回 waiting 队列。下次调度到该请求时，需要**从头重新 prefill**，重新计算所有 KV Cache。

代价：重算延迟 = prefill 的计算时间，与序列长度成正比。

```
重算延迟 ≈ prefill FLOPs / GPU 算力
```

对于短序列（如 256 token），prefill 很快，重算代价不大。但对于长序列（如 128K token），prefill 可能需要数秒甚至更久。

### 2.2 KV Cache Offloading（搬移到 CPU）

将 KV Cache 从 GPU 显存搬到 CPU 内存（通过 PCIe），需要时再搬回来。KV 数据不会丢失，无需重算。

代价：传输延迟 = KV 数据量 / PCIe 带宽。

```
传输延迟 ≈ KV size / PCIe 带宽
```

PCIe 4.0 x16 的带宽约 32GB/s，PCIe 5.0 约 64GB/s。

### 2.3 Break-even 分析：何时 Offload 比重算更划算？

关键洞察：**短序列重算快，长序列 offload 快**。

重算延迟随序列长度**二次增长**（attention 的复杂度是 O(n²)），而传输延迟随序列长度**线性增长**（KV 数据量与 seq_len 成正比）。因此存在一个临界点：

```
短序列：重算延迟 < 传输延迟 → 重算更快
长序列：重算延迟 > 传输延迟 → offload 更快
```

实际的 break-even point 取决于具体的模型参数、GPU 算力和 PCIe 带宽。

### 2.4 异步传输的影响

如果 offload 传输可以和 GPU 计算**重叠**（overlap），那么 offload 的有效延迟会更低：

- GPU 在处理其他请求的 forward 时，PCIe 可以同时在搬 KV Cache
- 这样 offload 的传输延迟被隐藏在计算中，几乎是"免费"的
- 但前提是 PCIe 带宽不被其他传输（如 NVLink 通信）争抢

异步传输让 offload 的 break-even point 向更短的序列方向移动，扩大了 offload 的优势区间。

## 3. KV Cache Offloading 的完整流程

### 3.1 Offload（GPU → CPU）

当 GPU 显存压力大、某个请求被抢占或暂时不需要时：

```
1. Scheduler 决定 offload 某个请求
2. 将该请求的 KV Cache block 通过 PCIe 异步拷贝到 CPU 内存
3. 等待拷贝完成后，释放 GPU 上的 KV Cache block（defer_block_free 机制）
4. 请求状态标记为 offloaded，保留 CPU 侧的 block 映射信息
```

### 3.2 Load（CPU → GPU）

当该请求需要继续推理时：

```
1. Scheduler 决定重新调度该请求
2. 在 GPU 上分配新的 KV Cache block
3. 将 CPU 内存中的 KV Cache 通过 PCIe 拷贝回 GPU
4. 请求恢复 running 状态，从上次中断的位置继续 decode
```

### 3.3 defer_block_free 机制

一个关键的同步问题：offload 是异步的（PCIe DMA），如果拷贝还没完成就释放了 GPU 上的 block，数据就丢失了。

`defer_block_free` 的做法是：**不在 offload 发起时立即释放 block，而是等到确认拷贝完成后再释放**。实现上是记录"这些 block 在第 N 步调度时被标记为 defer free"，等第 N 步的计算完成后才真正回收。这复用了 PD 分离中的 fencing 逻辑（Day 1 学过的）。

## 4. vLLM 中的两套 Offloading 实现

vLLM 提供了两套 offloading 实现，适用于不同场景：

### 4.1 Simple KV Offload（轻量级）

代码路径：`vllm/v1/simple_kv_offload/`

结构简单，直接做 GPU ↔ CPU 的 block 拷贝：
- `manager.py` — scheduler 侧管理：决定 offload/load 哪些 block，维护 GPU/CPU block 映射
- `worker.py` — worker 侧执行：实际的 CUDA memcpy 操作
- `copy_backend.py` — 拷贝后端（CUDA stream 管理）
- `cuda_mem_ops.py` — 底层的 GPU↔CPU 内存操作

适合简单场景：单机部署、不需要复杂的监控和调度策略。

### 4.2 Offloading Connector（完整框架）

代码路径：`vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` + `offloading/`

建立在 KV Connector 框架之上（和 PD 分离用同一套基础设施），提供完整的生产级能力：
- `scheduler.py` — offloading 调度策略（哪些 block 该 offload、优先级等）
- `worker.py` — 异步传输执行
- `metrics.py` — Prometheus 监控指标（offload/load 的吞吐、延迟、命中率等）
- `config.py` — 配置管理
- `events.py` — 事件系统

适合生产环境：需要监控、需要和 PD 分离复用基础设施、需要精细的 offload 策略控制。

### 4.3 选型依据

| 维度 | Simple KV Offload | Offloading Connector |
|------|-------------------|---------------------|
| 复杂度 | 低 | 高 |
| 监控 | 无 | Prometheus metrics |
| 调度策略 | 简单 | 可定制 |
| 与 PD 分离复用 | 否 | 是（KV Connector 框架） |
| 适用场景 | 单机、简单部署 | 生产环境、多节点 |

## 5. 长上下文内存规划

### 5.1 KV Cache 显存计算公式

```
每请求 KV Cache = num_layers × 2 × num_kv_heads × head_size × seq_len × dtype_bytes
```

其中：
- `num_layers`：transformer 层数
- `2`：K 和 V 两个张量
- `num_kv_heads`：KV head 数量（GQA/MQA 下远小于 Q head 数）
- `head_size`：每个 head 的维度（通常 128）
- `seq_len`：序列长度
- `dtype_bytes`：数据类型字节数（FP16=2, FP8=1）

### 5.2 实际计算示例：Llama-70B @ 128K context

模型参数：80 layers, 8 KV heads (GQA), 128 head_size, FP16

```
每请求 KV = 80 × 2 × 8 × 128 × 131072 × 2 bytes
         = 80 × 2 × 8 × 128 × 131072 × 2
         = 34,359,738,368 bytes
         ≈ 32 GB
```

单个 128K 请求的 KV Cache 就需要 32GB。H800 80GB 卡，模型权重约 140GB（FP16），TP=8 后每卡权重约 17.5GB，可用显存约 60GB。

- 60GB / 4GB (TP=8 后每请求 KV) = **15 个并发**
- 如果用 FP8 KV Cache：60GB / 2GB = **30 个并发**（翻倍）
- 如果再加 CP=2：60GB / 1GB = **60 个并发**（再翻倍）

### 5.3 降低 KV Cache 显存的手段

**FP8 KV Cache**：
- 将 KV Cache 从 FP16 量化到 FP8，显存减半
- 精度损失很小，大多数场景可接受
- vLLM 原生支持，通过配置开启

**Context Parallelism (CP)**：
- 将长序列切分到多张卡上，每张卡只存一部分 KV Cache
- CP=2 时每卡 KV 减半
- 需要 ring attention 或类似机制在卡间传递 attention 计算

**Chunked Prefill**：
- 超长输入不一次性 prefill，分块处理
- 控制单次 prefill 的 token 数量（`max_num_batched_tokens`）
- 避免 prefill 阶段的显存峰值过高
- 不减少 KV Cache 总量，但平滑了显存使用曲线

**KV Cache Offloading**：
- 不减少 KV Cache 总量，而是将不活跃请求的 KV Cache 搬到 CPU
- 扩大了有效的 KV Cache 容量（GPU + CPU）

### 5.4 部署 context length 上限评估方法

```
可用 KV 显存 = GPU 显存 × gpu_memory_utilization - 模型权重 / TP - 临时缓冲
max_context_per_req = 可用 KV 显存 / (num_layers × 2 × num_kv_heads/TP × head_size × dtype_bytes)
```

实际部署还需考虑：
- 并发请求数 × 每请求 KV 的总和不能超过可用 KV 显存
- 长短请求混合时，短请求的 KV Cache 会挤占长请求的空间
- 需要预留一定的显存余量应对突发

## 6. EC Connector（多模态 PD 分离）

### 6.1 背景

在 PD 分离（Prefill-Decode disaggregation）场景下，prefill 节点处理完输入后需要把计算结果传给 decode 节点。对于纯文本模型，传的是 decoder 的 KV Cache（通过 KV Connector）。

但对于多模态模型（如视觉-语言模型），除了 decoder KV Cache，还有**视觉编码器的输出（encoder cache）**需要传输。EC Connector 就是专门负责这部分传输的。

### 6.2 分工

```
KV Connector: prefill 节点 → decode 节点，传输 decoder KV Cache
EC Connector: prefill 节点 → decode 节点，传输 encoder embeddings（视觉编码器输出）
```

两者使用同一套 PD 分离基础设施，但传输的数据类型不同。EC Connector 只在多模态模型中使用。

## 7. 应用场景总结

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 短上下文、高并发 | Preemption（重算） | 短序列重算快，offload 传输反而更慢 |
| 长上下文（32K+）、显存紧张 | KV Cache Offloading | 长序列重算代价大，offload 更划算 |
| 长上下文、高并发 | FP8 KV + CP + Offloading 组合 | 多手段叠加降低显存压力 |
| 128K+ 超长上下文 | Chunked Prefill + FP8 KV + CP | 平滑显存峰值 + 减少每请求 KV 量 |
| 多模态 PD 分离 | KV Connector + EC Connector | decoder KV 和 encoder cache 分别传输 |
| 生产环境、需要监控 | Offloading Connector（完整框架） | Prometheus metrics + 可定制策略 |
| 简单部署、快速验证 | Simple KV Offload | 配置简单、开箱即用 |
