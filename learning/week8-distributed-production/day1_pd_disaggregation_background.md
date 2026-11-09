# PD 分离（Disaggregated Prefill/Decode）背景与原理

## PD 分离要解决什么问题

回忆 Week 2 学过的内容，Prefill 和 Decode 有本质差异：

| | Prefill | Decode |
|--|---------|--------|
| 计算模式 | 一次处理整个 prompt，compute-bound | 每步只处理 1 个 token，memory-bound |
| GPU 利用率 | 高（大矩阵乘，Tensor Core 打满） | 低（小矩阵乘，大量时间花在读 KV Cache） |
| 内存需求 | 一次性分配多个 block，用完可释放 | 长期占用 block，随生成逐步增长 |
| 延迟特性 | 与 prompt 长度成正比 | 基本恒定（TPOT） |

在传统的混合部署（Prefill 和 Decode 共享同一组 GPU）中，这两个阶段互相干扰：

### 问题 1：Prefill 阻塞 Decode

一个长 prompt 的 prefill 可能占用 GPU 几十毫秒，这段时间正在 decode 的请求被迫等待，TPOT 出现尖刺。Week 5 学的 Chunked Prefill 就是在缓解这个问题，但只是治标——把大 prefill 切成小块交替执行，干扰降低了但没消除。

### 问题 2：资源需求冲突

Prefill 想要更多算力（大 batch 并行处理多个 prompt），Decode 想要更多显存（容纳更多并发请求的 KV Cache）。同一组 GPU 上你只能取折中，两边都不满意。

### 问题 3：无法独立扩缩容

如果突然来了一波长 prompt 请求，你需要更多 prefill 算力；如果并发 decode 请求数暴增，你需要更多 KV Cache 显存。混合部署时只能整体扩容，资源利用率低。

## PD 分离的核心思想

既然两阶段资源需求本质不同，那就**物理分开**：

```
用户请求
   ↓
[Prefill 节点集群]          [Decode 节点集群]
 ┌──────────────┐          ┌──────────────┐
 │ 高算力配置    │   KV     │ 大显存配置    │
 │ 大 batch 处理 │ ──传输──→│ 高并发 decode │
 │ prompt        │          │ 低延迟生成    │
 └──────────────┘          └──────────────┘
```

- **Prefill 节点**：专注计算，可以用更激进的 batch 策略（大 `max_num_batched_tokens`），处理完 prompt 后把生成的 KV Cache 传给 Decode 节点，然后立即处理下一批 prompt
- **Decode 节点**：专注生成，不受 prefill 干扰，TPOT 稳定，显存全部用于 KV Cache 以最大化并发数
- **KV 传输**：Prefill 节点计算完 KV Cache 后，通过网络（InfiniBand/NVLink/RDMA）传给 Decode 节点

## 应用场景

### 适合 PD 分离的场景

1. **高 QPS 在线服务**：大量用户同时请求，需要低且稳定的 TPOT（用户体感是逐字输出的流畅度）。混合部署时 prefill 造成的 TPOT 尖刺是不可接受的。
2. **长 prompt 场景**（RAG、文档问答）：prompt 很长（几千到几万 token），prefill 计算量大，如果和 decode 混在一起会严重阻塞。
3. **大规模集群**：有足够的机器可以分别部署，且可以根据流量特征独立扩缩容 prefill 和 decode 节点。

### 不适合 PD 分离的场景

1. **单机部署**：只有一台机器，分离没有意义（我们的实验环境正是单机 8 卡，所以实验只能验证逻辑，无法体验真实收益）
2. **短 prompt + 长输出**：prefill 很快，decode 阶段占主导，分离带来的 KV 传输开销反而是额外负担
3. **低 QPS**：请求少，混合部署的 GPU 利用率就够了，不值得增加架构复杂度

## PD 分离引入的新问题

分离不是免费的，它引入了一个混合部署不存在的环节——**KV Cache 网络传输**：

### 1. 传输延迟

Prefill 完成后，第一个 token 已经在 Prefill 节点上产生，但 Decode 节点还没有拿到 KV Cache，无法生成第二个 token。系统通常等 KV 传输完成、Decode 端就绪后再开始返回 token 流，因此 TTFT 会增加 KV 传输时间：

```
混合部署 TTFT = 排队等待 + prefill 计算
PD 分离 TTFT  = 排队等待 + prefill 计算 + KV 传输 + Decode 端接收确认
```

另一种实现选择是 Prefill 节点直接返回第一个 token，Decode 端异步接收 KV 后接续 decode。这样 TTFT 不增加，但第一到第二个 token 之间会出现额外 gap。

### 2. 带宽压力

KV Cache 可能很大。比如 Llama-70B 处理 4K token 的 prompt：

```
KV size = num_layers × 2 × num_kv_heads × head_size × seq_len × dtype_bytes
        = 80 × 2 × 8 × 128 × 4096 × 2 bytes
        ≈ 1.3 GB
```

如果 InfiniBand 400Gbps（≈ 50GB/s），传输约 26ms。

### 3. 一致性保证

Decode 节点必须确认 KV 完整到达后才能开始 decode，否则 attention 计算会出错。这需要状态机和同步机制。

这就是为什么 vLLM 需要 KV Connector 框架和 Scheduler 层的配合——后续读代码时会看到 `WAITING_FOR_REMOTE_KVS` 状态、`defer_block_free`、`load_kv_async` 这些机制，它们都是在解决上面这些问题。

## 参考论文

- **DistServe**: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving
  - https://arxiv.org/abs/2401.09670
  - PD 分离的奠基工作，用 goodput 指标量化了分离带来的收益
  - 重点阅读前三节：问题定义、性能模型、分离架构设计
