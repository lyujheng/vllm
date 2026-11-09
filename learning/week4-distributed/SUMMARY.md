# Week 4 总结：分布式推理基础

> 学习时间：2026-08-28 ~ 2026-09-03
> 核心目标：掌握 TP/PP/DP/CP 四种并行策略在 vLLM 中的完整实现，理解 Executor 多 Worker 架构

---

## 📚 本周学习内容

### Day 1: TP 原理 + 并行 Linear 层实现 ✅

- Column/Row Parallel Linear 的权重切分与通信模式
- Attention 按 head 切分（GQA 按 KV head group 切分）
- MLP 层：Gate/Up 用 ColumnParallel，Down 用 RowParallel
- Sequence Parallelism 简要了解

### Day 2: 权重加载 + Group 管理 + 通信封装 ✅

- weight_loader 按 tp_rank 切分 safetensors 权重
- parallel_state.py 中 TP/PP/DP/EP group 初始化
- GroupCoordinator 封装 rank 列表与通信方法
- NCCL 通信原语与 custom_all_reduce（NVLink 优化）

### Day 3: Pipeline Parallelism（PP）✅

- vLLM 推理中 PP 是逐 stage 串行，非训练的 micro-batch pipelining
- 跨 stage hidden state 传输：P2P Send/Recv
- PP + TP 组合的 rank 分布

### Day 4: Data Parallelism（DP）+ DP Coordinator ✅

- DP 多副本独立推理，DPCoordinator 负载均衡分发请求
- DP Attention 概念：序列分布到不同 DP rank，attention 时跨 rank 通信

### Day 5: Context Parallelism（CP）✅

- PCP：长 prompt 的 token 序列均分到多卡，各卡只计算自己的 Q 部分，通过 all_gather 收集完整 KV 和 hidden states
- DCP：decode 阶段 KV Cache 分散存储，all-gather 获取远端 KV
- CP 切 sequence 维度减 KV Cache 显存，与 TP 切 head 维度互补

### Day 6: Executor 架构 + 多 Worker 启动流程 ✅

- Executor 抽象基类、MultiprocExecutor、RayExecutor 完整代码跟读
- Worker 进程创建、消息队列、PP 通信在各层的体现

### Day 7: 总结 ✅

---

## 🎯 核心知识点总结

### 1. 四种并行策略对比

| 策略 | 切分维度 | 通信模式 | 解决的问题 | 通信量 |
|------|----------|----------|------------|--------|
| TP | head / hidden 维度 | AllReduce / AllGather | 单层权重太大 | 每层 2 次集合通信 |
| PP | layer 维度 | P2P Send/Recv | 总权重太大 | 每 stage 传一次 hidden states |
| DP | 请求维度 | 无（独立推理） | 提高吞吐 | 几乎无 |
| CP | sequence 维度 | Ring P2P / AllGather | 超长序列 KV Cache 太大 | 每层传 KV 块 |

### 2. TP 权重切分模式

```
Transformer Layer 内的 TP 切分：

Attention:
  Q/K/V proj → ColumnParallel（按 head 切分，每卡 num_heads/tp_size）
  O proj     → RowParallel（forward 后 AllReduce）

MLP:
  Gate/Up    → ColumnParallel（按列切分，无需通信）
  Down       → RowParallel（forward 后 AllReduce）

Embedding:
  VocabParallelEmbedding → 按 vocab 维度切分

通信模式：
  ColumnParallel → RowParallel 之间：AllReduce
  每个 Transformer layer 共 2 次 AllReduce（Attention 一次 + MLP 一次）
```

### 3. PP 在各层的实现

PP 的核心设计：**Executor 层不感知 PP，PP 逻辑下推到 Worker 和 ModelRunner。**

```
Executor 层:
  - collective_rpc 统一广播给所有 Worker，不区分 TP/PP
  - 唯一的 PP 感知：output_rank 选最后一个 PP stage 的 rank 0
  - MultiprocExecutor: output_rank = world_size - tp_size * pcp_size

Worker 层 (gpu_worker.py execute_model):
  ┌─ First PP rank:
  │    intermediate_tensors = None
  │    output = model_runner.execute_model(scheduler_output, None)
  │    → 返回 IntermediateTensors
  │    → isend() 发给下一个 stage（异步）
  │    → return None
  │
  ├─ Middle PP rank:
  │    irecv() 从上一个 stage 接收（异步，包装为 AsyncIntermediateTensors）
  │    output = model_runner.execute_model(scheduler_output, intermediate_tensors)
  │    → 返回 IntermediateTensors
  │    → isend() 发给下一个 stage（异步）
  │    → return None
  │
  └─ Last PP rank:
       irecv() 从上一个 stage 接收（异步）
       output = model_runner.execute_model(scheduler_output, intermediate_tensors)
       → 返回 ModelRunnerOutput（有 logits + 采样结果）
       → return output

ModelRunner 层 (gpu_model_runner.py):
  - First rank: 跑 embedding + 前 N 层 transformer
  - Non-first rank: 跳过 embedding，用接收的 intermediate_tensors
  - Last rank: 跑 LM head → logits → sampling
  - Non-last rank: 不计算 logits，返回 IntermediateTensors
  - 多模态 encoder 只在 first rank 跑
  - Speculative decoding 只在 last rank 初始化
```

**PP 异步通信模式：**
```
iteration N:
  1. wait(上一轮的 send handle)     ← 确保上轮发送完成
  2. irecv() 异步接收上一 stage      ← 非阻塞
  3. model_runner.execute_model()    ← 访问 tensor 时才等待 recv 完成
  4. isend() 异步发送给下一 stage    ← 非阻塞，handle 存起来
  5. return None                     ← 中间 stage 不返回最终结果
```

### 4. Group 管理（parallel_state.py）

8 GPU，TP=4, PP=2 的 group 分布：
```
rank:     0  1  2  3  4  5  6  7
PP stage: 0  0  0  0  1  1  1  1
TP rank:  0  1  2  3  0  1  2  3

TP groups:  [0,1,2,3], [4,5,6,7]    ← 同 stage 内的卡，AllReduce
PP groups:  [0,4], [1,5], [2,6], [3,7]  ← 跨 stage 对应位置，P2P
```

`GroupCoordinator` 封装每个 group 的 rank 列表和通信方法：
- `get_tp_group()` → TP 组通信（AllReduce/AllGather）
- `get_pp_group()` → PP 组通信（Send/Recv），提供 `is_first_rank`/`is_last_rank` 判断

### 5. Executor 架构

```
Executor (抽象基类)
  ├── UniProcExecutor        单卡，supports_pp=False
  ├── MultiprocExecutor      单机多卡，supports_pp=True
  │     控制面: SharedMemory MessageQueue 广播
  │     数据面: Worker 内部 NCCL 通信
  │     PP: Worker 自行 isend/irecv，Executor 不感知
  │
  └── RayDistributedExecutor 多机多卡，supports_pp=True, uses_ray=True
        控制面: worker.execute_method.remote() (Ray RPC)
        数据面: Compiled DAG + NCCL/shm
        PP: DAG 显式构建 stage 拓扑
```

### 6. MultiprocExecutor vs RayExecutor

| | MultiprocExecutor | RayExecutor |
|---|---|---|
| Worker 形式 | 本地 multiprocessing.Process | Ray remote actor |
| execute_model | 走 collective_rpc 广播 | 走 Compiled DAG |
| PP 拓扑 | Worker 内部 isend/irecv 自行完成 | DAG 显式构建 pp_tp_workers[pp][tp] |
| 多节点 | leader-follower，手动配 node_rank | Ray 集群自动调度 |
| 结果收集 | 共享内存 response_mq | ray.get() 或 DAG ref.get() |
| Worker 监控 | WorkerMonitor sentinel + wait | 无（check_health 空实现）|
| 异步调度 | 支持（AsyncMultiprocExecutor）| 不支持 |

### 7. MultiprocExecutor 消息队列机制

```
主进程:
  rpc_broadcast_mq.enqueue((method, args, kwargs, output_rank))
      ↓ 广播
所有 Worker (worker_busy_loop):
  method, args, kwargs, output_rank = rpc_broadcast_mq.dequeue()
  func = getattr(self.worker, method)
  output = func(*args, **kwargs)
  if self.rank == output_rank:
      self.handle_output(output)  → 写入 worker_response_mq
      ↓
主进程:
  result = response_mqs[output_rank].dequeue()
```

本地 Worker: 共享内存队列（mmap，零拷贝）
远端 Worker: Worker 0 的 peer_worker_response_mqs 中转（网络）

队列就绪握手必须双方按相同顺序 wait_until_ready()，否则死锁。

### 8. PCP（Prefill Context Parallelism）详细原理

**解决的问题**：超长 prompt 的 prefill 阶段计算量大（attention 复杂度 O(n²)），单卡 prefill 延迟高。

**核心思路**：切分 Q（sequence 维度），不切分 KV Cache。每个 rank 只算部分 Q token 的 forward，KV 通过 all_gather 收集完整后写入每卡的本地 KV Cache，保证 decode 阶段不需要跨卡通信。

#### 8.1 整体执行流程

以一个 8 token 的 prompt、PCP=2 为例：

```
Prefill 阶段:

  Global Batch: [t0 t1 t2 t3 t4 t5 t6 t7]

  Step 1: PCPManager.partition_batch() — 按 rank 切分 Q token
    Rank 0 local batch: input_ids=[t0 t1 t2 t3], positions=[0 1 2 3]
    Rank 1 local batch: input_ids=[t4 t5 t6 t7], positions=[4 5 6 7]

  Step 2: model.forward() — 每个 rank 独立跑 forward
    每个 Transformer Layer 内部:
      ┌─────────────────────────────────────────────────────┐
      │  Rank 0                    Rank 1                   │
      │  Embedding(t0~t3)→h0~h3   Embedding(t4~t7)→h4~h7   │
      │          ↓                         ↓                │
      │  QKV proj → Q0,K0,V0      QKV proj → Q1,K1,V1      │
      │          ↓                         ↓                │
      │  Attention(Q0, K0, V0)    Attention(Q1, K1, V1)     │
      │  (注意: 这里只用了局部KV)  (注意: 这里只用了局部KV)    │
      │          ↓                         ↓                │
      │  KV all_gather: 收集 K0+K1, V0+V1                   │
      │  → 每卡写入完整 KV Cache [K0 K1 V0 V1]              │
      │          ↓                         ↓                │
      │  MLP(h0~h3)               MLP(h4~h7)                │
      └─────────────────────────────────────────────────────┘
    ... 重复 28 层 ...

  Step 3: PCPManager.restore_for_sampling() — all_gather hidden states
    Rank 0 hidden: [h0 h1 h2 h3]
    Rank 1 hidden: [h4 h5 h6 h7]
    all_gather → [h0 h1 h2 h3 h4 h5 h6 h7]
    按 hidden_restore_idx 恢复全局顺序 → sampling

Decode 阶段:
  PCP 不参与。每卡已有完整 KV Cache，正常自回归 decode。
```

#### 8.2 Attention 计算的关键问题

**Q: 每个 rank 只有部分 Q，KV Cache 完整还是不完整？**

**KV Cache 是完整的。** 从 MLA forward 的执行顺序可以确认（mla_attention.py:636-664）：

```python
# Step 1: all_gather 收集各 rank 的局部 KV
kv_for_cache, kpe_for_cache, layer_slot_mapping = (
    maybe_gather_mla_latent_cache_inputs(
        kv_c_normed, k_pe, layer_slot_mapping,
        attn_metadata.num_decode_tokens, self.use_pcp,
    )
)

# Step 2: 写入完整 KV Cache
self.impl.do_kv_cache_update(
    kv_for_cache, kpe_for_cache, self_kv_cache,
    layer_slot_mapping, self.kv_cache_dtype, self._k_scale,
)

# Step 3: attention 计算（从完整 KV Cache 读取）
self.forward_impl(q, kv_c_normed, k_pe, self_kv_cache, attn_metadata, ...)
```

每一层 Transformer 的执行流程：
1. QKV projection → 各 rank 产出**局部** K/V（只有自己那部分 token 的）
2. **all_gather** → 收集所有 rank 的 K/V
3. **写完整 KV Cache** → 每卡都有全量 KV
4. **Attention 计算** → 从完整 KV Cache 读取，Q 是局部的，KV 是完整的

所以每个 rank 的 attention 看到的是完整 KV，causal attention 结果正确。PCP 节省的是 QKV projection 和 attention 的计算量（Q 少了），而不是 KV Cache 的显存。

#### 8.3 Batch 分配策略（mixed prefill + decode）

实际推理时，一个 batch 中混合了 prefill 和 decode 请求。PCPManager 的分配策略：

```
PCP=2, global batch: [A(prefill) B(prefill) C(decode) D(prefill) E(prefill) F(prefill) G(decode)]

分配规则:
  - Prefill 请求：按 rank 均分 token
  - Decode 请求：在所有 rank 上复制（每 step 只 1 个 token，计算量极小）

Rank 0 local: [A B G]         (A、B 的 prefill token + G 的 decode token)
Rank 1 local: [D E F C]       (D、E、F 的 prefill token + C 的 decode token)
padded gather: [A B G _ | D E F C]

恢复全局顺序:
  hidden_restore_idx: [0, 1, 7, 4, 5, 6, 2, ...]
  global_hidden = gathered_hidden[hidden_restore_idx]
```

来自源码 `_build_batch_layout()` 的注释：
```
PCP=2 example:
  global batch:       [A B C D E F G]
  rank 0 / rank 1:    [A B G] / [C D E F]
  padded gathered:    [A B G _ | C D E F]
  hidden_restore_idx: [0, 1, 4, 5, 6, 7, 2]
  padded_gather_idx:  [0, 1, 6, 0, 2, 3, 4, 5]
```

#### 8.4 通信分析

| 通信操作 | 发生位置 | 数据量 | 次数/forward |
|----------|----------|--------|-------------|
| KV all_gather | 每层 attention 内部 | N/PCP × kv_dim × 2bytes × PCP | 28 层 × 1 次 |
| Hidden all_gather | model forward 结束后 | N × hidden_dim × 2bytes | 1 次 |

其中 N = prefill token 总数。

对比 TP 的 AllReduce（每层 2 次，传 hidden_dim 大小的数据），PCP 的通信更重（传的是 sequence 维度），但换来的是 prefill 计算量线性减少。对于超长 prompt（N 很大），计算节省 >> 通信开销。

#### 8.5 与其他 CP 方案对比

| | vLLM PCP（all_gather） | Ring Attention | 
|---|---|---|
| 通信模式 | all_gather 一次收集全部 KV | Ring P2P 逐步传递 KV 块 |
| KV Cache 存储 | 每卡存完整 KV（不省显存） | 每卡只存 1/CP 的 KV（省显存） |
| Attention 计算 | 各 rank 用局部 KV 计算局部 Q 的 attention | 各 rank 逐步累加 partial attention（online softmax） |
| 适用场景 | 加速 prefill 计算（KV Cache 放得下） | 超长序列 KV 放不下单卡 |
| 实现复杂度 | 低（标准 all_gather） | 高（ring schedule + log-sum-exp rescale） |
| 模型限制 | 当前只支持 MLA 模型 | 通用 |

#### 8.6 配置与限制

**配置参数**：
- `--prefill-context-parallel-size N`：PCP 并行度

**当前限制**（`PCPManager.validate_config()`）：
- 只支持 MLA 模型（DeepSeek V2/V3 等），因为实现绑定在 MLA attention backend
- 不支持 PP（pipeline parallelism）
- 不支持多模态模型
- 不支持 LoRA
- 不支持 Speculative Decoding
- Sparse MLA + PCP 不支持 CUDA Graph

#### 8.7 代码路径总结

```
配置入口:
  parallel_config.prefill_context_parallel_size → pcp_world_size

Group 管理:
  parallel_state.py → get_pcp_group() → PCP 通信组

Manager（调度层）:
  vllm/v1/worker/gpu/pcp_manager.py
    PCPManager.partition_batch()       → 切分 global batch 为 rank-local batch
    PCPManager.prepare_attn()          → 准备 local block_tables 和 slot_mappings
    PCPManager.prepare_slot_mappings() → 构建 gathered slot mappings（写完整 KV Cache）
    PCPManager.restore_for_sampling()  → all_gather hidden + 恢复全局 batch

算子层（KV Cache 写入）:
  vllm/model_executor/layers/attention/pcp.py
    _gather_prefill_cache_inputs()           → all_gather prefill KV 并写完整 cache
    maybe_gather_mla_latent_cache_inputs()   → MLA 专用：gather kv_c_normed + k_pe

Attention backend（MLA indexer）:
  vllm/v1/attention/backends/mla/indexer.py
    pcp_world_size 影响 slot_mapping 和 prefill chunk 的构建

ModelRunner 集成:
  vllm/v1/worker/gpu/model_runner.py
    maybe_partition_pcp_batch()        → forward 前切分
    pcp_manager.prepare_attn()         → 准备 attention metadata
    maybe_restore_pcp_for_sampling()   → forward 后恢复
```

### 8. WorkerWrapperBase 代理模式

```python
WorkerWrapperBase
  ├── 延迟初始化：init_worker() 时根据 worker_cls 字符串动态创建具体 Worker
  ├── 公共逻辑注入：init_device() 设置 vllm config 上下文
  ├── 插件扩展：worker_extension_cls 动态修改 __bases__
  └── 透传：__getattr__ 转发给内部 self.worker
```

### 9. RayExecutor Compiled DAG（PP=2, TP=4 示例）

```
input → [Worker 0,1,2,3]  (PP stage 0, TP=4 SPMD)
              ↓ NCCL/shm 传 intermediate tensors
        [Worker 4,5,6,7]  (PP stage 1, TP=4 SPMD) → output
```

构建代码核心：
```python
outputs = [input_data for _ in self.pp_tp_workers[0]]
for pp_rank, tp_group in enumerate(self.pp_tp_workers):
    outputs = [
        worker.execute_model_ray.bind(outputs[i])
        for i, worker in enumerate(tp_group)
    ]
```

两阶段执行：execute_model() 只存 scheduler_output，sample_tokens() 才触发 DAG。

### 10. 并行策略决策

```
模型能放进单卡？
  ├─ 是 → 单卡推理，DP 扩吞吐
  └─ 否 → TP 先切到单卡能放下
           TP=8 还不够？
             ├─ 是 → 加 PP（跨 stage 串行，增加延迟）
             └─ 否 → 超长序列 KV Cache 放不下？
                       ├─ 是 → 加 CP（切 sequence 维度）
                       └─ 否 → TP only

吞吐不够？
  → DP：多组 TP group 并行处理不同请求

通信带宽考虑：
  - TP 需要 NVLink（同节点）
  - PP 可跨节点（通信量小）
  - CP 需要高带宽（频繁传 KV）
```

---

## 关键代码位置

| 内容 | 文件 | 说明 |
|------|------|------|
| ColumnParallelLinear | `model_executor/layers/linear.py` | TP 按列切分 |
| RowParallelLinear | `model_executor/layers/linear.py` | TP 按行切分 + AllReduce |
| weight_loader | `model_executor/model_loader/` | 按 tp_rank 切分加载权重 |
| initialize_model_parallel | `distributed/parallel_state.py` | 创建 TP/PP/DP/EP groups |
| GroupCoordinator | `distributed/parallel_state.py` | 封装 group 通信方法 |
| custom_all_reduce | `distributed/device_communicators/custom_all_reduce.py` | NVLink 优化 AllReduce |
| Executor 抽象基类 | `v1/executor/abstract.py` | collective_rpc / execute_model 接口 |
| MultiprocExecutor | `v1/executor/multiproc_executor.py` | 单机多卡，共享内存 MQ |
| RayExecutor | `v1/executor/ray_executor.py` | 多机多卡，Compiled DAG |
| WorkerProc.worker_busy_loop | `v1/executor/multiproc_executor.py:997` | Worker RPC 主循环 |
| WorkerWrapperBase | `v1/worker/worker_base.py:187` | Worker 代理 + 延迟初始化 |
| GpuWorker.execute_model | `v1/worker/gpu_worker.py:1017` | PP stage 间 isend/irecv |
| GpuWorker.init_device | `v1/worker/gpu_worker.py:330` | NCCL group 建立 |
| DPCoordinator | `v1/engine/coordinator.py` | DP 请求分发 |
| cp_utils | `v1/worker/cp_utils.py` | DCP 工具函数 |

---

## Week 3 遗留问题解决情况

| 问题 | 状态 |
|------|------|
| 分布式场景多卡间 KV Cache 同步 | ✅ TP: 每卡独立存 num_kv_heads/tp_size；CP: 按 sequence 分散存储 |
| Pipeline Parallelism 的 KV Cache 管理 | ✅ 每个 PP stage 独立管理自己层的 KV Cache |

---

## 学习状态

- ✅ Day 1: TP 原理 + 并行 Linear 层实现
- ✅ Day 2: 权重加载机制 + Group 管理 + 通信封装
- ✅ Day 3: Pipeline Parallelism（PP）
- ✅ Day 4: Data Parallelism（DP）+ DP Coordinator
- ✅ Day 5: Context Parallelism（CP）
- ✅ Day 6: Executor 架构 + 多 Worker 启动流程
- ✅ Day 7: 总结

**Week 4 完成时间**: 2026-09-03

---

## 下周计划

**Week 5: 高级调度与优化**
- Day 1-2: Chunked Prefill 原理与实现
- Day 3: Prefix Caching（APC）机制
- Day 4-5: Speculative Decoding
- Day 6: CUDA Graph 深入
- Day 7: 综合实验

**重点问题**：
1. Chunked Prefill 的 chunk 大小如何确定？与 decode 如何交替？
2. Prefix Caching 的 hash 计算和命中逻辑细节？
3. Speculative Decoding 的 draft model 与 verify 流程？
