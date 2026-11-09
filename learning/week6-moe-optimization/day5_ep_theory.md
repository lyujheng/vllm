# Day 5: Expert Parallelism 理论基础

## 分布式并行策略对比

### Tensor Parallelism (TP) 回顾
```
Dense Model 的 TP:
  权重切分：W_up 按列切分到 GPU0, GPU1, ...
  输入广播：All tokens 到所有 GPU
  计算：每个 GPU 算部分维度
  输出聚合：All-Reduce (SUM)

通信：All-Reduce（每层 2 次，forward + backward）
```

### Expert Parallelism (EP) 原理
```
MoE Model 的 EP:
  Expert 切分：Expert 0-31 → GPU0, Expert 32-63 → GPU1
  Token 分发：根据 router 结果，token 发送到对应 GPU
  计算：每个 GPU 只算自己持有的 expert
  结果收回：All-to-All 通信把结果送回原 GPU

通信：All-to-All（每个 MoE 层 1 次）
```

## TP vs EP 的本质区别

| 维度 | Tensor Parallelism | Expert Parallelism |
|-----|-------------------|-------------------|
| 切分对象 | 权重矩阵（按维度） | Expert（按专家） |
| 输入分布 | 所有 GPU 收到所有 token | Token 按路由分发 |
| 通信原语 | All-Reduce | All-to-All |
| 负载均衡 | 天然均衡（切分均匀） | 依赖 router 输出 |
| 适用场景 | Dense layers | MoE layers |

## vLLM 的 EP 通信实现

vLLM 的 EP 有两种通信模式，取决于是否启用 DP：

### EP 与 DP 的约束关系

#### 约束 1：ep_size 由其他并行度推导，不独立设置

```python
# config.py:1113-1120, 1236
# 当 EP 启用时：
flatten_tp_size = dp_size * pcp_size * tp_size  # 所有 GPU 展平
ep_size = flatten_tp_size                        # EP = 展平后的总 GPU 数

# 即：ep_size = tp_size × dp_size × pcp_size
```

#### 约束 2：EP 启用需要手动开启

```python
# config.py:1204-1206
use_ep = (
    dp_size * pcp_size * tp_size > 1              # 至少有并行度 > 1
    and vllm_parallel_config.enable_expert_parallel  # 需要 --enable-expert-parallel
)
```

#### 约束 3：EP 启用时，MoE 层的 TP 降为 1

```python
# config.py:1234-1245
# EP 启用后，对 MoE 层：
ep_size = tp_size   # 展平后的 tp_size 全部给 EP
tp_size = 1         # MoE 层不再做 tensor parallelism
tp_rank = 0
```

MoE 层中 **EP 取代了 TP**：每个 expert 完整存储在一个 GPU 上，不做 tensor 切分。

#### 约束 4：experts 数量不必被 ep_size 整除

```python
# expert_map_manager.py:66-69
base_experts = global_num_experts // ep_size
remainder = global_num_experts % ep_size
local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts
```

vLLM 支持不均匀分配，但实际中大多数模型的 expert 数量能被 ep_size 整除。

#### 具体配置示例

```python
# 示例 1：4 GPU，纯 EP
--tensor-parallel-size=4 --enable-expert-parallel
→ 非 MoE 层：TP=4（tensor 切分）
→ MoE 层：  EP=4, TP=1（每 GPU 持有 1/4 experts，完整存储）

# 示例 2：4 GPU，EP + DP
--tensor-parallel-size=2 --data-parallel-size=2 --enable-expert-parallel
→ 非 MoE 层：TP=2（tensor 切分），DP=2（两个独立 engine）
→ MoE 层：  EP=4, TP=1（每 GPU 持有 1/4 experts）

# 示例 3：4 GPU，不启用 EP
--tensor-parallel-size=2 --data-parallel-size=2
→ 非 MoE 层：TP=2, DP=2
→ MoE 层：  TP=4, EP=1（每 GPU 持有所有 experts 的 1/4 tensor 切片）
```

**核心区别**：
- **启用 EP**：MoE 层按 expert 切分（每个 GPU 持有完整的部分 experts）
- **不启用 EP**：MoE 层按 tensor 切分（每个 GPU 持有所有 experts 的部分 tensor）

### 模式 1：纯 EP（dp_size=1，无 DP 切分）

```python
# 配置：--tensor-parallel-size=4 --enable-expert-parallel
ep_size = 4, dp_size = 1

特点：
  - 所有 GPU 接收相同的完整 batch（N tokens）
  - 不需要 dispatch（每个 GPU 已有所有 tokens）
  - 每个 GPU 只计算自己的 experts，输出部分结果
  - 1 次 All-Reduce 聚合所有 GPU 的部分结果（SUM）
  - 所有 GPU 得到相同的完整输出
  
通信量：
  - All-Reduce: N × D × 2 bytes
  
代码：make_moe_prepare_and_finalize_no_dp_ep()
      (all2all_utils.py:167)
      tensor_model_parallel_all_reduce(fused_output)
      (moe_runner.py:458)
```

### 模式 2：EP + DP（dp_size>1，有 DP 切分）

```python
# 配置：--tensor-parallel-size=2 --data-parallel-size=2 --enable-expert-parallel
ep_size = 4 (2×2), dp_size = 2

特点：
  - 多个 engine 实例，每个处理不同请求
  - tokens 按 DP 切分，每个 engine 持有部分 tokens
  - 需要 AllGatherv + ReduceScatterv 通信（v = variable，各 rank token 数不同）
  
代码：make_moe_prepare_and_finalize_naive_dp_ep()
      (all2all_utils.py:161)
```

vLLM 在 **EP + DP 模式** 下使用 **AllGatherv + ReduceScatterv**（v = variable，因各 rank token 数不同）而非传统的 All-to-All，避免复杂的 token 重排和 send/recv counts 计算。

### 通信模式（EP + DP）

```
Step 0: Token 初始分配
  GPU0: [token_0, token_1, token_2, token_3]  # 341 个 token
  GPU1: [token_4, token_5, token_6, token_7]  # 341 个 token
  GPU2: [token_8, token_9, ...]               # 342 个 token

Step 1: AllGatherv (Dispatch) - 收集所有数据（各 rank token 数不同，需传 sizes）
  GPU0: [all 1024 tokens, topk_ids, topk_weights]  # 完整数据
  GPU1: [all 1024 tokens, topk_ids, topk_weights]  # 完整数据
  GPU2: [all 1024 tokens, topk_ids, topk_weights]  # 完整数据

Step 2: Local Expert Compute - 每个 GPU 只计算自己负责的 expert
  Router 结果：
    token_0 → expert [2, 5] (权重 [0.6, 0.4])
    token_1 → expert [25, 40]
    token_2 → expert [3, 50]
    ...

  GPU0 (expert 0-21):
    expert_outputs[0] = 0.6*expert_2(x) + 0.4*expert_5(x)  # 完整计算
    expert_outputs[1] = 0                                   # 不负责 expert 25,40
    expert_outputs[2] = 0.6*expert_3(x)                     # 部分计算
    ...
  
  GPU1 (expert 22-42):
    expert_outputs[0] = 0                                   # 不负责 expert 2,5
    expert_outputs[1] = 0.7*expert_25(x) + 0.3*expert_40(x)
    expert_outputs[2] = 0
    ...
  
  GPU2 (expert 43-63):
    expert_outputs[0] = 0
    expert_outputs[1] = 0
    expert_outputs[2] = 0.4*expert_50(x)                    # 部分计算
    ...

Step 3: ReduceScatterv (Combine) - 求和并按各 rank 原始 token 数分散回原 GPU
  GPU0 收回 token 0-340:
    final[0] = expert_outputs_gpu0[0] + expert_outputs_gpu1[0] + expert_outputs_gpu2[0]
             = 0.6*expert_2(x) + 0.4*expert_5(x) + 0 + 0  # 完整结果
    final[2] = 0.6*expert_3(x) + 0 + 0.4*expert_50(x)     # 完整结果
  
  GPU1 收回 token 341-681
  GPU2 收回 token 682-1023
```

### 为什么需要 Reduce（求和）？

**因为每个 token 的完整输出 = 多个 expert 的输出之和**：

```python
# MoE 公式
output = Σ(topk_weights[i] * expert_i(hidden_states))

# token_2 选了 [expert_3, expert_50]，分布在不同 GPU
# GPU0 计算 expert_3 的贡献
# GPU2 计算 expert_50 的贡献
# 需要求和才能得到完整输出
```

### AllGatherv + ReduceScatterv vs All-to-All

| 维度 | AllGatherv + ReduceScatterv | All-to-All |
|-----|---------------------------|-----------|
| 实现复杂度 | 简单（不需要 token 重排，支持变长） | 复杂（需要计算 send/recv counts） |
| 内存开销 | 高（每个 GPU 临时存储全部 token） | 低（只传输必要的 token） |
| 通信量 | 相同 | 相同 |
| 计算效率 | 每个 GPU 遍历所有 token（条件判断） | 每个 GPU 只处理相关 token |
| 变长支持 | 原生支持（通过 sizes 参数） | 需要额外处理 |
| vLLM 选择 | ✅ naive 默认实现 | DeepEP P2P 实现 |

### 通信量分析（基于 vLLM 实际实现）

以下分析适用于 **EP + DP**、**EP + PCP**、**EP + SP** 等所有 tokens 分散在多个 rank 的场景。

只要满足 `dp_size > 1` 或 `pcp_size > 1` 或 `is_sequence_parallel`，
都走 AllGatherv + ReduceScatterv 路径（`naive_dp_ep`），区别仅在通信 group：

```python
# all2all.py:53-58
def _get_comm_group(self, is_sequence_parallel):
    if is_sequence_parallel:
        return get_ep_group()     # SP 模式：跨所有 EP ranks
    if self.dp_world_size > 1:
        return get_dp_group()     # DP 模式：跨 DP ranks
    return get_pcp_group()        # PCP 模式：跨 PCP ranks
```

本质原因相同：PCP 按序列切分 tokens，DP 按请求切分 tokens，
不同 rank 持有不同的 tokens → 需要 AllGatherv 收集 → ReduceScatterv 返回。

vLLM 使用 **AllGatherv + ReduceScatterv** 模式（[all2all.py:101-150](vllm/distributed/device_communicators/all2all.py#L101-L150)）

#### 阶段 1：dispatch（AllGatherv）

```python
# 每个 rank 收集所有 rank 的数据
gathered_tensors = dist_group.all_gatherv(
    [hidden_states, topk_weights, topk_ids],
    dim=0,
    sizes=sizes,
)
```

**重要理解**：
- **EP 切分的是 expert 维度**（模型权重）：GPU0 持有 expert 0-31，GPU1 持有 expert 32-63
- **DP 切分的是 token 维度**（输入数据）：dispatch 前，GPU0 持有 token 0-N，GPU1 持有 token N-2N
- **AllGatherv 是对 token 维度做全局收集**：让每个 GPU 拥有所有 tokens 的数据（v 表示各 rank token 数可不同）
- **expert 维度仍然是切分的**：每个 GPU 只计算自己负责的 experts

为什么需要 AllGatherv？
  - GPU0 的 token 可能路由到 GPU1 的 experts
  - GPU1 的 token 可能路由到 GPU0 的 experts
  - 每个 GPU 需要看到所有 tokens，才能处理路由给自己 experts 的请求

**通信量（每个 rank）**：
```
假设：
  - N tokens per rank（dispatch 前）
  - D hidden dimension
  - K top-k experts
  - P ranks（ep_size）

AllGatherv 通信量（每个 rank 接收）：
  = (P-1) * N * D * sizeof(fp16)           # hidden_states：expert 计算的输入
  + (P-1) * N * K * sizeof(fp16)           # topk_weights：kernel 内加权用
  + (P-1) * N * K * sizeof(int32)          # topk_ids：判断哪些 token 路由到本 rank
  
  ≈ (P-1) * N * (D + K*3 + K*4) bytes     # fp16=2 bytes, int32=4 bytes
  ≈ (P-1) * N * D * 2 bytes                # 主导项是 hidden_states

为什么 topk_weights 也要通信？
  - vLLM 的权重乘法在 expert kernel 内部完成（fused_moe.py:286-287）
  - 每个 rank 输出的是加权后的部分结果：weight[i] * Expert_i(x)
  - ReduceScatterv 只做求和，不做加权
  - 如果不传 weights，每个 rank 无法正确加权，reduce 结果错误
```

#### 阶段 2：本地 Expert 计算

```
每个 rank 只计算自己负责的 experts
  - 输入：ΣN_i tokens（AllGatherv 后，各 rank token 数之和）
  - 计算：只处理分配给本 rank experts 的 tokens
  - 输出：P*N * D（所有 tokens 的结果，但只有本 rank experts 的部分非零）
```

#### 阶段 3：combine（ReduceScatterv）

```python
# 对 hidden_states 进行 reduce-scatter
hidden_states = dist_group.reduce_scatterv(
    hidden_states, 
    dim=0, 
    sizes=sizes
)
```

**通信量（每个 rank）**：
```
Reduce-Scatterv 通信量（每个 rank 发送）：
  = (P-1) * N * D * sizeof(fp16)           # 发送给其他 ranks
  = (P-1) * N * D * 2 bytes

注意：ReduceScatterv = Reduce + Scatter（按各 rank 原始 token 数分散）
  - Reduce：对每个 token 位置，求和所有 ranks 的 expert 输出
  - Scatter：每个 rank 只保留自己负责的 N tokens
```

#### 总通信量

**每个 MoE layer，每个 rank**：
```
总通信量 = AllGatherv + ReduceScatterv
         = (P-1)*N*D*2 + (P-1)*N*D*2
         = 2*(P-1)*N*D*2 bytes
         ≈ 4*P*N*D bytes（当 P 较大时）

示例：P=4 ranks, N=1024 tokens/rank, D=4096
  = 4 * 4 * 1024 * 4096 * 2
  = 134 MB per layer per rank
  
对比传统 All-to-All（token-level shuffle）：
  - vLLM 的 AllGatherv + ReduceScatterv 是 tensor-level 操作
  - 通信量相同，但实现更简单（利用 NCCL 原语）
```

#### 通信量的关键因素

1. **与 ep_size (P) 成正比**：
   - P=2: 2*(P-1) = 2，通信量 = 2*N*D*2
   - P=4: 2*(P-1) = 6，通信量 = 6*N*D*2
   - P=8: 2*(P-1) = 14，通信量 = 14*N*D*2

2. **与 tokens per rank (N) 成正比**：
   - batch size 越大，通信量越大

3. **与 hidden dimension (D) 成正比**：
   - 模型越大，通信量越大

4. **与 top-k (K) 无关**（主导项）：
   - topk_weights 和 topk_ids 的通信量相对 hidden_states 很小
```

## DP 在 MoE 中的双层架构

### 非 MoE 层和 MoE 层的不同处理

基于前面的理解，vLLM 对不同层采用了**不同的并行策略**：

#### 非 MoE 层（Attention、普通 FFN）

```
每个 DP rank = 独立的 engine 实例

示例：TP=2, DP=2（4 GPU）
  Engine 0 (DP rank 0):
    - GPU 0, GPU 1 做 TP
    - 独立的 KV Cache
    - 独立的 Scheduler
    - 处理自己的请求队列
  
  Engine 1 (DP rank 1):
    - GPU 2, GPU 3 做 TP
    - 独立的 KV Cache
    - 独立的 Scheduler
    - 处理自己的请求队列

关键：两个 engine 之间无通信，完全隔离
```

#### MoE 层（Expert Parallel）

```
DP 参与 expert 切分：ep_size = tp_size × dp_size × pcp_size

示例：TP=2, DP=2, EP=True（4 GPU）
  ep_size = 2 × 2 × 1 = 4
  
  MoE 层的并行视图：
    - GPU 0: expert 0-15（假设 64 experts）
    - GPU 1: expert 16-31
    - GPU 2: expert 32-47
    - GPU 3: expert 48-63
  
  关键：4 个 GPU 跨 DP rank 协作
    - AllGatherv: 每个 GPU 收集所有 token（各 rank token 数可不同）
    - 本地计算：每个 GPU 只计算自己负责的 experts
    - ReduceScatterv: 求和并按原始 token 数分散回原 DP rank

通信模式：
  - Engine 0 的 GPU 0-1 和 Engine 1 的 GPU 2-3 
    在 MoE 层需要通信
  - 但在 Attention 层完全隔离
```

### 为什么 vLLM 采用这种双层架构？

**优势**：
1. **非 MoE 层高吞吐**：多个独立 engine 并行处理不同请求，无通信开销
2. **MoE 层细粒度并行**：利用所有 GPU 切分 experts，降低单 GPU 内存压力
3. **灵活的资源配置**：可以根据 GPU 数量和 expert 数量灵活调整 DP 和 EP

**代价**：
1. **MoE 层通信开销**：跨 DP rank 的 AllGatherv + ReduceScatterv
2. **实现复杂度**：需要区分两类层的并行策略

### Data Parallel (DP) 在 vLLM MoE 中的真实含义

**重要纠正**：在 vLLM 的 MoE 实现中，DP **不是独立副本**，而是**参与 Expert 的切分**！

#### 核心机制（[config.py:1170-1202](vllm/model_executor/layers/fused_moe/config.py#L1170-L1202)）

```python
# 当启用 EP 时，expert 切分维度计算：
ep_size = tp_size * dp_size * pcp_size

# 然后将 TP 设为 1（非 MoE 层仍用原始 TP）
tp_size_for_moe = 1
```

**关键引用**：
> "MoE layers will be sharded according to the product of the tensor, prefill-context, and data parallel sizes." — [parallel.py:130-131](vllm/config/parallel.py#L130-L131)

#### 具体例子（从源码注释）

**例子 1：TP=1, DP=2, EP=True（2 GPU）**
```
device 0: TP={1,0} DP={2,0} EP={2,0}
device 1: TP={1,0} DP={2,1} EP={2,1}

解释：
  - 有 2 个 engine 实例（DP=2）
  - experts 在 2 个设备间切分（ep_size = 1×2×1 = 2）
  - 每个 GPU 负责一半的 experts
  - 非 MoE 层：每个 engine 独立计算（无 TP）
  - MoE 层：2 个 GPU 协作，通过 AllGatherv + ReduceScatterv 通信
```

**例子 2：TP=2, DP=2, EP=True（4 GPU）**
```
device 0: TP={1,0} DP={2,0} EP={4,0}
device 1: TP={1,0} DP={2,0} EP={4,1}
device 2: TP={1,0} DP={2,1} EP={4,2}
device 3: TP={1,0} DP={2,1} EP={4,3}

解释：
  - 有 2 个 engine 实例（DP=2）
  - experts 在 4 个设备间切分（ep_size = 2×2×1 = 4）
  - 每个 GPU 负责 1/4 的 experts
  - 非 MoE 层：每个 engine 内部 2 GPU 做 TP（device 0-1 一组，device 2-3 一组）
  - MoE 层：4 个 GPU 跨 engine 协作，做 EP 通信
```

**例子 3：TP=4, DP=2, EP=False（4 GPU）**
```
device 0: TP={4,0} DP={2,0} EP={1,0}
device 1: TP={4,1} DP={2,0} EP={1,0}
device 2: TP={4,2} DP={2,1} EP={1,0}
device 3: TP={4,3} DP={2,1} EP={1,0}

解释：
  - 有 2 个 engine 实例（DP=2）
  - 没有启用 EP，MoE 使用 TP 切分
  - 非 MoE 层：所有 4 GPU 做 TP（跨 engine）
  - MoE 层：所有 4 GPU 做 TP（跨 engine）
```

### DP 在 MoE 中的作用总结

#### 非 MoE 层（Attention、FFN）
- **DP rank 之间完全隔离**
- 每个 DP rank 是独立的 engine 实例
- 独立的 KV Cache、Scheduler、请求队列
- 类似传统的数据并行（多副本）

#### MoE 层
- **DP 参与 expert 切分**：`ep_size = tp_size × dp_size × pcp_size`
- DP rank 之间**需要通信**（AllGatherv + ReduceScatterv）
- 跨 DP rank 的 GPU 协作处理 experts
- **不是独立副本**，而是更细粒度的 expert 并行

### 为什么 vLLM 这样设计？

**优势**：
1. **灵活的并行策略**：非 MoE 层用独立副本（高吞吐），MoE 层用细粒度 EP（降低内存压力）
2. **充分利用 GPU**：当 GPU 数量 > experts 时，DP 可以进一步切分 experts
3. **负载均衡**：多个 engine 实例可以并行处理不同请求，提高吞吐

**代价**：
1. **MoE 层通信**：DP rank 之间需要 AllGatherv + ReduceScatterv，增加通信开销
2. **实现复杂度**：需要区分 MoE 层和非 MoE 层的并行策略

### 应用场景

**场景 1：GPU 数量 > experts，想要更细粒度的 EP**
```
Mixtral-8x7B（8 experts），16 GPU
  - TP=1, DP=2, EP=True
  - ep_size = 1×2×1 = 2
  - 每个 expert 分配到 16/8 = 2 GPU？❌ 错误理解
  
正确理解：
  - ep_size = 2 表示 experts 在 2 个 GPU 间切分（但这里有 16 GPU）
  - 实际上需要 TP=8, DP=2, EP=True
  - ep_size = 8×2×1 = 16，每个 GPU 负责 8/16 的 experts
```

**场景 2：多 engine 实例 + EP**
```
32 GPU，Mixtral-8x7B
  - TP=8, DP=4, EP=True
  - 4 个 engine 实例（DP=4）
  - ep_size = 8×4×1 = 32
  - 每个 GPU 负责 8/32 = 1/4 个 expert
  - 非 MoE 层：4 个独立副本，各自处理不同请求
  - MoE 层：32 GPU 协作，细粒度 EP
```

### Lockstep 要求

**问题**：为什么 MoE + DP 必须同步步进？

**根因**：AllGatherv / ReduceScatterv 是集合通信，所有参与的 rank 必须同时调用，
缺一个就死锁。与处理的 request 是否相同无关。

```
EP + DP, 4 个 GPU:
  Rank 0: 持有 Expert 0-63,   处理 request A
  Rank 1: 持有 Expert 64-127,  处理 request B
  Rank 2: 持有 Expert 128-191, 处理 request C
  Rank 3: 持有 Expert 192-255, 处理 request D

MoE 层 forward:
  1. 各 rank 的 token 经 router 选中的 expert 可能在其他 rank 上
     例如 Rank 0 的 token 选中 Expert 70（在 Rank 1）
  2. AllGatherv：4 个 rank 把各自的 token 汇聚到一起（各 rank token 数可不同）
  3. 各 rank 用本地 expert 计算分配到自己的 token
  4. ReduceScatterv：把结果送回各 token 所属的 rank

死锁场景（不同步时）：
  Rank 0 step 5: 调用 AllGatherv() ← 等待 Rank 1 也调用
  Rank 1 step 3: 还在处理别的请求，没到 MoE 层
  → Rank 0 永远等下去 → 死锁

Lockstep 保证的是：
  ✗ 所有 rank 处理相同的请求（DP 就是处理不同请求的）
  ✓ 所有 rank 在同一时刻执行同一次集合通信调用
```

**vLLM 实现机制**（core.py DPEngineCoreProc）：
  - 有请求的 rank 正常执行 forward
  - 没请求的 rank 执行 dummy batch（空 batch）保持 step 对齐
  - 每 32 步做一次 all-reduce 检查全局是否还有 unfinished requests
  - 所有 rank 都没请求时才统一停止（wave 结束）

## EP + TP 组合策略

### vLLM 的 EP 实现：TP flatten 为 EP

**关键设计**：vLLM 开启 EP 后，MoE 层的 TP 被 flatten 成 EP。
每个 expert 完整存储在一个 rank 上，没有权重切分。

```python
# config.py FusedMoEParallelConfig:
# 开 EP 时，TP 被展平为 EP
ep_size = dp_size * pcp_size * tp_size   # 所有 GPU 参与 EP
tp_size = 1                                # MoE 层内不再做 TP
tp_rank = 0
```

### 示例：DP=2, TP=4, 256 experts

```
非 MoE 层（Attention 等）：
  DP rank 0: GPU 0-3，TP=4 切分 Attention 权重
  DP rank 1: GPU 4-7，TP=4 切分 Attention 权重

MoE 层（EP flatten）：
  ep_size = 2 × 4 = 8
  GPU 0: Expert 0-31   （完整存储，无权重切分）
  GPU 1: Expert 32-63
  GPU 2: Expert 64-95
  GPU 3: Expert 96-127
  GPU 4: Expert 128-159
  GPU 5: Expert 160-191
  GPU 6: Expert 192-223
  GPU 7: Expert 224-255

通信：
  - Attention 层: TP All-Reduce（DP rank 内部 4 个 GPU）
  - MoE 层: AllGatherv + ReduceScatterv（所有 8 个 GPU 参与）
```

### 不开 EP 的 DP + TP

```
不开 --enable-expert-parallel 时：
  MoE 层仍然被 flatten 到 dp_size × tp_size 个 GPU 上
  但走的是 naive AllGatherv+ReduceScatterv（非 DeepEP 等高效 all2all）
  ep_size = 1，tp_size = dp_size * tp_size

  即：不开 EP 时，MoE 层用 TP 跨所有 GPU 切分权重（每个 expert 被切分）
  开 EP 后，MoE 层用 EP 跨所有 GPU 切分 expert（每个 expert 完整）
```

### 选择策略

| 场景 | 推荐配置 | 原因 |
|-----|---------|------|
| 小 MoE (8 experts) | 不开 EP，纯 TP | expert 数量少，EP 的 all2all 通信不划算 |
| 大 MoE (64+ experts) | 开 EP | expert 完整存储，routing 通信更高效 |
| 超大 expert 单体 | 不开 EP 或 EP + 更大 TP | 单个 expert 放不下一张卡，需要 TP 切分 |
| 内存受限 | 开 EP | 减少单 GPU 的 expert 数量 |

## vLLM 代码路径

### EP 相关初始化
- EP Process Group: `vllm/distributed/parallel_state.py` → `_EP` group，
  通过 `init_model_parallel_group()` 创建
- DP Process Group: 同文件 `_DP` group，MoE DP lockstep 通信用这个 group
- FusedMoE 并行配置: `vllm/model_executor/layers/fused_moe/config.py` →
  `FusedMoEParallelConfig.make()`，决定 EP/TP flatten 逻辑

### FusedMoE 中的通信实现
- All2All 管理器: `vllm/model_executor/layers/fused_moe/all2all_utils.py` →
  `get_all2all_manager()`，根据配置选择通信后端
- 通信后端选择:
  - DeepEP low-latency: CUDA Graph 兼容，适合 decode
  - DeepEP high-throughput: 不兼容 CUDA Graph，适合 prefill
  - Naive AllGatherv+ReduceScatterv: 不开 EP 时的回退方案
- Modular kernel: `vllm/model_executor/layers/fused_moe/modular_kernel.py` →
  `FusedMoEModularKernel`，组合 prepare/finalize（通信）和 experts（计算）

### Engine Core 选择
- `vllm/v1/engine/core.py`:
  - MoE + DP → `DPEngineCoreProc`（lockstep 同步）
  - 非 MoE + DP → `EngineCoreProc`（完全独立，dp_size 设为 1）

## 实验准备（Day 6）

### 环境要求
- [ ] 2+ GPUs
- [ ] vLLM EP 功能已实现（检查 `--help` 是否有 EP 参数）
- [ ] MoE 模型已下载

### 实验设计（exp4）
对比配置：
1. **Single GPU**: 单卡跑完整 MoE（baseline）
2. **EP=2**: 2 卡 expert 并行
3. **TP=2**: 2 卡 tensor 并行（如果支持）

预期结果：
- EP 降低单卡内存 → 提高 max_num_seqs
- EP 引入通信开销 → 可能增加延迟
- TP 内存降低小于 EP（Attention 也需切分）

## 负载均衡问题与解决方案

### 问题：Expert 负载不均

**场景**：
```
假设 8 个 expert，EP=2（每个 GPU 4 个 expert）
Router 结果（1024 tokens）：
  Expert 0: 500 tokens  ← GPU0 负载重
  Expert 1: 300 tokens  ← GPU0
  Expert 2: 100 tokens  ← GPU0
  Expert 3: 50 tokens   ← GPU0
  Expert 4: 30 tokens   ← GPU1 负载轻
  Expert 5: 20 tokens   ← GPU1
  Expert 6: 15 tokens   ← GPU1
  Expert 7: 9 tokens    ← GPU1

结果：
  GPU0 处理 950 tokens，GPU1 处理 74 tokens
  → GPU1 空闲等待 GPU0，通信不均衡
```

### 解决方案 1：Load Balancing Loss

**原理**：训练时引入辅助 loss，鼓励 router 均匀分配 token

```
Auxiliary Loss = α * Σ(expert_i 的负载方差)

训练目标 = Task Loss + Load Balancing Loss
```

**效果**：
- 模型学习到更均匀的路由策略
- 生产环境无需额外操作

**局限**：
- 只能在训练时优化
- 推理时 router 已固定

### 解决方案 2：Expert Capacity（容量限制）

**原理**：限制每个 expert 最多处理的 token 数

```
Capacity = (total_tokens / num_experts) * capacity_factor

示例（1024 tokens, 8 experts, capacity_factor=1.25）：
  Capacity = (1024 / 8) * 1.25 = 160 tokens per expert

如果 Expert 0 被路由了 500 tokens：
  - 只处理前 160 个
  - 剩余 340 个 tokens 被丢弃或路由到第二选择
```

**Capacity Factor 权衡**：
- `factor=1.0`：严格均衡，但可能丢弃很多 token
- `factor=1.5`：允许 50% 冗余，减少丢弃
- `factor=2.0`：更宽松，但负载不均问题依然存在

**vLLM 实现**：
- 文件：`vllm/model_executor/layers/fused_moe/fused_moe.py`
- 参数：部分模型配置中有 `expert_capacity` 相关逻辑

### 解决方案 3：动态 Expert 分配（EPLB）

**原理**：运行时根据实际负载，将热门 expert 复制到多个 GPU

```
监控每个 expert 的调用频率 → 把热门 expert 复制到多个 GPU
```

**vLLM 已实现**：
- EPLB (Expert Parallelism Load Balancing):
  - 配置: `--num-redundant-experts N`，每个 rank 额外存储 N 个冗余热门 expert
  - 策略: `vllm/distributed/eplb/policy/default.py`
  - 执行: `vllm/distributed/eplb/rebalance_execute.py`
- Elastic EP (弹性 Expert Parallelism):
  - 运行时动态调整 EP rank 数量
  - `vllm/distributed/elastic_ep/elastic_state.py`

**注意**：
- 冗余 expert 需要额外内存
- EPLB 需要 `RoutedExpertsCapturer` 持续采集 routing 数据
  （`vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`）

## All-to-All 通信的实现细节

### vLLM 的实际通信方式

**注意**：vLLM 的 EP 通信**不是**原生 `torch.distributed.all_to_all_single`，
而是根据场景选择不同的实现：

**1. DeepEP（推荐，高性能）**：
```
DeepEP Low-Latency:
  - 适合 decode 阶段（小 batch）
  - CUDA Graph 兼容
  - 延迟优化

DeepEP High-Throughput:
  - 适合 prefill 阶段（大 batch）
  - 不兼容 CUDA Graph
  - 带宽优化
```

**2. Naive AllGatherv + ReduceScatterv（回退方案）**：
```
不开 EP 或不支持 DeepEP 时使用：
  Dispatch:  AllGatherv — 各 rank 的 token 汇聚到所有 rank（支持变长）
  Combine:   ReduceScatterv — expert 计算结果按各 rank 原始 token 数归约

与原生 all_to_all_single 的区别：
  - all_to_all: 点对点，每个 rank 只发需要的数据给目标 rank
  - AllGatherv: 广播式，所有 rank 收到所有数据（冗余但简单，支持变长）
```

### 通信流程（以 AllGatherv + ReduceScatterv 为例）

```
Step 1 - Dispatch (AllGatherv):
  Rank 0 有 300 tokens，Rank 1 有 500 tokens（各 rank token 数不同）
  AllGatherv 后：两个 rank 都拿到全部 800 tokens

Step 2 - Expert Compute:
  Rank 0: 用本地 experts 计算分配到自己的 tokens
  Rank 1: 用本地 experts 计算分配到自己的 tokens
  （通过 topk_ids 过滤，只计算路由到本地 expert 的 token）

Step 3 - Combine (ReduceScatterv):
  各 rank 的计算结果归约，每个 token 的结果回到原始 rank（按原始 token 数分散）
```

### Variable-length 支持

EP 场景下每个 GPU 的 token 数不同，AllGather 使用 `all_gatherv`：

```python
# all2all.py _get_sizes():
# 通过 DPMetadata 获取各 rank 的 token 数
dp_metadata = get_forward_context().dp_metadata
sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
# 例如 sizes = [300, 500] — 各 rank token 数不同
```

### 通信与计算 Overlap

DeepEP 等高效后端支持通信与计算重叠：
- 通信和 expert 计算可以 pipeline 化
- Low-latency 模式下通过 CUDA Graph capture 减少 kernel launch 开销

## Expert Capacity 的实现机制

### Capacity 计算

```python
num_tokens = len(topk_ids)  # 总 token 数
num_experts = 64
capacity_factor = 1.25

capacity_per_expert = (num_tokens // num_experts) * capacity_factor
```

### Token 丢弃策略

**策略 1：First-come-first-serve**
```python
for token_idx, expert_id in enumerate(topk_ids):
    if expert_counts[expert_id] < capacity:
        assign_token_to_expert(token_idx, expert_id)
        expert_counts[expert_id] += 1
    else:
        # 超过容量，丢弃或降级
        if use_second_expert:
            fallback_to_second_choice(token_idx)
        else:
            drop_token(token_idx)  # 输出 zero tensor
```

**策略 2：概率丢弃（随机采样）**
```python
# 如果 expert_i 收到 N 个 token，但 capacity=M
# 随机采样 M 个，保证公平性
selected_indices = random.sample(range(N), M)
```

**vLLM 的选择**：
- 默认无 capacity 限制（允许负载不均）
- 某些模型（如 DeepSeek-V2）训练时使用了 capacity，推理时保持兼容

## EP 的内存与通信开销分析

### 内存节省

**Single GPU（无 EP）**：
```
模型内存 = Attention weights + All Expert weights
         = A + E * W_expert

假设：
  A = 10 GB (Attention 层权重)
  E = 64 experts
  W_expert = 2 GB per expert
  Total = 10 + 64 * 2 = 138 GB  ← 单卡装不下
```

**EP=8**：
```
每个 GPU 的模型内存 = A + (E/8) * W_expert
                    = 10 + 8 * 2 = 26 GB  ← 单卡可以装下

节省：138 → 26 GB (81% 节省)
```

### 通信开销

**每个 MoE layer 的延迟构成**：
```
Total Latency = Dispatch All-to-All + Compute + Combine All-to-All

假设：
  - N = 1024 tokens per GPU
  - D = 4096 hidden_dim
  - K = 8 top-k
  - P = 8 GPUs
  - 带宽 = 300 GB/s (NVLink)

Dispatch All-to-All 时间：
  数据量 = N * K * D * 2 bytes = 1024 * 8 * 4096 * 2 = 64 MB
  时间 = 64 MB / 300 GB/s ≈ 0.2 ms

Compute 时间（每个 GPU 处理 N*K tokens）：
  FLOPs = N * K * (2 * D * intermediate) * 2 (GEMM1 + GEMM2)
        = 1024 * 8 * (2 * 4096 * 14336) * 2
        ≈ 2 TFLOPs
  时间 = 2 TFLOPs / 300 TFLOP/s (H100) ≈ 6.7 ms

Combine All-to-All：0.2 ms

Total ≈ 0.2 + 6.7 + 0.2 = 7.1 ms
通信占比 = 0.4 / 7.1 ≈ 5.6%
```

**结论**：
- 计算密集型场景（大 batch），通信占比小 → EP 高效
- 小 batch decode 阶段，通信占比增加 → EP 收益降低

## EP 适用场景总结

### 适合 EP 的场景

1. **大 MoE 模型**（64+ experts）
   - Expert 数量多 → 并行度高
   - 单卡装不下所有 expert

2. **Prefill 阶段**（大 batch）
   - Token 数多 → 通信占比小
   - 计算密集 → GPU 利用率高

3. **长序列生成**
   - Decode 步数多 → 分摊通信开销

4. **Expert 负载均衡**
   - 模型训练时用了 Load Balancing Loss
   - Router 输出较均匀

### 不适合 EP 的场景

1. **小 MoE 模型**（<16 experts）
   - 并行度低，EP 收益小
   - TP 可能更高效

2. **Decode 阶段**（小 batch）
   - Token 数少（可能只有几个）
   - 通信占比高 → 延迟增加

3. **Expert 负载极度不均**
   - 某些 expert 成为瓶颈
   - GPU 利用率低

4. **跨节点部署**
   - All-to-All 通信慢（依赖 InfiniBand）
   - 单节点内 NVLink 更适合 EP

## 学习检查点

- [ ] 能解释 All-to-All vs All-Reduce 的区别
- [ ] 理解为什么 MoE 更适合 EP 而非 TP
- [ ] 能画出 EP All-to-All 的 token 流动图
- [ ] 理解 DP Attention 和 Lockstep 的必要性
- [ ] 知道 EP + TP 的组合配置方法
- [ ] 理解负载不均衡的原因和 Expert Capacity 机制
- [ ] 能计算 EP 的内存节省和通信开销
- [ ] 知道 EP 适用和不适用的场景
