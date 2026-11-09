# EPLB（Expert-Parallel Load Balancing）背景与原理

## 1. 问题背景：MoE + EP 的负载不均

### MoE 推理回顾

MoE（Mixture of Experts）模型中，每个 MoE 层有 N 个 expert，每个 token 经 router 只选 top-K 个 expert 执行。

- Qwen2-57B-A14B：64 experts，top-8
- DeepSeek-V3：256 experts，top-8
- Mixtral-8x7B：8 experts，top-2

### Expert Parallelism（EP）

EP 将 expert 分到不同 GPU 上。以 64 expert / EP=8 为例，每卡持有 8 个 expert。

推理流程（vLLM 默认的 allgather_reducescatter 后端）：
```
dispatch 阶段：All-Gather，每张卡收集所有卡的 hidden_states + topk_ids
  → 各 GPU 根据 topk_ids 本地计算自己持有的 expert
  → combine 阶段：Reduce-Scatter，将结果聚合并分发回各卡
```

> 注意：vLLM 中 EP 的默认通信不是 All-to-All，而是 All-Gather + Reduce-Scatter。
> All-Gather 让每张卡拿到全部 token，各卡只计算自己持有的 expert 对应部分，
> 再通过 Reduce-Scatter 将结果合并回去。
> 代码见 `vllm/distributed/device_communicators/all2all.py` 的 dispatch（all_gatherv）和 combine（reduce_scatterv）。
> 另有 DeepEP、FlashInfer NVLink 等后端使用真正的 All-to-All 通信。

### 核心问题：负载倾斜

router 的选择不是均匀的。真实流量下，某些 expert 被高频选中（热点），某些几乎无 token 路由。

```
示例（64 experts, EP=8, 每卡 8 experts）：
  GPU 0: expert 0-7,   收到 5000 tokens  ← 热点集中
  GPU 1: expert 8-15,  收到  800 tokens
  GPU 2: expert 16-23, 收到  600 tokens
  ...
  GPU 7: expert 56-63, 收到  400 tokens  ← 几乎空闲
```

**木桶效应**：每次 MoE 层的计算时间由最慢的 GPU 决定（All-to-All 需要同步），GPU 0 处理 5000 tokens 时其他 GPU 全在等它。

### 为什么 TP 没有这个问题？

TP（Tensor Parallelism）是把每个 expert 的参数切分到所有 GPU 上，每张卡持有全部 expert 的一部分参数。所有 token 在每张卡上都要计算，负载天然均匀。

EP 是把不同 expert 整体放到不同 GPU，某张卡持有的 expert 恰好是热点就会过载。

## 2. EPLB 核心原理

### 核心概念：Logical Expert vs Physical Expert

vLLM 中 EPLB 的设计基于两个关键概念（见 `eplb_state.py` 注释）：

- **Logical Expert（逻辑 expert）**：模型定义中的 expert，数量固定。如 DeepSeek-R1 有 256 个。
- **Physical Expert（物理 expert）**：实际部署在 GPU 上的 expert 实例，数量 = logical + redundant。

```
num_physical_experts = num_logical_experts + num_redundant_experts
```

`num_redundant_experts` 是用户配置的额外槽位数（配置项 `eplb_config.num_redundant_experts`，默认 0）。
这些额外槽位就是用来放热点 expert 副本的空间。

### 示例：DeepSeek-R1，256 logical experts，32 redundant，EP=32

```
初始状态（均匀分配）：
  num_physical_experts = 256 + 32 = 288
  每张 GPU 持有 288 / 32 = 9 个 physical expert 槽位
  
  GPU  0: [expert 0, 1, 2, 3, 4, 5, 6, 7,  0']   ← 第9个槽位初始放 expert 0 的副本
  GPU  1: [expert 8, 9, 10,11,12,13,14,15,  1']
  ...
  GPU 31: [expert 248,...,255,                24']
```

初始时 redundant 槽位按 `i % num_routed_experts` 轮流分配（见 `build_initial_global_physical_to_logical_map`）。

### 重均衡算法（`DefaultEplbPolicy`）

算法来源于 [DeepSeek EPLB](https://github.com/deepseek-ai/eplb)，分三步（`rebalance_experts_hierarchical`）：

**Step 1：balanced_packing — 把 expert group 分配到 node**

将 expert 按 group 分组，用贪心装箱算法把各 group 均匀分配到各节点。
目标：每个节点内的 expert 负载总量尽量接近。
（利用节点内 NVLink 高带宽，节点间只需少量跨节点通信）

**Step 2：replicate_experts — 在节点内决定哪些 expert 要复制**

核心逻辑（`replicate_experts` 方法）：
```python
for i in range(num_log, num_phy):          # 遍历每个额外的物理槽位
    redundant_indices = argmax(weight / logcnt)  # 找"负载/副本数"最大的 expert
    phy2log[:, i] = redundant_indices            # 这个槽位放该 expert 的副本
    logcnt[redundant_indices] += 1               # 该 expert 的副本计数 +1
```

直觉：每次给当前"平均负载最高"的 expert 多加一份副本。
副本越多，该 expert 的 token 被分摊得越均匀。

**Step 3：balanced_packing — 把 physical expert 分配到 GPU**

将所有 physical expert（含副本）再次贪心装箱，分配到节点内各 GPU 上。
每个 physical expert 的有效负载 = `该 logical expert 的总负载 / 该 expert 的副本数`。
目标：每张 GPU 上的有效负载总量尽量接近。

### 映射关系

算法最终输出 `physical_to_logical_map`：shape = `[num_layers, num_physical_experts]`

```
示例（2 层 MoE，4 logical experts，6 physical experts，3 GPU）：

physical_to_logical_map:
  Layer 0: [0, 1, 2, 3, 0, 1]    ← 物理槽位 4 是 expert 0 的副本，槽位 5 是 expert 1 的副本
  Layer 1: [0, 2, 0, 1, 0, 3]    ← expert 0 有 3 份副本（槽位 0, 2, 4）

logical_replica_count:
  Layer 0: [2, 2, 1, 1]          ← expert 0 和 1 各有 2 份
  Layer 1: [3, 1, 1, 1]          ← expert 0 有 3 份（说明 Layer 1 中 expert 0 是热点）
```

注意：映射是**per-layer** 的，每层的热点 expert 可能不同，所以每层有独立的分配方案。

### 重均衡触发机制

基于固定步数间隔触发（不是基于负载方差阈值）：

```
配置参数：
  step_interval = 3000（默认）   每 3000 步触发一次重均衡
  window_size = 1000（默认）     用最近 1000 步的负载统计作为决策依据

流程（eplb_state.py 的 step() 方法）：
  每次 forward:
    expert_rearrangement_step += 1
    if 接近 step_interval（最后 window_size 步内）:
      记录本步各 physical expert 的 token 数到 sliding window
    if expert_rearrangement_step >= step_interval:
      AllReduce 收集全局负载统计
      调用 policy.rebalance_experts() 计算新分布
      执行权重迁移
      expert_rearrangement_step = 0
```

### 负载统计来源

每次 MoE 层的 router 产生 `topk_ids`（每个 token 选了哪些 physical expert）。
vLLM 在 router 内核中将每个 physical expert 处理的 token 数累加到 `expert_load_pass` 张量中。
滑动窗口 `expert_load_window` 记录最近 `window_size` 步的负载数据。

重均衡时先将 physical expert 负载通过 `physical_to_logical_map` 映射回 logical expert 负载，
再通过 AllReduce 跨所有 EP rank 汇总，得到全局 logical expert 负载统计。

### 权重迁移

计算出新的 `physical_to_logical_map` 后，需要实际搬移 GPU 上的 expert 权重：

- **同步模式**（`use_async=False`）：`rearrange_expert_weights_inplace` 直接在主线程执行，
  通过 `EplbCommunicator`（NCCL）在 GPU 间复制权重。迁移期间推理暂停。
- **异步模式**（`use_async=True`，默认）：后台线程逐层搬移权重到 `expert_buffer`，
  主线程在每步 `step()` 中检查是否完成，完成后调用 `_move_to_workspace` 将 buffer 
  内容写入实际 expert 权重并更新映射表。推理不中断，但新映射分层逐步生效。

### 槽位保持优化

`preserve_intragpu_slots`：重均衡后，如果某个 expert 还在同一张 GPU 上只是换了槽位，
就保持原来的槽位位置不变，避免不必要的权重拷贝。

## 3. 应用场景

### 适用条件

| 条件 | 说明 |
|------|------|
| MoE 模型 | Dense 模型没有 expert，不适用 |
| 使用 Expert Parallelism | 仅 TP 时每卡持有所有 expert 的部分参数，负载天然均匀 |
| 流量有倾斜 | 若 router 天然均匀，收益不大 |
| expert 数量 >> GPU 数 | 如 64 expert / 8 GPU，才有调度空间；8 expert / 8 GPU 每卡仅 1 个 expert，无法迁移 |

### 典型模型

| 模型 | expert 数 | top-K | 适合 EPLB |
|------|-----------|-------|-----------|
| DeepSeek-V3 | 256 | 8 | 非常适合（expert 多，倾斜明显） |
| Qwen2-57B-A14B | 64 | 8 | 适合 |
| Mixtral-8x7B | 8 | 2 | 勉强（expert 数少，调度空间有限） |
| Dense 模型 | 无 | - | 不适用 |

### EP vs TP 的决策参考

- **expert 数量多、GPU 数量多**：优先 EP，通信量小（只传 token，不传 expert 参数），但需要 EPLB 解决负载不均
- **expert 数量少或 GPU 数量少**：优先 TP，负载天然均匀，但通信量大（AllReduce）
- **实际部署常用 EP + TP 组合**：EP 切 expert 维度，TP 切 expert 内部参数维度

## 4. vLLM 中的 EPLB 实现概览

代码位置：`vllm/distributed/eplb/`

```
vllm/distributed/eplb/
├── policy/
│   ├── abstract.py        # 策略抽象接口：rebalance_experts(weight, ...) → physical_to_logical_map
│   └── default.py         # 默认策略（来自 DeepSeek EPLB）：三步层次化算法
├── eplb_state.py          # 核心状态管理：EplbState.step() 驱动整个流程
│                          #   负载统计滑动窗口、触发重均衡、映射表更新
├── eplb_communicator.py   # 通信后端：GPU 间 expert 权重迁移（NCCL）
├── rebalance_execute.py   # 执行层：rearrange_expert_weights_inplace / move_from_buffer
├── eplb_utils.py          # 工具函数
└── async_worker.py        # 异步执行器：后台线程逐层搬移权重
```

配置项（`vllm/config/parallel.py` 的 `EplbConfig`）：
```
num_redundant_experts = 0    # 额外物理槽位数（核心参数，> 0 才有 EPLB 效果）
step_interval = 3000         # 每 N 步触发一次重均衡
window_size = 1000           # 负载统计滑动窗口大小
use_async = True             # 异步模式（推理不中断）
policy = "default"           # 策略选择
```

> 后续代码追踪将从 `eplb_state.py` 的 `step()` 方法开始，
> 理解完整的 统计 → 触发 → 决策 → 迁移 → 提交 流程。

## 5. 核心链条总结

```
用户配置 num_redundant_experts > 0
  → 启动时：num_physical = num_logical + num_redundant，初始均匀分配
  → 推理中：router 的 topk_ids 记录各 physical expert 的 token 数
  → 每 step_interval 步：
      1. AllReduce 汇总全局 logical expert 负载
      2. replicate_experts：给"负载/副本数"最高的 expert 分配更多副本
      3. balanced_packing：将 physical expert 均匀装箱到各 GPU
      4. 输出新的 physical_to_logical_map（per-layer）
      5. 通过 NCCL 搬移 expert 权重到新位置
      6. 更新映射表，router 使用新分布继续推理
  → 目标：各 GPU 处理的有效 token 数趋于均匀，消除木桶效应
```
