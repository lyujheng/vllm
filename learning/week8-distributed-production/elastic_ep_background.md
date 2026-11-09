# Elastic EP（弹性专家并行）背景与原理

## 1. 问题背景：静态 EP 的资源瓶颈

### EP 的固有限制

在标准 Expert Parallelism 中，`data_parallel_size`（DP 副本数）在启动时确定且不可变。每个 DP 副本拥有一组 GPU，这组 GPU 上分布着 MoE 模型的全部专家。

这意味着：
- **启动时必须决定 GPU 数量**：即使流量低谷只需要一半的计算资源，也必须保持全部 GPU 运行
- **流量高峰时无法横向扩展**：如果 2 个 DP 副本满载，无法动态加入第 3 个 DP 副本来分担请求
- **GPU 资源利用率低**：在实际生产中，流量波动是常态，固定 GPU 数量导致资源浪费或服务质量下降

### 传统做法的代价

没有 Elastic EP 时，要应对流量变化只能：
1. **超额预留**：常驻最大规模 GPU，低谷时大量空转 → 成本高
2. **手动扩缩容**：停服 → 重新启动更多/更少 DP 副本 → 重新加载模型 → 恢复服务。这个过程需要分钟级时间，且服务中断

### 核心诉求

需要一种机制，能在服务不中断的前提下，运行时动态增减 DP 副本数量，实现 GPU 资源的弹性伸缩。

## 2. Elastic EP 的定义

Elastic EP（弹性专家并行）是 vLLM 中的一种运行时动态伸缩机制，允许在推理服务运行过程中，**不停服地**增加或减少 Data Parallel 副本数量。

关键特征：
- **在线扩缩容**：不需要停止推理服务
- **以 DP 副本为粒度**：每次增/减一个完整的 DP 副本（一组 GPU）
- **与 EPLB 深度耦合**：扩缩容后必须进行 EPLB 重均衡，确保新的专家分布合理
- **状态机驱动**：整个扩缩容过程被建模为多步状态机，每步可以异步执行，不阻塞正常推理

## 3. 核心原理

### 3.1 Scale Up（扩容）流程

场景：当前 DP=2（2 个 DP 副本），需要扩到 DP=3。

**涉及两类角色**：
- **Existing Engine**：已经在运行的 DP 副本（dp_rank 0 和 1）
- **New Engine**：新加入的 DP 副本（dp_rank 2）

#### Existing Engine 的状态机

```
CREATE_STANDBY_GROUPS → STAGE_QUANT_METHODS → TRANSFER_WEIGHTS
    → SYNC_KV_CACHE_MEMORY_SIZE → COMMIT_SCALE_UP → COMPLETE
```

每一步的含义：

1. **CREATE_STANDBY_GROUPS**：创建"候补"通信组（standby groups）。这是新拓扑下的 world/dp/ep/eplb 通信组，此时还不替换正在使用的活跃组，所以不影响正在进行的推理
2. **STAGE_QUANT_METHODS**（异步）：如果 EP size 发生变化，MoE 层的量化方法需要适配新的专家分片大小。这一步预先准备好新的 quant method，但还不生效
3. **TRANSFER_WEIGHTS**（异步）：现有 worker 通过 P2P 把模型权重（非 expert 权重部分）发送给新加入的 worker。采用 sender-receiver 配对策略，多个 existing worker 并行发送
4. **SYNC_KV_CACHE_MEMORY_SIZE**：新旧 worker 之间同步可用 KV cache 显存大小（取最小值），确保所有 DP 副本使用相同的 KV cache 配置
5. **COMMIT_SCALE_UP**（阻塞 forward）：
   - 用 standby groups 替换活跃通信组
   - 更新 parallel_config 中的 dp_size、dp_rank 等
   - 重新配置 MoE 模块的 EP 参数
   - 扩展 EPLB 状态张量（physical_to_logical_map 等）
   - **执行 EPLB 重均衡**：将专家重新分配到扩大后的 EP group 中
   - 重新 warmup 和捕获 CUDA graph

#### New Engine 的状态机

```
PRE_KV_INIT → PREPARE → COMPLETE
```

1. **PRE_KV_INIT**：
   - 接收来自 existing worker 的模型权重（非 expert 部分）
   - 同步 KV cache 显存大小
   - 准备通信 buffer
2. **PREPARE**：
   - kernel warmup
   - 同步 engine 运行状态（engines_running、current_wave、step_counter）
   - 执行 commit_scale_up，包含 EPLB 重均衡

### 3.2 Scale Down（缩容）流程

场景：当前 DP=3，需要缩到 DP=2。

**涉及两类角色**：
- **Remaining Engine**：继续运行的 DP 副本（dp_rank 0 和 1）
- **Removing Engine**：即将退出的 DP 副本（dp_rank 2）

#### Remaining Engine 的状态机

```
PREPARE → COMMIT_SCALE_DOWN → COMPLETE
```

1. **PREPARE**：创建 standby groups（缩容后的新拓扑通信组）
2. **COMMIT_SCALE_DOWN**（阻塞 forward）：
   - 先执行 EPLB 重均衡：此时 rank_mapping 会标记被移除的 rank 为 -1，EPLB 把它们的专家迁移到剩余 rank 上
   - 切换到新通信组
   - 重新 warmup + CUDA graph 捕获

#### Removing Engine 的状态机

```
PREPARE → COMPLETE
```

1. **PREPARE**：
   - 参与 EPLB 重均衡（把自己持有的专家权重发送给其他 rank）
   - 切换到空通信组（switch_and_remove）
2. **COMPLETE**：发送 SHUTDOWN_COMPLETE 通知 → 退出进程

### 3.3 Standby Groups 机制

这是 Elastic EP 的关键设计，目的是**不中断正在进行的推理**。

原理：
- 当前活跃的通信组（world/dp/ep/eplb）负责正常推理
- 新的通信组（standby groups）在后台创建，包含新旧所有 rank
- 准备工作（权重传输、quant method 预备）都通过 standby groups 进行
- 只在最终 COMMIT 阶段，一次性把 standby groups 替换为活跃组

这样，在 COMMIT 之前的所有步骤都不影响正常推理的 forward pass。

### 3.4 权重传输策略

扩容时，新 worker 需要获得模型权重。策略：

- **非 expert 权重**：通过 `batch_transfer_weights()` P2P 发送。Existing worker 和 new worker 配对，通过 standby DP group 的 device communicator 进行 batch_isend_irecv
- **expert 权重**：不直接传输。新 worker 初始加载 dummy weights，然后在 COMMIT 阶段的 EPLB 重均衡中，通过 EPLB 的 P2P 通信机制（EplbCommunicator）获得正确的 expert 权重分配

sender-receiver 配对逻辑：
```
num_new_workers = new_dp_size - old_dp_size
每个 existing worker 负责发送给 k 或 k+1 个 new worker（尽量均匀分配）
```

### 3.5 与 EPLB 的深度耦合

Elastic EP 强制要求 `enable_eplb=True`，因为：

1. **扩容后**：EP size 增大，原来的 physical_to_logical_map 需要扩展（新 rank 的专家槽位初始为 -1），必须重新运行 EPLB 策略把专家分配到新拓扑
2. **缩容前**：被移除 rank 上的专家必须先迁移到剩余 rank，否则这些专家就丢失了。通过 `rank_mapping={old_rank: -1}` 告知 EPLB "这个 rank 要消失"，EPLB 会把专家重新分配

关键代码逻辑（`elastic_execute.py` 中 `_perform_eplb_reshuffle()`）：
- 临时关闭异步 EPLB 模式（确保重均衡同步完成）
- 调用 `eplb_state.rearrange(rank_mapping=...)` 执行重均衡
- 重置 expert_rearrangement_step 确保所有 rank 同步
- 恢复异步模式

## 4. 应用场景

### 4.1 云上弹性部署

最典型的场景：根据流量自动调整 GPU 资源。

- 流量低谷（如凌晨）：缩减 DP 副本，释放 GPU 降低成本
- 流量高峰（如白天工作时段）：增加 DP 副本，提升吞吐
- 可与 Kubernetes HPA（Horizontal Pod Autoscaler）或云厂商的自动伸缩策略配合

### 4.2 故障恢复

当某个 DP 副本的 GPU 出现故障时：
- Scale Down 移除故障副本
- 服务不中断，剩余副本继续处理请求
- 故障 GPU 修复后，Scale Up 重新加入

### 4.3 混合优先级部署

高优请求 → 低优请求场景切换：
- 高优时段：全部 GPU 服务高优请求
- 低优时段：Scale Down 释放部分 GPU 给其他任务（训练、batch 推理等）

## 5. 关键配置

```python
# ParallelConfig 中的相关参数
enable_elastic_ep: bool = False   # 开启弹性 EP
enable_eplb: bool = True          # 必须同时开启 EPLB（强制要求）

# 约束条件
# - 不支持 Pipeline Parallelism（PP > 1）
# - 不支持 data_parallel_external_lb / data_parallel_hybrid_lb
# - 异步 EPLB 模式下需要 NIXL 包
```

## 6. 与 EPLB 的关系对比

| 维度 | EPLB | Elastic EP |
|------|------|-----------|
| 解决的问题 | 专家负载不均 | GPU 资源弹性伸缩 |
| 触发方式 | 周期性（step_interval） | 外部指令（ReconfigureDistributedRequest） |
| 改变什么 | 专家到 GPU 的映射 | DP 副本数量 + 专家映射 |
| 是否中断推理 | 异步模式不中断 | COMMIT 阶段短暂阻塞 |
| 依赖关系 | 独立运行 | 依赖 EPLB |
| 粒度 | 每层 MoE 独立重均衡 | 整个模型全局重配置 |

## 7. 整体流程总结（Scale Up 为例）

```
外部指令（如 K8s controller）
  │
  ▼
EngineCoreClient 发送 ReconfigureDistributedRequest
  │
  ▼
DPEngineCoreProc.reinitialize_distributed()
  │  创建 ElasticEPScalingState（状态机）
  ▼
run_busy_loop 中 progress() 驱动状态推进
  │
  ├─ [Existing Engine]                    ├─ [New Engine]
  │  1. 创建 standby groups               │  1. 接收权重
  │  2. 预备 quant methods (异步)          │  2. 同步 KV cache 大小
  │  3. P2P 传输权重 (异步)               │  3. kernel warmup
  │  4. 同步 KV cache 大小                │  4. commit → EPLB 重均衡
  │  5. ── 等待 commit 指令 ──            │
  │  6. commit → 切换通信组               │
  │     → EPLB 重均衡                     │
  │     → warmup + CUDA graph             │
  ▼                                        ▼
服务恢复，DP=3 正常推理
```
