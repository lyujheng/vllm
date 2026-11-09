# Week 8 总结：生产级分布式推理

**完成时间**：2026-10-10

---

## 学习内容总览

| Day | 主题 | 状态 |
|-----|------|:----:|
| Day 1-2 | PD 分离（Disaggregated Prefill/Decode） | ✅ |
| Day 3-4 | Expert Parallelism 深入与 EPLB | ✅ |
| Day 5 | 生产级投机解码（DFlash + DSpark） | ✅ |
| Day 6 | KV Cache Offloading + 长上下文内存规划 | ✅ |
| Day 7 | Week 8 总结 | ✅ |

---

## 1. PD 分离（Day 1-2）

### 核心思想

Prefill（计算密集）和 Decode（内存密集）分离到不同节点，各自独立扩缩容。

### KV Connector 框架

vLLM 通过 KV Connector 抽象统一了 KV Cache 传输的多种实现：

- **KVConnectorFactory**：工厂模式，通过 `kv_connector` 名称懒加载分发到具体 connector
- **KVConnectorBase_V1**：统一接口，所有 connector 实现相同的 5 个 scheduler 调用点 + worker 端钩子
- **MultiConnector**：包装多个子 connector（如 Nixl + SimpleCPUOffload），first-hit-wins 策略

### Scheduler 端 5 个调用点

| 调用点 | 时机 | 作用 |
|--------|------|------|
| `bind_gpu_block_pool` | 初始化 | 绑定 block pool 引用 |
| `get_num_new_matched_tokens` | 调度前 | 查询 external KV 命中数（CPU/远端） |
| `update_state_after_alloc` | block 分配后 | 建立 CPU↔GPU block 映射，touch 防淘汰 |
| `build_connector_meta` | schedule() 尾部 | 打包 load/store 清单发给 worker |
| `request_finished` | 请求完成 | 清理状态（offload 的清理，PD 的心跳停止） |

### Worker 端流程

```
pre_forward:  绑定 metadata + start_load_kv
     ↓
GPU forward:  模型计算（compute stream）
     ↓
post_forward: wait_for_save + get_finished（提交 load/store + 轮询完成事件）
```

### PD vs Offloading 机制对比

| 维度 | PD 分离 (Nixl) | KV Cache Offloading |
|------|---------------|-------------------|
| 传输方向 | GPU → 远端 GPU（网络） | GPU ↔ CPU（PCIe） |
| Block 保护 | 租约 + 心跳（30s lease, ~5s heartbeat） | touch ref_cnt |
| 传输粒度 | Nixl: 全量传输; LMCache: 逐层流水线 | block 级 DMA copy |
| 完成通知 | `finished_recving` | `finished_recving`（复用同一字段） |
| Store 触发 | 请求完成后传输 | `build_connector_meta` 每轮调度 |

### 代码产出

- `learning/week8-distributed-production/day1_pd_disaggregation_background.md`
- `learning/week8-distributed-production/day1_pd_disaggregation_code.md`
- `learning/week8-distributed-production/pd_disaggregation_workflow.md`
- `learning/week8-distributed-production/kv_transfer_latency_analysis.md`

---

## 2. EPLB 与弹性 EP（Day 3-4）

### EPLB（Expert-Parallel Load Balancing）

**问题**：MoE 模型中部分 expert 被高频选中（热点），EP 下导致 GPU 间负载不均。

**解决方案**：热点 expert 冗余复制到多卡，冷门 expert 合并。

**核心流程**：
1. **统计收集**：`base_router.py` 中 `_maybe_auto_rebalance()` 每隔 N 步统计各 expert 负载
2. **策略计算**：`policy/default.py` 的 `rebalance()` — 按负载排序，决定哪些 expert 复制/合并
3. **执行重均衡**：`rebalance_execute.py` — 跨 GPU 权重搬运（AllGather + 选择性写入）
4. **逻辑路由更新**：`eplb_state.py` — 更新 `logical_to_physical` 映射表，后续 router 按新映射分发

**设计亮点**：
- 重均衡在线执行，不需要停服
- 物理 expert 与逻辑 expert 解耦，router 只看逻辑 ID
- 策略可插拔（`policy/abstract.py` 定义接口）

### 弹性 EP

- 动态增减 GPU 参与 MoE 计算
- 核心状态机：STABLE → GROWING/SHRINKING → STABLE
- Cooperative scheduling：非阻塞分步执行，每个 forward step 推进一步状态迁移
- 代码较新，生产成熟度待验证

### 代码产出

- `learning/week8-distributed-production/eplb_background.md`
- `learning/week8-distributed-production/elastic_ep_background.md`

---

## 3. 生产级投机解码 — DFlash + DSpark（Day 5）

### DFlash：并行 Draft

- 一次前向生成整个 draft block（K 个 token），不逐 token 自回归
- Context KV 复用目标模型已有的 KV Cache
- Draft block 内部非因果 mask，所有位置并行计算
- 两层架构：proposer 编排层（`vllm/v1/spec_decode/dflash.py`）+ GPU speculator 层

### DSpark：DFlash + Markov Head

在 DFlash 并行骨干基础上，增加低秩 Markov Head 补偿位置间依赖：
- `logits[i] = base_logits[i] + markov_w2(markov_w1(prev_sampled_token))`
- 并行骨干保证速度，Markov Head 以极低代价提升接受率

### Dynamic SD Schedule

按 batch size 区间映射 `num_speculative_tokens`：高并发缩短 draft 长度，低并发拉长。本质是静态映射表，不是运行时动态决策。

### 实验数据（Qwen3-8B + dspark drafter，单卡 H800）

**DSpark vs Baseline（K=7）**：

| 指标 | Baseline | DSpark | 变化 |
|------|----------|--------|------|
| Output throughput (tok/s) | 1224.88 | 1250.91 | +2.1% |
| Mean TTFT (ms) | 28.12 | 36.92 | +31.3% ↑ |
| Mean TPOT (ms) | 6.92 | 3.28 | **-52.6% ↓** |
| Mean ITL (ms) | 6.92 | 9.20 | +32.9% ↑ |

**K sweep（K=1/3/5/7）**：

| 指标 | K=1 | K=3 | K=5 | K=7 |
|------|-----|-----|-----|-----|
| Mean TPOT (ms) | 6.75 | 3.75 | 3.44 | 3.19 |
| 平均接受长度 | 1.64 | 2.35 | 2.67 | 2.82 |
| Acceptance rate (%) | 63.68 | 45.12 | 33.37 | 26.00 |

K=3 为性价比拐点：TPOT 降幅最大（6.75→3.75, -44%），之后边际递减。

**K=1 异常**：TTFT 飙升至 1872ms，Peak concurrent=106。原因是 K=1 时每轮只猜 1 个 token，draft overhead 占比高但接受收益低，导致有效推进速度下降，请求堆积。

### 适用场景

- ✅ 延迟敏感 + 低并发：TPOT 降低 50%+
- ✅ 大 target 模型：draft-target 能力差距大，接受率更高
- ❌ 高并发：target 已 compute-bound，draft 验证开销拖累吞吐
- ❌ TTFT 极致敏感：首 token 延迟增加

### 代码产出

- `learning/week8-distributed-production/dflash_dspark_background.md`
- `learning/week8-distributed-production/dflash_dspark_code.md`
- `learning/week8-distributed-production/dflash_dspark_experiment.md`

---

## 4. KV Cache Offloading（Day 6）

### KV Connector 框架下的 Offloading

SimpleCPUOffloadConnector 复用 KV Connector 框架，实现 GPU ↔ CPU 的 KV Cache 搬运。

### 两种模式

| 模式 | 触发条件 | 适用场景 |
|------|---------|---------|
| **Eager** | 每个请求算完就异步存到 CPU | 请求模式不可预测，需要最大化 CPU 缓存利用 |
| **Lazy** | GPU block 被淘汰进入 free queue 时才存 | 请求可预测/显存紧张时减少不必要的搬运 |

两种模式在启动时固定，不可动态切换。

### DmaCopyBackend

- 后台 daemon 线程 + producer-consumer 队列
- 双 CUDA stream（load_stream, store_stream），与 compute stream 异步
- Store 需要 wait_event（等 compute 完成），Load 不需要（CPU pinned memory 稳定）
- CUDA event 实现跨 stream 同步

### 跨 step 加载

Offloading 的 load 在 step N 的 post_forward 提交，数据在 step N+1 的 forward 可用。请求在 load 完成前不会被调度进 forward。

### 实验数据（Qwen2.5-7B，单卡 H800，gpu-memory-utilization=0.7）

**冷启动（首轮）**：Baseline / Eager / Lazy 性能几乎一致（throughput ~9.54 req/s, TTFT ~100ms）

**暖启动（第二轮，prefix cache 已积累）**：

| 指标 | Baseline | Eager | Lazy |
|------|----------|-------|------|
| Mean TTFT (ms) | 37.13 | 34.47 | 37.41 |
| Mean TPOT (ms) | 6.44 | 6.51 | 6.35 |
| Peak concurrent | 24 | 24 | 24 |

**Server 日志关键指标**：

| 指标 | Baseline | Eager | Lazy |
|------|----------|-------|------|
| GPU KV cache 峰值使用率 | 3.7% | 5.1% | 5.7% |
| Prefix cache hit（第二轮） | 48.7% | 42.4% | 37.0% |
| **External prefix cache hit** | N/A | **0.0%** | **0.0%** |

### 关键发现

1. **Offloading 未触发**：External prefix cache hit 始终为 0%。原因是 GPU KV cache 未饱和（峰值仅 3-6%），block 不会被淘汰，offloading 无用武之地
2. **零退化**：Eager 的后台 CPU store 完全没有影响推理延迟，验证了 DmaCopyBackend 异步设计的零开销特性
3. **Prefix cache 本身效果显著**：暖启动 TTFT -65%，TPOT -41%

### 触发 Offloading 收益的条件

- GPU KV cache 饱和（低 `gpu-memory-utilization` 或更大模型/更长序列）
- 请求量超过 GPU KV cache 容量，迫使 block 淘汰
- 有共享前缀，被淘汰的 block 在后续请求中有复用价值

### 代码产出

- `learning/week8-distributed-production/kv_offloading_background.md`
- `learning/week8-distributed-production/kv_offloading_code_tracing.md`
- `learning/week8-distributed-production/day6_experiment_analysis.md`
- `learning/week8-distributed-production/experiments/exp_kv_offloading.sh`

---

## 5. 历史实验数据回顾（Week 4-7）

Week 8 的学习建立在前几周的基础之上，汇总关键实验数据：

### Week 4: 分布式推理基础

**TP 扩展性（Qwen2.5-7B，H800）**：

| 配置 | Output tok/s | Mean TTFT | Mean TPOT | 相对 TP=1 加速比 |
|------|:-----------:|:---------:|:---------:|:---------------:|
| TP=1 | — | 601.8ms | 8.20ms | 1.00x |
| TP=2 | — | 411.4ms | 6.23ms | 1.35x |
| TP=4 | — | 283.3ms | 4.39ms | 1.96x |

- TP 未达线性加速（TP=4 理论 4x，实际 1.96x），原因是 AllReduce 通信开销
- 模型单卡放得下时，DP 扩吞吐优于 TP（DP 零通信，吞吐约 ~4x）

**TP+PP 对比（4 卡）**：

| 配置 | Output tok/s | Mean TTFT | Mean TPOT |
|------|:-----------:|:---------:|:---------:|
| TP=4, PP=1 | — | 304.8ms | 4.46ms |
| TP=2, PP=2 | — | 336.1ms | 8.77ms |

- 纯 TP 吞吐是 TP+PP 的 1.64 倍
- PP 的 pipeline bubble 导致 decode 阶段有效计算量减半
- 结论：模型放得下时避免 PP，放不下时 TP 优先切到单卡能放、再加 PP

### Week 5: 高级调度优化

| 优化项 | 关键指标 | 变化 |
|--------|---------|------|
| Prefix Caching ON | Mean TTFT | 101.66 → 77.74ms（-23.5%） |
| Prefix Caching ON | P99 TTFT | 174.49 → 115.83ms（-33.6%） |
| Chunked Prefill 1024 | P99 ITL | 35.90 → 22.32ms（-37.8%） |
| Chunked Prefill 1024 | Mean TTFT | 88.01 → 94.88ms（+7.8%，预期内） |
| CUDA Graph ON | TPOT | 203.86 → 130.01ms（-36.2%） |
| 组合优化 | 吞吐 | 115.89 → 167.74 tok/s（+44.7%） |

### Week 6: MoE 推理

**MoE vs Dense**：

| 指标 | Dense (Qwen2.5-7B) | MoE (Qwen2-57B-A14B) |
|------|:------------------:|:--------------------:|
| Output throughput (tok/s) | 4402.86 | 1536.22 |
| 比值 | 1.00x | 0.35x |

MoE 57B（激活 14B）吞吐约为 Dense 7B 的 1/3，内存墙是主要瓶颈。

### Week 7: 量化

**量化方案对比（Qwen2.5-7B，H800）**：

| 配置 | 权重显存 | Output tok/s | Mean TPOT | 相对 BF16 |
|------|:-------:|:-----------:|:---------:|:---------:|
| BF16 基线 | 14.63 GiB | 5297 | 7.57ms | 1.00x |
| FP8 per-tensor | 8.59 GiB | **7506** | **5.69ms** | **1.42x** |
| INT4 GPTQ | 5.69 GiB | 6558 | 6.60ms | 1.24x |
| INT4 AWQ | 5.76 GiB | 6087 | 6.74ms | 1.15x |
| FP8 per-block | 8.59 GiB | 440 ⚠️ | 78.62ms | 异常 |
| BF16 + FP8 KV | 15.64 GiB | 5087 | 7.63ms | 0.96x |

- FP8 per-tensor 是 H800 最优方案（吞吐 +42%，权重显存减半）
- FP8 per-block 在线量化性能异常（kernel 未充分优化）
- INT4 AWQ/GPTQ 吞吐介于 BF16 和 FP8 之间

**LoRA 多租户**：
- 单 adapter 开销：TTFT +11%，TPOT +23%
- max-loras=1 导致 adapter 换入换出，被调度到非活跃 adapter 的请求 TTFT 劣化 5.3 倍（240→1280ms）
- max-loras=2 时两个 adapter 性能一致

---

## 6. 技术串联：各组件的交互关系

### KV Connector 是统一抽象

```
                    KVConnectorFactory
                         │
          ┌──────────────┼──────────────┐
          │              │              │
   NixlConnector   SimpleCPUOffload  LMCacheConnector
    (PD 分离)       (Offloading)     (逐层流水线)
          │              │              │
          └──────┬───────┘              │
                 │                      │
          MultiConnector ───────────────┘
         (PD + Offloading 组合)
```

PD 分离和 Offloading 复用同一框架，`finished_recving` 字段通用（PD: 网络接收完成; Offloading: CPU→GPU load 完成）。

### Scheduler 是决策中枢

```
Scheduler.schedule()
  ├─ get_num_new_matched_tokens()  → 查 GPU prefix cache + external（CPU/远端）
  ├─ update_state_after_alloc()    → 建映射、touch block
  ├─ build_connector_meta()        → 打包 load/store spec
  └─ request_finished()            → 清理（offload cleanup / PD heartbeat stop）

_update_from_kv_xfer_finished()
  ├─ connector.update_connector_output()  → 处理 load/store 完成事件
  ├─ finished_recving → 标记请求可参与 forward
  └─ finished_sending → PD 场景释放 block
```

### Worker 端的时序

```
step N:
  pre_forward()   → bind metadata + start_load_kv
  GPU forward     → compute stream
  post_forward()  → submit load/store to DmaCopyBackend → poll completion

step N+1:
  pre_forward()   → step N 的 load 已完成，请求可参与 forward
```

---

## 7. 关键认知收获

### 7.1 框架设计层面

- **Connector 抽象的威力**：PD 分离、CPU offloading、外部 KV cache 服务，三种完全不同的场景通过同一接口统一。新增传输后端只需实现 `KVConnectorBase_V1`，scheduler/worker 代码零修改
- **异步是核心设计原则**：DmaCopyBackend 的 daemon thread + dual CUDA stream、PD 的异步 KV 传输、投机解码的 draft-verify 流水线，都是通过异步解耦来 overlap 通信与计算

### 7.2 实验验证层面

- **Offloading 有前置条件**：GPU KV cache 必须饱和才有价值。H800 79GiB 对 7B 模型太富裕，200 个 2K 请求只用了 3-6% 的 KV cache
- **投机解码看场景**：TPOT -52.6% 很诱人，但 TTFT +31.3%、高并发吞吐仅 +2.1%。延迟敏感 + 低并发是甜点
- **FP8 per-tensor 是 H800 上的默认选择**：吞吐 +42%、权重减半、精度损失小、无需预量化
- **TP 优先于 PP**：同样 4 卡，TP=4 吞吐是 TP=2+PP=2 的 1.64 倍。PP 只在跨节点且 TP 切满时使用

### 7.3 生产部署决策

| 场景 | 推荐方案 |
|------|---------|
| Dense 模型单卡放得下 | DP 扩吞吐（零通信，线性扩展） |
| Dense 模型单卡放不下 | TP 切到单卡能放 → DP 扩吞吐 |
| TP 切满单节点还放不下 | 跨节点加 PP（通信量小） |
| MoE 模型 | EP（按 expert 切分）+ EPLB 负载均衡 |
| 超长上下文（128K+） | CP 切 sequence 维度减 KV cache 显存 |
| 高 QPS | PD 分离 + DP，prefill/decode 独立扩缩容 |
| 延迟敏感 + 低并发 | DSpark 投机解码（TPOT -50%+） |
| GPU KV cache 饱和 + 有共享前缀 | KV Cache Offloading（Eager/Lazy） |
| 多 LoRA 租户 | max-loras ≥ 活跃 adapter 数（避免换入换出） |
| H800 部署 | 默认 FP8 per-tensor 量化 |

---

## 8. 遗留问题与后续方向

1. **Offloading 实验未充分验证**：需要在 GPU KV cache 饱和场景下重测（低 gpu-memory-utilization + 大量长序列请求），才能观察到 external prefix cache hit > 0
2. **PD 分离单机局限**：单机 NVLink 传输无法体现真实网络带宽瓶颈，KV 传输延迟的理论推算（Day 1-2 文档）需要多节点实验验证
3. **EPLB 未做实验**：掌握了原理和代码结构，但未在 EP 配置下实测负载均衡效果
4. **DSpark 接受率偏低**：8B target 模型上 K=7 平均接受长度仅 2.82，更大模型（72B+）预期更好

---

## 文档索引

| 文件 | 说明 |
|------|------|
| `day1_pd_disaggregation_background.md` | PD 分离原理 |
| `day1_pd_disaggregation_code.md` | PD 分离代码追踪 |
| `pd_disaggregation_workflow.md` | PD 分离完整工作流 |
| `kv_transfer_latency_analysis.md` | KV 传输延迟理论分析 |
| `eplb_background.md` | EPLB 原理与代码追踪 |
| `elastic_ep_background.md` | 弹性 EP 原理 |
| `dflash_dspark_background.md` | DFlash + DSpark 原理 |
| `dflash_dspark_code.md` | DFlash + DSpark 代码追踪 |
| `dflash_dspark_experiment.md` | DSpark 实验数据 |
| `kv_offloading_background.md` | KV Offloading 原理 |
| `kv_offloading_code_tracing.md` | KV Offloading 完整代码追踪 |
| `day6_experiment_analysis.md` | KV Offloading 实验分析 |
| `experiments/exp_kv_offloading.sh` | KV Offloading 实验脚本 |
