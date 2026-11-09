# Week 2 总结：Continuous Batching、Scheduler 与请求生命周期

## 核心收获

### 1. Continuous Batching

传统 Static Batching 必须等整个 batch 的最慢请求完成才能处理下一批，GPU 利用率低。

Continuous Batching 以 iteration（一次 decode step）为粒度调度：
- 完成的请求立即移出，释放 KV Cache
- 新请求随时加入（下一个 iteration 就能 prefill）
- batch 组成每个 iteration 都可能变化

vLLM 的实现就是 `step()` 循环：
```
schedule()           → 每轮重新组 batch（谁留、谁进、谁出）
execute_model()      → GPU 执行当前 batch
update_from_output() → 检查完成、释放资源、返回输出
```

### 2. Scheduler 调度逻辑（schedule()）

调度分两大阶段：

**阶段 1：处理 Running 请求**
- 逐个计算 num_new_tokens（decode 通常=1，chunked prefill 可能更多）
- 调用 `kv_cache_manager.allocate_slots()` 分配新 blocks
- 分配失败 → 抢占优先级最低的 running 请求，释放其 blocks，重试
- 直到分配成功或无人可抢占

**阶段 2：处理 Waiting 请求**
- 前提：没有发生抢占（`not preempted_reqs`）
- 逐个检查 prefix cache 命中 → 计算需要的新 tokens → allocate_slots
- 分配失败 → 直接 break，不抢占 running
- 分配成功 → 从 waiting 移入 running

**关键约束**：
- `token_budget`：每轮调度的总 token 预算
- `max_num_running_reqs`：最大并发数
- `watermark`：为 running 请求预留 blocks（只对 waiting 请求施加）

### 3. update_from_output() 完整流程

GPU 执行完毕后处理结果的核心函数：

```
阶段 1：全局状态更新
  - 延迟释放 blocks（fence 机制：processed_step_seq++）
  - MoE 路由信息保存
  - 性能统计收集

阶段 2：逐请求处理（主循环）
  - 对账：num_in_flight_tokens -= num_tokens_scheduled
  - 过时输出检测（num_stale_output_tokens）
  - 跳过无效请求（failed KV load、已完成、被 abort）
  - 追加 token + check_stop
  - 结构化输出 grammar 推进
  - 处理停止的请求：finish_reason → _free_request → 从 running 移出

阶段 3：批量清理
  - 从 running/waiting 队列移除停止的请求

阶段 4：打包输出
  - 按 client_index 组织 EngineCoreOutput
  - 附加 finished_requests 通知 + scheduler_stats
```

### 4. KV Cache 与 Scheduler 的配合

**allocate_slots 判断逻辑**：
```python
available_blocks = block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None  # 告诉 Scheduler "分配失败"
```

**watermark 机制**：
- 只对 WAITING/PREEMPTED 请求生效
- 为 running 请求的后续 decode 预留空间
- 避免"新请求挤满 → running 下一步分配失败 → 被迫抢占"

**full_sequence_must_fit（默认开启）**：
- 在 chunked prefill 场景下，入场前先检查整个序列能否放下
- 防止"进来一半卡住"浪费已有计算

**remove_skipped_blocks**：
- 分配前先回收滑动窗口外的旧 blocks
- 不同 attention 类型通过多态 Manager 子类实现各自的回收策略

**抢占（Preemption）**：
- 释放被抢占请求的全部 KV Cache blocks
- num_computed_tokens 重置为 0（需要重新 prefill）
- output_token_ids 合并入 prompt_token_ids（不丢失已生成内容）
- 放回 waiting 队列头部

### 5. 异步调度与过时输出

**num_stale_output_tokens**：
- 异步调度下，抢占发生时 GPU 上可能还有 in-flight 的输出
- 这些输出对应的 KV Cache 已被释放，属于"过时"输出
- `drop_stale_output=True`：直接丢弃
- `drop_stale_output=False`：静默消费（不追加到 output）

**延迟释放 blocks（defer_block_free）**：
- 异步调度/PP 场景下，GPU 可能还在用某些 blocks
- fence 机制：sched_step_seq 和 processed_step_seq 对比
- 只有确认 GPU 不再使用后才实际释放

### 6. KV Cache Manager 架构

```
KVCacheManager（对外接口）
    └── Coordinator（协调多个 group）
            └── SingleTypeKVCacheManager 子类（每个 group 一个）
                    ├── FullAttentionManager     — 全注意力，不回收
                    ├── SlidingWindowManager     — 回收窗口外的 blocks
                    ├── RSWAManager              — 稀疏保留式滑动窗口
                    ├── ChunkedLocalAttentionManager
                    └── MambaManager             — 状态空间模型
```

初始化路径：
```
Scheduler.__init__()
  → KVCacheManager(kv_cache_config)
    → get_kv_cache_coordinator(kv_cache_config)
      → 遍历 kv_cache_config.kv_cache_groups
        → get_manager_for_kv_cache_spec(spec)  # 多态分发
```

Scheduler 不关心底层是什么 attention 类型，统一调用接口即可。

## 关键代码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| schedule() 主函数 | scheduler.py | 439 |
| Running 请求 allocate + 抢占循环 | scheduler.py | 576-629 |
| Waiting 请求 allocate | scheduler.py | 973-994 |
| update_from_output() | scheduler.py | 1670-2048 |
| _preempt_request() | scheduler.py | 1274-1315 |
| _free_request_blocks() | scheduler.py | 2340-2365 |
| allocate_slots() | kv_cache_manager.py | 344-560 |
| watermark 判断 | kv_cache_manager.py | 463-470 |
| full_sequence_must_fit 检查 | kv_cache_manager.py | 472-488 |
| remove_skipped_blocks() | single_type_kv_cache_manager.py | 622-659 |
| SlidingWindow get_num_skipped_tokens | single_type_kv_cache_manager.py | 1057-1083 |
| get_manager_for_kv_cache_spec() | single_type_kv_cache_manager.py | 1836 |

## 核心概念关系图

```
用户请求到达
     │
     ▼
┌──────────┐     allocate_slots() 成功
│ WAITING  │ ─────────────────────────────→ ┌──────────┐
└──────────┘                                │ RUNNING  │
     ▲                                      └──────────┘
     │ _preempt_request()                        │
     │ (释放 blocks, 重置进度)                     │ 每轮 decode
     │                                           ▼
     │                                    update_from_output()
     │                                           │
     │    allocate_slots() 失败                    │ check_stop
     ├────────────────────────────────────────────┤
     │    (抢占最低优先级请求)                       │
                                                 ▼
                                          ┌──────────┐
                                          │ FINISHED │
                                          └──────────┘
                                                 │
                                                 ▼
                                          _free_request()
                                          释放 KV Cache blocks
```

## 遗留问题（Week 3+ 解决）

1. ModelRunner 如何把 scheduler_output 转成 GPU 输入？（attention metadata 构建）
2. execute_model() 内部 TP 同步是怎么做的？
3. Prefix Caching 的 hash 计算和命中逻辑细节
4. Chunked Prefill 的 chunk 大小如何确定？

## 学习状态

- ✅ Day 1: Continuous Batching 原理
- ✅ Day 2-3: Scheduler 调度策略 + schedule() 代码
- ✅ Day 3-4: 请求生命周期 + update_from_output() 完整流程
- ✅ Day 5: KV Cache 与 Scheduler 配合（allocate_slots、watermark、抢占、滑动窗口回收）
- ✅ Day 6-7: Week 2 总结

**Week 2 完成时间**: 2026-08-20
