# Week 5 Day 1-2: Chunked Prefill

## 学习目标

1. 理解 Chunked Prefill 要解决的问题
2. 掌握 vLLM 中 chunked prefill 的调度实现
3. 通过实验对比 chunked prefill 对 TTFT 和 TPOT 的影响

## 1. 问题背景

### 为什么需要 Chunked Prefill？

**问题**：长 prompt 的 prefill 阶段会阻塞 decode 请求
- Prefill 是计算密集型（一次处理整个 prompt，token 数量可能很大）
- Decode 是 memory-bound（每次只生成 1 个 token，但有严格的延迟要求）
- 如果 prefill 占满 GPU，decode 请求会被饿死，导致用户感知延迟飙升

**解决思路**：
- 将长 prefill 拆分成多个 chunk
- 在 prefill chunk 之间插入 decode 请求执行
- 牺牲一部分 prefill 吞吐（TTFT 变长），换取 decode 延迟的稳定性（ITL 改善）

## 2. vLLM 实现原理

### 2.1 核心配置参数

**`max_num_batched_tokens`**：控制单次 forward 处理的最大 token 数
- 路径：`vllm/v1/core/sched/scheduler.py:113`
- 作用：决定 prefill 的 chunk 大小上界

### 2.2 核心代码路径

#### Scheduler 主调度逻辑
- **文件**：`vllm/v1/core/sched/scheduler.py`
- **关键方法**：
  - `schedule()` (line 439)：主调度入口
  - `_schedule_prefills()`：prefill 请求的调度和分块
  - `_clip_prefill_chunk_for_mamba()` (line 369)：处理 Mamba 模型的边界对齐

#### Chunk 标记逻辑
- **Line 1335-1342**：`is_prefill_chunk` 判定
  ```python
  # 如果 num_computed_tokens < total_tokens，说明还在 chunk 中
  is_prefill_chunk = (num_computed_tokens < total_tokens)
  ```

#### Chunk 队列管理
- **Line 358**：跟踪正在执行的 prefill chunks
- **Line 2219-2222**：chunk 完成后排队下一个 chunk

### 2.3 Chunking 执行流程

```
1. 新 prefill 请求进入 waiting queue
2. Scheduler 根据 max_num_batched_tokens 决定 chunk 大小
3. 如果 prompt_len > max_num_batched_tokens：
   - 切分成多个 chunk
   - 第一个 chunk 进入 running queue
   - 设置 is_prefill_chunk = True
4. Chunk 执行完成：
   - 更新 num_computed_tokens
   - 如果还有剩余 tokens，重新排队下一个 chunk
   - 否则进入 decode 阶段
5. Decode 请求在 chunk 间隙得到调度机会
```

### 2.4 与其他特性的交互

- **Speculative Decoding** (line 2156-2157)：prefill chunk 阶段忽略 draft tokens
- **Multi-modal** (line 1556-1560)：encoder input 可以独立控制是否 chunked
- **Structured Output** (line 1339, 1658)：使用 structured output 时有特殊处理

## 3. 实验设计

### Exp1: Chunked Prefill 对 TTFT 的影响

**目标**：观察不同 `max_num_batched_tokens` 对 TTFT 的影响

**实验配置**：
- 模型：Qwen2.5-7B
- Prompt length：固定 4096 tokens
- Batch size：1（排除 batch 干扰）
- 对比配置：
  - Config A: `max_num_batched_tokens=8192` (不 chunk)
  - Config B: `max_num_batched_tokens=2048` (切 2 个 chunk)
  - Config C: `max_num_batched_tokens=1024` (切 4 个 chunk)

**预期结果**：
- Config A TTFT 最短（一次性完成 prefill）
- Config B/C TTFT 变长（chunk 调度开销）

### Exp2: Chunked Prefill 对混合负载的 ITL 影响

**目标**：验证 chunked prefill 能否改善混合负载下的 decode 延迟

**实验配置**：
- 模型：Qwen2.5-7B
- 负载：
  - 5 个长 prompt 请求（4096 tokens）同时到达
  - 每隔 0.5s 发送 1 个短 decode 请求（128 tokens prompt + 生成 256 tokens）
- 对比配置：
  - Config A: `max_num_batched_tokens=8192` (baseline)
  - Config B: `max_num_batched_tokens=2048` (chunked)

**观察指标**：
- 长请求的 TTFT（会变长）
- 短请求的 ITL P50/P99（应该改善）
- 整体吞吐（可能略降）

**预期结果**：
- Config B 的短请求 ITL P99 明显降低（长 prefill 不再阻塞 decode）
- Config B 的长请求 TTFT 增加（chunk 调度开销）

## 4. 实验脚本

### 准备工作

```bash
cd /home/lvzheng/vllm
source .venv/bin/activate  # 如果有 venv
```

### Exp1 脚本框架

```bash
# TODO: 实现 exp1_ttft_comparison.sh
# 1. 启动 vLLM server（3 种配置）
# 2. 发送固定 4096 tokens 的 prompt
# 3. 记录 TTFT
# 4. 对比分析
```

### Exp2 脚本框架

```bash
# TODO: 实现 exp2_mixed_workload.sh
# 1. 启动 vLLM server（2 种配置）
# 2. 同时发送 5 个长 prompt
# 3. 每 0.5s 发送 1 个短请求
# 4. 记录所有请求的 TTFT 和 ITL
# 5. 对比 ITL P99
```

## 5. 代码阅读清单

Day 1 重点阅读：
- [ ] `vllm/v1/core/sched/scheduler.py:439` - `schedule()` 主调度逻辑
- [ ] `vllm/v1/core/sched/scheduler.py:1335-1342` - `is_prefill_chunk` 判定
- [ ] `vllm/v1/core/sched/scheduler.py:2219-2222` - chunk 队列管理
- [ ] `vllm/config.py` - 查找 `max_num_batched_tokens` 配置定义

Day 2 重点阅读：
- [ ] `vllm/v1/core/sched/scheduler.py:369-435` - `_clip_prefill_chunk_for_mamba()` 边界对齐
- [ ] Speculative decoding 与 chunked prefill 的交互代码

## 6. 学习笔记

### 2026-09-03

#### 核心发现

1. **没有固定 chunk size**：chunk 大小由 `min(剩余token数, long_prefill_token_threshold, token_budget剩余)` 动态决定，每个 step 可能不同

2. **调度流程**：
   - waiting queue 中的请求通过 `min(num_new_tokens, token_budget)` 切分
   - 未完成的 prefill 请求进入 running queue，通过 `num_computed_tokens` 追踪进度
   - `is_prefill_chunk` 在 `_update_after_schedule` 中动态计算：`num_computed_tokens < num_tokens`
   - 完成判断：`num_computed_tokens >= num_prompt_tokens` 时进入 decode

3. **KV Cache 分配**：chunked prefill 和 decode 走同一个 `allocate_slots`，没有特殊处理。每个 chunk 独立分配 block

4. **Attention 计算**：prefill 和 decode 都是先 `reshape_and_cache_flash` 写入 KV Cache，再从 cache 中读取做 attention。区别在于 prefill chunk 用 flash attention（多 Q 对多 KV），decode 用 paged attention（1 Q 对多 KV）

5. **抢占策略**：running queue 中 block 不够时，PRIORITY 策略选 priority 值最大（优先级最低）+ 最晚到达的请求抢占；FCFS 策略弹出末尾请求。被抢占请求 KV Cache 全部释放、num_computed_tokens 归零、放回 waiting queue

6. **`scheduler_reserve_full_isl` 默认 True**：waiting queue 新请求默认检查整个序列 KV Cache 能否放下，避免做了几个 chunk 后被迫抢占

#### 实验结果

实验配置：Qwen2.5-7B, GPU 7, random-input-len=2048, random-output-len=128, 30 prompts, request-rate=5

| 配置 | TTFT P99 (ms) | ITL P99 (ms) | ITL Mean (ms) | Throughput (tok/s) |
|------|--------------|-------------|-------------|-------------------|
| no_chunked (baseline) | 181 | 35.9 | 7.19 | 561 |
| chunked_8192 | 172 | 44.1 | 7.28 | 561 |
| chunked_2048 | 185 | 34.6 | 7.31 | 560 |
| chunked_1024 | 184 | **22.3** | 7.08 | 561 |

**结论**：
- ITL P99：chunked_1024 最低（22ms vs baseline 36ms），小 chunk 有效减少 decode 卡顿
- TTFT P99：各配置差异不大，chunked prefill 对 TTFT 影响很小
- Throughput：几乎不变，chunked prefill 不影响整体吞吐
- 更长 prompt 或更高并发下差异会更明显

#### 关键认知

- **TP 并行优化 TTFT**（降低 prefill 计算时间），**chunked prefill 优化 ITL P99**（减少 decode 被阻塞）
- 两者优化目标不同，可以组合使用
