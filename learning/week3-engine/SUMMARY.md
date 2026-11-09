# Week 3 总结：ModelRunner 执行流程

> 学习时间：2026-08-21 ~ 2026-08-27
> 核心目标：建立从 HTTP 请求到 token 输出的端到端全景认知

---

## 📚 本周学习内容

### Day 1: 端到端全景 ✅

- vLLM v1 Engine 分层架构（AsyncLLM → EngineCore → Worker → ModelRunner）
- 各层职责定位与进程间通信（ZMQ IPC）
- 请求处理全链路概览

### Day 2-3: ModelRunner 执行流程 ✅

- execute_model() 主流程概念
- 输入构造中的 slot_mapping / block_table 计算
- KV Cache 物理内存寻址机制

### Day 4-5: Attention Metadata 构建 + Sampling 层 ✅

- Attention Metadata 构建
- Sampling 层实现（Logits → Temperature → Top-K → Top-P → 采样）
- Structured Output（grammar-guided decoding）

### Day 6: CUDA Graph 机制 ✅

- CUDA Graph Capture/Replay 原理
- vLLM 按 batch size 预录不同大小 graph
- Decode 阶段 kernel launch overhead 优化

### Day 7: Attention 变体的系统适配 ✅

- MLA（Multi-head Latent Attention）完整理解
- Sliding Window Attention block 管理

### Day 8: 总结 + 完整 trace 串联 🔲

- 未完成完整 trace 串联

---

## 🎯 核心知识点总结

### 1. Engine 整体架构

```
用户 API（OpenAI 兼容 / FastAPI）
    │
    ▼
AsyncLLM（API 层入口，独立进程）
    │  ZMQ socket 通信
    ▼
EngineCore（核心引擎进程）
    ├── Scheduler（调度决策）
    └── ModelExecutor → Worker → ModelRunner → Model（GPU 执行）
```

关键设计：
- AsyncLLM 和 EngineCore 运行在不同进程，通过 ZMQ 解耦
- EngineCore 内部是同步的 `step()` 循环
- ModelRunner 负责构造 GPU 输入并执行模型 forward

### 2. MLA（Multi-head Latent Attention）

**核心思想**：通过低秩压缩 KV Cache，极大减少显存占用。

**KV Cache Shape 对比**：
```
标准 Attention: (num_blocks, num_kv_heads, block_size, 2*head_size)
    → 128 heads × 128 dim = 16384 维

MLA:           (num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim)
    → 512 + 64 = 576 维（压缩约 28 倍）
```

**压缩与解压流程**：
```
压缩（attention 外，显式执行）：
  hidden_states → kv_a_proj (ReplicatedLinear) → latent (576维)
  latent 存入 KV Cache

解压（fused 在 attention kernel 内）：
  读取 latent → kv_b_proj (ColumnParallelLinear) 解压 → 部分 K/V → 计算 attention
```

**为什么压缩在外，解压在内**：
- 压缩输出（latent）必须物化存入 KV Cache → 必须在外部显式执行
- 解压输出（完整 K/V）只在计算中使用，不需要存储 → fused 避免显存访问

**TP 并行策略**：
- `kv_a_proj`：`ReplicatedLinear`（每卡完整权重）→ 输出相同 → KV Cache replicate
- `kv_b_proj`：`ColumnParallelLinear`（按列切分）→ 每卡只解压自己负责的 head
- 不需要 AllReduce：输入相同 + 权重相同 = 输出相同

### 3. Sliding Window Attention（SWA）

**核心思想**：每个 token 只 attend 最近 `window_size` 个 token，窗口外的 block 可提前释放。

**释放逻辑**：
```python
# 计算窗口外 token 数（持续累加）
skipped_tokens = max(0, num_computed_tokens - sliding_window + 1)

# 向下取整：只释放整个 block 都在窗口外的（保证安全）
skipped_blocks = skipped_tokens // block_size

# 释放后用 null_block 占位，保持 list 索引不变
blocks[i] = null_block
```

**调用时机**（每次 allocate_slots 前）：
```
Scheduler.allocate_slots()
  → KVCacheManager.allocate_slots()
    → coordinator.remove_skipped_blocks()  # 先回收再分配
      → SlidingWindowManager.remove_skipped_blocks()
```

**不重复释放**：遍历时遇到已是 `null_block` 的就 break。

**与 Full Attention 对比**：

| | Full Attention | SWA |
|---|---|---|
| Block 释放 | 请求结束才释放 | 滑出窗口就释放 |
| 单请求最大 block 数 | 无上限 | 有上限（`window_size / block_size`）|
| Prefix cache 命中 | 任意连续命中 | 需连续 block ≥ `ceil(window_size / block_size)` |

### 4. 多类型 Attention 共存

每种 attention 类型有独立的 KV Cache group 和 block table：
```
KVCacheManager（对外接口）
    └── Coordinator（协调多个 group）
            └── SingleTypeKVCacheManager 子类（每个 group 一个）
                    ├── FullAttentionManager
                    ├── SlidingWindowManager
                    ├── RSWAManager
                    ├── ChunkedLocalAttentionManager
                    └── MambaManager
```

同一 group 内多层共享 block table；不同 group 独立管理。
新 attention 类型可通过实现新的 Manager 子类接入。

### 5. KV Cache 物理内存寻址

**虚拟页表机制**：
```
position → block_index = position // block_size        # 逻辑 block 编号
         → block_number = block_table[req][block_index] # 查表得物理 block ID
         → slot_id = block_number * block_size + offset  # 物理 slot 地址
```

**物理内存是一整块连续 tensor**：
- 启动时一次性分配：`torch.zeros(total_size, device=gpu)`
- 物理 block ID = tensor 第一维下标
- 不存在额外的"起始地址映射表"，数组下标即地址

---

## 关键代码位置

| 内容 | 文件 | 关键行 |
|------|------|--------|
| AsyncLLM 入口 | `vllm/v1/engine/async_llm.py` | — |
| EngineCore step 循环 | `vllm/v1/engine/core.py` | — |
| MLA forward | `vllm/model_executor/layers/mla.py` | 150-226 |
| MLA kv_a_proj (Replicated) | `vllm/model_executor/models/deepseek_v2.py` | 512 |
| SWA Manager | `v1/core/single_type_kv_cache_manager.py` | 878-1093 |
| SWA 释放调用 | `v1/core/kv_cache_manager.py` | 495-508 |
| slot_mapping Triton kernel | `v1/worker/block_table.py` | 380-443 |
| KV Cache tensor 分配 | `v1/worker/gpu_model_runner.py` | 7317-7345 |
| KVCacheGroupSpec 定义 | `v1/kv_cache_interface.py` | 958-970 |

---

## Week 2 遗留问题解决情况

| 问题 | 状态 |
|------|------|
| Prefix Caching 的 hash 计算和命中逻辑细节 | 🔲 Week 5 学习 |
| ModelRunner 如何把 scheduler_output 转成 GPU 输入 | ✅ 已理解 |
| execute_model() 内部 TP 同步是怎么做的 | ✅ 通过 MLA 理解了 TP 下权重切分和通信 |
| Chunked Prefill 的 chunk 大小如何确定 | 🔲 Week 5 学习 |

---

## 未完成内容

无。Week 3 计划内容全部完成。

---

## 学习状态

- ✅ Day 1: Engine 架构概览（AsyncLLM, EngineCore, IPC）
- ✅ Day 2-3: ModelRunner 执行流程（execute_model、slot_mapping/block_table 计算）
- ✅ Day 4-5: Attention Metadata 构建 + Sampling 层 + Structured Output
- ✅ Day 6: CUDA Graph 机制（Capture/Replay、Warmup）
- ✅ Day 7: Attention 变体系统适配（MLA + SWA）
- ✅ Day 8: 总结 + SUMMARY 编写

**Week 3 完成时间**: 2026-08-27

---

## 下周计划

**Week 4: 分布式推理基础**
- Day 1-2: Tensor Parallelism（TP）— Column/Row Parallel、权重分片加载
- Day 3-4: Pipeline Parallelism（PP）+ Data Parallelism（DP）
- Day 5-6: Context Parallelism + NCCL 通信原语
- Day 7: 多节点网络配置

**重点问题**：
1. TP 下各层的通信模式（AllReduce vs AllGather）？
2. PP 的 micro-batch 调度与 bubble 率？
3. DP Attention 如何实现？
