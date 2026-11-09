# vLLM 学习路径规划（v1 架构）
# 适用对象：熟悉CUDA/CUTLASS/Triton的算子工程师

## 总体目标
从底层算子向上构建完整的推理系统认知，成为大模型推理优化部署专家。
覆盖单机单卡到大规模分布式推理优化部署的全链路能力。

## 学习周期：10周
每周约15-20小时（每天2-3小时）

## 当前环境
- 代码路径：`/home/lvzheng/vllm/vllm`
- vLLM 版本：v1 架构
- 学习记录：`/home/lvzheng/vllm/vllm/learning/`

---

# 阶段1：算子到系统的映射（Week 1-2）

## Week 1: KV Cache 管理 - 从 Block Pool 到推理框架 ✅

### 学习目标
1. 理解 PagedAttention 的核心创新（虚拟内存式的 KV Cache 管理）
2. 掌握 Block Manager 的分配和释放机制
3. 理解 v1 架构的 attention backend 设计
4. 建立"算子-系统"的认知桥梁

### Day 1: PagedAttention 原理 ✅

**理论学习**（2小时）
1. 阅读 vLLM 论文关键章节（重点看 Figure 2-3）
   - 论文：https://arxiv.org/abs/2309.06180
   - 重点：传统 KV Cache vs Paged KV Cache 的内存碎片问题
   
2. 理解核心概念：
   - Block Table：逻辑地址到物理地址的映射
   - Block Size：通常 16 或 32 个 token
   - Block Pool：全局统一的物理 block 池
   - Copy-on-Write：前缀共享机制

**实践任务**（1小时）
```bash
# 1. 阅读 Block Manager 代码
cd /home/lvzheng/vllm/vllm

# 核心文件
cat vllm/v1/core/kv_cache_manager.py | less
# 重点：allocate_slots(), free(), get_num_free_blocks()

cat vllm/v1/core/block_pool.py | less
# 重点：get_new_blocks(), free_blocks(), BlockPool 初始化

# 2. 绘制数据结构图
mkdir -p /home/lvzheng/vllm/vllm/learning/week1-kv-cache
cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/block_manager.md << 'EOF'
# KV Cache 管理核心数据结构

## Block Table (per request)
- 每个请求维护一个 block_table：tuple[list[int], ...]
- 第一层 tuple：对应每个 KV Cache Group（大部分模型只有 1 个）
- 第二层 list：该 group 中的物理 block ID 列表
- 例如：([3, 7, 12], [15, 20]) 表示 group 0 用 block 3/7/12，group 1 用 15/20

## Block Pool (全局唯一)
- 预先分配的 GPU 内存池
- free_block_queue：空闲 block 的双端队列
- 每个 block 有 ref_cnt 标记占用状态
- 每个 block 大小：block_size * num_kv_heads * head_size * 2 (K+V)

## KV Cache Group
- 不同的 attention 类型可能需要不同的缓存策略
- 普通 decoder：1 个 group (Full Attention)
- Encoder-decoder：2 个 group (Self-Attention + Cross-Attention)

## 关键操作
1. allocate_slots: 从 free_queue 头部拿 block，ref_cnt++
2. free: 放回 free_queue，ref_cnt--（有 hash 放尾，无 hash 放头）
3. copy: CoW 时复制 block（Week 5 Prefix Caching）

[TODO: 画出具体的内存布局图]
EOF
```

### Day 2: Block Manager 核心逻辑 ✅

**代码深入**（2小时）
```bash
cd /home/lvzheng/vllm/vllm

# 1. 追踪 allocate_slots() 完整流程
# KVCacheManager.allocate_slots()
#   → Coordinator.get_num_blocks_to_allocate()  # 计算需要几个
#   → Coordinator.allocate_new_blocks()
#     → SingleTypeKVCacheManager.allocate_new_blocks()  # 遍历每个 group
#       → BlockPool.get_new_blocks()  # 实际分配 + 更新状态
#         → free_block_queue.popleft_n(num_blocks)  # 从队列拿
#         → block.ref_cnt += 1  # 标记占用

# 重点文件和行号
# vllm/v1/core/kv_cache_manager.py:445-552 (allocate_slots)
# vllm/v1/core/kv_cache_coordinator.py:262-271 (allocate_new_blocks)
# vllm/v1/core/single_type_kv_cache_manager.py:365 (调用 block_pool)
# vllm/v1/core/block_pool.py:661-677 (get_new_blocks 核心逻辑)

# 2. 追踪 free() 流程
# KVCacheManager.free()
#   → Coordinator.free()  # 遍历所有 group
#     → SingleTypeKVCacheManager.free()
#       → BlockPool.free_blocks()  # 倒序释放
#         → block.ref_cnt -= 1
#         → 如果 ref_cnt == 0，放回 free_queue
#           - 有 hash 的 append（放尾部，可能保留为缓存）
#           - 无 hash 的 prepend（放头部，优先复用）

# vllm/v1/core/kv_cache_manager.py:567-578 (free)
# vllm/v1/core/block_pool.py:731-742 (free_blocks)
```

**实践任务**（1小时）
```bash
# 创建流程分析文档
cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/allocation_flow.md << 'EOF'
# Block 分配与释放流程

## 分配流程（allocate_slots）

### 输入
- request: Request 对象
- num_tokens_need_slot: 本次需要多少 token 的空间
- num_encoder_tokens: encoder tokens（encoder-decoder 模型）

### 三层调用
1. **KVCacheManager 层**：计算需要多少新 block
   - 调用 coordinator.get_num_blocks_to_allocate()
   - 检查 block pool 是否有足够的空闲 block
   
2. **Coordinator 层**：遍历每个 KV Cache Group
   - 对每个 group 调用 manager.allocate_new_blocks()
   - 返回 tuple，每个元素对应一个 group 的 block 列表
   
3. **BlockPool 层**：实际分配
   - free_block_queue.popleft_n(num_blocks)：从队列头拿
   - block.ref_cnt += 1：标记为占用
   - 返回 block 列表

### 关键：多 group 共享同一个 block_pool
- 所有 manager 使用同一个 BlockPool 实例
- 串行分配，状态实时更新
- Group 0 拿走的 block，Group 1 不会再拿到

## 释放流程（free）

### 倒序释放
- 从尾部 block 开始释放（最近生成的）
- 头部 block（prompt）复用概率高，晚释放有利于 caching

### 放回策略
- **有 hash 的 block**：append 到队列尾
  - 保留更长时间，可能被 prefix caching 命中
- **无 hash 的 block**：prepend 到队列头
  - 优先复用，提高 GPU cache locality

## 数据结构示例
```python
# 请求 A 的 block table
req_to_blocks[request_A] = KVCacheBlocks(
    blocks=(
        [3, 7, 12],   # group 0: Full Attention
        [15, 20],     # group 1: Cross Attention (如果有)
    )
)

# Block Pool 状态
free_block_queue = deque([0, 1, 2, 4, 5, 6, 8, ...])  # 空闲的
block_3.ref_cnt = 1  # 被请求 A 占用
block_7.ref_cnt = 1
```
EOF
```

### Day 3: Attention Backend 调用链 ✅

**理解 v1 架构的 Attention Backend**（2小时）
```bash
cd /home/lvzheng/vllm/vllm

# v1 架构使用第三方 attention 库，不是自己实现的 paged_attention kernel
# 主要 backend：FlashInfer, FlashAttention, Triton

# 1. 查看可用的 attention backend
ls vllm/v1/attention/backends/
# flash_attn.py - FlashAttention backend
# flashinfer.py - FlashInfer backend (推荐，性能最好)
# triton_attn.py - Triton 实现
# rocm_*.py - AMD GPU 支持
# mla/ - Multi-head Latent Attention

# 2. 阅读 FlashInfer backend（最常用）
cat vllm/v1/attention/backends/flashinfer.py | grep -A 30 "def forward"

# 3. 理解 attention metadata 构建
# Metadata 包含 block_table、序列长度等信息
grep -A 20 "class.*Metadata" vllm/v1/attention/backends/flashinfer.py

# 4. 找到实际的 kernel 调用
grep -n "flashinfer\." vllm/v1/attention/backends/flashinfer.py | head -20
```

**追踪 block_table 的传递**（1小时）
```bash
# block_table 从哪里来，怎么传给 attention kernel？

# 1. Scheduler 决定哪些请求参与本轮
# vllm/v1/core/scheduler.py

# 2. 构造 model input，包含 block_table
# vllm/v1/worker/gpu_model_runner.py:prepare_model_input()

# 3. Attention metadata builder 封装 block_table
# vllm/v1/attention/backends/flashinfer.py: FlashInferMetadataBuilder

# 4. Attention forward 使用 metadata 调用 kernel
# vllm/v1/attention/backends/flashinfer.py: FlashInferImpl.forward()

cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/attention_backend.md << 'EOF'
# v1 架构的 Attention Backend

## 设计思想
- **不自己实现 kernel**：使用 FlashInfer/FlashAttention 等优化库
- **Backend 抽象**：统一接口，支持多种实现
- **Metadata 封装**：将 block_table、序列长度等信息打包

## 主要 Backend

### FlashInfer（默认推荐）
- 项目：https://github.com/flashinfer-ai/flashinfer
- 特点：高度优化的 paged attention kernel
- 优势：decode 性能最好，支持多种 attention pattern

### FlashAttention
- 项目：https://github.com/Dao-AILab/flash-attention
- 特点：IO-aware attention，减少 HBM 访问
- 优势：prefill 性能好

### Triton
- 纯 Python DSL 实现
- 优势：易于定制和调试
- 劣势：性能略低于 CUDA kernel

## Block Table 传递流程

```
Scheduler.schedule()
  → 决定哪些请求参与本轮
  → 返回 SchedulerOutput（包含 scheduled_requests）
    ↓
GPUModelRunner.execute_model()
  → prepare_model_input(scheduled_requests)
  → 构造 ModelInput（包含 attn_metadata）
    ↓
AttentionMetadataBuilder.build()
  → 从 KVCacheManager 获取每个请求的 block_table
  → 构造 GPU tensor: block_tables [num_seqs, max_blocks]
    ↓
AttentionImpl.forward(query, key, value, attn_metadata)
  → flashinfer.single_prefill_with_kv_cache()  # Prefill
  → flashinfer.single_decode_with_kv_cache()   # Decode
  → 传入 block_tables tensor
    ↓
FlashInfer CUDA Kernel
  → 根据 block_tables[seq_id] 找到物理 block
  → 读取 KV cache 计算 attention
```

## 关键数据结构

### AttentionMetadata
```python
@dataclass
class FlashInferMetadata:
    # Decode 阶段
    decode_wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper
    block_tables: torch.Tensor  # [num_seqs, max_blocks]
    
    # Prefill 阶段
    prefill_wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper
    # ...
```

### 从 KVCacheManager 获取 block_table
```python
# vllm/v1/worker/gpu_model_runner.py
for req in scheduled_requests:
    blocks = kv_cache_manager.get_block_table(req)  # tuple[list[int], ...]
    # 转成 tensor 传给 attention kernel
```

[TODO: 绘制完整的数据流图]
EOF
```

### Day 4: 实践 - 运行和观察 ✅

**任务1: 运行简单推理并观察日志**（1.5小时）
```bash
mkdir -p /home/lvzheng/vllm/vllm/learning/week1-kv-cache

# 1. 编写测试脚本
cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/test_inference.py << 'EOF'
import torch
from vllm import LLM, SamplingParams
import os

# 开启详细日志
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

# 使用小模型
model = LLM(
    model="facebook/opt-125m",
    max_num_seqs=4,
    block_size=16,
    gpu_memory_utilization=0.3,
)

prompts = [
    "Hello, my name is",
    "The capital of France is",
    "In a world where",
]

print("=" * 60)
print("Starting generation...")
print("=" * 60)

outputs = model.generate(
    prompts,
    SamplingParams(max_tokens=20, temperature=0)
)

for output in outputs:
    print(f"\nPrompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}")
    
print("\n" + "=" * 60)
print("Check logs for block allocation details")
print("=" * 60)
EOF

# 2. 运行并保存日志
cd /home/lvzheng/vllm/vllm
python learning/week1-kv-cache/test_inference.py 2>&1 | \
    tee learning/week1-kv-cache/inference.log

# 3. 分析日志中的关键信息
grep -i "block" learning/week1-kv-cache/inference.log | head -20
grep -i "allocat" learning/week1-kv-cache/inference.log | head -20
```

**任务2: 使用 torch.profiler 分析**（1小时）
```bash
cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/profile_attention.py << 'EOF'
import torch
from vllm import LLM, SamplingParams

model = LLM(
    model="facebook/opt-125m",
    max_num_seqs=2,
    gpu_memory_utilization=0.3,
)

prompts = ["Hello " * 30] * 2

# Warmup
_ = model.generate(prompts, SamplingParams(max_tokens=5))

# Profile
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    outputs = model.generate(
        prompts,
        SamplingParams(max_tokens=20, temperature=0)
    )

# 保存结果
prof.export_chrome_trace("learning/week1-kv-cache/profile_trace.json")
print("\nProfile saved to learning/week1-kv-cache/profile_trace.json")
print("Open chrome://tracing in Chrome to visualize")

# 打印 top 操作
print("\n" + "=" * 60)
print("Top CUDA operations:")
print("=" * 60)
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=10
))
EOF

python /home/lvzheng/vllm/vllm/learning/week1-kv-cache/profile_attention.py
```

**任务3: 观察不同 block_size 的影响**（0.5小时）
```bash
cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/compare_block_size.py << 'EOF'
import torch
import time
from vllm import LLM, SamplingParams

prompts = ["Hello " * 50] * 4

for block_size in [8, 16, 32]:
    print(f"\n{'='*60}")
    print(f"Testing with block_size={block_size}")
    print('='*60)
    
    model = LLM(
        model="facebook/opt-125m",
        block_size=block_size,
        gpu_memory_utilization=0.3,
    )
    
    # Warmup
    _ = model.generate(prompts, SamplingParams(max_tokens=1))
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    outputs = model.generate(
        prompts,
        SamplingParams(max_tokens=50, temperature=0)
    )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Total tokens generated: {total_tokens}")
    
    # 清理
    del model
    torch.cuda.empty_cache()
EOF

python /home/lvzheng/vllm/vllm/learning/week1-kv-cache/compare_block_size.py
```

### Day 5: 深入理解与总结 ✅

### Day 5: 深入理解与总结

**任务1: 追踪完整的 KV Cache 更新流程**（1.5小时）
```bash
cd /home/lvzheng/vllm/vllm

# 问题：生成的 KV Cache 什么时候写入 block？

# 1. Attention forward 同时更新 KV Cache
# vllm/v1/attention/backends/flashinfer.py
# FlashInfer 的 forward_includes_kv_cache_update = True
# 意味着 attention forward 时直接写入 kv_cache tensor

# 2. 找到 KV Cache tensor 的分配
grep -n "kv_cache" vllm/v1/worker/gpu_model_runner.py | head -20

# 3. 理解 KV Cache 的物理布局
# vllm/v1/core/kv_cache_manager.py: get_kv_cache()
# 返回的是整个 block pool 的 tensor 引用

cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/kv_cache_update.md << 'EOF'
# KV Cache 更新流程

## 物理存储
```python
# 整个 block pool 是一个大 tensor
kv_cache = torch.zeros(
    [num_blocks, 2, num_kv_heads, head_size, block_size],
    dtype=torch.float16,
    device="cuda"
)
# shape 说明：
# - num_blocks: 总共有多少物理 block
# - 2: Key 和 Value
# - num_kv_heads: KV head 数量
# - head_size: 每个 head 的维度
# - block_size: 每个 block 能存多少 token
```

## 写入时机
```
ModelRunner.execute_model()
  ↓
Model.forward()  # 例如 LlamaForCausalLM
  ↓
LlamaAttention.forward()
  ↓
AttentionImpl.forward(query, key, value, kv_cache, block_tables)
  ↓
FlashInfer Kernel
  - 计算 attention output
  - **同时**将当前 token 的 key/value 写入 kv_cache[block_id]
```

## Block Table 的作用
```python
# 假设 block_size = 16，请求 A 有 50 个 token

# 分配的 block_table
request_A.block_table = [3, 7, 12]  # 3 个 block

# Kernel 内部逻辑：
for token_id in range(50):
    block_idx = token_id // 16     # token 属于哪个逻辑 block
    offset = token_id % 16         # 在 block 内的偏移
    physical_block = block_table[block_idx]  # 查表得到物理 block
    
    # 写入 KV
    kv_cache[physical_block, 0, :, :, offset] = key[token_id]   # K
    kv_cache[physical_block, 1, :, :, offset] = value[token_id] # V
```

## 关键点
1. **KV Cache 是全局共享的大 tensor**，不是每个请求独立的
2. **Block Table 提供间接寻址**，让多个请求共享物理内存
3. **Attention forward 直接写入**，不需要单独的 copy 操作
4. **Paged 设计避免内存碎片**，可以动态增长

[TODO: 用实际代码验证 kv_cache tensor 的 shape]
EOF
```

**任务2: 编写 Week 1 总结**（1小时）
```bash
cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/SUMMARY.md << 'EOF'
# Week 1 总结：KV Cache 管理深入理解

## 核心收获

### 1. PagedAttention 的本质
- **虚拟内存思想**：Block Table 提供逻辑到物理的映射
- **解决的问题**：传统连续内存分配的内存碎片和浪费
- **关键创新**：
  - 动态分配：按需分配 block，用多少拿多少
  - 共享物理内存：多个请求通过不同 block_table 共享同一个 kv_cache tensor
  - Copy-on-Write：前缀相同的请求可以共享 block（Week 5 详细学习）

### 2. Block Manager 实现细节

**数据结构**：
- `BlockPool`: 全局唯一的物理 block 池，维护 `free_block_queue`
- `req_to_blocks`: 每个请求的 block table (tuple of lists)
- `KVCacheGroup`: 支持不同 attention 类型的分组管理

**分配流程**：
```
allocate_slots()
  → get_num_blocks_to_allocate()  # 计算需要几个
  → 检查是否有足够空闲 block
  → allocate_new_blocks()         # 实际分配
    → 遍历每个 group
    → block_pool.get_new_blocks() # 从 free_queue 拿，ref_cnt++
```

**释放流程**：
```
free()
  → coordinator.free()            # 遍历所有 group
  → block_pool.free_blocks()      # 倒序释放
    → ref_cnt--
    → 放回 free_queue（有 hash 放尾，无 hash 放头）
```

### 3. v1 架构的 Attention Backend

**不自己实现 kernel**：
- 使用 FlashInfer（推荐）、FlashAttention、Triton 等第三方库
- Backend 抽象：统一接口，可插拔设计
- Metadata 封装：将 block_table、序列长度等打包传给 kernel

**数据流**：
```
Scheduler → ModelRunner → AttentionMetadataBuilder
  → 从 KVCacheManager 获取 block_table
  → 构造 GPU tensor: block_tables [num_seqs, max_blocks]
  → AttentionImpl.forward() 调用 FlashInfer kernel
```

## 技术要点

### Block Table 映射机制
```python
# 逻辑位置 → 物理位置
token_id = 35, block_size = 16
logical_block = 35 // 16 = 2  # 第 2 个逻辑 block
offset = 35 % 16 = 3          # block 内偏移 3
physical_block = block_table[2]  # 查表
kv_cache[physical_block, :, :, :, offset]  # 实际访问
```

### Multi-Group 设计
- 普通 decoder：1 个 group（Full Attention）
- Encoder-decoder：2 个 group（Self + Cross Attention）
- 所有 group 共享同一个 BlockPool，串行分配

### KV Cache 物理布局
```python
kv_cache.shape = [num_blocks, 2, num_kv_heads, head_size, block_size]
# 全局唯一的大 tensor，通过 block_table 间接访问
```

## 实践经验

### 观察到的现象
1. **Block size 影响**：
   - block_size=8: 更灵活，但 block 数量多，查表开销大
   - block_size=32: block 数量少，但可能浪费空间
   - block_size=16: 平衡点（默认）

2. **内存使用**：
   - Prefill 阶段：大量分配 block
   - Decode 阶段：逐步增长，复用已分配的 block

3. **日志关键信息**：
   - `allocate_slots`: 分配多少 block
   - `free`: 释放哪些请求的 block
   - `num_free_blocks`: 当前空闲 block 数

## 遗留问题

1. **Prefix Caching 细节**（Week 5）：
   - `remove_skipped_blocks()` 的逻辑
   - 有 hash 的 block 如何被复用
   - Hash 计算和匹配机制

2. **分布式场景**（Week 4）：
   - 多卡之间的 KV Cache 同步
   - Pipeline Parallelism 的 KV Cache 管理

3. **Chunked Prefill**（Week 5）：
   - 长 prompt 如何分块处理
   - 与 block allocation 的配合

## 性能数据

[TODO: 填入实际测量的数据]
- OPT-125M, 4 prompts × 50 tokens
- block_size=16: XXX tokens/s
- block_size=32: XXX tokens/s
- Attention kernel 占比: XX%

## 下周计划

**Week 2: Continuous Batching 与 Scheduler**
- Day 1-2: Scheduler 调度策略
- Day 3-4: 请求生命周期管理
- Day 5: Prefill vs Decode 混合调度
- Day 6-7: 实践与总结

**重点问题**：
1. Scheduler 如何决策哪些请求分配 block？
2. 内存不足时的处理策略（swap、preempt）？
3. Prefill 和 Decode 如何混合调度以提高吞吐？
EOF
```

**任务3: 回顾与预习**（0.5小时）
```bash
# 1. 检查学习成果
ls -la /home/lvzheng/vllm/vllm/learning/week1-kv-cache/
# 应该有：
# - block_manager.md
# - allocation_flow.md
# - attention_backend.md
# - kv_cache_update.md
# - test_inference.py + inference.log
# - profile_attention.py + profile_trace.json
# - compare_block_size.py
# - SUMMARY.md

# 2. 预习 Week 2 内容
cat vllm/v1/core/scheduler.py | head -100
# 快速浏览 Scheduler 的接口和核心方法

# 3. 标记完成状态
cat >> /home/lvzheng/vllm/vllm/learning/week1-kv-cache/SUMMARY.md << 'EOF'

## 学习状态
- ✅ Day 1: PagedAttention 原理
- ✅ Day 2: Block Manager 核心逻辑
- ✅ Day 3: Attention Backend 调用链
- ✅ Day 4: 实践与观察
- ✅ Day 5: 深入理解与总结

**Week 1 完成时间**: $(date +%Y-%m-%d)
EOF
```

### Day 6-7: 周末巩固（可选）

### Day 6-7: 周末巩固（可选）

**高级任务1: 简化版 Block Manager 实现**
```bash
cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/simple_block_manager.py << 'EOF'
"""
简化版的 Block Manager，理解核心思想
不追求完整实现，只实现核心逻辑
"""
from dataclasses import dataclass
from typing import List, Dict
from collections import deque

@dataclass
class Block:
    """物理 block"""
    block_id: int
    ref_cnt: int = 0
    
class SimpleBlockManager:
    """简化版 Block Manager"""
    
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        
        # 初始化所有 block 为空闲状态
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_queue = deque(range(num_blocks))
        
        # 每个请求的 block table
        self.req_to_blocks: Dict[str, List[int]] = {}
    
    def allocate(self, request_id: str, num_tokens: int) -> bool:
        """为请求分配 block"""
        # 计算需要多少 block
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        # 检查是否有足够的空闲 block
        if len(self.free_queue) < num_blocks_needed:
            return False
        
        # 分配 block
        allocated_blocks = []
        for _ in range(num_blocks_needed):
            block_id = self.free_queue.popleft()
            self.blocks[block_id].ref_cnt = 1
            allocated_blocks.append(block_id)
        
        # 记录到 block table
        if request_id not in self.req_to_blocks:
            self.req_to_blocks[request_id] = []
        self.req_to_blocks[request_id].extend(allocated_blocks)
        
        return True
    
    def free(self, request_id: str):
        """释放请求的所有 block"""
        if request_id not in self.req_to_blocks:
            return
        
        block_ids = self.req_to_blocks.pop(request_id)
        
        # 倒序释放
        for block_id in reversed(block_ids):
            self.blocks[block_id].ref_cnt -= 1
            if self.blocks[block_id].ref_cnt == 0:
                self.free_queue.append(block_id)
    
    def get_block_table(self, request_id: str) -> List[int]:
        """获取请求的 block table"""
        return self.req_to_blocks.get(request_id, [])
    
    def get_num_free_blocks(self) -> int:
        """获取空闲 block 数量"""
        return len(self.free_queue)

# 测试
if __name__ == "__main__":
    manager = SimpleBlockManager(num_blocks=10, block_size=16)
    
    print("Initial free blocks:", manager.get_num_free_blocks())
    
    # 请求 A: 50 tokens (需要 4 个 block)
    success = manager.allocate("request_A", 50)
    print(f"\nAllocate 50 tokens for request_A: {success}")
    print(f"Block table: {manager.get_block_table('request_A')}")
    print(f"Free blocks: {manager.get_num_free_blocks()}")
    
    # 请求 B: 30 tokens (需要 2 个 block)
    success = manager.allocate("request_B", 30)
    print(f"\nAllocate 30 tokens for request_B: {success}")
    print(f"Block table: {manager.get_block_table('request_B')}")
    print(f"Free blocks: {manager.get_num_free_blocks()}")
    
    # 释放请求 A
    manager.free("request_A")
    print(f"\nFree request_A")
    print(f"Free blocks: {manager.get_num_free_blocks()}")
    
    # 再次分配
    success = manager.allocate("request_C", 80)
    print(f"\nAllocate 80 tokens for request_C: {success}")
    print(f"Block table: {manager.get_block_table('request_C')}")
    print(f"Free blocks: {manager.get_num_free_blocks()}")
EOF

python /home/lvzheng/vllm/vllm/learning/week1-kv-cache/simple_block_manager.py
```

**高级任务2: 可视化 Block 分配过程**
```bash
cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/visualize_blocks.py << 'EOF'
"""
可视化 Block 分配和释放过程
"""
from simple_block_manager import SimpleBlockManager

def visualize_blocks(manager: SimpleBlockManager, title: str):
    """可视化当前 block 状态"""
    print(f"\n{title}")
    print("=" * 60)
    
    # 显示每个 block 的状态
    print("Block States:")
    for i, block in enumerate(manager.blocks):
        status = "FREE" if block.ref_cnt == 0 else f"USED(ref={block.ref_cnt})"
        print(f"Block {i:2d}: [{status:12s}]", end="  ")
        if (i + 1) % 5 == 0:
            print()
    print()
    
    # 显示 block table
    print("\nBlock Tables:")
    for req_id, blocks in manager.req_to_blocks.items():
        print(f"  {req_id}: {blocks}")
    
    print(f"\nFree blocks: {manager.get_num_free_blocks()}")
    print("=" * 60)

# 模拟完整的分配-释放过程
manager = SimpleBlockManager(num_blocks=10, block_size=16)

visualize_blocks(manager, "Initial State")

# 场景1: 3个请求依次到达
manager.allocate("req_A", 50)  # 4 blocks
visualize_blocks(manager, "After allocating req_A (50 tokens)")

manager.allocate("req_B", 30)  # 2 blocks
visualize_blocks(manager, "After allocating req_B (30 tokens)")

manager.allocate("req_C", 20)  # 2 blocks
visualize_blocks(manager, "After allocating req_C (20 tokens)")

# 场景2: req_B 先完成
manager.free("req_B")
visualize_blocks(manager, "After req_B finished")

# 场景3: 新请求到达
manager.allocate("req_D", 25)  # 2 blocks
visualize_blocks(manager, "After allocating req_D (25 tokens)")

# 场景4: 清理所有请求
manager.free("req_A")
manager.free("req_C")
manager.free("req_D")
visualize_blocks(manager, "After all requests finished")
EOF

python /home/lvzheng/vllm/vllm/learning/week1-kv-cache/visualize_blocks.py
```

**总结与反思**
```bash
cat > /home/lvzheng/vllm/vllm/learning/week1-kv-cache/REFLECTION.md << 'EOF'
# Week 1 反思

## 最重要的三个收获
1. 
2. 
3. 

## 仍然困惑的点
1. 
2. 

## 对后续学习的启发
1. 
2. 

## 学习方法的调整
- 什么方法有效？
- 什么需要改进？
- 下周如何优化学习效率？
EOF
```

**Week 1 学习状态**
- ✅ Day 1: PagedAttention 原理
- ✅ Day 2: Block Manager 核心逻辑
- ✅ Day 3: Attention Backend 调用链
- ✅ Day 4: 实践与观察
- ✅ Day 5: 深入理解与总结
- ✅ Day 6-7: 周末巩固

**Week 1 完成时间**: 2026-08-13

---

## Week 2: 连续批处理（Continuous Batching） ✅

### 学习目标
1. 理解为什么需要动态批处理
2. 掌握 Scheduler 的调度策略
3. 理解 Prefill 和 Decode 的混合调度
4. 掌握请求生命周期管理

### Day 1: Continuous Batching 原理 ✅

**理论学习**（1.5小时）
```bash
# 核心问题理解
cat > /home/lvzheng/vllm/vllm/learning/week2-scheduler/continuous_batching.md << 'EOF'
# Continuous Batching 核心思想

## 传统 Static Batching 的问题

### 问题1: 等待最慢请求
```
Batch = [Request A, Request B, Request C]
完成时间：A=10s, B=8s, C=15s

传统方法：
- 等到最慢的 C 完成（15s）才释放整个 batch
- B 完成后闲置 7s，A 完成后闲置 5s
- GPU 利用率低
```

### 问题2: 长度不可预知
- 生成长度依赖于模型输出（遇到 EOS 或 max_tokens）
- 无法提前知道每个请求何时完成
- 批次大小固定导致资源浪费

## Continuous Batching 解决方案

### 核心思想: Iteration-level Scheduling
```
每个 iteration（生成一个 token）后重新调度：
- 完成的请求立即移出
- 新请求立即加入
- 动态调整 batch 大小
```

### 示例
```
Iteration 0: [A, B, C]       生成各自的第1个token
Iteration 1: [A, B, C, D]    D 新加入
Iteration 2: [A, C, D]       B 完成，移出
Iteration 3: [A, C, D, E]    E 新加入
Iteration 4: [C, D, E]       A 完成
...
```

### 优势
1. **更高的 GPU 利用率**: 完成的请求立即替换为新请求
2. **更低的延迟**: 新请求不需要等待整个 batch 完成
3. **更高的吞吐**: 动态批处理减少空闲时间

## vLLM 的实现

### 三个队列
1. **Waiting**: 等待调度的新请求
2. **Running**: 正在执行的请求
3. **Swapped**: 内存不足时被换出的请求

### 调度决策
每个 iteration:
1. 从 Running 中移除完成的请求
2. 尝试从 Waiting 加入新请求（如果有足够的 KV Cache）
3. 如果内存不足，可能需要 swap out 部分请求

[TODO: 画出状态转换图]
EOF

# 阅读相关博客
echo "推荐阅读: https://www.anyscale.com/blog/continuous-batching-llm-inference"
```

**代码初探**（1小时）
```bash
cd /home/lvzheng/vllm/vllm

# 1. Scheduler 核心接口
cat vllm/v1/core/scheduler.py | head -150

# 重点理解：
# - schedule() 方法：主调度逻辑
# - 三个队列的管理
# - 调度策略

# 2. 快速浏览关键数据结构
grep -n "class.*Request" vllm/v1/core/scheduler.py | head -10
grep -n "class.*Output" vllm/v1/core/scheduler.py | head -10
```

### Day 2-3: Scheduler 深入剖析 ✅

**任务1: 完整追踪 Scheduler.schedule() 方法**（2小时）
```bash
cd /home/lvzheng/vllm/vllm

# 1. 阅读 Scheduler 核心代码
cat vllm/v1/core/scheduler.py | less

# 关键方法和行号（需要在实际代码中确认）:
# - schedule(): 主调度逻辑
# - _schedule_running(): 处理正在运行的请求
# - _schedule_waiting(): 尝试调度等待中的请求
# - _schedule_swapped(): 尝试恢复被换出的请求

# 2. 创建分析文档
cat > /home/lvzheng/vllm/vllm/learning/week2-scheduler/scheduler_analysis.md << 'EOF'
# Scheduler 调度逻辑深入分析

## 数据结构

### Request 状态
```python
class RequestStatus(Enum):
    WAITING = "waiting"     # 等待调度
    RUNNING = "running"     # 正在执行
    SWAPPED = "swapped"     # 被换出到 CPU
    FINISHED = "finished"   # 已完成
    PREEMPTED = "preempted" # 被抢占
```

### Scheduler 维护的队列
```python
self.waiting: List[Request]   # 新提交的请求
self.running: List[Request]   # 正在执行的请求
self.swapped: List[Request]   # 被换出的请求
```

## schedule() 主流程

```python
def schedule() -> SchedulerOutput:
    # 1. 处理 running 队列
    #    - 完成的请求移出
    #    - 生成了下一个 token 的继续保留
    #    - 内存不足时可能 swap out
    
    # 2. 尝试从 swapped 恢复
    #    - 优先恢复被换出的请求（避免饥饿）
    #    - 需要有足够的 KV Cache 空间
    
    # 3. 尝试调度 waiting 队列
    #    - FCFS（先来先服务）或其他策略
    #    - 需要有足够的 KV Cache 空间
    #    - Prefill 需要连续的 slots
    
    # 4. 返回 SchedulerOutput
    #    - scheduled_requests: 本轮要执行的请求
    #    - num_prefill_tokens: prefill 阶段的 token 数
    #    - num_decode_tokens: decode 阶段的 token 数
```

## 调度策略

### 内存检查
```python
# 每次调度前检查：
free_blocks = kv_cache_manager.get_num_free_blocks()

# Prefill 请求需要的 blocks
num_blocks_needed = ceil(num_tokens / block_size)

# Decode 请求可能需要新 block（当前 block 满了）
if current_block_full:
    num_blocks_needed += 1
```

### Prefill vs Decode 混合

**策略1: Prefill 优先**
- 优点：新请求延迟低
- 缺点：可能阻塞 decode

**策略2: Decode 优先**  
- 优点：吞吐更稳定
- 缺点：新请求等待时间长

**vLLM 策略: 动态平衡**
- 设置 max_num_seqs 限制并发数
- 优先保证 running 请求的 decode
- 在有余量时调度 waiting 的 prefill

### Swap 策略

**何时 Swap Out?**
- 内存不足，无法为所有 running 请求分配 KV Cache
- 选择最后加入的请求换出（LRU 策略）

**何时 Swap In?**
- 有足够空闲内存
- 优先恢复等待最久的 swapped 请求

[TODO: 用实际代码验证这些策略]
EOF
```

**任务2: 追踪请求生命周期**（1小时）
```bash
cat > /home/lvzheng/vllm/vllm/learning/week2-scheduler/trace_request.py << 'EOF'
"""
追踪一个请求从提交到完成的完整流程
"""
from vllm import LLM, SamplingParams
import logging
import os

# 设置详细日志
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 小并发，观察调度
llm = LLM(
    model="facebook/opt-125m",
    max_num_seqs=2,  # 只允许 2 个并发
    gpu_memory_utilization=0.3,
)

# 提交 4 个请求，观察调度行为
prompts = [
    "Request A: " + "hello " * 5,
    "Request B: " + "world " * 10,
    "Request C: " + "test " * 3,
    "Request D: " + "demo " * 8,
]

print("\n" + "="*60)
print("Submitting 4 requests with max_num_seqs=2")
print("Expected: A,B running first, C,D waiting")
print("="*60 + "\n")

outputs = llm.generate(
    prompts,
    SamplingParams(max_tokens=10, temperature=0)
)

print("\n" + "="*60)
print("All requests completed")
print("="*60)
for i, output in enumerate(outputs):
    print(f"\nRequest {chr(65+i)}: {output.outputs[0].text[:50]}...")
EOF

python /home/lvzheng/vllm/vllm/learning/week2-scheduler/trace_request.py 2>&1 | \
    tee /home/lvzheng/vllm/vllm/learning/week2-scheduler/trace.log

# 分析日志
echo -e "\n\nAnalyzing schedule decisions..."
grep -i "schedule\|running\|waiting" /home/lvzheng/vllm/vllm/learning/week2-scheduler/trace.log | \
    head -50 > /home/lvzheng/vllm/vllm/learning/week2-scheduler/schedule_decisions.log
```

### Day 4: Prefill vs Decode 混合调度 ✅

**理解两阶段的差异**（2小时）
```bash
cat > /home/lvzheng/vllm/vllm/learning/week2-scheduler/prefill_vs_decode.md << 'EOF'
# Prefill vs Decode 两阶段

## 核心差异

### Prefill（预填充阶段）
**目的**: 处理输入 prompt，生成第一个 token
**特点**:
- 计算量大：需要对整个 prompt 计算 attention
- 一次性分配所有需要的 KV Cache blocks
- 类似于 batch inference
- 输入: prompt tokens，输出: 第1个生成的 token

**示例**:
```
Input: "Hello world, how are"  (4 tokens)
Prefill 处理:
  - 计算所有 4 个 token 的 KV Cache
  - 生成第 5 个 token: " you"
  - 分配 blocks: ⌈4/16⌉ = 1 个 block
```

### Decode（解码阶段）
**目的**: 逐个生成后续 token
**特点**:
- 计算量小：每次只处理 1 个新 token
- 渐进式分配：当前 block 满了才分配新 block
- Autoregressive 生成
- 输入: 上一个 token，输出: 下一个 token

**示例**:
```
已生成: "Hello world, how are you"
Decode 处理:
  - 输入: "you" (1 token)
  - 读取之前的 KV Cache
  - 生成下一个 token: " doing"
  - 可能需要分配新 block (如果第 1 个 block 满了)
```

## 调度挑战

### 问题1: 计算量不对称
- 1 个 prefill (100 tokens) ≈ 100 个 decode (1 token each)
- 如果混合调度，prefill 可能阻塞 decode

### 问题2: 内存分配不同
- Prefill 需要一次性分配多个 blocks
- Decode 通常复用现有 blocks
- Prefill 可能导致内存碎片

### 问题3: 延迟要求不同
- Decode: 延迟敏感（用户正在等待下一个 token）
- Prefill: 相对不敏感（反正要处理整个 prompt）

## vLLM 的解决方案

### 策略1: 优先保证 Decode
```python
# 伪代码
if running_requests:
    schedule_decode(running_requests)  # 优先
    
if has_free_memory and waiting_requests:
    schedule_prefill(waiting_requests)  # 有余量再调度
```

### 策略2: Chunked Prefill (Week 5 详细学习)
- 长 prompt 分块处理
- 每块类似一个 decode iteration
- 避免 prefill 阻塞太久

### 策略3: 动态 Batch Size
- Prefill 和 Decode 可以在同一个 iteration
- 根据内存和计算资源动态调整

## 性能影响

### 指标1: Time to First Token (TTFT)
- Prefill 阶段的延迟
- 受调度策略影响：prefill 可能需要等待

### 指标2: Time Per Output Token (TPOT)  
- Decode 阶段的延迟
- 应该保持稳定，不受 prefill 影响

### 指标3: Throughput
- 总吞吐量 = prefill tokens/s + decode tokens/s
- 需要平衡两者的调度

[TODO: 实验测量 TTFT 和 TPOT]
EOF
```

**实验: 测量 Prefill 和 Decode 性能**（1小时）
```bash
cat > /home/lvzheng/vllm/vllm/learning/week2-scheduler/measure_latency.py << 'EOF'
"""
测量 TTFT 和 TPOT
"""
import time
import torch
from vllm import LLM, SamplingParams

model = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.3,
)

# 测试不同长度的 prompt
prompt_lengths = [10, 50, 100, 200]

for length in prompt_lengths:
    prompt = "hello " * length
    
    # Warmup
    _ = model.generate([prompt], SamplingParams(max_tokens=1))
    
    # 测量 TTFT (Prefill)
    torch.cuda.synchronize()
    start = time.time()
    
    # 只生成 1 个 token
    output = model.generate([prompt], SamplingParams(max_tokens=1))[0]
    
    torch.cuda.synchronize()
    ttft = time.time() - start
    
    # 测量 TPOT (Decode)
    torch.cuda.synchronize()
    start = time.time()
    
    # 生成 20 个 token
    output = model.generate([prompt], SamplingParams(max_tokens=20))[0]
    
    torch.cuda.synchronize()
    total_time = time.time() - start
    
    # TPOT = (total_time - TTFT) / (num_tokens - 1)
    tpot = (total_time - ttft) / 19
    
    print(f"\nPrompt length: {length} tokens")
    print(f"  TTFT (Prefill): {ttft*1000:.2f} ms")
    print(f"  TPOT (Decode):  {tpot*1000:.2f} ms")
    print(f"  TTFT/TPOT ratio: {ttft/tpot:.2f}x")

print("\n" + "="*60)
print("Observation: TTFT grows with prompt length")
print("            TPOT remains relatively constant")
print("="*60)
EOF

python /home/lvzheng/vllm/vllm/learning/week2-scheduler/measure_latency.py
```

### Day 5: KV Cache 与 Scheduler 的配合 ✅

**理解内存压力下的调度决策**（2小时）
```bash
cat > /home/lvzheng/vllm/vllm/learning/week2-scheduler/memory_pressure.py << 'EOF'
"""
模拟内存压力：提交超过内存容量的请求
观察 Scheduler 如何处理
"""
from vllm import LLM, SamplingParams
import logging
import os

os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

# 故意设置很小的内存
llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.2,  # 只用 20% GPU 内存
    max_num_seqs=8,              # 允许较多并发
)

# 提交很多请求
prompts = [f"Request {i}: " + "test " * 30 for i in range(10)]

print("="*60)
print("Submitting 10 long requests with limited memory")
print("Expected: some requests will wait or swap")
print("="*60)

outputs = llm.generate(
    prompts,
    SamplingParams(max_tokens=20)
)

print("\n" + "="*60)
print("All completed despite memory pressure")
print("="*60)
EOF

python /home/lvzheng/vllm/vllm/learning/week2-scheduler/memory_pressure.py 2>&1 | \
    tee /home/lvzheng/vllm/vllm/learning/week2-scheduler/memory_pressure.log

# 分析日志中的内存相关决策
grep -i "memory\|swap\|preempt" \
    /home/lvzheng/vllm/vllm/learning/week2-scheduler/memory_pressure.log
```

**深入理解 can_allocate() 检查**（1小时）
```bash
cd /home/lvzheng/vllm/vllm

# 阅读内存检查逻辑
# vllm/v1/core/kv_cache_manager.py: can_allocate() 或类似方法

cat > /home/lvzheng/vllm/vllm/learning/week2-scheduler/memory_check.md << 'EOF'
# Scheduler 的内存检查机制

## 调度前的检查

### 对于 Prefill 请求
```python
# 需要计算整个 prompt 需要多少 blocks
num_tokens = len(prompt_tokens)
num_blocks_needed = ceil(num_tokens / block_size)

# 检查是否有足够的空闲 blocks
if kv_cache_manager.get_num_free_blocks() >= num_blocks_needed:
    can_schedule = True
```

### 对于 Decode 请求
```python
# 大多数情况下复用现有 block
# 只有当前 block 满了才需要新 block
if current_block_is_full():
    num_blocks_needed = 1
else:
    num_blocks_needed = 0

if kv_cache_manager.get_num_free_blocks() >= num_blocks_needed:
    can_schedule = True
```

## Watermark 机制

**目的**: 预留一些 blocks，避免完全用尽

```python
# 实际可用 = 空闲 blocks - watermark
available_blocks = free_blocks - watermark_blocks

# watermark 通常是总 blocks 的 1-5%
# 避免内存碎片和调度抖动
```

## Out of Memory (OOM) 处理

### 策略1: Preemption (抢占)
- 暂停某些 running 请求
- 释放它们的 KV Cache
- 等后续有空间再恢复

### 策略2: Swap (换出)
- 将 KV Cache 换出到 CPU 内存
- 需要时再换入
- 比 preemption 开销大，但保留了进度

### 策略3: Rejection (拒绝)
- 直接拒绝新请求
- 返回错误给客户端
- 最后的手段

[TODO: 验证 vLLM v1 使用哪种策略]
EOF
```

### Day 6-7: Week 2 总结

**周总结文档**（2小时）

### Day 6-7: Week 2 总结

**周总结文档**（2小时）
```bash
cat > /home/lvzheng/vllm/vllm/learning/week2-scheduler/SUMMARY.md << 'EOF'
# Week 2 总结：Continuous Batching 与 Scheduler

## 核心收获

### 1. Continuous Batching 的本质
- **Iteration-level Scheduling**: 每生成一个 token 就重新调度
- **动态批处理**: 完成的请求立即移出，新请求立即加入
- **提升 GPU 利用率**: 避免等待最慢请求，减少空闲时间

### 2. Scheduler 的三个队列
```
Waiting → Running → Finished
    ↓        ↓
    └──> Swapped
```

- **Waiting**: 新提交的请求，等待分配 KV Cache
- **Running**: 正在执行的请求，每个 iteration 生成一个 token
- **Swapped**: 内存不足时被换出到 CPU 的请求
- **Finished**: 完成生成的请求（遇到 EOS 或 max_tokens）

### 3. 调度策略

**内存检查**:
```python
# Prefill: 需要一次性分配多个 blocks
num_blocks = ceil(num_prompt_tokens / block_size)

# Decode: 通常复用现有 block，偶尔需要新 block
num_blocks = 1 if current_block_full else 0

# 调度条件
if free_blocks >= num_blocks + watermark:
    schedule(request)
```

**优先级**:
1. 保证 Running 请求的 Decode（避免饥饿）
2. 恢复 Swapped 请求（FIFO）
3. 调度 Waiting 的 Prefill（FCFS）

### 4. Prefill vs Decode 的差异

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算量 | 大（处理整个 prompt） | 小（1 个 token） |
| 内存分配 | 一次性分配多个 blocks | 渐进式，复用已有 |
| 延迟特点 | 与 prompt 长度成正比 | 基本恒定 |
| 指标 | TTFT (Time to First Token) | TPOT (Time Per Output Token) |
| 调度优先级 | 较低 | 较高 |

**实验观察**:
- Prompt 100 tokens: TTFT ≈ 20-50ms
- TPOT ≈ 5-10ms (decode)
- TTFT/TPOT ratio ≈ 4-10x

### 5. 内存压力处理

**Watermark 机制**:
- 预留 1-5% 的 blocks
- 避免完全用尽导致的调度抖动

**OOM 策略**:
1. **Preemption**: 暂停请求，释放 KV Cache（v0 架构）
2. **Swap**: 换出到 CPU 内存（如果支持）
3. **Rejection**: 拒绝新请求

## 技术要点

### Scheduler 与 KVCacheManager 的配合
```
Scheduler.schedule():
  1. 检查 running 请求是否完成
  2. 对每个待调度的请求:
     kv_cache_manager.can_allocate(request)  # 内存检查
  3. 调度通过检查的请求
  4. 返回 SchedulerOutput

Engine.step():
  1. 调用 scheduler.schedule()
  2. 对调度的请求:
     kv_cache_manager.allocate_slots(request)  # 实际分配
  3. 执行 model forward
  4. 更新请求状态
```

### 请求生命周期
```
用户提交
  ↓
WAITING 队列
  ↓ (有空闲内存)
Prefill (生成第 1 个 token)
  ↓
RUNNING 队列
  ↓ (每个 iteration)
Decode (生成 1 个 token)
  ↓ (遇到 EOS 或 max_tokens)
FINISHED
  ↓
释放 KV Cache
```

### Chunked Prefill (预习 Week 5)
- 长 prompt 分成多个 chunk
- 每个 chunk 像一个 decode iteration
- 避免单次 prefill 阻塞太久
- 提高 prefill 和 decode 的混合效率

## 实践经验

### 观察到的行为
1. **max_num_seqs 的影响**:
   - 太小：GPU 利用率低，吞吐下降
   - 太大：内存压力大，可能 OOM
   - 需要根据模型大小和 prompt 长度调优

2. **内存压力下的调度**:
   - 新请求在 waiting 队列等待
   - 只有 running 请求完成释放内存后，新请求才能调度
   - Swapped 请求优先于 waiting 恢复

3. **Prefill/Decode 混合**:
   - Running decode 始终优先
   - Prefill 在有余量时调度
   - 长 prompt 的 TTFT 明显高于短 prompt

## 遗留问题

1. **Chunked Prefill 细节**（Week 5）:
   - 如何确定 chunk 大小？
   - 如何在 chunk 之间保存 KV Cache？
   - 对调度器的影响？

2. **分布式调度**（Week 4）:
   - 多个 GPU worker 如何协调调度？
   - Pipeline Parallelism 的调度策略？

3. **高级调度策略**（Week 5）:
   - Priority scheduling（优先级调度）
   - Preemptive scheduling（抢占式调度）
   - Fair scheduling（公平调度）

## 性能数据

[TODO: 填入实际测量的数据]
- OPT-125M
- max_num_seqs=4 vs 8 vs 16
- 不同 prompt 长度的 TTFT
- 内存压力下的吞吐量

## 下周计划

**Week 3: LLMEngine 完整执行流程**
- Day 1-2: LLMEngine.step() 完整追踪
- Day 3: ModelRunner 的作用
- Day 4-5: Prefill 和 Decode 两阶段的实现
- Day 6-7: 端到端性能分析

**重点问题**:
1. 从 LLM.generate() 到底层 model.forward() 的完整流程？
2. ModelRunner 如何准备输入数据？
3. 如何处理变长输入的 batching？
4. Output 如何收集和返回给用户？
EOF
```

**可选任务: 绘制状态转换图**
```bash
cat > /home/lvzheng/vllm/vllm/learning/week2-scheduler/state_diagram.txt << 'EOF'
# Request State Transition Diagram

User Submit
     |
     v
┌─────────┐
│ WAITING │ <──────────────────┐
└─────────┘                    │
     |                         │
     | (has free memory)       │
     v                         │
┌─────────┐  Prefill          │
│ RUNNING │ ─────────>  Generate 1st token
└─────────┘                   │
     |                        │
     | (each iteration)       │
     v                        │
  Decode ──> Generate next token
     |                        │
     | (OOM)                  │
     v                        │
┌─────────┐                   │
│ SWAPPED │ ──────────────────┘
└─────────┘  (memory available)
     
     | (EOS or max_tokens)
     v
┌──────────┐
│ FINISHED │
└──────────┘
     |
     v
  Free KV Cache
EOF

cat > /home/lvzheng/vllm/vllm/learning/week2-scheduler/scheduling_flow.txt << 'EOF'
# Scheduler.schedule() Flow

Every Iteration:
                                                    
┌────────────────────────────────────────┐
│  1. Process RUNNING queue              │
│     - Check if finished (EOS)          │
│     - Move finished to output          │
│     - Keep running for next decode     │
└────────────────────────────────────────┘
                  |
                  v
┌────────────────────────────────────────┐
│  2. Try to restore SWAPPED requests    │
│     - Check memory availability        │
│     - Swap in KV Cache from CPU        │
│     - Move to RUNNING                  │
└────────────────────────────────────────┘
                  |
                  v
┌────────────────────────────────────────┐
│  3. Try to schedule WAITING requests   │
│     - FCFS or priority order           │
│     - Check memory for prefill         │
│     - Allocate KV Cache blocks         │
│     - Move to RUNNING                  │
└────────────────────────────────────────┘
                  |
                  v
┌────────────────────────────────────────┐
│  4. Return SchedulerOutput             │
│     - scheduled_requests               │
│     - num_prefill_tokens               │
│     - num_decode_tokens                │
└────────────────────────────────────────┘
EOF
```

**学习状态标记**
```bash
cat >> /home/lvzheng/vllm/vllm/learning/week2-scheduler/SUMMARY.md << 'EOF'

## 学习状态
- ✅ Day 1: Continuous Batching 原理
- ✅ Day 2-3: Scheduler 深入剖析
- ✅ Day 4: Prefill vs Decode 混合调度
- ✅ Day 5: KV Cache 与 Scheduler 配合
- ✅ Day 6-7: Week 2 总结

**Week 2 完成时间**: $(date +%Y-%m-%d)

**总学习时间**: Week 1 (15-20h) + Week 2 (15-20h) = 30-40h
**进度**: 2/8 weeks (25% 完成)
EOF
```

**Week 2 学习状态**
- ✅ Day 1: Continuous Batching 原理
- ✅ Day 2-3: Scheduler 调度策略 + schedule() 代码
- ✅ Day 3-4: 请求生命周期 + update_from_output() 完整流程
- ✅ Day 5: KV Cache 与 Scheduler 配合
- ✅ Day 6-7: Week 2 总结

**Week 2 完成时间**: 2026-08-20

---

## 阶段2：推理引擎与优化（Week 3-6）

## Week 3: ModelRunner 执行流程 ✅

### 学习目标
1. 建立从 HTTP 请求到 token 输出的端到端全景认知
2. 理解 ModelRunner 的输入构造与 execute_model() 流程
3. 掌握 Attention metadata 构建与变长 batch 技巧
4. 理解 Sampling 层的实现（Top-K/P kernel、structured output）
5. 理解 CUDA Graph 的基本机制（capture/replay/warmup）

### Day 1: 端到端全景 ✅

建立完整的请求处理链路框架概念，不深入每个组件细节：

- **全链路概览**：
  - `vllm serve` 启动 → FastAPI 接收 HTTP → AsyncLLMEngine 异步队列
  - → Engine loop → Scheduler.schedule() → ModelRunner.execute_model()
  - → Model.forward() → Sampling → token 输出 → SSE 返回
- **各层职责定位**：
  - API Server：协议解析、并发管理、流式响应
  - AsyncLLMEngine：异步解耦、后台 engine loop
  - Scheduler：调度决策、内存检查
  - ModelRunner：输入构造、GPU 执行
- **代码路径**（快速浏览，建立全局图）：
  - `vllm/entrypoints/openai/api_server.py` — HTTP 入口
  - `vllm/v1/engine/async_llm.py` — 异步引擎
  - `vllm/v1/engine/core.py` — Engine core（engine loop）
  - `vllm/v1/core/sched/scheduler.py` — 调度
  - `vllm/v1/worker/gpu_model_runner.py` — GPU 执行

### Day 2-3: ModelRunner 执行流程 ✅

- **execute_model() 主流程**：
  - 接收 SchedulerOutput → 准备输入 → 模型 forward → 返回输出
  - `_update_states()` → `_prepare_inputs()` → `model.forward()` → `_generate_output()`
- **输入构造**：
  - input_ids tensor 拼接（prefill tokens + decode tokens）
  - positions tensor 构造（每个 token 在序列中的绝对位置）
  - 区分 prefill 和 decode 请求的不同处理
- **模型 forward**：
  - LlamaForCausalLM 等模型的 forward 接口
  - 逐层计算：Embedding → Layers × N → RMSNorm → LM Head
  - 每层内部：Attention + MLP
- **代码路径**：
  - `vllm/v1/worker/gpu_model_runner.py` — ModelRunner 核心
  - `vllm/model_executor/models/` — 各模型实现

### Day 4-5: Attention Metadata 构建 + Sampling 层 ✅

- **Attention Metadata**：
  - FlashInfer/FlashAttention 需要的元数据
  - block_tables tensor：每个请求的物理 block 列表
  - seq_lens：每个请求的当前序列长度
  - Prefill 和 Decode 分别构造不同的 metadata
  - `AttentionMetadataBuilder` 的工作流程
- **Sampling 层**：
  - Logits → Temperature → Top-K → Top-P → 采样
  - GPU kernel 实现：Top-K/Top-P 的高效筛选
  - Repetition Penalty、Frequency Penalty
  - Structured Output：grammar-guided decoding 基础
    - FSM 约束 → bitmask → logits masking
  - `vllm/v1/sample/` — Sampling 实现
  - `vllm/v1/structured_output/` — 结构化输出

### Day 6: CUDA Graph 机制（引入级） ✅

- **问题**：Decode 阶段 kernel 很小，launch overhead 成为瓶颈
- **CUDA Graph 原理**：
  - Capture：录制 kernel 调用序列
  - Replay：单次 launch 执行整个 graph
  - 效果：decode 延迟降低 30-50%
- **vLLM 实现要点**：
  - 按 batch size 预先 capture 不同大小的 graph
  - Warmup 阶段完成所有 capture
  - Decode 时选择匹配 graph replay
  - Prefill 不用 graph（shape 每次不同）
- **限制**：固定 shape、无动态分支、固定内存地址
- **代码路径**：
  - `vllm/v1/worker/gpu_model_runner.py` — capture/replay 逻辑

### Day 7: Attention 变体的系统适配 ✅

不同 attention 架构对推理系统的影响——从系统视角理解，而非 kernel 实现细节。

- **GQA / MQA**：
  - KV Cache 大小变化：KV head 数量 < Query head 数量
  - 对 TP 切分的影响：按 KV head group 切分，而非按 query head
  - block 利用率变化：每个 block 存储的有效数据量不同
  - 代码路径：模型定义中 `num_kv_heads` 如何传递到 attention backend

- **MLA（Multi-head Latent Attention，DeepSeek-V2/V3）**：
  - 核心思想：KV Cache 存压缩后的低秩表示，而非原始 KV
  - 对 KV Cache 管理的颠覆：每 token 存储量大幅缩小
  - 对 attention kernel 的要求：需要在线解压（up-projection）
  - 对系统吞吐的影响：cache 小 → 能装更多请求 → 吞吐提升
  - 代码路径：`vllm/v1/attention/backends/mla/` 目录结构

- **Sliding Window Attention（Mistral 等）**：
  - 对 block 管理的影响：窗口外的旧 block 可提前释放
  - Scheduler 如何感知窗口大小并优化内存分配
  - 与 Prefix Caching 的交互：窗口外的 block 无需保留

- **系统设计 tradeoff 总结**：
  - 不同 attention 变体如何影响：内存占用、最大并发数、TP 策略选择
  - 作为算子工程师接入新 attention 变体时需要适配的接口

### Day 8: 总结 + 完整 trace 串联 ✅

- 从 `generate()` 到 token 输出的完整数据流追踪
- 绘制架构全景图
- 标记每一层的性能开销占比
- 编写 Week 3 SUMMARY.md

**Week 3 学习状态**
- ✅ Day 1: 端到端全景
- ✅ Day 2-3: ModelRunner 执行流程
- ✅ Day 4-5: Attention Metadata 构建 + Sampling 层
- ✅ Day 6: CUDA Graph 机制
- ✅ Day 7: Attention 变体的系统适配（MLA + SWA）
- ✅ Day 8: 总结 + 完整 trace 串联

**Week 3 完成时间**: 2026-08-27

---

## Week 4: 分布式推理基础 ✅

### 学习目标
1. 掌握 Tensor Parallelism 完整链路（并行 Linear → 通信 → 权重加载）
2. 理解 Pipeline Parallelism 在 vLLM 推理中的实际实现
3. 深入 Data Parallelism 与 DP Coordinator 调度机制
4. 了解 Context Parallelism（PCP/DCP）原理与代码结构
5. 理解 parallel_state.py 中的 group 管理与 Executor 多 worker 架构

### Day 1: TP 原理 + 并行 Linear 层实现 ✅

- **TP 原理**：
  - Column Parallel：按列切分权重，每卡计算部分输出，无需通信
  - Row Parallel：按行切分权重，forward 后需 AllReduce 聚合
  - Attention：按 head 切分（GQA 时按 KV head group 切分）
  - MLP：Gate/Up 用 Column Parallel，Down 用 Row Parallel
- **代码精读**：
  - `vllm/model_executor/layers/linear.py` — ColumnParallelLinear / RowParallelLinear / MergedColumnParallelLinear
  - `forward()` 中何时调用 AllReduce，`reduce_results` 参数含义
  - bias 的处理方式
  - `vllm/model_executor/layers/vocab_parallel_embedding.py` — Embedding 分片
- **TP 对 KV Cache 的影响**：
  - GQA：每卡只存 `num_kv_heads / tp_size` 个 head 的 KV
  - MLA：replicate 策略（Week 3 已了解）
- **Sequence Parallelism（简要）**：
  - 在非 tensor-parallel 区域（LayerNorm、Dropout）也做序列维度切分
  - 配合 TP 的 AllGather / ReduceScatter 替代 AllReduce

### Day 2: 权重加载机制 + Group 管理 + 通信封装 ✅

- **权重加载（weight_loader）**：
  - `vllm/model_executor/model_loader/` — 加载流程
  - `weight_loader()` 方法：根据 tp_rank 从 safetensors 切分权重
  - `MergedColumnParallelLinear`：QKV/Gate-Up 合并后按 head 切分
  - `RowParallelLinear`：按行切分加载
  - Lazy loading：按需加载 tensor，减少峰值内存
- **Group 管理**：
  - `vllm/distributed/parallel_state.py` — `initialize_model_parallel()`
  - TP/PP/DP/EP group 划分规则（哪些 rank 在同一 group）
  - `GroupCoordinator` 类：封装 rank 列表、通信方法
  - 8 卡场景举例：TP=4,PP=2 时的 group 分布
- **通信封装**：
  - `vllm/distributed/communication_op.py` — tensor_model_parallel_all_reduce 等高层接口
  - `vllm/distributed/device_communicators/pynccl.py` — NCCL Python 封装
  - `vllm/distributed/device_communicators/custom_all_reduce.py` — NVLink 优化的 AllReduce
  - NCCL 通信原语：AllReduce、AllGather、ReduceScatter、P2P Send/Recv

### Day 3: Pipeline Parallelism（PP） ✅

- **关键认知**：vLLM 推理中 PP 是逐 stage 串行执行，不是训练中的 micro-batch pipelining
- **代码精读**：
  - PP stage boundary 确定（模型层 → stage 映射）
  - 跨 stage 的 hidden state 传输：P2P Send/Recv
  - `get_pp_group()` 使用场景
  - ModelRunner 中 PP 逻辑：每个 worker 只执行本 stage 的层
- **PP + TP 组合**：
  - 8 GPU = TP=4 × PP=2 的 rank 分布示例
  - TP rank 间用 NVLink AllReduce，PP stage 间用 P2P
  - 何时选 PP：模型太大、TP 通信开销过高时用 PP 补充
- **延迟 vs 吞吐 tradeoff**：
  - PP 增加延迟（stage 串行）但可部署更大模型
  - 推理场景下 PP 的实际使用频率低于 TP

### Day 4: Data Parallelism（DP）+ DP Coordinator ✅

- **DP 原理**：多个完整模型副本（或 TP group），独立推理不同请求
- **vLLM DP 实现（重点）**：
  - `vllm/v1/engine/coordinator.py` — DPCoordinator / DPCoordinatorProc
  - 请求分发策略：如何负载均衡到不同 DP rank
  - DP rank 间的同步机制：何时 barrier、何时独立
  - `data_parallel_size` / `data_parallel_size_local` 配置
  - `data_parallel_backend`: "mp" vs Ray
- **DP Attention 概念**：
  - 传统 DP：每个副本独立 KV Cache，互不通信
  - DP Attention：序列分布到不同 DP rank，attention 时跨 rank 通信
  - 适用场景与 tradeoff（高 QPS vs 长序列）
- **与 Week 6 EP 的关联**：
  - MoE + DP 场景下 expert 分布与请求路由
  - EP 本质上是 expert 维度的 DP

### Day 5: Context Parallelism（CP） ✅

- **CP 核心原理**：
  - 解决的问题：超长序列（128K+）KV Cache 单卡放不下
  - 与 TP 的本质区别：TP 切 head 维度减权重显存，CP 切 sequence 维度减 KV Cache 显存
  - Ring Attention 算法：环形传递 KV + Flash Attention online softmax 合并
  - 双缓冲机制：buffer A/B 交替收发，通信与计算 overlap
  - 显存分析：每卡 3/CP 的 KV 显存（持久 1/CP + 临时 2/CP）
- **PCP（Prefill Context Parallel）详解**：
  - 长 prompt token 均分到多卡并行 prefill
  - Ring Attention 具体执行流程：partial attention + log-sum-exp rescale
  - 通信模式：P2P Send/Recv 沿环传递 KV 块
  - 与 Flash Attention 的配合：chunk-level ring
  - `prefill_context_parallel_size` 配置
  - `get_pcp_group()` 使用
  - 代码路径：PCP 相关的 attention backend 实现
- **DCP（Decode Context Parallel）详解**：
  - 与 PCP 的区别：decode 每步只有 1 个 Q token，计算量小，通信模式不同
  - KV 分配策略：round-robin 分散存储 vs 静态按位置分配
  - 通信模式：all-gather / P2P 获取远端 KV（非 Ring）
  - `decode_context_parallel_size` 配置
  - `vllm/v1/worker/cp_utils.py` — DCP 工具函数
  - 代码路径：DCP metadata 准备、KV Cache 分散写入逻辑
- **CP + TP 组合部署**：
  - rank 分布与 group 嵌套：TP group 内切 head，CP group 内切 sequence
  - 例：8 卡 TP=4, CP=2 时的 group 划分
  - 通信开销分析：TP AllReduce + CP Ring/AllGather 叠加
- **CP 部署决策**：
  - 何时开启 CP：序列长度 vs 单卡 KV Cache 容量的判断
  - CP size 的选择：通信开销与显存收益的 tradeoff
  - PCP only / DCP only / PCP+DCP 组合的适用场景
  - 与 PP 的对比：都能解决显存问题，选择依据

### Day 6: Executor 架构 + 多 Worker 启动流程 ✅

- **Executor 接口**：
  - `vllm/v1/executor/abstract.py` — Executor 基类定义
  - `execute_model()` / `start()` / `stop()` 接口
- **MultiprocExecutor**：
  - `vllm/v1/executor/multiproc_executor.py` — 多进程实现
  - `WorkerProc` 启动流程：spawn → init_device → init_model → ready
  - Worker 间通信机制（SharedMemory / socket）
- **RayExecutor**：
  - `vllm/v1/executor/ray_executor.py` — Ray 分布式实现
  - 跨节点 worker 管理
- **Worker 初始化**：
  - `vllm/v1/worker/gpu_worker.py` — init_device / load_model
  - 多 worker NCCL group 建立时机
  - TP 内同步 vs PP 跨 stage 同步在 Executor 中的体现

### Day 7: 实践验证 + 总结 ✅

- **实践（8 卡 H800）**：
  - TP=2 vs TP=4 vs TP=8 推理对比（throughput、latency、通信占比）
  - TP=4, PP=2 组合验证：观察 stage boundary 和 P2P 通信
  - NCCL_DEBUG=INFO 抓取通信日志分析
  - nvidia-smi / nsys 观察多卡利用率
- **总结**：
  - 各并行策略适用场景决策树（模型大小 × 卡数 × 延迟要求）
  - 编写 Week 4 SUMMARY.md

**Week 4 学习状态**
- ✅ Day 1: TP 原理 + 并行 Linear 层实现
- ✅ Day 2: 权重加载机制 + Group 管理 + 通信封装
- ✅ Day 3: Pipeline Parallelism（PP）
- ✅ Day 4: Data Parallelism（DP）+ DP Coordinator
- ✅ Day 5: Context Parallelism（CP）
- ✅ Day 6: Executor 架构 + 多 Worker 启动流程
- ✅ Day 7: 实践验证 + 总结

**Day 7 实验进度**
- ✅ SUMMARY.md 已编写完成（learning/week4-distributed/SUMMARY.md）
- ✅ 实验脚本已编写（learning/week4-distributed/exp1_tp_compare.sh, exp2_tp_pp_compare.sh, exp3_nccl_debug.sh）
- ✅ 模型路径更新（Qwen2.5-7B，新路径 /data/model/Qwen2.5-7B）
- ✅ 实验1: TP=1/2/4 推理对比（笔记：learning/week4-distributed/exp1_tp_analysis.md）
- ✅ 实验2: TP=4,PP=1 vs TP=2,PP=2 对比（笔记：learning/week4-distributed/exp2_tp_pp_analysis.md）
- ✅ 实验3: NCCL_DEBUG=INFO 通信日志分析（笔记：learning/week4-distributed/exp3_nccl_analysis.md）

**容器迁移说明（2026-09-01 更新）**
- ✅ /dev/shm 已扩展至 756GB，不再受限
- ✅ 模型路径已更新：/data/vllm/model/ → /data/model/
- ⚠️ 注意：在 /home/lvzheng/vllm/ 下不能直接 import vllm（源码目录冲突），需 cd /tmp 后执行
- 需要迁移到 vllm/vllm-openai:v0.27.1 官方镜像，加 --ipc=host 解决
- 迁移后需要：
  1. 恢复 learning/ 目录和 LearningPlan.md
  2. clone vllm 源码用于代码阅读（pip 版已预装，无需编译）
  3. 模型路径需确认（取决于 -v 挂载映射）
  4. 实验脚本中的 MODEL 路径可能需要更新

---

## Week 5: 高级调度与优化 ✅

### 学习目标
1. 掌握 Chunked Prefill 原理与实现
2. 理解 Prefix Caching（APC）机制
3. 掌握 Speculative Decoding 原理与 vLLM 实现
4. 深入 CUDA Graph 的调优实践

### Day 1-2: Chunked Prefill ✅

- **问题**：长 prompt prefill 阻塞 decode，影响在线延迟
- **方案**：将 prefill 拆成多个 chunk，与 decode 交替执行
- **vLLM 实现**：
  - `max_num_batched_tokens` 约束 chunk 大小
  - Scheduler 如何决定 chunk 划分
  - Partial prefill 的 KV cache 管理
  - `is_prefill_chunk` 标记
- **代码路径**：
  - `vllm/v1/core/sched/scheduler.py` — `_schedule_prefills()` chunking 逻辑
- **实践**：对比开启/关闭 chunked prefill 对 TTFT 和 TPOT 的影响

### Day 3: Prefix Caching（APC）✅

- **原理**：相同 prefix 的请求共享 KV cache blocks（hash 匹配）
- **vLLM 实现**：
  - Block hash 计算（token content → hash）
  - 命中时直接复用 blocks，跳过 prefill
  - LRU 淘汰：空闲 block 放 free_queue 尾部保留
  - 与 CoW 的关系
- **代码路径**：
  - `vllm/v1/core/block_pool.py` — `_get_cached_block()`
  - `request.update_block_hashes()`
- **适用场景**：system prompt 共享、few-shot examples、RAG context

### Day 4-5: Speculative Decoding ✅

- **原理**：小模型（drafter）猜测多个 token → 大模型（verifier）并行验证
- **收益**：一步验证 K 个 token，减少大模型 forward 次数
- **vLLM 实现**：
  - Drafter 模型集成（Eagle、Medusa、Draft model、N-gram）
  - `spec_token_ids`：Scheduler 中的投机 token 管理
  - 验证失败时的回退：`num_rejected` token 丢弃
  - 对 KV Cache 管理的影响
- **代码路径**：
  - `vllm/v1/spec_decode/` — 投机解码实现
  - `scheduler.update_draft_token_ids()`
  - `update_from_output()` 中 accepted/rejected 处理
- **实践**：对比不同 drafter 模型的加速比

### Day 6: CUDA Graph 深入 ✅

- **与 Week 3 引入级的区别**：深入调优和交互分析
- **Warmup 优化**：
  - Capture batch size 范围与 warmup 时间的权衡
  - `max_num_seqs` 影响需要 capture 的 graph 数量
- **Dynamic shape 交互**：
  - Padding 策略与内存开销
  - `enforce_eager=True` 的使用场景
- **与 Speculative Decoding 的交互**：
  - 投机 token 改变有效 batch shape
  - 需要额外的 graph capture
- **代码路径**：
  - `vllm/v1/worker/gpu_model_runner.py` — CUDAGraphRunner

### Day 7: 综合实验 ✅

- 搭建 benchmark 环境
- 逐一开启优化，量化每个优化的收益
- 组合优化时的交互影响分析
- 编写 Week 5 SUMMARY.md

**实验结果汇总**：

| 优化项 | 关键指标 | 数据 |
|--------|---------|------|
| Prefix Caching ON vs OFF | Mean TTFT | 77.74 vs 101.66 ms（↓23.5%） |
| Prefix Caching ON vs OFF | P99 TTFT | 115.83 vs 174.49 ms（↓33.6%） |
| Chunked Prefill 1024 vs 无 | P99 ITL | 22.32 vs 35.90 ms（↓37.8%） |
| Chunked Prefill 1024 vs 无 | Mean TTFT | 94.88 vs 88.01 ms（↑7.8%，符合预期） |
| CUDA Graph ON vs OFF | TPOT | 130.01 vs 203.86 ms（↓36.2%） |
| 组合优化（exp5） | 吞吐量 | baseline 115.89 → all_opt 167.74 tok/s（↑44.7%） |

---

## Week 6: MoE 模型推理优化 ✅

### 学习目标
1. 理解 MoE 架构原理与推理特性
2. 掌握 vLLM 中 FusedMoE kernel 实现
3. 理解 Expert Parallelism 基础与 All-to-All 通信
4. 掌握 MoE 模型的部署调优实践

### Day 0: 环境与模型准备

**模型下载**（提前完成，避免学习时等待）
```bash
cd /home/lvzheng/vllm

# 检查现有模型
ls -lh /data/model/ | grep -i "mixtral\|qwen.*moe"

# 如果没有 MoE 模型，下载小型 MoE 用于实验
# 选项 1：Qwen2-57B-A14B-Instruct（推荐，约 120GB，64 experts，top-8）
huggingface-cli download Qwen/Qwen2-57B-A14B-Instruct \
  --local-dir /data/model/Qwen2-57B-A14B-Instruct \
  --local-dir-use-symlinks False

# 选项 2：使用已有的 dense 模型作为对照
# Qwen2.5-7B（dense）vs Qwen2-MoE（sparse）
```

**实验目录初始化**
```bash
mkdir -p /home/lvzheng/vllm/learning/week6-moe-optimization
cd /home/lvzheng/vllm/learning/week6-moe-optimization

# 创建结果目录
mkdir -p results logs

# 创建 README
cat > README.md << 'EOF'
# Week 6: MoE 模型推理优化

## 实验列表
- exp1: MoE vs Dense 基础性能对比
- exp2: FusedMoE kernel profiling
- exp3: Expert 路由分析
- exp4: EP vs TP 性能对比（多卡环境）
- exp5: MoE 部署调优综合实验

## 模型
- Dense: /data/model/Qwen2.5-7B
- MoE: /data/model/Qwen2-57B-A14B-Instruct（64 experts, top-8）
EOF
```

### Day 1: MoE 架构原理 ✅

**理论学习**（2小时）

1. **MoE 基础概念**
   - Router（Gate）：为每个 token 选择 Top-K expert
   - Expert：通常是 FFN（Feed-Forward Network）
   - Load Balancing：auxiliary loss 防止部分 expert 过载
   - Capacity Factor：限制每个 expert 处理的最大 token 数
   
2. **MoE vs Dense 的权衡**
   ```
   Dense FFN: All tokens → Same FFN → Compute = N * D
   MoE:       Each token → Top-K of E experts → Compute ≈ N * D * K/E
   
   参数量：MoE = Dense * E（所有 expert 都需加载）
   计算量：MoE ≈ Dense * K/E（仅激活 K 个）
   
   示例：Mixtral 8x7B
   - 8 experts，每次选 top-2
   - 参数量 ≈ 47B（8 * 7B，expert 共享 attention）
   - 激活参数 ≈ 13B（2/8 的 expert 参与）
   ```

3. **推理的独特挑战**
   - 内存墙：所有 expert 权重必须驻留 GPU
   - 计算不均：token 分布不均 → 某些 expert 成为瓶颈
   - 批处理困难：batch 内 token 路由分散 → GPU 利用率低

**代码阅读**（1小时）
```bash
cd /home/lvzheng/vllm

# 1. 找到 MoE 模型定义（以 Qwen2MoE 为例）
find vllm/model_executor/models -name "*moe*.py"
# 预期输出：qwen2_moe.py, mixtral.py, deepseek_v2.py 等

# 2. 阅读 Qwen2MoE 的 forward 流程
grep -n "class Qwen2MoeDecoderLayer" vllm/model_executor/models/qwen2_moe.py
cat vllm/model_executor/models/qwen2_moe.py | sed -n '/class Qwen2MoeDecoderLayer/,/^class /p' | head -60
# 重点：
# - self.mlp = Qwen2MoeSparseMoeBlock(...) 
# - hidden_states = self.mlp(hidden_states) 的调用

# 3. 阅读 MoE Block 实现
grep -n "class Qwen2MoeSparseMoeBlock" vllm/model_executor/models/qwen2_moe.py
cat vllm/model_executor/models/qwen2_moe.py | sed -n '/class Qwen2MoeSparseMoeBlock/,/^class /p' | head -100
# 重点：
# - gate_up_proj: 计算 router logits
# - fused_moe(): 核心 MoE kernel 调用
# - topk: 选择多少个 expert
# - num_experts: 总 expert 数

# 4. 记录关键参数位置
cat > learning/week6-moe-optimization/day1_code_notes.md << 'EOF'
# Day 1: MoE 代码结构笔记

## Qwen2MoE 模型架构
- 文件：`vllm/model_executor/models/qwen2_moe.py`
- 关键类：`Qwen2MoeSparseMoeBlock`（行号：___）
- Router 实现：`self.gate` Linear 层计算 logits
- Expert 实现：`self.experts` ModuleList，每个是独立 FFN

## Router 逻辑
```python
# 伪代码流程（从代码中提取）
router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]
topk_weights, topk_ids = torch.topk(router_logits, k=self.top_k)  # 选 top-k
topk_weights = softmax(topk_weights)  # 归一化权重

# 调用 fused_moe kernel
output = fused_moe(
    hidden_states,
    w1=expert_w1,  # 所有 expert 的 gate_proj 权重
    w2=expert_w2,  # 所有 expert 的 down_proj 权重
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    ...
)
```

## 问题清单（Day 3-4 回答）
1. fused_moe() 如何避免为每个 expert 单独 kernel launch？
2. Token permutation 的目的是什么？
3. Expert Parallelism 时，all-to-all 通信发生在哪一步？
EOF
```

### Day 2: MoE 推理挑战分析 ✅

**实验1：MoE vs Dense 性能对比**（2小时）

创建基准测试脚本（参考完整脚本在前文已生成）：
```bash
cd /home/lvzheng/vllm/learning/week6-moe-optimization
# 使用前文生成的 exp1_moe_vs_dense.sh
chmod +x exp1_moe_vs_dense.sh
./exp1_moe_vs_dense.sh
```

**预期结果**：
- MoE 内存占用：约 dense 的 E/2 倍（假设 top_k = E/4）
- MoE TTFT：接近激活参数相当的 dense 模型
- MoE 吞吐量：可能低于 dense（负载不均 + permutation 开销）

**分析任务**（1小时）
填写 `learning/week6-moe-optimization/day2_analysis.md`（模板已在前文生成）

### Day 3: FusedMoE Kernel 实现（上）✅

**代码深入**（2.5小时）

```bash
cd /home/lvzheng/vllm

# 1. 定位 FusedMoE kernel 入口
grep -n "^def fused_moe" vllm/model_executor/layers/fused_moe/fused_moe.py | head -1
# 记录行号：___

# 2. 阅读 fused_moe() 函数签名和文档
# 重点参数：
# - hidden_states: [num_tokens, hidden_size]
# - w1/w2: expert 权重 [num_experts, ...]
# - topk_weights: [num_tokens, top_k] router 输出权重
# - topk_ids: [num_tokens, top_k] 选中的 expert ID

# 3. 追踪 Token Permutation 逻辑
grep -n "argsort\|sorted_token" vllm/model_executor/layers/fused_moe/fused_moe.py | head -20
# 理解如何按 expert 重排 token

# 4. 找到 Triton kernel 调用
grep -n "invoke_fused_moe_kernel\|@triton.jit" vllm/model_executor/layers/fused_moe/fused_moe.py | head -10

# 5. 绘制数据流图（填写模板）
# 模板已在前文生成：day3_kernel_dataflow.md
```

**关键理解点**：
- **Permutation 目的**：将相同 expert 的 token 聚集 → 批量计算
- **Fused Kernel**：所有 expert 的 GEMM 融合为一个 kernel launch
- **性能瓶颈**：Expert 负载不均 + sort 开销

### Day 4: FusedMoE Kernel 实现（下）+ Profiling ✅

**Profiling 实验**（2小时）

```bash
cd /home/lvzheng/vllm/learning/week6-moe-optimization
# 使用前文生成的 exp2_profile_moe.py
.venv/bin/python exp2_profile_moe.py > results/exp2_profile.txt 2>&1
cat results/exp2_profile.txt
```

**预期观察**：
- 小 batch (< 32): 低吞吐，GPU 未饱和
- 大 batch (> 256): 吞吐趋于平稳
- 对比 Dense FFN 的理论 FLOP

**分析任务**（1小时）
填写 `day4_profiling_analysis.md`（模板已生成）

### Day 5: Expert Routing 分析与 EP 基础 ✅

**理论学习：Expert Parallelism**（2小时）

阅读前文生成的 `day5_ep_theory.md`，重点理解：

1. **EP vs TP 的本质区别**
   - TP: 切分权重，所有 GPU 处理所有 token → All-Reduce
   - EP: 切分 expert，token 按路由分发 → All-to-All

2. **All-to-All 通信流程**
   ```
   Step 1: Dispatch（分发 token 到对应 GPU）
   Step 2: Local Expert Compute（各 GPU 计算其 expert）
   Step 3: Combine（收回结果到原 GPU）
   ```

3. **DP Attention 含义**
   - Attention 层：Data Parallel（无通信）
   - MoE 层：Expert Parallel（All-to-All 通信）

**代码阅读**（1小时）
```bash
cd /home/lvzheng/vllm

# EP 的 process group 初始化
grep -n "expert.*parallel\|ep_group" vllm/distributed/parallel_state.py | head -20

# FusedMoE 中的 EP 支持
grep -n "all.*to.*all\|expert.*parallel" vllm/model_executor/layers/fused_moe/fused_moe.py | head -20

# 检查 EP 参数是否可用
.venv/bin/python -m vllm.entrypoints.openai.api_server --help | grep -i "expert"
```

### Day 6: EP 多卡实验（需 2+ GPU） ✅

**实验4：EP vs TP 性能对比**（3小时，可选）

```bash
cd /home/lvzheng/vllm/learning/week6-moe-optimization
# 使用前文生成的 exp4_ep_vs_tp.sh
./exp4_ep_vs_tp.sh
```

**注意**：
- 需要 2+ GPU
- 如果 vLLM 版本不支持 EP，跳过实验，专注理论学习
- Week 8 会深入 EP 的生产实现（EPLB）

### Day 7: MoE 部署调优与总结 ✅

**综合实验**（2小时）

```bash
cd /home/lvzheng/vllm/learning/week6-moe-optimization
# 使用前文生成的 exp5_moe_tuning.sh
./exp5_moe_tuning.sh
```

测试配置：
1. Baseline: `gpu_memory_utilization=0.85, max_num_seqs=32`
2. High Memory: `gpu_memory_utilization=0.90, max_num_seqs=64`
3. + Chunked Prefill: `--enable-chunked-prefill`

**Week 6 总结**（1小时）

填写 `WEEK6_SUMMARY.md`（模板已生成），包括：
- 核心知识点总结
- 实验结果汇总（5 个实验的表格）
- 遇到的问题与解决
- 学习检查点（self-check）

完成后更新 LearningPlan.md 的 Week 6 状态为 ✅

---

## 阶段3：生产级系统能力（Week 7-10）

## Week 7: 量化与 LoRA ✅

### 学习目标
1. 理解量化技术原理（INT8/INT4/FP8/MXFP4）及其对推理的影响
2. 掌握 vLLM 量化注册体系、离线/在线量化方案的实现与 FP8 KV Cache
3. 掌握 LoRA adapter 的加载、Scheduler 约束与多租户推理
4. 理解量化与分布式（TP）、MoE、LoRA 的交叉组合
5. 掌握量化精度评估方法论

### Day 1-2: 量化技术原理 ✅

- **量化基础**：
  - 对称量化 vs 非对称量化
  - Per-tensor / Per-channel / Per-group / Per-block 量化粒度
  - 校准（Calibration）与量化误差评估
  - 哪些层对量化敏感（attention QKV 投影 vs MLP），敏感度直觉
- **主流方案对比**：
  - **FP8（E4M3/E5M2）**：精度损失最小，H100/H800 原生支持
  - **INT8（W8A8）**：权重和激活都量化，SmoothQuant
  - **INT4（W4A16）**：仅权重量化，GPTQ、AWQ
  - **MXFP4 / NVFP4**：Microscaling FP4，block-level scaling，更激进的量化；与 INT4 的区别（block scaling vs per-group scaling）
  - **GGUF**：llama.cpp 格式，混合精度量化
- **离线量化 vs 在线量化（Online Quantization）**：
  - 离线：预量化模型（GPTQ、AWQ 等），需要提前校准和导出
  - 在线：运行时动态量化（`--quantization fp8_per_tensor`），直接对 FP16 模型使用
  - 决策依据：精度要求、部署便捷性、是否有预量化模型
- **对推理的影响**：
  - 内存减少 → 可服务更多并发 / 更大模型
  - 计算吞吐提升（INT8/FP8 Tensor Core）
  - 精度评估方法：perplexity 对比、task accuracy 抽样、输出一致性 diff

### Day 3-4: vLLM 量化实现 + FP8 KV Cache ✅

- **量化注册与加载链路**（从配置到生效的完整路径）：
  - `QuantizationMethods` 类型枚举：当前 20+ 种方案的注册机制
  - `get_quantization_config(quant_method)` → 返回对应 Config 类
  - `QuantizationConfig.get_quant_method(layer)` → 决定每层的量化方式
  - 模型加载时如何将原始 Linear 替换为量化版本
  - `--quantization` 参数从 CLI → VllmConfig → 模型加载的传递链路
  - 代码：`vllm/model_executor/layers/quantization/__init__.py`
- **离线量化层实现**：
  - Marlin kernel（INT4 优化 GEMM）：`gptq_marlin`、`awq_marlin`
  - FP8 kernel 与 scaling factor 管理：`vllm/model_executor/layers/quantization/fp8.py`
  - 量化权重的特殊加载路径（GPTQ、AWQ、INT8）
  - `compressed-tensors`：通用量化格式支持
- **在线量化实现**：
  - `OnlineQuantizationConfig`：运行时动态量化的统一入口
  - 支持的 shorthand：`fp8_per_tensor`、`fp8_per_block`、`int8_per_channel_weight_only`、`nvfp4_per_token`、`mxfp8`
  - 与离线量化的代码路径差异：无需预量化权重，forward 时实时量化
  - 代码：`vllm/model_executor/layers/quantization/online/`
- **KV Cache 量化**：
  - FP8 KV Cache：容量翻倍，精度影响小
  - `cache_dtype` 参数配置
  - `BaseKVCacheMethod`：k_scale / v_scale 的加载与管理
  - 量化/反量化在 attention kernel 中的位置
  - 代码：`vllm/model_executor/layers/quantization/kv_cache.py`
- **量化与分布式的交互**：
  - 量化权重在 TP 下的分片：ColumnParallelLinear 的量化版本如何切分 scale factor
  - `experts_int8`：专门针对 MoE expert 的 INT8 量化方案
  - FP8 KV Cache + CP 的组合（为 Week 8 Day 5-6 铺垫）
- **代码路径**：
  - `vllm/model_executor/layers/quantization/` — 各方案实现
  - `vllm/model_executor/layers/quantization/__init__.py` — 注册与分发
  - `vllm/model_executor/layers/quantization/online/` — 在线量化
  - `vllm/model_executor/layers/quantization/kv_cache.py` — KV Cache 量化
- **实践**：
  - 同一模型 FP16 vs FP8（online）vs INT8 vs INT4（GPTQ）性能对比
  - FP8 KV Cache 容量与精度评估
  - Online vs Offline 量化对比：`--quantization fp8_per_tensor` vs 预量化 FP8 模型

### Day 5-6: LoRA Adapter ✅

- **LoRA 原理**：
  - 低秩分解：W + ΔW = W + B×A（r << d）
  - 推理时的额外计算：x@W + x@A@B×scaling
  - 多 adapter 场景：不同请求用不同 LoRA
- **vLLM 中的 LoRA 实现**：
  - `LoRAManager`（`vllm/lora/model_manager.py`）：adapter 加载与管理
  - `PunicaKernel`（SGMV）：多 LoRA batch 并行计算
  - 每个请求绑定 `lora_request`（adapter_id）
  - LoRA 权重的 GPU 内存管理（LRU swap）
  - 代码：`vllm/lora/punica_wrapper/` — Punica GPU/CPU 实现
- **Scheduler 层的 LoRA 约束**：
  - `max_loras` 配置：限制同时活跃的 adapter 数量
  - Scheduler `_schedule_prefills()` 中的 LoRA 检查：`scheduled_loras` 集合 + 超限拒绝
  - LoRA 约束与 KV Cache 约束的双重调度门控
  - 代码：`vllm/v1/core/sched/scheduler.py` — lora 相关逻辑（L674-L681, L724-L728）
- **Worker 侧 LoRA 生命周期**：
  - `worker_manager.py`：跨 Worker 的 adapter 分发与同步
  - `gpu_worker.py`：`add_lora()` / `remove_lora()` / `pin_lora()` 接口
  - ModelRunner 中的 LoRA 层替换与 warmup
- **性能考虑**：
  - LoRA rank 对延迟的影响
  - 多 adapter 并发 vs 单 adapter 的吞吐差异
  - LoRA + 量化（QLoRA 推理）：量化基座模型 + LoRA adapter 的组合
- **代码路径**：
  - `vllm/lora/` — LoRA 管理框架
  - `vllm/lora/layers/` — LoRA 层 forward
  - `vllm/lora/worker_manager.py` — Worker 侧生命周期管理

### Day 7: 综合实践 ✅

- **Day 3-4 量化实践（合并至此）**：
  - 同一模型 FP16 vs FP8（online）vs INT8 vs INT4（GPTQ）性能对比
  - FP8 KV Cache 容量与精度评估
  - Online vs Offline 量化对比：`--quantization fp8_per_tensor` vs 预量化 FP8 模型
- **量化部署全流程**：HF 模型 → 离线量化 / 在线量化 → 推理服务
- **多 LoRA adapter 并发服务搭建**：多租户场景验证 `max_loras` 调度行为
- **量化 + TP 组合验证**：TP=2/4 下的量化模型推理，观察 scale factor 分片
- **MoE + 量化组合部署**（如 `experts_int8` 或 MoE 模型 + FP8）
- **精度评估**：量化前后 perplexity 对比 + 输出 diff 抽样分析
- **性能 benchmark**：汇总所有配置的 TTFT / TPOT / Throughput
- 编写 Week 7 SUMMARY.md

**Week 7 学习状态**
- ✅ Day 1-2: 量化技术原理
- ✅ Day 3-4: vLLM 量化实现 + FP8 KV Cache
- ✅ Day 5-6: LoRA Adapter
- ✅ Day 7: 综合实践

**Week 7 完成时间**: 2026-09-24

**实验成果**：
- 量化对比实验：6 种配置性能数据（learning/week7-quantization/exp_quant_analysis.md）
- LoRA 并发实验：max-loras 调度行为验证（learning/week7-quantization/exp_lora_analysis.md）
- 关键发现：FP8 per-tensor 最优吞吐 +42%，FP8 KV Cache 高负载 +18%，max-loras=1 导致部分 adapter TTFT 劣化 5.3 倍

---

## Week 8: 生产级分布式推理 ✅

### 学习目标
1. 掌握 PD 分离（Disaggregated Prefill/Decode）架构的完整实现
2. 理解 KV Connector 的多种实现及选型依据，掌握 Prefill/Decode 两端 Scheduler 的行为差异
3. 掌握 Expert Parallelism 深入与负载均衡（EPLB）
4. 掌握生产级投机解码：DFlash 并行 draft 机制与 DSpark 半自回归框架
5. 掌握 KV Cache Offloading 的决策模型与长上下文内存规划的量化分析能力
6. 能根据业务场景选择合适的分布式部署方案，并给出量化依据

### Day 0: 环境与模型确认

**现有模型**
```bash
ls -lh /data/model/
# 已有：opt-125m, Qwen2.5-7B, Qwen2.5-7B-Instruct-AWQ, Qwen2.5-7B-Instruct-GPTQ-Int4
#       Mixtral-8x7B-Instruct-v0.1, Qwen2-57B-A14B-Instruct, lora-adapters
```

**各 Day 实验的模型需求**
- Day 1-2 PD 分离：Qwen2.5-7B 即可（验证状态机逻辑，不依赖模型规模）
- Day 3-4 EP/EPLB：Mixtral-8x7B 或 Qwen2-57B-A14B（已有）
- Day 5 DFlash/DSpark：代码阅读不需要模型；**实践实验需要配套的 target + drafter 模型对**（如 Qwen3-8B + dspark_qwen3_8b_block7），按需决定是否下载
- Day 6 Offloading/长上下文：Qwen2.5-7B 可验证 offloading 逻辑；内存规划是纸面计算
- Day 7 决策实践：复用上述实验模型

### Day 1-2: PD 分离架构 ✅

- **原理**：Prefill 和 Decode 分离到不同节点
  - Prefill 节点：计算密集，大 batch 高吞吐处理 prompt
  - Decode 节点：内存密集，专注低延迟 token 生成
  - 优势：解耦两阶段资源需求，各自独立扩缩容
- **KV Connector 机制**：
  - Prefill → Decode 的 KV Cache 传输
  - `connector.request_finished()` 通知机制
  - `WAITING_FOR_REMOTE_KVS` 状态管理
  - `delay_free_blocks`：等待传输完成再释放
  - `_inflight_prefills`：追踪正在传输的请求
- **KV Connector 多实现对比**（理解为什么需要多种实现，各自解决什么问题）：
  - `simple_connector`：基础实现，单机验证用
  - `flexkv_connector`：灵活的 KV 传输策略
  - `lmcache_connector` / `lmcache_mp_connector`：基于 LMCache 的高性能实现
  - `nixl/`：NVIDIA NIXL 通信库（RDMA/NVLink 优化）
  - `mooncake/`：Mooncake 分布式缓存后端
  - 选型依据：网络拓扑（NVLink vs InfiniBand vs RoCE）、延迟要求、是否需要持久化
- **Prefill 端与 Decode 端的 Scheduler 行为差异**：
  - Prefill 端：执行 prefill → 调用 `connector.request_finished()` → 触发 KV 传输 → `defer_block_free` 等传输完成后释放本地 blocks
  - Decode 端：请求进入 `WAITING_FOR_REMOTE_KVS` 状态 → 等待 KV 到达 → `load_kv_async` 异步加载 → `reserved_blocks` 预留空间 → `num_external_computed_tokens` 标记已计算量 → 转入正常 decode
  - 追踪任务：从 Prefill 端 `request_finished()` 到 Decode 端开始第一次 decode 的完整状态转换链路
- **Scheduler 配合**：
  - `load_kv_async`：异步加载远端 KV
  - `reserved_blocks`：为 in-flight 传输预留 blocks
  - `num_external_computed_tokens`：远端已计算的 token 数
- **代码路径**：
  - `vllm/distributed/kv_transfer/` — KV 传输框架
  - `vllm/distributed/kv_transfer/kv_connector/v1/` — 各 connector 实现
  - `vllm/v1/core/sched/scheduler.py` — connector 相关逻辑（`WAITING_FOR_REMOTE_KVS` L704-706，`_inflight_prefills` L360，`load_kv_async` L740+）
- **实践**：
  - 单机 8 卡模拟 PD 分离（rank 0-3 Prefill，rank 4-7 Decode），验证状态机和 KV 传输逻辑
  - ⚠️ **单机局限**：KV 传输走本地内存/NVLink，无法体现真实网络带宽瓶颈
  - **理论补偿**：基于网络参数推算 KV 传输延迟，判断传输是否成为瓶颈
    - KV size per request = num_layers × 2 × num_kv_heads × head_size × seq_len × dtype_bytes
    - 传输延迟 = KV size / 网络带宽（InfiniBand 400Gbps ≈ 50GB/s）
    - 与 prefill 计算时间对比：传输延迟 vs prefill latency，何时传输成为瓶颈
    - 参考论文：DistServe (https://arxiv.org/abs/2401.09670) 的 goodput 分析

### Day 3-4: Expert Parallelism 深入与 EPLB ✅

- **EP 深入**（基于 Week 6 基础）：
  - EP vs TP 的决策：何时用 EP 更优
  - EP + TP 组合策略的具体配置
  - All-to-All 通信优化
- **EPLB（Expert-Parallel Load Balancing）**：
  - 问题：部分 expert 被频繁选中（热点），负载不均
  - 方案：热点 expert 复制到多个 GPU，冷门 expert 合并
  - 负载统计与重均衡策略
  - EPLB policy 抽象设计：`policy/abstract.py` 定义接口，`policy/default.py` 默认策略实现
  - **重均衡触发机制**：触发条件（负载方差超过阈值？定期统计？）、重均衡频率、执行期间对在线推理的影响（是否需要暂停、如何做到 zero-downtime rebalance）
  - 代码：`vllm/distributed/eplb/` — `eplb_state.py`（状态管理）、`eplb_communicator.py`（通信）、`rebalance_execute.py`（执行）、`policy/`（策略）
- **弹性 Expert Parallelism**：
  - 动态调整 expert 在不同 GPU 上的分布
  - 弹性扩缩容：动态增减 GPU 参与计算
  - 代码：`vllm/distributed/elastic_ep/` — `elastic_execute.py`、`elastic_state.py`、`standby_state.py`
  - 注意：弹性 EP 在 vLLM 中相对较新，学习时先确认代码成熟度，重点放在 EPLB 核心逻辑
- **实践**：
  - 使用 Mixtral-8x7B 或 Qwen2-57B-A14B，对比 EP=2/4/8 配置下的吞吐和延迟
  - 监控 expert 负载分布（哪些 expert 被高频选中）
  - 验证 EPLB 开启前后的负载方差变化

> **完成状态**：
> - ✅ EPLB 背景与原理学习 → `learning/week8-distributed-production/eplb_background.md`
> - ✅ EPLB 代码追踪（eplb_state.py / policy/default.py / rebalance_execute.py / base_router.py）
> - ✅ Elastic EP 背景与原理学习 → `learning/week8-distributed-production/elastic_ep_background.md`
> - ✅ Elastic EP 状态机设计理念（cooperative scheduling、非阻塞分步执行）
> - ⏭️ EPLB 实验（跳过）

### Day 5: 生产级投机解码 — DFlash + DSpark ✅

> 前置知识：Week 5 Speculative Decoding 基础（Eagle/Medusa/N-gram 原理与验证流程）、Week 3 MLA 注意力机制。

- **回顾与动机**：
  - Week 5 学过的投机解码方法（Eagle 自回归 draft、Medusa 并行独立 head）各有瓶颈
  - Eagle：draft 质量高但逐 token 自回归慢；Medusa：并行快但位置间无依赖，接受率随 draft 长度衰减
  - DFlash 和 DSpark 是 DeepSeek 在生产环境中实际部署的方案，解决上述问题
- **DFlash：并行 Draft 机制**：
  - 核心思路：一次前向生成整个 draft block（如 7 个 token），不逐 token 自回归
  - Context-KV precompute：复用目标模型已有的 KV Cache 作为 context
  - 非因果 query-block forward：draft block 内部不做因果 mask，所有位置并行计算
  - Anchor token：用上一步的确认 token 作为锚点，从锚点开始预测
  - 与标准 Attention 的区别：draft block 内部是 fill-in（非因果），对 context 是因果
  - 代码（两层架构）：
    - `vllm/v1/spec_decode/dflash.py` — proposer 编排层（管"何时 draft"、与 Scheduler 交互）
    - `vllm/v1/worker/gpu/spec_decode/dflash/` — GPU speculator 实现（管"怎么 draft"、CUDA graph）
- **DSpark：半自回归框架（DFlash + Markov Head）**：
  - 两阶段架构：
    1. Parallel Backbone：复用 DFlash 的并行前向，一次出 N 个位置的 hidden states
    2. Sequential Markov Head：从左到右顺序采样，每一步注入前一 token 的转移偏置
  - Markov Head 低秩设计（`DSparkMarkovHead`）：
    - `markov_w1`：Embedding(vocab_size, r) — 把前一采样 token 映射到 r 维空间
    - `markov_w2`：Linear(r, vocab_size) — 投射回词表大小作为 bias
    - `logits[i] = base_logits[i] + markov_bias(prev_sampled_token)`
  - 为什么有效：并行骨干保证速度，Markov Head 以极低代价（低秩矩阵乘）补偿位置间依赖，提升接受率
  - `sample_from_anchor`：anchor-as-first-prediction（N 个 query 位置，每个都是预测）vs 1+N fill-in 布局
  - 代码：`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` — DSpark speculator（继承 DFlashSpeculator）
  - 代码：`vllm/model_executor/models/qwen3_dspark.py` — Qwen3 DSpark draft model（含 MarkovHead）
- **Dynamic Speculative Decoding Schedule（动态投机调度）**：
  - 按 batch size 区间映射 `num_speculative_tokens`：高并发时缩短 draft 长度，低并发时拉长
  - 本质是 throughput-oriented 的静态映射表，不是运行时基于 logits 置信度的动态决策
  - 目的：高 batch 时减少验证开销对吞吐的拖累，低 batch 时最大化加速比
  - 与 Week 5 固定 `num_speculative_tokens` 的对比：固定值在负载波动时无法适应
  - 代码：`vllm/v1/spec_decode/dynamic/` — `validate_and_normalize_dynamic_sd_schedule()`、`build_dynamic_sd_schedule_lookup()`
- **与 MLA 的交互**（串联 Week 3 知识）：
  - DeepSeek-V4 = MoE + MLA + DSpark，三者在 KV Cache 层面如何协作
  - DSpark drafter 如何复用 MLA 的压缩 KV Cache（latent KV）
  - 代码：`vllm/models/deepseek_v4/nvidia/dspark.py` — DeepSeek-V4 原生 DSpark
- **方法对比总结**：

  | 方法 | Draft 策略 | Token 间依赖 | 速度 | 接受率 |
  |------|-----------|-------------|------|--------|
  | Eagle | 自回归 | 完全依赖 | 慢 | 高 |
  | Medusa | 并行多 head | 无 | 快 | 中（衰减快） |
  | DFlash | 并行 backbone | 无 | 快 | 中 |
  | DSpark | 并行 + Markov Head | 低秩偏置 | 快 | 高 |

- **代码追踪路径汇总**：
  - `vllm/v1/spec_decode/dflash.py` — DFlash proposer 编排层
  - `vllm/v1/spec_decode/dynamic/` — Dynamic SD Schedule 实现
  - `vllm/v1/worker/gpu/spec_decode/dflash/` — DFlash GPU speculator
  - `vllm/v1/worker/gpu/spec_decode/dspark/` — DSpark GPU speculator（继承 DFlash）
  - `vllm/v1/worker/gpu/spec_decode/dspark/utils.py` — DSpark 模型加载
  - `vllm/model_executor/models/qwen3_dspark.py` — Qwen3 DSpark（MarkovHead 实现）
  - `vllm/model_executor/models/gemma4_dspark.py` — Gemma4 DSpark
  - `vllm/models/deepseek_v4/nvidia/dspark.py` — DeepSeek-V4 DSpark
  - `vllm/config/speculative.py` — DSpark 配置与自动检测逻辑
- **实践**：
  - 用 DSpark drafter（如 `deepseek-ai/dspark_qwen3_8b_block7`）跑 benchmark
  - 对比 Eagle vs DSpark 的加速比和接受率
  - 观察不同 `num_speculative_tokens` 对 DSpark 接受率的影响
  - 测试 Dynamic SD Schedule：配置不同 batch size 区间的 draft 长度，观察高/低并发下的吞吐变化
  - 论文参考：[DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)

> **完成状态**：
> - ✅ DFlash + DSpark 背景与原理学习 → `learning/week8-distributed-production/dflash_dspark_background.md`
> - ✅ 代码追踪（配置入口 / draft 模型核心 / speculator / proposer / Dynamic SD Schedule）→ `learning/week8-distributed-production/dflash_dspark_code.md`
> - ✅ 实验 06: DSpark vs Baseline（TPOT 降低 52.6%，TTFT 增加 31.3%）
> - ✅ 实验 07: K sweep（K=1/3/5/7，K=3 为性价比拐点）→ `learning/week8-distributed-production/dflash_dspark_experiment.md`

### Day 6: KV Cache Offloading + 长上下文内存规划 ✅

- **KV Cache Offloading 决策模型**：
  - 核心问题：什么时候 offload 比重算（preempt + re-prefill）更划算？
  - Offload 延迟 = KV size / PCIe 带宽（PCIe 4.0 x16 ≈ 32GB/s）
  - 重算延迟 = prefill FLOPs / GPU 算力（H800 FP16 ≈ 990 TFLOPS）
  - Break-even 分析：短序列重算快、长序列 offload 快，推导临界点
  - 异步传输（overlap with compute）对 break-even 的影响
  - `defer_block_free` 机制：等待 offload 完成后再释放 block，复用 PD 分离的 fencing 逻辑
- **两套 Offloading 实现**：
  - `vllm/v1/simple_kv_offload/` — 轻量级 GPU→CPU copy（`copy_backend.py`、`cuda_mem_ops.py`、`manager.py`、`worker.py`），适合简单场景
  - `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` + `offloading/` — 完整框架（含 scheduler、worker、metrics），适合生产环境需要监控和精细控制的场景
  - 选型依据：是否需要 metrics 监控、是否需要与 PD 分离复用基础设施
- **长上下文内存规划（量化计算）**：
  - KV Cache 公式：`per_request_kv = num_layers × 2 × num_kv_heads × head_size × seq_len × dtype_bytes`
  - 练习：给定 Llama-70B（80 layers, 8 KV heads/GQA, 128 head_size, FP16）
    - 128K context：每请求 KV = 80 × 2 × 8 × 128 × 131072 × 2 bytes ≈ 32GB
    - TP=8 后每卡 KV = 32/8 = 4GB，80GB 卡扣掉权重后能服务几个并发？
    - FP8 KV Cache：KV 减半 → 并发翻倍
    - CP=2：每卡 KV 再减半 → 进一步扩容
  - 分块策略：超长输入的 Chunked Prefill 配置（`max_num_batched_tokens` 如何选取）
  - CP + FP8 KV Cache 组合降低内存压力的量化分析
  - 实际部署的 context length 上限评估方法
- **EC Connector（多模态 PD 分离）**：
  - 用途：在 PD 分离场景下传输 encoder cache（视觉编码器输出），与 KV Connector 分工——KV Connector 传 decoder KV Cache，EC Connector 传 encoder embeddings
  - 代码：`vllm/distributed/ec_transfer/` — `ec_connector/base.py`（`ECConnectorBase`）、`ec_connector/cpu/`（CPU 实现）
- **代码路径**：
  - `vllm/v1/simple_kv_offload/` — 轻量级 offloading
  - `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` — 完整 offloading 框架
  - `vllm/distributed/ec_transfer/` — encoder cache 传输
  - `vllm/v1/core/single_type_kv_cache_manager.py` — `_pending_partial_tail_offloads`（offload 与 block 管理的交互）

> **完成状态**：
> - ✅ KV Connector 框架代码追踪（factory → base → multi_connector → simple_cpu_offload_connector）
> - ✅ Scheduler 5 个 connector 调用点详细追踪（bind_gpu_block_pool / get_num_new_matched_tokens / update_state_after_alloc / build_connector_meta / request_finished）
> - ✅ Worker 端流程追踪（pre_forward / post_forward / DmaCopyBackend daemon thread + dual CUDA stream）
> - ✅ PD 分离 vs Offloading 机制对比（心跳/租约、finished_recving 复用、逐层传输）
> - ✅ 代码追踪总结文档 → `learning/week8-distributed-production/kv_offloading_code_tracing.md`
> - ✅ 实验脚本编写 → `learning/week8-distributed-production/experiments/exp_kv_offloading.sh`
> - ✅ 实验执行与结果分析 → `learning/week8-distributed-production/day6_experiment_analysis.md`
> - 关键实验发现：GPU KV Cache 未饱和时 offloading 不产生收益（external prefix cache hit = 0%），但也不引入退化；Prefix cache 本身效果显著（TTFT -65%，TPOT -41%）

### Day 7: 分布式部署决策实践 ✅

- **场景分析与方案选择**：
  - Dense 模型（7B-70B）：TP/PP 选择依据
  - MoE 模型（Mixtral/DeepSeek）：EP vs TP 的决策树
  - 超长上下文（128K+）：CP 配置策略
  - 高 QPS 场景：PD 分离 + DP 组合
  - 延迟敏感场景：投机解码方案选择（Eagle vs DSpark vs N-gram，何时值得开启）
- **端到端 Benchmark**：
  - 对比不同分布式方案的 TTFT/TPOT/Throughput
  - **归因分析**：每个实验不只记录数字，要分析瓶颈来源（通信/显存/计算），标注"该结论在什么条件下可能反转"
  - ⚠️ **单机实验局限**：PD 分离和高 QPS DP 场景的网络传输瓶颈无法真实体现，用 Day 1-2 的理论推算补偿
- **交付物**：一份可复用的**部署决策文档**，包含：
  - 决策树：模型类型 × 规模 × 硬件 × 延迟/吞吐要求 → 推荐方案
  - 每种方案的适用边界条件与反转条件
  - 实验数据作为决策依据
  - 单机验证结果 + 多节点场景的理论分析
- 编写 Week 8 SUMMARY.md

> **完成状态**：
> - ✅ Week 8 SUMMARY.md → `learning/week8-distributed-production/WEEK8_SUMMARY.md`
> - 涵盖 PD 分离、EPLB、DSpark、KV Offloading 四大主题的技术总结、实验数据汇总、生产部署决策表

**Week 8 完成时间**: 2026-10-10

---

## Week 9: 服务配置调优与端到端 Benchmark ✅

### 学习目标
1. 掌握 AsyncLLMEngine 异步架构与请求生命周期的完整实现
2. 掌握 vLLM 服务核心参数的交互影响，能根据业务场景系统调优
3. 掌握 Metrics 监控体系（PerfStats / Prometheus），能读懂关键指标并定位瓶颈
4. 掌握端到端 benchmark 方法论与工具链（`vllm bench serve` / `benchmark_serving.py`）
5. 能搭建完整的"部署 → 压测 → 观测 → 调优"闭环并产出可复用的配置模板

### Day 1: AsyncLLMEngine 异步架构与请求生命周期 ✅

> 前置知识：Week 3 Day 1 建立了端到端全景（HTTP → Engine → Scheduler → ModelRunner），本 Day 深入 Engine 层异步实现细节。

- **AsyncLLM 核心架构**：
  - `AsyncLLM`（`vllm/v1/engine/async_llm.py`）作为 `EngineClient` 的入口角色
  - `generate()` 的异步实现：`add_request()` → `RequestTracker` → SSE stream 返回
  - `EngineCore`（`vllm/v1/engine/core.py`）与 `EngineCoreClient`（`vllm/v1/engine/core_client.py`）的进程间通信
  - Engine loop：`EngineCore.step()` → Scheduler → ModelRunner → 输出收集的单步循环
  - 多进程架构：API Server 进程 ↔ EngineCore 进程 ↔ Worker 进程（通过 zmq/shared memory）
- **请求追踪与流式输出**：
  - `OutputProcessor`（`vllm/v1/engine/output_processor.py`）：token 输出的收集、detokenize、增量推送
  - `Detokenizer`（`vllm/v1/engine/detokenizer.py`）：增量 detokenize 的实现（避免重复解码）
  - Streaming 实现：`AsyncStream` / `RequestOutput` 的生成与推送时机
  - 并行采样：`parallel_sampling.py` 如何处理 `n > 1` 的请求
- **Graceful Shutdown**：
  - `shutdown()` 方法：drain 模式（等待 in-flight 请求完成）
  - `wait_for_requests_to_drain()` 的超时机制
  - `shutdown_timeout` 配置：优雅关闭的最大等待时间
  - 信号处理：`launcher.py` 中 SIGTERM → shutdown_event → drain → server.shutdown() 链路
- **代码精读**：
  - `vllm/v1/engine/async_llm.py` — AsyncLLM 主类（重点：`generate()`、`shutdown()`、`_run_output_handler()`）
  - `vllm/v1/engine/core.py` — EngineCore（重点：`step()`、`_process_engine_step()`）
  - `vllm/v1/engine/core_client.py` — 进程间通信（重点：zmq socket 类型选择、序列化）
  - `vllm/entrypoints/launcher.py` — 启动与关闭流程

### Day 2: VllmConfig 参数体系与交互影响 ✅

> 前 8 周分散地学了各参数的原理，本 Day 系统性地理解参数之间的交互，建立调优思维。

- **配置架构总览**：
  - `VllmConfig`（`vllm/config/vllm.py`）：20+ 子 Config 的聚合，理解初始化时的交叉校验逻辑
  - 从 CLI 参数到 VllmConfig 的传递链路：`EngineArgs` → `VllmConfig.__post_init__()` → 各子 Config 的 validate
  - 关键子 Config：`SchedulerConfig`、`CacheConfig`、`ParallelConfig`、`SpeculativeConfig`、`ObservabilityConfig`
- **核心参数三角：max_num_seqs × max_num_batched_tokens × gpu_memory_utilization**：
  - `max_num_seqs`：最大并发请求数，影响 Scheduler 调度上限和 CUDA Graph capture 数量
  - `max_num_batched_tokens`：每步处理的 token 总预算，约束 Chunked Prefill 块大小和 prefill/decode 混合比例
  - `gpu_memory_utilization`：KV Cache 可用显存比例，决定 block pool 大小（进而决定最大并发和最大序列长度）
  - **交互关系**：三者共同决定了系统的 throughput/latency/max_context_length 权衡空间
  - `VllmConfig.__post_init__()` 中对这三者的交叉校验（如 `max_num_batched_tokens` 必须容纳 `num_speculative_tokens * max_num_seqs`）
- **Chunked Prefill 参数**：
  - `enable_chunked_prefill`：开启后 `max_num_batched_tokens` 成为 chunk 上限
  - Chunk 大小对 TTFT / ITL 的影响：大 chunk → TTFT 低但 ITL 波动大，小 chunk → ITL 平稳但 TTFT 高
  - Week 5 实验数据回顾：1024 vs 无 chunked prefill 的 P99 ITL 差异
- **Prefix Caching 参数**：
  - `enable_prefix_caching`：hash-based block 复用
  - Block pool 的 LRU 淘汰与 `free_queue` 尾部保留策略
  - 命中率与 `gpu_memory_utilization` 的关系：更大的 block pool → 更多缓存空间 → 更高命中率
- **CUDA Graph 相关参数**：
  - `enforce_eager`：关闭 CUDA Graph 的调试开关
  - `max_num_seqs` 影响需要 capture 的 graph 数量 → warmup 时间
  - Graph capture 对显存的额外占用
- **场景化配置模板**（本 Day 交付物之一）：
  - 高吞吐场景（batch 处理、离线推理）：大 `max_num_seqs` + 大 `max_num_batched_tokens` + 高 `gpu_memory_utilization`
  - 低延迟场景（在线服务、流式对话）：适中 `max_num_seqs` + 较小 `max_num_batched_tokens`（开 chunked prefill）
  - 长上下文场景（128K+ 输入）：低 `max_num_seqs` + 高 `gpu_memory_utilization` + FP8 KV Cache
  - 多 LoRA 场景：`max_loras` + `max_num_seqs` 的双重约束
- **代码路径**：
  - `vllm/config/vllm.py` — VllmConfig 聚合与交叉校验（`__post_init__()` 重点阅读）
  - `vllm/config/scheduler.py` — SchedulerConfig（`max_num_seqs`、`max_num_batched_tokens`、`enable_chunked_prefill`）
  - `vllm/config/cache.py` — CacheConfig（`gpu_memory_utilization`、`block_size`、`enable_prefix_caching`）
  - `vllm/config/observability.py` — ObservabilityConfig

### Day 3: Metrics 监控体系 ✅

> Week 10 Day 1-2 原有 Metrics 内容前移至此，与服务调优形成闭环。

- **vLLM Metrics 架构**：
  - `vllm/v1/metrics/` 目录结构：`perf.py`（性能统计）、`prometheus.py`（Prometheus 导出）、`stats.py`（统计聚合）、`loggers.py`（日志输出）
  - 数据流：ModelRunner 采集原始计时 → `PerfStats` 聚合 → `PerfMetricsLogging` 日志输出 / Prometheus 导出
- **PerfStats 与 ComponentMetrics**：
  - `PerfStats`（`vllm/v1/metrics/perf.py:94`）：单步执行的性能统计
  - `ComponentMetrics` 体系：`AttentionMetrics` / `MLAAttentionMetrics` / `FfnMetrics` / `UnembedMetrics`
  - `ModelMetrics`（`vllm/v1/metrics/perf.py:1268`）：模型级聚合
  - 理解每个 metric 的含义：prefill_time、decode_time、attention_time、ffn_time、scheduling_time
- **Prometheus 指标精读**：
  - 请求级：`vllm:num_requests_running` / `vllm:num_requests_waiting` / `vllm:num_requests_swapped`
  - 吞吐级：`vllm:avg_generation_throughput_toks_per_s` / `vllm:avg_prompt_throughput_toks_per_s`
  - 延迟级：`vllm:e2e_request_latency_seconds` / `vllm:time_to_first_token_seconds` / `vllm:time_per_output_token_seconds`
  - 资源级：`vllm:gpu_cache_usage_perc`（KV Cache 使用率）/ `vllm:cpu_cache_usage_perc`
  - 调度级：`vllm:num_preemptions_total` / `vllm:num_generation_tokens_total`
  - 每个指标对应的代码位置和更新时机
- **KV Cache Metrics**：
  - `KVCacheMetricsCollector`（`vllm/v1/core/kv_cache_metrics.py`）：block 使用率、碎片率、prefix cache 命中率
  - `BlockMetricsState`：prefix_cache_hit_rate、block_utilization 的计算方法
  - 如何从 metrics 判断是否需要调大 `gpu_memory_utilization` 或开启 prefix caching
- **实践**：
  - 启动 vLLM 服务，访问 `/metrics` 端点，理解每个 Prometheus 指标的含义
  - 发送不同并发的请求，观察 `num_requests_running`、`gpu_cache_usage_perc`、`num_preemptions_total` 的变化
  - 识别关键告警阈值：KV Cache 使用率 > 90%（即将 OOM）、preemption 频繁（内存不足）、waiting 队列堆积（吞吐不足）
- **代码路径**：
  - `vllm/v1/metrics/perf.py` — 性能统计核心（PerfStats / ComponentMetrics / ModelMetrics）
  - `vllm/v1/metrics/prometheus.py` — Prometheus 指标注册与导出
  - `vllm/v1/metrics/stats.py` — 统计聚合
  - `vllm/v1/core/kv_cache_metrics.py` — KV Cache 专项指标

### Day 4: 参数扫描实验 ✅

> 前 3 天建立了理论基础，本 Day 用实验验证参数交互影响。

- **实验 1：max_num_seqs 扫描**：
  - 固定模型（Qwen2.5-7B, TP=1）、固定负载（ShareGPT / 固定长度 prompt）
  - 扫描 `max_num_seqs` = 4 / 8 / 16 / 32 / 64 / 128
  - 测量指标：Throughput、Mean TTFT、P99 TPOT、GPU Cache 使用率
  - 预期：小 → GPU 利用率低；过大 → KV Cache 不足导致 preemption、延迟飙升
  - 找到当前模型/硬件下的 sweet spot
- **实验 2：max_num_batched_tokens 扫描**：
  - 开启 chunked prefill
  - 扫描 `max_num_batched_tokens` = 512 / 1024 / 2048 / 4096 / 8192
  - 使用混合负载（短 prompt + 长 prompt 混合）
  - 测量指标：Mean TTFT、P99 ITL、Throughput
  - 验证 Week 5 结论：小值平稳 ITL 但 TTFT 高，大值反之
- **实验 3：gpu_memory_utilization 扫描**：
  - 扫描 `gpu_memory_utilization` = 0.7 / 0.8 / 0.85 / 0.9 / 0.95
  - 高并发负载（request_rate 逐步增大直到打满）
  - 测量指标：Max concurrent requests、Preemption count、Throughput at saturation
  - 观察 prefix caching 命中率随可用 block 数的变化
- **实验 4：组合调优**：
  - 基于实验 1-3 的 sweet spot，组合测试
  - 对比 baseline（默认参数）vs 调优后的配置
  - 量化整体收益：Throughput 提升百分比、尾延迟改善
- **交付物**：
  - 参数扫描结果表格（每个实验的原始数据 + 可视化趋势）
  - `learning/week9-serving/param_tuning_guide.md`：调优手册，记录每个参数的 sweet spot 范围和选择依据
- **代码路径（benchmark 工具）**：
  - `benchmarks/benchmark_serving.py` — 在线服务 benchmark
  - `vllm/entrypoints/cli/benchmark/serve.py` — `vllm bench serve` 子命令
  - `vllm/entrypoints/cli/benchmark/throughput.py` — 离线吞吐 benchmark
  - `vllm/entrypoints/cli/benchmark/latency.py` — 延迟 benchmark

### Day 5: Benchmark 工具链与方法论 ✅

> 从 Day 4 的实验中提炼方法论，掌握 benchmark 工具的正确使用方式。

- **vLLM 官方 Benchmark 工具**：
  - `vllm bench serve`：在线服务 benchmark（Poisson arrivals / 固定 QPS）
  - `vllm bench throughput`：离线吞吐 benchmark
  - `vllm bench latency`：单请求延迟 benchmark
  - `vllm bench startup`：启动时间 benchmark（含 warmup + CUDA Graph capture）
  - `benchmarks/benchmark_serving.py`：更灵活的自定义 benchmark（支持 ShareGPT、sonnet、random 等数据集）
- **Benchmark 设计原则**：
  - **预热（Warmup）**：首轮请求触发 CUDA Graph capture、JIT 编译等一次性开销，必须排除
  - **统计显著性**：最少 100+ 请求，关注 P50/P95/P99 而非 mean（尾延迟更能反映真实体验）
  - **真实流量模式**：Poisson arrivals（随机到达）比固定间隔更接近真实场景
  - **数据集选择**：ShareGPT traces（真实对话长度分布）vs 固定长度（控制变量）vs sonnet（长 prompt 压力测试）
  - **控制变量**：一次只变一个参数，其余固定
- **指标体系**：
  - **TTFT（Time to First Token）**：P50/P95/P99，反映 prefill 性能和排队延迟
  - **TPOT（Time Per Output Token）**/ **ITL（Inter-Token Latency）**：P50/P95/P99，反映 decode 性能
  - **Throughput（tokens/s）**：output tokens/s + prompt tokens/s，反映系统整体处理能力
  - **Request throughput（req/s）**：在给定 SLO 下能承载的最大 QPS
  - **Goodput**：满足 SLO（如 TTFT < 500ms, TPOT < 100ms）的请求占比
  - GPU 利用率：`nvidia-smi` 观察 SM 占用率和显存使用
- **常见 Benchmark 陷阱**：
  - 未预热：首批请求的 CUDA Graph capture 时间被计入
  - 请求数太少：统计波动大，P99 不可信
  - 忽略排队延迟：TTFT 包含排队时间，高并发下 TTFT 飙升不一定是 prefill 慢
  - 未区分 prefill-bound vs decode-bound：需要分别分析
- **实践**：
  - 用 `vllm bench serve` 和 `benchmarks/benchmark_serving.py` 分别跑同一配置，对比结果一致性
  - 用 ShareGPT 数据集做真实流量模拟，逐步增大 `request_rate` 直到系统饱和
  - 找到当前配置的 QPS 上限（Goodput 开始下降的拐点）

### Day 6: 故障模拟与诊断实练 ✅

> 把 Week 10 Day 5 的故障排查前移，在已搭建好的服务上实际触发和诊断。

- **故障场景 1：KV Cache 耗尽（OOM）**：
  - 触发方法：设置低 `gpu_memory_utilization`（如 0.3）+ 高 `max_num_seqs` + 长 prompt
  - 观察 Metrics：`gpu_cache_usage_perc` 接近 100%、`num_preemptions_total` 激增、`num_requests_waiting` 堆积
  - 定位路径：Prometheus → Scheduler preemption 日志 → `kv_cache_manager.can_allocate()` 返回 False
  - 解决方案：调大 `gpu_memory_utilization` / 调小 `max_num_seqs` / 开启 FP8 KV Cache / 开启 prefix caching
- **故障场景 2：Scheduler 饥饿（TTFT 飙升）**：
  - 触发方法：发送大量长 prompt（如 4K tokens）且不开 chunked prefill
  - 现象：新请求 TTFT 极高（排队等待 prefill 完成）
  - 观察 Metrics：`num_requests_waiting` 持续增长、`time_to_first_token_seconds` P99 飙升
  - 定位路径：单个长 prefill 阻塞 → decode 请求被延迟
  - 解决方案：开启 chunked prefill + 设置合理的 `max_num_batched_tokens`
- **故障场景 3：吞吐突降**：
  - 触发方法：负载从低并发突增到高并发（burst traffic）
  - 现象：短时间内 throughput 下降，然后逐渐恢复
  - 观察 Metrics：prefix cache 命中率骤降（新 prefix 大量涌入）、block 分配压力增大
  - 定位路径：Burst 导致 prefix cache 失效 → 大量 prefill 重算 → 吞吐临时下降
- **故障场景 4：TP 通信异常**：
  - 触发方法：TP=4 部署，用 `NCCL_DEBUG=INFO` 观察通信日志
  - 了解 NCCL timeout 的默认值和配置方法
  - `NCCL_TIMEOUT` / `NCCL_ASYNC_ERROR_HANDLING` 的作用
  - 常见 NCCL 错误及对应的排查方向
- **诊断工具箱**（串联前 8 周已学工具）：
  - `nvidia-smi dmon -s u`：实时 GPU 利用率和显存
  - `/metrics` 端点：Prometheus 指标（Day 3 已学）
  - `VLLM_LOGGING_LEVEL=DEBUG`：详细调度日志
  - `NCCL_DEBUG=INFO`：通信日志（Week 4 已学）
  - `torch.profiler`：kernel 级别定位（Week 1 已用）
- **交付物**：
  - `learning/week9-serving/fault_diagnosis_playbook.md`：故障诊断手册
    - 每种故障的症状、Metrics 特征、定位步骤、解决方案
    - 诊断决策树：TTFT 高 → 检查 waiting 队列 / 检查 chunked prefill / 检查 prefix cache 命中率

### Day 7: 部署架构概览（VLM + Executor + 容器化）+ 总结 ✅

> 将 VLM 部署差异、Executor 多后端架构、容器化/K8s 基础认知整合为 1 天。重点是"理解设计 + 知道配置要点"，不做实操部署。

- **多模态推理部署要点**（理论，不深入代码实现）：
  - VLM 部署与 dense 模型的关键差异：
    - 图像 token 占用 KV Cache：一张图像可能展开为 576+ 个 token → 大幅增加 KV Cache 消耗
    - `EncoderCacheManager`：编码器输出的独立缓存池，与 decoder KV Cache 分开管理
    - `SchedulerConfig.max_num_encoder_input_tokens`：控制每步最大 encoder token 数
    - `EncoderCacheManagerConfig`：encoder cache 的内存预算配置
  - 部署调优：根据图像分辨率估算 token 数 → 调整 `max_num_seqs` 和 `gpu_memory_utilization`
  - 代码路径（快速浏览）：
    - `vllm/v1/core/encoder_cache_manager.py` — encoder cache 分配与释放
    - `vllm/v1/worker/gpu/mm/` — encoder runner 与 CUDA graph
    - `vllm/multimodal/` — 多模态输入解析
- **Executor 多后端架构**（代码阅读）：
  - 三种 Executor 的定位与选择逻辑：
    - `UniprocExecutor`（`vllm/v1/executor/uniproc_executor.py`）：TP=1 单卡，无多进程开销
    - `MultiprocExecutor`（`vllm/v1/executor/multiproc_executor.py`）：单机多卡默认，spawn 进程 + shared memory / zmq
    - `RayExecutor`（`vllm/v1/executor/ray_executor.py`）：跨节点，Ray Actor 封装 Worker，gRPC / object store
  - 选择逻辑：TP=1 → Uniproc；单机多卡 → Multiproc；多节点 → Ray
  - `MultiprocExecutor` 精读（单机环境实际使用的后端）：
    - `WorkerProc` 生命周期：spawn → init_device → load_model → ready → execute_model loop
    - Worker 间通信：zmq PUSH/PULL 传递 SchedulerOutput，shared memory 传递大 tensor
  - `RayExecutor` 概览（代码阅读，了解与 Multiproc 的差异即可）：
    - Ray Actor 如何封装 Worker，跨节点 NCCL group 建立
    - 故障恢复：Ray Actor restart 机制
  - 代码路径：
    - `vllm/v1/executor/abstract.py` — Executor 基类接口
    - `vllm/v1/executor/multiproc_executor.py` — 单机多卡实现（重点阅读）
    - `vllm/v1/executor/ray_executor.py` — Ray 实现（概览）
- **容器化与 K8s 部署基础**（理论认知，不做实操）：
  - **vLLM 官方 Dockerfile 结构**：
    - `docker/Dockerfile`：多阶段构建（build stage → runtime stage），理解为什么需要分阶段（减小镜像体积 + 编译依赖不进运行时）
    - GPU 运行时依赖：`nvidia-container-runtime`、`--gpus all`、`--ipc=host`（shared memory）
    - 常见的 Docker 启动参数：`--shm-size`（/dev/shm 大小，影响 NCCL 通信）、`-v` 模型挂载
  - **Health Probe 实现**：
    - `/health` 端点：`vllm/entrypoints/serve/instrumentator/health.py` — 调用 `client.check_health()` 检查引擎状态
    - gRPC Health：`vllm/entrypoints/grpc_server.py` — `VllmHealthServicer`，shutdown 时设置 `NOT_SERVING`
    - DP Supervisor 的 health 聚合：`vllm/entrypoints/openai/dp_supervisor.py` — 聚合多个 DP worker 的 health 状态
    - K8s 场景：Liveness probe（进程是否存活）→ `/health`；Readiness probe（是否可接收流量）→ 模型加载完成后才 ready
  - **Graceful Shutdown 与 K8s 的配合**：
    - K8s 发送 SIGTERM → `launcher.py` 捕获信号 → `shutdown_event.set()` → drain in-flight 请求 → `server.shutdown()`
    - `shutdown_timeout` 配置：必须小于 K8s 的 `terminationGracePeriodSeconds`，否则被强制 SIGKILL
    - 滚动更新（Rolling Update）：新 Pod ready 后旧 Pod 才 drain，保证服务不中断
  - **关键配置清单**（了解即可，供未来上生产时查阅）：
    - GPU 资源声明：`nvidia.com/gpu: N`
    - 内存资源：需预留足够 CPU 内存给模型加载和 CPU offloading
    - Node selector / tolerations：GPU 节点调度
    - HPA（Horizontal Pod Autoscaler）：基于 QPS / GPU 利用率的扩缩容策略
- **Week 9 总结与交付物检查**（编写 SUMMARY.md）：
  - 核心收获汇总
  - 参数调优手册（Day 2 + Day 4 产出）
  - 故障诊断手册（Day 6 产出）
  - Benchmark 实验数据汇总（Day 4-5 产出）
  - 与 Week 10 的衔接：Week 10 聚焦 profiling 工具链和端到端实战项目

---

## Week 10: Profiling 深入 + 端到端实战项目

### 学习目标
1. 掌握性能分析工具链（torch.profiler / nsys / Roofline），覆盖单卡和分布式场景
2. 建立"瓶颈特征 → 优化决策"的结构化映射框架
3. 完成双场景端到端优化实战（Dense + MoE），验证前 9 周全部分布式能力的落地闭环
4. 掌握新模型上线评估的全链路方法（架构分析 → 资源估算 → 方案选择 → 部署验证）
5. 理解成本优化方法并产出可交付的优化报告

> Metrics 监控和故障诊断已在 Week 9 Day 3 和 Day 6 覆盖，本周聚焦 kernel 级 profiling 和综合实战。

### Day 1: 性能分析工具链 ✅

- **torch.profiler**：
  - `torch.profiler.profile()` 的配置：`activities`（CPU/CUDA）、`record_shapes`、`with_stack`
  - Chrome Trace 可视化：`chrome://tracing` 或 Perfetto UI
  - `key_averages().table()` 的解读：top CUDA kernels、CPU/GPU 时间占比
  - 如何从 trace 中识别：attention kernel、MLP kernel、AllReduce 通信、CUDA Graph replay
- **Nsight Systems（nsys）**：
  - `nsys profile -o output vllm serve ...`：抓取完整的 kernel 时间线
  - Timeline 解读：CUDA kernel 执行、内存拷贝、CPU/GPU 同步点、kernel launch overhead
  - 与 torch.profiler 的互补：nsys 看系统级全景（含 NCCL 通信），torch.profiler 看 Python 级调用栈
- **瓶颈定位方法论**：
  - **CPU-bound**：特征是 GPU 空闲（SM occupancy 低），CPU 端 tokenization / scheduling / Python overhead 成为瓶颈
  - **GPU compute-bound**：特征是 SM occupancy 高、memory throughput 未饱和（如大 batch prefill）
  - **GPU memory-bound**：特征是 SM occupancy 低但 HBM 带宽接近峰值（如小 batch decode）
  - Roofline model 简化分析：arithmetic intensity（FLOPs/byte）vs 硬件 roofline，判断是算力受限还是带宽受限
  - H800 参数：FP16 算力 ~990 TFLOPS、HBM 带宽 ~3.35 TB/s、compute/bandwidth roofline 交叉点
- **vLLM 特有的 profiling 配置**：
  - `ProfilerConfig`（`vllm/config/profiler.py`）：内置 profiling 支持
  - `--collect-detailed-traces` 参数
  - `VLLM_TORCH_PROFILER_DIR` 环境变量
- **实践**：
  - 对 Qwen2.5-7B TP=1 的单步 forward 做 profiling，标注各组件耗时占比
  - Prefill vs Decode 的 kernel 分布对比：Prefill 以 GEMM 为主（compute-bound），Decode 以 attention 为主（memory-bound）
  - 标注 Qwen2.5-7B 单步 forward 的耗时分解：Attention XX%、MLP XX%、LayerNorm XX%、其他 XX%
- **代码路径**：
  - `vllm/config/profiler.py` — profiling 配置
  - `vllm/v1/metrics/perf.py` — ComponentMetrics 体系（Attention / FFN / Unembed 各组件的计时）

### Day 2: 分布式 Profiling + 瓶颈→决策映射框架 ✅

> Day 1 建立了单卡 profiling 能力，Day 2 扩展到分布式场景，并构建优化决策的核心思维工具。

- **分布式场景 Profiling**：
  - **TP 通信开销归因**：
    - TP=1/2/4 分别 profiling，观察 AllReduce 占总延迟的比例随 TP size 的增长曲线
    - 区分 AllReduce 实际通信时间 vs kernel launch + 同步等待时间
    - 用 nsys 的 NCCL 通信 timeline 标注通信与计算的 overlap 程度
    - 实践：Qwen2.5-7B TP=4 的 decode 步 profiling，标注 AllReduce 在总延迟中的占比
  - **MoE 特有 Profiling**：
    - FusedMoE kernel 内部分解：Token Permutation（sort/scatter）时间 vs Grouped GEMM 时间
    - Expert 负载不均时的 GPU idle 分析：热门 expert 对应的 GEMM tile 大、冷门 expert 小 → SM 利用率不均
    - 实践：Mixtral-8x7B 的 MoE 层 profiling，标注 permute 占比和 GEMM 占比
  - **Speculative Decoding Overhead Profiling**：
    - Drafter forward 时间 vs Verifier forward 时间 vs 验证+回退开销
    - Acceptance rate 低时 overhead 分析：draft 白算 + 重新 sample 的总代价
    - 实践（如有 DSpark drafter）：标注 draft + verify 各占一步的比例
- **瓶颈→决策映射表**（本 Day 核心交付物）：

  > 构建一张结构化的映射表，作为 Day 3-5 实战的前置决策工具。

  | 瓶颈特征（观测指标） | 瓶颈判定 | 优化方向 | 涉及 Week |
  |---------------------|---------|---------|----------|
  | Decode TPOT 高、SM 占用低、HBM 带宽接近峰值 | Memory-bound（decode） | FP8 KV Cache 减少读取量；增大 batch 提高计算密度；Speculative Decoding 减少 decode 步数 | W7/W5 |
  | Prefill TTFT 高、GEMM kernel 占比 >80%、SM 占用高 | Compute-bound（prefill） | FP8 量化加速 GEMM；Chunked Prefill 缓解排队 | W7/W5 |
  | AllReduce 占比 >15-20% | Communication-bound（TP） | 减小 TP size + 加 DP；用 custom AllReduce（NVLink 优化）；考虑 PP 替代部分 TP | W4 |
  | waiting 队列堆积、KV Cache 使用率 >90%、preemption 频繁 | Memory-capacity-bound | 增大 gpu_memory_utilization；FP8 KV Cache；prefix caching 提高复用；减小 max_num_seqs | W9/W7/W5 |
  | TTFT P99 远高于 P50（但 GPU 未饱和） | Scheduling-bound | 开启 Chunked Prefill；调小 max_num_batched_tokens 限制单次 prefill 阻塞 | W5/W9 |
  | MoE permute 占比 >30%、expert GEMM 时间方差大 | Load-imbalance-bound（MoE） | EPLB 负载均衡；增大 batch 平摊 permute 开销；EP 替代 TP 减少每卡 expert 数量 | W6/W8 |
  | Drafter overhead > acceptance rate 收益 | Spec-decode overhead | 减少 num_speculative_tokens；Dynamic SD Schedule 根据 batch size 调整；关闭投机 | W5/W8 |

  - **使用方法**：实战中先 profiling → 对照表判定瓶颈类型 → 选择对应优化方向 → 实施并验证
  - **注意**：多个瓶颈可能同时存在，需按优先级逐个解决；解决一个后瓶颈可能转移到另一个

- **交付物**：
  - `learning/week10-optimization/bottleneck_decision_map.md`：瓶颈→决策映射表（含每种瓶颈的 profiling 判定标准和对应优化项）
  - `learning/week10-optimization/distributed_profiling_notes.md`：分布式 profiling 实验笔记

### Day 3: 场景 A — Dense 模型端到端优化（Qwen2.5-7B）

> 综合运用 Week 1-9 全部知识，完成 Dense 模型从基线到全面优化的完整闭环。

- **基线建立**：
  - **部署配置**：Qwen2.5-7B，TP=1（单卡基线），默认参数
  - 跑 `vllm bench serve`（ShareGPT 数据集，request_rate 逐步增大），记录：
    - TTFT P50/P95/P99、TPOT P50/P95/P99、Throughput（tokens/s）、QPS 上限（Goodput 开始下降的拐点）
  - 用 Week 9 Day 3 学的 `/metrics` 端点观测基线运行状态：`gpu_cache_usage_perc`、`num_requests_running`、`num_preemptions_total`
  - 用 Day 1-2 的 profiling 工具标注基线的耗时分解，对照瓶颈→决策映射表判定初始瓶颈类型

- **逐步优化**（每次只变一个，量化边际收益）：
  1. **TP 扩展**：TP=1 → TP=2 → TP=4，观察吞吐提升和通信开销增长，找到当前模型的 TP sweet spot
  2. **Prefix Caching ON**：在 TP sweet spot 上开启，测量 TTFT 改善和 cache 命中率
  3. **Chunked Prefill ON**（`max_num_batched_tokens=2048`）：测量 P99 ITL 改善 vs TTFT 代价
  4. **参数调优**：应用 Week 9 Day 4 的参数扫描结论（`max_num_seqs` / `gpu_memory_utilization` sweet spot）
  5. **FP8 在线量化**（`--quantization fp8_per_tensor`）：测量吞吐提升，抽样检查输出一致性
  6. **FP8 KV Cache**（`--kv-cache-dtype fp8_e4m3`）：测量高并发下的容量提升效果
  7. **Speculative Decoding**（如有 drafter 模型）：测量 TPOT 改善和 acceptance rate
  - 每步记录完整 benchmark 数据，对照基线计算提升百分比

- **组合优化 + 归因**：
  - 叠加所有正收益的优化，测量综合收益
  - **归因分析**：每个优化的收益来源（减少 prefill 阻塞 / 增大 KV Cache 容量 / 减少 forward 时间 / 减少 decode 步数），标注在什么条件下该收益可能消失或反转
  - 关注组合时的交互效应：例如 FP8 量化 + TP=4 的组合收益是否等于各自收益之和？

- **交付物**：
  - `learning/week10-optimization/dense_baseline.md`：基线数据 + profiling 耗时分解 + 初始瓶颈判定
  - `learning/week10-optimization/dense_optimization_log.md`：逐步优化的完整数据记录（每步的配置、指标、对比基线的 delta）

### Day 4: 场景 B — MoE 模型端到端优化（Mixtral-8x7B）

> 验证前 9 周 MoE 相关能力（Week 6 FusedMoE + Week 8 EPLB/EP）在实际部署中的闭环。

- **基线建立**：
  - **部署配置**：Mixtral-8x7B-Instruct-v0.1，TP=4（模型约 95GB，需多卡），默认参数
  - 跑 `vllm bench serve`（ShareGPT 数据集），记录基线指标（同 Day 3）
  - Profiling：用 Day 2 的 MoE profiling 方法，标注 FusedMoE 层的 permute/GEMM 占比和 expert 负载分布
  - 对照瓶颈→决策映射表判定：是 communication-bound（AllReduce 占比高）还是 load-imbalance-bound（expert 不均）？

- **TP vs EP 决策验证**：
  - **配置 1**：TP=4（所有 expert 复制到每卡）
  - **配置 2**：TP=2, EP=2（expert 分散，MoE 层 All-to-All 通信）
  - **配置 3**：EP=4（纯 EP，Attention 层 DP + MoE 层 EP）
  - 对比三种配置的 Throughput、TPOT、通信开销占比
  - 验证 Week 4 / Week 6 / Week 8 的理论预测：TP 适合小 expert 数量（通信少但冗余高），EP 适合大 expert 数量（无冗余但 All-to-All 开销）
  - 标注当前模型（8 experts）的最优选择和原因

- **MoE 特有优化**：
  - **EPLB 开启**：观察 expert 负载方差变化（如果当前 vLLM 版本支持）
  - **gpu_memory_utilization 调优**：MoE 模型权重显存大，KV Cache 预算相对少，需更精细平衡
  - **FP8 量化**：MoE 模型量化收益可能更大（expert 权重是显存大头）
  - **max_num_seqs 调优**：MoE decode 计算量比 Dense 大（top-K expert GEMM），对并发的容忍度不同

- **组合优化 + 归因**：
  - 叠加最优并行策略 + 量化 + 参数调优，测量综合收益
  - **与 Dense 对比**：相同硬件下，MoE 模型 vs Dense 模型的吞吐/延迟/成本效率差异
  - **归因分析**：MoE 模型的瓶颈转移路径（初始可能是 memory-capacity-bound → 量化后变为 communication-bound → 调整并行策略后变为 compute-bound）

- **交付物**：
  - `learning/week10-optimization/moe_baseline.md`：基线数据 + MoE profiling 分解 + expert 负载分布
  - `learning/week10-optimization/moe_optimization_log.md`：TP/EP 对比 + 逐步优化数据记录
  - `learning/week10-optimization/dense_vs_moe_comparison.md`：两种架构的横向对比总结

### Day 5: 优化报告撰写 + 成本优化分析

> 将 Day 3-4 的实验数据提炼为可交付的优化报告，并补充成本维度分析。

- **优化报告**（核心交付物）：
  - **报告结构**：
    1. **执行摘要**：一页总结关键发现和推荐配置
    2. **硬件与模型概况**：8 × H800、Qwen2.5-7B（Dense）、Mixtral-8x7B（MoE）
    3. **场景 A（Dense）优化详情**：
       - 基线指标
       - TP 扩展收益曲线（TP=1/2/4）
       - 各优化项边际收益表格
       - 组合优化最终指标
       - 瓶颈转移路径
    4. **场景 B（MoE）优化详情**：
       - 基线指标
       - TP vs EP 决策数据与结论
       - MoE 特有优化收益
       - 组合优化最终指标
       - 瓶颈转移路径
    5. **双场景横向对比**：Dense vs MoE 在同硬件下的效率差异
    6. **归因分析总结**：每个优化项的收益来源 + 适用条件 + 反转条件
    7. **推荐的生产配置模板**：分 Dense / MoE / 高吞吐 / 低延迟 四类
  - 报告路径：`learning/week10-optimization/optimization_report.md`

- **成本优化分析**：
  - **GPU 机型对比**（纸面计算）：
    - A100 80GB vs H100 80GB vs H800 80GB：算力、带宽、价格（公开云价格或租赁价格）
    - 推理卡（L40S、A10G）vs 训练卡：适用场景差异
    - 对于给定模型和 QPS 要求，推算不同机型的 TCO（Total Cost of Ownership）
  - **Right-sizing**：
    - 根据 Day 3-4 实测数据反推：当前 8 卡配置的实际利用率
    - 是否存在过度配置？能否用更少的卡达到 SLO？
    - TP=4 vs TP=2+DP=2 的吞吐/成本对比（Day 3 已有数据，此处量化成本差异）
  - **量化的成本节省**：
    - FP8 vs FP16：同等吞吐下需要更少的卡 → 每月节省多少
    - INT4 vs FP16：内存减半但精度代价，适用场景（对话 vs 代码生成 vs 摘要）
  - **MoE 的成本效率**：
    - 同等质量下 MoE 需要的卡数 vs Dense 需要的卡数
    - EP 策略对成本的影响：减少冗余权重 → 更少的总显存需求
  - **交付物**：`learning/week10-optimization/cost_analysis.md`

### Day 6: 新模型上线评估演练

> 验证"全链路能力"的最终环节：拿到一个没部署过的模型配置，能快速走完评估到方案选择的全流程。

- **练习 1：DeepSeek-V3 部署方案纸面推演**（不需要实际模型权重，纯参数推导）：
  - **Step 1 — 架构分析**：
    - 查阅公开信息：671B 参数、MoE（256 experts, top-8 activated）、MLA attention、128K context
    - 判定关键特征：超大 MoE + MLA（KV Cache 极小）+ 超长上下文
  - **Step 2 — 资源估算**：
    - 权重显存：671B × 2 bytes（FP16）= 1342 GB；FP8 = 671 GB
    - 每请求 KV Cache（MLA 压缩后）：推算 per-token KV 大小，计算 128K context 的 KV 需求
    - 最少卡数：权重 / 单卡显存（H800 80GB）→ FP16 至少 17 卡，FP8 至少 9 卡
  - **Step 3 — 并行策略选择**：
    - TP vs EP 决策：256 experts → EP 更适合（TP 会让每卡复制全部 expert，显存爆炸）
    - EP=8（单机 8 卡）下每卡 32 个 expert，Attention 层 DP
    - 单机 8 卡 FP8 能否装下？671/8 ≈ 84 GB > 80 GB → 需要 16 卡或更激进量化（INT4/MXFP4）
    - 推导可行的最小配置：如 16 卡 EP=16，或 8 卡 + INT4 expert 权重
  - **Step 4 — 性能预估**：
    - MLA 的 KV Cache 优势：每请求 KV 远小于标准 MHA → 支持更高并发
    - 256 expert 的 All-to-All 通信量预估：对比 Mixtral 8 expert 场景的通信开销放大倍数
    - EPLB 的必要性：256 expert 的负载不均风险远高于 8 expert
  - **Step 5 — 产出推荐方案**：一页纸的部署方案建议（含并行策略、量化选择、关键参数、风险项）

- **练习 2：已有模型的快速评估 SOP 提炼**：
  - 基于 Day 3-4 的实战经验和练习 1 的推演，提炼一套通用的"新模型上线评估 SOP"：
    1. 读 `config.json`：提取 `num_hidden_layers`、`hidden_size`、`num_attention_heads`、`num_key_value_heads`、`num_experts`、`num_experts_per_tok`、`max_position_embeddings`
    2. 判定架构类型：Dense / MoE / MLA / Sliding Window → 查阅 Week 3 Day 7 总结
    3. 资源估算公式：权重显存 = params × dtype_bytes；KV Cache = 公式（Week 8 Day 6 已推导）
    4. 并行策略初选：查阅 Week 8 Day 7 部署决策树
    5. 基线部署 + profiling → 对照 Day 2 瓶颈→决策映射表 → 逐步优化
  - **交付物**：`learning/week10-optimization/model_onboarding_sop.md`

- **⚠️ 注意**：练习 1 是纸面推演，不实际部署 DeepSeek-V3（8 卡 H800 装不下）。目的是训练"架构分析 → 资源估算 → 方案选择"的全链路思维。

### Day 7: 10 周学习全景回顾 + 自评

- **全景回顾**：
  - 按阶段梳理核心知识点：
    - 阶段 1（Week 1-2）：KV Cache 管理 + Continuous Batching
    - 阶段 2（Week 3-6）：推理引擎 + 分布式 + 高级调度 + MoE
    - 阶段 3（Week 7-10）：量化/LoRA + 生产分布式 + 服务调优 + Profiling/实战
  - 标记每个主题的掌握程度：**熟练**（能独立实操+归因）/ **理解**（能解释原理+看懂代码）/ **了解**（知道概念和适用场景）
  - 标记遗留问题和后续深入方向
- **自评检查**：
  - 逐项对照评估标准（本文末尾的初级/中级/高级 checklist）
  - 诚实标注每项的实际达成情况
  - 对未达标项给出补强计划
- **交付物汇总**：
  - 10 份周 SUMMARY.md
  - Week 8 部署决策文档
  - Week 9 参数调优手册 + 故障诊断手册
  - Week 10 优化报告（双场景）+ 成本分析 + 瓶颈→决策映射表 + 新模型上线 SOP
- **专家级能力方向标注**（为后续持续成长指引）：
  - 基于 10 周学习中发现的兴趣点和能力差距，选择 2-3 个专家级方向作为后续深入目标
  - 例如：kernel 优化（FlashInfer 贡献）/ 调度策略设计 / MoE 路由优化 / vLLM 社区贡献
- 编写 Week 10 SUMMARY.md + 全局总结

---

# 评估标准

## 初级（Week 1-2 完成）
- [ ] 能解释 PagedAttention 的核心思想（虚拟内存式 KV Cache 管理）
- [ ] 理解 Block Table 的映射机制和 Block Pool 的管理
- [ ] 能追踪 KV Cache 分配和释放的完整流程
- [ ] 理解 Continuous Batching 和 Scheduler 的调度策略
- [ ] 能区分 Prefill 和 Decode 两阶段的差异
- [ ] 理解请求生命周期与资源释放链路

## 中级（Week 3-6 完成）
- [ ] 能画出从 HTTP 请求到 token 输出的端到端架构图
- [ ] 理解 ModelRunner 的输入构造与 execute_model() 流程
- [ ] 理解 Sampling 层实现（Top-K/P、structured output）
- [ ] 理解 CUDA Graph 的 capture/replay 机制
- [ ] 能配置和使用多卡推理（TP/PP）
- [ ] 理解权重分片机制（ColumnParallel/RowParallel）
- [ ] 理解 CP 原理与适用场景
- [ ] 掌握 NCCL 通信原语与多节点网络配置
- [ ] 理解 Chunked Prefill、Prefix Caching、Speculative Decoding
- [ ] 理解 MoE 推理特性和 FusedMoE kernel 实现
- [ ] 能选择 TP vs EP 部署 MoE 模型

## 高级（Week 7-10 完成）
- [ ] 掌握量化技术（FP8/INT8/INT4/MXFP4）的原理与部署
- [ ] 理解 vLLM 量化注册体系（QuantizationConfig → 层替换完整链路）
- [ ] 掌握离线量化与在线量化（Online Quantization）的使用与区别
- [ ] 掌握 FP8 KV Cache 配置与精度评估
- [ ] 理解量化与 TP 分片、MoE（experts_int8）的交互
- [ ] 掌握 LoRA adapter 多租户推理（含 Scheduler 约束与 Worker 生命周期）
- [ ] 掌握量化精度评估方法（perplexity、输出一致性）
- [ ] 掌握 PD 分离架构完整实现（KV Connector、Scheduler 协作）
- [ ] 掌握 EPLB 负载均衡原理与调优
- [ ] 理解 KV Cache Offloading 策略
- [ ] 能根据模型特征和业务场景选择分布式方案
- [ ] 理解 AsyncLLMEngine 异步架构与 Graceful Shutdown 实现
- [ ] 掌握 VllmConfig 核心参数的交互影响，能系统性调优
- [ ] 掌握 Metrics 监控体系（PerfStats / Prometheus），能从指标定位瓶颈
- [ ] 掌握 Benchmark 工具链与方法论（warmup、统计显著性、Goodput）
- [ ] 能诊断常见故障（OOM、Scheduler 饥饿、吞吐突降）并给出解决方案
- [ ] 掌握 profiling 工具链（torch.profiler / nsys），覆盖单卡和分布式场景（TP 通信归因、MoE profiling）
- [ ] 掌握"瓶颈特征→优化决策"的结构化映射框架，能从 profiling 数据推导优化方向
- [ ] 能完成 Dense + MoE 双场景端到端部署优化并撰写优化报告（含归因分析与瓶颈转移路径）
- [ ] 理解成本优化方法与 TCO 分析
- [ ] 能对未部署过的新模型完成全链路评估（架构分析→资源估算→并行策略选择→部署方案）

## 专家级（额外）
- [ ] 能修改和优化底层 attention kernel（FlashAttention/FlashInfer）
- [ ] 能设计新的调度策略并验证效果
- [ ] 能实现新的 KV Cache 压缩方法
- [ ] 能优化 FusedMoE kernel 或设计新的 expert routing 策略
- [ ] 能设计分布式推理策略（异构推理、弹性扩缩容）
- [ ] 能贡献代码到 vLLM 项目
- [ ] 能在论文级别理解和创新推理系统技术

---

# 学习资源

## 必读论文
1. **vLLM**: Efficient Memory Management for Large Language Model Serving with PagedAttention
   - https://arxiv.org/abs/2309.06180
2. **FlashAttention**: Fast and Memory-Efficient Exact Attention with IO-Awareness
   - https://arxiv.org/abs/2205.14135
3. **FlashAttention-2**: Faster Attention with Better Parallelism and Work Partitioning
   - https://arxiv.org/abs/2307.08691
4. **Orca**: A Distributed Serving System for Transformer-Based Generative Models
   - https://www.usenix.org/system/files/osdi22-yu.pdf
5. **Megatron-LM**: Training Multi-Billion Parameter Language Models Using Model Parallelism
   - https://arxiv.org/abs/1909.08053（TP/PP 原理）
6. **DistServe**: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving
   - https://arxiv.org/abs/2401.09670（PD 分离）

## 推荐博客
1. **vLLM Blog**: https://blog.vllm.ai/
2. **Anyscale Continuous Batching**: https://www.anyscale.com/blog/continuous-batching-llm-inference
3. **FlashInfer Documentation**: https://docs.flashinfer.ai/

## 工具链

### Profiling 工具
```bash
# torch.profiler
python -c "import torch; ..."  # 生成 trace.json → chrome://tracing

# Nsight Systems
nsys profile -o output python script.py

# Nsight Compute
ncu --set full -o output python script.py
```

### 监控工具
```bash
watch -n 1 nvidia-smi        # GPU 使用情况
nvidia-smi dmon -s u         # GPU 内存监控
nvtop                        # 交互式 GPU 监控
```

### Benchmark 工具
```bash
# vLLM 官方 benchmark
python benchmarks/benchmark_serving.py \
    --model <model> --num-prompts 1000 \
    --request-rate 10 --dataset-name sharegpt

# NCCL 通信测试
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2
```

---

# 每周检查点

## 每周五完成
1. **周总结文档**（`SUMMARY.md`）：核心收获、技术要点、遗留问题、下周计划
2. **代码实验验证**：至少 2-3 个可运行的脚本
3. **性能数据记录**：Throughput、Latency 等指标对比
4. **疑难问题列表**

## 每周日
1. 回顾本周学习，检查理解深度
2. 预习下周内容，准备问题清单
3. 调整学习节奏

---

# 学习方法建议

1. **理论与实践结合**：先理论 → 后实践 → 再反思
2. **循序渐进**：按周顺序推进，遇到困难先标记继续
3. **主动输出**：写文档、画图、写代码、假想讲解
4. **善用工具**：grep 搜索、IDE 追踪、Git blame、GitHub issues
5. **保持节奏**：每天 2-3 小时，固定时间，适当休息

---

# 附录：常用命令速查

## vLLM 基本使用
```bash
cd /home/lvzheng/vllm/vllm
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
.venv/bin/python -m pytest tests/core/test_block_manager.py -v
pre-commit run --all-files
```

## 常用 grep 命令
```bash
grep -rn "def allocate_slots" vllm/
grep -rn "class.*Scheduler" vllm/
grep -r "keyword" vllm/ --include="*.py"
```

## Git 相关
```bash
git log --follow vllm/v1/core/kv_cache_manager.py
git blame vllm/v1/core/kv_cache_manager.py -L 445,552
git show <commit_hash>
```

---

# 更新日志

## 2026-08-20（第三次更新 — 计划重构）
- ✅ **重新审视并调整全局计划结构**
  - 目标明确：大模型推理优化部署专家（单机单卡 + 大规模分布式）
  - 阶段划分调整：阶段2 Week 3-6（引擎+优化），阶段3 Week 7-10（生产级）
- ✅ **Week 3 重构**：精简为 ModelRunner 执行流程
  - Day 1 新增：端到端全景（建立框架概念，不深入 API Server）
  - 去除 API Server 深入内容（移至 Week 9）
  - 新增 Sampling 层学习
- ✅ **Week 4 加厚**：分布式推理核心周
  - 合并原 Week 7 的权重分片/加载内容
  - 新增多节点网络配置（IB/RoCE、NCCL env tuning）
  - PP 补充 micro-batch 调度 + bubble 分析
- ✅ **Week 7 量化提前**（原 Week 9）
  - 在 MoE 之后、PD 分离之前，FP8 知识作为 Week 8 前置
- ✅ **Week 9 新增：服务化与生产部署**
  - API Server 生产配置（限流、健康检查、背压）
  - 多模态推理（原 Week 7 内容）
  - 多实例部署（负载均衡、routing、auto-scaling）— 新增
  - 部署基础设施（容器化、K8s、滚动更新）— 新增
  - 基准测试方法论 — 新增
- ✅ **Week 10 补充**
  - 新增：容错机制（GPU 故障恢复、NCCL watchdog）
  - 新增：成本优化（机型选择、TCO 分析）
- ✅ **删除两个补充模块**：内容已全部融入主计划
- ✅ **更新评估标准**：反映新的周次安排和技能要求
- ✅ **新增论文**：Megatron-LM（TP/PP）、DistServe（PD分离）
- ✅ **新增 Benchmark 工具**速查

## 2026-09-17
- ✅ Week 8 新增 Day 5：生产级投机解码（DFlash + DSpark 半自回归框架）
- ✅ 原 Day 5-6（KV Cache Offloading + 长上下文）压缩为 Day 6
- ✅ Day 7 场景分析补充投机解码方案选择决策项
- ✅ 学习目标新增第 4 条：DFlash/DSpark

## 2026-09（Week 8 内容修订）
- ✅ **学习目标修订**：细化第 2、5、6 条，明确 Connector 选型、量化分析能力、决策需要量化依据
- ✅ **新增 Day 0**：模型确认（列出各 Day 对模型的实际需求，不强制下载）
- ✅ **Day 1-2 PD 分离深度补充**：
  - 新增 KV Connector 多实现对比（simple/flexkv/lmcache/nixl/mooncake）及选型依据
  - 新增 Prefill 端与 Decode 端 Scheduler 行为差异的完整追踪任务
  - 实践任务增加 KV 传输延迟的理论推算（补偿单机无网络瓶颈的局限）
  - 补充代码行号定位
- ✅ **Day 3-4 EPLB 深度补充**：
  - 新增 EPLB policy 抽象设计（abstract.py / default.py）
  - 新增重均衡触发机制、频率、对在线推理影响的学习要求
  - 补充 EPLB 子模块文件说明
  - 弹性 EP 标注代码成熟度注意事项
- ✅ **Day 5 事实修正**：
  - Confidence-Scheduled Verification → Dynamic SD Schedule（按 batch size 映射 draft 长度）
  - 修正描述：从"运行时基于 logits 置信度"改为"throughput-oriented 的静态映射表"
  - DFlash 补充两层架构说明（proposer 编排层 + GPU speculator 层）
  - 代码路径新增：`vllm/v1/spec_decode/dflash.py`（proposer）、`vllm/v1/spec_decode/dynamic/`
  - 实践新增 Dynamic SD Schedule 测试
- ✅ **Day 6 重写**：
  - 从概念罗列改为决策模型：offload vs 重算的 break-even 量化分析
  - 新增两套 Offloading 实现的区分（simple_kv_offload vs offloading_connector）
  - 长上下文从空洞描述改为 KV Cache 内存规划量化练习（给定模型参数推导并发数）
  - EC Connector 路径修正：`vllm/distributed/kv_transfer/` → `vllm/distributed/ec_transfer/`
  - 补充完整代码路径
- ✅ **Day 7 强化**：
  - 新增归因分析要求（每个实验分析瓶颈来源，标注结论反转条件）
  - 标注单机实验局限性
  - 明确交付物为可复用的部署决策文档（含决策树、边界条件、实验数据）

## 2026-10（Week 10 重写）
- ✅ **Day 1 拆分为单卡 Profiling**：保留原 torch.profiler / nsys / Roofline 内容，聚焦单卡场景
- ✅ **Day 2 新增分布式 Profiling + 瓶颈→决策映射框架**：
  - 新增 TP 通信开销归因（TP=1/2/4 对比 AllReduce 占比曲线）
  - 新增 MoE 特有 profiling（FusedMoE permute/GEMM 分解、expert 负载不均 idle 分析）
  - 新增 Speculative Decoding overhead profiling
  - 新增"瓶颈特征→优化决策"结构化映射表（7 种瓶颈类型 × 优化方向 × 关联 Week）
- ✅ **Day 3-4 拆为双场景实战**（原 Day 3-5 单场景 → 双场景）：
  - Day 3：场景 A（Dense，Qwen2.5-7B）— 含 TP=1/2/4 扩展 + 逐步优化 + 组合优化
  - Day 4：场景 B（MoE，Mixtral-8x7B）— 含 TP vs EP 决策验证 + EPLB + MoE 特有调优
  - 新增 Dense vs MoE 横向对比交付物
- ✅ **Day 5 合并报告撰写 + 成本分析**（原 Day 5 报告 + Day 6 成本 → 合并）：
  - 报告结构升级：双场景分列 + 横向对比 + 瓶颈转移路径
  - 成本分析新增 MoE 成本效率分析
- ✅ **Day 6 新增新模型上线评估演练**：
  - 练习 1：DeepSeek-V3 纸面部署推演（671B MoE + MLA + 128K context）
  - 练习 2：通用新模型上线评估 SOP 提炼
  - 训练"架构分析→资源估算→方案选择"的全链路思维
- ✅ **Day 7 强化**：
  - 新增掌握程度三级标记（熟练/理解/了解）
  - 新增专家级方向选择
  - 交付物新增瓶颈→决策映射表 + 新模型上线 SOP
- ✅ **评估标准更新**：
  - profiling 项新增"覆盖分布式场景"要求
  - 新增"瓶颈→决策映射框架"项
  - 端到端优化项改为"双场景"要求
  - 新增"新模型全链路评估"项

## 2026-10（Week 9-10 重构）
- ✅ **Week 9 重构**：从"服务化概念普及"转向"服务配置调优与端到端 Benchmark"
  - Day 1：新增 AsyncLLMEngine 异步架构深入（generate() 异步流、graceful shutdown、进程间通信）
  - Day 2：新增 VllmConfig 参数体系（核心参数三角 + 场景化配置模板）
  - Day 3：Metrics 监控从 Week 10 前移（PerfStats / Prometheus / KVCacheMetrics）
  - Day 4：新增参数扫描实验（max_num_seqs / max_num_batched_tokens / gpu_memory_utilization 系统性扫描）
  - Day 5：Benchmark 工具链与方法论（vllm bench 工具精读 + 常见陷阱 + Goodput 概念）
  - Day 6：故障模拟与诊断从 Week 10 前移（4 种故障场景 + 诊断决策树）
  - Day 7：部署架构概览（VLM 部署要点 + Executor 多后端精读 + 容器化/K8s 基础认知）+ 总结
  - 删除：K8s/容器化实操（单机无法实践，改为理论认知：Dockerfile 结构、health probe、graceful shutdown 与 K8s 配合）
  - 删除：Ray 多节点实践（改为代码阅读，MultiprocExecutor 重点精读 + RayExecutor 概览）
  - 删除：VLM 2天深入（压缩为部署差异要点）
- ✅ **Week 10 重构**：从"系统优化与诊断"转向"Profiling 深入 + 端到端实战"
  - Day 1-2：Profiling 工具链（torch.profiler / nsys / Roofline），含 H800 参数和 compute vs memory bound 判定
  - Day 3-5：端到端实战项目（选型 → 基线 → 逐步优化 → 归因分析 → 报告），3天完整闭环
  - Day 6：成本优化（机型对比 + right-sizing + 量化成本收益）
  - Day 7：10 周学习全景回顾 + 自评
  - 删除：Metrics 和故障诊断（已移至 Week 9）、生产环境调优（与 Week 9 Day 4 合并）
- ✅ **评估标准更新**：反映 Week 9-10 修订后的实际技能要求
- ✅ **约束适配**：全部内容限定为单机 8 卡 H800 可实践范围，多节点内容降为代码阅读或理论分析

## 2026-08-10（第二次更新）
- ✅ 补充 Week 3 Day 4-5：API Server 与多用户并发架构

## 2026-08-10（第一次更新）
- ✅ 完整重写 Week 1-2：适配 v1 架构
- ✅ 框架化 Week 3-8
- ✅ 更新学习资源和工具链

---

## TODO
- [ ] 根据实际学习进度，细化每周的详细实验脚本
- [ ] Week 4 补充具体的多卡实验配置
- [ ] 根据实际代码验证所有流程图和伪代码

---

**祝学习顺利！成为大模型推理优化部署专家！🚀**
