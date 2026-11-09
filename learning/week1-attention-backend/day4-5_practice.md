# Week 1 Day 4-5: 实践与总结

## 时间分配：3-4 小时

---

## 实践目标

通过运行实际推理代码，验证你在 Day 1-3 学到的概念：
- ✅ Attention Backend 的工作流程
- ✅ Block Table 的传递和使用
- ✅ Prefill 和 Decode 的分离
- ✅ KV Cache 的存储格式

---

## 实践 1：运行基础推理（30 分钟）

### 1.1 创建测试脚本

```bash
cat > learning/week1-attention-backend/test_inference.py << 'EOF'
"""Week 1 Day 4: 基础推理测试"""

from vllm import LLM, SamplingParams

# 使用小模型快速测试
# 如果模型已下载到本地，使用本地路径
llm = LLM(
    model="./models/opt-125m",  # 或 "facebook/opt-125m" 自动下载
    max_num_seqs=2,              # 允许 2 个并发请求
    block_size=16,               # 观察 block 分配
    max_model_len=512,           # 限制序列长度
)

# 测试场景 1：单个请求（观察 Prefill → Decode 过程）
print("=" * 60)
print("测试 1: 单个请求")
print("=" * 60)

prompts = ["Hello, my name is"]
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=10,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()

# 测试场景 2：多个并发请求（观察 Continuous Batching）
print("=" * 60)
print("测试 2: 并发请求")
print("=" * 60)

prompts = [
    "The capital of France is",
    "Python is a programming language that",
]

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Request {i}:")
    print(f"  Prompt: {output.prompt}")
    print(f"  Generated: {output.outputs[0].text}")
    print()

print("=" * 60)
print("推理完成")
print("=" * 60)
EOF
```

### 1.2 运行测试

```bash
cd /home/lvzheng/vllm/vllm

# 先下载模型（可选，如果已下载跳过）
# 方法 1：使用 huggingface-cli
huggingface-cli download facebook/opt-125m --local-dir ./models/opt-125m

# 方法 2：或使用 Python 下载
python3 << 'DOWNLOAD_EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "facebook/opt-125m"
local_path = "./models/opt-125m"
print(f"下载模型到 {local_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)
print("下载完成！")
DOWNLOAD_EOF

# 运行测试（使用系统 Python，不用虚拟环境）
python3 learning/week1-attention-backend/test_inference.py
```

### 1.3 观察输出

运行日志中会看到（简化）：
```
INFO: Initializing an LLM engine with config: ...
INFO: Using FlashInfer backend
INFO: KV cache shape: (num_blocks, num_kv_heads, block_size, 2*head_size)
INFO: # GPU blocks: 1234
...
测试 1: 单个请求
Prompt: Hello, my name is
Generated:  John. I'm a software engineer...
...
```

**关键观察**：
- 使用的 backend（FlashInfer / FlashAttention）
- GPU blocks 数量（物理 block 池大小）
- 推理速度（tokens/s）

---

## 实践 2：追踪 Block Table（可选，45 分钟）

### 2.1 添加日志到 forward()

临时修改 `vllm/v1/attention/backends/flashinfer.py`：

```python
# 在 forward() 开头（Line 1790 附近）添加：
def forward(self, ...):
    if attn_metadata is None:
        return output.fill_(0)
    
    # ===== 添加日志 =====
    import sys
    print(f"\n[Forward] num_decode_tokens={attn_metadata.num_decode_tokens}, "
          f"num_prefill_tokens={attn_metadata.num_prefill_tokens}",
          file=sys.stderr)
    
    if hasattr(attn_metadata, 'prefill') and attn_metadata.prefill:
        print(f"[Prefill] num_prefill_seqs={attn_metadata.prefill.num_prefill_seqs}",
              file=sys.stderr)
    
    if hasattr(attn_metadata, 'decode') and attn_metadata.decode:
        print(f"[Decode] num_decode_seqs={attn_metadata.decode.num_decode_seqs}",
              file=sys.stderr)
    # ====================
    
    # 原有代码继续...
```

### 2.2 重新运行测试

```bash
# 重新安装（因为修改了源码）
# 如果不使用虚拟环境，使用系统 Python
VLLM_USE_PRECOMPILED=1 pip3 install -e . --torch-backend=auto

# 运行测试，日志输出到文件
python3 learning/week1-attention-backend/test_inference.py 2> forward_log.txt

# 查看日志
cat forward_log.txt | grep -E "\[Forward\]|\[Prefill\]|\[Decode\]"
```

### 2.3 分析日志

你会看到类似：
```
[Forward] num_decode_tokens=0, num_prefill_tokens=5    ← 第一次：只有 Prefill
[Prefill] num_prefill_seqs=1

[Forward] num_decode_tokens=1, num_prefill_tokens=0    ← 后续：只有 Decode
[Decode] num_decode_seqs=1

[Forward] num_decode_tokens=1, num_prefill_tokens=0
[Decode] num_decode_seqs=1
...
```

**观察点**：
- 第一个 forward：只有 prefill（处理整个 prompt）
- 后续 forward：只有 decode（每次生成 1 个 token）
- 如果有并发请求，会看到 decode + prefill 混合的情况

---

## 实践 3：对比不同 Backend（可选，30 分钟）

### 3.1 测试 FlashAttention Backend

```python
# 修改 test_inference.py，添加 backend 参数
llm = LLM(
    model="facebook/opt-125m",
    max_num_seqs=2,
    attention_backend="FLASH_ATTN",  # 显式指定 FlashAttention
)
```

### 3.2 对比性能

```bash
# 测试 FlashInfer
time python3 learning/week1-attention-backend/test_inference.py

# 修改代码使用 FlashAttention 后再测试
time python3 learning/week1-attention-backend/test_inference.py
```

**观察点**：
- 初始化时间
- 推理速度（tokens/s）
- 日志中显示的 backend 名称

---

## 实践 4：Week 1 总结（60 分钟）

### 4.1 创建总结文档

```bash
cat > learning/week1-summary.md << 'EOF'
# Week 1 学习总结：Attention Backend

## 学习时间
- 开始：[填写日期]
- 完成：[填写日期]
- 总时长：约 12-15 小时

---

## 核心概念掌握

### 1. PagedAttention 机制

**问题**：传统 Attention 的 KV Cache 怎么存储？有什么问题？

**你的理解**：
[填写：连续存储 vs Paged 存储的区别]

**关键代码**：
```python
# KV Cache shape
kv_cache.shape = (num_blocks, num_kv_heads, block_size, 2 * head_size)
                                                        ^^^^^^^^^^^
                                                        K+V 交错存储
```

---

### 2. FlashInfer Backend 三层架构

**架构图**：
```
FlashInferBackend (配置层)
    ├─ get_kv_cache_shape()
    ├─ get_impl_cls()
    └─ get_builder_cls()
         ↓
FlashInferMetadataBuilder (准备层)
    └─ build() → 初始化 Wrapper，传入 Block Table
         ↓
FlashInferImpl (执行层)
    └─ forward() → 调用 wrapper.run()
```

**你的理解**：
[填写：为什么要分三层？各层职责是什么？]

---

### 3. Block Table 传递路径

**完整路径**：
```
Scheduler 构建 block_table_tensor
    ↓
CommonAttentionMetadata.block_table_tensor
    ↓
MetadataBuilder.build() 提取 block_table
    ↓
wrapper.plan(kv_indices=block_table_tensor)  ← 传给 Wrapper
    ↓
forward() → wrapper.run()
    ↓
CUDA Kernel 使用 block_table 读取 KV Cache
```

**你的理解**：
[填写：Block Table 是什么？为什么需要它？]

---

### 4. Prefill vs Decode

| 维度 | Prefill | Decode |
|------|---------|--------|
| Query Tokens | 多个（整个 prompt） | 1 个/请求 |
| Wrapper | BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper |
| 计算特点 | 计算密集（QK^T 矩阵大） | 访存密集（读长 KV Cache） |
| forward() 中位置 | query[num_decode_tokens:] | query[:num_decode_tokens] |

**你的理解**：
[填写：为什么要分开处理？]

---

### 5. Continuous Batching

**概念**：
不同请求可以处于不同阶段（有的在 Prefill，有的在 Decode），vLLM 把它们合并成一个 batch 处理。

**代码体现**：
```python
if num_prefill_tokens > 0:
    prefill_wrapper.run(query[num_decode_tokens:], ...)

if num_decode_tokens > 0:
    decode_wrapper.run(query[:num_decode_tokens], ...)
```

**实践观察**：
[填写：在日志中看到了什么？]

---

## 关键文件位置

| 文件 | 核心内容 | 行号 |
|------|---------|------|
| `vllm/v1/attention/backends/flashinfer.py` | Backend/Builder/Impl 三层架构 | 343, 620, 1530 |
| `vllm/v1/attention/backends/flashinfer.py` | forward() 方法 | 1681-2200 |
| `vllm/v1/cudagraph_dispatcher.py` | CUDA Graph padding 逻辑 | 72-91 |
| `vllm/config/compilation.py` | cudagraph_capture_sizes 配置 | 692-707 |

---

## 遗留问题（Week 2 解决）

1. Block Table 是怎么生成的？（Scheduler）
2. 物理 blocks 怎么分配和释放？（BlockManager）
3. slot_mapping 是什么？（KV Cache 写入）
4. 多个请求如何调度？（Scheduler）

---

## Week 1 成果检验

完成以下问题，检验你的理解：

### Q1: KV Cache 的 shape 是什么？为什么最后一维是 2*head_size？

**你的答案**：


### Q2: FlashInfer 的 Plan/Run 两阶段 API 的作用是什么？

**你的答案**：


### Q3: 同一个 batch 中可能同时有 Prefill 和 Decode 请求吗？如果有，forward() 如何处理？

**你的答案**：


### Q4: CUDA Graph padding 的目的是什么？padding 后如何保证计算正确性？

**你的答案**：


---

## 下一步：Week 2

重点内容：
- BlockManager：物理 block 的分配和释放
- Scheduler：请求调度和 batch 构建
- slot_mapping：新 token 的 KV 如何写入 Cache

预计时间：12-15 小时
EOF
```

### 4.2 填写总结文档

花时间完整填写上面的模板，用自己的话解释核心概念。

### 4.3 回顾关键代码

重新快速浏览以下代码位置，确认理解：
- [flashinfer.py:343](vllm/v1/attention/backends/flashinfer.py#L343) - FlashInferBackend
- [flashinfer.py:620](vllm/v1/attention/backends/flashinfer.py#L620) - MetadataBuilder
- [flashinfer.py:1681](vllm/v1/attention/backends/flashinfer.py#L1681) - forward()

---

## ✅ Day 4-5 完成标准

- [ ] 成功运行基础推理脚本
- [ ] 观察到 Prefill → Decode 的过程
- [ ] （可选）添加日志，看到 num_prefill_tokens 和 num_decode_tokens 的变化
- [ ] 完成 Week 1 总结文档
- [ ] 能用自己的话解释三层架构和 Block Table 传递

**预计时间**: 3-4 小时

---

## 下一步

完成后进入 **Week 2: KV Cache 管理与 Scheduler**。
