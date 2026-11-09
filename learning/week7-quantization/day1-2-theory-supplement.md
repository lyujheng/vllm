# Week 7 Day 1-2 理论补充

## 一、校准（Calibration）

### 什么是校准

离线量化需要确定 scale（缩放因子），而 scale 的计算依赖于数据的实际分布。校准就是：**用一小批代表性数据跑一次前向推理，统计权重和激活的数值分布，据此确定最优的 scale。**

```
校准流程:
原始 FP16 模型 + 校准数据集（通常 128~512 条样本）
    → 逐层前向推理
    → 统计每层的权重/激活分布
    → 计算 scale
    → 导出量化模型（权重 + scale）
```

### 各方案的校准方式

| 方案 | 是否需要校准 | 校准目的 | 校准数据量 |
|------|------------|---------|-----------|
| AWQ | 是 | 统计激活值，找重要通道 | 128~256 条 |
| GPTQ | 是 | 计算 Hessian 矩阵 H = XᵀX，做逐列误差补偿 | 128~256 条 |
| SmoothQuant (INT8) | 是 | 统计激活值范围，计算平滑因子 | 128~512 条 |
| FP8 static (W8A8) | 是 | 确定激活的固定 scale | 128~512 条 |
| FP8 dynamic (W8A8) | 否 | scale 每次 forward 当场算 | — |
| FP8 online (W8A16) | 否 | 权重加载时直接 cast | — |
| MXFP4 | 是 | 确定每个 block 的 E8M0 scale | 128~256 条 |

### AWQ 校准细节

AWQ 的核心观察：**不是所有权重通道同等重要，1% 的显著通道对模型精度影响极大。**

校准过程：
1. 跑校准数据，收集每层的激活值 X
2. 统计每个输入通道的激活均值：`channel_importance = mean(|X|, dim=batch)`
3. 重要通道（激活值大的）给更高的保护——乘以一个 per-channel 缩放因子 s
4. 权重对应除以 s，保持数学等价：`Y = X @ W = (X * s) @ (W / s)`
5. 缩放后权重的数值范围更均匀，量化误差更小

```
校准前:
  通道 0（不重要）: 激活值 ≈ 0.01，权重 = 3.5  → 量化误差影响小
  通道 5（重要）:   激活值 ≈ 50.0，权重 = 0.002 → 量化误差被放大 50 倍

校准后（对通道 5 乘以 s=10）:
  权重变成 0.002 * 10 = 0.02 → 在量化范围内分辨率更高
  激活变成 50.0 / 10 = 5.0   → 数学结果不变
```

### GPTQ 校准细节

GPTQ 基于 OBS（Optimal Brain Surgeon）理论，目标是**最小化量化引入的输出误差**。

校准过程：
1. 跑校准数据，收集每层的输入 X
2. 计算 Hessian 矩阵：`H = XᵀX`（描述输入的二阶统计信息）
3. 逐列量化权重，每量化一列，用 Hessian 的逆来补偿剩余列的误差

```
逐列量化伪代码:
for col in range(K):
    # 量化当前列
    w_quant = round(W[:, col] / scale) * scale
    error = W[:, col] - w_quant

    # 用 Hessian 逆把误差补偿到未量化的列
    W[:, col+1:] -= error @ H_inv[col, col+1:] / H_inv[col, col]
```

关键直觉：量化第 i 列产生的误差，可以通过微调第 i+1, i+2, ... 列来部分抵消。Hessian 矩阵告诉我们"调整哪些列、调整多少"最优。

### FP8 Static 校准

最简单的校准——只需要确定激活的范围：

```
1. 跑校准数据，记录每层激活的最大绝对值
2. scale = max(|activation|) / 448   (FP8 E4M3 最大值 448)
3. 保存 scale 到 checkpoint 的 input_scale 字段
```

推理时用这个固定 scale 量化激活，不需要每次 forward 重新计算。比 dynamic 快（省了一次 reduce_max），但如果实际输入分布偏离校准数据，精度可能下降。

---

## 二、量化敏感层分析

### 哪些层对量化敏感

不是模型的每一层都能安全量化。经验规律：

| 层类型 | 敏感度 | 原因 |
|--------|--------|------|
| Attention QKV 投影 | **高** | Q 和 K 的点积对数值精度敏感，微小误差被 softmax 放大 |
| Attention Output 投影 | 中 | 输出经过残差连接，有一定容错 |
| MLP Gate/Up 投影 | **低** | 激活函数（SiLU/GELU）对输入不太敏感 |
| MLP Down 投影 | 中 | 直接加到残差流，误差会累积 |
| Embedding | **高** | 离散查表，量化会改变 token 表示 |
| LM Head | **高** | 输出 logits，影响 token 选择概率 |
| LayerNorm/RMSNorm | **高** | 归一化参数量小但影响全局 |

### 敏感度直觉

核心原则：**误差会沿残差流累积**。

```
Transformer 的残差结构:
x → [Attention] → + → [MLP] → + → [Attention] → + → ...
       ↑              ↑              ↑
     残差连接        残差连接        残差连接

每层的量化误差 ε 都会加到残差流上:
x_out = x + f_quant(x) = x + f(x) + ε

经过 N 层后，累积误差 ≈ ε₁ + ε₂ + ... + εₙ
```

越靠近模型前端（embedding）和后端（lm_head）的层越敏感：
- 前端误差传播距离最长，影响所有后续层
- 后端直接决定输出 token 概率

### 实践中的应对

1. **混合精度量化**：敏感层保持 FP16，其余层量化
   ```bash
   # vLLM 的 ignore 机制
   --quantization-config '{"ignore": ["lm_head", "model.embed_tokens"]}'
   ```

2. **不同层用不同量化精度**：
   - Attention: W8A8 (INT8/FP8)
   - MLP: W4A16 (INT4)
   - Embedding/LM Head: FP16（不量化）

3. **AWQ/GPTQ 自动处理**：校准过程天然会给敏感通道更多保护

---

## 三、量化误差评估方法

### Perplexity（困惑度）对比

最常用的量化精度评估指标。Perplexity 衡量模型对文本的预测能力，越低越好。

```
评估流程:
1. 准备评估数据集（通常用 WikiText-2 或 C4）
2. 分别用 FP16 模型和量化模型计算 perplexity
3. 对比差距

参考标准（7B 模型，WikiText-2）:
  FP16:           5.68
  FP8 (W8A8):     5.69  (差距 < 0.02，几乎无损)
  INT8 (W8A8):    5.71  (差距 < 0.05，可接受)
  INT4 AWQ:       5.82  (差距 < 0.2，轻微损失)
  INT4 GPTQ:      5.79  (差距 < 0.15，轻微损失)
  MXFP4:          5.90  (差距 < 0.3，需要验证下游任务)
```

### Task Accuracy 抽样

Perplexity 是全局指标，可能掩盖特定任务的退化。需要在下游任务上验证：

```
常用 benchmark:
- MMLU (多选知识题)
- HumanEval (代码生成)
- GSM8K (数学推理)
- HellaSwag (常识推理)

如果某个任务 accuracy 下降 > 2%，说明量化对该能力有显著影响
```

### 输出一致性 Diff

最直观的方法——对比量化前后的输出差异：

```
方法:
1. 准备 100~500 条测试 prompt
2. FP16 模型生成参考输出
3. 量化模型生成对比输出
4. 计算一致性:
   - 完全匹配率 (exact match %)
   - Token 级别 overlap rate
   - 语义相似度 (embedding cosine similarity)

典型结果:
  FP8:  95%+ exact match（几乎一致）
  INT8: 85-90% exact match
  INT4: 60-75% exact match（表述不同但语义接近）
```

### 评估建议

```
快速验证: perplexity 对比（5 分钟）
  → 差距 < 0.1: 可以直接用
  → 差距 0.1~0.5: 需要下游任务验证
  → 差距 > 0.5: 量化方案可能不适合该模型

深度验证: task accuracy + 输出 diff（1-2 小时）
  → 在你关心的具体任务上测试
```

---

## 四、GGUF 混合精度格式

### 什么是 GGUF

GGUF（GPT-Generated Unified Format）是 llama.cpp 定义的模型格式，核心特点是**混合精度量化**——模型的不同层可以用不同的量化精度。

### 与 GPTQ/AWQ 的区别

| | GPTQ/AWQ | GGUF |
|--|---------|------|
| 量化粒度 | 整个模型统一一种精度 | 每层可以不同精度 |
| 典型配置 | 全部 INT4 | 重要层 Q8，其他层 Q4/Q2 |
| 运行时 | GPU（cuBLAS/Marlin） | CPU 为主，也支持 GPU offload |
| 生态 | HuggingFace + vLLM/TRT-LLM | llama.cpp + ollama |
| 命名规则 | 按方法命名（awq, gptq） | 按精度命名（Q4_K_M, Q5_K_S） |

### GGUF 量化类型命名

```
Q4_K_M:
  Q4  = 4-bit 量化
  K   = K-quant（使用 k-means 聚类优化量化）
  M   = Medium（中等精度，平衡速度和质量）

常见类型（精度从高到低）:
  Q8_0     — 8-bit，接近无损，体积大
  Q6_K     — 6-bit，高精度
  Q5_K_M   — 5-bit，推荐的高质量选项
  Q4_K_M   — 4-bit，最常用的平衡选项
  Q3_K_M   — 3-bit，明显精度损失
  Q2_K     — 2-bit，极端压缩，精度差

混合精度示例（Q4_K_M）:
  Attention 层: Q6 量化（敏感层用高精度）
  MLP 层:      Q4 量化（不敏感层用低精度）
  Embedding:   Q6 量化
```

### vLLM 对 GGUF 的支持

vLLM 支持加载 GGUF 格式模型，但主要定位是 GPU 推理。GGUF 在 vLLM 中会被转换为统一的内部表示，然后走正常的量化 kernel 路径。

```python
# vLLM 中的 GGUF 支持
# 代码: vllm/model_executor/layers/quantization/gguf.py
```

GGUF 在 CPU 推理（llama.cpp）生态里是主流格式，但在 GPU 推理（vLLM）生态里，GPTQ/AWQ/FP8 更常用。了解 GGUF 主要是为了理解混合精度的思路——这个思路在其他方案中也有体现（比如 `--quantization-config` 的 `ignore` 字段就是最简单的混合精度）。

---

## 五、总结对比

| 维度 | FP8 | INT8 (SmoothQuant) | INT4 (AWQ) | INT4 (GPTQ) | MXFP4 | GGUF |
|------|-----|-----------|------------|-------------|-------|------|
| 权重精度 | 8-bit | 8-bit | 4-bit | 4-bit | 4-bit | 混合 |
| 激活精度 | 8/16-bit | 8-bit | 16-bit | 16-bit | 16-bit | 16-bit |
| 需要校准 | static 需要 | 是 | 是 | 是 | 是 | 是 |
| 校准复杂度 | 低 | 中 | 中 | 高 | 低 | 中 |
| 精度损失 | 极小 | 小 | 小~中 | 小~中 | 中 | 取决于配置 |
| 压缩比 | 2x | 2x | 4x | 4x | 4x~8x | 2x~8x |
| GPU 支持 | H100/H800+ | 广泛 | 广泛 | 广泛 | Blackwell+ | GPU offload |
| Marlin 加速 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| vLLM 推荐场景 | H800 首选 | 通用 | 显存受限 | 显存受限 | MoE 模型 | 兼容性需求 |
