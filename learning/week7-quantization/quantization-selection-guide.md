# 量化方案选择指南

## 一、什么时候需要量化

### 需要量化的场景

**场景 1: 模型装不下**

```
70B 模型 FP16 需要 140GB 显存，单张 H800 (80GB) 放不下
  → 量化到 FP8:  140GB / 2 = 70GB，单卡能跑
  → 量化到 INT4: 140GB / 4 = 35GB，单卡能跑，还有大量空间给 KV Cache
```

**场景 2: 要更大吞吐**

```
模型能装下，但显存剩余不多，KV Cache 空间小，并发上不去
  → 量化后权重更小 → 省出显存给 KV Cache → 支持更多并发请求
```

**场景 3: 要更低延迟**

```
小 batch 推理是 memory-bound（瓶颈在从 HBM 读权重到 SM）
  → 量化后权重体积减小 → 访存量减少 → 单次推理延迟降低
```

### 不需要量化的场景

```
- 显存充足 + 精度要求极高（医疗、法律、金融等敏感场景） → 用 FP16/BF16
- 小模型（7B 以下）+ 显存够 → FP16 就行，量化收益不大
```

---

## 二、方案选择优先级

### 第一选择: FP8 (W8A8)

```
条件:    H800 / H100 / Ada 等支持 FP8 Tensor Core 的 GPU
精度:    几乎无损（perplexity 差距 < 0.02）
压缩比:  2x
速度:    最快（FP8 Tensor Core 原生计算，不需要反量化）
```

用法:
```bash
# Online（最方便，不需要预量化模型）
vllm serve /path/to/bf16-model --quantization fp8_per_tensor

# Offline（精度略好，用预量化 checkpoint）
vllm serve /path/to/fp8-model
```

适合: **H800 上绝大多数场景的首选方案**

### 第二选择: INT4 AWQ / GPTQ (W4A16)

```
条件:    显存不够用 FP8（FP8 只压缩 2x，INT4 压缩 4x）
精度:    轻微损失（perplexity 差距 0.1~0.2）
压缩比:  4x
速度:    比 FP8 慢（需要 kernel 内反量化），但比 FP16 快（访存少）
```

AWQ vs GPTQ 选择:
```
AWQ:  量化速度快，精度通常稍好，优先选这个
GPTQ: 生态更早，HuggingFace 上预量化模型资源更多
```

用法:
```bash
# 下载预量化模型直接用
vllm serve /path/to/model-awq
vllm serve /path/to/model-gptq
```

适合: **显存受限，需要单卡跑大模型**

### 第三选择: INT8 (W8A8)

```
条件:    GPU 不支持 FP8（V100 / A100 等），但支持 INT8 Tensor Core
精度:    损失小（perplexity 差距 < 0.05）
压缩比:  2x
速度:    INT8 Tensor Core 原生计算
```

用法:
```bash
# Weight-only（最简单）
vllm serve /path/to/model --quantization int8_per_channel_weight_only

# W8A8 需要 SmoothQuant 预量化模型
vllm serve /path/to/smoothquant-model
```

适合: **A100 等不支持 FP8 的 GPU**

### 第四选择: MXFP4 (W4A16)

```
条件:    MoE 模型 + Blackwell GPU（或 H800 用 Marlin fallback）
精度:    中等损失（perplexity 差距 0.2~0.3）
压缩比:  ~4x（权重 E2M1 + scale E8M0）
当前局限: vLLM 只实现了 MoE 层的量化，普通 Linear 回退到 FP16
```

适合: **MoE 模型（expert 数量多，权重占比大，压缩收益最大）**

---

## 三、决策流程图

```
显存够不够放 FP16 模型？
│
├─ 够 → 精度要求极高？
│        ├─ 是 → 不量化，用 FP16/BF16
│        └─ 否 → 要更大并发 / 更低延迟？
│                 ├─ 是 → FP8 online（最方便）
│                 └─ 否 → 不量化
│
└─ 不够 → GPU 支持 FP8？
           │
           ├─ 是 → FP8 压缩 2x 后能放下？
           │        ├─ 是 → FP8（精度最好）
           │        └─ 否 → INT4 AWQ（压缩 4x）
           │
           └─ 否 → GPU 支持 INT8？
                     ├─ 是 → INT8 压缩 2x 后能放下？
                     │        ├─ 是 → INT8
                     │        └─ 否 → INT4 AWQ/GPTQ
                     └─ 否 → INT4 AWQ/GPTQ
```

---

## 四、实际部署经验

### H800 单卡 (80GB) 参考

| 模型规模 | FP16 显存 | 推荐方案 | 量化后显存 | 备注 |
|---------|----------|---------|----------|------|
| 7B | 14GB | 不量化 | — | 显存充足，量化收益小 |
| 13B | 26GB | 不量化 或 FP8 | 13GB | 想要更多并发可用 FP8 |
| 34B | 68GB | FP8 | 34GB | FP16 能放下但 KV Cache 空间太小 |
| 70B | 140GB | FP8 或 INT4 | 70GB / 35GB | FP8 刚好单卡，INT4 更宽裕 |
| 405B | 810GB | 多卡 TP + FP8 | ~405GB (6卡) | 或 INT4 减少卡数 |

### 速查表

```
追求精度:   FP8，几乎无损
追求压缩:   INT4 AWQ，4x 压缩比
追求方便:   FP8 online，不需要预量化模型，加个参数就行
追求兼容:   INT4 GPTQ，预量化模型资源最多
MoE 模型:   MXFP4（如果 GPU 支持）或 FP8
```

### 量化后务必验证

无论选择哪种方案，上线前都要做精度验证:

```
1. 快速验证: perplexity 对比（5 分钟）
   → 差距 < 0.1: 可以直接用
   → 差距 0.1~0.5: 需要下游任务验证
   → 差距 > 0.5: 换更高精度的方案

2. 深度验证: 在实际业务场景的测试集上跑一轮（1-2 小时）
   → 对比量化前后的输出质量
   → 关注关键能力（推理、代码、数学等）有无退化
```
