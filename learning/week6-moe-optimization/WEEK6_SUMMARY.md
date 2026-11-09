# Week 6 学习总结：MoE 模型推理优化

**完成时间**：2026-09-16

---

## 一、核心知识点

### 1. MoE 架构原理

#### 基本概念
- **Router（Gate）机制**：为每个 token 计算 expert 选择概率，选择 Top-K
- **Expert 结构**：通常是 FFN（gate_proj, up_proj, down_proj）
- **激活模式**：
  - 总参数：E * FFN_params（所有 expert 都需加载）
  - 激活参数：K * FFN_params（每个 token 只用 K 个）
  - 示例：Qwen2-57B-A14B（64 experts, top-8）
    - 总参数 ~57B
    - 激活参数 ~14B

#### MoE vs Dense 权衡
| 维度 | MoE | Dense |
|-----|-----|-------|
| 参数量 | E × FFN_params | FFN_params |
| 单次计算量 | K/E × Dense | 1 × Dense |
| 内存占用 | 高（所有 expert） | 低 |
| 适合场景 | 大模型、稀疏激活 | 通用 |

### 2. FusedMoE Kernel 实现

#### Kernel 执行流程
```
1. Token Permutation（重排）
   - 目的：将相同 expert 的 token 聚集，提高内存访问连续性
   - 实现：argsort by expert_id
   - 开销：O(N*K * log(N*K))

2. Fused Expert Computation
   - 所有 expert 在一个 Triton kernel 完成
   - 避免 E 次 kernel launch 开销
   - 每个 expert 批量处理其分配的 tokens

3. Unpermutation + Router Weight
   - Scatter 结果回原始顺序
   - 应用 router 输出的权重
```

#### 关键代码位置
- **入口函数**：`vllm/model_executor/layers/fused_moe/fused_moe.py` - `fused_moe()`
- **Triton Kernel**：`fused_moe_kernel()` 
- **模型调用**：`vllm/model_executor/models/qwen2_moe.py` - `Qwen2MoeSparseMoeBlock`

### 3. Expert Parallelism (EP)

#### EP vs TP 对比
| | Tensor Parallelism | Expert Parallelism |
|---|-------------------|-------------------|
| 切分对象 | 权重矩阵（列/行） | Expert（整体） |
| 通信原语 | AllReduce | AllGatherv + ReduceScatterv |
| 适用层 | 所有 Dense 层 | MoE 层 |
| 负载均衡 | 天然均衡 | 依赖 router |

#### EP 通信流程（DP+EP 模式）
```
Step 1 - Dispatch (AllGatherv):
  各 rank 的 token 按 expert 归属聚合，变长 gather 到所有 rank

Step 2 - Local Expert Compute:
  每个 rank 计算其持有的 expert

Step 3 - Combine (ReduceScatterv):
  各 rank 的计算结果归约，每个 token 的结果回到原始 rank（按原始 token 数分散）
```

#### DP Attention 架构
- **Attention 层**：Data Parallel（各 GPU 独立，无通信）
- **MoE 层**：Expert Parallel（AllGatherv/ReduceScatterv）
- **ep_size = tp_size × dp_size**

### 4. Benchmark 指标体系

#### 延迟类（单个请求体验）
| 指标 | 含义 | 反映什么 |
|------|------|---------|
| Mean TTFT | 首 token 延迟 | Prefill 阶段性能 |
| P99 TTFT | 最差首 token 延迟 | 排队和调度稳定性 |
| Mean TPOT | 平均每 token 生成耗时 | Decode 阶段性能 |
| P99 ITL | 最差 token 间延迟 | Decode 稳定性 |

#### 吞吐类（系统整体能力）
| 指标 | 含义 | 分子 |
|------|------|------|
| Output throughput | 每秒生成 token 数 | 只算输出 token |
| Total throughput | 每秒处理总 token 数 | 输入 + 输出 token |

**核心关系**：延迟和吞吐是矛盾的。并发越高，吞吐越大，但每个请求延迟越高。

---

## 二、实验结果汇总

### Exp1: MoE vs Dense 性能对比
| 指标 | Dense (Qwen2.5-7B) | MoE (Qwen2-57B-A14B) | 比值 |
|-----|-----------|---------------|------|
| Output throughput (tok/s) | 4402.86 | 1536.22 | 0.35x |
| Total throughput (tok/s) | 22014.32 | 7681.12 | 0.35x |
| Mean TTFT (ms) | 1097.94 | 3090.71 | 2.8x |
| Mean TPOT (ms) | 8.84 | 26.62 | 3.0x |

**结论**：MoE 虽然激活参数只有 14B，但 57B 的总参数量导致内存访问和 KV cache 开销远高于 7B Dense。吞吐约为 Dense 的 1/3，延迟约为 3 倍。

### Exp2: FusedMoE Kernel Profiling（H800）
| Batch Size | Time (ms) | TFLOP/s | Tokens/s |
|-----------|----------|---------|----------|
| 16 | 13.56 | 2.22 | 1180 |
| 64 | 16.02 | 7.51 | 3994 |
| 256 | 17.03 | 28.24 | 15030 |
| 1024 | 24.87 | 77.36 | 41171 |
| 2048 | 42.85 | 89.81 | 47794 |

- **Peak TFLOP/s**: 89.81（batch_size=2048）
- **关键发现**：小 batch（<64）GPU 利用率极低（<8 TFLOP/s），batch=2048 仍未饱和

**结论**：MoE kernel 对 batch size 极度敏感，生产中需要尽量攒大 batch。

### Exp4: EP vs TP（4 卡 H800，50 prompts）
| 配置 | Output (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | P99 ITL (ms) |
|-----|---------------|---------------|---------------|-------------|
| TP=4 | 2081.68 | 1008.95 | 16.08 | 156.50 |
| TP=4 + EP | 2040.84 | **894.81** | 17.36 | 180.45 |
| TP=2, DP=2 + EP | 1774.30 | 1353.66 | 17.55 | **77.48** |

**结论**：
- EP 改善 prefill 延迟（TTFT 降低 11%），对 decode 吞吐影响不大
- DP+EP 在小请求量下吞吐不如纯 TP，但 P99 ITL 稳定性大幅提升（每个 engine 负载减半）
- DP 的吞吐优势需要高 QPS 场景才能体现

### Exp5: 部署调优（4 卡 H800，100 prompts）
| 配置 | Output (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | P99 ITL (ms) |
|-----|---------------|---------------|---------------|-------------|
| Baseline (mem=0.85, seqs=32) | 1463.73 | 4029.20 | **13.38** | **45.69** |
| High Mem (mem=0.90, seqs=64) | **1796.85** | **2557.85** | 21.79 | 197.38 |
| Chunked Prefill | 1573.40 | 2706.89 | 15.18 | 95.28 |

**结论**：
- High Memory 吞吐最高（+23%），但 decode 延迟最差（P99 ITL 4x）
- Chunked Prefill 在 TTFT 和 ITL 之间做折中
- Baseline 延迟最稳定，吞吐最低
- **注意**：这些是 vLLM 通用部署参数，非 MoE 特有

---

## 三、遇到的问题与解决

### 问题 1：实验脚本使用旧版 vLLM 指令
- **问题**：生成的脚本使用 `python -m vllm.entrypoints.openai.api_server` 和 `python benchmarks/benchmark_serving.py`
- **解决**：统一改为 `vllm serve` 和 `vllm bench serve`，记录到 `vllm_usage_guide.md`

### 问题 2：Qwen2-57B 单卡放不下
- **问题**：原脚本 Test 1 设计为单卡 baseline，57B 模型无法在单卡运行
- **解决**：去掉单卡测试，最小配置 TP=4

### 问题 3：EP 参数名不正确
- **问题**：脚本使用 `--expert-parallel-size`，实际参数是 `--enable-expert-parallel`
- **解决**：查阅 vLLM 源码 `vllm/config/parallel.py` 确认正确参数名

---

## 四、学习检查点（Self-Check）

### 理论理解
- [x] 能解释 MoE Router 的作用和 Top-K 选择机制
- [x] 理解 MoE 的"参数量大但计算量小"的本质
- [x] 能画出 FusedMoE kernel 的数据流（permute → compute → unpermute）
- [x] 理解 Token Permutation 的目的（聚集相同 expert 的 tokens）
- [x] 能解释 AllGatherv/ReduceScatterv vs AllReduce 的区别
- [x] 理解 DP Attention 架构和 ep_size = tp_size × dp_size

### 代码追踪
- [x] 找到 `fused_moe()` 函数定义
- [x] 找到 MoE 模型的 forward 调用（Qwen2MoeSparseMoeBlock）
- [x] 找到 EP 相关配置（`vllm/config/parallel.py` enable_expert_parallel）
- [x] 理解 topk_weights 和 topk_ids 的数据结构

### 实验技能
- [x] 能运行 MoE vs Dense 对比实验
- [x] 能使用 Python 脚本 profile FusedMoE kernel 性能
- [x] 能分析不同 batch size 下的性能曲线
- [x] 能解读 benchmark 输出的 TTFT/TPOT/ITL/Throughput 指标

### 部署决策
- [x] 知道何时选择 MoE 而非 Dense
- [x] 知道 MoE kernel 对 batch size 极度敏感（小 batch 利用率低）
- [x] 理解 EP vs TP 的性能差异和选择依据
- [x] 理解 gpu_memory_utilization / max_num_seqs / chunked_prefill 的权衡

---

## 五、与前几周的关联

| 周次 | 关联点 |
|------|--------|
| Week 1-2 KV Cache | MoE 总参数量大 → 留给 KV Cache 的空间更少 |
| Week 3 ModelRunner | MoE 层的 forward 在 ModelRunner.execute_model() 中执行 |
| Week 4 分布式推理 | EP 是 MoE 特有的并行策略，与 TP/PP 互补 |
| Week 5 高级优化 | Chunked Prefill 对 MoE 同样适用（exp5 验证） |

---

## 六、关键洞察

1. **MoE 的核心矛盾是"参数多、算力少"**：所有 expert 权重必须驻留 GPU，但每次只激活 K/E，导致内存墙问题远比 Dense 严重
2. **EP 的价值不在于吞吐提升，而在于通信模式优化**：EP 让每个 expert 完整存储，避免了 MoE 层的 AllReduce，对 prefill 改善最明显
3. **DP 的价值不仅是吞吐，还有延迟稳定性**：P99 ITL 的改善来自每个 engine 负载减半
4. **没有最优部署配置，只有最适合场景的配置**：吞吐 vs 延迟的权衡贯穿所有实验
