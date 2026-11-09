# Week 6: MoE 模型推理优化

## 学习目标
1. 理解 MoE 架构原理与推理特性
2. 掌握 vLLM 中 FusedMoE kernel 实现
3. 理解 Expert Parallelism 基础与 All-to-All 通信
4. 掌握 MoE 模型的部署调优实践

## 目录结构
```
week6-moe-optimization/
├── README.md                      # 本文件
├── day1_code_notes.md             # Day 1: 代码结构笔记
├── day2_analysis.md               # Day 2: MoE vs Dense 分析
├── day3_kernel_dataflow.md        # Day 3: FusedMoE 数据流
├── day4_profiling_analysis.md     # Day 4: Profiling 分析
├── day5_ep_theory.md              # Day 5: EP 理论基础
├── WEEK6_SUMMARY.md               # Week 6 总结
│
├── exp1_moe_vs_dense.sh           # 实验1: MoE vs Dense 对比
├── exp2_profile_moe.py            # 实验2: Kernel Profiling
├── exp4_ep_vs_tp.sh               # 实验4: EP vs TP（需2+ GPU）
├── exp5_moe_tuning.sh             # 实验5: 部署调优
│
├── results/                       # 实验结果输出
└── logs/                          # 服务器日志
```

## 实验列表

| 实验 | 目标 | 需求 | 预期时长 |
|-----|------|------|---------|
| exp1 | MoE vs Dense 性能对比 | MoE 模型 | 1-2h |
| exp2 | FusedMoE kernel profiling | GPU | 1h |
| exp3 | Expert 路由分析（可选） | - | 1h |
| exp4 | EP vs TP 对比 | 2+ GPU | 2h |
| exp5 | 部署调优综合实验 | MoE 模型 | 1-2h |

## 模型准备

### 推荐模型
- **Qwen2-57B-A14B-Instruct**（~120GB）
  - 64 experts, top-8
  - 较新，社区支持好
  
- **Mixtral-8x7B-Instruct-v0.1**（~90GB）
  - 8 experts, top-2
  - 经典 MoE 模型

### 下载命令
```bash
# Qwen2 MoE（推荐）
huggingface-cli download Qwen/Qwen2-57B-A14B-Instruct \
  --local-dir /data/model/Qwen2-57B-A14B-Instruct \
  --local-dir-use-symlinks False

# 或 Mixtral
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --local-dir /data/model/Mixtral-8x7B-Instruct-v0.1 \
  --local-dir-use-symlinks False
```

### 检查现有模型
```bash
ls -lh /data/model/ | grep -i "mixtral\|qwen.*moe\|deepseek"
```

## 学习路径

### Day 0: 环境准备（0.5h）
- [ ] 模型下载（后台运行）
- [ ] 创建实验目录
- [ ] 检查 GPU 可用性

### Day 1: MoE 架构原理（3h）
- [ ] 阅读 MoE 基础理论（2h）
- [ ] 代码阅读：Qwen2MoeSparseMoeBlock（1h）
- [ ] 填写 `day1_code_notes.md`

**关键问题**：
- MoE 的参数量 vs 计算量权衡？
- Router 如何选择 expert？
- 为什么所有 expert 都需加载到 GPU？

### Day 2: MoE 推理挑战（3h）
- [ ] 运行 exp1：MoE vs Dense 对比（1-2h）
- [ ] 分析结果，填写 `day2_analysis.md`（1h）

**关键观察**：
- 内存占用比例
- 吞吐量和延迟对比
- 验证三大挑战：内存墙、负载不均、批处理效率

### Day 3: FusedMoE Kernel（上）（3h）
- [ ] 阅读 `fused_moe()` 代码（2h）
- [ ] 追踪 Token Permutation 逻辑
- [ ] 填写 `day3_kernel_dataflow.md`

**关键理解**：
- Permutation 的目的
- Fused Kernel 如何避免多次 launch
- Triton kernel 的执行流程

### Day 4: FusedMoE Kernel（下）+ Profiling（3h）
- [ ] 运行 exp2：Kernel Profiling（1h）
- [ ] 分析不同 batch size 的性能（1h）
- [ ] 填写 `day4_profiling_analysis.md`（1h）

**关键指标**：
- 吞吐饱和点
- GPU 利用率估算
- 与 Dense FFN 的对比

### Day 5: Expert Parallelism 理论（3h）
- [ ] 阅读 `day5_ep_theory.md`（2h）
- [ ] 理解 All-to-All 通信
- [ ] 理解 DP Attention 架构
- [ ] 代码阅读：EP process group（1h）

**关键概念**：
- EP vs TP 的本质区别
- All-to-All 通信量计算
- Lockstep 的必要性

### Day 6: EP 多卡实验（3h，可选）
- [ ] 检查 GPU 数量（需 2+）
- [ ] 检查 vLLM EP 支持
- [ ] 运行 exp4：EP vs TP（2h）
- [ ] 分析通信开销（1h）

**如果不满足条件**：
- 跳过实验，专注理论
- Week 8 会深入 EPLB 实现

### Day 7: 部署调优与总结（3h）
- [ ] 运行 exp5：调优实验（1-2h）
- [ ] 填写 `WEEK6_SUMMARY.md`（1-2h）
- [ ] 更新 `LearningPlan.md` Week 6 为 ✅

**总结内容**：
- 5 个实验的结果表格
- 核心知识点梳理
- 学习检查点 self-check
- 与前几周的关联

## 快速启动

### 运行所有实验（按顺序）
```bash
cd /home/lvzheng/vllm/learning/week6-moe-optimization

# Day 2
./exp1_moe_vs_dense.sh

# Day 4
.venv/bin/python exp2_profile_moe.py

# Day 6 (需 2+ GPU)
./exp4_ep_vs_tp.sh

# Day 7
./exp5_moe_tuning.sh
```

### 查看结果
```bash
# 所有 benchmark 结果
ls -lh results/

# 服务器日志（如果出错）
tail -100 logs/*.log
```

## 注意事项

### 实验环境
- **GPU 内存**：至少 40GB（A100/A6000），MoE 模型很大
- **存储空间**：至少 150GB（模型 + 结果）
- **vLLM 版本**：v1 架构

### 常见问题

#### Q1: 模型下载太慢
**A**: 使用镜像站或分片下载：
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...
```

#### Q2: OOM (Out of Memory)
**A**: 降低 `gpu_memory_utilization` 或 `max_num_seqs`：
```bash
--gpu-memory-utilization 0.80  # 默认 0.90
--max-num-seqs 16              # 默认 32
```

#### Q3: EP 参数不存在
**A**: 当前 vLLM 版本可能不支持 EP，跳过 exp4，Week 8 会覆盖

#### Q4: Benchmark 失败
**A**: 检查服务器日志 `logs/exp*_server.log`，确认服务启动成功

## 学习检查点

完成 Week 6 后，你应该能：

### 理论
- [ ] 解释 MoE 的 Router 机制和 Top-K 选择
- [ ] 理解 MoE 三大推理挑战（内存、负载、批处理）
- [ ] 画出 FusedMoE kernel 数据流图
- [ ] 解释 All-to-All vs All-Reduce 的区别
- [ ] 理解 DP Attention 和 Lockstep

### 代码
- [ ] 找到 MoE 模型的 forward 流程
- [ ] 找到 fused_moe() 函数定义
- [ ] 理解 topk_weights 和 topk_ids 的作用
- [ ] 找到 EP process group 初始化代码

### 实践
- [ ] 运行 MoE vs Dense 对比实验
- [ ] Profile FusedMoE kernel 性能
- [ ] 分析不同 batch size 的性能曲线
- [ ] （可选）运行 EP vs TP 对比

### 部署
- [ ] 知道何时选择 MoE 而非 Dense
- [ ] 知道 MoE 的最优 batch size 范围
- [ ] 理解 EP vs TP 的选择依据
- [ ] 知道 `gpu_memory_utilization` 对 MoE 的影响

## 参考资源

### 论文
- **Switch Transformers**: https://arxiv.org/abs/2101.03961
- **Mixtral of Experts**: https://arxiv.org/abs/2401.04088
- **DeepSeek-V2**: https://arxiv.org/abs/2405.04434

### 博客
- Hugging Face: Understanding Mixture of Experts
- vLLM Blog: Efficient MoE Serving

### 代码
- vLLM FusedMoE: `vllm/model_executor/layers/fused_moe/`
- Qwen2-MoE: `vllm/model_executor/models/qwen2_moe.py`
- Mixtral: `vllm/model_executor/models/mixtral.py`

---

**Week 6 学习周期**：7 天  
**预计总时长**：15-20 小时  
**前置知识**：Week 1-5（特别是 Week 4 分布式基础）  
**后续学习**：Week 7 量化（MoE 量化特别重要）、Week 8 EPLB
