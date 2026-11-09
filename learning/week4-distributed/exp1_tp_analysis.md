# 实验1笔记：TP=1/2/4 在线推理性能对比

> 实验时间：2026-09-02
> 配置：Qwen2.5-7B, gpu_memory_utilization=0.85, random dataset (input_len=256, output_len=128, 100 prompts, request_rate=inf)
> 环境：8× H800 80GB, NVLink, vLLM 0.27.1 (V1 引擎, online serving)

---

## 1. 测试结果

| 指标 | TP=1 | TP=2 | TP=4 |
|------|------|------|------|
| Successful requests | 100 | 100 | 100 |
| Request throughput (req/s) | 59.03 | 79.47 | 115.59 |
| Output token throughput (tok/s) | 7,556 | 10,172 | 14,795 |
| Mean TTFT (ms) | 601.76 | 411.38 | 283.32 |
| Mean TPOT (ms) | 8.20 | 6.23 | 4.39 |
| Mean ITL (ms) | 8.27 | 6.23 | 4.39 |

## 2. 加速比分析

| 对比 | 吞吐加速比 | TTFT 加速比 | TPOT 加速比 |
|------|-----------|------------|------------|
| TP=2 vs TP=1 | 1.35x | 1.46x | 1.32x |
| TP=4 vs TP=1 | 1.96x | 2.12x | 1.87x |

TP 增大未达到线性加速（TP=4 理想值为 4x），原因：

### 2.1 通信开销

- 每个 Transformer layer 有 2 次 AllReduce（Attention + MLP）
- Qwen2.5-7B 有 28 层 → 每次 forward 共 56 次 AllReduce
- Ring AllReduce 步数为 n-1，TP=4 时需 3 步（vs TP=2 的 1 步）
- 虽然 NVLink 带宽高（H800 单向约 400 GB/s），但每次 AllReduce 有固定启动延迟，56 次累积可观

### 2.2 不可并行部分（Amdahl 定律）

以下操作未被 TP 切分或需要额外同步：
- RMSNorm：每卡完整执行
- Sampling（argmax/top-k）：不并行
- CPU 侧：Scheduling、KV Cache 管理、Tokenizer 编解码

### 2.3 TP vs DP 的取舍

同样 4 张卡，两种方案对比：
- **TP=4, DP=1**：4 卡一起处理同一 batch，每层通信，吞吐 ~1.96x
- **TP=1, DP=4**：4 卡独立处理各自 batch，零通信，吞吐 ~4x

TP 的优势在于**降低单请求延迟**（TTFT 从 601→283ms），DP 的优势在于**最大化吞吐**。

生产决策：
- 模型单卡放得下 → 优先 DP 扩吞吐
- 模型单卡放不下 → TP 切到单卡能放，再用 DP 扩吞吐

## 3. 模型兼容性

Qwen2.5-7B 有 28 个 attention head，TP 必须能整除 28。
- 可用：TP=1, 2, 4, 7, 14, 28
- 不可用：TP=8（28/8=3.5，无法整除，启动报错）

## 4. 踩坑记录

### 4.1 源码导入陷阱
在 `/home/lvzheng/vllm/` 源码目录下执行 `python3 -m vllm.entrypoints.openai.api_server` 会导入本地源码而非安装的包，报 `ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'`。使用 `vllm serve` CLI 命令不受影响。

### 4.2 HTTP 代理导致 502
终端环境设置了 `http_proxy`，curl 请求 localhost 被转发到代理服务器，返回 502 Bad Gateway。解决：脚本开头 `export no_proxy=localhost,127.0.0.1`。

### 4.3 Server 就绪判断
`/health` endpoint 在 uvicorn 启动时就返回 200，但此时 EngineCore 可能还在初始化。正确做法：监听 server 日志中的 `Application startup complete` 标志。

### 4.4 GPU 显存泄漏
`kill -9` 强杀 vLLM 进程会导致子进程（EngineCore）残留和 GPU 显存泄漏。应使用 `kill`（SIGTERM）正常退出，让 vLLM 走 graceful shutdown 流程清理资源。

### 4.5 gpu_memory_utilization
默认 0.92 在 TP=1 时可能导致 OOM（模型权重 + KV Cache + CUDA Graph 后剩余不足）。降到 0.85 解决。
