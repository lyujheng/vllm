# vLLM 正确使用方式总结

**更新日期**：2024  
**vLLM 版本**：最新 CLI 接口

---

## 一、vLLM Server 启动

### 1.1 基本启动命令

```bash
vllm serve <MODEL_PATH> [OPTIONS]
```

### 1.2 常用参数

| 参数 | 说明 | 示例 |
|-----|------|------|
| `--port` | API 服务端口 | `--port 8000` |
| `--host` | 监听地址 | `--host 0.0.0.0` |
| `--gpu-memory-utilization` | GPU 显存利用率 | `--gpu-memory-utilization 0.85` |
| `--max-num-seqs` | 最大并发序列数 | `--max-num-seqs 64` |
| `--tensor-parallel-size` | Tensor Parallel 大小 | `--tensor-parallel-size 2` |
| `--pipeline-parallel-size` | Pipeline Parallel 大小 | `--pipeline-parallel-size 2` |
| `--trust-remote-code` | 信任远程代码 | `--trust-remote-code` |

### 1.3 启动示例

#### 单 GPU 启动
```bash
vllm serve /data/model/Qwen2.5-7B \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64
```

#### TP=2 启动（2 GPU Tensor Parallel）
```bash
vllm serve /data/model/Qwen2-57B-A14B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.85
```

#### 后台运行并记录日志
```bash
vllm serve /data/model/Qwen2.5-7B \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    > server.log 2>&1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
```

### 1.4 检查服务就绪

```bash
# 方法 1：检查日志中的启动完成标志
for i in $(seq 1 120); do
    if grep -q "Application startup complete" server.log 2>/dev/null; then
        echo "✓ Server ready!"
        break
    fi
    sleep 5
done

# 方法 2：使用 curl 检查健康状态
curl http://localhost:8000/health
```

### 1.5 停止服务

```bash
# 优雅停止（推荐）
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

# 强制停止
kill -9 $SERVER_PID

# 等待端口释放
sleep 10
```

---

## 二、vLLM Benchmark

### 2.1 基本命令

```bash
vllm bench serve [OPTIONS]
```

**注意**：
- ❌ 旧方式：`python benchmarks/benchmark_serving.py`（已废弃）
- ✅ 新方式：`vllm bench serve`

### 2.2 常用参数

| 参数 | 说明 | 示例 |
|-----|------|------|
| `--model` | 模型名称或路径 | `--model /data/model/Qwen2.5-7B` |
| `--dataset-name` | 数据集类型 | `--dataset-name random` |
| `--random-input-len` | 随机输入长度 | `--random-input-len 256` |
| `--random-output-len` | 随机输出长度 | `--random-output-len 128` |
| `--num-prompts` | 请求数量 | `--num-prompts 100` |
| `--request-rate` | 请求速率（QPS） | `--request-rate inf` (无限速) |
| `--port` | 服务端口 | `--port 8000` |

### 2.3 Benchmark 示例

#### 基础性能测试
```bash
vllm bench serve \
    --model /data/model/Qwen2.5-7B \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts 100 \
    --request-rate inf \
    --port 8000
```

#### 保存结果到文件
```bash
vllm bench serve \
    --model /data/model/Qwen2.5-7B \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --num-prompts 100 \
    --request-rate inf \
    --port 8000 \
    2>&1 | tee benchmark_results.txt
```

#### 不同负载测试
```bash
# 测试不同 request rate
for rate in 1 5 10 inf; do
    echo "Testing request rate: $rate"
    vllm bench serve \
        --model /data/model/Qwen2.5-7B \
        --dataset-name random \
        --random-input-len 256 \
        --random-output-len 128 \
        --num-prompts 100 \
        --request-rate $rate \
        --port 8000 \
        2>&1 | tee benchmark_rate_${rate}.txt
done
```

### 2.4 输出指标说明

Benchmark 输出的关键指标：

| 指标 | 含义 |
|-----|------|
| **Throughput** | 吞吐量（tokens/s 或 requests/s） |
| **Mean TTFT** | 平均首 token 延迟 (Time To First Token) |
| **Mean TPOT** | 平均每 token 延迟 (Time Per Output Token) |
| **P50/P95/P99 TTFT** | TTFT 的百分位数 |
| **P50/P95/P99 TPOT** | TPOT 的百分位数 |

---

## 三、完整实验脚本模板

### 3.1 单模型性能测试

```bash
#!/bin/bash
set -e

MODEL="/data/model/Qwen2.5-7B"
OUTPUT_DIR="./results"
mkdir -p $OUTPUT_DIR

# 避免 localhost 请求走代理
export no_proxy=localhost,127.0.0.1

echo "Starting vLLM server..."
vllm serve $MODEL \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    > $OUTPUT_DIR/server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等待服务就绪
echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if grep -q "Application startup complete" $OUTPUT_DIR/server.log 2>/dev/null; then
        echo "✓ Server ready!"
        break
    fi
    sleep 5
done

# 运行 benchmark
echo "Running benchmark..."
vllm bench serve \
    --model $MODEL \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts 100 \
    --request-rate inf \
    --port 8000 \
    2>&1 | tee $OUTPUT_DIR/benchmark.txt

# 提取关键指标
echo ""
echo "=== Results ==="
grep -E "Throughput|Mean TTFT|Mean TPOT|P99" $OUTPUT_DIR/benchmark.txt

# 清理
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "Done!"
```

### 3.2 对比实验模板（不同配置）

```bash
#!/bin/bash
set -e

MODEL="/data/model/Qwen2.5-7B"
OUTPUT_DIR="./results"
mkdir -p $OUTPUT_DIR

export no_proxy=localhost,127.0.0.1

# 测试不同 TP 配置
for TP in 1 2 4; do
    echo "============================================"
    echo "Testing TP=$TP"
    echo "============================================"

    vllm serve $MODEL \
        --tensor-parallel-size $TP \
        --port 8000 \
        --gpu-memory-utilization 0.85 \
        > $OUTPUT_DIR/tp${TP}_server.log 2>&1 &
    SERVER_PID=$!

    # 等待服务就绪
    for i in $(seq 1 120); do
        if grep -q "Application startup complete" $OUTPUT_DIR/tp${TP}_server.log 2>/dev/null; then
            echo "✓ Server ready!"
            break
        fi
        sleep 5
    done

    # 运行 benchmark
    vllm bench serve \
        --model $MODEL \
        --dataset-name random \
        --random-input-len 256 \
        --random-output-len 128 \
        --num-prompts 100 \
        --request-rate inf \
        --port 8000 \
        2>&1 | tee $OUTPUT_DIR/tp${TP}_benchmark.txt

    # 清理
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 10
done

echo "All tests completed! Results in $OUTPUT_DIR/"
```

---

## 四、常见问题

### 4.1 端口被占用

```bash
# 检查端口占用
lsof -i :8000

# 杀掉占用端口的进程
kill -9 $(lsof -t -i:8000)
```

### 4.2 服务启动失败

```bash
# 检查日志
tail -f server.log

# 常见原因：
# 1. 显存不足 → 降低 --gpu-memory-utilization
# 2. 模型路径错误 → 检查 MODEL 变量
# 3. CUDA 版本不兼容 → 重新安装 vLLM
```

### 4.3 Benchmark 连接失败

```bash
# 确认服务已启动
curl http://localhost:8000/health

# 设置代理绕过（如果使用了代理）
export no_proxy=localhost,127.0.0.1
```

### 4.4 显存溢出 (OOM)

```bash
# 解决方案 1：降低显存利用率
--gpu-memory-utilization 0.7

# 解决方案 2：减少并发
--max-num-seqs 32

# 解决方案 3：增加 TP
--tensor-parallel-size 2
```

---

## 五、与旧版本对比

| 功能 | 旧版本（已废弃） | 新版本（推荐） |
|-----|----------------|---------------|
| **启动 server** | `python -m vllm.entrypoints.openai.api_server --model ...` | `vllm serve <model>` |
| **Benchmark** | `python benchmarks/benchmark_serving.py --backend vllm --model ...` | `vllm bench serve --model ...` |
| **Throughput 测试** | `python benchmarks/benchmark_throughput.py` | `vllm bench throughput` |
| **延迟测试** | `python benchmarks/benchmark_latency.py` | `vllm bench latency` |

---

## 六、参考资料

### 官方文档
- vLLM CLI 文档：https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- Benchmark 工具：https://docs.vllm.ai/en/latest/performance/benchmarks.html

### 帮助命令
```bash
# 查看 serve 帮助
vllm serve --help

# 查看 benchmark 帮助
vllm bench serve --help

# 查看所有子命令
vllm --help
```

---

**重要提示**：
- ✅ 始终使用 `vllm serve` 启动服务
- ✅ 始终使用 `vllm bench serve` 运行 benchmark
- ✅ 设置 `export no_proxy=localhost,127.0.0.1` 避免代理问题
- ✅ 等待 "Application startup complete" 日志后再运行 benchmark
- ✅ 测试完成后使用 `kill` 正常停止服务，并 `sleep 10` 等待端口释放
