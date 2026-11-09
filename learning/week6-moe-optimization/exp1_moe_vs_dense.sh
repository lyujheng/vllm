#!/bin/bash
# Week 6 Experiment 1: MoE vs Dense 性能对比
# 目标：对比 MoE 和 Dense 模型在相同负载下的性能差异

set -e

# 配置
DENSE_MODEL="/data/model/Qwen2.5-7B"
MOE_MODEL="/data/model/Qwen2-57B-A14B-Instruct"
# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"

# Benchmark 参数
NUM_PROMPTS=100
INPUT_LEN=512
OUTPUT_LEN=128

mkdir -p $RESULTS_DIR $LOGS_DIR

echo "============================================"
echo "Week 6 Exp1: MoE vs Dense Performance"
echo "============================================"

# 检查模型是否存在
if [ ! -d "$DENSE_MODEL" ]; then
    echo "❌ Dense model not found: $DENSE_MODEL"
    exit 1
fi

if [ ! -d "$MOE_MODEL" ]; then
    echo "⚠️  MoE model not found: $MOE_MODEL"
    echo "Please download a MoE model or update MOE_MODEL variable"
    echo "Skipping MoE benchmarks..."
    MOE_AVAILABLE=false
else
    MOE_AVAILABLE=true
fi

# Test 1: Dense model baseline
echo ""
echo "Test 1: Dense Model Baseline"
echo "--------------------------------------------"
echo "Model: $DENSE_MODEL"

CUDA_VISIBLE_DEVICES=5 vllm serve $DENSE_MODEL \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    > $LOGS_DIR/exp1_dense_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等待服务启动
echo "Waiting for server to start..."
for i in {1..60}; do
    if grep -q "Application startup complete" $LOGS_DIR/exp1_dense_server.log 2>/dev/null; then
        echo "✓ Server started"
        break
    fi
    sleep 2
done

sleep 5

# 运行 benchmark
echo "Running benchmark..."
vllm bench serve \
    --model $DENSE_MODEL \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS \
    --request-rate inf \
    --port 8000 \
    2>&1 | tee $RESULTS_DIR/exp1_dense_benchmark.txt

# 提取关键指标
echo ""
echo "Dense Model Results:"
grep -E "Throughput|Mean TTFT|Mean TPOT|P99 TTFT|P99 TPOT" $RESULTS_DIR/exp1_dense_benchmark.txt || echo "Failed to extract metrics"

# 清理
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 10

# Test 2: MoE model
if [ "$MOE_AVAILABLE" = true ]; then
    echo ""
    echo "Test 2: MoE Model"
    echo "--------------------------------------------"
    echo "Model: $MOE_MODEL"

    CUDA_VISIBLE_DEVICES=5,6 vllm serve $MOE_MODEL \
        --port 8000 \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 64 \
        --tensor-parallel-size 2 \
        > $LOGS_DIR/exp1_moe_server.log 2>&1 &

    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    # 等待服务启动
    echo "Waiting for server to start..."
    for i in {1..120}; do
        if grep -q "Application startup complete" $LOGS_DIR/exp1_moe_server.log 2>/dev/null; then
            echo "✓ Server started"
            break
        fi
        sleep 2
    done

    sleep 5

    # 运行 benchmark
    echo "Running benchmark..."
    vllm bench serve \
        --model $MOE_MODEL \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --request-rate inf \
        --port 8000 \
        2>&1 | tee $RESULTS_DIR/exp1_moe_benchmark.txt

    # 提取关键指标
    echo ""
    echo "MoE Model Results:"
    grep -E "Throughput|Mean TTFT|Mean TPOT|P99 TTFT|P99 TPOT" $RESULTS_DIR/exp1_moe_benchmark.txt || echo "Failed to extract metrics"

    # 清理
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 10
fi

# 生成对比报告
echo ""
echo "============================================"
echo "Summary"
echo "============================================"
echo "Results saved to:"
echo "  - $RESULTS_DIR/exp1_dense_benchmark.txt"
if [ "$MOE_AVAILABLE" = true ]; then
    echo "  - $RESULTS_DIR/exp1_moe_benchmark.txt"
fi
echo ""
echo "Next step: Fill day2_analysis.md with the comparison"
