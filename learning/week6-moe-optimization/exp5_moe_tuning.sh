#!/bin/bash
# Week 6 Experiment 5: MoE 部署调优综合实验
# 目标：测试不同部署配置对 MoE 推理性能的影响
#
# 测试配置（Qwen2-57B-A14B-Instruct，TP=4）：
#   Test 1: Baseline — gpu_memory_utilization=0.85, max_num_seqs=32
#   Test 2: High Memory — gpu_memory_utilization=0.90, max_num_seqs=64
#   Test 3: Chunked Prefill — baseline + --enable-chunked-prefill
#
# 使用 GPU 1,2,3,4

set -e

export CUDA_VISIBLE_DEVICES=1,2,3,4
export no_proxy=localhost,127.0.0.1

# 配置
MOE_MODEL="/data/model/Qwen2-57B-A14B-Instruct"
RESULTS_DIR="./results"
LOGS_DIR="./logs"
PORT=8000
TP_SIZE=4

# Benchmark 参数
NUM_PROMPTS=100
INPUT_LEN=512
OUTPUT_LEN=128

mkdir -p $RESULTS_DIR $LOGS_DIR

echo "============================================"
echo "Week 6 Exp5: MoE Deployment Tuning"
echo "Model: Qwen2-57B-A14B-Instruct (64 experts, top-8)"
echo "GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================"

# 检查模型
if [ ! -d "$MOE_MODEL" ]; then
    echo "❌ MoE model not found: $MOE_MODEL"
    exit 1
fi

# 通用函数：启动 server 并等待就绪
start_server_and_wait() {
    local LOG_FILE=$1
    local TIMEOUT=300

    echo "Waiting for server to start (timeout: ${TIMEOUT}s)..."
    for i in $(seq 1 $TIMEOUT); do
        if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then
            echo "✓ Server started (${i}s)"
            sleep 5
            return 0
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ Server process exited unexpectedly. Check log: $LOG_FILE"
            tail -20 "$LOG_FILE"
            return 1
        fi
        sleep 1
    done
    echo "❌ Server start timeout. Check log: $LOG_FILE"
    tail -20 "$LOG_FILE"
    return 1
}

# 通用函数：运行 benchmark
run_benchmark() {
    local RESULT_FILE=$1
    echo "Running benchmark (${NUM_PROMPTS} prompts, input=${INPUT_LEN}, output=${OUTPUT_LEN})..."
    vllm bench serve \
        --model $MOE_MODEL \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --request-rate inf \
        --port $PORT \
        2>&1 | tee "$RESULT_FILE"
}

# 通用函数：停止 server
stop_server() {
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null || true
    sleep 10
}

# ===========================================
# Test 1: Baseline
# gpu_memory_utilization=0.85, max_num_seqs=32
# ===========================================
echo ""
echo "============================================"
echo "Test 1: Baseline"
echo "  gpu_memory_utilization=0.85, max_num_seqs=32"
echo "============================================"

vllm serve $MOE_MODEL \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    > $LOGS_DIR/exp5_baseline_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if start_server_and_wait "$LOGS_DIR/exp5_baseline_server.log"; then
    run_benchmark "$RESULTS_DIR/exp5_baseline_benchmark.txt"
    echo ""
    echo "Baseline Results:"
    grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp5_baseline_benchmark.txt || echo "Failed to extract metrics"
fi

stop_server

# ===========================================
# Test 2: High Memory Utilization
# gpu_memory_utilization=0.90, max_num_seqs=64
# ===========================================
echo ""
echo "============================================"
echo "Test 2: High Memory Utilization"
echo "  gpu_memory_utilization=0.90, max_num_seqs=64"
echo "============================================"

vllm serve $MOE_MODEL \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    > $LOGS_DIR/exp5_highmem_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if start_server_and_wait "$LOGS_DIR/exp5_highmem_server.log"; then
    run_benchmark "$RESULTS_DIR/exp5_highmem_benchmark.txt"
    echo ""
    echo "High Memory Results:"
    grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp5_highmem_benchmark.txt || echo "Failed to extract metrics"
fi

stop_server

# ===========================================
# Test 3: Chunked Prefill
# baseline + --enable-chunked-prefill
# ===========================================
echo ""
echo "============================================"
echo "Test 3: Chunked Prefill"
echo "  gpu_memory_utilization=0.85, max_num_seqs=32"
echo "  + --enable-chunked-prefill"
echo "============================================"

vllm serve $MOE_MODEL \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --enable-chunked-prefill \
    > $LOGS_DIR/exp5_chunked_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if start_server_and_wait "$LOGS_DIR/exp5_chunked_server.log"; then
    run_benchmark "$RESULTS_DIR/exp5_chunked_benchmark.txt"
    echo ""
    echo "Chunked Prefill Results:"
    grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp5_chunked_benchmark.txt || echo "Failed to extract metrics"
fi

stop_server

# ===========================================
# 生成对比报告
# ===========================================
echo ""
echo "============================================"
echo "Experiment 5 Summary"
echo "============================================"

echo ""
echo "--- Test 1: Baseline (mem=0.85, seqs=32) ---"
grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp5_baseline_benchmark.txt 2>/dev/null || echo "  (no data)"

echo ""
echo "--- Test 2: High Memory (mem=0.90, seqs=64) ---"
grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp5_highmem_benchmark.txt 2>/dev/null || echo "  (no data)"

echo ""
echo "--- Test 3: Chunked Prefill ---"
grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp5_chunked_benchmark.txt 2>/dev/null || echo "  (no data)"

echo ""
echo "============================================"
echo "Analysis hints:"
echo "  1. mem 0.85→0.90 + seqs 32→64: KV cache 更大，能缓存更多并发请求"
echo "  2. Chunked Prefill: prefill 被拆成小块与 decode 交替，观察 TTFT 和 ITL 变化"
echo "============================================"
echo ""
echo "All results saved to $RESULTS_DIR/"
echo "Next: Fill WEEK6_SUMMARY.md with findings"
