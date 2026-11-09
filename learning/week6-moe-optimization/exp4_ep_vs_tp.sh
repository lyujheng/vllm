#!/bin/bash
# Week 6 Experiment 4: EP vs TP 性能对比（单机 8 卡 H800）
# 目标：对比 Expert Parallelism 和 Tensor Parallelism 在 MoE 模型上的表现
#
# 测试配置（Qwen2-57B-A14B-Instruct，64 experts，top-8）：
#   Test 1: TP=4（纯 TP，MoE 层按 tensor 切分，每个 expert 被切成 4 片）
#   Test 2: TP=4 + EP（MoE 层 EP=4，每卡持有完整的 16 个 expert）
#   Test 3: TP=2, DP=2 + EP（MoE 层 EP=4，同时有 DP 吞吐提升）
#
# 注意：Qwen2-57B 单卡放不下，最少需要 TP=4
# 使用 GPU 1,2,3,4

set -e

export CUDA_VISIBLE_DEVICES=1,2,3,4
export no_proxy=localhost,127.0.0.1

# 配置
MOE_MODEL="/data/model/Qwen2-57B-A14B-Instruct"
RESULTS_DIR="./results"
LOGS_DIR="./logs"

# Benchmark 参数
NUM_PROMPTS=50
INPUT_LEN=512
OUTPUT_LEN=128
PORT=8000

mkdir -p $RESULTS_DIR $LOGS_DIR

echo "============================================"
echo "Week 6 Exp4: EP vs TP Comparison"
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
    echo "Running benchmark..."
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
# Test 1: TP=4（纯 TP，不开 EP）
# MoE 层：每个 expert 被 TP 切成 4 片，通信用 AllReduce
# ===========================================
echo ""
echo "============================================"
echo "Test 1: TP=4 (Pure TP, no EP)"
echo "  MoE layers: TP=4, each expert split across 4 GPUs"
echo "  Communication: AllReduce"
echo "============================================"

vllm serve $MOE_MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 4 \
    > $LOGS_DIR/exp4_tp4_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if start_server_and_wait "$LOGS_DIR/exp4_tp4_server.log"; then
    run_benchmark "$RESULTS_DIR/exp4_tp4_benchmark.txt"
    echo ""
    echo "TP=4 Results:"
    grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp4_tp4_benchmark.txt || echo "Failed to extract metrics"
fi

stop_server

# ===========================================
# Test 2: TP=4 + EP（开启 EP）
# MoE 层：EP=4，每卡持有完整的 16 个 expert（64/4）
# 非 MoE 层：TP=4
# ===========================================
echo ""
echo "============================================"
echo "Test 2: TP=4 + EP (Expert Parallel enabled)"
echo "  MoE layers: EP=4, each GPU holds 16 complete experts"
echo "  Non-MoE layers: TP=4"
echo "============================================"

vllm serve $MOE_MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    > $LOGS_DIR/exp4_tp4_ep_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if start_server_and_wait "$LOGS_DIR/exp4_tp4_ep_server.log"; then
    run_benchmark "$RESULTS_DIR/exp4_tp4_ep_benchmark.txt"
    echo ""
    echo "TP=4 + EP Results:"
    grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp4_tp4_ep_benchmark.txt || echo "Failed to extract metrics"
fi

stop_server

# ===========================================
# Test 3: TP=2, DP=2 + EP
# MoE 层：EP=4（2*2），每卡持有 16 个完整 expert，跨 DP 通信
# 非 MoE 层：TP=2，DP=2（两个独立 engine）
# ===========================================
echo ""
echo "============================================"
echo "Test 3: TP=2, DP=2 + EP"
echo "  MoE layers: EP=4 (2*2), AllGatherv + ReduceScatterv across DP"
echo "  Non-MoE layers: TP=2, DP=2 (two independent engines)"
echo "  Expected: higher throughput from DP, but EP communication overhead"
echo "============================================"

vllm serve $MOE_MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    > $LOGS_DIR/exp4_tp2_dp2_ep_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if start_server_and_wait "$LOGS_DIR/exp4_tp2_dp2_ep_server.log"; then
    run_benchmark "$RESULTS_DIR/exp4_tp2_dp2_ep_benchmark.txt"
    echo ""
    echo "TP=2, DP=2 + EP Results:"
    grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp4_tp2_dp2_ep_benchmark.txt || echo "Failed to extract metrics"
fi

stop_server

# ===========================================
# 生成对比报告
# ===========================================
echo ""
echo "============================================"
echo "Summary: EP vs TP Comparison"
echo "============================================"

echo ""
echo "--- Test 1: TP=4 (Pure TP) ---"
grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp4_tp4_benchmark.txt 2>/dev/null || echo "  (no data)"

echo ""
echo "--- Test 2: TP=4 + EP ---"
grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp4_tp4_ep_benchmark.txt 2>/dev/null || echo "  (no data)"

echo ""
echo "--- Test 3: TP=2, DP=2 + EP ---"
grep -E "Throughput|Mean TTFT|Mean TPOT|Mean ITL" $RESULTS_DIR/exp4_tp2_dp2_ep_benchmark.txt 2>/dev/null || echo "  (no data)"

echo ""
echo "============================================"
echo "Key comparisons to analyze:"
echo "  1. TP=4 vs TP=4+EP: EP 是否降低 MoE 层通信开销？"
echo "  2. TP=4+EP vs TP=2,DP=2+EP: DP 带来的吞吐提升 vs 跨 DP 的 AllGatherv 开销"
echo "  3. 显存占用差异：nvidia-smi 观察各配置的显存使用"
echo "============================================"
echo ""
echo "Results saved to: $RESULTS_DIR/exp4_*.txt"
echo "Server logs: $LOGS_DIR/exp4_*.log"
