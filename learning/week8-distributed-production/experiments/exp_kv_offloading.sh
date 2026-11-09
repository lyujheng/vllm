#!/bin/bash
set -e

########################################################################
# Day 6 实验：KV Cache Offloading 效果验证
#
# 实验目标：
#   1. 对比 baseline（无 offloading）vs offloading（eager/lazy）在高并发长上下文下的表现
#   2. 观察 offloading 对 prefix cache 命中率的影响
#   3. 对比不同 GPU 显存利用率下 offloading 的收益
#
# 实验设计：
#   实验 A：Baseline vs Eager vs Lazy（固定 gpu-memory-utilization=0.7）
#          - 低显存利用率模拟显存紧张场景，放大 offloading 效果
#          - 使用共享前缀请求，触发 prefix cache + offloading 交互
#   实验 B：不同显存利用率下 offloading 的收益（0.5 / 0.7 / 0.9）
#          - 观察显存越紧张 offloading 收益越大的预期
#
# 模型：Qwen2.5-7B（单卡）
# 硬件：单机 8xH800，实验使用 CUDA_VISIBLE_DEVICES 限制为单卡
########################################################################

MODEL="/data/model/Qwen2.5-7B"
PORT=8000
OUTPUT_DIR="./results/day6_kv_offloading"
CPU_OFFLOAD_BYTES=42949672960  # 40GB CPU 内存用于 offloading

export no_proxy=localhost,127.0.0.1
export CUDA_VISIBLE_DEVICES=2

mkdir -p $OUTPUT_DIR

########################################################################
# 工具函数
########################################################################

start_server() {
    local server_args="$1"
    local log_file="$2"

    echo "  Starting server: vllm serve $MODEL $server_args"
    eval "vllm serve $MODEL $server_args > $log_file 2>&1 &"
    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID"

    echo "  Waiting for server to start..."
    for i in $(seq 1 180); do
        if grep -q "Application startup complete" "$log_file" 2>/dev/null; then
            echo "  ✓ Server ready!"
            return 0
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "  ✗ Server process died. Check $log_file"
            return 1
        fi
        sleep 5
    done
    echo "  ✗ Server start timeout. Check $log_file"
    return 1
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        echo "  Stopping server (PID=$SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        sleep 10
    fi
}

run_benchmark() {
    local bench_args="$1"
    local result_file="$2"

    echo "  Running benchmark..."
    eval "vllm bench serve --model $MODEL --port $PORT $bench_args" \
        2>&1 | tee "$result_file"
}

########################################################################
# 实验 A：Baseline vs Eager vs Lazy
#
# 场景：gpu-memory-utilization=0.7 模拟显存紧张
# 请求：共享前缀（input_len=2048）+ 多轮请求（num_prompts=200）
#       共享前缀可触发 prefix cache，offloading 可扩大缓存容量
########################################################################

echo "============================================================"
echo "实验 A：Baseline vs Eager vs Lazy 对比"
echo "============================================================"

GPU_MEM_UTIL=0.7
INPUT_LEN=2048
OUTPUT_LEN=128
NUM_PROMPTS=200
REQUEST_RATE=10  # 适中速率，模拟实际在线场景

# --- A1: Baseline（无 offloading，有 prefix cache）---
echo ""
echo "--- A1: Baseline (no offloading) ---"
start_server "--port $PORT --gpu-memory-utilization $GPU_MEM_UTIL --max-num-seqs 64" \
    "$OUTPUT_DIR/A1_baseline_server.log"

run_benchmark "--dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS --request-rate $REQUEST_RATE \
    --sharegpt-output-len $OUTPUT_LEN" \
    "$OUTPUT_DIR/A1_baseline_bench.txt"

# 再跑一轮相同请求，观察 prefix cache 命中
echo "  Running second round (prefix cache warm)..."
run_benchmark "--dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS --request-rate $REQUEST_RATE \
    --sharegpt-output-len $OUTPUT_LEN" \
    "$OUTPUT_DIR/A1_baseline_bench_warm.txt"

stop_server

# --- A2: Eager offloading ---
echo ""
echo "--- A2: Eager offloading ---"
start_server "--port $PORT --gpu-memory-utilization $GPU_MEM_UTIL --max-num-seqs 64 \
    --kv-transfer-config '{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU_OFFLOAD_BYTES}}'" \
    "$OUTPUT_DIR/A2_eager_server.log"

run_benchmark "--dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS --request-rate $REQUEST_RATE \
    --sharegpt-output-len $OUTPUT_LEN" \
    "$OUTPUT_DIR/A2_eager_bench.txt"

echo "  Running second round (prefix cache + CPU offload warm)..."
run_benchmark "--dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS --request-rate $REQUEST_RATE \
    --sharegpt-output-len $OUTPUT_LEN" \
    "$OUTPUT_DIR/A2_eager_bench_warm.txt"

stop_server

# --- A3: Lazy offloading ---
echo ""
echo "--- A3: Lazy offloading ---"
start_server "--port $PORT --gpu-memory-utilization $GPU_MEM_UTIL --max-num-seqs 64 \
    --kv-transfer-config '{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU_OFFLOAD_BYTES,\"lazy_offload\":true}}'" \
    "$OUTPUT_DIR/A3_lazy_server.log"

run_benchmark "--dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS --request-rate $REQUEST_RATE \
    --sharegpt-output-len $OUTPUT_LEN" \
    "$OUTPUT_DIR/A3_lazy_bench.txt"

echo "  Running second round (prefix cache + CPU offload warm)..."
run_benchmark "--dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS --request-rate $REQUEST_RATE \
    --sharegpt-output-len $OUTPUT_LEN" \
    "$OUTPUT_DIR/A3_lazy_bench_warm.txt"

stop_server

########################################################################
# 实验 B：不同显存利用率下 offloading 的收益
#
# 固定 eager 模式，对比不同 gpu-memory-utilization
########################################################################

echo ""
echo "============================================================"
echo "实验 B：不同显存利用率下 Eager Offloading 的收益"
echo "============================================================"

for GPU_UTIL in 0.5 0.7 0.9; do
    echo ""
    echo "--- B: gpu-memory-utilization=$GPU_UTIL (no offloading) ---"
    start_server "--port $PORT --gpu-memory-utilization $GPU_UTIL --max-num-seqs 64" \
        "$OUTPUT_DIR/B_util${GPU_UTIL}_baseline_server.log"

    run_benchmark "--dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS --request-rate $REQUEST_RATE \
        --sharegpt-output-len $OUTPUT_LEN" \
        "$OUTPUT_DIR/B_util${GPU_UTIL}_baseline_bench.txt"

    stop_server

    echo ""
    echo "--- B: gpu-memory-utilization=$GPU_UTIL (eager offloading) ---"
    start_server "--port $PORT --gpu-memory-utilization $GPU_UTIL --max-num-seqs 64 \
        --kv-transfer-config '{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU_OFFLOAD_BYTES}}'" \
        "$OUTPUT_DIR/B_util${GPU_UTIL}_eager_server.log"

    run_benchmark "--dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS --request-rate $REQUEST_RATE \
        --sharegpt-output-len $OUTPUT_LEN" \
        "$OUTPUT_DIR/B_util${GPU_UTIL}_eager_bench.txt"

    stop_server
done

########################################################################
# 汇总结果
########################################################################

echo ""
echo "============================================================"
echo "实验结果汇总"
echo "============================================================"

echo ""
echo "--- 实验 A：Baseline vs Eager vs Lazy ---"
for f in A1_baseline_bench.txt A1_baseline_bench_warm.txt \
         A2_eager_bench.txt A2_eager_bench_warm.txt \
         A3_lazy_bench.txt A3_lazy_bench_warm.txt; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        echo ""
        echo "[$f]"
        grep -E "Throughput|Mean TTFT|Mean TPOT|P99|Prefix cache" "$OUTPUT_DIR/$f" 2>/dev/null || true
    fi
done

echo ""
echo "--- 实验 B：不同显存利用率 ---"
for f in $OUTPUT_DIR/B_*.txt; do
    if [ -f "$f" ]; then
        echo ""
        echo "[$(basename $f)]"
        grep -E "Throughput|Mean TTFT|Mean TPOT|P99|Prefix cache" "$f" 2>/dev/null || true
    fi
done

echo ""
echo "All experiments completed! Results in $OUTPUT_DIR/"
