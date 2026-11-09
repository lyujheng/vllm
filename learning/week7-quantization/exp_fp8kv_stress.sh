#!/bin/bash
# Week 7 补充实验：FP8 KV Cache 高并发长上下文对比
# 目标：验证 KV Cache 成为瓶颈时 FP8 KV Cache 的吞吐提升
#
# 对比组：
#   A. BF16 权重 + BF16 KV Cache（基线）
#   B. BF16 权重 + FP8 KV Cache
#
# 设计思路：
#   - 用长 input（4096 token）让每请求 KV 占用大
#   - 高并发（500 请求）尝试打满 KV Cache
#   - request_rate=inf 一次性涌入，最大化并发压力

set -e

MODEL="/data/model/Qwen2.5-7B"
OUTPUT_DIR="/home/lvzheng/vllm/results/week7-quant"
PORT=8000

export no_proxy=localhost,127.0.0.1

mkdir -p "$OUTPUT_DIR"

wait_server_ready() {
    local logfile=$1
    local timeout=${2:-300}
    echo "等待服务就绪 (最长 ${timeout}s)..."
    for i in $(seq 1 "$timeout"); do
        if grep -q "Application startup complete" "$logfile" 2>/dev/null; then
            echo "✓ 服务就绪！(${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "✗ 服务启动超时，查看日志: $logfile"
    return 1
}

stop_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        echo "停止服务 (PID=$pid)..."
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
    fi
    sleep 5
    if lsof -t -i:"$PORT" >/dev/null 2>&1; then
        kill -9 "$(lsof -t -i:"$PORT")" 2>/dev/null
        sleep 3
    fi
}

run_benchmark() {
    local model_name=$1
    local output_file=$2
    local num_prompts=$3
    local input_len=$4
    local output_len=$5

    echo "  Benchmark: prompts=$num_prompts, input=$input_len, output=$output_len"
    vllm bench serve \
        --model "$model_name" \
        --tokenizer "$MODEL" \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --num-prompts "$num_prompts" \
        --request-rate inf \
        --port "$PORT" \
        2>&1 | tee "$output_file"
}

###############################################################################
# 实验 A：BF16 KV Cache（基线）
###############################################################################

echo "============================================================"
echo "实验 A：BF16 权重 + BF16 KV Cache（长上下文高并发）"
echo "============================================================"

vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    > "$OUTPUT_DIR/exp7a_bf16kv_long_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp7a_bf16kv_long_server.log"; then
    # 预热
    run_benchmark "$MODEL" "$OUTPUT_DIR/exp7a_bf16kv_long_warmup.txt" 20 4096 512
    # 正式测试
    run_benchmark "$MODEL" "$OUTPUT_DIR/exp7a_bf16kv_long_bench.txt" 500 4096 512
fi

stop_server $SERVER_PID

###############################################################################
# 实验 B：FP8 KV Cache
###############################################################################

echo ""
echo "============================================================"
echo "实验 B：BF16 权重 + FP8 KV Cache（长上下文高并发）"
echo "============================================================"

vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --kv-cache-dtype fp8 \
    > "$OUTPUT_DIR/exp7b_fp8kv_long_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp7b_fp8kv_long_server.log"; then
    # 预热
    run_benchmark "$MODEL" "$OUTPUT_DIR/exp7b_fp8kv_long_warmup.txt" 20 4096 512
    # 正式测试
    run_benchmark "$MODEL" "$OUTPUT_DIR/exp7b_fp8kv_long_bench.txt" 500 4096 512
fi

stop_server $SERVER_PID

###############################################################################
# 结果汇总
###############################################################################

echo ""
echo "============================================================"
echo "结果汇总"
echo "============================================================"
echo ""
for f in exp7a_bf16kv_long_bench.txt exp7b_fp8kv_long_bench.txt; do
    echo "=== $f ==="
    grep -E "Successful|Failed|duration|throughput|Mean TTFT|Mean TPOT|P99 TPOT" "$OUTPUT_DIR/$f" 2>/dev/null
    echo ""
done

echo "完成！"
