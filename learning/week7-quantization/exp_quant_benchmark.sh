#!/bin/bash
# Week 7 Day 7 实验：量化方案性能对比
# 实验目标：
#   1. BF16 基线
#   2. FP8 在线量化（per-tensor）
#   3. FP8 在线量化（per-block）
#   4. INT4 AWQ 离线量化
#   5. INT4 GPTQ 离线量化
#   6. FP8 KV Cache（BF16 权重 + FP8 KV Cache）
#
# 硬件：H800（单卡）
# 模型：Qwen2.5-7B / Qwen2.5-7B-Instruct-AWQ / Qwen2.5-7B-Instruct-GPTQ-Int4

set -e

MODEL_BF16="/data/model/Qwen2.5-7B"
MODEL_AWQ="/data/model/Qwen2.5-7B-Instruct-AWQ"
MODEL_GPTQ="/data/model/Qwen2.5-7B-Instruct-GPTQ-Int4"
OUTPUT_DIR="/home/lvzheng/vllm/results/week7-quant"
PORT=8000

export no_proxy=localhost,127.0.0.1

mkdir -p "$OUTPUT_DIR"

###############################################################################
# 工具函数
###############################################################################

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
    local tokenizer=$2
    local output_file=$3
    local num_prompts=${4:-100}
    local request_rate=${5:-inf}

    echo "  Benchmark: model=$model_name, prompts=$num_prompts, rate=$request_rate"
    vllm bench serve \
        --model "$model_name" \
        --tokenizer "$tokenizer" \
        --dataset-name random \
        --random-input-len 256 \
        --random-output-len 128 \
        --num-prompts "$num_prompts" \
        --request-rate "$request_rate" \
        --port "$PORT" \
        2>&1 | tee "$output_file"
}

###############################################################################
# 实验 1：BF16 基线
###############################################################################

echo "============================================================"
echo "实验 1：BF16 基线"
echo "============================================================"

vllm serve "$MODEL_BF16" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    > "$OUTPUT_DIR/exp1_bf16_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp1_bf16_server.log"; then
    # 预热
    run_benchmark "$MODEL_BF16" "$MODEL_BF16" "$OUTPUT_DIR/exp1_bf16_warmup.txt" 20 inf
    # 正式测试
    run_benchmark "$MODEL_BF16" "$MODEL_BF16" "$OUTPUT_DIR/exp1_bf16_bench.txt" 100 inf
fi

stop_server $SERVER_PID

###############################################################################
# 实验 2：FP8 在线量化（per-tensor）
###############################################################################

echo ""
echo "============================================================"
echo "实验 2：FP8 per-tensor 在线量化"
echo "============================================================"

vllm serve "$MODEL_BF16" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --quantization fp8_per_tensor \
    > "$OUTPUT_DIR/exp2_fp8_pt_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp2_fp8_pt_server.log"; then
    run_benchmark "$MODEL_BF16" "$MODEL_BF16" "$OUTPUT_DIR/exp2_fp8_pt_warmup.txt" 20 inf
    run_benchmark "$MODEL_BF16" "$MODEL_BF16" "$OUTPUT_DIR/exp2_fp8_pt_bench.txt" 100 inf
fi

stop_server $SERVER_PID

###############################################################################
# 实验 3：FP8 在线量化（per-block）
###############################################################################

echo ""
echo "============================================================"
echo "实验 3：FP8 per-block 在线量化"
echo "============================================================"

vllm serve "$MODEL_BF16" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --quantization fp8_per_block \
    > "$OUTPUT_DIR/exp3_fp8_pb_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp3_fp8_pb_server.log"; then
    run_benchmark "$MODEL_BF16" "$MODEL_BF16" "$OUTPUT_DIR/exp3_fp8_pb_warmup.txt" 20 inf
    run_benchmark "$MODEL_BF16" "$MODEL_BF16" "$OUTPUT_DIR/exp3_fp8_pb_bench.txt" 100 inf
fi

stop_server $SERVER_PID

###############################################################################
# 实验 4：INT4 AWQ 离线量化
###############################################################################

echo ""
echo "============================================================"
echo "实验 4：INT4 AWQ"
echo "============================================================"

vllm serve "$MODEL_AWQ" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    > "$OUTPUT_DIR/exp4_awq_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp4_awq_server.log"; then
    run_benchmark "$MODEL_AWQ" "$MODEL_AWQ" "$OUTPUT_DIR/exp4_awq_warmup.txt" 20 inf
    run_benchmark "$MODEL_AWQ" "$MODEL_AWQ" "$OUTPUT_DIR/exp4_awq_bench.txt" 100 inf
fi

stop_server $SERVER_PID

###############################################################################
# 实验 5：INT4 GPTQ 离线量化
###############################################################################

echo ""
echo "============================================================"
echo "实验 5：INT4 GPTQ"
echo "============================================================"

vllm serve "$MODEL_GPTQ" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    > "$OUTPUT_DIR/exp5_gptq_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp5_gptq_server.log"; then
    run_benchmark "$MODEL_GPTQ" "$MODEL_GPTQ" "$OUTPUT_DIR/exp5_gptq_warmup.txt" 20 inf
    run_benchmark "$MODEL_GPTQ" "$MODEL_GPTQ" "$OUTPUT_DIR/exp5_gptq_bench.txt" 100 inf
fi

stop_server $SERVER_PID

###############################################################################
# 实验 6：BF16 权重 + FP8 KV Cache
###############################################################################

echo ""
echo "============================================================"
echo "实验 6：BF16 + FP8 KV Cache"
echo "============================================================"

vllm serve "$MODEL_BF16" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --kv-cache-dtype fp8_e4m3 \
    > "$OUTPUT_DIR/exp6_fp8kv_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp6_fp8kv_server.log"; then
    run_benchmark "$MODEL_BF16" "$MODEL_BF16" "$OUTPUT_DIR/exp6_fp8kv_warmup.txt" 20 inf
    run_benchmark "$MODEL_BF16" "$MODEL_BF16" "$OUTPUT_DIR/exp6_fp8kv_bench.txt" 100 inf
fi

stop_server $SERVER_PID

###############################################################################

echo ""
echo "============================================================"
echo "所有实验完成！结果目录: $OUTPUT_DIR"
echo "============================================================"
ls -lh "$OUTPUT_DIR"
