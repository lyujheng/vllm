#!/bin/bash
# Week 7 Day 7 实验：LoRA 多 Adapter 并发服务
# 实验目标：
#   1. 单 adapter 基线性能
#   2. 双 adapter 并发服务（max-loras=2）
#   3. max-loras=1 vs max-loras=2 调度行为对比
#
# 硬件：8×H800，实验使用单卡即可

set -e

MODEL="/data/model/Qwen2.5-7B"
ADAPTER_A="/data/model/lora-adapters/adapter-reasoning"
ADAPTER_B="/data/model/lora-adapters/adapter-rp"
OUTPUT_DIR="/home/lvzheng/vllm/results/week7-lora"
PORT=8000

export no_proxy=localhost,127.0.0.1

mkdir -p "$OUTPUT_DIR"

###############################################################################
# 工具函数
###############################################################################

wait_server_ready() {
    local logfile=$1
    local timeout=${2:-180}
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
    # 确保端口释放
    if lsof -t -i:"$PORT" >/dev/null 2>&1; then
        kill -9 "$(lsof -t -i:"$PORT")" 2>/dev/null
        sleep 3
    fi
}

run_benchmark() {
    local model_name=$1
    local output_file=$2
    local num_prompts=${3:-50}
    local request_rate=${4:-inf}

    echo "  Benchmark: model=$model_name, prompts=$num_prompts, rate=$request_rate"
    vllm bench serve \
        --model "$model_name" \
        --tokenizer "$MODEL" \
        --dataset-name random \
        --random-input-len 256 \
        --random-output-len 128 \
        --num-prompts "$num_prompts" \
        --request-rate "$request_rate" \
        --port "$PORT" \
        2>&1 | tee "$output_file"
}

###############################################################################
# 实验 1：无 LoRA 基线
###############################################################################

echo "============================================================"
echo "实验 1：无 LoRA 基线"
echo "============================================================"

vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    > "$OUTPUT_DIR/exp1_baseline_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp1_baseline_server.log"; then
    run_benchmark "$MODEL" "$OUTPUT_DIR/exp1_baseline_bench.txt" 50 inf
fi

stop_server $SERVER_PID

###############################################################################
# 实验 2：单 adapter 服务
###############################################################################

echo ""
echo "============================================================"
echo "实验 2：单 adapter 服务 (adapter-reasoning)"
echo "============================================================"

vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 16 \
    --lora-modules adapter-reasoning="$ADAPTER_A" \
    > "$OUTPUT_DIR/exp2_single_lora_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp2_single_lora_server.log"; then
    # 用 adapter 名称作为 model 参数发请求
    run_benchmark "adapter-reasoning" "$OUTPUT_DIR/exp2_single_lora_bench.txt" 50 inf
fi

stop_server $SERVER_PID

###############################################################################
# 实验 3：双 adapter 并发，max-loras=2
###############################################################################

echo ""
echo "============================================================"
echo "实验 3：双 adapter 并发 (max-loras=2)"
echo "============================================================"

vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank 16 \
    --lora-modules adapter-reasoning="$ADAPTER_A" adapter-rp="$ADAPTER_B" \
    > "$OUTPUT_DIR/exp3_dual_lora_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp3_dual_lora_server.log"; then
    echo "--- 3a: 先发 adapter-reasoning 请求 ---"
    run_benchmark "adapter-reasoning" "$OUTPUT_DIR/exp3a_reasoning_only.txt" 50 inf

    echo ""
    echo "--- 3b: 再发 adapter-rp 请求 ---"
    run_benchmark "adapter-rp" "$OUTPUT_DIR/exp3b_rp_only.txt" 50 inf

    echo ""
    echo "--- 3c: 对调顺序，先发 adapter-rp 请求 ---"
    run_benchmark "adapter-rp" "$OUTPUT_DIR/exp3c_rp_first.txt" 50 inf

    echo ""
    echo "--- 3d: 对调顺序，再发 adapter-reasoning 请求 ---"
    run_benchmark "adapter-reasoning" "$OUTPUT_DIR/exp3d_reasoning_second.txt" 50 inf
fi

stop_server $SERVER_PID

###############################################################################
# 实验 4：双 adapter 混合并发请求（Python 脚本）
###############################################################################

echo ""
echo "============================================================"
echo "实验 4：双 adapter 混合并发请求"
echo "============================================================"

vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank 16 \
    --lora-modules adapter-reasoning="$ADAPTER_A" adapter-rp="$ADAPTER_B" \
    > "$OUTPUT_DIR/exp4_mixed_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp4_mixed_server.log"; then
    python3 /home/lvzheng/vllm/learning/week7-quantization/exp_lora_mixed_requests.py \
        --port "$PORT" \
        --num-prompts 100 \
        --output "$OUTPUT_DIR/exp4_mixed_results.json"
fi

stop_server $SERVER_PID

###############################################################################
# 实验 5：max-loras=1 + 双 adapter 注册，观察调度限制
###############################################################################

echo ""
echo "============================================================"
echo "实验 5：max-loras=1 + 双 adapter（调度瓶颈观察）"
echo "============================================================"

vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 16 \
    --max-cpu-loras 2 \
    --lora-modules adapter-reasoning="$ADAPTER_A" adapter-rp="$ADAPTER_B" \
    > "$OUTPUT_DIR/exp5_maxlora1_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if wait_server_ready "$OUTPUT_DIR/exp5_maxlora1_server.log"; then
    python3 /home/lvzheng/vllm/learning/week7-quantization/exp_lora_mixed_requests.py \
        --port "$PORT" \
        --num-prompts 100 \
        --output "$OUTPUT_DIR/exp5_maxlora1_results.json"
fi

stop_server $SERVER_PID

###############################################################################
# 汇总
###############################################################################

echo ""
echo "============================================================"
echo "全部实验完成！结果目录: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "结果文件:"
ls -la "$OUTPUT_DIR"/*.txt "$OUTPUT_DIR"/*.json 2>/dev/null
echo ""
echo "关键指标对比（从 benchmark 输出中提取）:"
for f in "$OUTPUT_DIR"/exp*_bench.txt "$OUTPUT_DIR"/exp3*.txt; do
    if [ -f "$f" ]; then
        echo "--- $(basename "$f") ---"
        grep -E "Throughput|Mean TTFT|Mean TPOT|P99" "$f" 2>/dev/null || true
        echo ""
    fi
done
