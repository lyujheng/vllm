#!/bin/bash
# 实验 07: DSpark 不同 num_speculative_tokens 对比
# 观察不同 draft 长度对接受率和性能的影响
#
# 环境要求：
#   - /data/model/Qwen3-8B (target 模型)
#   - /data/model/dspark_qwen3_8b_block7 (DSpark draft 模型)
#
# 实验设计：
#   K=1, 3, 5, 7 四组对比
#   观察 TTFT、TPOT、Throughput 随 K 的变化趋势

set -e

export CUDA_VISIBLE_DEVICES=2

TARGET_MODEL="/data/model/Qwen3-8B"
DRAFT_MODEL="/data/model/dspark_qwen3_8b_block7"
OUTPUT_DIR="/home/lvzheng/vllm/results/day5_dspark_k_sweep"
SONNET_PATH="/home/lvzheng/vllm/benchmarks/sonnet.txt"
PORT=8300

mkdir -p $OUTPUT_DIR
export no_proxy=localhost,127.0.0.1

for K in 1 3 5 7; do
    echo "============================================"
    echo "测试 num_speculative_tokens = $K"
    echo "============================================"

    # 清理端口
    kill -9 $(lsof -t -i:$PORT) 2>/dev/null || true
    sleep 5

    vllm serve $TARGET_MODEL \
        --port $PORT \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 64 \
        --speculative-config '{"model": "'$DRAFT_MODEL'", "num_speculative_tokens": '$K'}' \
        > $OUTPUT_DIR/k${K}_server.log 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    echo "等待服务就绪..."
    for i in $(seq 1 180); do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null | grep -q "200"; then
            echo "✓ Server ready!"
            break
        fi
        if [ $i -eq 180 ]; then
            echo "✗ Server failed to start for K=$K. Check $OUTPUT_DIR/k${K}_server.log"
            kill $SERVER_PID 2>/dev/null
            continue 2
        fi
        sleep 5
    done

    echo "运行 Benchmark (K=$K)..."
    vllm bench serve \
        --model $TARGET_MODEL \
        --dataset-name sonnet \
        --dataset-path $SONNET_PATH \
        --sonnet-input-len 256 \
        --sonnet-output-len 128 \
        --num-prompts 200 \
        --request-rate 10 \
        --port $PORT \
        2>&1 | tee $OUTPUT_DIR/k${K}_bench.txt

    echo ""
    echo "=== K=$K Results ==="
    grep -E "Throughput|Mean TTFT|Mean TPOT|P99" $OUTPUT_DIR/k${K}_bench.txt

    # 清理
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 10
done

echo ""
echo "============================================"
echo "所有 K 值对比汇总"
echo "============================================"
for K in 1 3 5 7; do
    echo ""
    echo "--- K=$K ---"
    grep -E "Throughput|Mean TTFT|Mean TPOT" $OUTPUT_DIR/k${K}_bench.txt 2>/dev/null || echo "(no results)"
done
echo ""
echo "Done! 详细结果在 $OUTPUT_DIR/"
