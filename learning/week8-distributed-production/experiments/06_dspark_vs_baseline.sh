#!/bin/bash
# 实验 06: DSpark 投机解码 vs 无投机解码 对比实验
# 对比 Qwen3-8B 在有/无 DSpark 投机解码下的性能差异
#
# 环境要求：
#   - /data/model/Qwen3-8B (target 模型)
#   - /data/model/dspark_qwen3_8b_block7 (DSpark draft 模型)
#   - 8x H800 GPU
#
# 实验设计：
#   实验 A: Qwen3-8B 无投机解码 (baseline)
#   实验 B: Qwen3-8B + DSpark (num_speculative_tokens=7)
#   对比指标: Throughput, TTFT, TPOT

set -e

export CUDA_VISIBLE_DEVICES=2

TARGET_MODEL="/data/model/Qwen3-8B"
DRAFT_MODEL="/data/model/dspark_qwen3_8b_block7"
OUTPUT_DIR="/home/lvzheng/vllm/results/day5_dspark"
SONNET_PATH="/home/lvzheng/vllm/benchmarks/sonnet.txt"
PORT=8300

mkdir -p $OUTPUT_DIR
export no_proxy=localhost,127.0.0.1

# 清理可能残留的进程
kill -9 $(lsof -t -i:$PORT) 2>/dev/null || true
sleep 5

echo "============================================"
echo "实验 A: Baseline（无投机解码）"
echo "============================================"


vllm serve $TARGET_MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    > $OUTPUT_DIR/baseline_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "等待服务就绪..."
for i in $(seq 1 120); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null | grep -q "200"; then
        echo "✓ Server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "✗ Server failed to start. Check $OUTPUT_DIR/baseline_server.log"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done

echo "运行 Benchmark (baseline)..."
vllm bench serve \
    --model $TARGET_MODEL \
    --dataset-name sonnet \
    --dataset-path $SONNET_PATH \
    --sonnet-input-len 256 \
    --sonnet-output-len 128 \
    --num-prompts 200 \
    --request-rate 10 \
    --port $PORT \
    2>&1 | tee $OUTPUT_DIR/baseline_bench.txt

echo ""
echo "=== Baseline Results ==="
grep -E "Throughput|Mean TTFT|Mean TPOT|P99" $OUTPUT_DIR/baseline_bench.txt

# 清理
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 10

echo ""
echo "============================================"
echo "实验 B: DSpark 投机解码 (num_speculative_tokens=7)"
echo "============================================"

# 清理端口
kill -9 $(lsof -t -i:$PORT) 2>/dev/null || true
sleep 5


vllm serve $TARGET_MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --speculative-config '{"model": "'$DRAFT_MODEL'", "num_speculative_tokens": 7}' \
    > $OUTPUT_DIR/dspark_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "等待服务就绪（DSpark 加载可能需要更长时间）..."
for i in $(seq 1 180); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null | grep -q "200"; then
        echo "✓ Server ready!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "✗ Server failed to start. Check $OUTPUT_DIR/dspark_server.log"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done

echo "运行 Benchmark (DSpark)..."
vllm bench serve \
    --model $TARGET_MODEL \
    --dataset-name sonnet \
    --dataset-path $SONNET_PATH \
    --sonnet-input-len 256 \
    --sonnet-output-len 128 \
    --num-prompts 200 \
    --request-rate 10 \
    --port $PORT \
    2>&1 | tee $OUTPUT_DIR/dspark_bench.txt

echo ""
echo "=== DSpark Results ==="
grep -E "Throughput|Mean TTFT|Mean TPOT|P99" $OUTPUT_DIR/dspark_bench.txt

# 清理
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo ""
echo "============================================"
echo "对比结果汇总"
echo "============================================"
echo ""
echo "--- Baseline ---"
grep -E "Throughput|Mean TTFT|Mean TPOT" $OUTPUT_DIR/baseline_bench.txt
echo ""
echo "--- DSpark (K=7) ---"
grep -E "Throughput|Mean TTFT|Mean TPOT" $OUTPUT_DIR/dspark_bench.txt
echo ""
echo "Done! 详细结果在 $OUTPUT_DIR/"
