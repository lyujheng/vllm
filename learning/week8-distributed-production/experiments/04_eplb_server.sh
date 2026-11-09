#!/bin/bash
# EPLB 实验：TP=2 DP=1 → EP=2（单 EngineCore，避免 DP 跨进程同步问题）
# 模型：Qwen2-57B-A14B-Instruct (64 experts, top-8)

set -e

MODEL=/data/model/Qwen2-57B-A14B-Instruct
RESULTS_DIR=/home/lvzheng/vllm/results/week8_eplb
mkdir -p $RESULTS_DIR

# 清理残留进程
pkill -f "vllm serve\|api_server" 2>/dev/null || true
sleep 2

echo "=== 启动 vLLM with EPLB ==="
echo "TP=2, DP=1, EP=2 (单 EngineCore)"
echo "num_redundant_experts=8, step_interval=10, window_size=5, log_balancedness=true"
echo ""

export no_proxy=localhost,127.0.0.1

cd /tmp

CUDA_VISIBLE_DEVICES=0,1 vllm serve $MODEL \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-config '{"num_redundant_experts": 8, "step_interval": 10, "window_size": 5, "log_balancedness": true}' \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --port 8300 \
    2>&1 | tee $RESULTS_DIR/server.log
