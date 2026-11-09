#!/bin/bash
# EPLB 实验：持续高负载触发 EPLB 重均衡
# 需要足够多的 forward step（每个 decode step 算 1 个 forward step）
# 用大量请求 + 长输出 + 高并发确保持续负载
set -e

export no_proxy=localhost,127.0.0.1

SERVER=http://localhost:8300
RESULTS_DIR=/home/lvzheng/vllm/results/week8_eplb
mkdir -p $RESULTS_DIR

echo "=== 等待服务就绪 ==="
for i in $(seq 1 120); do
    if curl -s -o /dev/null -w "%{http_code}" $SERVER/health | grep -q "200"; then
        echo "服务已就绪 (${i}s)"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "超时，服务未就绪"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== 用 vllm bench serve 持续高并发打压 ==="
echo "2000 个请求, output_len=256, request_rate=inf (无限速)"
echo "step_interval=10, 需要 >=10 个 forward step 触发重均衡"
echo ""

cd /tmp

vllm bench serve \
    --model /data/model/Qwen2-57B-A14B-Instruct \
    --dataset-name random \
    --random-input-len 64 \
    --random-output-len 256 \
    --num-prompts 2000 \
    --request-rate inf \
    --port 8300 \
    2>&1 | tee $RESULTS_DIR/bench_results.txt

echo ""
echo "=== Benchmark 完成 ==="
echo ""

# 提取 EPLB 相关日志
echo "=== EPLB 相关日志摘录 ===" | tee $RESULTS_DIR/eplb_logs.txt
grep -i "EPLB step\|Rearranging\|rearrange\|balancedness\|Skip rearrange\|expert.*migrat" $RESULTS_DIR/server.log >> $RESULTS_DIR/eplb_logs.txt 2>/dev/null || echo "(未找到 EPLB 重均衡日志)" | tee -a $RESULTS_DIR/eplb_logs.txt

echo ""
cat $RESULTS_DIR/eplb_logs.txt
