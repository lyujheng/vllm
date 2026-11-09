#!/bin/bash
# 实验3: NCCL 通信日志分析
# 抓取 TP=4 时的 NCCL 通信日志，观察用了哪些通信原语、走 NVLink 还是 PCIe

MODEL="/data/model/Qwen2.5-7B"
OUTPUT_DIR="/home/lvzheng/vllm/learning/week4-distributed/results"
mkdir -p $OUTPUT_DIR

echo "============================================"
echo "Running TP=4 with NCCL_DEBUG=INFO"
echo "============================================"

# 只跑少量请求，主要看 NCCL 日志
NCCL_DEBUG=INFO python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='$MODEL',
    tensor_parallel_size=4,
    gpu_memory_utilization=0.8,
)

prompts = [
    'What is the capital of France?',
    'Explain quantum computing in simple terms.',
    'Write a short poem about the ocean.',
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=50, temperature=0.8))

for i, output in enumerate(outputs):
    print(f'--- Prompt {i} ---')
    print(f'Output: {output.outputs[0].text[:100]}')
    print()
" 2>&1 | tee $OUTPUT_DIR/nccl_debug.log

# 提取关键信息
echo ""
echo "=== NCCL Transport Summary ==="
grep -i "transport\|nvlink\|pcie\|ib\|net\|shm\|p2p" $OUTPUT_DIR/nccl_debug.log | head -30
echo ""
echo "=== NCCL AllReduce/AllGather ==="
grep -i "allreduce\|allgather\|reducescatter\|broadcast" $OUTPUT_DIR/nccl_debug.log | head -20
echo ""
echo "Full log: $OUTPUT_DIR/nccl_debug.log"
