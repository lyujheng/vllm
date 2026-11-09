"""
Week 6 Experiment 2: FusedMoE Kernel Profiling
目标：分析 MoE 层在不同 batch size 下的性能特征
"""

import torch
import time
from typing import List, Dict
import sys
import os

# 清理 Python 路径，避免导入本地源码
if '/home/lvzheng/vllm' in sys.path:
    sys.path.remove('/home/lvzheng/vllm')

# 检查 vLLM 是否可导入
try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
except ImportError:
    print("Error: Cannot import fused_experts from vLLM")
    print("Make sure vLLM is installed: pip install vllm")
    sys.exit(1)


def profile_fused_moe(
    num_tokens: int,
    hidden_size: int = 4096,
    num_experts: int = 64,
    top_k: int = 8,
    intermediate_size: int = 14336,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> Dict[str, float]:
    """
    Profile FusedMoE kernel performance

    Args:
        num_tokens: Number of tokens in batch
        hidden_size: Hidden dimension
        num_experts: Total number of experts
        top_k: Number of experts to activate per token
        intermediate_size: FFN intermediate size
        num_warmup: Warmup iterations
        num_iters: Benchmark iterations

    Returns:
        Performance metrics
    """
    device = torch.device("cuda:0")

    # 准备输入
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float16)

    # Router 输出 (模拟)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)

    # Expert 权重 (模拟 gate_up_proj 和 down_proj)
    w1 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, device=device, dtype=torch.float16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(num_warmup):
        _ = fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
        )

    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(num_iters):
        output = fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
        )
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / num_iters * 1000

    # 计算理论 FLOP
    # 每个 token 激活 top_k 个 expert
    # 每个 expert: 2 次 GEMM (gate_up 和 down)
    flops_per_token = top_k * (
        2 * hidden_size * intermediate_size +  # gate_up_proj
        2 * intermediate_size * hidden_size     # down_proj
    )
    total_flops = num_tokens * flops_per_token
    tflops = (total_flops / avg_time_ms / 1e9)  # TFLOP/s

    return {
        "num_tokens": num_tokens,
        "avg_time_ms": avg_time_ms,
        "tflops": tflops,
        "throughput_tokens_per_sec": num_tokens / (avg_time_ms / 1000),
    }


def main():
    print("=" * 60)
    print("Week 6 Exp2: FusedMoE Kernel Profiling")
    print("=" * 60)
    print()

    # 检查 GPU
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 配置 (模拟 Qwen2-57B-A14B MoE 层)
    config = {
        "hidden_size": 4096,
        "num_experts": 64,
        "top_k": 8,
        "intermediate_size": 14336,
    }

    print("MoE Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # 测试不同 batch size
    batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]

    print("Profiling Results:")
    print("-" * 60)
    print(f"{'Batch Size':<12} {'Time (ms)':<12} {'TFLOP/s':<12} {'Tok/s':<12}")
    print("-" * 60)

    results = []
    for bs in batch_sizes:
        try:
            result = profile_fused_moe(
                num_tokens=bs,
                **config,
                num_warmup=5,
                num_iters=50,
            )
            results.append(result)

            print(f"{result['num_tokens']:<12} "
                  f"{result['avg_time_ms']:<12.2f} "
                  f"{result['tflops']:<12.2f} "
                  f"{result['throughput_tokens_per_sec']:<12.1f}")
        except Exception as e:
            print(f"{bs:<12} Error: {str(e)}")

    print("-" * 60)
    print()

    # 分析
    if results:
        print("Analysis:")
        print(f"  Peak TFLOP/s: {max(r['tflops'] for r in results):.2f} "
              f"(at batch_size={max(results, key=lambda r: r['tflops'])['num_tokens']})")

        # GPU 利用率估算 (假设 A100 理论峰值 312 TFLOP/s FP16)
        theoretical_peak = 312  # A100 FP16 Tensor Core
        peak_tflops = max(r['tflops'] for r in results)
        utilization = (peak_tflops / theoretical_peak) * 100
        print(f"  Estimated GPU Utilization: {utilization:.1f}% (assuming A100)")

        # 找到吞吐饱和点
        throughputs = [r['tflops'] for r in results]
        for i in range(1, len(throughputs)):
            if throughputs[i] < throughputs[i-1] * 1.05:  # 增长 < 5%
                print(f"  Throughput saturation at batch_size ≈ {results[i-1]['num_tokens']}")
                break

    print()
    print("Next step: Fill day4_profiling_analysis.md with observations")


if __name__ == "__main__":
    main()
