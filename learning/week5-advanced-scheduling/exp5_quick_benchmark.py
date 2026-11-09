#!/usr/bin/env python3
"""
Week 5 Day 7 实验5：快速性能验证
使用 vLLM Python API 直接测试，无需启动 server
"""

import time
import json
from typing import List, Dict
from vllm import LLM, SamplingParams

RESULTS_DIR = "/home/lvzheng/vllm/learning/week5-advanced-scheduling/results"

def run_benchmark(
    config_name: str,
    model_name: str,
    prompts: List[str],
    **llm_kwargs
) -> Dict:
    """运行一次 benchmark"""

    print(f"\n{'='*60}")
    print(f"配置: {config_name}")
    print(f"参数: {llm_kwargs}")
    print(f"{'='*60}")

    # 初始化模型
    start_init = time.time()
    llm = LLM(model=model_name, **llm_kwargs)
    init_time = time.time() - start_init
    print(f"模型加载耗时: {init_time:.2f}s")

    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
    )

    # Warmup
    print("Warmup...")
    _ = llm.generate(prompts[:1], sampling_params)

    # 实际测试
    results = []
    for i, prompt in enumerate(prompts):
        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        latency = time.time() - start

        output = outputs[0]
        num_tokens = len(output.outputs[0].token_ids)

        results.append({
            "prompt_id": i,
            "latency": latency,
            "num_tokens": num_tokens,
            "tokens_per_sec": num_tokens / latency if latency > 0 else 0
        })

        print(f"  Prompt {i}: {latency:.3f}s, {num_tokens} tokens, "
              f"{num_tokens/latency:.2f} tok/s")

    # 汇总
    total_tokens = sum(r["num_tokens"] for r in results)
    total_time = sum(r["latency"] for r in results)
    avg_throughput = total_tokens / total_time if total_time > 0 else 0

    summary = {
        "config": config_name,
        "init_time": init_time,
        "results": results,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_throughput": avg_throughput
    }

    print(f"\n总结:")
    print(f"  总 tokens: {total_tokens}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均吞吐: {avg_throughput:.2f} tok/s")

    return summary


def main():
    MODEL = "/data/model/Qwen2.5-7B"

    # 测试 prompts - 包含相同 prefix 以测试 prefix caching
    PROMPTS = [
        "你是一个专业的AI助手。请简要介绍什么是机器学习？",
        "你是一个专业的AI助手。请简要介绍什么是深度学习？",
        "你是一个专业的AI助手。请简要介绍什么是强化学习？",
        "你是一个专业的AI助手。请简要介绍什么是自然语言处理？",
    ]

    all_results = []

    # 配置1: Baseline - 所有优化关闭
    print("\n" + "="*60)
    print("开始实验 - 对比不同优化配置")
    print("="*60)

    result1 = run_benchmark(
        config_name="baseline",
        model_name=MODEL,
        prompts=PROMPTS,
        enforce_eager=True,  # 关闭 CUDA Graph
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    all_results.append(result1)

    # 配置2: 只开启 CUDA Graph
    result2 = run_benchmark(
        config_name="cudagraph_only",
        model_name=MODEL,
        prompts=PROMPTS,
        enforce_eager=False,  # 开启 CUDA Graph
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    all_results.append(result2)

    # 配置3: CUDA Graph + Prefix Caching
    result3 = run_benchmark(
        config_name="cudagraph_prefix",
        model_name=MODEL,
        prompts=PROMPTS,
        enforce_eager=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    all_results.append(result3)

    # 配置4: 所有优化 (CUDA Graph + Prefix Caching + Chunked Prefill)
    result4 = run_benchmark(
        config_name="all_optimizations",
        model_name=MODEL,
        prompts=PROMPTS,
        enforce_eager=False,
        enable_prefix_caching=True,
        max_num_batched_tokens=512,  # Chunked prefill
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    all_results.append(result4)

    # 生成对比报告
    print("\n" + "="*80)
    print("性能对比总结")
    print("="*80)
    print(f"{'配置':<25} {'平均吞吐(tok/s)':<20} {'相对加速':<15} {'初始化(s)':<12}")
    print("-"*80)

    baseline_throughput = all_results[0]["avg_throughput"]

    for result in all_results:
        speedup = result["avg_throughput"] / baseline_throughput if baseline_throughput > 0 else 1.0
        print(f"{result['config']:<25} "
              f"{result['avg_throughput']:<20.2f} "
              f"{speedup:<15.2f}x "
              f"{result['init_time']:<12.2f}")

    print("="*80)

    # 保存结果
    output_file = f"{RESULTS_DIR}/exp5_quick_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n结果已保存到: {output_file}")

    # 生成建议
    print("\n" + "="*80)
    print("优化建议")
    print("="*80)

    best_config = max(all_results, key=lambda x: x["avg_throughput"])
    print(f"最佳配置: {best_config['config']}")
    print(f"最高吞吐: {best_config['avg_throughput']:.2f} tok/s")
    print(f"相比 baseline 提升: {best_config['avg_throughput']/baseline_throughput:.2f}x")

    print("\n关键发现:")
    print("1. CUDA Graph 对 decode 阶段的加速效果")
    print("2. Prefix Caching 对相同前缀请求的优化")
    print("3. Chunked Prefill 对长 prompt 的调度优化")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()
