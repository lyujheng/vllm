"""
Week 7 Day 7: LoRA 混合 adapter 并发请求实验
向 vLLM 服务交替发送两种 adapter 请求，统计各 adapter 的延迟和吞吐。
"""

import argparse
import asyncio
import json
import time

import aiohttp


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    first_token_time = None
    generated_tokens = 0

    # 使用 streaming 来测量 TTFT
    payload["stream"] = True
    try:
        async with session.post(url, json=payload) as resp:
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data:"):
                    continue
                data_str = decoded[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                generated_tokens += 1
    except Exception as e:
        return {
            "request_id": request_id,
            "model": model,
            "error": str(e),
        }

    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else None
    total_time = end - start
    tpot = ((end - first_token_time) / max(generated_tokens - 1, 1)
            if first_token_time and generated_tokens > 1 else None)

    return {
        "request_id": request_id,
        "model": model,
        "ttft": ttft,
        "tpot": tpot,
        "total_time": total_time,
        "generated_tokens": generated_tokens,
    }


async def run_mixed_benchmark(
    port: int,
    num_prompts: int,
    max_tokens: int = 128,
) -> list[dict]:
    url = f"http://localhost:{port}/v1/completions"
    adapters = ["adapter-reasoning", "adapter-rp"]
    prompt = "Explain the concept of machine learning in simple terms."

    tasks = []
    async with aiohttp.ClientSession() as session:
        for i in range(num_prompts):
            model = adapters[i % 2]  # 交替使用两个 adapter
            tasks.append(
                send_request(session, url, model, prompt, max_tokens, i)
            )
        # 全部并发发出
        results = await asyncio.gather(*tasks)

    return list(results)


def print_summary(results: list[dict]):
    by_model: dict[str, list[dict]] = {}
    for r in results:
        if "error" in r:
            print(f"  请求 {r['request_id']} ({r['model']}): 错误 - {r['error']}")
            continue
        by_model.setdefault(r["model"], []).append(r)

    print("\n===== 混合并发请求结果汇总 =====\n")
    for model, items in sorted(by_model.items()):
        ttfts = [r["ttft"] for r in items if r["ttft"] is not None]
        tpots = [r["tpot"] for r in items if r["tpot"] is not None]
        totals = [r["total_time"] for r in items]
        tokens = sum(r["generated_tokens"] for r in items)

        print(f"--- {model} ({len(items)} 请求) ---")
        if ttfts:
            ttfts.sort()
            print(f"  TTFT  均值: {sum(ttfts)/len(ttfts)*1000:.1f} ms"
                  f"  P50: {ttfts[len(ttfts)//2]*1000:.1f} ms"
                  f"  P99: {ttfts[int(len(ttfts)*0.99)]*1000:.1f} ms")
        if tpots:
            tpots.sort()
            print(f"  TPOT  均值: {sum(tpots)/len(tpots)*1000:.1f} ms"
                  f"  P50: {tpots[len(tpots)//2]*1000:.1f} ms"
                  f"  P99: {tpots[int(len(tpots)*0.99)]*1000:.1f} ms")
        total_time = max(totals) if totals else 0
        print(f"  总 tokens: {tokens}"
              f"  吞吐: {tokens/total_time:.1f} tok/s" if total_time > 0 else "")
        print()

    # 总体统计
    all_ok = [r for r in results if "error" not in r]
    if all_ok:
        wall_time = max(r["total_time"] for r in all_ok)
        total_tokens = sum(r["generated_tokens"] for r in all_ok)
        print(f"总体: {len(all_ok)} 请求成功, "
              f"总 tokens={total_tokens}, "
              f"端到端吞吐={total_tokens/wall_time:.1f} tok/s")
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"失败: {len(errors)} 请求")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 混合 adapter 并发请求实验")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出 JSON 文件路径")
    args = parser.parse_args()

    print(f"发送 {args.num_prompts} 个混合请求 "
          f"(adapter-reasoning / adapter-rp 交替)...")

    results = asyncio.run(
        run_mixed_benchmark(args.port, args.num_prompts, args.max_tokens)
    )

    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
