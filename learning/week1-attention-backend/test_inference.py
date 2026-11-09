"""Week 1 Day 4: 基础推理测试"""

from vllm import LLM, SamplingParams

# 使用预下载的本地模型
llm = LLM(
    model="/data/vllm/model/opt-125m",  # 本地路径
    max_num_seqs=2,                      # 允许 2 个并发请求
    block_size=16,                       # 观察 block 分配
    max_model_len=512,                   # 限制序列长度
    tensor_parallel_size=1,              # 使用 1 块 GPU（默认值）
    # tensor_parallel_size=2,            # 如果要用 2 块 GPU，取消注释
)

# 测试场景 1：单个请求（观察 Prefill → Decode 过程）
print("=" * 60)
print("测试 1: 单个请求")
print("=" * 60)

prompts = ["Hello, my name is"]
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=100,
    repetition_penalty=1.2,  # 惩罚重复，减少循环
    stop=["\n\n", "."],      # 遇到双换行或句号就停止
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    finish_reason = output.outputs[0].finish_reason  # 查看停止原因
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print(f"Finish reason: {finish_reason}")  # 'stop' (遇到EOS) 或 'length' (达到max_tokens)
    print()

# 测试场景 2：多个并发请求（观察 Continuous Batching）
print("=" * 60)
print("测试 2: 并发请求")
print("=" * 60)

prompts = [
    "The capital of France is",
    "Python is a programming language that",
]

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Request {i}:")
    print(f"  Prompt: {output.prompt}")
    print(f"  Generated: {output.outputs[0].text}")
    print()

print("=" * 60)
print("推理完成")
print("=" * 60)
