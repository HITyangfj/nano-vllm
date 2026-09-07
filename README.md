<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

## Decode 优先混合批处理

新增 `enable_mixed_batching=True`（默认开启）：每轮先为已有 Decode 请求分配
1 token，再用剩余 token budget 执行 Prefill chunk。混合批次只做一次完整模型
forward；未完成的 chunk 不采样，最后一个 chunk 采样首个输出 token。

```python
llm = LLM("/YOUR/MODEL/PATH", enable_mixed_batching=True,
          max_num_batched_tokens=512, max_num_seqs=32)
outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=256))
```

设置 `enable_mixed_batching=False` 可使用 Prefill 优先、分阶段的对比模式。
`generate()` 返回格式不变；`step()` 的第二个返回值改为 `StepStats`，分别记录
Prefill、Decode、实际输出 token 和 `sampled_tokens`。可开启
`collect_request_metrics=True` 获取请求级入队及逐 token CPU 可用时间戳。

详见 [实现、验证和可复现评测说明](docs/MIXED_BATCHING.md)。
本次本地验证为 **60 项 CPU 测试通过，2 个 GPU 测试模块跳过**；GPU 数值验证、
CUDA Graph 实机验证、TP=2 和性能 A/B 尚未执行。下方原有性能表是项目历史数据，
不代表此次混合批处理改动的收益。

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
