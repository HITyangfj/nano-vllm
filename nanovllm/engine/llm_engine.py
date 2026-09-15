# 【阅读定位】引擎总入口，负责把各组件串起来。
# 外部调用 generate()，或者自行循环 add_request()/step()。
# 每个 step 的主线：Scheduler 选请求 -> ModelRunner 计算/采样 -> Scheduler 提交进度。
# Sequence 保存请求，BlockManager 管缓存页，Tokenizer 负责文本与 token ID 的转换。
# 当前版本支持 Prefill/Decode 混合执行，因此“一轮”不再只有一个全局阶段。
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.metrics import RequestMetrics, StepStats


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 延迟导入模型/Tokenizer 依赖，使仅导入引擎模块的 CPU 测试不必加载 GPU 后端。
        from transformers import AutoTokenizer
        from nanovllm.engine.model_runner import ModelRunner

        # 只把 Config 中声明过的键传给配置对象；当前实现会忽略其他 kwargs。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.request_metrics: dict[int, RequestMetrics] = {}
        Sequence.block_size = config.kvcache_block_size
        # TP（Tensor Parallel，张量并行）：多个进程/显卡共同执行同一模型的一次前向。
        # 主进程承担 rank 0，其余 rank 各启动一个进程；这不是把不同请求分给不同卡。
        self.ps = []
        self.events = []
        # spawn 启动新的 Python 进程，避免 fork 继承已经初始化的 CUDA 运行时状态。
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 每个 worker 的 ModelRunner 初始化后会进入命令循环；rank 0 留在本进程。
        # ModelRunner 先测量显存并分配真实 KV 池，之后 Scheduler 才能拿到最终块数。
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 进程正常退出时兜底清理共享内存、子进程和分布式通信资源。
        atexit.register(self.exit)

    def exit(self):
        # 允许手动调用 exit() 后，atexit 再次调用而不重复释放。
        if not hasattr(self, "model_runner"):
            return
        # call() 会把退出命令同步发送给 TP worker，再在 rank 0 本地执行。
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 支持字符串 Prompt 和预先分词的 token ID 列表；这里只入队，不执行模型。
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        # 观测开关关闭时不保留逐 token 记录，避免长时间运行累计大量统计数据。
        if self.config.collect_request_metrics:
            self.request_metrics[seq.seq_id] = RequestMetrics(seq.seq_id, perf_counter())
        return seq.seq_id

    def step(self):
        # 1. 生成本轮不可变计划：每个请求的计算区间、token、块表和采样需求。
        # 调用者应在 is_finished()==False 时调用；空队列不会执行空 batch forward。
        started_at = perf_counter()
        plan = self.scheduler.schedule()
        # 2. 同一轮的所有请求合并为一次模型前向；返回 {seq_id: 新采样的 token_id}。
        # 部分 Prefill 请求没有采样结果，所以不能把这个结果直接与所有请求 zip。
        token_ids = self.model_runner.call("run", plan)
        # run() 已把 rank 0 的采样结果取回 CPU，这个时间不是 CUDA 异步发射时间。
        available_at = perf_counter()
        # 在 postprocess 释放完成请求之前记录占用，更接近本轮执行时的 KV 占用。
        used_blocks = len(self.scheduler.block_manager.used_block_ids)
        # 3. 提交 KV 进度、追加有效输出，判断结束条件并释放引用。
        finished = self.scheduler.postprocess(plan, token_ids)
        if self.config.collect_request_metrics:
            for seq_id, token_id in token_ids.items():
                metric = self.request_metrics[seq_id]
                metric.token_timestamps.append(available_at)
                metric.token_ids.append(token_id)
            for seq in finished:
                self.request_metrics[seq.seq_id].finished_at = available_at
        # “计算 token”与“输出 token”不同：算 510 个 Prompt token，可能一个都没输出。
        # StepStats 也携带本轮 sampled_tokens，让未完成请求的新增 token 可被调用者取得。
        stats = StepStats(plan.num_prefill_tokens, plan.num_decode_tokens, len(token_ids),
                          sum(r.num_recomputed_tokens for r in plan.requests), plan.num_preemptions,
                          used_blocks, started_at, available_at, token_ids)
        # outputs 只包含本轮刚完成请求的完整 completion；不含 Prompt token。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished]
        return outputs, stats

    def is_finished(self):
        # waiting 和 running 都为空，才说明整个引擎没有剩余请求。
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 离线批量接口：先把全部 Prompt 入队，再循环 step() 直到所有请求完成。
        # 学习时注意：尽管这里保留了旧的 list[str] 类型标注，实际返回值是
        # [{'text': 解码文本, 'token_ids': completion列表}, ...]，以函数末尾为准。
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 单个 SamplingParams 复用于所有请求；也可以为每个请求分别提供参数。
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, stats = self.step()
            elapsed = perf_counter() - t
            # 这里展示本轮的计算速率；混合 batch 的 Prefill/Decode 两项可同时非零。
            # 它不是只用 completion 数计算的端到端输出吞吐。
            prefill_throughput = stats.num_prefill_tokens / elapsed
            decode_throughput = stats.num_decode_tokens / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 请求完成顺序可能不同。按自增 seq_id 排序，恢复本次提交 Prompt 的先后顺序。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 到此才统一把完整 completion ID 列表转为文本；GPU 模型本身不处理字符串。
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
