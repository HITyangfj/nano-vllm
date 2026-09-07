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
        from transformers import AutoTokenizer
        from nanovllm.engine.model_runner import ModelRunner

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.request_metrics: dict[int, RequestMetrics] = {}
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        if self.config.collect_request_metrics:
            self.request_metrics[seq.seq_id] = RequestMetrics(seq.seq_id, perf_counter())
        return seq.seq_id

    def step(self):
        started_at = perf_counter()
        plan = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", plan)
        # run() materializes rank-0 samples on CPU, so this is not launch time.
        available_at = perf_counter()
        used_blocks = len(self.scheduler.block_manager.used_block_ids)
        finished = self.scheduler.postprocess(plan, token_ids)
        if self.config.collect_request_metrics:
            for seq_id, token_id in token_ids.items():
                metric = self.request_metrics[seq_id]
                metric.token_timestamps.append(available_at)
                metric.token_ids.append(token_id)
            for seq in finished:
                self.request_metrics[seq.seq_id].finished_at = available_at
        stats = StepStats(plan.num_prefill_tokens, plan.num_decode_tokens, len(token_ids),
                          sum(r.num_recomputed_tokens for r in plan.requests), plan.num_preemptions,
                          used_blocks, started_at, available_at, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished]
        return outputs, stats

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
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
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
