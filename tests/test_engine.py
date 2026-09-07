from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams


class DeterministicRunner:
    """Models CPU-ready results only; the real scheduler and engine do all commits."""
    def call(self, method, plan):
        assert method == 'run'
        return {sid: 42 for sid in reversed(plan.sample_seq_ids)}


def engine():
    e = LLMEngine.__new__(LLMEngine)
    e.config = SimpleNamespace(max_num_seqs=3, max_num_batched_tokens=4,
        max_model_len=4096, enable_mixed_batching=True, eos=999,
        kvcache_block_size=256, num_kvcache_blocks=32, collect_request_metrics=True)
    e.scheduler = Scheduler(e.config)
    e.model_runner = DeterministicRunner()
    e.request_metrics = {}
    return e


def test_step_progress_and_cpu_ready_metrics_use_ids_not_zip_order():
    e = engine()
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    a = e.add_request([1], params)
    b = e.add_request([2], params)
    e.step()
    c = e.add_request([3]*7, params)
    finished, stats = e.step()
    assert stats.num_decode_tokens == 2 and stats.num_prefill_tokens == 2
    assert stats.num_output_tokens == 2 and set(stats.sampled_tokens) == {a, b}
    assert not e.request_metrics[c].token_timestamps
    while not e.is_finished():
        e.step()
    for metric in e.request_metrics.values():
        assert metric.token_ids == [42]*3
        assert len(metric.token_timestamps) == 3
        assert metric.enqueued_at <= metric.token_timestamps[0] <= metric.finished_at


def test_generate_retains_public_output_format():
    e = engine()
    e.tokenizer = SimpleNamespace(decode=lambda ids: ','.join(map(str, ids)))
    outputs = e.generate([[1]*7, [2]*2], SamplingParams(max_tokens=2), use_tqdm=False)
    assert outputs == [{'text': '42,42', 'token_ids': [42, 42]}]*2
