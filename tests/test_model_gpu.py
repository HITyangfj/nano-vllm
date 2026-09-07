"""Set NANO_VLLM_TEST_MODEL to local Qwen3 weights; fixed histories, no RNG comparison."""
import os
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('flash_attn')
pytest.importorskip('triton')
MODEL = os.environ.get('NANO_VLLM_TEST_MODEL')
if not MODEL or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
    pytest.skip('Needs NANO_VLLM_TEST_MODEL and a compatible GPU', allow_module_level=True)

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler_output import ScheduledRequest, SchedulerOutput
from nanovllm.utils.context import reset_context


@pytest.fixture(scope='module')
def engine():
    if os.environ.get('NANO_VLLM_TEST_TP') == '2':
        pytest.skip('TP=2 run only; keep NCCL groups separate')
    llm = LLM(MODEL, enforce_eager=False, max_num_seqs=17, max_num_batched_tokens=512,
              max_model_len=2048, gpu_memory_utilization=0.8)
    yield llm
    llm.exit()


@torch.inference_mode()
def logits(runner, plan, eager=True):
    previous = runner.enforce_eager
    runner.enforce_eager = eager
    try:
        ids, positions, _ = runner.prepare_batch(plan)
        hidden = runner.run_model(ids, positions)
        if not plan.sample_seq_ids:
            return None
        return runner.model.compute_logits(hidden).float().cpu()
    finally:
        runner.enforce_eager = previous
        reset_context()


def make_sequence(bm, tokens):
    seq = Sequence(tokens)
    cached = bm.can_allocate(seq)
    assert cached >= 0
    bm.allocate(seq, cached)
    return seq


def execute(runner, bm, seqs, counts, decode_flags, eager=True):
    plan = SchedulerOutput(tuple(ScheduledRequest.from_sequence(s, n, d)
                                 for s, n, d in zip(seqs, counts, decode_flags)))
    result = logits(runner, plan, eager)
    for s, r in zip(seqs, plan.requests):
        s.num_scheduled_tokens = r.num_scheduled_tokens
        bm.hash_blocks(s)
        s.num_cached_tokens = r.end_pos
        s.num_scheduled_tokens = 0
    return result


def test_fixed_history_mixed_singleton_prefix_and_recompute(engine):
    runner = engine.model_runner
    bm = BlockManager(engine.config.num_kvcache_blocks, 256)
    histories = [[(j + i * 79) % 1000 for j in range(n)] for i, n in enumerate([257, 17, 513])]
    reference = []
    for history in histories:
        seq = make_sequence(bm, history)
        reference.append(execute(runner, bm, [seq], [len(history)-seq.num_cached_tokens], [False]))
        bm.deallocate(seq)
    bm = BlockManager(engine.config.num_kvcache_blocks, 256)
    seqs = [make_sequence(bm, h) for h in histories]
    assert execute(runner, bm, seqs, [256, 16, 239], [False]*3) is None
    actual = execute(runner, bm, seqs, [1, 1, 274], [True, True, False])
    torch.testing.assert_close(actual, torch.cat(reference), atol=5e-2, rtol=1e-2)
    # Hit two complete prefix blocks; a one-token prefill must stay off decode graphs.
    shared = make_sequence(bm, histories[2])
    assert shared.num_cached_tokens == 512
    hit = execute(runner, bm, [shared], [1], [False], eager=False)
    torch.testing.assert_close(hit, reference[2], atol=5e-2, rtol=1e-2)
    # Evict the shared prefix metadata too, forcing partial recomputation.
    for seq in [*seqs, shared]:
        bm.deallocate(seq)
    bm = BlockManager(engine.config.num_kvcache_blocks, 256)
    recompute = make_sequence(bm, histories[2])
    assert execute(runner, bm, [recompute], [511], [False]) is None
    result = execute(runner, bm, [recompute], [2], [False])
    torch.testing.assert_close(result, reference[2], atol=5e-2, rtol=1e-2)


@pytest.mark.parametrize('size', [1, 3, 9, 16, 17, 18])
def test_decode_graph_eager_and_uncovered_bucket(engine, size):
    runner = engine.model_runner
    bm = BlockManager(engine.config.num_kvcache_blocks, 256)
    seqs = [make_sequence(bm, [i+1, 2, 3]) for i in range(size)]
    execute(runner, bm, seqs, [2]*size, [False]*size)
    plan = SchedulerOutput(tuple(ScheduledRequest.from_sequence(s, 1, True) for s in seqs))
    eager = logits(runner, plan, eager=True)
    graph = logits(runner, plan, eager=False)
    torch.testing.assert_close(graph, eager, atol=5e-2, rtol=1e-2)


def test_no_sample_chunk_skips_head_and_sampler(engine):
    runner = engine.model_runner
    bm = BlockManager(engine.config.num_kvcache_blocks, 256)
    seq = make_sequence(bm, [1]*513)
    plan = SchedulerOutput((ScheduledRequest.from_sequence(seq, 255, False),))
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError('No-sample chunks must skip this operation')
        patch.setattr(runner.model, 'compute_logits', forbidden)
        patch.setattr(runner.sampler, 'forward', forbidden)
        assert runner.run(plan) == {}


def test_tp_two_smoke():
    # Run this test separately: its NCCL group must not overlap the module fixture.
    if os.environ.get('NANO_VLLM_TEST_TP') != '2' or torch.cuda.device_count() < 2:
        pytest.skip('Opt in to a separate two-GPU run with NANO_VLLM_TEST_TP=2')
    llm = LLM(MODEL, tensor_parallel_size=2, enforce_eager=True,
              max_num_batched_tokens=16, max_num_seqs=3, max_model_len=2048)
    try:
        llm.add_request([1]*2, SamplingParams(max_tokens=8, ignore_eos=True))
        llm.add_request([2]*3, SamplingParams(max_tokens=8, ignore_eos=True))
        llm.step()
        llm.add_request([3]*513, SamplingParams(max_tokens=3, ignore_eos=True))
        mixed = empty_sample = False
        while not llm.is_finished():
            _, stats = llm.step()
            mixed |= bool(stats.num_decode_tokens and stats.num_prefill_tokens)
            empty_sample |= stats.num_output_tokens == 0
        assert mixed and empty_sample
    finally:
        llm.exit()
