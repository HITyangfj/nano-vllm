import pickle
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler_output import prepare_batch_inputs, graph_batch_sizes
from nanovllm.sampling_params import SamplingParams


def scheduler(budget=512, slots=16, blocks=64, mixed=True):
    Sequence.block_size = 256
    return Scheduler(SimpleNamespace(max_num_seqs=slots, max_num_batched_tokens=budget,
        max_model_len=8192, enable_mixed_batching=mixed, eos=999,
        kvcache_block_size=256, num_kvcache_blocks=blocks))


def request(n, output=4, value=1, ignore_eos=True):
    return Sequence([value] * n, SamplingParams(max_tokens=output, ignore_eos=ignore_eos))


def commit(s, plan=None, token=7):
    plan = plan or s.schedule()
    s.postprocess(plan, {sid: token for sid in plan.sample_seq_ids})
    invariants(s)
    return plan


def invariants(s):
    seqs = [*s.running, *s.waiting]
    assert len({x.seq_id for x in seqs}) == len(seqs)
    refs = Counter(b for x in seqs for b in x.block_table)
    bm = s.block_manager
    assert len(set(bm.free_block_ids)) == len(bm.free_block_ids)
    assert set(bm.free_block_ids).isdisjoint(bm.used_block_ids)
    assert len(bm.free_block_ids) + len(bm.used_block_ids) == len(bm.blocks)
    for b in bm.blocks:
        assert b.ref_count == refs[b.block_id]
        assert (b.block_id in bm.used_block_ids) == (b.ref_count > 0)
    for x in s.running:
        assert x.status == SequenceStatus.RUNNING and not x.is_prefill
        assert x.num_cached_tokens == len(x) - 1
    for x in s.waiting:
        assert x.status == SequenceStatus.WAITING and x.is_prefill
        assert x.num_cached_tokens < len(x)


def test_acceptance_and_mapping():
    s = scheduler()
    a, b, c = request(10), request(20), request(1200)
    s.add(a)
    s.add(b)
    commit(s)
    s.add(c)
    plan = s.schedule()
    assert [r.num_scheduled_tokens for r in plan.requests] == [1, 1, 510]
    assert plan.num_decode_tokens == 2 and plan.num_prefill_tokens == 510
    assert plan.sample_seq_ids == (a.seq_id, b.seq_id)
    data = prepare_batch_inputs(plan, 256)
    assert data.query_start_loc == [0, 1, 2, 512]
    assert data.positions == [10, 20, *range(510)]
    assert data.logits_indices == [0, 1]
    assert len(data.slot_mapping) == 512
    commit(s, plan)
    assert c.num_cached_tokens == 510 and c.num_completion_tokens == 0
    commit(s)
    assert c.num_cached_tokens == 1020 and c.num_completion_tokens == 0
    plan = s.schedule()
    assert plan.requests[-1].start_pos == 1020
    assert plan.sample_seq_ids[-1] == c.seq_id
    commit(s, plan)
    assert c.num_completion_tokens == 1


@pytest.mark.parametrize('budget,slots', [(1, 5), (2, 1), (3, 2), (512, 3)])
def test_budget_limits_round_robin_and_completion(budget, slots):
    s = scheduler(budget, slots)
    seqs = [request(1, output=5, value=i) for i in range(8)]
    for seq in seqs:
        s.add(seq)
    for _ in range(100):
        if s.is_finished():
            break
        plan = s.schedule()
        assert 0 < plan.total_num_tokens <= budget
        assert len(plan.requests) <= slots
        assert len({r.seq_id for r in plan.requests}) == len(plan.requests)
        commit(s, plan)
    assert s.is_finished()
    assert all(seq.num_completion_tokens == 5 for seq in seqs)


def test_decode_rotation():
    s = scheduler()
    seqs = [request(1, output=10, value=i) for i in range(5)]
    for seq in seqs:
        s.add(seq)
    commit(s)
    s.max_num_batched_tokens = 2
    ids = [r.seq_id for _ in range(3) for r in commit(s).requests]
    assert ids[:5] == [x.seq_id for x in seqs]


@pytest.mark.parametrize('length', [1, 255, 256, 257, 511, 512, 513])
def test_boundaries_prefix_publication_and_shared_refs(length):
    s = scheduler(budget=255)
    a = request(length, output=3)
    s.add(a)
    while not a.num_completion_tokens:
        plan = s.schedule()
        data = prepare_batch_inputs(plan, 256)
        r = plan.requests[0]
        for pos, slot in zip(data.positions, data.slot_mapping):
            assert slot == a.block_table[pos // 256] * 256 + pos % 256
        # Uncomputed full blocks must not be published yet.
        for i in range(a.num_cached_tokens // 256, len(a.block_table)):
            assert s.block_manager.blocks[a.block_table[i]].hash == -1
        commit(s, plan)
    b = request(length, output=2)
    s.add(b)
    plan = s.schedule()
    rb = next(r for r in plan.requests if r.seq_id == b.seq_id)
    assert rb.start_pos == ((length - 1) // 256) * 256
    for block in b.block_table[:rb.start_pos // 256]:
        assert block in a.block_table
        assert s.block_manager.blocks[block].ref_count == 2
    commit(s, plan)
    while not s.is_finished():
        commit(s)
    assert len(s.block_manager.free_block_ids) == 64


def test_selected_cache_is_protected_and_recompute_does_not_duplicate_output():
    s = scheduler(budget=512, blocks=3)
    a, b, c = (request(256, output=4, value=i) for i in range(3))
    s.add(a)
    s.add(b)
    commit(s)
    s.add(c)
    old_a_blocks = tuple(a.block_table)
    plan = s.schedule()
    assert plan.requests[0].seq_id == a.seq_id
    assert tuple(a.block_table[:1]) == old_a_blocks
    assert b in s.waiting and b.num_completion_tokens == 1
    assert plan.num_preemptions == 1
    # The last free block belongs to selected A; B cannot reclaim it.
    assert all(s.block_manager.blocks[x].ref_count > 0 for x in a.block_table)
    commit(s, plan)
    recomputed = 0
    for _ in range(100):
        if s.is_finished():
            break
        plan = s.schedule()
        recomputed += sum(r.num_recomputed_tokens for r in plan.requests)
        commit(s, plan)
    assert s.is_finished() and recomputed > 0
    assert [x.num_completion_tokens for x in (a, b, c)] == [4, 4, 4]


def test_waiting_partial_prefill_can_be_evicted_for_decode():
    s = scheduler(budget=256, blocks=3)
    a = request(256)
    s.add(a)
    commit(s)
    c = request(257, value=2)
    s.add(c)
    # Reserve C's prompt, modeling a resident incomplete chunk before A needs a slot.
    s.block_manager.allocate(c, 0)
    plan = s.schedule()
    assert plan.requests[0].seq_id == a.seq_id
    assert c.block_table == [] and plan.num_preemptions == 1
    commit(s, plan)


def test_pickle_preserves_worker_inputs_and_sequence_state():
    s = scheduler(budget=2)
    a = request(1)
    s.add(a)
    commit(s)
    c = request(10)
    s.add(c)
    plan = s.schedule()
    restored = pickle.loads(pickle.dumps(plan))
    assert restored == plan
    assert prepare_batch_inputs(restored, 256) == prepare_batch_inputs(plan, 256)
    for seq in (a, c):
        assert pickle.loads(pickle.dumps(seq)).__dict__ == seq.__dict__
    a.append_token(42)  # Plan is an immutable snapshot, not a live Sequence view.
    assert restored.requests[0].input_ids == (7,)


def test_eos_and_result_validation():
    s = scheduler()
    a = request(1, ignore_eos=False)
    s.add(a)
    plan = s.schedule()
    with pytest.raises(ValueError, match='IDs'):
        s.postprocess(plan, {})
    with pytest.raises(RuntimeError, match='previous batch'):
        s.schedule()
    commit(s, plan, token=999)
    assert s.is_finished() and a.is_finished
    with pytest.raises(ValueError, match='pool'):
        scheduler(blocks=1).add(request(257))
    with pytest.raises(ValueError, match='max_model_len'):
        s.add(request(8190))
    with pytest.raises(ValueError):
        Sequence([])


def test_legacy_mode_is_separate_prefill_first():
    s = scheduler(mixed=False)
    a = request(1)
    s.add(a)
    commit(s)
    c = request(700)
    s.add(c)
    first = commit(s)
    assert first.num_prefill_tokens == 512 and first.num_decode_tokens == 0
    assert a.num_completion_tokens == 1
    commit(s)
    assert commit(s).is_pure_decode


@pytest.mark.parametrize('seed', range(10))
def test_finite_random_arrivals_under_pressure(seed):
    rng = random.Random(seed)
    s = scheduler(budget=rng.choice([1, 17, 255, 512]), slots=3, blocks=5)
    seqs = [request(rng.choice([1, 255, 256, 257, 513]), output=3, value=i) for i in range(9)]
    for step in range(12000):
        if step < len(seqs):
            s.add(seqs[step])
        if s.is_finished() and step >= len(seqs):
            break
        plan = s.schedule()
        assert 0 < plan.total_num_tokens <= s.max_num_batched_tokens
        commit(s, plan)
    assert s.is_finished()
    assert all(x.num_completion_tokens == 3 for x in seqs)


@pytest.mark.parametrize('size', [1, 3, 9, 17, 31, 512])
def test_graph_buckets_fit_capacity(size):
    buckets = graph_batch_sizes(size)
    assert buckets[-1] == size and all(x <= size for x in buckets)
    for actual in range(1, size + 1):
        assert any(actual <= x <= size for x in buckets)
