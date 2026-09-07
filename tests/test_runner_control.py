"""Run real runner control flow on CPU, replacing only unavailable GPU imports."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nanovllm.engine.scheduler_output import ScheduledRequest, SchedulerOutput, graph_batch_sizes
from nanovllm.utils.context import get_context, set_context


@pytest.fixture
def runner_type(monkeypatch):
    # Load an isolated copy, never replace the production module used by GPU tests.
    for name, members in {
        'nanovllm.models.qwen3': {'Qwen3ForCausalLM': object},
        'nanovllm.layers.attention': {'validate_attention_backend': lambda: None},
        'nanovllm.layers.sampler': {'Sampler': object},
        'nanovllm.utils.loader': {'load_model': lambda *a: None},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[1] / 'nanovllm/engine/model_runner.py'
    spec = importlib.util.spec_from_file_location('_runner_control_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ModelRunner


@pytest.mark.parametrize('sample', [False, True])
def test_run_one_forward_and_optional_head_and_sampler(runner_type, monkeypatch, sample):
    runner = runner_type.__new__(runner_type)
    runner.rank = 0
    calls = []
    def prepare(plan):
        set_context(logits_indices=torch.tensor([1] if sample else [], dtype=torch.long))
        return torch.tensor([1, 2]), torch.tensor([0, 1]), torch.tensor([0.6])
    runner.prepare_batch = prepare
    runner.run_model = lambda *a: calls.append('forward') or torch.ones(2, 4)
    runner.model = SimpleNamespace(compute_logits=lambda h: calls.append('head') or h[[1]])
    runner.sampler = lambda *a: calls.append('sample') or torch.tensor([123])
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: calls.append('synchronize'))
    plan = SchedulerOutput((ScheduledRequest(37, 0, 2, 2 if sample else 3, (1, 2), (0,), 0.6, False),))
    assert runner.run(plan) == ({37: 123} if sample else {})
    assert calls == (['forward', 'head', 'sample'] if sample else ['forward', 'synchronize'])
    assert get_context().logits_indices is None


@pytest.mark.parametrize('actual,pure_decode', [(1, True), (3, True), (9, True), (17, True), (18, True), (3, False)])
def test_graph_padding_and_eager_fallback(runner_type, actual, pure_decode):
    runner = runner_type.__new__(runner_type)
    runner.enforce_eager = False
    runner.graph_bs = graph_batch_sizes(17)
    called = []
    runner.model = lambda *args: called.append('eager') or torch.zeros(actual, 4)
    runner.graph_vars = {
        'input_ids': torch.full((17,), 99), 'positions': torch.full((17,), 99),
        'slot_mapping': torch.full((17,), 99), 'context_lens': torch.full((17,), 99),
        'block_tables': torch.full((17, 3), 99), 'outputs': torch.ones(17, 4),
    }
    def replay():
        called.append('graph')
        v = runner.graph_vars
        assert torch.all(v['slot_mapping'][actual:] == -1)
        assert torch.all(v['context_lens'][actual:] == 0)
        assert torch.all(v['block_tables'][actual:] == 0)
        assert torch.all(v['block_tables'][:, 1:] == 0)
        assert torch.all(v['slot_mapping'][:actual] == 7)
    runner.graphs = {n: SimpleNamespace(replay=replay) for n in runner.graph_bs}
    set_context(is_pure_decode=pure_decode, slot_mapping=torch.full((actual,), 7),
                context_lens=torch.full((actual,), 8), block_tables=torch.zeros(actual, 1))
    out = runner.run_model(torch.ones(actual, dtype=torch.long), torch.ones(actual, dtype=torch.long))
    assert out.shape == (actual, 4)
    assert called == (['graph'] if pure_decode and actual <= 17 else ['eager'])


@pytest.mark.parametrize('requested,peer_capacity', [(-1, 2), (3, 2), (-1, -1)])
def test_tp_capacity_is_agreed_before_errors(runner_type, monkeypatch, requested, peer_capacity):
    runner = runner_type.__new__(runner_type)
    runner.enforce_eager, runner.world_size, runner.block_size = True, 1, 256
    hf = SimpleNamespace(num_key_value_heads=1, hidden_size=2, num_attention_heads=1,
                         num_hidden_layers=1, dtype=torch.float16)
    runner.config = SimpleNamespace(hf_config=hf, gpu_memory_utilization=1,
                                   num_kvcache_blocks=requested)
    layer = SimpleNamespace(k_cache=None, v_cache=None)
    runner.model = SimpleNamespace(modules=lambda: [layer])
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (8192, 8192))
    monkeypatch.setattr(torch.cuda, 'memory_stats', lambda: {
        'allocated_bytes.all.peak': 0, 'allocated_bytes.all.current': 0})
    tensor = torch.tensor
    monkeypatch.setattr(torch, 'tensor', lambda data, **kwargs: tensor(data, dtype=kwargs.get('dtype')))
    calls = []
    def agree(capacity, op):
        calls.append('all_reduce')
        capacity.fill_(peer_capacity)
    monkeypatch.setattr(torch.distributed, 'all_reduce', agree)
    if peer_capacity <= 0 or requested > peer_capacity:
        with pytest.raises(ValueError):
            runner.allocate_kv_cache()
    else:
        runner.allocate_kv_cache()
        assert runner.config.num_kvcache_blocks == 2
        assert layer.k_cache.shape == (2, 256, 1, 2)
    assert calls == ['all_reduce']
