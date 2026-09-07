import pytest

torch = pytest.importorskip('torch')
from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.utils.context import set_context, reset_context


def test_lm_head_selects_only_explicit_sample_rows(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 1)
    head = ParallelLMHead(4, 4)
    with torch.no_grad():
        head.weight.copy_(torch.eye(4))
    states = torch.arange(512*4, dtype=torch.float32).reshape(512, 4)
    # A/B decode and a partial C: C's final hidden row must never reach sampling.
    set_context(logits_indices=torch.tensor([0, 1]))
    try:
        assert torch.equal(head(states), states[:2])
        set_context(logits_indices=torch.tensor([0, 1, 511]))
        assert torch.equal(head(states), states[[0, 1, 511]])
    finally:
        reset_context()
