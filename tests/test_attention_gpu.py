"""Actual paged attention versus an absolute-position fp32 reference (no weights)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flash_attn")
pytest.importorskip("triton")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
    pytest.skip("FlashAttention-2 requires a compatible CUDA GPU", allow_module_level=True)

from nanovllm.engine.scheduler_output import ScheduledRequest, SchedulerOutput, prepare_batch_inputs
from nanovllm.layers.attention import Attention, validate_attention_backend
from nanovllm.utils.context import set_context, reset_context


def absolute_attention(q, k, v, start):
    k = k.float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.float().repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    score = torch.einsum("qhd,khd->hqk", q.float(), k) * q.shape[-1] ** -0.5
    allowed = torch.arange(k.shape[0], device=q.device)[None, :] <= (
        start + torch.arange(q.shape[0], device=q.device))[:, None]
    score.masked_fill_(~allowed, float('-inf'))
    return torch.einsum("hqk,khd->qhd", score.softmax(-1), v)


@pytest.mark.parametrize('lengths,starts,decode', [
    ([17, 255, 257], [0, 0, 0], False),
    ([256, 512, 513], [255, 511, 512], True),
    ([257, 513, 511], [256, 512, 1], False),
    ([513, 257, 512], [511, 255, 256], False),
])
@torch.inference_mode()
def test_paged_gqa(lengths, starts, decode):
    validate_attention_backend()
    torch.manual_seed(123)
    device = 'cuda'
    heads, kv_heads, dim, block_size = 4, 2, 64, 256
    num_blocks = sum((n + 255) // 256 for n in lengths)
    pages = torch.randperm(num_blocks).tolist()
    # Future slots have deliberately unrelated data: valid KV lengths must bound reads.
    kc = torch.randn(num_blocks, block_size, kv_heads, dim, device=device, dtype=torch.float16)
    vc = torch.randn_like(kc)
    requests, qs, ks, vs, expected = [], [], [], [], []
    for i, (end, start) in enumerate(zip(lengths, starts)):
        table, pages = pages[:(end + 255) // 256], pages[(end + 255) // 256:]
        k = torch.randn(end, kv_heads, dim, device=device, dtype=torch.float16)
        v = torch.randn_like(k)
        q = torch.randn(end - start, heads, dim, device=device, dtype=torch.float16)
        for p in range(start):
            kc[table[p // 256], p % 256] = k[p]
            vc[table[p // 256], p % 256] = v[p]
        requests.append(ScheduledRequest(i, start, end, end, (1,) * (end - start),
                                         tuple(table), 1.0, decode))
        qs.append(q)
        ks.append(k[start:])
        vs.append(v[start:])
        expected.append(absolute_attention(q, k, v, start))
    plan = SchedulerOutput(tuple(requests))
    data = prepare_batch_inputs(plan, block_size)
    tensor = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    set_context(is_pure_decode=decode, cu_seqlens_q=tensor(data.query_start_loc),
                cu_seqlens_k=tensor(data.cu_seqlens_k), max_seqlen_q=max(e-s for e,s in zip(lengths, starts)),
                max_seqlen_k=max(lengths), slot_mapping=tensor(data.slot_mapping),
                context_lens=tensor(lengths), block_tables=tensor(data.block_tables))
    attn = Attention(heads, dim, dim ** -0.5, kv_heads)
    attn.k_cache, attn.v_cache = kc, vc
    try:
        actual = attn(torch.cat(qs), torch.cat(ks), torch.cat(vs)).reshape(-1, heads, dim)
        torch.testing.assert_close(actual.float(), torch.cat(expected), atol=3e-3, rtol=3e-3)
        # Verify the real Triton store writes exactly the scheduled slots.
        slots = torch.tensor(data.slot_mapping, device=device)
        torch.testing.assert_close(kc.flatten(0, 1)[slots], torch.cat(ks), atol=0, rtol=0)
    finally:
        reset_context()


@torch.inference_mode()
def test_cache_free_warmup():
    q = torch.randn(17, 4, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(17, 2, 64, device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    cu = torch.tensor([0, 17], device='cuda', dtype=torch.int32)
    set_context(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=17, max_seqlen_k=17)
    try:
        actual = Attention(4, 64, 64 ** -0.5, 2)(q, k, v)
        torch.testing.assert_close(actual.float(), absolute_attention(q, k, v, 0), atol=3e-3, rtol=3e-3)
    finally:
        reset_context()
