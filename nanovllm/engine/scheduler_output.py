"""Immutable, CPU-only execution snapshots; workers never need Sequence history."""
from dataclasses import dataclass

from nanovllm.engine.sequence import Sequence


@dataclass(frozen=True)
class ScheduledRequest:
    seq_id: int
    start_pos: int
    end_pos: int
    num_tokens: int
    input_ids: tuple[int, ...]
    block_table: tuple[int, ...]
    temperature: float
    is_decode: bool
    num_recomputed_tokens: int = 0

    @classmethod
    def from_sequence(cls, seq: Sequence, count: int, is_decode: bool):
        start = seq.num_cached_tokens
        end = start + count
        if count <= 0 or end > len(seq):
            raise ValueError("Invalid scheduled query interval")
        return cls(seq.seq_id, start, end, len(seq), tuple(seq[start:end]),
                   tuple(seq.block_table), seq.temperature, is_decode,
                   max(0, min(end, seq.num_computed_tokens) - start))

    @property
    def num_scheduled_tokens(self):
        return self.end_pos - self.start_pos

    @property
    def needs_sample(self):
        return self.end_pos == self.num_tokens


@dataclass(frozen=True)
class SchedulerOutput:
    requests: tuple[ScheduledRequest, ...]
    num_preemptions: int = 0

    @property
    def sample_seq_ids(self):
        return tuple(r.seq_id for r in self.requests if r.needs_sample)

    @property
    def num_prefill_tokens(self):
        return sum(r.num_scheduled_tokens for r in self.requests if not r.is_decode)

    @property
    def num_decode_tokens(self):
        return sum(r.num_scheduled_tokens for r in self.requests if r.is_decode)

    @property
    def total_num_tokens(self):
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_pure_decode(self):
        return bool(self.requests) and all(r.is_decode for r in self.requests)


@dataclass
class BatchInputs:
    input_ids: list[int]
    positions: list[int]
    query_start_loc: list[int]
    cu_seqlens_k: list[int]
    context_lens: list[int]
    slot_mapping: list[int]
    block_tables: list[list[int]] | None
    logits_indices: list[int]
    temperatures: list[float]


def prepare_batch_inputs(output: SchedulerOutput, block_size: int) -> BatchInputs:
    if not output.requests:
        raise ValueError("Cannot execute an empty batch")
    batch = BatchInputs([], [], [0], [0], [], [], None, [], [])
    paged = any(r.block_table for r in output.requests)
    if paged and not all(r.block_table for r in output.requests):
        raise ValueError("Cannot mix cached requests with cache-free warmup")
    if paged:
        width = max(len(r.block_table) for r in output.requests)
        batch.block_tables = [list(r.block_table) + [0] * (width - len(r.block_table))
                              for r in output.requests]
    for r in output.requests:
        if len(r.input_ids) != r.num_scheduled_tokens or r.num_scheduled_tokens <= 0:
            raise ValueError("Input tokens and scheduled interval disagree")
        if r.is_decode and (r.num_scheduled_tokens != 1 or not r.needs_sample):
            raise ValueError("Ordinary decode must compute exactly the last token")
        if not paged and r.start_pos:
            raise ValueError("Cache-free execution cannot read history")
        batch.input_ids.extend(r.input_ids)
        batch.positions.extend(range(r.start_pos, r.end_pos))
        batch.query_start_loc.append(len(batch.input_ids))
        batch.cu_seqlens_k.append(batch.cu_seqlens_k[-1] + r.end_pos)
        batch.context_lens.append(r.end_pos)
        batch.slot_mapping.extend(
            r.block_table[p // block_size] * block_size + p % block_size if paged else -1
            for p in range(r.start_pos, r.end_pos))
        if r.needs_sample:
            batch.logits_indices.append(len(batch.input_ids) - 1)
            batch.temperatures.append(r.temperature)
    return batch


def graph_batch_sizes(max_batch_size: int):
    return sorted({n for n in [1, 2, 4, 8, *range(16, max_batch_size + 1, 16), max_batch_size]
                   if 0 < n <= max_batch_size})
