from collections import deque
from typing import TYPE_CHECKING

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler_output import ScheduledRequest, SchedulerOutput

if TYPE_CHECKING:
    from nanovllm.config import Config


class Scheduler:
    """WAITING owns new/partial/recompute prefill; RUNNING owns ordinary decode.

    Scheduling reserves slots but never commits progress. Selected requests are
    protected until postprocess; max_num_seqs limits requests in this batch only.
    """

    def __init__(self, config: "Config"):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.enable_mixed_batching = config.enable_mixed_batching
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        if min(self.max_num_seqs, self.max_num_batched_tokens, config.num_kvcache_blocks) <= 0:
            raise ValueError("Batch limits and KV block count must be positive")
        self.block_manager = BlockManager(config.num_kvcache_blocks, self.block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._pending: SchedulerOutput | None = None
        self.num_preemptions = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def _check_capacity(self, seq, num_tokens=None):
        num_blocks = ((num_tokens if num_tokens is not None else len(seq)) + self.block_size - 1) // self.block_size
        if num_blocks > len(self.block_manager.blocks):
            raise ValueError(f"Request {seq.seq_id} needs {num_blocks} KV blocks, "
                             f"but the entire pool has {len(self.block_manager.blocks)}")

    def add(self, seq: Sequence):
        if len(seq) + seq.max_tokens > self.max_model_len:
            raise ValueError("Prompt length + max_tokens exceeds max_model_len")
        # The final sampled token is returned without computing its KV.
        self._check_capacity(seq, len(seq) + seq.max_tokens - 1)
        if seq.status != SequenceStatus.WAITING or seq in self.waiting or seq in self.running:
            raise ValueError("Request is already queued or finished")
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        if self._pending is not None:
            raise RuntimeError("Commit the previous batch before scheduling again")
        requests = []
        protected = set()
        budget = self.max_num_batched_tokens
        preemptions_before = self.num_preemptions

        def select(seq, count, decode):
            nonlocal budget
            seq.num_scheduled_tokens = count
            requests.append(ScheduledRequest.from_sequence(seq, count, decode))
            protected.add(seq.seq_id)
            budget -= count

        def has_room():
            return budget > 0 and len(requests) < self.max_num_seqs

        def prefill():
            for seq in list(self.waiting):
                if not has_room():
                    break
                self._check_capacity(seq)
                cached = None
                if not seq.block_table:
                    cached = self.block_manager.can_allocate(seq)
                    if cached == -1:
                        continue
                    remaining = len(seq) - cached * self.block_size
                else:
                    remaining = len(seq) - seq.num_cached_tokens
                # Comparison mode preserves the old prefill-first chunk policy.
                if not self.enable_mixed_batching and requests and remaining > budget:
                    break
                if cached is not None:
                    self.block_manager.allocate(seq, cached)
                select(seq, min(remaining, budget), False)

        def decode():
            selected = []
            while self.running and has_room():
                seq = self.running.popleft()
                self._check_capacity(seq)
                while not self.block_manager.can_append(seq):
                    # Previously selected requests are outside this queue.
                    if self.running:
                        self.preempt(self.running.pop())
                        continue
                    victim = next((s for s in reversed(self.waiting)
                                   if s.block_table and s.seq_id not in protected), None)
                    if victim is not None:
                        self.waiting.remove(victim)
                        self.preempt(victim)
                        continue
                    self.preempt(seq)
                    break
                else:
                    self.block_manager.may_append(seq)
                    select(seq, 1, True)
                    selected.append(seq)
            # Round robin when the token/request limit is smaller than running.
            self.running.extend(selected)

        if self.enable_mixed_batching:
            decode()
            prefill()
        else:
            prefill()
            if not requests:
                decode()
                if not requests:
                    prefill()
        if not requests:
            raise RuntimeError("No executable requests; KV capacity or queue state prevents progress")
        self._pending = SchedulerOutput(tuple(requests), self.num_preemptions - preemptions_before)
        return self._pending

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def postprocess(self, output: SchedulerOutput, token_ids: dict[int, int]):
        if output is not self._pending:
            raise ValueError("Result does not belong to the pending batch")
        if set(token_ids) != set(output.sample_seq_ids):
            raise ValueError("Sampled sequence IDs do not match the batch plan")
        seqs = {s.seq_id: s for s in (*self.running, *self.waiting)}
        finished = []
        for r in output.requests:
            seq = seqs[r.seq_id]
            if seq.num_cached_tokens != r.start_pos or len(seq) != r.num_tokens:
                raise RuntimeError("Sequence changed while batch was in flight")
            # hash_blocks consumes the OLD progress interval, before publishing it.
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens = r.end_pos
            seq.num_computed_tokens = max(seq.num_computed_tokens, r.end_pos)
            seq.num_scheduled_tokens = 0
            if not r.needs_sample:
                continue
            if seq.status == SequenceStatus.WAITING:
                self.waiting.remove(seq)
                self.running.append(seq)
            seq.status = SequenceStatus.RUNNING
            seq.is_prefill = False
            token_id = token_ids[r.seq_id]
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished.append(seq)
        self._pending = None
        return finished
