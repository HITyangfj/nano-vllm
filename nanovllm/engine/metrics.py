from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    seq_id: int
    enqueued_at: float
    token_timestamps: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    finished_at: float | None = None


@dataclass(frozen=True)
class StepStats:
    num_prefill_tokens: int
    num_decode_tokens: int
    num_output_tokens: int
    num_recomputed_tokens: int
    num_preemptions: int
    num_used_kv_blocks: int
    started_at: float
    tokens_available_at: float
    # New tokens only, available to the caller even before a request finishes.
    sampled_tokens: dict[int, int]
