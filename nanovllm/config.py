from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_mixed_batching: bool = True
    collect_request_metrics: bool = False
    cuda_graph_memory_reserve: int = 256 * 1024 * 1024
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        if min(self.max_num_batched_tokens, self.max_num_seqs, self.max_model_len) <= 0:
            raise ValueError("Token budget, batch request limit and model length must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.num_kvcache_blocks != -1 and self.num_kvcache_blocks <= 0:
            raise ValueError("num_kvcache_blocks must be -1 (automatic) or positive")
        if self.cuda_graph_memory_reserve < 0:
            raise ValueError("cuda_graph_memory_reserve must be nonnegative")
        if not os.path.isdir(self.model):
            raise ValueError(f"Model directory does not exist: {self.model}")
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        from transformers import AutoConfig
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
