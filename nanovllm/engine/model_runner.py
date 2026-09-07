import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler_output import SchedulerOutput, ScheduledRequest, prepare_batch_inputs, graph_batch_sizes
from nanovllm.layers.attention import validate_attention_backend
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        validate_attention_backend()
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n + 4 > self.shm.size:
            raise ValueError(f"TP batch payload ({n} bytes) exceeds shared memory capacity")
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        budget = min(self.config.max_num_batched_tokens,
                     self.config.max_num_seqs * self.config.max_model_len)
        count = min(budget, self.config.max_num_seqs)
        lengths = [budget // count + (i < budget % count) for i in range(count)]
        # Exercise the full token budget AND the maximum number of logits rows.
        seqs = [Sequence([0] * length) for length in lengths]
        self.run(SchedulerOutput(tuple(ScheduledRequest.from_sequence(s, len(s), False)
                                       for s in seqs)))
        # Also exercise long varlen queries; their attention workspace can differ.
        long_lengths = []
        remaining = budget
        while remaining:
            length = min(remaining, self.config.max_model_len)
            long_lengths.append(length)
            remaining -= length
        if long_lengths != lengths:
            seqs = [Sequence([0] * length) for length in long_lengths]
            self.run(SchedulerOutput(tuple(ScheduledRequest.from_sequence(s, len(s), False)
                                           for s in seqs)))
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        dtype_bytes = torch.empty((), dtype=hf_config.dtype, device="cpu").element_size()
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * dtype_bytes
        reserve = 0 if self.enforce_eager else config.cuda_graph_memory_reserve
        available = int(total * config.gpu_memory_utilization - used - peak + current - reserve) // block_bytes
        # Agree before raising capacity errors, so peers do not enter a collective
        # that the failing rank will never reach.
        capacity = torch.tensor(available, dtype=torch.int64, device="cuda")
        dist.all_reduce(capacity, op=dist.ReduceOp.MIN)
        available = int(capacity.item())
        if available <= 0:
            raise ValueError("No memory remains for KV cache after activation/graph reservation")
        if config.num_kvcache_blocks == -1:
            config.num_kvcache_blocks = available
        elif config.num_kvcache_blocks > available:
            raise ValueError(f"Requested KV pool exceeds estimated capacity ({available} blocks)")
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_batch(self, output: SchedulerOutput):
        batch = prepare_batch_inputs(output, self.block_size)

        def tensor(data, dtype=torch.int32):
            return torch.tensor(data, dtype=dtype, device="cpu", pin_memory=True).cuda(non_blocking=True)

        set_context(
            is_pure_decode=output.is_pure_decode,
            cu_seqlens_q=tensor(batch.query_start_loc),
            cu_seqlens_k=tensor(batch.cu_seqlens_k),
            max_seqlen_q=max(r.num_scheduled_tokens for r in output.requests),
            max_seqlen_k=max(batch.context_lens),
            slot_mapping=tensor(batch.slot_mapping),
            context_lens=tensor(batch.context_lens),
            block_tables=tensor(batch.block_tables) if batch.block_tables is not None else None,
            logits_indices=tensor(batch.logits_indices, torch.int64),
        )
        return (tensor(batch.input_ids, torch.int64), tensor(batch.positions, torch.int64),
                tensor(batch.temperatures, torch.float32) if self.rank == 0 else None)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        context = get_context()
        bs = input_ids.size(0)
        bucket = next((n for n in getattr(self, "graph_bs", ()) if n >= bs), None)
        use_graph = (context.is_pure_decode and not self.enforce_eager and bucket is not None
                     and bucket in self.graphs and context.block_tables is not None
                     and context.block_tables.size(1) <= self.graph_vars["block_tables"].size(1))
        if not use_graph:
            return self.model(input_ids, positions)
        graph_vars = self.graph_vars
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"].zero_()
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        self.graphs[bucket].replay()
        return graph_vars["outputs"][:bs]

    @torch.inference_mode()
    def run(self, output: SchedulerOutput) -> dict[int, int] | None:
        try:
            input_ids, positions, temperatures = self.prepare_batch(output)
            hidden_states = self.run_model(input_ids, positions)
            # Every rank takes this branch together, including no-sample chunks.
            if not output.sample_seq_ids:
                if self.rank == 0:
                    torch.cuda.synchronize()
                return {} if self.rank == 0 else None
            logits = self.model.compute_logits(hidden_states)
            if self.rank == 0:
                token_ids = self.sampler(logits, temperatures).tolist()
                return dict(zip(output.sample_seq_ids, token_ids))
            return None
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, self.config.max_num_batched_tokens, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = graph_batch_sizes(max_bs)
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(is_pure_decode=True, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
