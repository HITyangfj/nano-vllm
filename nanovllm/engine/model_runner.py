# 【阅读定位】把 SchedulerOutput 转成 GPU 工作，并返回采样 token 的执行器。
# 初始化：建 TP 通信组 -> 建模型/加载权重 -> warmup -> 建 KV 池 -> 捕获 Decode Graph。
# 每轮：prepare_batch() 准备输入和 Context -> run_model() 前向 -> LM Head -> Sampler。
# Prefill 是一段输入的计算，Decode 是单个新增输入的计算；混合 batch 只做一次完整前向。
# 张量并行 TP 切分模型权重/计算，各 rank 处理同一批请求；最终仅 rank 0 返回采样结果。
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
        # 确认已安装的 FlashAttention 暴露所需 paged varlen 参数；不是数值正确性测试。
        validate_attention_backend()
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # NCCL 负责 GPU 间 collective（如 all_reduce/gather）；rank 是当前进程编号。
        # world_size=1 时也创建通信组，因为模型的并行层会读取 rank/world_size。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        # 临时改变默认设备/浮点类型，让模型参数和随后创建的 KV、Graph buffer 位于
        # 当前 GPU，并使用模型配置的 dtype；初始化结束后恢复默认浮点类型和 CPU 设备。
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 先 warmup 测激活峰值，再决定把多少剩余显存用于 KV；此时还不存在真实 KV 池。
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # 共享内存传“CPU 调度命令和计划”，不传模型权重或大块 GPU K/V 张量。
                # 这里使用固定名 nanovllm 和 1 MiB 容量，属于轻量的单引擎通信实现。
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # barrier 保证 rank 0 已创建共享内存，worker 才按名称连接。
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # worker 初始化不会立即返回；它在这里等待 rank 0 后续发送 run/exit。
                self.loop()

    def exit(self):
        # 所有 rank 都要参与清理/同步；只有创建共享内存的 rank 0 删除其名称。
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
        # 非 0 rank 的命令循环：反序列化参数，在本进程执行同名方法。
        # 各 rank 的前向/LM Head collective 次序必须一致，否则会互相等待。
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        # 协议：[4 字节 payload 长度，小端] + [pickle 编码的 method_name 与参数]。
        # Event 用于唤醒 worker；读取后清除信号，等待下一条命令。
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        # rank 0 完整写好消息后才 set() 各 worker 的 Event，避免读取未完成的 payload。
        # 传输的 SchedulerOutput 已携带本轮 token/位置/页表快照，无需 worker 访问
        # rank 0 的 waiting/running 队列，也不依赖 Sequence 的全量历史切片。
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
        # 统一入口：rank 0 先通知其他 rank，然后自己执行；worker 调用时不会再广播。
        # getattr 把字符串 "run"/"exit" 转为对应的绑定方法。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        # empty_cache 释放 PyTorch allocator 中可归还的空闲缓存，不会删除模型参数。
        # reset_peak_memory_stats 后的峰值，用于估算前向激活还需要多少额外显存。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        budget = min(self.config.max_num_batched_tokens,
                     self.config.max_num_seqs * self.config.max_model_len)
        count = min(budget, self.config.max_num_seqs)
        # 第一组形状把预算分摊给尽可能多的请求，覆盖大量 logits 行的内存需求。
        # 例如 budget=10、count=3，得到长度 [4,3,3]，总 token 数仍为 10。
        lengths = [budget // count + (i < budget % count) for i in range(count)]
        # 这些是虚拟 token，请求没有 block_table，Attention 使用无缓存的普通 varlen。
        seqs = [Sequence([0] * length) for length in lengths]
        self.run(SchedulerOutput(tuple(ScheduledRequest.from_sequence(s, len(s), False)
                                       for s in seqs)))
        # 第二组尽量拼成长 query；长序列 Attention 的 workspace 可能不同于短序列批次。
        # 两组只是测量不同形状的峰值，不对应真实用户请求，也不需要提交 Scheduler 状态。
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
        # 本函数真正分配 GPU K/V 存储；BlockManager 只管理这些存储页的编号和引用。
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # used 来自 CUDA 驱动，包含当前设备上的已用显存；peak/current 是 PyTorch
        # allocator 的统计。peak-current 估计需要留给运行时临时激活的额外空间。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # GQA 模型的 KV head 数可以少于 Query head 数；KV 容量计算必须使用 KV heads。
        # 某些模型显式配置 head_dim，不能总是假定 hidden_size / attention_heads。
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        dtype_bytes = torch.empty((), dtype=hf_config.dtype, device="cpu").element_size()
        # 每页在当前 rank 的字节数：K/V 两份 × 层数 × 每页 token 数 × 本 rank 的 KV
        # head 数 × 每个 head 维度 × dtype 字节数。物理页 ID 在所有层共同使用。
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * dtype_bytes
        # Graph 会占用额外显存，因此在估算 KV 容量前预留配置值；这仍是容量估计。
        reserve = 0 if self.enforce_eager else config.cuda_graph_memory_reserve
        available = int(total * config.gpu_memory_utilization - used - peak + current - reserve) // block_bytes
        # 各 rank 的可用显存可能不同，取最小容量，使同一块号在每个 rank 都合法。
        # 必须先参加 collective 再一致报容量错误，不能一张卡提前报错而其他卡卡在同步。
        capacity = torch.tensor(available, dtype=torch.int64, device="cuda")
        dist.all_reduce(capacity, op=dist.ReduceOp.MIN)
        available = int(capacity.item())
        if available <= 0:
            raise ValueError("No memory remains for KV cache after activation/graph reservation")
        # -1 表示自动估算；显式正数用于固定 KV 池大小（如公平比较调度策略）。
        if config.num_kvcache_blocks == -1:
            config.num_kvcache_blocks = available
        elif config.num_kvcache_blocks > available:
            raise ValueError(f"Requested KV pool exceeds estimated capacity ({available} blocks)")
        # 形状：[K/V, 层, 物理页, 页内 token, 本 rank KV heads, head_dim]。
        # empty() 不初始化内容；只有实际写过/复用的有效位置才能被 Attention 读取。
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 为每层 Attention 挂上大池的视图，共享底层存储，不复制整份缓存。
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_batch(self, output: SchedulerOutput):
        # prepare_batch_inputs 在 CPU 上按计划顺序拼接请求，具体映射见 scheduler_output.py。
        # A/B Decode 各 1 token、C Prefill 510 token 时，query_start_loc=[0,1,2,512]。
        # positions 保持每个请求自己的绝对位置，C 的后续 chunk 不会重新从 0 开始。
        batch = prepare_batch_inputs(output, self.block_size)

        def tensor(data, dtype=torch.int32):
            # 先建 pinned（锁页）CPU 内存，再向 GPU 异步复制；相同 CUDA stream 的
            # 后续计算会遵守执行顺序。整数索引通常 int32，token/position 使用 int64。
            return torch.tensor(data, dtype=dtype, device="cpu", pin_memory=True).cuda(non_blocking=True)

        # Context 是本进程本轮前向共享的元数据，Attention 和 LM Head 会读取它。
        # cu_seqlens_q：query 的累加边界；cu_seqlens_k：各请求有效 KV 长度的累加值。
        # context_lens[i]=end_i，包含本轮刚写入的 K/V，不是仅有历史的 start_i。
        # slot_mapping 每个输入 token 一项：物理页 ID*block_size + 页内偏移。
        # block_tables 每请求一行，描述逻辑页到物理页的映射；warmup 时可以为 None。
        # logits_indices 只指向需要采样请求的最后一个 hidden-state 行。
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
        # inference_mode 关闭反向图及相关追踪开销；这里只返回 hidden states，不采样。
        # bs 在此是实际输入 token 数；仅在纯 Decode 中，它也等于请求数。
        context = get_context()
        bs = input_ids.size(0)
        bucket = next((n for n in getattr(self, "graph_bs", ()) if n >= bs), None)
        # eager = 直接运行模型；Graph = 重放预先捕获、固定形状/地址的 GPU 计算。
        # 一个 q_len=1 的末尾 Prefill chunk 仍不是普通 Decode，所以检查显式阶段。
        # warmup 尚未捕获 Graph、batch 太大或页表太宽时，都必须正确走 eager。
        use_graph = (context.is_pure_decode and not self.enforce_eager and bucket is not None
                     and bucket in self.graphs and context.block_tables is not None
                     and context.block_tables.size(1) <= self.graph_vars["block_tables"].size(1))
        if not use_graph:
            # 混合 batch 整体做这一次 forward，投影和 MLP 共用合批输入；Attention
            # 内部选择 varlen paged 路径，不是先做一次 Prefill forward 再做 Decode。
            return self.model(input_ids, positions)
        graph_vars = self.graph_vars
        # 捕获的 Graph 引用的是固定 buffer 地址，不能仅给 Python 变量换一个新 Tensor；
        # 必须把本轮数据复制进去，再 replay。较小 batch 用 padding 填充到 bucket。
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        # slot=-1 让 KV 写入 kernel 跳过 padding 行；长度=0 防止读取虚构历史。
        # 先清理全部 buffer，避免前一轮大 batch 的遗留数据污染本轮小 batch。
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"].zero_()
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        self.graphs[bucket].replay()
        # 只返回真实请求行；padding 行不会参与后续 LM Head 和采样。
        return graph_vars["outputs"][:bs]

    @torch.inference_mode()
    def run(self, output: SchedulerOutput) -> dict[int, int] | None:
        # run() 是引擎每步调用的完整入口。返回值：rank 0 为 {seq_id: token_id}，
        # 其他 rank 为 None；rank 0 无需采样时返回 {}，仍然完成了本轮 KV 计算。
        try:
            input_ids, positions, temperatures = self.prepare_batch(output)
            hidden_states = self.run_model(input_ids, positions)
            # hidden_states 形状 [本轮输入 token 总数, hidden_size]，不是 Q 的多头形状。
            # 所有 rank 从同一个计划判断是否采样，保证同时跳过/参加 LM Head 的 gather。
            if not output.sample_seq_ids:
                if self.rank == 0:
                    # 无采样时没有 .tolist() 隐式等待，用显式同步保证 step 返回前执行完毕。
                    torch.cuda.synchronize()
                return {} if self.rank == 0 else None
            # LM Head 按 Context.logits_indices 选行，再投影到词表；TP 模式各 rank
            # 都参加词表分片的 gather，完整 logits 只汇总在 rank 0。
            logits = self.model.compute_logits(hidden_states)
            if self.rank == 0:
                # temperatures 的顺序与 sample_seq_ids 一致，不与全部请求列表强行对齐。
                # .tolist() 把 CUDA 采样结果取回 CPU，此后才能记录 CPU 可用时间戳。
                token_ids = self.sampler(logits, temperatures).tolist()
                return dict(zip(output.sample_seq_ids, token_ids))
            return None
        finally:
            # 无论正常返回还是发生异常，都不把本 batch 的元数据留给下一次前向。
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        # 只为普通 Decode 捕获 Graph；每请求恰好一个输入 token，形状主要由请求数决定。
        # Prefill/混合批次 query 长度变化大，当前版本使用 eager 路径。
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, self.config.max_num_batched_tokens, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 以下 buffer 在捕获后长期保留；当前初始化阶段默认设备仍是当前 rank 的 CUDA。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = graph_batch_sizes(max_bs)
        # bucket 例：上限 17 时为 [1,2,4,8,16,17]，实际 9 个请求可用 16 的 Graph。
        # 最后一个 bucket 不超过 buffer 容量，同时覆盖配置允许的最大 batch。
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            # 从大到小捕获，并复用 graph_pool，以便各个互斥重放的 Graph 共享内存池。
            graph = torch.cuda.CUDAGraph()
            set_context(is_pure_decode=True, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 先在捕获外执行一次当前形状，完成必要的惰性初始化。
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                # 此时记录固定地址上的模型前向；LM Head/采样留在 Graph 外处理真实行。
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存持久 Tensor 引用，run_model() 每轮向这些 buffer 拷贝数据并读取结果。
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
