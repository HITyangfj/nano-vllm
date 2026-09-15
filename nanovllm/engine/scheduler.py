# 【阅读定位】调度器回答“这一轮计算哪些请求、各计算多少输入 token”。
# 它操作 CPU 请求队列和 KV 页元数据，不执行神经网络。
# 当前混合策略：先给普通 Decode 每请求 1 token，再把剩余预算分给 Prefill chunk。
# 读法：add() 入队 -> schedule() 保留资源/生成快照 -> 模型执行 -> postprocess() 提交。
from collections import deque
from typing import TYPE_CHECKING

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler_output import ScheduledRequest, SchedulerOutput

if TYPE_CHECKING:
    # 仅供类型检查器使用；运行时导入 Scheduler 不需要导入模型配置依赖。
    from nanovllm.config import Config


class Scheduler:
    # WAITING = 新 Prefill / 未完成 chunk / 抢占后的重算；RUNNING = 普通 Decode。
    # 调度时分配槽位，但不提前宣称“已经算完”；只有 postprocess 才推进计算进度。
    """WAITING owns new/partial/recompute prefill; RUNNING owns ordinary decode.

    Scheduling reserves slots but never commits progress. Selected requests are
    protected until postprocess; max_num_seqs limits requests in this batch only.
    """

    def __init__(self, config: "Config"):
        # 两个约束分别是“本轮不同请求数”和“本轮输入 token 总数”。
        # max_num_seqs 不限制 waiting/running 的队列总长度，Decode 也要消耗 token 预算。
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
        # 当前已下发、尚未提交的 batch；约束同一调度器最多有一个在途计划。
        self._pending: SchedulerOutput | None = None
        self.num_preemptions = 0

    def is_finished(self):
        # 部分 Prefill 仍在 waiting 中，因此只检查 running 是否为空是不够的。
        return not self.waiting and not self.running

    def _check_capacity(self, seq, num_tokens=None):
        # 区分“暂时被其他请求占满”与“单个请求永远放不进整个池”。后者直接报错。
        num_blocks = ((num_tokens if num_tokens is not None else len(seq)) + self.block_size - 1) // self.block_size
        if num_blocks > len(self.block_manager.blocks):
            raise ValueError(f"Request {seq.seq_id} needs {num_blocks} KV blocks, "
                             f"but the entire pool has {len(self.block_manager.blocks)}")

    def add(self, seq: Sequence):
        # 在入队前检查最坏长度；max_tokens 是输出上限，不是 Prompt+输出的总长度。
        if len(seq) + seq.max_tokens > self.max_model_len:
            raise ValueError("Prompt length + max_tokens exceeds max_model_len")
        # 最后一个采样 token 直接返回，不会再拿去做下一轮前向，因此它不需要自己的 KV。
        self._check_capacity(seq, len(seq) + seq.max_tokens - 1)
        if seq.status != SequenceStatus.WAITING or seq in self.waiting or seq in self.running:
            raise ValueError("Request is already queued or finished")
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        # 不允许上一个计划尚未执行/提交，就为同一批请求重新调度或释放其缓存。
        if self._pending is not None:
            raise RuntimeError("Commit the previous batch before scheduling again")
        requests = []
        # protected 保存本轮已选请求的 ID，防止显存紧张时把在途请求当牺牲者。
        protected = set()
        budget = self.max_num_batched_tokens
        preemptions_before = self.num_preemptions

        def select(seq, count, decode):
            # nonlocal 修改外层本轮剩余预算；不在这里推进 num_cached_tokens。
            # 快照固定 start/end、输入 token 和块表，避免执行端从可变 Sequence 反推输入。
            nonlocal budget
            seq.num_scheduled_tokens = count
            requests.append(ScheduledRequest.from_sequence(seq, count, decode))
            protected.add(seq.seq_id)
            budget -= count

        def has_room():
            # 即便 token 预算仍有余量，达到请求数上限后也不能继续纳入新请求。
            return budget > 0 and len(requests) < self.max_num_seqs

        def prefill():
            # 列表快照固定本次遍历顺序；部分 chunk 执行后仍留在 waiting，等待下一轮。
            for seq in list(self.waiting):
                if not has_room():
                    break
                self._check_capacity(seq)
                cached = None
                if not seq.block_table:
                    # 新请求或刚被抢占的请求：先检查前缀能复用多少完整块、能否分配页。
                    cached = self.block_manager.can_allocate(seq)
                    if cached == -1:
                        # 暂时没有空间就跳过，不能让一个大请求阻塞其他已有缓存的请求。
                        continue
                    remaining = len(seq) - cached * self.block_size
                else:
                    # 已经执行过部分 Prefill，缓存/页表仍在；从已有进度接着算。
                    remaining = len(seq) - seq.num_cached_tokens
                # 关闭混批时，沿用旧的“只有本 batch 第一个请求可以切 chunk”策略。
                # 混批模式允许 Decode 已占 2 token 后，剩余 510 token 继续给长 Prefill。
                if not self.enable_mixed_batching and requests and remaining > budget:
                    break
                if cached is not None:
                    # None 表示已有页表，无需 allocate；0 表示没命中前缀，但仍需分配。
                    self.block_manager.allocate(seq, cached)
                select(seq, min(remaining, budget), False)

        def decode():
            # 已选 Decode 暂放 selected，从 running 队列中移开，避免后续抢占误伤。
            selected = []
            while self.running and has_room():
                seq = self.running.popleft()
                self._check_capacity(seq)
                while not self.block_manager.can_append(seq):
                    # 第一级：抢占尚未选中的 running 尾部请求，让当前请求优先取得下一页。
                    if self.running:
                        self.preempt(self.running.pop())
                        continue
                    # 第二级：回收尚未选中的、已持有缓存的部分 Prefill 请求。
                    # 纯新请求没有 block_table，抢占它无法腾出任何空间。
                    victim = next((s for s in reversed(self.waiting)
                                   if s.block_table and s.seq_id not in protected), None)
                    if victim is not None:
                        self.waiting.remove(victim)
                        self.preempt(victim)
                        continue
                    self.preempt(seq)
                    # 再无可回收对象，只能让当前请求也退回 waiting，以后重算。
                    break
                else:
                    # Python 的 while...else：只有 while 没有通过 break 退出时才执行。
                    # 这里意味着所需 KV 已能满足；被抢占的请求不会误入此分支继续 Decode。
                    self.block_manager.may_append(seq)
                    select(seq, 1, True)
                    selected.append(seq)
            # 把本轮执行过的请求放到尾部，没轮到的留在前面，形成轮转。
            # 有限请求集可继续推进；持续 Decode 填满预算时，新 Prefill 仍可能长期等待。
            self.running.extend(selected)

        if self.enable_mixed_batching:
            # A/B 各 Decode 1 token，预算 512 时，C 最多获得 510-token Prefill chunk。
            decode()
            prefill()
        else:
            # 对比模式优先 Prefill；本轮只要有 Prefill，就不再加入 Decode。
            prefill()
            if not requests:
                decode()
                if not requests:
                    # Decode 抢占可能使所有请求转为 waiting，立即尝试重算以推进工作。
                    prefill()
        if not requests:
            raise RuntimeError("No executable requests; KV capacity or queue state prevents progress")
        # tuple 内的 ScheduledRequest 是冻结快照；本轮完成哪些请求要等模型结果回来。
        self._pending = SchedulerOutput(tuple(requests), self.num_preemptions - preemptions_before)
        return self._pending

    def preempt(self, seq: Sequence):
        # 调用方先把 seq 从原队列移除，这里再插入 waiting 头部，避免重复入队。
        # 抢占释放 KV 引用，但保留 Prompt 和已有 completion；以后重算恢复 KV，
        # 追上全部已知 token 之前不生成新 completion。
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def postprocess(self, output: SchedulerOutput, token_ids: dict[int, int]):
        # 只接受当前在途计划的结果；用 seq_id 集合检查采样是否缺失、错配或多余。
        # 无采样的部分 Prefill batch 合法，期望结果就是空字典 {}。
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
            # 顺序不能交换：hash_blocks 根据“旧进度 + 本轮计算量”找新完成的整块。
            # 如果先推进 num_cached_tokens，会错过本轮新写好的块，甚至发布错误区间。
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens = r.end_pos
            seq.num_computed_tokens = max(seq.num_computed_tokens, r.end_pos)
            seq.num_scheduled_tokens = 0
            if not r.needs_sample:
                # chunk 没算完已有历史：仅提交 KV 进度，继续留在 waiting。
                # 不追加 token，不计 completion，不判断 EOS，也不进入普通 Decode。
                continue
            if seq.status == SequenceStatus.WAITING:
                self.waiting.remove(seq)
                self.running.append(seq)
            seq.status = SequenceStatus.RUNNING
            seq.is_prefill = False
            # Prefill 末块和普通 Decode 都会走这里；查 ID 而非按整个 batch 的位置配对。
            token_id = token_ids[r.seq_id]
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                # 完成后释放本请求的页引用；其他请求仍在共享的前缀页不会被提前回收。
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished.append(seq)
        self._pending = None
        # 返回本轮刚完成的 Sequence，由 LLMEngine 提取 completion 并记录完成时间。
        return finished
