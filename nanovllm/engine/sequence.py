# 【阅读定位】一个 Sequence 就是一个生成请求的 CPU 侧记录。
# 它保存 token 历史、计算进度和逻辑块到物理块的映射；真正的 K/V 张量在
# ModelRunner 的 GPU 缓存池里，Sequence 本身不保存模型的 hidden states 或 K/V。
# 建议先读本文件，再读 llm_engine.py 的 step()，最后跟进 Scheduler 和 ModelRunner。
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    # WAITING 不只表示“从未执行”：部分 Prefill、被抢占后需要重算的请求也在这里。
    WAITING = auto()
    # RUNNING 表示已经完成 Prefill、正在逐 token Decode，不表示此刻一定占用 GPU。
    # 预算不足时，一个 RUNNING 请求可能本轮没有被选入 batch。
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # 每个逻辑块能容纳的 token 数；LLMEngine 初始化时用配置覆盖这个类属性。
    # 同一引擎内，Sequence、BlockManager 和 ModelRunner 必须使用同样的块大小。
    block_size = 256
    # 进程内自增请求 ID，用于把模型返回的采样结果对应到请求，而非依赖列表位置。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 初始 token_ids 是已经分词的 Prompt；采样产生的 completion 后续追加在尾部。
        if not token_ids:
            raise ValueError("A request must contain at least one prompt token")
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        # 复制列表，避免 append_token() 连带修改调用方传进来的 Prompt 列表。
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        # num_tokens = Prompt 长度 + 已生成的 completion 数；Prompt 长度保持不变。
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        # 当前有效 KV 前缀覆盖 [0, num_cached_tokens)。其中既可能有本请求刚计算的
        # KV，也可能有 Prefix Cache 复用的 KV；它不是 block_table 的总容量。
        self.num_cached_tokens = 0
        # 本轮计划计算多少个输入 token；schedule() 设置，postprocess() 提交后清零。
        self.num_scheduled_tokens = 0
        # 请求级的 Prefill/重算标记；混合 batch 不能拿它当整批共用的阶段标记。
        self.is_prefill = True
        # 历次执行提交过的最高结束位置：抢占后保留，用来统计重复计算的 token。
        # 注意它不能代替 num_cached_tokens：曾经计算过，不代表 KV 现在仍在缓存池里。
        self.num_computed_tokens = 0
        # block_table[i] = 请求的第 i 个逻辑块对应的物理块 ID。
        # 例：block_size=256、block_table=[7, 2]，绝对位置 260 对应物理槽 2*256+4。
        self.block_table = []
        # temperature 控制采样分布；max_tokens 只限制新增 completion，不含 Prompt。
        # ignore_eos=True 时忽略 EOS 提前结束条件，但仍会在 max_tokens 处停止。
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        # 让 len(seq) 表示当前已知 token 的总数，不是“已经算过的 token 数”。
        return self.num_tokens

    def __getitem__(self, key):
        # 支持 seq[i] 和 seq[start:end]；调度快照用切片取出本轮真实输入 token。
        return self.token_ids[key]

    @property
    def is_finished(self):
        # @property 让调用方使用 seq.is_finished，而不是 seq.is_finished()。
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        # 部分 Prefill 只推进 KV 计算进度，不追加 token，因此不会增加这个值。
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        # 用固定的 Prompt 边界切分输入和输出；切片返回新的列表。
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 向上取整 ceil(num_tokens / block_size)：256 个 token 需 1 块，257 个需 2 块。
        # 表示容纳当前全部 token 所需的逻辑块数；Decode 刚追加 token 时，物理表
        # 可能暂时少一块，下一次调度会通过 may_append() 补齐。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一块包含几个 token，范围为 1..block_size；整除时结果是 block_size，
        # 因此不能直接写成 num_tokens % block_size（整除时会错误地得到 0）。
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        # 获取第 i 个逻辑块的 token 内容，供 Prefix Cache 计算哈希和校验命中。
        # 最后一块可能不满；是否允许发布/复用，由 BlockManager 决定。
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        # 自回归生成的关键时间差：采样出一个 token，只代表“知道了它的 ID”，
        # 还没有计算这个新 token 自己的 K/V。它将在下一轮作为模型输入。
        # 例：Prompt 长 10，Prefill 后采样一次，则 num_tokens=11、num_cached_tokens=10。
        # 所以普通 Decode 在每轮提交后通常满足 num_cached_tokens == num_tokens-1。
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # pickle 序列化钩子：保留全部实例字段，包括请求状态和完整 token 历史。
        # 旧方案 Decode 时仅传 last_token，会使其他进程无法切片历史做混批/重算。
        # 当前实际 TP 执行传输的是 SchedulerOutput 快照，不必每轮传整个 Sequence。
        return self.__dict__.copy()

    def __setstate__(self, state):
        # pickle 反序列化时恢复实例字典；不会重新调用 __init__ 分配新的 seq_id。
        self.__dict__.update(state)
