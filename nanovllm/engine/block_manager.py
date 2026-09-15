# 【阅读定位】Paged KV Cache 的 CPU 侧管理器：管理页编号、归属和可复用前缀。
# 逻辑块：某请求 token 序列按 block_size 切出来的第 0、1、2... 块。
# 物理块：GPU KV 池中的页，用整数 block_id 标识；不同请求可以引用同一物理块。
# 本文件只维护元数据，不分配/搬运 GPU 张量；真正的张量由 ModelRunner 创建，
# Attention 按 slot_mapping 写入。一个物理块 ID 对应各层 KV 池中的同一页编号。
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        # 有多少请求的 block_table 引用了这一页；0 表示可重新分配，不一定没有旧 KV。
        self.ref_count = 0
        # -1 表示没有已发布的完整块哈希；尚未计算完的页不能参与 Prefix Cache 命中。
        self.hash = -1
        # 用于检查哈希命中的 token 内容；这里是 Python token 列表，不是 K/V 张量。
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        # 仅在对应完整块的 KV 已计算后，更新供后续请求查找的前缀元数据。
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        # 物理页改派给新内容时调用：新请求取得第一个引用，原内容的哈希作废。
        # 不需要把 GPU 内存清零；后续查询只能读到本请求已经有效写入的区间。
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # “完整前缀的哈希 -> 可复用物理页”。页即使暂时无人引用，也可能仍在此索引中。
        self.hash_to_block_id: dict[int, int] = dict()
        # free 与 used 是互斥集合：free 页的 ref_count=0，used 页的 ref_count>0。
        # deque 使从头分配/向尾回收更方便；这里没有实现复杂的 LRU 内存管理策略。
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 链式哈希 H_i = hash(H_(i-1), 本块 token)，把前文也纳入身份。
        # 仅当前块 token 相同不够：前文不同，同一个 token 的 Attention/KV 也可能不同。
        # prefix=-1 表示第一块，没有前一块哈希；返回值是 64 位无符号整数。
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        # 分配前调用方已检查容量；取出一页并使旧的 Prefix Cache 身份失效。
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 同一哈希的索引可能已指向另一物理页，只删除“仍指向当前页”的旧映射。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        # 最后一个引用释放后才把页归还 free；保留 hash/token 元数据和原 GPU 内容，
        # 这样在它真正被重新分配之前，另一个相同前缀请求仍有机会复用。
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        # 只做可行性检查，不改队列、页表或引用计数。
        # 返回 >=0：可复用的连续前缀块数（0 也代表可以分配）；-1：当前空闲页不足。
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # 故意不复用最后一个逻辑块，即使它恰好满块也保留计算：
        # 必须至少计算最后一个输入位置，才能获得用来采样下一个 token 的 logits。
        # 最后一块也可能继续增长，让它保持独占可避免写入其他请求共享的尾块。
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 前缀必须连续命中；某块没命中后，后续块不能脱离前文独立复用。
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                # 共享已被占用的页，不会消耗 free 页；复用“缓存仍在的 free 页”
                # 则需要把它从 free 移走，所以那种情况不能在这里扣减所需空闲页数。
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # 与 can_allocate() 配套：先取得可复用页，再分配剩余逻辑块。
        # 这里仍为当前已知的整个 Prompt/重算历史分配空间，尚未做 Prompt KV 增量分配。
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # 已有其他请求持有它：共享同一份只读前缀 KV，引用计数加一。
                block.ref_count += 1
            else:
                # 页无人使用但旧数据尚未覆盖：重新激活，不能 reset() 丢掉其缓存身份。
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # 只有复用部分有“有效 KV”；新分配的页只是容量，不能算进缓存进度。
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        # 请求完成/被抢占都会走这里：逐页释放“本请求的引用”，不是直接删除共享页。
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        # 保留 token 历史和 num_computed_tokens，供抢占后重算；仅清除当前缓存归属。
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # Decode 新 token 是否跨入下一块？对比所需逻辑块数与已分配页表长度。
        # 例：已经有 1 个 256-token 块，当前已知 257 个 token，则还需要 1 页。
        return len(self.free_block_ids) >= max(0, seq.num_blocks - len(seq.block_table))

    def may_append(self, seq: Sequence):
        # 普通 Decode 每轮只增加 1 个 token，因此这里最多补 1 页。
        # 未跨块时只会在已有尾块中写入下一槽，不需要修改 block_table。
        if seq.num_blocks > len(seq.block_table):
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        # 在本轮模型执行完成之后、num_cached_tokens 推进之前调用。
        # floor(old_progress/block_size) 到 floor(new_progress/block_size) 之间，
        # 正好是本轮新完成的完整逻辑块；未满的尾块不能发布为可复用缓存。
        # 例：block_size=256，进度从 255 到 257，则新完成的是第 0 块 [0,256)。
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        # 本轮可能从中间块开始，因此沿用前一个已完成块的链式哈希。
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
