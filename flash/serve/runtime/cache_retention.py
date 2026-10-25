"""Bounded second chances for reusable, idle vLLM GPU cache blocks.

The native pool still owns hashes, reference counts, and allocations. Only its
free-list eviction order changes; even a fully hot cache always yields blocks.
"""

import time

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue


class ReuseFreeQueue(FreeKVCacheBlockQueue):
    def __init__(self, blocks):
        super().__init__(blocks)
        self._reuse = {}

    def _credit(self, block, now):
        saved = self._reuse.get(block.block_id)
        if saved is None or saved[0] != block.block_hash:
            return 0.0
        return saved[1] * 2 ** (-max(0.0, now - saved[2]) / 60.0)

    def remove(self, block):
        # BlockPool.touch removes idle blocks on a prefix hit. Record only a
        # successful native removal, never an allocation or a cache lookup.
        super().remove(block)
        if block.block_hash is not None:
            now = time.monotonic()
            # One prefill uses several allocations; award four bounded passes
            # per idle reuse so those chunks do not immediately erase history.
            self._reuse[block.block_id] = (
                block.block_hash,
                min(32.0, self._credit(block, now) + 4.0),
                now,
            )

    def _pop(self, now):
        # Limit both scan work and repeated chances. Rotating uses the native
        # intrusive list so all block identities and references stay intact.
        for _ in range(min(32, max(0, self.num_free_blocks - 1))):
            block = self.fake_free_list_head.next_free_block
            credit = self._credit(block, now)
            if block.block_hash is None or credit < 0.5:
                break
            super().append(super().popleft())
            self._reuse[block.block_id] = (block.block_hash, credit - 1.0, now)
        block = super().popleft()
        self._reuse.pop(block.block_id, None)
        return block

    def popleft(self):
        return self._pop(time.monotonic())

    def popleft_n(self, n):
        assert 0 <= n <= self.num_free_blocks
        if not self._reuse:
            return super().popleft_n(n)
        now = time.monotonic()
        return [self._pop(now) for _ in range(n)]


class CacheRetention:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manager = self.kv_cache_manager
        if manager.enable_caching:
            pool = manager.block_pool
            pool.free_block_queue = ReuseFreeQueue(pool.free_block_queue.get_all_free_blocks())
