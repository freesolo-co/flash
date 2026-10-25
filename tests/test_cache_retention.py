"""Run against the serving image's native vLLM block objects and free list."""

import pytest

native = pytest.importorskip("vllm.v1.core.kv_cache_utils")
if not hasattr(native, "FreeKVCacheBlockQueue"):
    pytest.skip("requires native vLLM", allow_module_level=True)

from flash.serve.runtime.cache_retention import ReuseFreeQueue


def blocks(n):
    result = [native.KVCacheBlock(i) for i in range(n)]
    for block in result:
        block.block_hash = bytes([block.block_id])
    return result


def test_reuse_survives_scan_without_pinning():
    items = blocks(4)
    queue = ReuseFreeQueue(items)
    queue.remove(items[0])
    queue.prepend_n([items[0]])
    assert queue.popleft() is items[1]
    remaining = queue.popleft_n(3)
    assert {b.block_id for b in remaining} == {0, 2, 3}
    assert queue.num_free_blocks == 0
    assert queue.get_all_free_blocks() == []
    assert all(b.ref_cnt == 0 and b.next_free_block is None for b in items)


def test_hot_cache_allocation_and_preflight():
    items = blocks(40)
    queue = ReuseFreeQueue(items)
    for _ in range(8):
        for item in items:
            queue.remove(item)
            queue.append(item)
    with pytest.raises(AssertionError):
        queue.popleft_n(41)
    assert queue.num_free_blocks == 40
    assert len({b.block_id for b in queue.popleft_n(40)}) == 40
    assert not queue._reuse
    queue.append_n(items)
    assert queue.get_all_free_blocks() == items


def test_expired_or_replaced_hash_does_not_protect(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("flash.serve.runtime.cache_retention.time.monotonic", lambda: now[0])
    for replace in (False, True):
        items = blocks(2)
        queue = ReuseFreeQueue(items)
        queue.remove(items[0])
        queue.prepend_n([items[0]])
        if replace:
            items[0].reset_hash()
            items[0].block_hash = b"replacement"
        else:
            now[0] += 240.0
        assert queue.popleft() is items[0]
        assert not queue._reuse


def test_cold_batch_preserves_native_order():
    items = blocks(4)
    queue = ReuseFreeQueue(items)
    assert queue.popleft_n(0) == []
    assert queue.popleft_n(4) == items
    with pytest.raises(ValueError, match="No free blocks available"):
        queue.popleft()
