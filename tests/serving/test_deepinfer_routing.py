from types import SimpleNamespace

import pytest

from flash.serving.deepinfer.routing import CacheRouter, fingerprint


def entry(namespace, tokens):
    return {"namespace": namespace, "length": len(tokens), "digest": fingerprint(tokens)}


def test_verified_prefix_beats_idle_cold_rank_but_not_large_queue():
    router = CacheRouter(2)
    router.update(1, [entry("tenant", [1] * 512)])
    tokens = [1] * 600
    assert router.choose("tenant", tokens, [0, 0], {}) == 1
    pending = {"busy": SimpleNamespace(rank=1, work=1000)}
    assert router.choose("tenant", tokens, [0, 1], pending) == 0
    assert router.choose("other", tokens, [0, 0], {}) == 0
    assert router.choose("tenant", [2] * 600, [0, 0], {}) == 0


def test_eviction_expiry_and_capacity_remove_affinity():
    router = CacheRouter(2)
    router.update(1, [entry("tenant", [1] * 512)])
    assert router.choose("tenant", [1] * 600, [0, 8], {}) == 0
    router.update(1, [])
    assert router.choose("tenant", [1] * 600, [0, 0], {}) == 0
    router.update(1, [entry("tenant", [1] * 512)])
    router.max_age = -1
    assert router.choose("tenant", [1] * 600, [0, 0], {}) == 0
    with pytest.raises(OverflowError):
        router.choose("tenant", [1] * 600, [8, 8], {})


def test_queued_work_retires_prefill_once_and_tracks_decode():
    import asyncio

    from flash.serving.deepinfer.runtime import Pending, Runtime

    runtime = Runtime()
    pending = Pending(0, asyncio.Queue(), asyncio.Event(), 1160, 1000)
    runtime.pending["request"] = pending
    runtime._deliver({"type": "tokens", "id": "request", "tokens": [1]})
    assert pending.work == 144
    assert pending.prefill == 0
    runtime._deliver({"type": "tokens", "id": "request", "tokens": [2, 3]})
    assert pending.work == 112
