"""Concurrent cache misses retain one donor while every request still runs."""

import sys
from types import SimpleNamespace

import pytest

from flash.serving.deepinfer.runtime.prefix_cache import PrefixCache


@pytest.fixture
def setup_cache(monkeypatch):
    added, detached, forks = [], [], []
    sequences = {}

    def add_request(**kwargs):
        added.append(kwargs)
        seq = SimpleNamespace(num_cached_tokens=0, request_id=kwargs["request_id"])
        sequences[seq.request_id] = seq
        return seq

    def detach(request_id):
        detached.append(request_id)
        return sequences[request_id]

    engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            get_num_free_gpu_blocks=lambda: 10000,
            detach_waiting_for_prefix_cache=detach,
            free_seq=lambda seq: None,
        ),
        cache_config=SimpleNamespace(block_size=128),
        add_request=add_request,
        flush_inflight_reqs=list,
    )
    cache = PrefixCache(engine, max_entries=8, max_tokens=4096)
    monkeypatch.setattr(cache, "_fork", lambda *args: forks.append(args))
    monkeypatch.setitem(
        sys.modules,
        "deepinfer._C.engine",
        SimpleNamespace(RequestOutput=SimpleNamespace(clone=lambda x: x)),
    )
    return cache, added, sequences, detached, forks


def command(index, namespace="tenant", prefix=None):
    return {
        "id": str(index),
        "cache_namespace": namespace,
        "tokens": (list(range(128)) if prefix is None else prefix)
        + [1000 + index, 2000 + index, 3000],
        "prefix_tokens": 128,
    }


def test_burst_prefills_every_request_but_retains_only_one_snapshot(setup_cache):
    cache, added, sequences, detached, forks = setup_cache
    for i in range(8):
        cache.admit(command(i), None)
    assert len(added) == 8
    assert [x["request_id"] for x in added if x["chunk_checkpoints"]] == ["1"]
    assert set(cache.pending) == {"1"}
    assert all(x == 0 for x in cache.hits.values())
    sequences["1"].num_cached_tokens = 128
    cache.capture_ready()
    assert detached == ["1"]
    assert len(cache.entries) == 1
    assert not cache.pending_prefixes
    cache.admit(command(8), None)
    assert cache.hits["8"] == 128
    assert len(added) == 8
    assert len(forks) == 2


def test_checkpoint_cancellation_allows_another_donor(setup_cache):
    cache, added, *_ = setup_cache
    for i in range(3):
        cache.admit(command(i), None)
    cache.forget("2")
    cache.admit(command(3), None)
    assert added[-1]["chunk_checkpoints"] is None
    cache.forget("1")
    cache.admit(command(4), None)
    assert added[-1]["chunk_checkpoints"] == [128]
    assert set(cache.pending) == {"4"}


def test_deduplication_separates_tenants_and_exact_prefixes(setup_cache):
    cache, added, *_ = setup_cache
    for i in range(2):
        cache.admit(command(i), None)
        cache.admit(command(i + 10, namespace="other"), None)
        cache.admit(command(i + 20, prefix=[999, *range(1, 128)]), None)
    assert len(cache.pending) == 3
    assert sum(bool(x["chunk_checkpoints"]) for x in added) == 3
    cache.close()
    assert not cache.pending_prefixes


def test_failed_native_admission_does_not_claim_a_checkpoint(setup_cache):
    cache, added, *_ = setup_cache
    cache.admit(command(0), None)
    add_request = cache.engine.add_request
    cache.engine.add_request = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="refused"):
        cache.admit(command(1), None)
    assert not cache.pending_prefixes
    cache.engine.add_request = add_request
    cache.admit(command(2), None)
    assert added[-1]["chunk_checkpoints"] == [128]
