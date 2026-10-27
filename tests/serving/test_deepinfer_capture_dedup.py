"""Duplicate checkpoints keep independent continuations without evicting useful prefixes."""

import sys
from types import SimpleNamespace

import pytest

from flash.serving.deepinfer.runtime.prefix_cache import PrefixCache


@pytest.fixture
def setup_cache(monkeypatch):
    added, freed, forks, trace = [], [], [], []
    sequences = {}

    def add_request(**kwargs):
        added.append(kwargs)
        seq = SimpleNamespace(num_cached_tokens=0, request_id=kwargs["request_id"])
        sequences[seq.request_id] = seq
        return seq

    def free(seq):
        freed.append(seq)
        trace.append(("free", seq))

    def fork(seq, *args):
        forks.append((seq, args))
        trace.append(("fork", seq))

    engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            get_num_free_gpu_blocks=lambda: 10000,
            detach_waiting_for_prefix_cache=lambda rid: sequences[rid],
            free_seq=free,
        ),
        cache_config=SimpleNamespace(block_size=128),
        add_request=add_request,
        flush_inflight_reqs=list,
    )
    cache = PrefixCache(engine, max_entries=8, max_tokens=4096)
    monkeypatch.setattr(cache, "_fork", fork)
    monkeypatch.setitem(
        sys.modules,
        "deepinfer._C.engine",
        SimpleNamespace(RequestOutput=SimpleNamespace(clone=lambda x: x)),
    )
    return cache, added, sequences, freed, forks, trace


def command(index, namespace="tenant", prefix=None):
    return {
        "id": str(index),
        "cache_namespace": namespace,
        "tokens": (list(range(128)) if prefix is None else prefix)
        + [1000 + index, 2000 + index, 3000],
        "prefix_tokens": 128,
    }


def capture(cache):
    for seq, _, tokens, _, _ in cache.pending.values():
        seq.num_cached_tokens = len(tokens)
    cache.capture_ready()


def test_burst_preserves_checkpoints_and_frees_duplicates_after_forking(setup_cache):
    cache, added, sequences, freed, forks, trace = setup_cache
    for i in range(8):
        cache.admit(command(i), None)
    assert len(added) == 8
    assert sum(bool(row["chunk_checkpoints"]) for row in added) == 7
    assert all(value == 0 for value in cache.hits.values())
    capture(cache)
    assert len(cache.entries) == 1
    assert len(forks) == 7
    assert len(freed) == 6
    assert next(iter(cache.entries.values())).sequence is sequences["1"]
    for seq in freed:
        assert trace.index(("fork", seq)) < trace.index(("free", seq))
    cache.admit(command(8), None)
    assert cache.hits["8"] == 128
    assert len(added) == 8


def test_duplicate_does_not_evict_an_unrelated_prefix_at_entry_limit(setup_cache):
    cache, _, sequences, freed, *_ = setup_cache
    cache.max_entries = 2
    for i in range(2):
        cache.admit(command(i), None)
    capture(cache)
    for i in range(2, 5):
        cache.admit(command(i, prefix=[999, *range(1, 128)]), None)
    capture(cache)
    assert len(cache.entries) == 2
    assert len(freed) == 1
    assert freed[0] is sequences["4"]
    assert any(prefix.sequence is sequences["1"] for prefix in cache.entries.values())


def test_retention_separates_tenants_and_exact_prefixes(setup_cache):
    cache, _, _, freed, *_ = setup_cache
    for i in range(2):
        cache.admit(command(i), None)
        cache.admit(command(i + 10, namespace="other"), None)
        cache.admit(command(i + 20, prefix=[999, *range(1, 128)]), None)
    capture(cache)
    assert len(cache.entries) == 3
    assert not freed


def test_cancelled_checkpoint_is_not_captured_or_freed_again(setup_cache):
    cache, _, sequences, freed, forks, _ = setup_cache
    for i in range(4):
        cache.admit(command(i), None)
    cache.forget("2")
    capture(cache)
    assert len(cache.entries) == 1
    assert len(forks) == 2
    assert len(freed) == 1
    assert freed[0] is sequences["3"]
    assert all(seq is not sequences["2"] for seq, _ in forks)
