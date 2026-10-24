"""Shared preparation, native prefix ownership, and token delivery invariants."""

from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flash.serve.runtime.prompt_cache import cached_prompt_tokens
from flash.serve.runtime.token_batch import TokenBatch
from flash.serving.deepinfer.prefix_cache import Prefix, PrefixCache
from flash.serving.deepinfer.protocol import parse_request
from flash.serving.deepinfer.runtime import Runtime
from tests.serving.test_deepinfer import body


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_shared_cache_hits_are_detached_and_eviction_is_lru():
    cache = OrderedDict()
    calls = []

    def encode():
        calls.append(1)
        return [1, 2]

    result = await cached_prompt_tokens(cache, "a", encode, max_entries=2)
    result.append(99)
    assert await cached_prompt_tokens(cache, "a", encode, max_entries=2) == [1, 2]
    await cached_prompt_tokens(cache, "b", encode, max_entries=2)
    await cached_prompt_tokens(cache, "a", encode, max_entries=2)
    await cached_prompt_tokens(cache, "c", encode, max_entries=2)
    assert list(cache) == ["a", "c"]
    assert len(calls) == 3


@pytest.mark.anyio
async def test_deepinfer_prompt_cache_separates_thinking_tenant_and_checks_output_budget():
    runtime = Runtime()
    calls = []

    def encode(request):
        calls.append(request)
        return [2] * 32000

    runtime.tokenize = encode
    request = parse_request(body(max_tokens=1))
    await runtime.prepare_tokens(request)
    await runtime.prepare_tokens(request)
    await runtime.prepare_tokens(replace(request, thinking=True))
    await runtime.prepare_tokens(replace(request, cache_namespace="a" * 64))
    assert len(calls) == 3
    with pytest.raises(ValueError, match="context"):
        await runtime.prepare_tokens(replace(request, max_tokens=1024))
    assert len(calls) == 3


def test_token_delivery_first_token_deadline_size_and_terminal_flush():
    batch = TokenBatch(3, 0.05)
    batch.add([1], 10)
    assert batch.take(10) == [1]
    batch.add([2], 10.01)
    assert batch.take(10.02) == []
    assert batch.take(10.07) == [2]
    batch.add([3, 4, 5], 10.08)
    assert batch.take(10.08) == [3, 4, 5]
    batch.add([6], 10.09)
    assert batch.take(10.09, finished=True) == [6]
    assert not batch.pending


@pytest.mark.parametrize("limit", [1, 4, 16])
def test_batching_preserves_every_token_in_order(limit):
    batch = TokenBatch(limit, 0.05)
    expected = list(range(100))
    emitted = []
    for i, token in enumerate(expected):
        now = i * 0.007
        batch.add([token], now)
        emitted.extend(batch.take(now, finished=i == 99))
    assert emitted == expected


@pytest.mark.parametrize("now", [1.0, 10000.0])
def test_prefix_eviction_releases_native_state_and_expired_entries(monkeypatch, now):
    monkeypatch.setattr("flash.serving.deepinfer.prefix_cache.time.monotonic", lambda: now)
    released = []
    engine = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        scheduler=SimpleNamespace(free_seq=released.append, get_num_free_gpu_blocks=lambda: 0),
    )
    cache = PrefixCache(engine)
    cache.entries["old"] = Prefix("a", (1,), "old-state", now - cache.ttl)
    cache.prune()
    assert released == ["old-state"]
    cache.entries["pressure"] = Prefix("a", (1,), "pressure-state", now)
    cache.prune(256)
    assert released == ["old-state", "pressure-state"]
    assert not cache.entries


def test_prefix_lookup_requires_exact_tokens_and_same_namespace(monkeypatch):
    created = []
    engine = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        scheduler=SimpleNamespace(free_seq=lambda seq: None, get_num_free_gpu_blocks=lambda: 10000),
        add_request=lambda **kwargs: created.append(kwargs) or SimpleNamespace(num_cached_tokens=0),
    )
    cache = PrefixCache(engine)
    cache.entries["one"] = Prefix("tenant-a", tuple(range(128)), "snapshot", float("inf"))
    forks = []
    monkeypatch.setattr(cache, "_fork", lambda *args: forks.append(args))
    command = {"id": "hit", "tokens": list(range(130)), "cache_namespace": "tenant-a"}
    cache.admit(command, object())
    assert cache.hits["hit"] == 128
    assert len(forks) == 1
    cache.admit({**command, "id": "other", "cache_namespace": "tenant-b"}, object())
    assert cache.hits["other"] == 0
    assert len(created) == 1
    cache.admit({**command, "id": "different", "tokens": [999, *range(1, 130)]}, object())
    assert cache.hits["different"] == 0
    assert len(created) == 2
    cache.forget("hit")
    assert "one" in cache.entries


def test_prefix_budget_counts_allocated_suffix_blocks_too():
    added = []
    engine = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        scheduler=SimpleNamespace(free_seq=lambda seq: None, get_num_free_gpu_blocks=lambda: 10000),
        add_request=lambda **kwargs: added.append(kwargs) or SimpleNamespace(num_cached_tokens=0),
    )
    cache = PrefixCache(engine, max_tokens=256)
    cache.admit(
        {
            "id": "too-large",
            "tokens": list(range(400)),
            "prefix_tokens": 128,
            "cache_namespace": "tenant",
        },
        object(),
    )
    assert added[-1]["chunk_checkpoints"] is None
    assert not cache.pending
    cache.admit(
        {
            "id": "fits",
            "tokens": list(range(200)),
            "prefix_tokens": 128,
            "cache_namespace": "tenant",
        },
        object(),
    )
    assert added[-1]["chunk_checkpoints"] == [128]
    assert cache.pending["fits"][-1] == 256
