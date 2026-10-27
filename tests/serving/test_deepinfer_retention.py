from types import SimpleNamespace

from flash.serving.deepinfer.runtime.prefix_cache import Prefix, PrefixCache


def cache():
    freed = []
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(free_seq=freed.append, get_num_free_gpu_blocks=lambda: 0),
        cache_config=SimpleNamespace(block_size=128),
    )
    result = PrefixCache(engine)
    result.entries.update(
        {
            name: Prefix("tenant", (1,) * 256, name, 100, 256, frequency_at=100)
            for name in ("hot", "cold")
        }
    )
    return result, freed


def test_reuse_survives_scan_and_ages_out(monkeypatch):
    monkeypatch.setattr("flash.serving.deepinfer.runtime.prefix_cache.time.monotonic", lambda: 100)
    frequency, _ = cache()
    for _ in range(3):
        frequency._touch("hot", frequency.entries["hot"])
    frequency._touch("cold", frequency.entries["cold"])
    assert frequency._victim() == "cold"
    monkeypatch.setattr("flash.serving.deepinfer.runtime.prefix_cache.time.monotonic", lambda: 400)
    frequency._touch("cold", frequency.entries["cold"])
    assert frequency._victim() == "hot"


def test_ttl_still_frees_popular_entries_and_cost_counts_full_allocation(monkeypatch):
    monkeypatch.setattr("flash.serving.deepinfer.runtime.prefix_cache.time.monotonic", lambda: 100)
    value, freed = cache()
    value.entries["hot"].frequency = 3
    value.entries["hot"].storage_tokens = 2048
    assert value._victim() == "hot"
    value.entries["hot"].frequency = 1000
    value.ttl = 1
    monkeypatch.setattr("flash.serving.deepinfer.runtime.prefix_cache.time.monotonic", lambda: 102)
    value.prune()
    assert set(freed) == {"hot", "cold"}
    assert not value.entries


def test_one_off_prompts_skip_checkpoints_but_reused_tenant_prefixes_are_retained():
    added = []
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(get_num_free_gpu_blocks=lambda: 10000),
        cache_config=SimpleNamespace(block_size=128),
        add_request=lambda **kwargs: added.append(kwargs) or SimpleNamespace(num_cached_tokens=0),
    )
    value = PrefixCache(engine)
    command = {
        "id": "first",
        "tokens": list(range(160)),
        "prefix_tokens": 128,
        "cache_namespace": "tenant-a",
    }
    value.admit(command, None)
    assert added[-1]["chunk_checkpoints"] is None
    assert not value.pending
    value.admit({**command, "id": "other-tenant", "cache_namespace": "tenant-b"}, None)
    assert added[-1]["chunk_checkpoints"] is None
    value.admit({**command, "id": "different", "tokens": [999, *range(1, 160)]}, None)
    assert added[-1]["chunk_checkpoints"] is None
    # The continuation may differ; admission depends on the exact reusable prefix.
    value.admit({**command, "id": "reused", "tokens": [*range(128), *range(500, 550)]}, None)
    assert added[-1]["chunk_checkpoints"] == [128]
    assert set(value.pending) == {"reused"}
    assert all(
        count == 0 for count in value.hits.values()
    )  # Observation is not a billed cache hit.


def test_admission_history_is_bounded_expires_and_does_not_retain_token_lists(monkeypatch):
    module = "flash.serving.deepinfer.runtime.prefix_cache"
    monkeypatch.setattr(module + ".ADMISSION_HISTORY_LIMIT", 2)
    monkeypatch.setattr(module + ".time.monotonic", lambda: 100)
    value = PrefixCache(None, ttl=10)
    assert not value._checkpoint_key("a", [1] * 128)
    assert not value._checkpoint_key("a", [2] * 128)
    assert value._checkpoint_key("a", [1] * 128)
    assert not value._checkpoint_key("a", [3] * 128)
    assert len(value.observed) == 2
    assert all(len(key[1]) == 64 for key in value.observed)
    assert not value._checkpoint_key("a", [2] * 128)  # Evicted observation is cold again.
    monkeypatch.setattr(module + ".time.monotonic", lambda: 111)
    assert not value._checkpoint_key("a", [2] * 128)
    assert len(value.observed) == 1
    value.close()
    assert not value.observed
