from types import SimpleNamespace

from flash.serving.deepinfer.prefix_cache import Prefix, PrefixCache


def cache(policy):
    freed = []
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(free_seq=freed.append, get_num_free_gpu_blocks=lambda: 0),
        cache_config=SimpleNamespace(block_size=128),
    )
    result = PrefixCache(engine, retention=policy)
    result.entries.update(
        {
            name: Prefix("tenant", (1,) * 256, name, 100, 256, frequency_at=100)
            for name in ("hot", "cold")
        }
    )
    return result, freed


def test_reuse_survives_scan_and_ages_out(monkeypatch):
    monkeypatch.setattr("flash.serving.deepinfer.prefix_cache.time.monotonic", lambda: 100)
    frequency, _ = cache("frequency")
    lru, _ = cache("lru")
    for value in (frequency, lru):
        for _ in range(3):
            value._touch("hot", value.entries["hot"])
        value._touch("cold", value.entries["cold"])
    assert frequency._victim() == "cold"
    assert lru._victim() == "hot"
    monkeypatch.setattr("flash.serving.deepinfer.prefix_cache.time.monotonic", lambda: 400)
    frequency._touch("cold", frequency.entries["cold"])
    assert frequency._victim() == "hot"


def test_ttl_still_frees_popular_entries_and_cost_counts_full_allocation(monkeypatch):
    monkeypatch.setattr("flash.serving.deepinfer.prefix_cache.time.monotonic", lambda: 100)
    value, freed = cache("frequency")
    value.entries["hot"].frequency = 3
    value.entries["hot"].storage_tokens = 2048
    assert value._victim() == "hot"
    value.entries["hot"].frequency = 1000
    value.ttl = 1
    monkeypatch.setattr("flash.serving.deepinfer.prefix_cache.time.monotonic", lambda: 102)
    value.prune()
    assert set(freed) == {"hot", "cold"}
    assert not value.entries
