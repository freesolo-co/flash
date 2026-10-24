from types import SimpleNamespace

from flash.serving.deepinfer.approximate_cache import ApproximatePrefixCache, recompute_boundary


def test_only_small_position_preserving_edits_with_long_tail_are_candidates():
    previous = [1] * 2048
    current = previous.copy()
    assert recompute_boundary(previous, current, 2048, 128) is None
    current[300] = 2
    assert recompute_boundary(previous, current, 2048, 128) == 512
    assert recompute_boundary(previous, current, 2047, 128) is None
    current[1800] = 2
    assert recompute_boundary(previous, current, 2048, 128) is None
    current = [2] * 33 + previous[33:]
    assert recompute_boundary(previous, current, 2048, 128) is None


def test_disabled_by_default_and_cancellation_releases_snapshot(monkeypatch):
    monkeypatch.delenv("FLASH_DEEPINFER_APPROXIMATE_NAMESPACES", raising=False)
    value = ApproximatePrefixCache(SimpleNamespace(), cpu_bytes=0)
    assert not value.namespaces
    assert value.blend_snapshots is None
    value.blends["cancelled"] = object()
    value.approximate.add("cancelled")
    value.hits["cancelled"] = 0
    value.forget("cancelled")
    assert not value.blends
    assert not value.approximate
    assert not value.hits
