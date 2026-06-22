"""Unit tests for the pool registry + balancing policy (pure, no IO)."""

from __future__ import annotations

import pytest

from flash.pool.state import Adapter, Backend, NoCapacityError, PoolState


def _state(*backends: Backend) -> PoolState:
    s = PoolState()
    for b in backends:
        s.add_backend(b)
    return s


def test_pick_for_base_least_inflight():
    s = _state(
        Backend(id="a", url="http://a", base_model="Q"),
        Backend(id="b", url="http://b", base_model="Q"),
    )
    s.backends["a"].inflight = 5
    assert s.pick_for_base("Q").id == "b"


def test_pick_for_base_no_capacity():
    s = _state(Backend(id="a", url="http://a", base_model="Q"))
    with pytest.raises(NoCapacityError):
        s.pick_for_base("OTHER")


def test_pick_for_base_skips_saturated_backend():
    # "a" has fewer in-flight but is OVER its cap; "b" is below cap. Even though least-loaded by raw
    # inflight is "a", a saturated backend must not get more work while a healthy one is free.
    s = _state(
        Backend(id="a", url="http://a", base_model="Q", max_concurrency=2),
        Backend(id="b", url="http://b", base_model="Q", max_concurrency=10),
    )
    s.backends["a"].inflight = 2  # saturated (>= cap)
    s.backends["b"].inflight = 5  # below cap
    assert s.backends["a"].saturated and not s.backends["b"].saturated
    assert s.pick_for_base("Q").id == "b"


def test_pick_for_base_all_saturated_falls_back_to_least_loaded():
    # When EVERY backend is saturated we must still return one (queue, don't drop) — the least loaded.
    s = _state(
        Backend(id="a", url="http://a", base_model="Q", max_concurrency=2),
        Backend(id="b", url="http://b", base_model="Q", max_concurrency=2),
    )
    s.backends["a"].inflight = 5
    s.backends["b"].inflight = 3
    assert s.backends["a"].saturated and s.backends["b"].saturated
    assert s.pick_for_base("Q").id == "b"  # least-loaded saturated one, no NoCapacityError


def test_adapter_lazy_load_then_warm_reuse():
    s = _state(Backend(id="a", url="http://a", base_model="Q"))
    s.register_adapter(Adapter(name="run1", base_model="Q", uri="/lora/run1"))
    # first pick: must load
    d = s.pick_backend("run1")
    assert d.backend.id == "a"
    assert d.needs_load is True
    s.mark_loaded("a", "run1")
    # second pick: warm, no load
    d2 = s.pick_backend("run1")
    assert d2.backend.id == "a"
    assert d2.needs_load is False


def test_pick_backend_warm_skips_saturated():
    # Both backends host "run" (warm). "a" has fewer in-flight but is saturated; the warm pick must
    # prefer the non-saturated "b" rather than driving "a" further into overload.
    s = _state(
        Backend(id="a", url="http://a", base_model="Q", max_concurrency=2),
        Backend(id="b", url="http://b", base_model="Q", max_concurrency=10),
    )
    s.register_adapter(Adapter(name="run", base_model="Q", uri="/run"))
    s.mark_loaded("a", "run")
    s.mark_loaded("b", "run")
    s.backends["a"].inflight = 2  # saturated
    s.backends["b"].inflight = 5  # below cap
    d = s.pick_backend("run")
    assert d.backend.id == "b"
    assert d.needs_load is False  # still a warm hit, just the non-saturated one


def test_pick_backend_warm_all_saturated_still_returns():
    # Every warm backend is saturated: still serve from the least-loaded warm one (no eviction churn,
    # no NoCapacityError) — the adapter is already loaded so we queue rather than reload elsewhere.
    s = _state(
        Backend(id="a", url="http://a", base_model="Q", max_concurrency=2),
        Backend(id="b", url="http://b", base_model="Q", max_concurrency=2),
    )
    s.register_adapter(Adapter(name="run", base_model="Q", uri="/run"))
    s.mark_loaded("a", "run")
    s.mark_loaded("b", "run")
    s.backends["a"].inflight = 5
    s.backends["b"].inflight = 3
    d = s.pick_backend("run")
    assert d.backend.id == "b"  # least-loaded saturated warm backend
    assert d.needs_load is False


def test_reregister_with_changed_base_model_clears_stale_placements():
    # Re-registering an adapter NAME with a DIFFERENT base_model must drop placements on the old-base
    # backend, or the warm path would route requests to a backend serving the wrong base model.
    s = _state(
        Backend(id="qgpu", url="http://q", base_model="Q"),
        Backend(id="rgpu", url="http://r", base_model="R"),
    )
    ad = s.register_adapter(Adapter(name="run", base_model="Q", uri="/lora/run"))
    s.mark_loaded("qgpu", "run")
    assert ad.placements == {"qgpu"}
    assert "run" in s.backends["qgpu"].adapters
    # now re-register the SAME name under base model R
    again = s.register_adapter(Adapter(name="run", base_model="R", uri="/lora/run"))
    assert again is ad
    assert ad.base_model == "R"
    assert ad.placements == set()  # old-base placement cleared
    assert ad.loaded_version == {}
    assert "run" not in s.backends["qgpu"].adapters  # detached from the old-base backend
    # the next pick must (re)load onto the correct-base backend, not warm-route to the old one
    d = s.pick_backend("run")
    assert d.backend.id == "rgpu"
    assert d.needs_load is True


def test_multiple_adapters_on_one_gpu():
    # One backend, two runs -> both land on the same GPU (multi-LoRA), no extra GPU needed.
    s = _state(Backend(id="a", url="http://a", base_model="Q", max_loras=8))
    for r in ("run1", "run2", "run3"):
        s.register_adapter(Adapter(name=r, base_model="Q", uri=f"/lora/{r}"))
        d = s.pick_backend(r)
        assert d.backend.id == "a"
        s.mark_loaded("a", r)
    assert s.backends["a"].adapters == {"run1", "run2", "run3"}


def test_free_slot_spreads_across_gpus():
    # New adapters prefer the backend with the FEWEST adapters (spread, not pile-up).
    s = _state(
        Backend(id="a", url="http://a", base_model="Q", max_loras=4),
        Backend(id="b", url="http://b", base_model="Q", max_loras=4),
    )
    s.register_adapter(Adapter(name="r1", base_model="Q", uri="/1"))
    s.mark_loaded("a", "r1")  # a now has 1 adapter, b has 0
    s.register_adapter(Adapter(name="r2", base_model="Q", uri="/2"))
    d = s.pick_backend("r2")
    assert d.backend.id == "b"
    assert d.needs_load is True


def test_eviction_when_all_full():
    s = _state(Backend(id="a", url="http://a", base_model="Q", max_loras=2))
    for r in ("r1", "r2"):
        s.register_adapter(Adapter(name=r, base_model="Q", uri=f"/{r}"))
        s.mark_loaded("a", r)
    s.acquire("a", "r2")  # touch r2 so r1 is the LRU
    s.release("a")
    s.register_adapter(Adapter(name="r3", base_model="Q", uri="/r3"))
    d = s.pick_backend("r3")
    assert d.backend.id == "a"
    assert d.needs_load is True
    assert d.evict == "r1"


def test_failover_excludes_tried_backend():
    s = _state(
        Backend(id="a", url="http://a", base_model="Q"),
        Backend(id="b", url="http://b", base_model="Q"),
    )
    s.register_adapter(Adapter(name="run", base_model="Q", uri="/run"))
    s.mark_loaded("a", "run")
    s.mark_loaded("b", "run")
    first = s.pick_backend("run").backend.id
    second = s.pick_backend("run", exclude={first}).backend.id
    assert {first, second} == {"a", "b"}


def test_reregister_with_changed_uri_marks_placements_stale():
    # Re-registering an adapter whose weight PATH changed must invalidate warm placements (bump the
    # version) so the router reloads the new weights instead of treating backends as already serving
    # the new policy.
    s = _state(Backend(id="a", url="http://a", base_model="Q"))
    ad = s.register_adapter(Adapter(name="run", base_model="Q", uri="/lora/run/v0"))
    s.mark_loaded("a", "run")
    assert ad.stale_placements() == set()  # warm at v0
    again = s.register_adapter(Adapter(name="run", base_model="Q", uri="/lora/run/v1"))
    assert again is ad  # same live entry, placements kept
    assert ad.uri == "/lora/run/v1"
    assert ad.version == 1  # bumped because the weights changed
    assert ad.stale_placements() == {"a"}  # backend now flagged for reload
    # the router's pick will see it as stale -> hot-swap
    d = s.pick_backend("run")
    assert d.backend.id == "a"
    stale = ad.loaded_version.get("a", -1) != ad.version
    assert stale is True


def test_reregister_same_uri_does_not_bump_version():
    # An idempotent re-register (same uri) must NOT churn a reload — only a real weight change does.
    s = _state(Backend(id="a", url="http://a", base_model="Q"))
    ad = s.register_adapter(Adapter(name="run", base_model="Q", uri="/lora/run/v0"))
    s.mark_loaded("a", "run")
    s.register_adapter(Adapter(name="run", base_model="Q", uri="/lora/run/v0", replicas=3))
    assert ad.version == 0
    assert ad.replicas == 3  # other fields still updated
    assert ad.stale_placements() == set()


def test_weight_version_staleness_tracks_placements():
    s = _state(Backend(id="a", url="http://a", base_model="Q"))
    ad = s.register_adapter(Adapter(name="run", base_model="Q", uri="/run"))
    s.mark_loaded("a", "run")
    assert ad.stale_placements() == set()  # loaded at v0
    s.bump_adapter_version("run")  # a weight sync
    assert ad.stale_placements() == {"a"}  # placement now behind
    s.mark_loaded("a", "run")  # router reloaded
    assert ad.stale_placements() == set()


def test_plan_placements_reaches_replica_count():
    s = _state(
        Backend(id="a", url="http://a", base_model="Q"),
        Backend(id="b", url="http://b", base_model="Q"),
        Backend(id="c", url="http://c", base_model="Q"),
    )
    s.register_adapter(Adapter(name="run", base_model="Q", uri="/run", replicas=2))
    decisions = s.plan_placements("run")
    assert len(decisions) == 2
    assert all(d.needs_load for d in decisions)
    assert len({d.backend.id for d in decisions}) == 2


def test_remove_backend_clears_placements():
    s = _state(Backend(id="a", url="http://a", base_model="Q"))
    ad = s.register_adapter(Adapter(name="run", base_model="Q", uri="/run"))
    s.mark_loaded("a", "run")
    s.remove_backend("a")
    assert ad.placements == set()
    with pytest.raises(NoCapacityError):
        s.pick_backend("run")


def test_snapshot_summary():
    s = _state(
        Backend(id="a", url="http://a", base_model="Q", cost_per_hour=2.0),
        Backend(id="b", url="http://b", base_model="R", cost_per_hour=3.0),
    )
    snap = s.snapshot()
    assert snap["summary"]["backends"] == 2
    assert snap["summary"]["healthy_backends"] == 2
    assert set(snap["summary"]["base_models"]) == {"Q", "R"}
    assert snap["summary"]["cost_per_hour"] == 5.0
