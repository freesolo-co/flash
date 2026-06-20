"""Provider symmetry + the pluggable ``base.Provider`` interface (CPU-only, offline).

Asserts the central property of this refactor: RunPod and Vast expose the IDENTICAL
provider interface behind the IDENTICAL per-provider module layout, the registry hands
out conforming ``Provider`` objects, and the cross-provider allocator ranks/pins/falls
back correctly.
"""

from __future__ import annotations

import importlib

import pytest

# The fixed module set every provider subpackage must expose (symmetry contract).
PROVIDER_MODULES = ("api", "auth", "pricing", "gpus", "jobs", "train", "preflight")
# The fixed method set base.Provider defines.
PROVIDER_METHODS = (
    "is_configured",
    "preflight",
    "gpu_classes",
    "hourly_rate",
    "submit_run",
    "poll",
    "cancel",
    "destroy",
    "gc",
    "sweep_orphans",
)


def test_registry_lists_both_providers():
    from flash.providers import PROVIDER_NAMES, get_provider

    assert PROVIDER_NAMES == ("runpod", "vast")
    assert {get_provider(n).name for n in PROVIDER_NAMES} == {"runpod", "vast"}
    assert get_provider("RunPod ").name == "runpod"  # case/space-insensitive
    with pytest.raises(KeyError):
        get_provider("lambda")


def test_both_providers_implement_the_interface():
    from flash.providers import PROVIDER_NAMES, get_provider
    from flash.providers.base import Provider

    for prov in [get_provider(n) for n in PROVIDER_NAMES]:
        assert isinstance(prov, Provider)  # runtime_checkable Protocol
        for meth in PROVIDER_METHODS:
            assert callable(getattr(prov, meth)), f"{prov.name} missing {meth}"


@pytest.mark.parametrize("provider", ["runpod", "vast"])
def test_identical_module_layout(provider):
    """Both subpackages expose the SAME public module set (the symmetry contract)."""
    for mod in PROVIDER_MODULES:
        importlib.import_module(f"flash.providers.{provider}.{mod}")
    pkg = importlib.import_module(f"flash.providers.{provider}")
    assert hasattr(pkg, "PROVIDER")


def test_method_signatures_match_across_providers():
    """The interface methods take the same parameters on both providers (swappable)."""
    import inspect

    from flash.providers import get_provider

    rp, va = get_provider("runpod"), get_provider("vast")
    for meth in (
        "gpu_classes",
        "hourly_rate",
        "submit_run",
        "poll",
        "cancel",
        "destroy",
        "gc",
        "sweep_orphans",
        "is_configured",
        "preflight",
    ):
        rs = inspect.signature(getattr(rp, meth))
        vs = inspect.signature(getattr(va, meth))
        assert list(rs.parameters) == list(vs.parameters), f"{meth} param mismatch"


def test_gpu_classes_partition_the_table():
    """Each provider owns its rows of the shared table: runpod == enum_member rows,
    vast == vast_name rows; the Vast-only classes are present only on vast."""
    from flash.providers import get_provider
    from flash.providers.base import GPU_INFO

    rp = {g.name for g in get_provider("runpod").gpu_classes()}
    va = {g.name for g in get_provider("vast").gpu_classes()}
    assert rp == {g.name for g in GPU_INFO.values() if g.enum_member}
    assert va == {g.name for g in GPU_INFO.values() if g.vast_name}
    # the three vast-only classes #163 removed are back, and only on vast
    for only in ("L40S", "RTX Pro 4000", "A100 SXM 40GB"):
        assert only in va
        assert only not in rp


def test_sweep_orphans_is_part_of_the_protocol():
    """sweep_orphans is a base.Provider method on BOTH providers (Fix #9): RunPod is a
    no-op (serverless endpoints self-reap), Vast reaps instances. The server iterates
    providers generically through this hook — never a hardcoded `get_provider('vast')`."""
    from flash.providers import PROVIDER_NAMES, get_provider
    from flash.providers.base import Provider

    for prov in [get_provider(n) for n in PROVIDER_NAMES]:
        assert hasattr(prov, "sweep_orphans")
    # RunPod's is a no-op and never raises (no live API call).
    assert get_provider("runpod").sweep_orphans(active_labels={"flash-x"}) == []
    # The method is part of the runtime-checkable protocol surface.
    assert "sweep_orphans" in dir(Provider)


@pytest.mark.parametrize("provider", ["runpod", "vast"])
def test_static_pricing_offline(provider, monkeypatch):
    """Offline, every provider's hourly_rate falls back to the static snapshot."""
    monkeypatch.setenv("FLASH_SKIP_NET", "1")
    pricing = importlib.import_module(f"flash.providers.{provider}.pricing")
    rate = pricing.hourly_rate("RTX 5090")
    assert rate == pytest.approx(0.99)  # the static GpuClass.hourly_usd snapshot


def test_runpod_live_pricing_mock(monkeypatch):
    """Live RunPod rates override the static snapshot (mocked SDK)."""
    from flash.providers.runpod import pricing

    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)
    monkeypatch.setattr(pricing, "_fetch_live_rates", lambda: {"RTX 5090": 1.23})
    monkeypatch.setattr(pricing, "_CACHE_PATH", __import__("pathlib").Path("/nonexistent/x.json"))
    pricing._MEM.update(ts=0.0, rates={})
    assert pricing.hourly_rate("RTX 5090") == 1.23


def test_vast_live_pricing_from_offers_mock(monkeypatch):
    """Vast's hourly_rate is the cheapest live offer for the class (mocked offers)."""
    from flash.providers.vast import jobs, pricing

    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)
    monkeypatch.setenv("VAST_API_KEY", "vk")

    def fake_offers(min_vram_gb, disk_gb, exclude_machine_ids=frozenset()):
        from flash.providers.vast.jobs import VastOffer

        return [
            VastOffer(1, 1, "RTX 3090", 24, 0.19, 12.8, 200.0, 0.99, 5000.0, "CZ"),
            VastOffer(2, 2, "RTX 3090", 24, 0.40, 12.8, 200.0, 0.99, 5000.0, "DE"),
        ]

    monkeypatch.setattr(jobs, "usable_offers", fake_offers)
    assert pricing.hourly_rate("RTX 3090") == 0.19  # cheapest live offer wins


# ---------------------------------------------------------------------------
# cross-provider allocation
# ---------------------------------------------------------------------------
def _mock_both_available(monkeypatch):
    """Make BOTH providers 'available' regardless of env/network for ranking tests."""
    import flash.providers as P

    monkeypatch.setattr(P, "available_providers", lambda: ("runpod", "vast"))
    # allocator imports available_providers by name, so patch its module ref too.
    from flash.providers import allocator

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "vast"))


def _mock_vast_offers(monkeypatch, offers):
    from flash.providers.vast import jobs

    monkeypatch.setattr(jobs, "usable_offers", lambda *a, **k: list(offers))


def _offer(gpu="RTX 3090", dph=0.10, oid=1, mid=1):
    from tests._helpers.vast import make_vast_offer

    return make_vast_offer(offer_id=oid, machine_id=mid, gpu=gpu, dph_total=dph)


def test_allocator_picks_cheaper_vast_over_runpod(monkeypatch):
    from flash.providers.allocator import allocate

    _mock_both_available(monkeypatch)
    # a cheap vast RTX 3090 offer undercuts the cheapest runpod validated 24 GB class
    _mock_vast_offers(monkeypatch, [_offer(gpu="RTX 3090", dph=0.10)])
    a = allocate("Qwen/Qwen3.5-0.8B", "sft")
    assert a.provider == "vast"
    assert a.gpu == "RTX 3090"
    assert a.offer is not None  # the chosen offer is carried for provisioning
    assert a.provider_offers  # full offer book preserved for the live-market walk


def test_allocator_prefers_runpod_on_price_tie(monkeypatch):
    from flash.providers.allocator import allocate

    _mock_both_available(monkeypatch)
    # Same class at the same price on both providers: runpod wins the tie (registry order).
    # Pin the class so the tie is between providers, not classes (no gate now, so an
    # unpinned 0.8B run would otherwise pick the globally-cheapest class).
    _mock_vast_offers(monkeypatch, [_offer(gpu="RTX A5000", dph=0.27)])
    a = allocate("Qwen/Qwen3.5-0.8B", "sft", gpu="RTX A5000")
    assert (a.provider, a.gpu) == ("runpod", "RTX A5000")


def test_allocator_falls_back_to_runpod_when_vast_search_fails(monkeypatch):
    from flash.providers.allocator import allocate
    from flash.providers.vast import jobs

    _mock_both_available(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("vast offer search 500")

    monkeypatch.setattr(jobs, "usable_offers", boom)
    # A vast offer-search failure degrades to runpod-only and never raises (no provider pin).
    a = allocate("Qwen/Qwen3.5-0.8B", "sft")
    assert a.provider == "runpod"


def test_allocator_gpu_pin_chooses_provider(monkeypatch):
    """Pinning a vast-only class routes to vast; the class is honored, provider derived."""
    from flash.providers.allocator import allocate

    _mock_both_available(monkeypatch)
    _mock_vast_offers(monkeypatch, [_offer(gpu="L40S", dph=0.80)])
    a = allocate("Qwen/Qwen3.5-0.8B", "sft", gpu="L40S")
    assert (a.provider, a.gpu) == ("vast", "L40S")


def test_allocator_pinned_larger_gpu_searches_at_pin_size(monkeypatch):
    """Fix #5: pinning a larger Vast class for a small model searches offers at the
    PINNED class's VRAM, not the (smaller) model requirement — otherwise the cheapest-N
    offer window can fill with small cards and never surface the pinned class."""
    from flash.providers import allocator
    from flash.providers.base import GPU_INFO

    _mock_both_available(monkeypatch)
    captured = {}

    def fake_offers(min_vram_gb, disk_gb, exclude_machine_ids=frozenset()):
        captured["min_vram_gb"] = min_vram_gb
        return [_offer(gpu="A40", dph=0.40)]  # A40 = 48 GB, the pinned class

    from flash.providers.vast import jobs

    monkeypatch.setattr(jobs, "usable_offers", fake_offers)
    # Qwen3-0.6B needs ~24 GB, but the user pinned the 48 GB A40 -> search at 48
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "sft", gpu="A40")
    assert (a.provider, a.gpu) == ("vast", "A40")
    assert captured["min_vram_gb"] == GPU_INFO["A40"].vram_gb  # searched at the pin size


def test_allocation_summary_non_vast_provider(monkeypatch):
    """Fix #6: allocation_summary treats `offer` as an opaque per-provider hint — it only
    formats Vast specifics when provider=='vast', never misformatting/raising otherwise."""
    from flash.providers.allocator import allocation_summary
    from flash.providers.base import Allocation, Candidate

    # A non-vast allocation carrying a NON-vast offer object (no offer_id/geolocation):
    # allocation_summary must not touch it.
    sentinel = object()  # would raise AttributeError if treated as a vast offer
    cand = Candidate("runpod", "RTX A5000", 0.27, 24, offer=sentinel)
    alloc = Allocation(
        provider="runpod",
        gpu="RTX A5000",
        hourly_usd=0.27,
        min_vram_gb=24,
        candidates=(cand,),
        offer=sentinel,
    )
    out = allocation_summary(alloc)  # must not raise
    assert "RTX A5000 on runpod" in out
    assert "vast offer" not in out


# ---------------------------------------------------------------------------
# handle round-trip through the interface
# ---------------------------------------------------------------------------
def test_jobhandle_roundtrip_tags_provider():
    from flash.providers.base import JobHandle

    for name, extra in (
        ("runpod", {"endpoint_id": "ep", "job_id": "j"}),
        ("vast", {"instance_id": 7, "offer_id": 5}),
    ):
        h = JobHandle(provider=name, data=extra)
        d = h.to_dict()
        assert d["provider"] == name
        back = JobHandle.from_dict(d)
        assert back.provider == name
        assert back.data == extra
    # a pre-provider handle (no `provider` key) defaults to runpod
    assert JobHandle.from_dict({"endpoint_id": "ep", "job_id": "j"}).provider == "runpod"


def test_provider_cancel_destroy_dispatch(monkeypatch):
    """cancel/destroy on each provider hit that provider's REST surface."""
    from flash.providers import get_provider
    from flash.providers.base import JobHandle

    # runpod
    from flash.providers.runpod import api as rp_api

    cancelled, deleted = [], []
    monkeypatch.setattr(rp_api, "cancel_job", lambda e, j: cancelled.append((e, j)))
    monkeypatch.setattr(rp_api, "delete_endpoint", lambda e: deleted.append(e) or True)
    rh = JobHandle("runpod", {"endpoint_id": "ep", "job_id": "j"})
    get_provider("runpod").cancel(rh)
    get_provider("runpod").destroy(rh)
    assert cancelled == [("ep", "j")]
    assert deleted == ["ep"]

    # vast
    from flash.providers.vast import api as v_api

    destroyed = []
    monkeypatch.setattr(v_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    vh = JobHandle("vast", {"instance_id": 9})
    get_provider("vast").cancel(vh)
    get_provider("vast").destroy(vh)
    assert destroyed == [9, 9]
