"""usable_offers: every quality/safety filter, alias mapping, and ordering (mocked API)."""

from __future__ import annotations

import pytest


def _wall_capped_spec(max_wall_seconds: float):
    """A rentable vast spec whose gpu carries an explicit wall grant.

    max_wall_seconds is platform-managed and stripped from the public spec, so set it the way the
    runner does -- by replacing gpu on a built spec rather than passing it through from_dict.
    """
    from dataclasses import replace

    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "gpu": {"type": "H100", "count": 1, "provider": "vast"},
            "train": {"max_examples": 4},
        }
    )
    return replace(spec, gpu=replace(spec.gpu, max_wall_seconds=max_wall_seconds))


def _offer(**kw) -> dict:
    """A fully-passing verified-datacenter RTX 4090 offer; override fields per case."""
    base = {
        "id": 1000,
        "machine_id": 1,
        "gpu_name": "RTX 4090",
        "gpu_ram": 24576,
        # single-card box. the count is load-bearing twice over (it sizes the rented box and divides
        # dph_total into the per-card rate), so usable_offers re-checks it client-side rather than
        # trusting the server honoured the num_gpus filter -- a row without it is dropped.
        "num_gpus": 1,
        "dph_total": 0.25,
        "verification": "verified",
        "hosting_type": 1,
        "reliability2": 0.995,
        "cuda_max_good": 12.8,
        "disk_space": 200.0,
        "inet_down": 5000.0,
        "geolocation": "Czechia, CZ",
    }
    base.update(kw)
    return base


FIXTURE = [
    _offer(id=1, dph_total=0.25),  # KEEP: cheapest valid 4090
    _offer(
        id=2, gpu_ram=24564, dph_total=0.34
    ),  # KEEP: under-reported VRAM ok (board < class nominal)
    _offer(id=3, gpu_name="RTX A6000", gpu_ram=49140, dph_total=0.80),  # DROP: retired class
    _offer(id=4, gpu_name="A100 SXM4", gpu_ram=40960, dph_total=0.87),  # KEEP: 40GB SXM4 variant
    _offer(id=5, gpu_name="A100 SXM4", gpu_ram=81920, dph_total=1.20),  # KEEP: 80GB SXM4 variant
    _offer(id=10, dph_total=0.08, hosting_type=0),  # DROP: community host (secrets ship to it)
    _offer(id=11, dph_total=0.10, verification="unverified"),  # DROP: not verified
    _offer(id=12, dph_total=0.11, reliability2=0.80),  # DROP: reliability floor
    _offer(
        id=13, gpu_name="Tesla T4", gpu_ram=16384, dph_total=0.05
    ),  # DROP: sub-Ampere (unmanaged)
    _offer(id=14, dph_total=0.12, disk_space=20.0),  # DROP: too little disk
    _offer(id=15, dph_total=0.13, inet_down=50.0),  # DROP: too slow to pull weights
    _offer(id=16, gpu_name="RTX 5090", gpu_ram=32607, dph_total=0.20, cuda_max_good=12.4),
    # ^ DROP: Blackwell class on a pre-CUDA-13 driver (PTX JIT would fail)
    _offer(id=17, gpu_name="RTX 5090", gpu_ram=32607, dph_total=0.35, cuda_max_good=13.1),
    # ^ KEEP: Blackwell with a CUDA-13 driver
]


def test_vast_a100_pcie_offer_resolves_by_vram():
    # Codex: Vast lists 40 GB A100 PCIe boards as "A100 PCIE" (the vast_name of the 80 GB A100 PCIe
    # class). A 40 GB "A100 PCIE" offer fails the 80 GB VRAM gate, so without aliasing it onto the 40 GB
    # class it drops real A100 capacity for 35-40 GB runs. The largest-fitting-class rule keeps an 80 GB
    # board on the 80 GB class; only a 40 GB one lands on the 40 GB class.
    from flash.providers.core.base import vast_gpu_for_offer

    assert vast_gpu_for_offer("A100 PCIE", 80 * 1024) == "A100 PCIe"  # 80 GB -> 80 GB class
    assert (
        vast_gpu_for_offer("A100 PCIE", 40 * 1024) == "A100 SXM 40GB"
    )  # 40 GB -> 40 GB class (was dropped)


def test_usable_offers_filters_and_order(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    captured = {}

    def fake_search(
        min_vram_mb,
        *,
        min_disk_gb=0,
        min_reliability=0.95,
        min_duration_seconds=0,
        limit=64,
        num_gpus=1,
        extra_q=None,
    ):
        captured["min_vram_mb"] = min_vram_mb
        captured["min_disk_gb"] = min_disk_gb
        return list(FIXTURE)

    monkeypatch.setattr(vast_api, "search_offers", fake_search)
    out = vast.usable_offers(24, disk_gb=60)
    # community host (id=10) dropped — datacenter-only (run secrets ship to the box)
    assert [o.offer_id for o in out] == [1, 2, 17, 4, 5]  # dph ascending, junk gone
    assert captured["min_disk_gb"] == 60
    # the server-side VRAM filter carries slack for under-reporting boards
    assert captured["min_vram_mb"] == int(24 * 1024 * 0.92)
    by_id = {o.offer_id: o for o in out}
    assert by_id[2].gpu == "RTX 4090"
    assert by_id[2].vram_gb == 24
    assert by_id[4].gpu == "A100 SXM 40GB"
    assert by_id[5].gpu == "A100 SXM"
    assert by_id[17].gpu == "RTX 5090"


def test_usable_offers_always_datacenter_only(monkeypatch):
    """Community/marketplace hosts (hosting_type 0) are ALWAYS rejected — run secrets ship to the
    box, so even a verified community host (id=10) never makes the cut."""
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(vast_api, "search_offers", lambda *a, **k: list(FIXTURE))
    out = vast.usable_offers(24, disk_gb=60)
    # the community host (id=10) is dropped; only verified datacenter hosts survive
    assert [o.offer_id for o in out] == [1, 2, 17, 4, 5]


def test_usable_offers_vram_gate(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setattr(vast_api, "search_offers", lambda *a, **k: list(FIXTURE))
    out = vast.usable_offers(32, disk_gb=60)
    # every 24 GB class drops; only the active 32/40/80 GB survivors remain (dph ascending)
    assert [o.offer_id for o in out] == [17, 4, 5]


def test_usable_offers_exclude_machines(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    rows = [_offer(id=1, machine_id=7, dph_total=0.25), _offer(id=2, machine_id=8, dph_total=0.30)]
    monkeypatch.setattr(vast_api, "search_offers", lambda *a, **k: rows)
    assert [o.offer_id for o in vast.usable_offers(24, 60, exclude_machine_ids={7})] == [2]


def test_usable_offers_search_page_spans_all_classes(monkeypatch):
    # the price-sorted search page must be wide enough to span EVERY managed class (callers bucket
    # by class); the old limit=64 let a flood of one cheap class hide a larger fitting class with
    # usable offers just past the page.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    captured = {}

    def fake_search(
        min_vram_mb,
        *,
        min_disk_gb=0,
        min_reliability=0.95,
        min_duration_seconds=0,
        limit=64,
        num_gpus=1,
        extra_q=None,
    ):
        captured["limit"] = limit
        captured["min_duration_seconds"] = min_duration_seconds
        return []

    monkeypatch.setattr(vast_api, "search_offers", fake_search)
    vast.usable_offers(24, disk_gb=60)
    assert captured["limit"] >= 256  # wide default page (was 64)
    assert captured["min_duration_seconds"] == 0  # no deadline -> duration filter off
    vast.usable_offers(24, disk_gb=60, limit=512)
    assert captured["limit"] == 512  # explicit override honored


def test_usable_offers_exact_h100_threads_name_and_vram_filters(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    captured = {}

    def fake_search(min_vram_mb, **kwargs):
        captured.update(min_vram_mb=min_vram_mb, **kwargs)
        return []

    monkeypatch.setattr(vast_api, "search_offers", fake_search)
    vast.usable_offers(24, disk_gb=60, gpu_type="H100")

    assert captured["min_vram_mb"] == int(80 * 1024 * vast._SEARCH_VRAM_SLACK)
    assert captured["max_vram_mb"] == 80 * 1024
    assert captured["gpu_names"] == ("H100 SXM", "H100 PCIE")


def test_usable_offers_exact_40gb_bounds_shared_name_server_side(monkeypatch):
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    captured = {}

    def fake_search(min_vram_mb, **kwargs):
        captured.update(min_vram_mb=min_vram_mb, **kwargs)
        return []

    monkeypatch.setattr(vast_api, "search_offers", fake_search)
    vast.usable_offers(24, disk_gb=60, gpu_type="A100 SXM 40GB")

    assert captured["min_vram_mb"] == int(40 * 1024 * vast._SEARCH_VRAM_SLACK)
    assert captured["max_vram_mb"] == 40 * 1024
    # Exact A100 SXM 40GB drops the "A100 PCIE" capacity alias: a PCIe board canonicalizes to the distinct
    # "A100 PCIe" class, so seeding it would rent a board the box's device attestation (verify_gpu) then
    # rejects. The alias is still honored for NON-exact capacity capture via vast_gpu_for_offer (above).
    assert captured["gpu_names"] == ("A100 SXM4",)

    captured.clear()
    vast.usable_offers(24, disk_gb=60)
    assert "max_vram_mb" not in captured


def test_usable_offers_threads_duration_floor(monkeypatch):
    # when a wall cap is supplied, usable_offers does not extend it with provisioning grace.
    # a zero wall keeps the duration filter disabled.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    captured = {}

    def fake_search(min_vram_mb, *, min_duration_seconds=0, num_gpus=1, **k):
        captured["min_duration_seconds"] = min_duration_seconds
        return []

    monkeypatch.setattr(vast_api, "search_offers", fake_search)
    vast.usable_offers(24, disk_gb=60, max_wall_seconds=7200.0)
    assert captured["min_duration_seconds"] == 7200.0
    # a sub-60s wall is floored at the minimum provider creation allowance.
    vast.usable_offers(24, disk_gb=60, max_wall_seconds=30.0)
    assert captured["min_duration_seconds"] == 60.0
    vast.usable_offers(24, disk_gb=60)  # no wall cap -> filter stays off
    assert captured["min_duration_seconds"] == 0


def test_rent_search_outlasts_the_boxs_deadline_not_just_the_wall_grant(monkeypatch):
    # an unarmed workload profile's box deadline carries a provisioning allowance on top of its work
    # budget, but gpu.max_wall_seconds still names the work budget alone. Searching on the grant
    # accepts a host whose remaining duration outlasts 600s of work yet expires part-way through the
    # boot the allowance exists to survive -- the box then dies mid-provisioning on a host that was
    # never rentable for the window it was handed.
    from flash.providers.vast import jobs as vast

    captured = {}

    def fake_offers(*a, max_wall_seconds=0, **k):
        captured["max_wall_seconds"] = max_wall_seconds
        raise vast.vast_api.VastApiError("stop before renting")

    spec = _wall_capped_spec(600.0)
    monkeypatch.setattr(vast, "usable_offers", fake_offers)
    # the box holds a 600s work budget plus a 1200s provisioning allowance.
    now = 1_800_000_000.0
    monkeypatch.setattr(vast.time, "time", lambda: now)
    with pytest.raises(vast.vast_api.VastApiError):
        vast.submit_attempt_vast(spec, deadline_at=now + 1800.0)
    assert captured["max_wall_seconds"] == 1800.0, (
        "offer search must require a host that outlasts the deadline the box enforces"
    )


def test_rent_search_never_shortens_below_the_granted_wall(monkeypatch):
    # the ordinary case: the deadline sits exactly one wall grant out, so the floor is unchanged.
    # a deadline already inside the grant (a late retry) must not lower the duration requirement
    # below the wall the worker is still allowed to use.
    from flash.providers.vast import jobs as vast

    captured = {}

    def fake_offers(*a, max_wall_seconds=0, **k):
        captured["max_wall_seconds"] = max_wall_seconds
        raise vast.vast_api.VastApiError("stop before renting")

    spec = _wall_capped_spec(600.0)
    monkeypatch.setattr(vast, "usable_offers", fake_offers)
    now = 1_800_000_000.0
    monkeypatch.setattr(vast.time, "time", lambda: now)
    with pytest.raises(vast.vast_api.VastApiError):
        vast.submit_attempt_vast(spec, deadline_at=now + 600.0)
    assert captured["max_wall_seconds"] == 600.0
    with pytest.raises(vast.vast_api.VastApiError):
        vast.submit_attempt_vast(spec, deadline_at=now + 120.0)
    assert captured["max_wall_seconds"] == 600.0


def test_live_rates_gates_on_min_disk(monkeypatch):
    # live pricing must gate on MIN_DISK_GB (what create() enforces), not disk_gb=0 — otherwise it
    # prices off "cheapest" offers that aren't actually provisionable.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import pricing

    captured = {}

    def fake_usable(min_vram_gb, disk_gb, *a, **k):
        captured["disk_gb"] = disk_gb
        return []

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    pricing.live_rates(refresh=True)
    assert captured["disk_gb"] == vast.MIN_DISK_GB


def test_live_rates_floors_query_at_smallest_managed_vram(monkeypatch):
    # live pricing must floor the market query at the SMALLEST managed Vast class's VRAM, not 0 —
    # min_vram_gb=0 lets tiny UNMANAGED low-VRAM offers fill the fixed-size price-sorted page and
    # crowd managed classes off it, so hourly_rate() falls back to static rates even when live
    # offers exist. The floor keeps it to one market query while making the page relevant.
    from flash.providers.core.base import GPU_INFO
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import pricing

    captured = {}

    def fake_usable(min_vram_gb, disk_gb, *a, **k):
        captured["min_vram_gb"] = min_vram_gb
        return []

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    pricing.live_rates(refresh=True)
    expected = int(min(i.vram_gb for i in GPU_INFO.values() if i.vast_name))
    assert captured["min_vram_gb"] == expected
    assert captured["min_vram_gb"] > 0  # not the old crowding-prone 0


def test_live_rates_threads_wall_cap_and_bypasses_cache(monkeypatch):
    # a duration-bound estimate must price against offers that OUTLAST the run — thread
    # max_wall_seconds into usable_offers (the same duration floor the allocator/submit use), so a
    # cheap short-lived offer that the launch-time filter rejects can't set the rate. Duration-bound
    # queries must also NOT pollute the shared duration-agnostic cache (the `flash gpus` path).
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import pricing

    captured = {}

    def fake_usable(min_vram_gb, disk_gb, *a, max_wall_seconds=0, **k):
        captured["max_wall_seconds"] = max_wall_seconds
        return []

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    monkeypatch.setattr(pricing, "_rates_cache", {"ts": 0.0, "data": None})  # isolate
    pricing.hourly_rate("RTX 4090", max_wall_seconds=7200.0)
    assert captured["max_wall_seconds"] == 7200.0
    # the duration-bound query did NOT write the shared cache (would serve a narrowed set to `flash gpus`)
    assert pricing._rates_cache["data"] is None


def test_live_rates_caches_within_ttl_and_refresh_bypasses(monkeypatch):
    # repeated live_rates() within the TTL must share ONE market fetch (the refresh param was
    # previously ignored); refresh=True forces a fresh query.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import pricing

    calls = {"n": 0}

    def fake_usable(min_vram_gb, disk_gb, *a, **k):
        calls["n"] += 1
        return []

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    monkeypatch.setattr(
        pricing, "_rates_cache", {"ts": 0.0, "data": None}
    )  # isolate from other tests
    pricing.live_rates()  # first call -> one fetch
    pricing.live_rates()  # within TTL -> served from cache, no new fetch
    assert calls["n"] == 1
    pricing.live_rates(refresh=True)  # forced -> fetches again
    assert calls["n"] == 2


def test_usable_offers_threads_and_rechecks_card_count(monkeypatch):
    """The count reaches the search AND is re-checked on the rows that come back.

    Vast bakes the card count into the offer (create_instance takes no count), so num_gpus is the
    only way to reach a multi-card box. It is load-bearing twice: it sizes the rented box, and it
    divides dph_total into the per-card rate the allocator ranks on. A server that ignored the
    filter would otherwise hand back single-card rows that get priced as if they were multi-card.
    """
    from flash.providers.vast import jobs as vast
    from flash.providers.vast.client import api as vast_api

    captured = {}

    def fake_search(min_vram_mb, **kwargs):
        captured.update(kwargs)
        # honour the filter for id=1, ignore it for id=2 (a server that lies about the count)
        return [
            _offer(id=1, num_gpus=int(kwargs.get("num_gpus", 1)), dph_total=0.25),
            _offer(id=2, num_gpus=1, dph_total=0.20),
        ]

    monkeypatch.setattr(vast_api, "search_offers", fake_search)
    out = vast.usable_offers(24, disk_gb=60, num_gpus=4)

    assert captured["num_gpus"] == 4, "the count never reached the search"
    # id=2 is cheaper and would sort first, so dropping it proves the client-side re-check fired
    # rather than the ordering happening to hide it.
    assert [o.offer_id for o in out] == [1], "a wrong-count row survived the client-side re-check"
