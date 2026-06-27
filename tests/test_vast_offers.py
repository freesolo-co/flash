"""usable_offers: every quality/safety filter, alias mapping, and ordering (mocked API)."""

from __future__ import annotations


def _offer(**kw) -> dict:
    """A fully-passing verified-datacenter RTX 4090 offer; override fields per case."""
    base = {
        "id": 1000,
        "machine_id": 1,
        "gpu_name": "RTX 4090",
        "gpu_ram": 24576,
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
    _offer(id=3, gpu_name="RTX A6000", gpu_ram=49140, dph_total=0.80),  # KEEP: 48GB Ampere
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


def test_usable_offers_filters_and_order(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    captured = {}

    def fake_search(min_vram_mb, *, min_disk_gb=0, min_reliability=0.95, limit=64, extra_q=None):
        captured["min_vram_mb"] = min_vram_mb
        captured["min_disk_gb"] = min_disk_gb
        return list(FIXTURE)

    monkeypatch.setattr(vast_api, "search_offers", fake_search)
    out = vast.usable_offers(24, disk_gb=60)
    # community host (id=10) dropped — datacenter-only (run secrets ship to the box)
    assert [o.offer_id for o in out] == [1, 2, 17, 3, 4, 5]  # dph ascending, junk gone
    assert captured["min_disk_gb"] == 60
    # the server-side VRAM filter carries slack for under-reporting boards
    assert captured["min_vram_mb"] == int(24 * 1024 * 0.92)
    by_id = {o.offer_id: o for o in out}
    assert by_id[2].gpu == "RTX 4090"
    assert by_id[2].vram_gb == 24
    assert by_id[3].gpu == "RTX A6000"
    assert by_id[4].gpu == "A100 SXM 40GB"
    assert by_id[5].gpu == "A100 SXM"
    assert by_id[17].gpu == "RTX 5090"


def test_usable_offers_always_datacenter_only(monkeypatch):
    """Community/marketplace hosts (hosting_type 0) are ALWAYS rejected — run secrets ship to the
    box, so even a verified community host (id=10) never makes the cut."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    monkeypatch.setattr(vast_api, "search_offers", lambda *a, **k: list(FIXTURE))
    out = vast.usable_offers(24, disk_gb=60)
    # the community host (id=10) is dropped; only verified datacenter hosts survive
    assert [o.offer_id for o in out] == [1, 2, 17, 3, 4, 5]


def test_usable_offers_vram_gate(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    monkeypatch.setattr(vast_api, "search_offers", lambda *a, **k: list(FIXTURE))
    out = vast.usable_offers(32, disk_gb=60)
    # every 24 GB class drops; only the 32/40/48/80 GB survivors remain (dph ascending)
    assert [o.offer_id for o in out] == [17, 3, 4, 5]


def test_usable_offers_exclude_machines(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    rows = [_offer(id=1, machine_id=7, dph_total=0.25), _offer(id=2, machine_id=8, dph_total=0.30)]
    monkeypatch.setattr(vast_api, "search_offers", lambda *a, **k: rows)
    assert [o.offer_id for o in vast.usable_offers(24, 60, exclude_machine_ids={7})] == [2]


def test_usable_offers_search_page_spans_all_classes(monkeypatch):
    # Codex Mr5nO: the price-sorted search page must be wide enough to span EVERY managed class
    # (callers bucket by class); the old limit=64 let a flood of one cheap class hide a larger fitting
    # class with usable offers just past the page.
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    captured = {}

    def fake_search(min_vram_mb, *, min_disk_gb=0, min_reliability=0.95, limit=64, extra_q=None):
        captured["limit"] = limit
        return []

    monkeypatch.setattr(vast_api, "search_offers", fake_search)
    vast.usable_offers(24, disk_gb=60)
    assert captured["limit"] >= 256  # wide default page (was 64)
    vast.usable_offers(24, disk_gb=60, limit=512)
    assert captured["limit"] == 512  # explicit override honored


def test_live_rates_gates_on_min_disk(monkeypatch):
    # Codex Mr4re: live pricing must gate on MIN_DISK_GB (what create() enforces), not disk_gb=0 —
    # otherwise it prices off "cheapest" offers that aren't actually provisionable.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast import pricing

    captured = {}

    def fake_usable(min_vram_gb, disk_gb, *a, **k):
        captured["disk_gb"] = disk_gb
        return []

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    pricing.live_rates(refresh=True)
    assert captured["disk_gb"] == vast.MIN_DISK_GB


def test_live_rates_caches_within_ttl_and_refresh_bypasses(monkeypatch):
    # Copilot Msbs9: repeated live_rates() within the TTL must share ONE market fetch (the refresh
    # param was previously ignored); refresh=True forces a fresh query.
    from flash.providers.vast import jobs as vast
    from flash.providers.vast import pricing

    calls = {"n": 0}

    def fake_usable(min_vram_gb, disk_gb, *a, **k):
        calls["n"] += 1
        return []

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    monkeypatch.setattr(pricing, "_rates_cache", {"ts": 0.0, "data": None})  # isolate from other tests
    pricing.live_rates()  # first call -> one fetch
    pricing.live_rates()  # within TTL -> served from cache, no new fetch
    assert calls["n"] == 1
    pricing.live_rates(refresh=True)  # forced -> fetches again
    assert calls["n"] == 2
