"""usable_offers: every quality/safety filter, alias mapping, and ordering (mocked API)."""

from __future__ import annotations


def _offer(**kw) -> dict:
    """A fully-passing verified-datacenter RTX 3090 offer; override fields per case."""
    base = {
        "id": 1000,
        "machine_id": 1,
        "gpu_name": "RTX 3090",
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
    _offer(id=1, dph_total=0.25),  # KEEP: cheapest valid 3090
    _offer(id=2, gpu_name="L4", gpu_ram=23034, dph_total=0.34),  # KEEP: under-reported VRAM ok
    _offer(id=3, gpu_name="RTX 6000Ada", gpu_ram=49140, dph_total=0.80),  # KEEP: alias mapping
    _offer(id=4, gpu_name="A100 SXM4", gpu_ram=40960, dph_total=0.87),  # KEEP: 40GB variant
    _offer(id=5, gpu_name="A100 SXM4", gpu_ram=81920, dph_total=1.20),  # KEEP: 80GB variant
    _offer(id=10, dph_total=0.08, hosting_type=0),  # dropped: community host (secrets ship to it)
    _offer(id=11, dph_total=0.10, verification="unverified"),  # DROP: not verified
    _offer(id=12, dph_total=0.11, reliability2=0.80),  # DROP: reliability floor
    _offer(id=13, gpu_name="Tesla T4", gpu_ram=16384, dph_total=0.05),  # DROP: sub-Ampere
    _offer(id=14, dph_total=0.12, disk_space=20.0),  # DROP: too little disk
    _offer(id=15, dph_total=0.13, inet_down=50.0),  # DROP: too slow to pull weights
    _offer(id=16, gpu_name="RTX PRO 4000", gpu_ram=24467, dph_total=0.20, cuda_max_good=12.4),
    # ^ DROP: Blackwell class on a pre-CUDA-13 driver (PTX JIT would fail)
    _offer(id=17, gpu_name="RTX PRO 4000", gpu_ram=24467, dph_total=0.35, cuda_max_good=13.1),
    # ^ KEEP: Blackwell with a CUDA-13 driver
]


def test_usable_offers_filters_and_order(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    captured = {}

    def fake_search(
        min_vram_mb, *, min_disk_gb=0, min_reliability=0.95, num_gpus=1, limit=64, extra_q=None
    ):
        captured["min_vram_mb"] = min_vram_mb
        captured["min_disk_gb"] = min_disk_gb
        captured["num_gpus"] = num_gpus
        return list(FIXTURE)

    monkeypatch.setattr(vast_api, "search_offers", fake_search)
    out = vast.usable_offers(24, disk_gb=60)
    # community host (id=10) dropped — datacenter-only (run secrets ship to the box)
    assert [o.offer_id for o in out] == [1, 2, 17, 3, 4, 5]  # dph ascending, junk gone
    assert captured["min_disk_gb"] == 60
    # the server-side VRAM filter carries slack for under-reporting boards
    assert captured["min_vram_mb"] == int(24 * 1024 * 0.92)
    by_id = {o.offer_id: o for o in out}
    assert by_id[2].gpu == "L4"
    assert by_id[2].vram_gb == 24
    assert by_id[3].gpu == "RTX 6000 Ada"
    assert by_id[4].gpu == "A100 SXM 40GB"
    assert by_id[5].gpu == "A100 SXM"
    assert by_id[17].gpu == "RTX Pro 4000"


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
    # every 24 GB class drops; only the 40/48/80 GB survivors remain
    assert [o.offer_id for o in out] == [3, 4, 5]


def test_usable_offers_exclude_machines(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    rows = [_offer(id=1, machine_id=7, dph_total=0.25), _offer(id=2, machine_id=8, dph_total=0.30)]
    monkeypatch.setattr(vast_api, "search_offers", lambda *a, **k: rows)
    assert [o.offer_id for o in vast.usable_offers(24, 60, exclude_machine_ids={7})] == [2]


def test_usable_offers_threads_num_gpus_into_search(monkeypatch):
    """A multi-GPU (disaggregated) request must reach search_offers as an exact GPU count, so a
    2-GPU run actually searches 2-GPU machines instead of silently searching 1-GPU offers (which
    would masquerade as 'no capacity'). Default stays 1 for single-GPU runs."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    captured = {}

    def fake_search(min_vram_mb, *, min_disk_gb=0, min_reliability=0.95, num_gpus=1, **k):
        captured["num_gpus"] = num_gpus
        return []

    monkeypatch.setattr(vast_api, "search_offers", fake_search)
    vast.usable_offers(24, disk_gb=60, num_gpus=3)
    assert captured["num_gpus"] == 3
    # default is single-GPU
    vast.usable_offers(24, disk_gb=60)
    assert captured["num_gpus"] == 1


def test_usable_offers_populates_num_gpus_from_row(monkeypatch):
    """VastOffer.num_gpus is read from the Vast offer row so multi-GPU offers are distinguishable
    from single-GPU ones downstream (cost reporting / disaggregated plumbing); absent -> 1."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    rows = [
        _offer(id=1, machine_id=1, dph_total=0.25, num_gpus=2),  # multi-GPU row
        _offer(id=2, machine_id=2, dph_total=0.30),  # no num_gpus key -> defaults to 1
    ]
    monkeypatch.setattr(vast_api, "search_offers", lambda *a, **k: rows)
    by_id = {o.offer_id: o for o in vast.usable_offers(24, 60)}
    assert by_id[1].num_gpus == 2
    assert by_id[2].num_gpus == 1
