"""Cross-provider allocation: VRAM sizing, cheapest-wins ranking, pins, gates, fallbacks."""

from __future__ import annotations

import dataclasses

import pytest

from tests.test_vast_offers import _offer


def _vast_validated(monkeypatch, *names):
    """Mark classes as vast-validated for the duration of a test."""
    from autoslm.flash import gpus

    for name in names:
        info = gpus.GPU_INFO[name]
        monkeypatch.setitem(
            gpus.GPU_INFO,
            name,
            dataclasses.replace(info, validated_on=(*info.validated_on, "vast")),
        )


def _wire(monkeypatch, offers, providers=("runpod", "vast")):
    """Static runpod rates + a fake vast offer book."""
    from autoslm.providers import allocator, vast, vast_api

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")  # static, deterministic runpod rates
    monkeypatch.setattr(allocator, "available_providers", lambda: tuple(providers))
    monkeypatch.setattr(vast_api, "search_offers", lambda *a, **k: list(offers))
    return allocator, vast


def test_required_vram_catalog_and_open(monkeypatch):
    from autoslm.engine import vram
    from autoslm.providers.allocator import required_vram_gb

    assert required_vram_gb("Qwen/Qwen3-4B-Instruct-2507", "grpo") == 24
    assert required_vram_gb("Qwen/Qwen3-8B", "sft") == 32
    # open model: sized for GRPO (the heavier phase of the usual SFT+GRPO run) + headroom
    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda m: 4.0)
    est = vram.estimate_vram_gb(4.0, "grpo")
    import math

    assert required_vram_gb("org/some-4b", "sft") == math.ceil(est * 1.15)
    # unknown size -> the 24 GB tier, like resolve_gpu_policy
    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda m: None)
    assert required_vram_gb("org/mystery", "grpo") == 24


def test_cheapest_across_providers(monkeypatch):
    allocator, _ = _wire(monkeypatch, [_offer(id=1, dph_total=0.25)])
    _vast_validated(monkeypatch, "RTX 3090")
    # vast 3090 $0.25 beats the cheapest validated runpod 24GB class (A5000 $0.27)
    a = allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo")
    assert (a.provider, a.gpu) == ("vast", "RTX 3090")
    assert a.hourly_usd == 0.25
    assert a.offer is not None
    assert a.offer.offer_id == 1
    # runpod candidates still ranked behind it for failover
    assert any(c.provider == "runpod" for c in a.candidates)
    assert "vast" in allocator.allocation_summary(a)


def test_runpod_wins_when_vast_pricier(monkeypatch):
    allocator, _ = _wire(monkeypatch, [_offer(id=1, dph_total=0.55)])
    _vast_validated(monkeypatch, "RTX 3090")
    a = allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo")
    assert (a.provider, a.gpu) == ("runpod", "RTX A5000")  # $0.27 static


def test_vram_gating_excludes_small_cards(monkeypatch):
    allocator, _ = _wire(monkeypatch, [])
    a = allocator.allocate("Qwen/Qwen3-8B", "grpo")  # needs 32 GB
    assert a.min_vram_gb == 32
    assert all(c.vram_gb >= 32 for c in a.candidates)
    assert a.gpu == "RTX 5090"  # cheapest validated >=32GB runpod class


def test_pinned_undersized_gpu_rejected(monkeypatch):
    from autoslm.flash.gpus import UnsupportedGpuError

    allocator, _ = _wire(monkeypatch, [])
    # Qwen3-8B needs 32 GB; pinning a 24 GB card must reject up front (the model
    # requirement is the floor regardless of the pin), not provision a 24 GB OOM.
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate("Qwen/Qwen3-8B", "grpo", gpu="RTX 4090")
    # a pin that DOES fit still works
    a = allocator.allocate("Qwen/Qwen3-8B", "grpo", gpu="RTX 5090")
    assert a.gpu == "RTX 5090"
    assert a.min_vram_gb == 32  # model requirement, not the pinned card's own VRAM


def test_validation_gate_and_opt_in(monkeypatch):
    allocator, _ = _wire(monkeypatch, [_offer(id=1, gpu_name="L4", gpu_ram=23034, dph_total=0.20)])
    monkeypatch.delenv("AUTOSLM_GPU_ALLOW_UNVALIDATED", raising=False)
    # L4 is validated nowhere -> excluded by default...
    a = allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo")
    assert (a.provider, a.gpu) == ("runpod", "RTX A5000")
    # ...but the explicit opt-in admits it (and it's the cheapest)
    a = allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo", allow_unvalidated=True)
    assert (a.provider, a.gpu) == ("vast", "L4")


def test_provider_pins(monkeypatch):
    allocator, _ = _wire(monkeypatch, [_offer(id=1, dph_total=0.25)])
    _vast_validated(monkeypatch, "RTX 3090")
    a = allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo", provider="runpod")
    assert a.provider == "runpod"
    assert all(c.provider == "runpod" for c in a.candidates)
    a = allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo", provider="vast")
    assert a.provider == "vast"
    assert all(c.provider == "vast" for c in a.candidates)


def test_provider_vast_unavailable_fails_fast(monkeypatch):
    from autoslm.flash.gpus import UnsupportedGpuError

    allocator, _ = _wire(monkeypatch, [], providers=("runpod",))
    with pytest.raises(UnsupportedGpuError, match="VAST_API_KEY"):
        allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo", provider="vast")


def test_gpu_pin_picks_cheaper_provider(monkeypatch):
    # same class on both substrates: runpod static $0.46 vs vast offer $0.25
    allocator, _ = _wire(monkeypatch, [_offer(id=1, dph_total=0.25)])
    _vast_validated(monkeypatch, "RTX 3090")
    monkeypatch.setenv("AUTOSLM_GPU_ALLOW_UNVALIDATED", "1")  # 3090 isn't runpod-validated
    a = allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo", gpu="RTX 3090")
    assert (a.provider, a.gpu, a.hourly_usd) == ("vast", "RTX 3090", 0.25)
    assert {c.provider for c in a.candidates} == {"runpod", "vast"}


def test_vast_api_down_degrades_to_runpod(monkeypatch):
    from autoslm.providers import allocator, vast_api

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "vast"))

    def boom(*a, **k):
        raise vast_api.VastApiError("503")

    monkeypatch.setattr(vast_api, "search_offers", boom)
    a = allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo")
    assert a.provider == "runpod"
    from autoslm.flash.gpus import UnsupportedGpuError

    with pytest.raises(UnsupportedGpuError, match="offer search failed"):
        allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo", provider="vast")


def test_skip_net_matches_static_cheapest(monkeypatch):
    from autoslm.flash.gpus import cheapest_gpu
    from autoslm.providers import allocator

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")  # real available_providers: runpod only
    a = allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo")
    assert a.provider == "runpod"
    assert a.gpu == cheapest_gpu(24)


def test_nothing_fits_names_constraint(monkeypatch):
    from autoslm.flash.gpus import UnsupportedGpuError
    from autoslm.providers import allocator

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 4096)
    with pytest.raises(UnsupportedGpuError, match="4096 GB"):
        allocator.allocate("Qwen/Qwen3-4B-Instruct-2507", "grpo")
