"""RunPod allocation: VRAM sizing, cheapest-wins ranking, pins, gates, fallbacks."""

from __future__ import annotations

import pytest


def test_required_vram_catalog_and_open(monkeypatch):
    from autoslm.engine import vram
    from autoslm.providers.allocator import required_vram_gb

    assert required_vram_gb("Qwen/Qwen3.5-0.8B", "grpo") == 12
    assert required_vram_gb("Qwen/Qwen3.5-4B", "sft") == 32
    # open model: sized for GRPO (the heavier phase of the usual SFT+GRPO run) + headroom
    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda m: 4.0)
    est = vram.estimate_vram_gb(4.0, "grpo")
    import math

    assert required_vram_gb("org/some-4b", "sft") == math.ceil(est * 1.15)
    # unknown size -> the 24 GB tier, like resolve_gpu_policy
    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda m: None)
    assert required_vram_gb("org/mystery", "grpo") == 24


def test_cheapest_runpod_class(monkeypatch):
    from autoslm.providers import allocator

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")  # static, deterministic runpod rates
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    # cheapest validated runpod 24GB class
    assert (a.provider, a.gpu) == ("runpod", "RTX A5000")
    assert "runpod" in allocator.allocation_summary(a)


def test_vram_gating_excludes_small_cards(monkeypatch):
    from autoslm.providers import allocator

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    a = allocator.allocate("Qwen/Qwen3.5-4B", "grpo")  # needs 32 GB
    assert a.min_vram_gb == 32
    assert all(c.vram_gb >= 32 for c in a.candidates)
    assert a.gpu == "RTX 5090"  # cheapest validated >=32GB runpod class


def test_pinned_undersized_gpu_rejected(monkeypatch):
    from autoslm.providers import allocator
    from autoslm.providers.base import UnsupportedGpuError

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    # Qwen3-8B needs 32 GB; pinning a 24 GB card must reject up front (the model
    # requirement is the floor regardless of the pin), not provision a 24 GB OOM.
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate("Qwen/Qwen3.5-4B", "grpo", gpu="RTX 4090")
    # a pin that DOES fit still works
    a = allocator.allocate("Qwen/Qwen3.5-4B", "grpo", gpu="RTX 5090")
    assert a.gpu == "RTX 5090"
    assert a.min_vram_gb == 32  # model requirement, not the pinned card's own VRAM


def test_validation_gate_and_opt_in(monkeypatch):
    from autoslm.providers import allocator

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    monkeypatch.delenv("AUTOSLM_GPU_ALLOW_UNVALIDATED", raising=False)
    # L4 is validated nowhere -> excluded by default...
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert (a.provider, a.gpu) == ("runpod", "RTX A5000")
    # ...but the explicit opt-in admits cheaper unvalidated classes
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo", allow_unvalidated=True)
    assert a.provider == "runpod"
    assert a.hourly_usd <= 0.27  # an unvalidated class is at least as cheap as the A5000


def test_provider_pin_runpod(monkeypatch):
    from autoslm.providers import allocator

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo", provider="runpod")
    assert a.provider == "runpod"
    assert all(c.provider == "runpod" for c in a.candidates)


def test_unknown_provider_rejected(monkeypatch):
    from autoslm.providers import allocator
    from autoslm.providers.base import UnsupportedGpuError

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    # A name not in PROVIDER_NAMES is an "unknown provider".
    with pytest.raises(UnsupportedGpuError, match="unknown provider"):
        allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo", provider="lambda")


def test_known_provider_unavailable_offline(monkeypatch):
    from autoslm.providers import allocator
    from autoslm.providers.base import UnsupportedGpuError

    # vast is a known provider but offline (AUTOSLM_SKIP_NET) it is not available.
    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    with pytest.raises(UnsupportedGpuError, match="not available"):
        allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo", provider="vast")


def test_skip_net_matches_static_cheapest(monkeypatch):
    from autoslm.providers import allocator
    from autoslm.providers.base import cheapest_gpu

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")  # real available_providers: runpod only
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert a.provider == "runpod"
    assert a.gpu == cheapest_gpu(24)


def test_nothing_fits_names_constraint(monkeypatch):
    from autoslm.providers import allocator
    from autoslm.providers.base import UnsupportedGpuError

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 4096)
    with pytest.raises(UnsupportedGpuError, match="4096 GB"):
        allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
