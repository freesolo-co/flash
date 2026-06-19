"""RunPod allocation: VRAM sizing, cheapest-wins ranking, pins, gates, fallbacks."""

from __future__ import annotations

import pytest


def test_required_vram_catalog_and_open(monkeypatch):
    from autoslm.engine import vram
    from autoslm.providers import allocator
    from autoslm.providers.allocator import required_vram_gb

    # MEASURED: tiny-model GRPO OOMs a 20 GB card (vLLM-colocate engine overhead the param
    # estimate missed); floored to the 24 GB vLLM-colocate minimum (_VLLM_COLOCATE_FLOOR_GB).
    assert required_vram_gb("Qwen/Qwen3.5-0.8B", "grpo") == 24
    assert required_vram_gb("Qwen/Qwen3.5-4B", "sft") == 17  # matrix: 4B SFT seq1k down-routes (rank32 default)
    # open model: sized for GRPO (the heavier phase of the usual SFT+GRPO run) + headroom
    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda m: 4.0)
    est = vram.estimate_vram_gb(4.0, "grpo")

    # Default GRPO (no [train].max_length) sizes at the run's REAL engine length, mirroring
    # run_rl()'s max(1024, rl.max_prompt_len + completion) = 2048 + 320 = 2368 tokens (NOT a flat
    # 1024). At 2368 the 4.7B param estimate is ~31.8 GB raw -> 35 GB with headroom, so a 32 GB card
    # is no longer a safe fit and the allocator escalates to the cheapest validated >=35 GB class.
    a = allocator.allocate("Qwen/Qwen3.5-4B", "grpo")
    assert a.min_vram_gb == 35
    assert all(c.vram_gb >= 35 for c in a.candidates)


def test_pinned_undersized_gpu_escalates_never_below_need(monkeypatch):
    """An undersized / unavailable pin must NOT hard-fail: it escalates to the cheapest FITTING
    class across providers (the pin is a preference, not a hard cap). The matrix ``need`` stays the
    absolute floor (escalation never drops below it -> no OOM). Only a need that exceeds EVERY GPU
    across EVERY provider raises (all options exhausted)."""
    from autoslm.providers import allocator
    from autoslm.providers.base import UnsupportedGpuError

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    # Qwen3-4B GRPO at the default engine length (2368 tokens, mirroring run_rl) needs ~35 GB;
    # pinning a 24 GB RTX 4090 escalates UP to the cheapest fitting class (never provisions an
    # undersized OOM card, never raises).
    a = allocator.allocate("Qwen/Qwen3.5-4B", "grpo", gpu="RTX 4090")
    assert a.min_vram_gb == 35  # the model requirement floor is preserved
    assert a.gpu != "RTX 4090"  # escalated off the undersized pin to a fitting class
    # a pin that already fits the 35 GB floor is honored as-is
    a = allocator.allocate("Qwen/Qwen3.5-4B", "grpo", gpu="A100 PCIe")
    assert a.gpu == "A100 PCIe"
    assert a.min_vram_gb == 35
    # a requirement larger than every available GPU still raises (genuinely unsatisfiable)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 4096)
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate("Qwen/Qwen3.5-4B", "grpo", gpu="RTX 4090")


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


def test_estimator_matches_measured_seq_boundaries():
    """The raw VRAM physics reproduces the MEASURED RunPod capacity sweep: each anchor
    is a real train/OOM boundary observed on a pinned card (the calibration ground truth).
    estimate_vram_gb is the accurate estimate; model_required adds the safety headroom."""
    from autoslm.engine.vram import estimate_vram_gb as e

    # 0.8B (0.9B): seq up to 32k fits the cheapest 24 GB card, both algos (measured: A5000)
    assert e(0.9, "grpo", seq_len=4096) <= 24
    assert e(0.9, "grpo", seq_len=32768) <= 24
    assert e(0.9, "sft", seq_len=32768) <= 24
    # 4B (4.7B) SFT: seq 32k fits a 32 GB card (measured: 5090)
    assert e(4.7, "sft", seq_len=32768) <= 32
    # 4B GRPO: default + seq 16k fit a 32 GB card, seq 32k steps OVER it (measured boundary)
    assert e(4.7, "grpo", seq_len=1024, max_tokens=256) <= 32
    assert e(4.7, "grpo", seq_len=16384, max_tokens=4096) <= 32
    assert e(4.7, "grpo", seq_len=32768, max_tokens=8192) > 32
    # 9B GRPO is weights-driven: exceeds a 48 GB card -> the 80 GB tier (measured: A100)
    assert e(9.7, "grpo", seq_len=1024) > 48
    # a 36B model in bf16 never fits 32 GB (the 4-bit MoE is floored in model_required)
    assert e(36.0, "sft", seq_len=4096) > 32


def test_required_vram_policy_floors_and_downrouting():
    """model_required_vram_gb: hard floors never under-provision; small runs down-route to
    a cheaper card; bigger context/group/thinking only ever size UP (never down)."""
    from autoslm.engine.vram import model_required_vram_gb as need

    m4 = "Qwen/Qwen3.5-4B"
    # context length lifts GRPO need monotonically
    short = need(m4, "grpo", train={"max_length": 1024, "max_tokens": 256})
    long = need(m4, "grpo", train={"max_length": 16384, "max_tokens": 4096})
    assert long > short
    # sub-1B GRPO fits a 24 GB card; 2B GRPO OOMs 24 (MEASURED) -> needs the 32 tier; small SFT
    # drops below the catalog default (32)
    assert need("Qwen/Qwen3.5-0.8B", "grpo") <= 24
    assert 24 < need("Qwen/Qwen3.5-2B", "grpo") <= 32
    assert need(m4, "sft", train={"max_length": 1024, "lora_rank": 8}) < 32
    # 9B GRPO is 4-bit QLoRA -> fits a 24-32GB card (was 80 GB bf16).
    assert need("Qwen/Qwen3.5-9B", "grpo") <= 32  # QLoRA 4-bit: fits a ~24-32GB card
    assert need("Qwen/Qwen3.5-9B", "grpo", train={"max_length": 8192, "max_tokens": 2048, "group_size": 8}) <= 48  # 4-bit
    assert need("Qwen/Qwen3.5-9B", "grpo", train={"max_length": 4096, "group_size": 8}) <= 48
    # group size and thinking never DECREASE the requirement
    base = need(m4, "grpo", train={"max_length": 4096, "max_tokens": 1024, "group_size": 4})
    assert need(m4, "grpo", train={"max_length": 4096, "max_tokens": 1024, "group_size": 16}) >= base
    assert need(m4, "grpo", train={"max_length": 4096, "max_tokens": 1024, "group_size": 4}, thinking=True) >= base
    # max_tokens (completion length) lifts the fp32-logits term -> a longer completion never sizes
    # DOWN, and a much longer one sizes UP (the term the estimator previously ignored).
    short_c = need(m4, "grpo", train={"max_length": 8192, "max_tokens": 256, "group_size": 8})
    long_c = need(m4, "grpo", train={"max_length": 8192, "max_tokens": 8192, "group_size": 8})
    assert long_c >= short_c


def test_estimator_logits_term_uses_max_tokens_and_caps_at_budget():
    """The GRPO estimate must include the fp32-logits term (it scales with max_tokens, NOT
    seq_len) and cap it at the per-device logits budget so it never over-reserves."""
    from autoslm.engine import vram
    from autoslm.engine.vram import estimate_vram_gb as e

    # Holding seq fixed, a longer completion (max_tokens) raises the GRPO train-phase estimate.
    # Use the non-colocate path (use_vllm=False) so the rollout term doesn't mask the train peak
    # (for small vLLM models the 2nd-copy rollout dominates and hides the logits term).
    lo = e(2.0, "grpo", seq_len=4096, max_tokens=256, group_size=8, use_vllm=False)
    hi = e(2.0, "grpo", seq_len=4096, max_tokens=8192, group_size=8, use_vllm=False)
    assert hi > lo
    # ... but never by more than the logits budget (the per_device=1 floor is capped there)
    assert hi - lo <= vram._LOGITS_BUDGET_GB + 1e-6
    # SFT path ignores max_tokens entirely (no logits-over-completion term there)
    assert e(2.0, "sft", seq_len=4096, max_tokens=256) == e(2.0, "sft", seq_len=4096, max_tokens=8192)


def test_vram_headroom_consistent_across_sizing_paths():
    """resolve_gpu_policy (parse-time) and required_vram_gb (submit-time) must size with the SAME
    headroom (a validated constant), so they never disagree (PR #176 review)."""
    from autoslm.providers import allocator

    assert allocator.vram_headroom() == 1.1
    # both paths feed model_required_vram_gb the same headroom -> identical sizing
    a_need = allocator.required_vram_gb("Qwen/Qwen3.5-4B", "grpo", train={"max_length": 4096})
    from autoslm.engine.vram import model_required_vram_gb

    direct = model_required_vram_gb("Qwen/Qwen3.5-4B", "grpo", train={"max_length": 4096}, headroom=1.1)
    assert a_need == direct


def test_allocate_never_selects_below_matrix_need(monkeypatch):
    """The core anti-OOM invariant: the GPU the allocator picks ALWAYS has >= the matrix's
    required VRAM, across a sweep of model x algo x seq x group x batch. If this ever fails,
    auto-allocation could provision a too-small card and OOM a paid worker."""
    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    from autoslm.providers.allocator import allocate, required_vram_gb
    from autoslm.providers.base import get_gpu_info

    grid = [
        ("Qwen/Qwen3.5-0.8B", "grpo", {"max_length": 1024, "group_size": 4}),
        ("Qwen/Qwen3.5-0.8B", "grpo", {"max_length": 32768, "group_size": 16}),
        ("Qwen/Qwen3.5-2B", "sft", {"max_length": 8192}),
        ("Qwen/Qwen3.5-4B", "grpo", {"max_length": 1024, "group_size": 4}),
        ("Qwen/Qwen3.5-4B", "grpo", {"max_length": 16384, "max_tokens": 4096, "group_size": 8}),
        ("Qwen/Qwen3.5-4B", "sft", {"max_length": 32768}),
        ("Qwen/Qwen3.5-9B", "grpo", {"max_length": 8192, "group_size": 8}),
    ]
    for model, algo, tr in grid:
        need = required_vram_gb(model, algo, train=tr)
        alloc = allocate(model, algo, provider="runpod", allow_unvalidated=True, train=tr)
        assert get_gpu_info(alloc.gpu).vram_gb >= need, (model, algo, tr, alloc.gpu, need)
