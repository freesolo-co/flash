"""RunPod allocation: VRAM sizing, cheapest-wins ranking, fallbacks (no pins, no gates)."""

from __future__ import annotations

import pytest


def test_required_vram_catalog_and_open(monkeypatch):
    from flash.engine import vram
    from flash.providers import allocator
    from flash.providers.allocator import required_vram_gb

    # MEASURED: tiny-model GRPO OOMs a 20 GB card (vLLM-colocate engine overhead the param
    # estimate missed); floored to the 24 GB vLLM-colocate minimum (_VLLM_COLOCATE_FLOOR_GB).
    assert required_vram_gb("Qwen/Qwen3.5-0.8B", "grpo") == 24
    assert required_vram_gb("Qwen/Qwen3.5-4B", "sft") == 17  # matrix: 4B SFT seq1k down-routes (rank32 default)
    # open model: sized for GRPO (the heavier phase of the usual SFT+GRPO run) + headroom
    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda m, **k: 4.0)
    est = vram.estimate_vram_gb(4.0, "grpo")

    # Default GRPO (no [train].max_length) sizes at the run's REAL engine length, mirroring
    # run_rl()'s max(1024, rl.max_prompt_len + completion) = 2048 + 320 = 2368 tokens (NOT a flat
    # 1024). At 2368 the 4.7B param estimate is ~31.8 GB raw -> 35 GB with headroom, so a 32 GB card
    # is no longer a safe fit and the allocator escalates to the cheapest validated >=35 GB class.
    a = allocator.allocate("Qwen/Qwen3.5-4B", "grpo")
    assert a.min_vram_gb == 35
    assert all(c.vram_gb >= 35 for c in a.candidates)


def test_allocation_restricted_to_validated_pool(monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import VALIDATED

    # The deployed control plane rejects a submit for any non-validated class, so client-side
    # allocation must only ever pick a class in the live-validated pool — across ALL candidates,
    # not just the chosen one. Offline only RunPod is available; 0.8B GRPO needs the 24 GB tier
    # whose cheapest VALIDATED RunPod class is RTX 3090 @ $0.46 (the cheaper RTX A5000 @ $0.27 is
    # runpod_mig_risk and skipped on RunPod; cheaper unvalidated 24 GB classes like L4 are excluded
    # too). 16 GB unvalidated classes (RTX 2000 Ada @ $0.24) never appear.
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert a.provider == "runpod"
    assert all(c.gpu in VALIDATED for c in a.candidates), [
        c.gpu for c in a.candidates if c.gpu not in VALIDATED
    ]
    assert a.gpu == "RTX 3090"  # the cheapest VALIDATED non-MIG 24 GB RunPod class


def test_allocation_skips_cheaper_unvalidated_class(monkeypatch):
    """A small SFT fits a 16-17 GB card; the absolute-cheapest fitting class (RTX 2000 Ada /
    RTX A4000, ~$0.24-0.25) is UNVALIDATED and must be skipped for the cheapest VALIDATED one,
    so the run actually submits (the live `flash train` default-submit failure this fixes)."""
    from flash.providers import allocator
    from flash.providers.base import GPU_INFO, VALIDATED

    # 4B SFT (seq 1024, rank 8) down-routes below 24 GB in the matrix (see test_required_vram_*).
    a = allocator.allocate(
        "Qwen/Qwen3.5-4B", "sft", train={"max_length": 1024, "lora_rank": 8}
    )
    assert a.min_vram_gb < 24  # a sub-24 GB run where unvalidated cheap cards exist
    assert all(c.gpu in VALIDATED for c in a.candidates)
    # The cheapest fitting UNVALIDATED RunPod class would have been chosen without the gate.
    assert any(
        (not g.validated) and g.enum_member and g.vram_gb >= a.min_vram_gb
        for g in GPU_INFO.values()
    )


def test_allocate_excludes_gpu_class_walks_to_different_validated(monkeypatch):
    """A MIG / unusable-GPU retry (runtime safety net) must re-allocate OFF the failed CLASS to a
    DIFFERENT validated one. On RunPod the validated, non-MIG fitting pool for 0.8B GRPO (>=24 GB)
    ranks RTX 3090 ($0.46) < RTX A6000 ($0.49) < RTX 4090 ($0.69) < ... (RTX A5000 is already skipped
    up front as runpod_mig_risk); excluding the cheapest (RTX 3090) makes the allocator pick the
    next validated class, never re-pick the excluded one."""
    from flash.providers import allocator

    base = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert base.gpu == "RTX 3090"  # cheapest validated non-MIG 24 GB class

    walked = allocator.allocate(
        "Qwen/Qwen3.5-0.8B", "grpo", exclude_gpu_classes=frozenset({"RTX 3090"})
    )
    assert walked.gpu != "RTX 3090"  # walked off the excluded class
    # the excluded class is gone from EVERY candidate, not just the chosen one
    assert all(c.gpu != "RTX 3090" for c in walked.candidates)
    # ... onto the next-cheapest validated full GPU (a card RunPod can't MIG-slice)
    assert walked.gpu == "RTX A6000"


def test_runpod_allocation_never_returns_mig_substituted_types(monkeypatch):
    """REGRESSION (live MIG burn): RunPod serves some validated GPU *types* as Blackwell MIG slices
    (RTX A5000, RTX Pro 6000 WK), which crash training (PyTorch's CUDA allocator NVML-asserts on MIG
    partitions). The allocator must NEVER pick — or even rank — a runpod_mig_risk class on RunPod for
    a model that fits a cheaper full GPU. The 0.8B GRPO (24 GB) lands on RTX 3090 and the 9B GRPO
    (80 GB) on A100 PCIe — both full, non-MIG cards RunPod can't partition."""
    from flash.providers import allocator
    from flash.providers.base import GPU_INFO

    mig_types = {g.name for g in GPU_INFO.values() if g.runpod_mig_risk}
    assert mig_types == {"RTX A5000", "RTX Pro 6000 WK"}  # the two confirmed-substituted types

    a08 = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert a08.provider == "runpod"
    assert a08.gpu == "RTX 3090"  # cheapest validated, non-MIG, full 24 GB RunPod card
    a9 = allocator.allocate("Qwen/Qwen3.5-9B", "grpo")
    assert a9.provider == "runpod"
    assert a9.gpu == "A100 PCIe"  # cheapest validated, non-MIG, full 80 GB RunPod card
    # No MIG-substituted type appears in EITHER ranked candidate list (not just the chosen one).
    for a in (a08, a9):
        assert not (mig_types & {c.gpu for c in a.candidates}), [
            c.gpu for c in a.candidates if c.gpu in mig_types
        ]


def test_runpod_mig_risk_types_stay_validated_and_submittable(monkeypatch):
    """The exclusion is allocator-only: the MIG-risk types must STILL be in the validated pool so
    the deployed control plane's validation gate keeps accepting them as submittable classes (we
    change which class the ALLOCATOR picks for RunPod, not the schema's allowlist)."""
    from flash.providers.base import GPU_INFO, VALIDATED

    for name in ("RTX A5000", "RTX Pro 6000 WK"):
        g = GPU_INFO[name]
        assert g.runpod_mig_risk is True
        assert g.validated is True  # stays in the validated pool ...
        assert name in VALIDATED  # ... so the server's validation gate still accepts it
        # ... and stays available on Vast (the MIG substitution is a RunPod behavior; Vast rents
        # whole boards), so the exclusion must NOT remove its Vast identity.
        assert g.vast_name is not None


def test_default_max_retries_raised_for_mig_walk():
    """The GPU retry budget is raised (2 -> 6) so the runtime MIG-walk (exclude_class) reliably
    reaches a full GPU as a safety net for any MIG slice that slips past the allocator-side skip.
    Covers both the GpuSpec default and the JobSpec.from_dict default (the worker payload path)."""
    from flash.spec import GpuSpec, JobSpec

    assert GpuSpec().max_retries == 6
    assert JobSpec.from_dict({}).gpu.max_retries == 6
    assert JobSpec.from_dict({"gpu": {}}).gpu.max_retries == 6
    # An explicit value still wins (the default is only the fallback).
    assert JobSpec.from_dict({"gpu": {"max_retries": 3}}).gpu.max_retries == 3


def test_cheapest_gpu_skips_mig_risk_class(monkeypatch):
    """cheapest_gpu (the RunPod-static, parse-time provisional) must also skip runpod_mig_risk
    classes so the provisional matches what the RunPod allocator path actually provisions — else a
    `flash train` dry-run would display RTX A5000 while the real submit lands on RTX 3090."""
    from flash.providers.base import cheapest_gpu

    assert cheapest_gpu(24) == "RTX 3090"  # NOT RTX A5000 (runpod_mig_risk)
    assert cheapest_gpu(80) == "A100 PCIe"  # NOT RTX Pro 6000 WK (runpod_mig_risk)


def test_allocate_exclude_all_fitting_classes_raises(monkeypatch):
    """Excluding every fitting validated class leaves nothing to allocate -> UnsupportedGpuError
    (the run terminates cleanly rather than re-picking a banned class)."""
    from flash.providers import allocator
    from flash.providers.base import UnsupportedGpuError

    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    all_classes = frozenset(c.gpu for c in a.candidates)
    with pytest.raises(UnsupportedGpuError, match="excluding GPU classes"):
        allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo", exclude_gpu_classes=all_classes)


def test_offline_allocates_static_cheapest(monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import cheapest_gpu

    # No live pricing/offers (RunPod-only, static rates): allocation matches cheapest_gpu.
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert a.provider == "runpod"
    assert a.gpu == cheapest_gpu(24)


def test_exclude_gpu_classes_walks_to_next_cheapest(monkeypatch):
    # The capacity-walk uses exclude_gpu_classes to skip a throttled class WITHOUT lowering the
    # VRAM floor: re-allocation then returns the NEXT-cheapest fitting class, never raises just
    # because the cheapest was excluded (as long as something else still fits).
    from flash.providers import allocator

    monkeypatch.setenv("FLASH_SKIP_NET", "1")
    base = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    nxt = allocator.allocate(
        "Qwen/Qwen3.5-0.8B", "grpo", exclude_gpu_classes=frozenset({base.gpu})
    )
    assert nxt.gpu != base.gpu  # walked past the excluded (throttled) class
    assert nxt.min_vram_gb == base.min_vram_gb  # same VRAM floor — never sized below need
    # the excluded class is gone from the candidate pool entirely
    assert all(c.gpu != base.gpu for c in nxt.candidates)


def test_exclude_gpu_classes_provider_scoped(monkeypatch):
    # PR #4 review (thread 2): a no_capacity failure is RunPod-only (Vast has no IN_QUEUE), so the
    # exclusion is passed as a (provider, class) pair and must drop ONLY that provider's offer of
    # the class — the same class on the other provider must survive so an available Vast A100 isn't
    # skipped because RunPod's A100 queue was starved.
    from flash.providers import allocator
    from flash.providers.base import Candidate

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ["runpod", "vast"])
    monkeypatch.setattr(
        allocator,
        "_runpod_candidates",
        lambda *a, **k: [Candidate("runpod", "A100 SXM", 1.50, 80, True)],
    )
    monkeypatch.setattr(
        allocator,
        "_vast_candidates",
        lambda *a, **k: ([Candidate("vast", "A100 SXM", 1.00, 80, True)], ()),
    )

    # Scoped exclusion of the RunPod A100 leaves the (cheaper) Vast A100 selectable.
    a = allocator.allocate(
        "Qwen/Qwen3.5-0.8B", "grpo", exclude_gpu_classes=frozenset({("runpod", "A100 SXM")})
    )
    assert (a.provider, a.gpu) == ("vast", "A100 SXM")
    assert all(c.provider != "runpod" for c in a.candidates)

    # A bare class string still excludes the class on EVERY provider (legacy market-wide form).
    from flash.providers.base import UnsupportedGpuError

    with pytest.raises(UnsupportedGpuError):
        allocator.allocate(
            "Qwen/Qwen3.5-0.8B", "grpo", exclude_gpu_classes=frozenset({"A100 SXM"})
        )


def test_provider_pin_restricts_or_raises(monkeypatch):
    """The opt-in provider pin restricts allocation to one substrate: provider="runpod" stays on
    RunPod; provider=None is unchanged (cross-provider cheapest); provider="vast" without a key
    raises a clear UnsupportedGpuError instead of silently falling back to RunPod."""
    from flash.providers import allocator
    from flash.providers.base import UnsupportedGpuError

    # Offline harness: only RunPod is configured (VAST_API_KEY deleted in conftest).
    # provider=None -> unchanged cross-provider cheapest-wins (here: the static RunPod cheapest).
    base = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    pinned_rp = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo", provider="runpod")
    assert pinned_rp.provider == "runpod"
    assert pinned_rp.gpu == base.gpu  # RunPod-only either way offline, so identical

    # Pinning an unavailable provider is a CLEAN config error, never a silent RunPod fall-through.
    with pytest.raises(UnsupportedGpuError, match=r"provider 'vast' pinned but not available"):
        allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo", provider="vast")


def test_provider_pin_vast_returns_vast_allocation(monkeypatch):
    """With VAST_API_KEY present and a fitting offer, provider="vast" allocates on Vast (and
    excludes RunPod from the candidate pool entirely)."""
    from flash.providers import allocator
    from flash.providers.vast import jobs as vast_jobs
    from tests._helpers.vast import make_vast_offer

    # Make Vast "available" and feed the allocator a single fitting, validated offer.
    monkeypatch.setenv("VAST_API_KEY", "x")
    offer = make_vast_offer(gpu="RTX 3090", vram_gb=24, dph_total=0.20)
    monkeypatch.setattr(vast_jobs, "usable_offers", lambda *a, **k: [offer])

    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo", provider="vast")
    assert a.provider == "vast"
    assert a.gpu == "RTX 3090"
    assert all(c.provider == "vast" for c in a.candidates)  # RunPod excluded by the pin


def test_nothing_fits_names_constraint(monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import UnsupportedGpuError

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 4096)
    with pytest.raises(UnsupportedGpuError, match="4096 GB"):
        allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")


def test_estimator_matches_measured_seq_boundaries():
    """The raw VRAM physics reproduces the MEASURED RunPod capacity sweep: each anchor
    is a real train/OOM boundary observed on a pinned card (the calibration ground truth).
    estimate_vram_gb is the accurate estimate; model_required adds the safety headroom."""
    from flash.engine.vram import estimate_vram_gb as e

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
    # a 36B model in bf16 never fits 32 GB (~72 GB of weights alone)
    assert e(36.0, "sft", seq_len=4096) > 32


def test_required_vram_policy_floors_and_downrouting():
    """model_required_vram_gb: hard floors never under-provision; small runs down-route to
    a cheaper card; bigger context/group/thinking only ever size UP (never down)."""
    from flash.engine.vram import model_required_vram_gb as need

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
    # 9B GRPO is bf16 (QLoRA dropped: the 4-bit vLLM-rollout merge collapsed the GRPO
    # importance ratio -> no learning), so colocated GRPO needs an 80GB-class card.
    assert need("Qwen/Qwen3.5-9B", "grpo") >= 80  # bf16 colocate: 80GB floor
    assert need("Qwen/Qwen3.5-9B", "grpo", train={"max_length": 8192, "max_tokens": 2048, "group_size": 8}) >= 80
    assert need("Qwen/Qwen3.5-9B", "grpo", train={"max_length": 4096, "group_size": 8}) >= 80
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
    from flash.engine import vram
    from flash.engine.vram import estimate_vram_gb as e

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
    """provisional_gpu (parse-time) and required_vram_gb (submit-time) must size with the SAME
    headroom (a validated constant), so they never disagree (PR #176 review)."""
    from flash.providers import allocator

    assert allocator.vram_headroom() == 1.1
    # both paths feed model_required_vram_gb the same headroom -> identical sizing
    a_need = allocator.required_vram_gb("Qwen/Qwen3.5-4B", "grpo", train={"max_length": 4096})
    from flash.engine.vram import model_required_vram_gb

    direct = model_required_vram_gb("Qwen/Qwen3.5-4B", "grpo", train={"max_length": 4096}, headroom=1.1)
    assert a_need == direct


def test_allocate_never_selects_below_matrix_need(monkeypatch):
    """The core anti-OOM invariant: the GPU the allocator picks ALWAYS has >= the matrix's
    required VRAM, across a sweep of model x algo x seq x group x batch. If this ever fails,
    auto-allocation could provision a too-small card and OOM a paid worker."""
    from flash.providers.allocator import allocate, required_vram_gb
    from flash.providers.base import get_gpu_info

    grid = [
        ("Qwen/Qwen3.5-0.8B", "grpo", {"max_length": 1024, "group_size": 4}),
        ("Qwen/Qwen3.5-0.8B", "grpo", {"max_length": 32768, "group_size": 16}),
        # un-fused big-vocab SFT (the documented OOM case): a <3B model at a <2048-token ctx, where
        # the lm_head materializes the [per_device, seq, ~248k] fp32 logits the SFT branch once ignored.
        ("Qwen/Qwen3.5-0.8B", "sft", {"max_length": 1024}),
        ("Qwen/Qwen3.5-2B", "sft", {"max_length": 1536}),  # near the Liger threshold
        ("Qwen/Qwen3.5-2B", "sft", {"max_length": 8192}),
        ("Qwen/Qwen3.5-4B", "grpo", {"max_length": 1024, "group_size": 4}),
        ("Qwen/Qwen3.5-4B", "grpo", {"max_length": 16384, "max_tokens": 4096, "group_size": 8}),
        ("Qwen/Qwen3.5-4B", "sft", {"max_length": 32768}),
        ("Qwen/Qwen3.5-9B", "grpo", {"max_length": 8192, "group_size": 8}),
    ]
    for model, algo, tr in grid:
        need = required_vram_gb(model, algo, train=tr)
        alloc = allocate(model, algo, train=tr)
        assert get_gpu_info(alloc.gpu).vram_gb >= need, (model, algo, tr, alloc.gpu, need)


def test_sft_big_vocab_logits_term_present_and_bounded():
    """SFT must reserve the [per_device, seq, vocab] fp32-logits VRAM whenever the worker's fused CE
    is OFF (a <3B model AND a <2048-token ctx) -- the big-vocab SFT OOM driver the SFT branch
    previously ignored entirely. The term is bounded by the logits budget (the per-device cap keeps
    it there) and VANISHES once the worker fuses the CE, so it can't OOM and never over-reserves."""
    from flash.engine import vram
    from flash.engine.vram import estimate_vram_gb as e

    V = 248_320  # Qwen3.5's padded vocab
    # un-fused (0.9B, seq 1024): a big-vocab run reserves real logits vs a ~zero-vocab run ...
    with_logits = e(0.9, "sft", seq_len=1024, vocab=V, batch_size=4)
    no_logits = e(0.9, "sft", seq_len=1024, vocab=1, batch_size=4)
    assert with_logits > no_logits
    # ... but never by more than the budget (the per-device cap bounds the term, like GRPO)
    assert with_logits - no_logits <= vram._LOGITS_BUDGET_GB + 1e-6
    # fused gate: at seq >= 2048 the fused CE removes the term -> vocab no longer moves the estimate
    assert e(0.9, "sft", seq_len=2048, vocab=V) == e(0.9, "sft", seq_len=2048, vocab=1)
    # ... and a >= 3B model fuses at any ctx -> vocab-independent there too (unchanged from before)
    assert e(4.7, "sft", seq_len=1024, vocab=V) == e(4.7, "sft", seq_len=1024, vocab=1)


def test_sft_per_device_cap_keeps_unfused_logits_within_budget():
    """The worker's SFT per-device cap (sft_per_device) keeps the un-fused [pd, seq, vocab] logits
    within _LOGITS_BUDGET_GB whenever pd CAN be reduced -- the SFT mirror of rl_per_device_comps. At
    the pd=1 floor the logits are irreducible (a near-2048 big-vocab ctx can exceed the budget); the
    estimator then reserves that true floor (no clamp), so what's reserved still == what runs."""
    from flash.engine import vram

    V = 248_320
    for seq in (256, 512, 1024, 1536, 2000):  # all < 2048 -> un-fused for a small model
        pd = vram.sft_per_device(4, seq_len=seq, vocab=V, fused=False)
        logits_gb = pd * seq * V * vram._SFT_LOGITS_BYTES_PER_ELEM / 1e9
        assert 1 <= pd <= 4
        # cap holds logits <= budget unless pd is already floored to 1 (irreducible)
        assert pd == 1 or logits_gb <= vram._LOGITS_BUDGET_GB + 1e-6, (seq, pd, logits_gb)
        # the cap is TIGHT: one more micro-batch would breach the budget
        assert (pd + 1) * seq * V * vram._SFT_LOGITS_BYTES_PER_ELEM / 1e9 > vram._LOGITS_BUDGET_GB or pd == 4
    # fused -> no cap, the full micro-batch runs (no needless throughput loss)
    assert vram.sft_per_device(4, seq_len=4096, vocab=V, fused=True) == 4


def test_required_vram_sft_small_model_includes_big_vocab_logits():
    """REGRESSION (big-vocab SFT OOM): a sub-3B short-ctx SFT on a ~248k-vocab model must size for
    its lm_head logits. Pre-fix the SFT branch ignored them (need ~8 GB for a ~17 GB real peak that
    OOMs a 24 GB card near the Liger threshold); now the equation reserves the logits-inclusive need
    so the allocator can't under-provision. A >=2048 ctx fuses the CE, dropping the term back out."""
    from flash.providers.allocator import required_vram_gb

    n_short = required_vram_gb("Qwen/Qwen3.5-0.8B", "sft", train={"max_length": 1024})
    assert n_short >= 12  # base+act (~7) + capped logits (~4), x1.1 -- well above the pre-fix ~8
    n_fused = required_vram_gb("Qwen/Qwen3.5-0.8B", "sft", train={"max_length": 2048})
    assert n_fused < n_short  # the fused CE removes the logits term at >= 2048 ctx


def test_sft_equation_covers_honest_peak_across_seq_boundary():
    """Boundary regression: for EVERY catalog SFT model across the seq grid (straddling the 2048
    Liger threshold) x batch x rank, the equation must reserve >= the INDEPENDENT honest peak (incl.
    the capped big-vocab logits), and a validated card must fit. This is the in-CI distillation of
    the 148k-config offline sweep -- if the SFT logits term is ever dropped again, this fails."""
    import math

    from flash.catalog import MODELS, vocab_size_for
    from flash.engine import vram
    from flash.engine.vram import params_b_from_str, sft_logits_fused, sft_per_device
    from flash.providers.allocator import required_vram_gb
    from flash.providers.base import GPU_INFO

    validated = [g.vram_gb for g in GPU_INFO.values() if getattr(g, "validated", False)]

    def honest_peak(pb, seq, vocab, quant, rank, bs):
        bpp = vram._BYTES_PER_PARAM.get(quant, 2.0)
        width = math.sqrt(max(pb, 0.1))
        base = pb * bpp + vram._BASE_OVERHEAD_GB + (rank / 16.0) * (0.3 + 0.04 * pb)
        fused = sft_logits_fused(pb, seq)
        pd = sft_per_device(bs, seq_len=seq, vocab=vocab, fused=fused)
        act = vram._ACT_COEF * pd * (seq / 1024.0) * width
        logits = 0.0 if fused else pd * seq * vocab * vram._SFT_LOGITS_BYTES_PER_ELEM / 1e9
        return base + act + logits

    for mid, info in MODELS.items():
        if "sft" not in info.algos:
            continue
        pb = info.params_b or params_b_from_str(info.params) or 0.0
        vocab, quant = vocab_size_for(mid), getattr(info, "quant", "bf16") or "bf16"
        for seq in (512, 1024, 1536, 2047, 2048, 4096, 32768):
            for bs in (1, 4, 8, 32):
                for rank in (8, 32, 128):
                    tr = {"max_length": seq, "batch_size": bs, "lora_rank": rank}
                    need = required_vram_gb(mid, "sft", train=tr)
                    peak = math.ceil(honest_peak(pb, seq, vocab, quant, rank, bs) * 1.1)
                    assert need >= peak, (mid, seq, bs, rank, need, peak)
                    assert any(gb >= need for gb in validated), (mid, seq, bs, rank, need)


# ---------------------------------------------------------------------------
# Fix 3: SFT per-device micro-batch sized by the real vocab (big-vocab OOM guard)
# ---------------------------------------------------------------------------
def test_sft_logits_cap_shrinks_per_device_for_big_vocab():
    """A small, short-context SFT (fused CE OFF) over the ~248k Qwen3.5 vocab must cap the
    per-device micro-batch so the [per_device, seq, vocab] fp32 logits+grad fit the budget;
    grad-accum rises so the realized effective batch is never below the request (the OOM the
    fix addresses: a 0.8B SFT OOM'd a 24 GB card in backward)."""
    from flash.engine.vram import (
        _LOGITS_BUDGET_GB,
        _SFT_LOGITS_BYTES_PER_ELEM,
        sft_grad_accum,
        sft_logits_per_device_cap,
    )

    vocab = 248_320
    seq = 1024
    # Fused OFF (small model, short ctx): per_device is vocab-capped below the fixed 4.
    pd, ga = sft_grad_accum(8, seq_len=seq, vocab=vocab, fused=False)
    assert pd < 4  # capped below the fixed micro-batch default
    assert pd * ga >= 8  # realized effective batch never below the request
    # the capped per-device logits stay within the budget
    assert pd * seq * vocab * _SFT_LOGITS_BYTES_PER_ELEM / 1e9 <= _LOGITS_BUDGET_GB + 1e-9
    assert pd == sft_logits_per_device_cap(seq, vocab) or pd == 4


def test_sft_logits_cap_no_regression_small_vocab_or_fused():
    """The cap must NOT shrink the micro-batch for a small-vocab model, nor when the fused CE is
    on (Liger fuses the logits away) — those keep the fixed per-device 4."""
    from flash.engine.vram import sft_grad_accum, sft_per_device

    # Small vocab (e.g. ~32k): the [pd, seq, vocab] logits are tiny -> no cap, keep 4.
    assert sft_per_device(8, seq_len=1024, vocab=32_000, fused=False) == 4
    # Fused CE on (the worker fuses for >=3B model OR >=2048 ctx): logits never materialize -> 4.
    assert sft_per_device(8, seq_len=1024, vocab=248_320, fused=True) == 4
    # Default call (no seq/vocab/fused) is the old fixed behavior, so existing callers are unchanged.
    assert sft_grad_accum(8) == (4, 2)


def test_sft_logits_fused_conservative_size_gate():
    """sft_logits_fused is the allocator's CONSERVATIVE fused-CE gate: it banks on the saving only
    for a >=3B model OR a >=2048-token context. chalk's FLCE actually fuses every run, but a small
    short-context run is sized as if un-fused (so the GPU is never undersized) and the per-device
    logits cap applies there."""
    from flash.engine.vram import sft_logits_fused

    assert sft_logits_fused(0.8, 1024) is False  # small + short -> conservatively not banked -> cap applies
    assert sft_logits_fused(4.0, 1024) is True  # >=3B
    assert sft_logits_fused(0.8, 2048) is True  # long ctx
    assert sft_logits_fused(None, 1024) is False  # unknown size -> memory-safe (cap applies)


def test_sft_estimate_includes_capped_logits_term():
    """The SFT VRAM estimate must include the big-vocab logits term (previously ignored), but
    bounded by the per-device logits cap so it never over-reserves — and a long context that
    fuses the CE drops the term entirely."""
    from flash.engine.vram import _LOGITS_BUDGET_GB
    from flash.engine.vram import estimate_vram_gb as e

    big = e(0.8, "sft", seq_len=1024, vocab=248_320, batch_size=8)
    small = e(0.8, "sft", seq_len=1024, vocab=8_000, batch_size=8)
    assert big > small  # the big-vocab logits term is real and now counted
    # bounded: the per-device cap keeps the logits term within the budget (plus the activation diff)
    assert big - small <= _LOGITS_BUDGET_GB + 2.0
    # >=2048 ctx fuses the CE -> no logits term -> big vocab no longer inflates the estimate
    fused_big = e(0.8, "sft", seq_len=2048, vocab=248_320, batch_size=8)
    fused_small = e(0.8, "sft", seq_len=2048, vocab=8_000, batch_size=8)
    assert abs(fused_big - fused_small) < 1e-6
