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
    # chunked nll bounds the vocab projection to one verl FusedLinearForPPO chunk (512 token rows,
    # NOT 256): 512 * 248320 * 16 B = 2.03 GB, which is what the child actually allocates.
    assert required_vram_gb("Qwen/Qwen3.5-4B", "sft") == 20
    # sized for GRPO (the heavier phase of the usual SFT+GRPO run) + headroom
    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda m, **k: 4.0)
    est = vram.estimate_vram_gb(4.0, "grpo")

    # Default GRPO (no [train].max_context_tokens) sizes at the run's REAL engine length, mirroring
    # run_rl()'s max(1024, rl.max_prompt_len + completion) = 2048 + 320 = 2368 tokens (NOT a flat
    # 1024). At 2368 the 4.7B param estimate is ~31.8 GB raw -> 35 GB with headroom, so a 32 GB card
    # is no longer a safe fit and the allocator escalates to the cheapest validated >=35 GB class.
    a = allocator.allocate("Qwen/Qwen3.5-4B", "grpo")
    assert a.min_vram_gb == 35
    assert all(c.vram_gb >= 35 for c in a.candidates)


def test_allocation_restricted_to_validated_pool():
    from flash.providers import allocator
    from flash.providers.base import VALIDATED

    # The deployed control plane rejects a submit for any non-validated class, so client-side
    # allocation must only ever pick a class in the validated pool — across ALL candidates,
    # not just the chosen one. Offline only RunPod is available; 0.8B GRPO needs the 24 GB tier
    # whose cheapest VALIDATED RunPod class is RTX 4090 @ $0.69. 24 GB is the floor — sub-24 GB
    # classes were dropped.
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert a.provider == "runpod"
    assert all(c.gpu in VALIDATED for c in a.candidates), [
        c.gpu for c in a.candidates if c.gpu not in VALIDATED
    ]
    assert a.gpu == "RTX 4090"  # the cheapest VALIDATED RunPod class that fits 24 GB


def test_allocation_skips_cheaper_unvalidated_class(monkeypatch):
    """The allocator must skip a cheaper UNVALIDATED class for the cheapest VALIDATED one (so the
    deployed control plane accepts the submit). The managed catalog is now fully validated, so
    inject a synthetic unvalidated RunPod class cheaper than any real one and confirm it is
    excluded from the candidate set."""
    from flash.providers import allocator
    from flash.providers.base import GPU_INFO, VALIDATED, GpuClass

    fake = GpuClass("FAKE Cheap", "NVIDIA_FAKE", 24, "fakecheap", "sm80", 0.10)
    assert not fake.validated
    monkeypatch.setitem(GPU_INFO, "FAKE Cheap", fake)

    # 4B SFT (seq 1024, rank 8) down-routes below 24 GB in the matrix (see test_required_vram_*).
    a = allocator.allocate(
        "Qwen/Qwen3.5-4B", "sft", train={"max_context_tokens": 1024, "lora_rank": 8}
    )
    assert a.min_vram_gb < 24  # a sub-24 GB run the synthetic unvalidated card also fits
    # The synthetic class is cheaper and fits, yet is excluded because it is unvalidated.
    assert any(
        (not g.validated) and g.enum_member and g.vram_gb >= a.min_vram_gb
        for g in GPU_INFO.values()
    )
    assert all(c.gpu in VALIDATED for c in a.candidates)
    assert "FAKE Cheap" not in [c.gpu for c in a.candidates]


def test_runpod_allocation_lands_on_full_validated_cards():
    """Allocation lands on the card with the cheapest dollars-per-step among validated classes."""
    from flash.providers import allocator

    # ranking is on cost per step rather than $/hr, but on MEASURED throughput the RTX 4090 is also
    # the best value in the pool (~4.2 $/PFLOP-hr vs the H100 PCIe's ~6.6), so a small SFT run stays
    # there. it wins on both bases here; the cases where the two bases DISAGREE are covered by
    # test_total_cost_ranking_beats_hourly_rate below.
    a08_sft = allocator.allocate("Qwen/Qwen3.5-0.8B", "sft")
    assert a08_sft.provider == "runpod"
    assert a08_sft.gpu == "RTX 4090"
    # grpo spends most of a step waiting on reward grading, which no card shortens, so the extra
    # throughput cannot pay for itself and the ranking collapses back toward the cheapest rate.
    a08_grpo = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert a08_grpo.provider == "runpod"
    assert a08_grpo.gpu == "RTX 4090"
    a9 = allocator.allocate("Qwen/Qwen3.5-9B", "grpo")
    assert a9.provider == "runpod"
    assert a9.gpu == "A100 PCIe"  # cheapest validated 80 GB RunPod card
    a27_sft = allocator.allocate("Qwen/Qwen3.6-27B", "sft")
    assert a27_sft.provider == "runpod"
    assert a27_sft.gpu == "A100 PCIe"
    a27_grpo = allocator.allocate("Qwen/Qwen3.6-27B", "grpo")
    assert a27_grpo.provider == "runpod"
    assert (
        a27_grpo.gpu == "B200"
    )  # colocated GRPO (trainer + vLLM rollout = two ~54GB copies) needs B200


def test_total_cost_ranking_beats_hourly_rate():
    """A card that costs more per hour wins when it finishes enough sooner to pay for itself.

    This is the whole point of ranking on job cost: the cheapest RENTAL and the cheapest RUN are
    different cards whenever throughput differs enough. Stubbed so the assertion is about the
    ranking rule and not about whichever real classes happen to be priced today.
    """
    from flash.cost.analytical import step_cost_key
    from flash.cost.types import RunConfig

    key = step_cost_key(RunConfig(model_id="Qwen/Qwen3.5-4B", method="sft", steps=1))
    assert key is not None
    # A10: 125 TFLOPS at $1.29. RTX 4090: 165 TFLOPS at $0.69. The 4090 is both cheaper and faster.
    assert key("RTX 4090", 0.69) < key("A10", 1.29)
    # Now price the A10 BELOW the 4090. It is still the more expensive way to run the job, because
    # the extra wall time costs more than the rate saves -- which $/hr ranking cannot see.
    assert key("RTX 4090", 0.69) < key("A10", 0.60)


def test_step_cost_ranking_declines_unknown_classes():
    """A class with no throughput data is not ranked on a placeholder speed.

    Returning a constant leaves such classes ordered by the $/hr tie-break instead of inventing a
    speed difference the hardware may not have.
    """
    from flash.cost.analytical import step_cost_key
    from flash.cost.types import RunConfig

    key = step_cost_key(RunConfig(model_id="Qwen/Qwen3.5-4B", method="sft", steps=1))
    assert key("definitely not a real gpu", 1.00) == key("also not real", 99.00) == 0.0


def test_step_cost_key_none_for_uncatalogued_model():
    """An unpriceable model degrades to $/hr ranking rather than failing allocation."""
    from flash.cost.analytical import step_cost_key
    from flash.cost.types import RunConfig

    assert (
        step_cost_key(RunConfig(model_id="some/model-not-in-catalog", method="sft", steps=1))
        is None
    )


def test_latency_bound_step_ignores_gpu_speed():
    """When a step is dominated by waits no card shortens, ranking collapses back toward $/hr.

    GRPO spends most of a step waiting on concurrent reward grading, so buying more FLOPs cannot
    pay for itself and the cheaper rental legitimately wins.
    """
    from flash.cost.analytical import step_seconds_split
    from flash.cost.types import RunConfig

    gpu_bound, fixed = step_seconds_split(
        RunConfig(model_id="Qwen/Qwen3.5-0.8B", method="grpo", steps=1), "H100"
    )
    assert fixed > gpu_bound  # the wait, not the math, is what the step is made of


def test_default_max_retries():
    """The GPU retry budget default (5) covers infra-shaped flakes (worker loss / stall / timeout)
    and matches INFRA_RETRY_FLOOR (runner.lifecycle), which the runner already floored the effective
    budget to — so the declared default now reflects the real GPU-walk budget. Covers both the
    GpuSpec default and the JobSpec.from_dict default (the worker payload path)."""
    from flash.runner.lifecycle import INFRA_RETRY_FLOOR
    from flash.spec import GpuSpec, JobSpec

    assert GpuSpec().max_retries == 5
    assert GpuSpec().max_retries == INFRA_RETRY_FLOOR  # default tracks the runner's infra floor
    assert JobSpec.from_dict({}).gpu.max_retries == 5
    assert JobSpec.from_dict({"gpu": {}}).gpu.max_retries == 5
    # An explicit value still wins (the default is only the fallback).
    assert JobSpec.from_dict({"gpu": {"max_retries": 3}}).gpu.max_retries == 3


def test_cheapest_gpu_picks_cheapest_validated_runpod_class():
    """cheapest_gpu (the RunPod-static, parse-time provisional) picks the cheapest VALIDATED
    RunPod-provisionable class that fits, matching what the RunPod allocator path provisions."""
    from flash.providers.base import cheapest_gpu

    assert cheapest_gpu(24) == "RTX 4090"  # cheapest validated RunPod class that fits 24 GB
    assert cheapest_gpu(80) == "A100 PCIe"  # cheapest validated 80 GB RunPod class


def test_offline_allocates_static_cheapest():
    from flash.providers import allocator
    from flash.providers.base import cheapest_gpu

    # RunPod-only static rates: allocation matches cheapest_gpu.
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert a.provider == "runpod"
    assert a.gpu == cheapest_gpu(24)


def test_nothing_fits_names_constraint(monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import UnsupportedGpuError

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 4096)
    with pytest.raises(UnsupportedGpuError, match="4096 GB"):
        allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")


def test_allocate_provider_constraint_never_falls_through(monkeypatch):
    from flash.providers import allocator, get_provider
    from flash.providers.base import Candidate

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "lambda"))
    monkeypatch.setattr(
        get_provider("runpod"),
        "live_candidates",
        lambda need, constraints: [Candidate("runpod", "RTX 4090", 0.50, 24)],
    )
    monkeypatch.setattr(
        get_provider("lambda"),
        "live_candidates",
        lambda need, constraints: [Candidate("lambda", "A10", 1.29, 24)],
    )

    allocation = allocator.allocate(
        "Qwen/Qwen3.5-0.8B",
        "grpo",
        provider="lambda",
    )

    assert allocation.provider == "lambda"
    assert allocation.gpu == "A10"
    assert {candidate.provider for candidate in allocation.candidates} == {"lambda"}


def test_allocate_rejects_unconfigured_provider(monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import UnsupportedGpuError

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    with pytest.raises(UnsupportedGpuError, match="not configured"):
        allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo", provider="lambda")


def test_allocate_gpu_type_never_widens_or_escalates(monkeypatch):
    from flash.providers import allocator, get_provider
    from flash.providers.base import Candidate

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "lambda"))
    monkeypatch.setattr(
        get_provider("runpod"),
        "live_candidates",
        lambda need, constraints: [
            Candidate("runpod", "RTX 4090", 0.50, 24),
            Candidate("runpod", "H100", 3.29, 80),
        ],
    )
    monkeypatch.setattr(
        get_provider("lambda"),
        "live_candidates",
        lambda need, constraints: [Candidate("lambda", "H100", 2.49, 80)],
    )

    allocation = allocator.allocate(
        "Qwen/Qwen3.5-0.8B",
        "grpo",
        gpu_type="h100",
    )

    assert allocation.gpu == "H100"
    assert {candidate.gpu for candidate in allocation.candidates} == {"H100"}


def test_allocate_gpu_type_enforces_vram_and_provider_support(monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import UnsupportedGpuError

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 80)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "lambda"))
    with pytest.raises(UnsupportedGpuError, match="requires at least"):
        allocator.allocate(
            "Qwen/Qwen3.5-9B",
            "grpo",
            gpu_type="RTX 4090",
        )
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    with pytest.raises(UnsupportedGpuError, match="cannot provision"):
        allocator.allocate(
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            provider="lambda",
            gpu_type="RTX 4090",
        )


@pytest.mark.parametrize("provider", ["lambda", "vast"])
def test_exact_dynamic_provider_empty_capacity_is_retryable(monkeypatch, provider):
    from flash.providers import allocator, get_provider
    from flash.providers.base import CapacityLookupError

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: (provider,))
    monkeypatch.setattr(get_provider(provider), "live_candidates", lambda need, constraints: [])

    with pytest.raises(CapacityLookupError, match="currently has no capacity"):
        allocator.allocate(
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            provider=provider,
            gpu_type="H100",
        )


def test_exact_runpod_empty_capacity_stays_terminal(monkeypatch):
    from flash.providers import allocator, get_provider
    from flash.providers.base import UnsupportedGpuError

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    monkeypatch.setattr(get_provider("runpod"), "live_candidates", lambda need, constraints: [])

    with pytest.raises(UnsupportedGpuError, match="no allocatable capacity"):
        allocator.allocate(
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            provider="runpod",
            gpu_type="H100",
        )


def test_allocate_gpu_type_ignores_ineligible_provider_blip(monkeypatch):
    from flash.providers import allocator, get_provider
    from flash.providers.base import CapacityLookupError, UnsupportedGpuError

    calls: list[str] = []
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "lambda"))
    monkeypatch.setattr(
        get_provider("runpod"),
        "live_candidates",
        lambda need, constraints: calls.append("runpod") or [],
    )

    def lambda_blip(need, constraints):
        calls.append("lambda")
        raise CapacityLookupError("lambda live capacity lookup failed")

    monkeypatch.setattr(get_provider("lambda"), "live_candidates", lambda_blip)

    with pytest.raises(UnsupportedGpuError) as exc_info:
        allocator.allocate(
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            gpu_type="H200",
        )

    assert not isinstance(exc_info.value, CapacityLookupError)
    assert calls == ["runpod"]


def _raise_capacity_blip(*a, **k):
    from flash.providers.base import CapacityLookupError

    raise CapacityLookupError("vast live capacity lookup failed") from RuntimeError("market blip")


def _stub_alloc(monkeypatch, *, runpod, lambda_, vast):
    from flash.providers import allocator, get_provider

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "lambda", "vast"))
    # allocate() now sources each provider's candidates via provider.live_candidates(need, constraints);
    # adapt the per-provider stubs (runpod/lambda take need; vast takes need + disk/wall) onto that seam.
    monkeypatch.setattr(
        get_provider("runpod"), "live_candidates", lambda need, constraints: runpod(need)
    )
    monkeypatch.setattr(
        get_provider("lambda"), "live_candidates", lambda need, constraints: lambda_(need)
    )
    monkeypatch.setattr(
        get_provider("vast"),
        "live_candidates",
        lambda need, constraints: vast(need, constraints.disk_gb, constraints.max_wall_seconds),
    )


def test_transient_capacity_blip_is_retryable_not_terminal(monkeypatch):
    """A live capacity-lookup outage that is the SOLE reason nothing fits raises the RETRYABLE
    CapacityLookupError, NOT the terminal UnsupportedGpuError — so the runner infra-retries the blip
    (isinstance check must stay False, since lifecycle terminal-fails only on UnsupportedGpuError)."""
    from flash.providers import allocator
    from flash.providers.base import CapacityLookupError, UnsupportedGpuError

    _stub_alloc(
        monkeypatch,
        runpod=lambda need: [],  # no RunPod class fits
        lambda_=lambda need: [],
        vast=_raise_capacity_blip,  # Vast (the only possible source) blipped
    )
    with pytest.raises(CapacityLookupError) as ei:
        allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert not isinstance(ei.value, UnsupportedGpuError)


def test_capacity_blip_degrades_to_fitting_provider(monkeypatch):
    """A Vast blip must NOT abort allocation when another provider has a fitting class — degrade to it."""
    from flash.providers import allocator
    from flash.providers.base import Candidate

    _stub_alloc(
        monkeypatch,
        runpod=lambda need: [Candidate("runpod", "RTX 4090", 0.69, 24)],
        lambda_=lambda need: [],
        vast=_raise_capacity_blip,
    )
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")
    assert a.provider == "runpod"  # degraded past the blip, no error


def test_genuine_no_fit_without_blip_stays_terminal(monkeypatch):
    """No blip, just nothing fits -> terminal UnsupportedGpuError (unchanged contract)."""
    from flash.providers import allocator
    from flash.providers.base import UnsupportedGpuError

    _stub_alloc(
        monkeypatch,
        runpod=lambda need: [],
        lambda_=lambda need: [],
        vast=lambda need, disk_gb=0.0, max_wall_seconds=0.0: [],
    )
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate("Qwen/Qwen3.5-0.8B", "grpo")


def test_estimator_matches_measured_seq_boundaries():
    """The raw VRAM physics reproduces the MEASURED RunPod capacity sweep: each anchor
    is a real train/OOM boundary observed on a pinned card (the calibration ground truth).
    estimate_vram_gb is the accurate estimate; model_required adds the safety headroom."""
    from flash.engine.vram import estimate_vram_gb as e

    # 0.8B (0.9B): GRPO seq up to 32k fits the cheapest 24 GB card. Real SFT materializes
    # dense logits, so only a shorter context stays in that class.
    assert e(0.9, "grpo", seq_len=4096) <= 24
    assert e(0.9, "grpo", seq_len=32768) <= 24
    assert e(0.9, "sft", seq_len=4096) <= 24
    # the direct estimator defaults to conservative plain nll when no model identity is available.
    assert e(4.7, "sft", seq_len=8192, vocab=248_320) > 32
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
    short = need(m4, "grpo", train={"max_context_tokens": 1024, "max_completion_tokens": 256})
    long = need(m4, "grpo", train={"max_context_tokens": 16384, "max_completion_tokens": 4096})
    assert long > short
    # sub-1B GRPO fits a 24 GB card; 2B GRPO OOMs 24 (MEASURED) -> needs the 32 tier; small SFT
    # drops below the catalog default (32)
    assert need("Qwen/Qwen3.5-0.8B", "grpo") <= 24
    assert 24 < need("Qwen/Qwen3.5-2B", "grpo") <= 32
    assert need(m4, "sft", train={"max_context_tokens": 1024, "lora_rank": 8}) < 32
    # 9B GRPO is bf16 (QLoRA dropped: the 4-bit vLLM-rollout merge collapsed the GRPO
    # importance ratio -> no learning), so colocated GRPO needs an 80GB-class card.
    assert need("Qwen/Qwen3.5-9B", "grpo") >= 80  # bf16 colocate: 80GB floor
    assert (
        need(
            "Qwen/Qwen3.5-9B",
            "grpo",
            train={"max_context_tokens": 8192, "max_completion_tokens": 2048, "group_size": 8},
        )
        >= 80
    )
    assert (
        need("Qwen/Qwen3.5-9B", "grpo", train={"max_context_tokens": 4096, "group_size": 8}) >= 80
    )
    need_27b_sft = need("Qwen/Qwen3.6-27B", "sft")
    need_27b_grpo = need("Qwen/Qwen3.6-27B", "grpo")
    assert need_27b_sft == 80
    assert need_27b_grpo == 150  # colocated-GRPO resident peak -> B200
    # group size and thinking never DECREASE the requirement
    base = need(
        m4,
        "grpo",
        train={"max_context_tokens": 4096, "max_completion_tokens": 1024, "group_size": 4},
    )
    assert (
        need(
            m4,
            "grpo",
            train={"max_context_tokens": 4096, "max_completion_tokens": 1024, "group_size": 16},
        )
        >= base
    )
    assert (
        need(
            m4,
            "grpo",
            train={"max_context_tokens": 4096, "max_completion_tokens": 1024, "group_size": 4},
            thinking=True,
        )
        >= base
    )
    # max_tokens (completion length) lifts the fp32-logits term -> a longer completion never sizes
    # DOWN, and a much longer one sizes UP (the term the estimator previously ignored).
    short_c = need(
        m4,
        "grpo",
        train={"max_context_tokens": 8192, "max_completion_tokens": 256, "group_size": 8},
    )
    long_c = need(
        m4,
        "grpo",
        train={"max_context_tokens": 8192, "max_completion_tokens": 8192, "group_size": 8},
    )
    assert long_c >= short_c


def test_required_vram_sizes_weights_from_curated_params_b_not_display_string():
    """Cursor Medium: model_required_vram_gb must size the resident WEIGHT term from the curated
    ``ModelInfo.params_b`` (the single source of truth resolve_params_b / the cost model read).
    params_b is now a required numeric field and the ``params`` display string is display-only
    (params_b_from_str was removed): re-parsing the string was fragile for an MoE whose string lists
    BOTH counts ("35B total / ~3B active") — the first parsed token could be the ~3B active count,
    sizing the ~70 GB resident weights ~10x too small and under-provisioning the card."""
    from flash.catalog import MODELS, ModelInfo
    from flash.engine.vram import model_required_vram_gb

    fake_id = "test/moe-active-first-string"
    # A pathological display string that lists the ACTIVE count FIRST — the exact footgun the curated
    # params_b avoids now that the string is never parsed.
    fake = ModelInfo(
        id=fake_id,
        display_name="fake MoE (active-first string)",
        params="~3B active / 35B total (MoE)",
        algos=("sft", "grpo"),
        min_vram_gb=141,
        params_b=35.0,
        active_params_b=3.0,
        vocab_size=248_320,
    )
    MODELS[fake_id] = fake
    try:
        # Sized from the curated 35.0 -> the ~70 GB bf16 resident weights dominate, so the SFT need is
        # far above any ~3B estimate (~12 GB). >= 70 proves we used 35B, not the parsed 3B.
        assert model_required_vram_gb(fake_id, "sft") >= 70
    finally:
        del MODELS[fake_id]


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
    assert e(2.0, "sft", seq_len=4096, max_tokens=256) == e(
        2.0, "sft", seq_len=4096, max_tokens=8192
    )


def test_opd_vram_estimate_reserves_one_dense_image_loss_peak():
    from flash.engine.vram import estimate_vram_gb as e

    seq_len = 4096
    completion = 512
    vocab = 248_320
    with_vocab = e(
        2.0,
        "opd",
        seq_len=seq_len,
        max_tokens=completion,
        vocab=vocab,
        batch_size=4,
        group_size=4,
    )
    without_vocab = e(
        2.0,
        "opd",
        seq_len=seq_len,
        max_tokens=completion,
        vocab=0,
        batch_size=4,
        group_size=4,
    )
    expected = (seq_len * 4 + completion * 8) * vocab / 1e9
    assert with_vocab - without_vocab == pytest.approx(expected)

    short = e(2.0, "opd", seq_len=seq_len, max_tokens=128, vocab=vocab)
    long = e(2.0, "opd", seq_len=seq_len, max_tokens=2048, vocab=vocab)
    assert long > short


def test_opd_uses_opd_sizing_not_grpo():
    """OPD must size on its own dense-logit estimator, never on the GRPO colocate path.

    Regression (codex[bot], vram.py): a sizing branch hardcoded ``_need(params_b, 'grpo', ...)``, so
    an OPD run was sized as a colocated-vLLM GRPO job -- rejecting fitting runs or routing them to
    pricier GPUs. The real algorithm must reach the estimator, so the two diverge.
    """
    from flash.engine.vram import model_required_vram_gb

    train = {"max_context_tokens": 8192, "max_completion_tokens": 8192, "lora_rank": 16}
    for model_id in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"):
        opd_need = model_required_vram_gb(model_id, "opd", train=train)
        grpo_need = model_required_vram_gb(model_id, "grpo", train=train)
        assert opd_need != grpo_need, f"{model_id} OPD must not size as the GRPO colocate path"
        assert opd_need > 0


def test_opd_applies_the_colocated_vllm_floor():
    """OPD starts a resident colocated vLLM engine, so a tiny model cannot be admitted on its tiny
    training estimate -- the engine's own footprint sets a floor the training term never reaches."""
    from flash.engine.vram import model_required_vram_gb

    train = {
        "max_context_tokens": 1536,
        "max_completion_tokens": 128,
        "batch_size": 1,
        "group_size": 1,
    }
    # the smallest catalog model: its training estimate is far under the floor, so the 24 GB it
    # reports IS the floor rather than a coincidence of the sizing equations.
    assert model_required_vram_gb("Qwen/Qwen3.5-0.8B", "opd", train=train, headroom=1.0) == 24


def test_vram_headroom_consistent_across_sizing_paths():
    """provisional_gpu (parse-time) and required_vram_gb (submit-time) must size with the SAME
    headroom (a validated constant), so they never disagree (PR #176 review)."""
    from flash.providers import allocator

    assert allocator.vram_headroom() == 1.1
    # both paths feed model_required_vram_gb the same headroom -> identical sizing
    a_need = allocator.required_vram_gb(
        "Qwen/Qwen3.5-4B", "grpo", train={"max_context_tokens": 4096}
    )
    from flash.engine.vram import model_required_vram_gb

    direct = model_required_vram_gb(
        "Qwen/Qwen3.5-4B", "grpo", train={"max_context_tokens": 4096}, headroom=1.1
    )
    assert a_need == direct


def test_allocate_never_selects_below_matrix_need():
    """The core anti-OOM invariant: the GPU the allocator picks ALWAYS has >= the matrix's
    required VRAM, across a sweep of model x algo x seq x group x batch. If this ever fails,
    auto-allocation could provision a too-small card and OOM a paid worker."""
    from flash.providers.allocator import allocate, required_vram_gb
    from flash.providers.base import get_gpu_info

    grid = [
        ("Qwen/Qwen3.5-0.8B", "grpo", {"max_context_tokens": 1024, "group_size": 4}),
        ("Qwen/Qwen3.5-0.8B", "grpo", {"max_context_tokens": 32768, "group_size": 16}),
        # chunked-nll qwen sft cases across short and long contexts.
        ("Qwen/Qwen3.5-0.8B", "sft", {"max_context_tokens": 1024}),
        ("Qwen/Qwen3.5-2B", "sft", {"max_context_tokens": 1536}),
        ("Qwen/Qwen3.5-2B", "sft", {"max_context_tokens": 8192}),
        ("Qwen/Qwen3.5-4B", "grpo", {"max_context_tokens": 1024, "group_size": 4}),
        (
            "Qwen/Qwen3.5-4B",
            "grpo",
            {"max_context_tokens": 16384, "max_completion_tokens": 4096, "group_size": 8},
        ),
        ("Qwen/Qwen3.5-4B", "sft", {"max_context_tokens": 32768}),
        ("Qwen/Qwen3.5-9B", "grpo", {"max_context_tokens": 8192, "group_size": 8}),
    ]
    for model, algo, tr in grid:
        need = required_vram_gb(model, algo, train=tr)
        alloc = allocate(model, algo, train=tr)
        assert get_gpu_info(alloc.gpu).vram_gb >= need, (model, algo, tr, alloc.gpu, need)


def test_observed_qwen2_opd_vllm_case_routes_off_32gb_cards(monkeypatch):
    """Regression: Qwen3.5-2B OPD with the vLLM rollout engine failed startup on RTX 5090.

    The aggregate estimate said the run needed only the 28 GB colocate floor, but vLLM initializes
    after the HF/PEFT student is already resident and requires its executor budget to be free. Keep
    this observed shape off consumer 32 GB cards and 40 GB fallback classes."""
    from flash.cost import RunConfig, estimate_cost
    from flash.providers import allocator
    from flash.providers.allocator import required_vram_gb
    from flash.providers.base import get_gpu_info, provisional_gpu

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    train = {"epochs": 1, "max_completion_tokens": 128, "lora_rank": 32}

    need = required_vram_gb("Qwen/Qwen3.5-2B", "opd", train=train)
    assert need > 40

    preview_gpu = provisional_gpu("Qwen/Qwen3.5-2B", "opd", train=train)
    alloc = allocator.allocate("Qwen/Qwen3.5-2B", "opd", train=train)
    estimate = estimate_cost(
        RunConfig(
            "Qwen/Qwen3.5-2B",
            "opd",
            60,
            completion_len=128,
            lora_rank=32,
            provider="runpod",
        )
    )

    assert preview_gpu == "A100 PCIe"
    assert alloc.gpu == preview_gpu
    assert alloc.min_vram_gb == need
    assert estimate.required_vram_gb == need
    assert estimate.gpu == preview_gpu
    assert get_gpu_info(preview_gpu).vram_gb >= need


@pytest.mark.parametrize(
    ("label", "train", "expected_gpu"),
    [
        (
            "max_completion_128",
            {"epochs": 1, "max_completion_tokens": 128, "lora_rank": 32},
            "A100 PCIe",
        ),
        (
            "max_completion_256",
            {"epochs": 1, "max_completion_tokens": 256, "lora_rank": 32},
            "A100 PCIe",
        ),
        (
            "max_completion_512",
            {"epochs": 1, "max_completion_tokens": 512, "lora_rank": 32},
            "A100 PCIe",
        ),
        (
            "max_completion_1024",
            {"epochs": 1, "max_completion_tokens": 1024, "lora_rank": 32},
            "A100 PCIe",
        ),
        (
            "max_context_2048",
            {
                "epochs": 1,
                "max_context_tokens": 2048,
                "max_completion_tokens": 128,
                "lora_rank": 32,
            },
            "A100 PCIe",
        ),
        (
            "max_context_8192",
            {
                "epochs": 1,
                "max_context_tokens": 8192,
                "max_completion_tokens": 128,
                "lora_rank": 32,
            },
            "A100 PCIe",
        ),
        (
            # the dense image fallback grows with context and keeps this mixed-modality-safe route
            # above the 96 gb class.
            "max_context_16384",
            {
                "epochs": 1,
                "max_context_tokens": 16384,
                "max_completion_tokens": 128,
                "lora_rank": 32,
            },
            "H200",
        ),
        (
            # b200, not h200: every catalog model is a gdn hybrid, so the opd rollout runs a bf16 kv
            # cache (the worker refuses fp8 for them) and sizing must reserve the full cache.
            "max_context_24576",
            {
                "epochs": 1,
                "max_context_tokens": 24576,
                "max_completion_tokens": 128,
                "lora_rank": 32,
            },
            "B200",
        ),
        (
            "group_size_8",
            {"epochs": 1, "group_size": 8, "max_completion_tokens": 128, "lora_rank": 32},
            "A100 PCIe",
        ),
    ],
)
def test_observed_qwen2_opd_sweep_never_downroutes_to_32_or_40gb(
    monkeypatch, label, train, expected_gpu
):
    """Attachment regression: 2B OPD/vLLM dry-runs stayed on 4090/5090-class GPUs.

    The submitted worker then OOMed during vLLM rollout initialization. These are the same sweep axes
    from the report: completion length, context length, and group size. All control-plane views must
    agree on a >40 GB requirement and a fitting non-consumer GPU before any paid worker is created.
    """
    from flash.cost import RunConfig, estimate_cost
    from flash.providers import allocator
    from flash.providers.allocator import required_vram_gb
    from flash.providers.base import get_gpu_info, provisional_gpu
    from flash.schema import spec_from_dict

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))

    model = "Qwen/Qwen3.5-2B"
    need = required_vram_gb(model, "opd", train=train)
    run_config = RunConfig(
        model,
        "opd",
        int(train.get("epochs", 1)),
        seq_len=train.get("max_context_tokens"),
        completion_len=train.get("max_completion_tokens"),
        group_size=train.get("group_size"),
        lora_rank=train.get("lora_rank"),
        provider="runpod",
    )

    assert need > 40
    if expected_gpu is None:
        from flash.providers.base import GPU_INFO, UnsupportedGpuError

        assert need > max(g.vram_gb for g in GPU_INFO.values() if g.validated)
        with pytest.raises(UnsupportedGpuError):
            provisional_gpu(model, "opd", train=train)
        with pytest.raises(UnsupportedGpuError):
            allocator.allocate(model, "opd", train=train)
        with pytest.raises(ValueError, match="no GPU class fits"):
            estimate_cost(run_config)
        return

    preview_gpu = provisional_gpu(model, "opd", train=train)
    alloc = allocator.allocate(model, "opd", train=train)
    estimate = estimate_cost(run_config)
    spec = spec_from_dict(
        {
            "model": model,
            "algorithm": "opd",
            "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
            "train": dict(train),
            "gpu": {},
        },
        run_id=f"opd-sweep-{label}",
    )

    assert preview_gpu == expected_gpu
    assert alloc.gpu == expected_gpu
    assert estimate.gpu == expected_gpu
    assert spec.gpu.type == ""
    assert alloc.min_vram_gb == need
    assert estimate.required_vram_gb == need
    assert get_gpu_info(expected_gpu).vram_gb >= need
    assert get_gpu_info(expected_gpu).vram_gb > 40


def test_observed_qwen4b_opd_vllm_startup_case_routes_off_40gb_cards(monkeypatch):
    """Regression: Qwen3.5-4B OPD at 8k ctx / 128 rollout tokens failed vLLM startup on 40 GB.

    The trainer is resident before the colocated vLLM engine initializes, so the allocator must not
    consider the 40 GB A100 class viable even though the rollout token budget is intentionally small.
    """
    from flash.cost import RunConfig, estimate_cost
    from flash.providers import allocator
    from flash.providers.allocator import required_vram_gb
    from flash.providers.base import get_gpu_info, provisional_gpu

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    train = {
        "epochs": 1,
        "max_context_tokens": 8192,
        "max_completion_tokens": 128,
        "lora_rank": 32,
    }

    need = required_vram_gb("Qwen/Qwen3.5-4B", "opd", train=train)
    # the dense image fallback keeps this run above the 80 gb class while still fitting the 96 gb
    # rtx pro 6000.
    assert 80 < need <= 96

    preview_gpu = provisional_gpu("Qwen/Qwen3.5-4B", "opd", train=train)
    alloc = allocator.allocate("Qwen/Qwen3.5-4B", "opd", train=train)
    estimate = estimate_cost(
        RunConfig(
            "Qwen/Qwen3.5-4B",
            "opd",
            1,
            seq_len=8192,
            completion_len=128,
            lora_rank=32,
            provider="runpod",
        )
    )

    assert preview_gpu == "RTX Pro 6000"
    assert alloc.gpu == preview_gpu
    assert alloc.min_vram_gb == need
    assert estimate.required_vram_gb == need
    assert estimate.gpu == preview_gpu
    assert get_gpu_info(preview_gpu).vram_gb >= need


def test_opd_catalog_model_config_gpu_matrix_routes_to_fitting_cards(monkeypatch):
    """OPD-specific matrix guard: each catalog OPD model across representative train configs must
    resolve to a GPU that satisfies the shared VRAM requirement, or reject before provisioning when
    the config exceeds every managed single-GPU class."""
    from flash.catalog import MODELS
    from flash.cost import RunConfig, estimate_cost
    from flash.providers import allocator
    from flash.providers.allocator import required_vram_gb
    from flash.providers.base import (
        GPU_INFO,
        UnsupportedGpuError,
        get_gpu_info,
        providers_for,
        provisional_gpu,
    )
    from flash.schema import ConfigError, spec_from_dict

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    max_managed_vram = max(g.vram_gb for g in GPU_INFO.values() if g.validated)
    configured_gpu_types = tuple(name for name, gpu_info in GPU_INFO.items() if gpu_info.validated)
    configs = {
        # Matches the failed continuation shape from the OPD/vLLM RTX 5090 report.
        "observed_2b_128tok_r32": {
            "epochs": 1,
            "max_completion_tokens": 128,
            "lora_rank": 32,
        },
        "recipe_default": {"epochs": 1},
        "opd_prompt_batch": {
            "epochs": 1,
            "batch_size": 8,
            "group_size": 1,
            "max_context_tokens": 1536,
            "max_completion_tokens": 512,
            "lora_rank": 16,
        },
        "longer_context": {
            "epochs": 1,
            "batch_size": 8,
            "group_size": 1,
            "max_context_tokens": 4096,
            "max_completion_tokens": 512,
            "lora_rank": 16,
        },
        "longer_completion": {
            "epochs": 1,
            "batch_size": 1,
            "group_size": 1,
            "max_context_tokens": 4096,
            "max_completion_tokens": 2048,
            "lora_rank": 16,
        },
        "wide_rollout_batch": {
            "epochs": 1,
            "batch_size": 8,
            "group_size": 4,
            "max_context_tokens": 4096,
            "max_completion_tokens": 512,
            "lora_rank": 16,
        },
    }
    checked: set[tuple[str, str]] = set()
    rejected: set[tuple[str, str]] = set()

    for model_id, info in MODELS.items():
        if "opd" not in info.algos:
            continue
        for label, train in configs.items():
            need = required_vram_gb(model_id, "opd", train=train)
            rc = RunConfig(
                model_id,
                "opd",
                int(train.get("epochs", 1)),
                seq_len=train.get("max_context_tokens"),
                completion_len=train.get("max_completion_tokens"),
                batch_size=train.get("batch_size"),
                group_size=train.get("group_size"),
                lora_rank=train.get("lora_rank"),
                provider="runpod",
            )

            if need > max_managed_vram:
                with pytest.raises(UnsupportedGpuError):
                    allocator.allocate(model_id, "opd", train=train)
                # Cost preflight is deliberately offline so a capacity lookup cannot consume a
                # lifecycle retry before the run exists. It still rejects the same impossible shape.
                with pytest.raises(ValueError, match="no GPU class fits"):
                    estimate_cost(rc)
                rejected.add((model_id, label))
                continue

            preview_gpu = provisional_gpu(model_id, "opd", train=train)
            preview_info = get_gpu_info(preview_gpu)
            assert preview_info.validated
            assert preview_info.vram_gb >= need, (model_id, label, preview_gpu, need)

            alloc = allocator.allocate(model_id, "opd", train=train)
            alloc_info = get_gpu_info(alloc.gpu)
            assert alloc.provider == "runpod"
            assert alloc.min_vram_gb == need
            assert alloc.gpu == preview_gpu
            assert alloc_info.validated
            assert alloc_info.vram_gb >= need, (model_id, label, alloc.gpu, need)
            assert all(c.vram_gb >= need for c in alloc.candidates)

            estimate = estimate_cost(rc)
            assert estimate.required_vram_gb == need
            assert estimate.gpu == preview_gpu
            assert estimate.gpu_vram_gb >= need, (model_id, label, estimate.gpu, need)

            for configured_gpu in configured_gpu_types:
                raw = {
                    "model": model_id,
                    "algorithm": "opd",
                    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
                    "train": dict(train),
                    "gpu": {"type": configured_gpu},
                }
                if get_gpu_info(configured_gpu).vram_gb < need:
                    with pytest.raises(ConfigError, match="requires at least"):
                        spec_from_dict(raw, run_id="opd-matrix")
                    continue

                spec = spec_from_dict(raw, run_id="opd-matrix")
                assert spec.gpu.type == configured_gpu
                assert providers_for(spec.gpu.type)
                assert get_gpu_info(spec.gpu.type).vram_gb >= need

            checked.add((model_id, label))

    expected = {
        (model_id, label)
        for model_id, info in MODELS.items()
        if "opd" in info.algos
        for label in configs
    }
    assert checked | rejected == expected
    assert checked
    assert rejected


def test_catalog_model_algorithm_gpu_matrix_routes_to_fitting_cards(monkeypatch):
    """Full catalog matrix guard: every supported model x algorithm route must pick a card that
    meets the shared VRAM requirement across schema preview, submit allocation, and cost estimate."""
    from flash.catalog import ALGORITHMS, MODELS
    from flash.cost import RunConfig, estimate_cost
    from flash.providers import allocator
    from flash.providers.base import GPU_INFO, combined_vram_gb, get_gpu_info, provisional_gpu

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    expected = {
        (model_id, algo)
        for model_id, info in MODELS.items()
        for algo in ALGORITHMS
        if algo in info.algos
    }
    checked = set()

    for model_id, info in MODELS.items():
        for algo in ALGORITHMS:
            if algo not in info.algos:
                continue
            train = {}
            need = allocator.required_vram_gb(model_id, algo, train=train, thinking=False)
            # most cells fit one card. the 35B MoE trains 256 routed experts per fused tensor, so its
            # GRPO/OPD peaks exceed every single card and the route is only defined across two.
            cards = 2 if need > max(g.vram_gb for g in GPU_INFO.values() if g.validated) else 1

            preview_gpu = provisional_gpu(
                model_id, algo, train=train, thinking=False, gpu_count=cards
            )
            preview_info = get_gpu_info(preview_gpu)
            assert preview_info.validated
            assert preview_info.enum_member
            # sharding is not free: combined_vram_gb applies the per-card overhead, so two cards hold
            # less than twice one card. sizing with a naive sum would overstate what the shape holds.
            assert combined_vram_gb(preview_info.vram_gb, cards) >= need, (
                model_id,
                algo,
                preview_gpu,
                need,
            )

            alloc = allocator.allocate(
                model_id, algo, train=train, thinking=False, max_gpu_count=cards
            )
            alloc_info = get_gpu_info(alloc.gpu)
            assert alloc.provider == "runpod"
            assert alloc.min_vram_gb == need
            assert alloc_info.validated
            assert alloc_info.enum_member
            assert combined_vram_gb(alloc_info.vram_gb, cards) >= need, (
                model_id,
                algo,
                alloc.gpu,
                need,
            )
            assert all(combined_vram_gb(c.vram_gb, cards) >= need for c in alloc.candidates)

            estimate = estimate_cost(
                RunConfig(model_id, algo, 1, provider="runpod", gpu_count=cards)
            )
            assert estimate.gpu_vram_gb * estimate.gpu_count >= estimate.required_vram_gb, (
                model_id,
                algo,
                estimate.gpu,
                estimate.required_vram_gb,
            )
            checked.add((model_id, algo))

    assert checked == expected
    assert {algo for _, algo in checked} == set(ALGORITHMS)


def test_catalog_model_algorithm_config_gpu_matrix_enforces_pins(monkeypatch):
    """Every active validated GPU pin is preserved when it fits and rejected when it does not."""
    from flash.catalog import ALGORITHMS, MODELS
    from flash.providers import allocator
    from flash.providers.base import GPU_INFO, get_gpu_info, providers_for
    from flash.schema import ConfigError, spec_from_dict

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    configured_gpu_types = tuple(name for name, gpu_info in GPU_INFO.items() if gpu_info.validated)
    expected = {
        (model_id, algo, configured_gpu)
        for model_id, info in MODELS.items()
        for algo in ALGORITHMS
        if algo in info.algos
        for configured_gpu in configured_gpu_types
    }
    checked = set()
    rejected = set()

    for model_id, info in MODELS.items():
        for algo in ALGORITHMS:
            if algo not in info.algos:
                continue
            train = {"epochs": 1, "max_examples": 8}
            need = allocator.required_vram_gb(model_id, algo, train=train)

            for configured_gpu in configured_gpu_types:
                raw = {
                    "model": model_id,
                    "algorithm": algo,
                    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
                    "train": train,
                    "gpu": {"type": configured_gpu},
                }
                key = (model_id, algo, configured_gpu)
                if get_gpu_info(configured_gpu).vram_gb < need:
                    # two distinct rejections, both correct: a pin that is merely too small names the
                    # shortfall, while a run that outgrows EVERY validated class (35B GRPO/OPD, once
                    # the routed experts train) fails earlier with no fitting class at all.
                    biggest = max(g.vram_gb for g in GPU_INFO.values() if g.validated)
                    # opd words its over-capacity error differently from the generic allocator one,
                    # so match the shared "no ... validated GPU" shape rather than either wording.
                    reason = (
                        "requires at least"
                        if need <= biggest
                        else r"(no validated GPU class has|more than any single validated GPU)"
                    )
                    with pytest.raises(ConfigError, match=reason):
                        spec_from_dict(raw, run_id="matrix")
                    rejected.add(key)
                    continue

                spec = spec_from_dict(raw, run_id="matrix")
                assert spec.gpu.type == configured_gpu
                resolved_info = get_gpu_info(spec.gpu.type)
                assert resolved_info.validated
                assert providers_for(spec.gpu.type)
                assert resolved_info.vram_gb >= need
                checked.add(key)

    assert checked | rejected == expected
    assert checked
    assert rejected


def test_sft_big_vocab_logits_term_present_and_bounded():
    """plain nll reserves dense logits while chunked nll reserves one bounded projection chunk."""
    from flash.engine import vram
    from flash.engine.vram import estimate_vram_gb as e

    V = 248_320  # Qwen3.5's padded vocab
    with_logits = e(0.9, "sft", seq_len=1024, vocab=V, batch_size=4)
    no_logits = e(0.9, "sft", seq_len=1024, vocab=1, batch_size=4)
    assert with_logits > no_logits
    assert with_logits - no_logits <= vram._LOGITS_BUDGET_GB + 1e-6

    chunked_big = e(0.9, "sft", seq_len=2048, vocab=V, sft_fused_ce=True)
    chunked_small = e(0.9, "sft", seq_len=2048, vocab=1, sft_fused_ce=True)
    assert chunked_big > chunked_small
    assert chunked_big - chunked_small <= (
        vram._SFT_CHUNKED_NLL_TOKENS * V * vram._SFT_LOGITS_BYTES_PER_ELEM / 1e9
    )
    assert chunked_big < e(0.9, "sft", seq_len=2048, vocab=V, sft_fused_ce=False)


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
        assert (
            pd + 1
        ) * seq * V * vram._SFT_LOGITS_BYTES_PER_ELEM / 1e9 > vram._LOGITS_BUDGET_GB or pd == 4
    # fused -> no cap, the full micro-batch runs (no needless throughput loss)
    assert vram.sft_per_device(4, seq_len=4096, vocab=V, fused=True) == 4


def test_required_vram_qwen_chunked_nll_drops_big_vocab_logits():
    """validated qwen sft sizing bounds vocab logits while retaining activation growth."""
    import math

    from flash.catalog import MODELS, vocab_size_for
    from flash.engine.vram import estimate_vram_gb
    from flash.providers.allocator import required_vram_gb

    model_id = "Qwen/Qwen3.5-0.8B"
    info = MODELS[model_id]
    n_short = required_vram_gb(model_id, "sft", train={"max_context_tokens": 1024})
    expected = math.ceil(
        estimate_vram_gb(
            info.params_b,
            "sft",
            seq_len=1024,
            vocab=vocab_size_for(model_id),
            sft_fused_ce=True,
        )
        * 1.1
    )
    # 10, not 9: the reserved projection is one 512-row verl fused-CE chunk, not 256 rows.
    assert n_short == expected == 10
    n_long = required_vram_gb(model_id, "sft", train={"max_context_tokens": 2048})
    assert n_long >= n_short


def test_qwen4b_sft_8192_chunked_nll_routes_to_32gb_card(monkeypatch):
    """chunked nll removes the dense-logit term that previously forced this shape onto 80 gb."""
    from flash.cost import RunConfig, estimate_cost
    from flash.providers import allocator
    from flash.providers.allocator import required_vram_gb
    from flash.providers.base import get_gpu_info, provisional_gpu

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    train = {"epochs": 1, "max_examples": 4020, "max_context_tokens": 8192, "lora_rank": 32}

    need = required_vram_gb("Qwen/Qwen3.5-4B", "sft", train=train)
    assert need == 28

    preview_gpu = provisional_gpu("Qwen/Qwen3.5-4B", "sft", train=train)
    alloc = allocator.allocate("Qwen/Qwen3.5-4B", "sft", train=train)
    estimate = estimate_cost(
        RunConfig(
            "Qwen/Qwen3.5-4B",
            "sft",
            1,
            seq_len=8192,
            lora_rank=32,
            provider="runpod",
        )
    )

    assert preview_gpu == "RTX 5090"
    assert alloc.gpu == preview_gpu
    assert alloc.min_vram_gb == need
    assert estimate.required_vram_gb == need
    assert estimate.gpu == preview_gpu
    assert get_gpu_info(preview_gpu).vram_gb >= need


def test_required_vram_sft_plain_nll_fallback_keeps_logits_term(monkeypatch):
    import math

    from flash.catalog import vocab_size_for
    from flash.engine import vram
    from flash.engine.vram import estimate_vram_gb, model_required_vram_gb

    mid = "meta-llama/Llama-3.2-1B"
    params_b = 1.2
    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda model_id: params_b)
    train = {"max_context_tokens": 4096, "batch_size": 4}
    need = model_required_vram_gb(mid, "sft", train=train, headroom=1.0)
    unfused = math.ceil(
        estimate_vram_gb(
            params_b,
            "sft",
            seq_len=train["max_context_tokens"],
            batch_size=train["batch_size"],
            vocab=vocab_size_for(mid),
            sft_fused_ce=False,
        )
    )
    fused = math.ceil(
        estimate_vram_gb(
            params_b,
            "sft",
            seq_len=train["max_context_tokens"],
            batch_size=train["batch_size"],
            vocab=vocab_size_for(mid),
            sft_fused_ce=True,
        )
    )

    assert need == unfused
    assert need > fused


def test_sft_equation_covers_honest_peak_across_seq_boundary():
    """the allocator must mirror chunked qwen and plain-nll fallback peaks across the catalog."""
    import math

    from flash.catalog import MODELS, vocab_size_for
    from flash.engine import vram
    from flash.engine.vram import sft_chunked_nll_enabled, sft_per_device
    from flash.providers.allocator import required_vram_gb
    from flash.providers.base import GPU_INFO

    validated = [g.vram_gb for g in GPU_INFO.values() if getattr(g, "validated", False)]

    def honest_peak(mid, pb, seq, vocab, quant, rank, bs, active_b=None):
        # moe activations and optimizer scale with the active backbone while weights stay on total params.
        bpp = vram._BYTES_PER_PARAM.get(quant, 2.0)
        eff = float(active_b) if active_b else pb
        width = math.sqrt(max(eff, 0.1))
        base = pb * bpp + vram._BASE_OVERHEAD_GB + (rank / 16.0) * (0.3 + 0.04 * eff)
        fused = sft_chunked_nll_enabled(mid)
        pd = sft_per_device(bs, seq_len=seq, vocab=vocab, fused=fused)
        act = vram._ACT_COEF * pd * (seq / 1024.0) * width
        projected = min(pd * seq, vram._SFT_CHUNKED_NLL_TOKENS) if fused else pd * seq
        logits = projected * vocab * vram._SFT_LOGITS_BYTES_PER_ELEM / 1e9
        return base + act + logits

    for mid, info in MODELS.items():
        if "sft" not in info.algos:
            continue
        pb = info.params_b  # required curated field
        active_b = float(getattr(info, "active_params_b", 0.0) or 0.0)
        vocab, quant = vocab_size_for(mid), getattr(info, "quant", "bf16") or "bf16"
        for seq in (512, 1024, 1536, 2047, 2048, 4096, 32768):
            for bs in (1, 4, 8, 32):
                for rank in (8, 32, 128):
                    tr = {"max_context_tokens": seq, "batch_size": bs, "lora_rank": rank}
                    need = required_vram_gb(mid, "sft", train=tr)
                    peak = math.ceil(
                        honest_peak(mid, pb, seq, vocab, quant, rank, bs, active_b) * 1.1
                    )
                    # The conservative estimate must always cover the honest peak (universal).
                    assert need >= peak, (mid, seq, bs, rank, need, peak)
                    # Fitting configs must have a validated target; configs above every managed
                    # single-GPU class are still safe because schema/allocation rejects them before
                    # provisioning.
                    assert need > max(validated) or any(gb >= need for gb in validated), (
                        mid,
                        seq,
                        bs,
                        rank,
                        need,
                    )


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


def test_sft_chunked_nll_restores_qwen_microbatch_and_gc_gate():
    from flash.engine.vram import sft_chunked_nll_enabled, sft_grad_accum
    from flash.engine.worker.perf import grad_checkpointing_on

    model_id = "Qwen/Qwen3.5-0.8B"
    chunked = sft_chunked_nll_enabled(model_id)
    assert sft_grad_accum(8, seq_len=1024, vocab=248_320, fused=chunked) == (4, 2)
    assert (
        grad_checkpointing_on(
            model_id,
            1024,
            allow_disable=True,
            card_vram_gb=80,
            capability=(9, 0),
            active_params_b=0.9,
            hidden=1024,
            num_layers=24,
            fused_ce=chunked,
            per_device_bs=4,
            lora_rank=32,
        )
        is False
    )


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


def test_sft_chunked_nll_model_gate_mirrors_worker():
    from flash.engine.vram import sft_chunked_nll_enabled

    assert sft_chunked_nll_enabled("Qwen/Qwen3.5-0.8B") is True
    assert sft_chunked_nll_enabled("Qwen/Qwen3.5-9B") is True
    assert sft_chunked_nll_enabled("Qwen/Qwen3.6-27B") is True
    assert sft_chunked_nll_enabled("Qwen/Qwen3.6-35B-A3B") is True
    assert sft_chunked_nll_enabled("meta-llama/Llama-3.2-1B") is False
    assert sft_chunked_nll_enabled("org/unknown") is False


def test_every_sft_catalog_model_is_sized_for_the_fused_loss():
    """sizing must mirror the worker, which sets use_fused_kernels=true for EVERY model.

    the enumerated gate above cannot fail when a NEW catalog model is added and left out of the
    set, which is exactly how Qwen3.6-27B came to be sized for dense logits it never allocates.
    every catalog model is a qwen3_5/qwen3_5_moe checkpoint, and verl dispatches both to the fused
    torch backend, so the sft-capable catalog and the set must stay identical.
    """
    from flash.catalog import MODELS
    from flash.engine.vram import sft_chunked_nll_enabled

    missing = sorted(
        mid
        for mid, info in MODELS.items()
        if "sft" in info.algos and not sft_chunked_nll_enabled(mid)
    )
    assert not missing, (
        f"sft-capable catalog models sized for dense logits the fused worker never builds: {missing}"
    )


def test_sft_estimate_includes_capped_logits_term():
    """the direct conservative estimate keeps dense logits; chunked mode uses a smaller fixed term."""
    from flash.engine.vram import _LOGITS_BUDGET_GB
    from flash.engine.vram import estimate_vram_gb as e

    big = e(0.8, "sft", seq_len=1024, vocab=248_320, batch_size=8)
    small = e(0.8, "sft", seq_len=1024, vocab=8_000, batch_size=8)
    assert big > small  # the big-vocab logits term is real and now counted
    # bounded: the per-device cap keeps the logits term within the budget (plus the activation diff)
    assert big - small <= _LOGITS_BUDGET_GB + 2.0
    # Real SFT still carries the logits term at >=2048 ctx.
    long_big = e(0.8, "sft", seq_len=2048, vocab=248_320, batch_size=8)
    long_small = e(0.8, "sft", seq_len=2048, vocab=8_000, batch_size=8)
    assert long_big > long_small
    # chunked mode still accounts for one vocab projection chunk, but stays far below plain nll.
    chunked_big = e(0.8, "sft", seq_len=2048, vocab=248_320, batch_size=8, sft_fused_ce=True)
    assert chunked_big < long_big


def test_vast_candidates_searches_at_effective_disk(monkeypatch):
    # Codex Mslml: the allocator's Vast capacity search must use the SAME effective disk floor
    # (max(disk_gb, MIN_DISK_GB)) the submit path provisions with — else a high-disk run is advertised
    # Vast capacity that only exists at the 60 GB floor and then can't actually rent (an impossible
    # attempt a max_retries=0 run never escapes).
    from flash.providers import get_provider
    from flash.providers.base import AllocationConstraints
    from flash.providers.vast import jobs as vast_jobs

    captured = {}

    def fake_usable(vram_floor, disk_gb, *a, **k):
        captured["disk_gb"] = disk_gb
        return []

    monkeypatch.setattr(vast_jobs, "usable_offers", fake_usable)
    vast = get_provider("vast")
    vast.live_candidates(16, AllocationConstraints())  # default -> floored at MIN_DISK_GB
    assert captured["disk_gb"] == vast_jobs.MIN_DISK_GB
    vast.live_candidates(
        16, AllocationConstraints(disk_gb=200.0)
    )  # high-disk run searches at the request
    assert captured["disk_gb"] == 200.0
    vast.live_candidates(16, AllocationConstraints(disk_gb=10.0))  # below the floor still clamps up
    assert captured["disk_gb"] == vast_jobs.MIN_DISK_GB


def test_vast_candidates_threads_max_wall_seconds(monkeypatch):
    # Codex Msvb0: the allocator's Vast capacity search must thread the run's wall cap so usable_offers
    # applies the duration floor — else the allocator advertises Vast classes whose only live offers
    # expire before the run finishes (fatal for a max_retries=0 run).
    from flash.providers import get_provider
    from flash.providers.base import AllocationConstraints
    from flash.providers.vast import jobs as vast_jobs

    captured = {}

    def fake_usable(vram_floor, disk_gb, *a, **k):
        captured["max_wall_seconds"] = k.get("max_wall_seconds")
        return []

    monkeypatch.setattr(vast_jobs, "usable_offers", fake_usable)
    vast = get_provider("vast")
    vast.live_candidates(
        16, AllocationConstraints()
    )  # default -> no deadline threaded (0 = duration filter off)
    assert captured["max_wall_seconds"] == 0.0
    vast.live_candidates(
        16, AllocationConstraints(max_wall_seconds=7200.0)
    )  # long run threads its wall cap
    assert captured["max_wall_seconds"] == 7200.0


def _stub_provider(monkeypatch, allocator, candidates_by_need):
    """stub a single provider whose live_candidates returns fixed candidates filtered by per-card need.

    Each supplied candidate is offered at every rentable count up to ``constraints.max_gpu_count``,
    which is what a real provider now does: providers report the shapes they can genuinely rent and
    the allocator only decides which one fits. A stub that returned single-card shapes only could
    never produce a multi-card candidate, so every combination assertion below would be unfailable.
    """
    from dataclasses import replace

    from flash.providers.base import Candidate, rentable_gpu_counts

    class _P:
        name = "runpod"

        def live_candidates(self, need, constraints):
            return [
                replace(c, gpu_count=count)
                for c in candidates_by_need
                if c.vram_gb >= need
                for count in rentable_gpu_counts(constraints.max_gpu_count)
            ]

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    monkeypatch.setattr(allocator, "get_provider", lambda name: _P())
    return Candidate


def test_combo_default_single_gpu_behavior_unchanged(monkeypatch):
    # max_gpu_count=1 (default): identical to classic cheapest single-class allocation.
    from flash.providers import allocator
    from flash.providers.base import Candidate

    cands = [
        Candidate(provider="runpod", gpu="A100 PCIe", hourly_usd=1.5, vram_gb=80),
        Candidate(provider="runpod", gpu="H200", hourly_usd=4.0, vram_gb=141),
    ]
    _stub_provider(monkeypatch, allocator, cands)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 100)
    a = allocator.allocate("m", "sft")
    assert (a.gpu, a.gpu_count) == ("H200", 1)  # only class fitting 100 GB alone


def test_combo_two_cheap_cards_beat_one_expensive(monkeypatch):
    # 2 x A100 ($3.00 total, 160 GB * 0.85 = 136 GB effective) beats 1 x H200 ($4.00) for a 100 GB need.
    from flash.providers import allocator
    from flash.providers.base import Candidate

    cands = [
        Candidate(provider="runpod", gpu="A100 PCIe", hourly_usd=1.5, vram_gb=80),
        Candidate(provider="runpod", gpu="H200", hourly_usd=4.0, vram_gb=141),
    ]
    _stub_provider(monkeypatch, allocator, cands)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 100)
    a = allocator.allocate("m", "sft", max_gpu_count=4)
    assert (a.gpu, a.gpu_count) == ("A100 PCIe", 2)
    assert a.hourly_usd == 1.5  # per-card rate preserved
    assert a.candidates[0].total_hourly_usd == 3.0


def test_combo_single_kept_when_cheaper_than_combination(monkeypatch):
    # 1 x H200 ($2.00) beats 2 x A100 ($3.00): combinations only win on total cost.
    from flash.providers import allocator
    from flash.providers.base import Candidate

    cands = [
        Candidate(provider="runpod", gpu="A100 PCIe", hourly_usd=1.5, vram_gb=80),
        Candidate(provider="runpod", gpu="H200", hourly_usd=2.0, vram_gb=141),
    ]
    _stub_provider(monkeypatch, allocator, cands)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 100)
    a = allocator.allocate("m", "sft", max_gpu_count=4)
    assert (a.gpu, a.gpu_count) == ("H200", 1)


def test_combo_uses_smallest_fitting_count_and_shard_margin(monkeypatch):
    """The smallest RENTABLE count that fits, with the shard margin actually deciding the boundary.

    Counts are powers of two (verl shards over them: num_attention_heads % sp_size != 0 aborts at
    step 0), so on 80 GB cards with the replicated-floor model, usable = n*(80-8)*0.85 + 8:
    2 cards = 130.4 GB, 4 cards = 252.8 GB.

    Both needs are asserted because they pin different halves of the rule. 200 GB pins
    smallest-fitting-count (2 is too small, so 4). 140 GB pins the shard MARGIN itself: it sits in
    the gap between the discounted 2-card capacity (130.4) and the undiscounted one (152), so an
    allocator that forgot to discount would rent 2 cards and OOM on a run that needs 4.
    """
    from flash.providers import allocator
    from flash.providers.base import Candidate

    cands = [Candidate(provider="runpod", gpu="A100 PCIe", hourly_usd=1.5, vram_gb=80)]
    _stub_provider(monkeypatch, allocator, cands)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 200)
    a = allocator.allocate("m", "sft", max_gpu_count=4)
    assert (a.gpu, a.gpu_count) == ("A100 PCIe", 4)

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 140)
    a = allocator.allocate("m", "sft", max_gpu_count=4)
    assert (a.gpu, a.gpu_count) == ("A100 PCIe", 4), "the shard margin was not applied"


def test_combo_replicated_floor_excludes_tiny_cards(monkeypatch):
    # cards at/below the replicated floor can never combine, regardless of count.
    from flash.providers import allocator
    from flash.providers.base import Candidate, UnsupportedGpuError

    cands = [Candidate(provider="runpod", gpu="TINY 8GB", hourly_usd=0.1, vram_gb=8)]
    _stub_provider(monkeypatch, allocator, cands)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 100)
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate("m", "sft", max_gpu_count=4)


def test_combo_summary_shows_count_and_total(monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import Candidate

    cands = [
        Candidate(provider="runpod", gpu="A100 PCIe", hourly_usd=1.5, vram_gb=80),
        Candidate(provider="runpod", gpu="H200", hourly_usd=4.0, vram_gb=141),
    ]
    _stub_provider(monkeypatch, allocator, cands)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 100)
    a = allocator.allocate("m", "sft", max_gpu_count=2)
    s = allocator.allocation_summary(a)
    assert "2x A100 PCIe" in s
    assert "$3.00/hr" in s


def test_fused_ce_chunk_matches_verl_default():
    """the reserved vocab projection must equal what the verl child actually allocates.

    both sizing paths bound the projection by verl's ``FusedLinearForPPO(chunk_size=...)`` default.
    reserving fewer rows than the child projects under-reserves, which admits a job that then OOMs
    on a paid gpu -- the failure this pins. the literal 512 is deliberate: deriving it from
    ``VERL_FUSED_CE_CHUNK_TOKENS`` would move with the constant and never fail.
    """
    from flash.engine import vram

    assert vram.VERL_FUSED_CE_CHUNK_TOKENS == 512
    assert vram._SFT_CHUNKED_NLL_TOKENS == 512
    assert vram.OPD_CE_CHUNK_SIZE == 512


def test_sft_default_context_tracks_thinking_mode():
    """unauthored sft context must size at the length the worker trains on, per mode.

    ``sft_max_length`` trims rows to ``RECIPE.sft.max_seq_len_thinking`` when thinking is on, so a
    flat non-thinking default sized activations for half the real sequence.
    """
    from flash.engine.recipe import RECIPE
    from flash.engine.vram import model_required_vram_gb

    assert RECIPE.sft.max_seq_len_thinking > RECIPE.sft.max_seq_len
    mid = "Qwen/Qwen3.5-4B"
    plain = model_required_vram_gb(mid, "sft", thinking=False)
    thinking = model_required_vram_gb(mid, "sft", thinking=True)
    assert thinking > plain, (plain, thinking)
