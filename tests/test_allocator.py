"""RunPod allocation: VRAM sizing, cheapest-wins ranking, fallbacks (no pins, no gates)."""

from __future__ import annotations

import pytest


def test_allocation_skips_cheaper_unvalidated_class(monkeypatch):
    """A cheaper fitting class stays excluded until it is validated."""
    from flash.providers.core import allocator
    from flash.providers.core.base import GPU_INFO, VALIDATED, GpuClass

    synthetic = GpuClass(
        "synthetic cheap gpu",
        "NVIDIA_SYNTHETIC_CHEAP",
        24,
        "syntheticcheap",
        "sm80",
        0.01,
    )
    monkeypatch.setitem(GPU_INFO, synthetic.name, synthetic)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *args, **kwargs: 24)

    allocation = allocator.allocate("test/catalog-independent-model", "sft")

    assert synthetic.vram_gb >= allocation.min_vram_gb
    assert synthetic.hourly_usd < allocation.hourly_usd
    assert synthetic.name not in {candidate.gpu for candidate in allocation.candidates}
    assert all(candidate.gpu in VALIDATED for candidate in allocation.candidates)


def test_runpod_allocation_lands_on_full_validated_cards():
    """Allocation lands on the card with the cheapest dollars-per-step among validated classes."""
    from flash.providers.core import allocator

    a9 = allocator.allocate("Qwen/Qwen3.5-9B", "grpo")
    assert a9.provider == "runpod"
    assert a9.gpu == "A100 PCIe"  # cheapest validated 80 GB RunPod card
    a27_sft = allocator.allocate("Qwen/Qwen3.8-27B", "sft")
    assert a27_sft.provider == "runpod"
    assert a27_sft.gpu == "A100 PCIe"
    a27_grpo = allocator.allocate("Qwen/Qwen3.8-27B", "grpo")
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

    key = step_cost_key(RunConfig(model_id="Qwen/Qwen3.5-9B", method="sft", steps=1))
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

    key = step_cost_key(RunConfig(model_id="Qwen/Qwen3.5-9B", method="sft", steps=1))
    assert key("definitely not a real gpu", 1.00) == key("also not real", 99.00) == 0.0


def test_step_cost_key_none_for_uncatalogued_model():
    """An unpriceable model degrades to $/hr ranking rather than failing allocation."""
    from flash.cost.analytical import step_cost_key
    from flash.cost.types import RunConfig

    assert (
        step_cost_key(RunConfig(model_id="some/model-not-in-catalog", method="sft", steps=1))
        is None
    )


def test_default_max_retries():
    """The GPU retry budget default (5) covers infra-shaped flakes (worker loss / stall / timeout)
    and matches INFRA_RETRY_FLOOR (runner.lifecycle), which the runner already floored the effective
    budget to — so the declared default now reflects the real GPU-walk budget. Covers both the
    GpuSpec default and the JobSpec.from_dict default (the worker payload path)."""
    from flash.core.spec import GpuSpec, JobSpec
    from flash.runner.supervise.retry_decision import INFRA_RETRY_FLOOR

    assert GpuSpec().max_retries == 5
    assert GpuSpec().max_retries == INFRA_RETRY_FLOOR  # default tracks the runner's infra floor
    assert JobSpec.from_dict({}).gpu.max_retries == 5
    assert JobSpec.from_dict({"gpu": {}}).gpu.max_retries == 5
    # An explicit value still wins (the default is only the fallback).
    assert JobSpec.from_dict({"gpu": {"max_retries": 3}}).gpu.max_retries == 3


def test_cheapest_gpu_picks_cheapest_validated_runpod_class():
    """cheapest_gpu (the RunPod-static, parse-time provisional) picks the cheapest VALIDATED
    RunPod-provisionable class that fits, matching what the RunPod allocator path provisions."""
    from flash.providers.core.base import cheapest_gpu

    assert cheapest_gpu(24) == "RTX 4090"  # cheapest validated RunPod class that fits 24 GB
    assert cheapest_gpu(80) == "A100 PCIe"  # cheapest validated 80 GB RunPod class


def test_offline_allocates_static_cheapest():
    from flash.providers.core import allocator
    from flash.providers.core.base import cheapest_gpu

    # RunPod-only static rates: allocation matches cheapest_gpu.
    a = allocator.allocate("Qwen/Qwen3.5-9B", "grpo")
    assert a.provider == "runpod"
    assert a.gpu == cheapest_gpu(a.min_vram_gb)


def test_nothing_fits_names_constraint(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import UnsupportedGpuError

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 4096)
    with pytest.raises(UnsupportedGpuError, match="4096 GB"):
        allocator.allocate("Qwen/Qwen3.5-9B", "grpo")


def test_allocate_provider_constraint_never_falls_through(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate
    from flash.providers.core.registry import get_provider

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
        "Qwen/Qwen3.5-9B",
        "grpo",
        provider="lambda",
    )

    assert allocation.provider == "lambda"
    assert allocation.gpu == "A10"
    assert {candidate.provider for candidate in allocation.candidates} == {"lambda"}


def test_soft_provider_preference_ranks_ahead_of_cost_without_dropping_fallbacks(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate
    from flash.providers.core.registry import get_provider

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("vast", "runpod", "lambda"))
    monkeypatch.setattr(allocator, "_step_cost_ranker", lambda *a, **k: None)
    monkeypatch.setattr(
        get_provider("vast"),
        "live_candidates",
        lambda need, constraints: [Candidate("vast", "RTX 4090", 0.10, 24)],
    )
    monkeypatch.setattr(
        get_provider("runpod"),
        "live_candidates",
        lambda need, constraints: [Candidate("runpod", "RTX 4090", 3.00, 24)],
    )
    monkeypatch.setattr(
        get_provider("lambda"),
        "live_candidates",
        lambda need, constraints: [Candidate("lambda", "A10", 0.05, 24)],
    )

    allocation = allocator.allocate("Qwen/Qwen3.5-9B", "grpo", providers=("runpod", "vast"))

    assert allocation.provider == "runpod"
    assert [candidate.provider for candidate in allocation.candidates] == [
        "runpod",
        "vast",
        "lambda",
    ]


def test_soft_provider_preference_preserves_cost_order_within_one_rank(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate
    from flash.providers.core.registry import get_provider

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "lambda", "vast"))
    monkeypatch.setattr(allocator, "_step_cost_ranker", lambda *a, **k: None)
    monkeypatch.setattr(
        get_provider("runpod"),
        "live_candidates",
        lambda need, constraints: [Candidate("runpod", "RTX 4090", 3.00, 24)],
    )
    monkeypatch.setattr(
        get_provider("lambda"),
        "live_candidates",
        lambda need, constraints: [Candidate("lambda", "A10", 1.00, 24)],
    )
    monkeypatch.setattr(
        get_provider("vast"),
        "live_candidates",
        lambda need, constraints: [Candidate("vast", "RTX 4090", 0.10, 24)],
    )

    allocation = allocator.allocate("Qwen/Qwen3.5-9B", "grpo", providers=("runpod",))

    assert [candidate.provider for candidate in allocation.candidates] == [
        "runpod",
        "vast",
        "lambda",
    ]


def test_allocate_rejects_provider_pin_with_preferences():
    from flash.providers.core import allocator
    from flash.providers.core.base import UnsupportedGpuError

    with pytest.raises(UnsupportedGpuError, match="provider and providers cannot both be set"):
        allocator.allocate(
            "Qwen/Qwen3.5-9B",
            "grpo",
            provider="runpod",
            providers=("vast",),
        )
    with pytest.raises(UnsupportedGpuError, match="must name at least one provider"):
        allocator.allocate("Qwen/Qwen3.5-9B", "grpo", providers=[])


def test_allocate_rejects_unconfigured_provider(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import UnsupportedGpuError

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    with pytest.raises(UnsupportedGpuError, match="not configured"):
        allocator.allocate("Qwen/Qwen3.5-9B", "grpo", provider="lambda")


def test_allocate_gpu_type_never_widens_or_escalates(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate
    from flash.providers.core.registry import get_provider

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
        "Qwen/Qwen3.5-9B",
        "grpo",
        gpu_type="h100",
    )

    assert allocation.gpu == "H100"
    assert {candidate.gpu for candidate in allocation.candidates} == {"H100"}


def test_allocate_gpu_type_fallbacks_widen_the_search_without_dictating_the_winner(monkeypatch):
    """An ordered pin restricts allocation to the named classes and no others, which is what gives a
    pinned run somewhere to go when its first class is out of capacity. Order is preference, not
    priority: the survivors still compete on cost, so naming a class first does not make the run pay
    more for it."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate
    from flash.providers.core.registry import get_provider

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    monkeypatch.setattr(
        get_provider("runpod"),
        "live_candidates",
        lambda need, constraints: [
            Candidate("runpod", "RTX 4090", 0.50, 24),
            Candidate("runpod", "A100 PCIe", 1.19, 80),
            Candidate("runpod", "H100", 3.29, 80),
        ],
    )

    allocation = allocator.allocate(
        "Qwen/Qwen3.5-9B",
        "grpo",
        gpu_type="H100",
        gpu_type_fallbacks=("A100 PCIe",),
    )

    # only the named classes survive; rtx 4090 is offered and fitting but was not asked for.
    assert {candidate.gpu for candidate in allocation.candidates} == {"H100", "A100 PCIe"}
    # and the cheaper of the two wins despite h100 being named first.
    assert allocation.gpu == "A100 PCIe"


def test_allocate_rejects_an_unsatisfiable_fallback(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, UnsupportedGpuError
    from flash.providers.core.registry import get_provider

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 80)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    monkeypatch.setattr(
        get_provider("runpod"),
        "live_candidates",
        lambda need, constraints: [Candidate("runpod", "H100", 3.29, 80)],
    )

    with pytest.raises(UnsupportedGpuError, match=r"RTX 4090.*requires at least 80 GB"):
        allocator.allocate(
            "Qwen/Qwen3.5-9B",
            "grpo",
            gpu_type="H100",
            gpu_type_fallbacks=("RTX 4090",),
            max_gpu_count=1,
        )


def test_allocate_ordered_lambda_pin_classifies_impossible_shapes_as_unsupported(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, UnsupportedGpuError
    from flash.providers.core.registry import get_provider

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("lambda",))

    def only_b200(_need, constraints):
        if constraints.gpu_type:
            raise UnsupportedGpuError(
                f"lambda does not offer a rentable {constraints.gpu_type} shape up to 4 cards"
            )
        return [Candidate("lambda", "B200", 5.00, 180, gpu_count=4)]

    monkeypatch.setattr(get_provider("lambda"), "live_candidates", only_b200)

    for fallbacks in ((), ("H100",)):
        with pytest.raises(UnsupportedGpuError, match="lambda does not offer"):
            allocator.allocate(
                "Qwen/Qwen3.5-9B",
                "grpo",
                provider="lambda",
                gpu_type="A10",
                gpu_type_fallbacks=fallbacks,
                max_gpu_count=4,
            )


def test_allocate_gpu_type_enforces_vram_and_provider_support(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import UnsupportedGpuError

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 80)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "lambda"))
    with pytest.raises(UnsupportedGpuError, match=r"requires at least 80 GB.*--gpus"):
        allocator.allocate(
            "Qwen/Qwen3.5-9B",
            "grpo",
            gpu_type="RTX 4090",
            max_gpu_count=1,
        )
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    with pytest.raises(UnsupportedGpuError, match="cannot provision"):
        allocator.allocate(
            "Qwen/Qwen3.5-9B",
            "grpo",
            provider="lambda",
            gpu_type="RTX 4090",
        )


@pytest.mark.parametrize("provider", ["lambda", "vast"])
def test_exact_dynamic_provider_empty_capacity_is_retryable(monkeypatch, provider):
    from flash.providers.core import allocator
    from flash.providers.core.base import CapacityLookupError
    from flash.providers.core.registry import get_provider

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: (provider,))
    monkeypatch.setattr(get_provider(provider), "live_candidates", lambda need, constraints: [])

    with pytest.raises(CapacityLookupError, match="currently has no capacity"):
        allocator.allocate(
            "Qwen/Qwen3.5-9B",
            "grpo",
            provider=provider,
            gpu_type="H100",
        )


@pytest.mark.parametrize("gpu_type", ["", "H100"])
def test_sft_width_that_never_fits_is_terminal_not_a_capacity_retry(monkeypatch, gpu_type):
    """Regression: a width the fit filter always rejects was reported as sold-out capacity.

    The filter credits only the ranks an sft run launches, and an unpacked batch of 1 launches ONE
    however many cards are rented. When that empties the candidate set, the classification has to
    agree: it asked whether any offered class fits at the RENTED count, so it still found a shape,
    called the miss retryable, and `_allocate_attempt` turned it into a `poll_error`. On a
    Lambda/Vast-only fleet that re-polls a live market for capacity that cannot help -- no lookup
    makes a batch of 1 use more ranks -- burning the infra budget instead of failing with the reason.

    Parametrized over both branches because a pin and an unpinned search classify separately.
    """
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, CapacityLookupError, UnsupportedGpuError
    from flash.providers.core.registry import get_provider

    # 200 GB does not fit one H100 card, and the clamp means one card is all that ever launches.
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 200)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("lambda",))
    # capacity is HEALTHY: the provider offers the shape, so nothing here is a stock or outage
    # artifact. only the executed-width clamp removes it, which is exactly the terminal case.
    monkeypatch.setattr(
        get_provider("lambda"),
        "live_candidates",
        lambda need, constraints: [
            Candidate(provider="lambda", gpu="H100", hourly_usd=2.0, vram_gb=80, gpu_count=n)
            for n in (1, 2, 4, 8)
        ],
    )

    with pytest.raises(UnsupportedGpuError) as ei:
        allocator.allocate(
            "Qwen/Qwen3.5-9B",
            "sft",
            train={"batch_size": 1},
            gpu_type=gpu_type,
            max_gpu_count=8,
        )
    assert not isinstance(ei.value, CapacityLookupError), (
        "an unpacked sft run launches one rank, so no market lookup can widen it -- classifying "
        "this as capacity makes the runner retry a deterministic rejection until the budget dies"
    )


def test_lookup_blip_is_only_retryable_when_a_launchable_shape_exists(monkeypatch):
    """Regression: the lookup-failure branch returned retryable before consulting executed width.

    A live-capacity blip is retryable because the outage may be HIDING a shape that fits. When the
    run's executed width fits no advertised class, the outage is not what stands in the way and a
    retry cannot help -- `_allocate_attempt` turns it into `poll_error` and burns the infra budget on
    a deterministic miss. The advertised class list is static and readable during the outage, so this
    is decidable exactly when it matters.

    Both halves are asserted: the blip must STILL be retryable when a launchable shape does exist, or
    this guard would trade a retry bug for an outage that kills every run.
    """
    from flash.providers.core import allocator
    from flash.providers.core.base import CapacityLookupError, UnsupportedGpuError
    from flash.providers.core.registry import get_provider

    def _blip(need, constraints):
        raise CapacityLookupError("lambda live capacity lookup failed")

    monkeypatch.setattr(allocator, "available_providers", lambda: ("lambda",))
    monkeypatch.setattr(get_provider("lambda"), "live_candidates", _blip)

    # 200 GB against an sft run clamped to one rank: no advertised class holds it at any count.
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 200)
    with pytest.raises(UnsupportedGpuError) as ei:
        allocator.allocate("Qwen/Qwen3.5-9B", "sft", train={"batch_size": 1}, max_gpu_count=8)
    assert not isinstance(ei.value, CapacityLookupError), (
        "the blip did not cause this and cannot cure it -- no lookup makes a batch of 1 use more "
        "ranks, so retrying spends the infra budget on a shape that will never exist"
    )

    # same blip, same provider, but a need one card CAN hold: still retryable, as before.
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    with pytest.raises(CapacityLookupError):
        allocator.allocate("Qwen/Qwen3.5-9B", "sft", train={"batch_size": 1}, max_gpu_count=8)


def test_exact_runpod_empty_capacity_stays_terminal(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import UnsupportedGpuError
    from flash.providers.core.registry import get_provider

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 24)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    monkeypatch.setattr(get_provider("runpod"), "live_candidates", lambda need, constraints: [])

    with pytest.raises(UnsupportedGpuError, match="no allocatable capacity"):
        allocator.allocate(
            "Qwen/Qwen3.5-9B",
            "grpo",
            provider="runpod",
            gpu_type="H100",
        )


def test_allocate_gpu_type_ignores_ineligible_provider_blip(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import CapacityLookupError, UnsupportedGpuError
    from flash.providers.core.registry import get_provider

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
            "Qwen/Qwen3.5-9B",
            "grpo",
            gpu_type="H200",
        )

    assert not isinstance(exc_info.value, CapacityLookupError)
    assert calls == ["runpod"]


def _raise_capacity_blip(*a, **k):
    from flash.providers.core.base import CapacityLookupError

    raise CapacityLookupError("vast live capacity lookup failed") from RuntimeError("market blip")


def _stub_alloc(monkeypatch, *, runpod, lambda_, vast):
    from flash.providers.core import allocator
    from flash.providers.core.registry import get_provider

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
    from flash.providers.core import allocator
    from flash.providers.core.base import CapacityLookupError, UnsupportedGpuError

    _stub_alloc(
        monkeypatch,
        runpod=lambda need: [],  # no RunPod class fits
        lambda_=lambda need: [],
        vast=_raise_capacity_blip,  # Vast (the only possible source) blipped
    )
    with pytest.raises(CapacityLookupError) as ei:
        allocator.allocate("Qwen/Qwen3.5-9B", "grpo")
    assert not isinstance(ei.value, UnsupportedGpuError)


def test_capacity_blip_degrades_to_fitting_provider(monkeypatch):
    """A Vast blip must NOT abort allocation when another provider has a fitting class — degrade to it."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate

    _stub_alloc(
        monkeypatch,
        runpod=lambda need: [Candidate("runpod", "RTX 4090", 0.69, 24)],
        lambda_=lambda need: [],
        vast=_raise_capacity_blip,
    )
    a = allocator.allocate("Qwen/Qwen3.5-9B", "grpo")
    assert a.provider == "runpod"  # degraded past the blip, no error


def test_genuine_no_fit_without_blip_stays_terminal(monkeypatch):
    """No blip, just nothing fits -> terminal UnsupportedGpuError (unchanged contract)."""
    from flash.providers.core import allocator
    from flash.providers.core.base import UnsupportedGpuError

    _stub_alloc(
        monkeypatch,
        runpod=lambda need: [],
        lambda_=lambda need: [],
        vast=lambda need, disk_gb=0.0, max_wall_seconds=0.0: [],
    )
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate("Qwen/Qwen3.5-9B", "grpo")


def test_estimator_matches_measured_seq_boundaries():
    """The raw VRAM physics reproduces the MEASURED RunPod capacity sweep: each anchor
    is a real train/OOM boundary observed on a pinned card (the calibration ground truth).
    estimate_vram_gb is the accurate estimate; model_required adds the safety headroom."""
    from flash.engine.plan.vram import estimate_vram_gb as e

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


def test_required_vram_sizes_weights_from_curated_params_b_not_display_string():
    """model_required_vram_gb must size the resident WEIGHT term from the curated
    ``ModelInfo.params_b`` (the single source of truth resolve_params_b / the cost model read).
    params_b is now a required numeric field and the ``params`` display string is display-only
    (params_b_from_str was removed): re-parsing the string was fragile for an MoE whose string lists
    BOTH counts ("35B total / ~3B active") — the first parsed token could be the ~3B active count,
    sizing the ~70 GB resident weights ~10x too small and under-provisioning the card."""
    from flash.core.catalog import MODELS, ModelInfo
    from flash.engine.plan.vram import model_required_vram_gb

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
    from flash.engine.plan import vram
    from flash.engine.plan.vram import estimate_vram_gb as e

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
    from flash.engine.plan.vram import estimate_vram_gb as e

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

    Regression (vram.py): a sizing branch hardcoded ``_need(params_b, 'grpo', ...)``, so
    an OPD run was sized as a colocated-vLLM GRPO job -- rejecting fitting runs or routing them to
    pricier GPUs. The real algorithm must reach the estimator, so the two diverge.
    """
    from flash.engine.plan.vram import model_required_vram_gb

    train = {"max_context_tokens": 8192, "max_completion_tokens": 8192, "lora_rank": 16}
    for model_id in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B"):
        opd_need = model_required_vram_gb(model_id, "opd", train=train)
        grpo_need = model_required_vram_gb(model_id, "grpo", train=train)
        assert opd_need != grpo_need, f"{model_id} OPD must not size as the GRPO colocate path"
        assert opd_need > 0


def test_opd_sizes_on_the_authored_prompts_per_step():
    """The rl optimizer batch is named ``prompts_per_step``, and sizing must read THAT key.

    Regression: the sizer only ever read ``batch_size``. Once the rl batch was split out under its
    own name, an rl spec's ``batch_size`` was always None, so every authored ``prompts_per_step``
    fell through to the recipe default -- a wide batch sized as if it were the default one and
    under-provisioned the card it was then rented on. Sizing has to move with the knob, on the
    TrainSpec the submit path passes and on the raw dict the parse gate passes.
    """
    from flash.engine.plan.vram import model_required_vram_gb
    from flash.schema import spec_from_dict

    def _train(**over):
        return {
            "epochs": 1,
            "group_size": 1,
            "max_context_tokens": 1536,
            "max_completion_tokens": 512,
            "lora_rank": 16,
            **over,
        }

    narrow = model_required_vram_gb("Qwen/Qwen3.5-9B", "opd", train=_train(prompts_per_step=1))
    wide = model_required_vram_gb("Qwen/Qwen3.5-9B", "opd", train=_train(prompts_per_step=32))
    assert wide > narrow, "an authored prompts_per_step must move the opd vram floor"

    # the submit path allocates from the parsed TrainSpec, not the raw dict, and that object carries
    # the batch ONLY under prompts_per_step -- so it has to size identically to the dict above.
    def _spec_need(pps):
        spec = spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "opd",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "gpu": {},
                "train": _train(prompts_per_step=pps),
            },
            run_id="pps",
        )
        return model_required_vram_gb(spec.model, "opd", train=spec.train)

    assert _spec_need(1) == narrow
    assert _spec_need(32) == wide


def test_vram_headroom_consistent_across_sizing_paths():
    """provisional_gpu (parse-time) and required_vram_gb (submit-time) must size with the SAME
    headroom (a validated constant), so they never disagree."""
    from flash.providers.core import allocator

    assert allocator.vram_headroom() == 1.1
    # both paths feed model_required_vram_gb the same headroom -> identical sizing
    a_need = allocator.required_vram_gb(
        "Qwen/Qwen3.5-9B", "grpo", train={"max_context_tokens": 4096}
    )
    from flash.engine.plan.vram import model_required_vram_gb

    direct = model_required_vram_gb(
        "Qwen/Qwen3.5-9B", "grpo", train={"max_context_tokens": 4096}, headroom=1.1
    )
    assert a_need == direct


def test_allocate_never_selects_below_matrix_need():
    """The core anti-OOM invariant: the GPU the allocator picks ALWAYS has >= the matrix's
    required VRAM, across a sweep of model x algo x seq x group x batch. If this ever fails,
    auto-allocation could provision a too-small card and OOM a paid worker."""
    from flash.providers.core.allocator import allocate, required_vram_gb
    from flash.providers.core.base import get_gpu_info

    grid = [
        ("Qwen/Qwen3.5-9B", "grpo", {"max_context_tokens": 1024, "group_size": 4}),
        ("Qwen/Qwen3.5-9B", "grpo", {"max_context_tokens": 32768, "group_size": 8}),
        # chunked-nll qwen sft cases across short and long contexts.
        ("Qwen/Qwen3.5-9B", "sft", {"max_context_tokens": 1024}),
        ("Qwen/Qwen3.5-9B", "sft", {"max_context_tokens": 1536}),
        ("Qwen/Qwen3.5-9B", "sft", {"max_context_tokens": 8192}),
        ("Qwen/Qwen3.5-9B", "grpo", {"max_context_tokens": 1024, "group_size": 4}),
        (
            "Qwen/Qwen3.5-9B",
            "grpo",
            {"max_context_tokens": 16384, "max_completion_tokens": 4096, "group_size": 8},
        ),
        ("Qwen/Qwen3.5-9B", "sft", {"max_context_tokens": 32768}),
        ("Qwen/Qwen3.5-9B", "grpo", {"max_context_tokens": 8192, "group_size": 8}),
    ]
    for model, algo, tr in grid:
        need = required_vram_gb(model, algo, train=tr)
        alloc = allocate(model, algo, train=tr)
        assert get_gpu_info(alloc.gpu).vram_gb >= need, (model, algo, tr, alloc.gpu, need)


def test_opd_catalog_model_config_gpu_matrix_routes_to_fitting_cards(monkeypatch):
    """opd configs auto-size unpinned shapes while exact type pins keep single-card validation."""
    from flash.core.catalog import MODELS
    from flash.cost import RunConfig, estimate_cost
    from flash.providers.core import allocator
    from flash.providers.core.allocator import required_vram_gb
    from flash.providers.core.base import (
        GPU_INFO,
        UnsupportedGpuError,
        get_gpu_info,
        providers_for,
        provisional_gpu,
        provisional_gpu_count,
    )
    from flash.providers.core.sharding import (
        MAX_COMBINATION_CARDS,
        combined_vram_gb,
    )
    from flash.schema import ConfigError, spec_from_dict

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    configured_gpu_types = tuple(name for name, gpu_info in GPU_INFO.items() if gpu_info.validated)
    configs = {
        # Matches the failed continuation shape from the OPD/vLLM RTX 5090 report.
        "observed_2b_128tok_r32": {
            "epochs": 1,
            "max_completion_tokens": 128,
            "lora_rank": 32,
        },
        "recipe_default": {"epochs": 1},
        # opd takes the optimizer batch as `prompts_per_step`; `batch_size` is sft-only and is
        # rejected outright, so spelling it here would fail the parse rather than size a shape.
        "opd_prompt_batch": {
            "epochs": 1,
            "prompts_per_step": 8,
            "group_size": 1,
            "max_context_tokens": 1536,
            "max_completion_tokens": 512,
            "lora_rank": 16,
        },
        "longer_context": {
            "epochs": 1,
            "prompts_per_step": 8,
            "group_size": 1,
            "max_context_tokens": 4096,
            "max_completion_tokens": 512,
            "lora_rank": 16,
        },
        # this case varies the COMPLETION length, so its batch only has to be wide enough not to
        # bound the rank count: one prompt in a group of one is a single sequence, which launches one
        # rank however many cards are rented (see `rl_data_parallel_cards`) and would make this a test
        # of the width clamp rather than of long completions.
        "longer_completion": {
            "epochs": 1,
            "prompts_per_step": 8,
            "group_size": 1,
            "max_context_tokens": 4096,
            "max_completion_tokens": 2048,
            "lora_rank": 16,
        },
        "wide_rollout_batch": {
            "epochs": 1,
            "prompts_per_step": 8,
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
                batch_size=train.get("prompts_per_step"),
                group_size=train.get("group_size"),
                lora_rank=train.get("lora_rank"),
                provider="runpod",
            )

            auto_count = provisional_gpu_count(model_id, "opd", train=train)
            auto_cap = allocator.geometry_safe_gpu_cap(model_id, MAX_COMBINATION_CARDS)
            max_auto_vram = max(
                combined_vram_gb(gpu.vram_gb, auto_cap)
                for gpu in GPU_INFO.values()
                if gpu.enum_member and gpu.validated
            )
            if need > max_auto_vram:
                with pytest.raises(UnsupportedGpuError):
                    allocator.allocate(model_id, "opd", train=train)
                with pytest.raises(ValueError, match="opd needs"):
                    estimate_cost(rc)
                rejected.add((model_id, label))
                continue

            preview_gpu = provisional_gpu(model_id, "opd", train=train)
            preview_info = get_gpu_info(preview_gpu)
            assert preview_info.validated
            assert combined_vram_gb(preview_info.vram_gb, auto_count) >= need, (
                model_id,
                label,
                preview_gpu,
                auto_count,
                need,
            )

            alloc = allocator.allocate(model_id, "opd", train=train)
            alloc_info = get_gpu_info(alloc.gpu)
            assert alloc.provider == "runpod"
            assert alloc.min_vram_gb == need
            assert alloc.gpu == preview_gpu
            assert alloc.gpu_count == auto_count
            assert alloc_info.validated
            assert combined_vram_gb(alloc_info.vram_gb, alloc.gpu_count) >= need
            assert all(
                combined_vram_gb(candidate.vram_gb, candidate.gpu_count) >= need
                for candidate in alloc.candidates
            )

            estimate = estimate_cost(rc)
            assert estimate.required_vram_gb == need
            assert estimate.gpu == alloc.gpu
            assert estimate.gpu_count == alloc.gpu_count
            assert combined_vram_gb(estimate.gpu_vram_gb, estimate.gpu_count) >= need

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


def test_catalog_model_algorithm_gpu_matrix_routes_to_fitting_cards(monkeypatch):
    """Full catalog matrix guard: every supported model x algorithm route must pick a card that
    meets the shared VRAM requirement across schema preview, submit allocation, and cost estimate."""
    from flash.core.catalog import ALGORITHMS, MODELS
    from flash.cost import RunConfig, estimate_cost
    from flash.providers.core import allocator
    from flash.providers.core.base import (
        GPU_INFO,
        get_gpu_info,
        provisional_gpu,
    )
    from flash.providers.core.sharding import combined_vram_gb

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
    from flash.core.catalog import ALGORITHMS, MODELS
    from flash.providers.core import allocator
    from flash.providers.core.base import GPU_INFO, get_gpu_info, providers_for
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
                    # a type pin with no authored count keeps the historical one-card constraint, so
                    # its own shortfall is always the actionable error even when every class is small.
                    with pytest.raises(ConfigError, match="requires at least"):
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
    from flash.engine.plan import vram
    from flash.engine.plan.vram import estimate_vram_gb as e

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
    from flash.engine.plan import vram

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


def test_sft_equation_covers_honest_peak_across_seq_boundary():
    """the allocator must mirror chunked qwen and plain-nll fallback peaks across the catalog."""
    import math

    from flash.core.catalog import MODELS, vocab_size_for
    from flash.engine.plan import vram
    from flash.engine.plan.vram import sft_chunked_nll_enabled, sft_per_device
    from flash.providers.core.allocator import required_vram_gb
    from flash.providers.core.base import GPU_INFO

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
    from flash.engine.plan.vram import (
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
    from flash.engine.plan.vram import sft_grad_accum, sft_per_device

    # Small vocab (e.g. ~32k): the [pd, seq, vocab] logits are tiny -> no cap, keep 4.
    assert sft_per_device(8, seq_len=1024, vocab=32_000, fused=False) == 4
    # Fused CE on (the worker fuses for >=3B model OR >=2048 ctx): logits never materialize -> 4.
    assert sft_per_device(8, seq_len=1024, vocab=248_320, fused=True) == 4
    # Default call (no seq/vocab/fused) is the old fixed behavior, so existing callers are unchanged.
    assert sft_grad_accum(8) == (4, 2)


def test_sft_chunked_nll_model_gate_mirrors_worker():
    from flash.engine.plan.vram import sft_chunked_nll_enabled

    assert sft_chunked_nll_enabled("Qwen/Qwen3.5-9B") is True
    assert sft_chunked_nll_enabled("Qwen/Qwen3.8-27B") is True
    assert sft_chunked_nll_enabled("Qwen/Qwen3.6-35B-A3B") is True
    assert sft_chunked_nll_enabled("meta-llama/Llama-3.2-1B") is False
    assert sft_chunked_nll_enabled("org/unknown") is False


def test_every_sft_catalog_model_is_sized_for_the_fused_loss():
    """sizing must mirror the worker, which sets use_fused_kernels=true for EVERY model.

    the enumerated gate above cannot fail when a NEW catalog model is added and left out of the
    set, which is exactly how Qwen3.8-27B came to be sized for dense logits it never allocates.
    every catalog model is a qwen3_5/qwen3_5_moe checkpoint, and verl dispatches both to the fused
    torch backend, so the sft-capable catalog and the set must stay identical.
    """
    from flash.core.catalog import MODELS
    from flash.engine.plan.vram import sft_chunked_nll_enabled

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
    from flash.engine.plan.vram import _LOGITS_BUDGET_GB
    from flash.engine.plan.vram import estimate_vram_gb as e

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
    # the allocator's Vast capacity search must use the SAME effective disk floor (max(disk_gb,
    # MIN_DISK_GB)) the submit path provisions with — else a high-disk run is advertised Vast
    # capacity that only exists at the 60 GB floor and then can't actually rent (an impossible
    # attempt a max_retries=0 run never escapes).
    from flash.providers.core.base import AllocationConstraints
    from flash.providers.core.registry import get_provider
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
    # the allocator's Vast capacity search must thread the run's wall cap so usable_offers applies
    # the duration floor — else the allocator advertises Vast classes whose only live offers expire
    # before the run finishes (fatal for a max_retries=0 run).
    from flash.providers.core.base import AllocationConstraints
    from flash.providers.core.registry import get_provider
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

    from flash.providers.core.base import Candidate, rentable_gpu_counts

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
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate

    cands = [
        Candidate(provider="runpod", gpu="A100 PCIe", hourly_usd=1.5, vram_gb=80),
        Candidate(provider="runpod", gpu="H200", hourly_usd=4.0, vram_gb=141),
    ]
    _stub_provider(monkeypatch, allocator, cands)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 100)
    a = allocator.allocate("m", "sft")
    assert (a.gpu, a.gpu_count) == ("H200", 1)  # only class fitting 100 GB alone


def test_unset_count_auto_sizes_the_27b_grpo_run_to_two_cards(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate

    cands = [
        Candidate(provider="runpod", gpu="H200", hourly_usd=4.39, vram_gb=141),
        Candidate(provider="runpod", gpu="B200", hourly_usd=5.89, vram_gb=180),
    ]
    _stub_provider(monkeypatch, allocator, cands)
    allocation = allocator.allocate(
        "Qwen/Qwen3.8-27B",
        "grpo",
        train={"max_context_tokens": 8192, "max_completion_tokens": 4096},
    )
    assert allocation.gpu_count == 2
    assert all(candidate.gpu_count <= 2 for candidate in allocation.candidates)


def test_a_pinned_gpu_type_without_a_count_stays_one_card_in_allocate(monkeypatch):
    """`allocate()` is the THIRD boundary that decides this, and it decides independently.

    The parse gate and the offline quote already keep a pinned class at one card when no count was
    authored. `allocate()` resolved its own ceiling from `max_gpu_count`, so a direct call with a
    pinned type and no count auto-sized the pinned class: measured, a 24 GB RTX 4090 resolved to 8
    cards for an 80 GB run and would have rented them. Auto-sizing applies only when NEITHER the
    class nor the count is authored.
    """
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, UnsupportedGpuError

    cands = [Candidate(provider="runpod", gpu="RTX 4090", hourly_usd=0.69, vram_gb=24)]
    _stub_provider(monkeypatch, allocator, cands)
    with pytest.raises(UnsupportedGpuError) as exc:
        allocator.allocate("Qwen/Qwen3.5-9B", "grpo", gpu_type="RTX 4090")
    # the run is REJECTED rather than silently widened, and the wider shape is offered as an
    # opt-in remedy the author must name -- never taken on their behalf.
    message = str(exc.value)
    assert "requires at least 80 GB" in message
    assert "--gpus 8" in message


def test_every_boundary_reads_the_authored_ceiling_from_one_predicate():
    """The four sizing boundaries must not re-derive "did the author choose a shape?" themselves.

    This rule drifted three times in review because each boundary spelled it differently, and a
    test at one boundary cannot fail for a bug at another. `authored_gpu_ceiling` is now the single
    definition; this pins its truth table so a future edit to any one caller cannot quietly
    reintroduce a fourth dialect.
    """
    from flash.providers.core.base import authored_gpu_ceiling

    # nothing authored -> auto-size (the only case that may widen)
    assert authored_gpu_ceiling("", None) is None
    # a bare class pin is a ONE-CARD pin, never an invitation to widen
    assert authored_gpu_ceiling("RTX 4090", None) == 1
    # an authored count is a hard ceiling, with or without a class
    assert authored_gpu_ceiling("", 4) == 4
    assert authored_gpu_ceiling("RTX 4090", 2) == 2


def test_unset_count_quote_prices_the_auto_sized_shape():
    from flash.cost import RunConfig, estimate_cost

    estimate = estimate_cost(
        RunConfig(
            model_id="Qwen/Qwen3.8-27B",
            method="grpo",
            steps=1,
            seq_len=8192,
            completion_len=4096,
            gpu_count=None,
        )
    )
    assert estimate.gpu_count == 2
    assert estimate.required_vram_gb == 235


def test_explicit_two_card_pin_never_escalates_to_four(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, UnsupportedGpuError

    cands = [Candidate(provider="runpod", gpu="A100 PCIe", hourly_usd=1.39, vram_gb=80)]
    _stub_provider(monkeypatch, allocator, cands)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 200)

    with pytest.raises(UnsupportedGpuError) as exc:
        allocator.allocate(
            "Qwen/Qwen3.5-9B",
            "sft",
            gpu_type="A100 PCIe",
            max_gpu_count=2,
        )
    message = str(exc.value)
    # the pin is rejected AT the authored width and the remedy names the width that fits. wording
    # comes from `wider_shape_remedy`, the one searched helper every fit rejection routes through.
    assert "2-card combination" in message
    assert "--gpus 4" in message


def test_combo_two_cheap_cards_beat_one_expensive(monkeypatch):
    # 2 x A100 ($3.00 total, 160 GB * 0.85 = 136 GB effective) beats 1 x H200 ($4.00) for a 100 GB need.
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate

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
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate

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

    Counts are powers of two (the rollout engine shards heads over them: num_attention_heads %
    tp_size != 0 aborts at init), so on 80 GB cards with the replicated-floor model,
    usable = n*(80-8)*0.85 + 8:
    2 cards = 130.4 GB, 4 cards = 252.8 GB.

    Both needs are asserted because they pin different halves of the rule. 200 GB pins
    smallest-fitting-count (2 is too small, so 4). 140 GB pins the shard MARGIN itself: it sits in
    the gap between the discounted 2-card capacity (130.4) and the undiscounted one (152), so an
    allocator that forgot to discount would rent 2 cards and OOM on a run that needs 4.
    """
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate

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
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, UnsupportedGpuError

    cands = [Candidate(provider="runpod", gpu="TINY 8GB", hourly_usd=0.1, vram_gb=8)]
    _stub_provider(monkeypatch, allocator, cands)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 100)
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate("m", "sft", max_gpu_count=4)


def test_combo_summary_shows_count_and_total(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate

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
    from flash.engine.plan import vram

    assert vram.VERL_FUSED_CE_CHUNK_TOKENS == 512
    assert vram._SFT_CHUNKED_NLL_TOKENS == 512
    assert vram.OPD_CE_CHUNK_SIZE == 512


def test_sft_default_context_tracks_thinking_mode():
    """unauthored sft context must size at the length the worker trains on, per mode.

    ``sft_max_length`` trims rows to ``RECIPE.sft.max_seq_len_thinking`` when thinking is on, so a
    flat non-thinking default sized activations for half the real sequence.
    """
    from flash.engine.plan.recipe import RECIPE
    from flash.engine.plan.vram import model_required_vram_gb

    assert RECIPE.sft.max_seq_len_thinking > RECIPE.sft.max_seq_len
    mid = "Qwen/Qwen3.5-9B"
    plain = model_required_vram_gb(mid, "sft", thinking=False)
    thinking = model_required_vram_gb(mid, "sft", thinking=True)
    assert thinking > plain, (plain, thinking)


def test_pinned_gpu_fit_failure_names_the_card_count_that_fixes_it():
    """A pin that fails only at the authored ceiling must name the width that works.

    `[gpu] count` is a user-authored ceiling, so this rejection is one flag from success. Without
    the remedy the message states the shortfall and stops, and the user cannot tell an
    unsatisfiable run apart from one that fits on two cards -- the difference between abandoning
    the run and passing `--gpus 2`.
    """
    from flash.providers.core.allocator import _resolve_exact_gpu
    from flash.providers.core.base import GPU_INFO, UnsupportedGpuError

    need = GPU_INFO["H200"].vram_gb + 40  # fits on 2 cards, never on 1
    with pytest.raises(UnsupportedGpuError, match=r"--gpus 2") as single:
        _resolve_exact_gpu(
            "H200",
            need=need,
            cap=1,
            max_gpu_count=1,
            provider="",
            available=("runpod",),
            widest_cap=8,
        )
    assert "it fits on 2 cards" in str(single.value)

    # the multi-card rejection carries the same remedy, measured from ITS cap rather than 1.
    with pytest.raises(UnsupportedGpuError, match=r"--gpus 4") as pair:
        _resolve_exact_gpu(
            "H200",
            need=GPU_INFO["H200"].vram_gb * 3,
            cap=2,
            max_gpu_count=2,
            provider="",
            available=("runpod",),
            widest_cap=8,
        )
    assert "even as a 2-card combination" in str(pair.value)


def test_pinned_gpu_fit_failure_stays_a_dead_end_when_no_width_fits():
    """A genuinely unsatisfiable run must NOT be told to raise the ceiling.

    The remedy is only useful if it is true. Suggesting `--gpus 8` for a run no shape can hold
    would trade one dead end for a second, slower one -- so the suggestion is searched, not
    assumed, and absent when nothing fits.
    """
    from flash.providers.core.allocator import _resolve_exact_gpu
    from flash.providers.core.base import UnsupportedGpuError

    with pytest.raises(UnsupportedGpuError) as exc:
        _resolve_exact_gpu(
            "H200",
            need=100_000,
            cap=1,
            max_gpu_count=1,
            provider="",
            available=("runpod",),
            widest_cap=8,
        )
    assert "--gpus" not in str(exc.value)


def test_wider_shape_remedy_is_bounded_by_the_geometry_cap():
    """A suggested width must be one the model's head geometry allows.

    `geometry_safe_gpu_cap` exists because vllm rejects `num_attention_heads % tp_size != 0` at
    rollout-engine init -- AFTER the box is rented. A remedy that ignored it would bill the user for
    a box that cannot start. (This gate outlived Ulysses: sequence parallelism is pinned off on all
    three algorithms now, but grpo and opd still hand the rented width to the rollout engine as
    `tensor_model_parallel_size`, so head divisibility is still load-bearing.)
    """
    from flash.providers.core.base import GPU_INFO, wider_shape_remedy

    vram = GPU_INFO["H200"].vram_gb
    need = vram + 40
    assert "--gpus 2" in wider_shape_remedy((vram,), need, ceiling=8, above=1)
    # capped below the fitting width: report no remedy rather than an unrentable one.
    assert wider_shape_remedy((vram,), need, ceiling=1, above=1) == ""
    # `above` excludes widths already tried, so a remedy never restates the failing shape: at
    # above=2 the fitting width 2 is suppressed and the next one that fits is named instead.
    assert "--gpus 4" in wider_shape_remedy((vram,), need, ceiling=8, above=2)
    # ...and once no untried width fits, there is nothing to suggest.
    assert wider_shape_remedy((vram,), need, ceiling=8, above=8) == ""


def test_wider_shape_remedy_names_the_cheapest_fitting_width():
    """The remedy must name the smallest shape that works, not the widest on offer.

    Suggesting 8 cards for a run that fits on 2 would quadruple the bill to fix a fit error.
    """
    from flash.providers.core.base import GPU_INFO, wider_shape_remedy

    vram = GPU_INFO["H200"].vram_gb
    assert "--gpus 2" in wider_shape_remedy((vram,), vram + 40, ceiling=8, above=1)


def test_remedy_never_names_a_width_the_run_will_not_launch_on():
    """Regression: `--gpus N` was searched with the RENTED count, not the ranks that join.

    The fit gate credits only launched ranks, so an unpacked sft run (batch 1, one rank however many
    cards are rented) is rejected at every width. The remedy searched the same failure with the full
    count and answered `--gpus 2`: the user pays for a second card that contributes no memory and
    fails identically. Advice has to be proved with the rule that will judge the retry.

    Codex's shape exactly: a batch-1 4B at 32k needs 28 GB and does not fit a 24 GB card.
    """
    from flash.providers.core.base import wider_shape_remedy

    need, card = 28.0, 24
    assert (
        wider_shape_remedy((card,), need, ceiling=8, above=1, executed_width=lambda _n: 1) == ""
    ), (
        "an sft run pinned to one rank gains nothing from more cards, so no width is a remedy -- "
        "naming one sends the user to pay twice for the same failure"
    )
    # the default is unchanged for everything that DOES launch what it rents, so this cannot
    # silently suppress a real remedy on grpo/opd.
    assert "--gpus 2" in wider_shape_remedy((card,), need, ceiling=8, above=1)
    assert "--gpus 2" in wider_shape_remedy(
        (card,), need, ceiling=8, above=1, executed_width=lambda n: n
    )


def test_pin_rejection_names_the_width_it_actually_credited():
    """Regression: the message claimed a `cap`-card combination the VRAM math never tried.

    Once the precheck credits `launched(cap)`, saying "cannot fit even as an 8-card combination" for
    a run that launches one rank points the operator at the card ceiling -- they raise `--gpus` and
    hit the identical failure. The real limiter is the batch that bounds the rank count, so the
    message has to name the width it credited and why it is smaller than the one allowed.
    """
    from flash.providers.core.allocator import _resolve_exact_gpu
    from flash.providers.core.base import UnsupportedGpuError

    def reject(executed_width):
        with pytest.raises(UnsupportedGpuError) as ei:
            _resolve_exact_gpu(
                "H100",
                need=500.0,
                cap=8,
                max_gpu_count=8,
                provider="",
                available=("runpod",),
                widest_cap=8,
                executed_width=executed_width,
            )
        return str(ei.value)

    clamped = reject(lambda _n: 1)
    assert "1-card combination" in clamped, (
        "the math credited one rank, so claiming a wider combination was tried sends the operator "
        "to raise a ceiling that is not the limiter"
    )
    assert "8-card combination" not in clamped
    assert "only 1 joins this run" in clamped, "the message must say WHY the width is smaller"

    # a run that launches what it rents keeps the original wording, with no confusing aside.
    full = reject(lambda n: n)
    assert "8-card combination" in full
    # "join this run" also covers the singular "joins this run", which contains it.
    assert "join this run" not in full


def test_width_search_credits_only_the_ranks_that_join():
    """Regression: the shared width search credited rented cards on every consumer.

    `smallest_fitting_gpu_count` backs the auto-sized ceiling, the authored-ceiling check, and both
    catalog hints. Crediting cards that never join made all of them name a width that cannot hold
    the run -- an unpinned sft run with `gpu.count=1` was told to raise the ceiling to two, and the
    resubmission failed identically because `executed_width(2)` is still one.

    Fixing the shared helper rather than each caller is what makes the ceiling search, the pinned
    precheck, and the `--gpus N` advice answer one question.
    """
    from flash.providers.core.base import GPU_INFO, smallest_fitting_gpu_count

    need = GPU_INFO["H100"].vram_gb + 40.0  # fits on 2 rented cards, never on 1

    assert (
        smallest_fitting_gpu_count(
            need, max_gpu_count=8, gpu_names=("H100",), executed_width=lambda _n: 1
        )
        is None
    ), "a run clamped to one rank has no fitting width, so no ceiling can be promised"

    # unchanged for runs that launch what they rent, and unchanged when no rule is supplied at all.
    assert smallest_fitting_gpu_count(need, max_gpu_count=8, gpu_names=("H100",)) == 2
    assert (
        smallest_fitting_gpu_count(
            need, max_gpu_count=8, gpu_names=("H100",), executed_width=lambda n: n
        )
        == 2
    )


def test_width_search_finds_a_shape_the_executed_width_reaches_non_monotonically():
    """Crediting the executed width is not enough on its own: it must be VALUED as a rank count.

    Two things break a first-hit search once the width rule is applied. The executed width is not
    monotonic in the rented count -- sft over 3 rows with batch 3 launches 1 rank on 2 cards but 3
    on 4 -- so a search that stops at the first miss abandons a wider shape that does fit. And a
    rank count is not a rentable count, so valuing it through the rentable snap floors 3 ranks to
    2, under-crediting a combination by a whole card.

    Together they made a run that fits on 4 cards report that no width could hold it, which the
    caller turns into a terminal rejection of a job the allocator would have launched.
    """
    from flash.engine.plan.steps import sft_data_parallel_cards
    from flash.providers.core.base import (
        GPU_INFO,
        smallest_fitting_gpu_count,
    )
    from flash.providers.core.sharding import combined_vram_gb

    width = lambda count: sft_data_parallel_cards(count, 3, 3)  # noqa: E731
    assert (width(2), width(4)) == (1, 3), "premise: the executed width dips, then climbs"

    vram = GPU_INFO["H100"].vram_gb
    need = combined_vram_gb(vram, 2) + 1.0  # over 2 ranks, under 3
    assert combined_vram_gb(vram, 3) >= need > combined_vram_gb(vram, 2)

    assert (
        smallest_fitting_gpu_count(need, max_gpu_count=8, gpu_names=("H100",), executed_width=width)
        == 4
    ), "4 rented cards launch 3 ranks, which hold the run -- the search must not stop at 2"


def test_catalog_hint_is_withheld_when_the_width_would_not_launch():
    """The `--gpus N` catalog hint searched widths crediting rented cards, like the remedy did.

    `smallest_fitting_gpu_count` has no executed-width notion, so for a clamped sft run it names a
    count that buys nothing -- sending the user to ask a provider to confirm a SKU that cannot help.
    """
    from flash.providers.core.allocator import _resolve_exact_gpu
    from flash.providers.core.base import GPU_INFO, UnsupportedGpuError

    need = GPU_INFO["H100"].vram_gb + 40.0  # fits on 2 rented cards, never on 1

    def message(executed_width):
        with pytest.raises(UnsupportedGpuError) as ei:
            _resolve_exact_gpu(
                "H100",
                need=need,
                cap=1,
                max_gpu_count=1,
                provider="lambda",
                available=("lambda",),
                widest_cap=8,
                executed_width=executed_width,
            )
        return str(ei.value)

    assert "--gpus" not in message(lambda _n: 1), (
        "a clamped sft run stays at one rank, so asking lambda to confirm a 2-card SKU is a round "
        "trip that cannot fix the run"
    )
    # unchanged for runs that launch what they rent: the hint is real advice there.
    assert "--gpus 2" in message(lambda n: n)


def test_provider_incompatible_pin_reports_the_incompatibility_not_a_fit_remedy():
    """A class the pinned provider does not carry is a provider error at EVERY width.

    H200 is RunPod-only, so a Lambda-pinned H200 that also misses on VRAM must report the pairing.
    Checking fit first made the message name `--gpus 2` -- a flag that cannot help, because no
    Lambda H200 exists at any count. The user would raise the ceiling and fail identically.
    """
    from flash.providers.core.allocator import _resolve_exact_gpu
    from flash.providers.core.base import GPU_INFO, UnsupportedGpuError, providers_for

    assert "lambda" not in providers_for("H200")
    need = GPU_INFO["H200"].vram_gb + 40  # would fit on 2 cards, were they purchasable
    with pytest.raises(UnsupportedGpuError) as exc:
        _resolve_exact_gpu(
            "H200",
            need=need,
            cap=1,
            max_gpu_count=1,
            provider="lambda",
            available=("lambda",),
            widest_cap=8,
        )
    assert "cannot provision exact GPU 'H200'" in str(exc.value)
    assert "--gpus" not in str(exc.value)


def test_unpurchasable_width_names_a_remedy_the_user_can_actually_apply():
    """The remedy must depend on whether dropping the pin would actually change the pool.

    Three cases collapse to the same narrow ``available`` tuple and must not get the same advice:
    a pin over a fleet that also has RunPod (dropping it really does buy the wider shape), a pin
    over a Lambda-only fleet (dropping it lands on the identical pool and the identical rejection),
    and no pin at all (a knob the operator never set). ``available`` cannot separate them, so the
    pre-pin fleet has to travel with it.

    Driven through ``_resolved_gpu_count`` rather than the message builder directly: the pin
    context is LOST at that boundary, so a builder-only test would pass against the defect.
    """
    from flash.providers.core.allocator import _resolved_gpu_count
    from flash.providers.core.base import UnsupportedGpuError

    shared = {
        "need": 188.0,
        "requested_gpu_count": 1,
        "model_revision": "",
        "exact": "",
    }

    def reject(available, unpinned, requested=1, need=None):
        overrides = {"requested_gpu_count": requested}
        if need is not None:
            overrides["need"] = need
        with pytest.raises(UnsupportedGpuError) as exc:
            _resolved_gpu_count(
                "Qwen/Qwen3.6-35B-A3B",
                "grpo",
                available=available,
                unpinned=unpinned,
                **{**shared, **overrides},
            )
        return str(exc.value)

    # a pin worth dropping: the fleet behind it carries RunPod, which rents the width freely.
    droppable = reject(("lambda",), ("lambda", "runpod"))
    assert "Drop the provider pin" in droppable

    # reaching this branch means the authored ceiling ALSO failed, so a remedy naming only the
    # provider buys a second rejection that finally reveals `--gpus N`. both halves, one message.
    assert "`--gpus 2`" in droppable
    # and the advice has to actually work: following it verbatim allocates rather than re-rejecting.
    assert (
        _resolved_gpu_count(
            "Qwen/Qwen3.6-35B-A3B",
            "grpo",
            available=("lambda", "runpod"),
            unpinned=None,
            **{**shared, "requested_gpu_count": 2},
        )
        == 2
    )

    # the same pin over a LAMBDA-ONLY fleet. dropping it leaves the identical pool, so the advice
    # would send the user in a circle back to this exact error.
    futile = reject(("lambda",), ("lambda",))
    assert "Drop the provider pin" not in futile
    assert "configure a provider that rents card counts directly (RunPod)" in futile

    # nothing pinned: a narrow fleet is simply the whole configured fleet.
    unpinned = reject(("lambda", "vast"), None)
    assert "Drop the provider pin" not in unpinned
    assert "configure a provider that rents card counts directly (RunPod)" in unpinned

    # these two have no arbitrary-count provider behind them, so the width can only be OFFERED to
    # try against the provider's own catalog -- `live_capacity` means "confirm dynamically", not
    # "the wider SKU is absent". Lambda really does resolve gpu_4x_h100_pcie.
    for message in (futile, unpinned):
        assert "`--gpus 2`" in message
        assert "check it against their catalog" in message
        # so the message must not assert non-existence it cannot prove offline.
        assert "no available provider sells" not in message

    # a raise clause may only appear when the ceiling really does have to rise. the unpinned pool
    # can carry a BIGGER class the pin hid (Vast tops out at 80 GB/card, RunPod has H200/B200), and
    # then the same width fits and "--gpus 1" would name the ceiling the user already set.
    # 100 GB: two 80 GB Vast cards, but ONE RunPod H200/B200. the width does not rise.
    same_width = reject(("vast",), ("runpod", "vast"), need=100.0)
    assert "Drop the provider pin" in same_width
    assert "raise the card ceiling" not in same_width
    assert "--gpus" not in same_width
    # and the DESCRIPTION must agree with that remedy: the fallback is a bigger card at the same
    # count, so claiming the run "fits only on a multi-card shape" contradicts the very next
    # sentence. the obstacle is the pin hiding a large enough class, not the width.
    assert "fits only on a multi-card shape" not in same_width
    assert "at 1 card:" in same_width
    assert "the pinned provider's largest card is too small" in same_width

    # no message may claim the run exceeds every class -- it fits, it just cannot be bought.
    for message in (droppable, futile, unpinned):
        assert "more than any" not in message


def test_exact_gpu_rejection_reports_a_pin_that_hides_a_wider_count():
    """A pin can be the only reason an exact class has no offerable width.

    `_resolve_exact_gpu` narrows to providers carrying the class, and offers a `--gpus N` remedy
    only when one of them rents counts freely. A Lambda pin on a fleet that also has RunPod
    suppresses that entirely -- yet RunPod carries the very same H100 and rents it at any count, so
    dropping the pin is a real fix the bare shortfall message hides.
    """
    from flash.providers.core.allocator import _resolve_exact_gpu
    from flash.providers.core.base import UnsupportedGpuError

    def reject(provider, available, unpinned, need=188.0):
        with pytest.raises(UnsupportedGpuError) as exc:
            _resolve_exact_gpu(
                "H100",
                need=need,
                cap=1,
                max_gpu_count=1,
                provider=provider,
                available=available,
                unpinned=unpinned,
                widest_cap=8,
            )
        return str(exc.value)

    hidden = reject("lambda", ("lambda",), ("runpod", "lambda"))
    assert "Drop the provider pin" in hidden
    # the authored ceiling already failed too, so the pin clause alone would buy a second
    # rejection. both halves, as the sibling `vram_fit_error_message` path does.
    assert "`--gpus 4`" in hidden

    # an OVERSIZED pin gets no hint at all: 8x H100 is 497.6 GB, so dropping the pin cannot help
    # and promising it would be false. routed through `wider_shape_remedy`, which proves the width.
    oversized = reject("lambda", ("lambda",), ("runpod", "lambda"), need=900.0)
    assert "Drop the provider pin" not in oversized
    assert "--gpus" not in oversized

    # no pin, and a pin with nothing better behind it, must stay silent rather than invent advice.
    assert "Drop the provider pin" not in reject("", ("lambda",), None)
    assert "Drop the provider pin" not in reject("lambda", ("lambda",), ("lambda",))

    # where a width IS offerable the existing remedy stands; the hint must not displace it.
    runpod = reject("runpod", ("runpod",), None)
    assert "`--gpus 4`" in runpod
    assert "Drop the provider pin" not in runpod


def test_fit_remedy_is_withheld_when_only_fixed_count_sku_providers_remain():
    """A width is only PROMISED when a provider in play rents card counts freely.

    Lambda names the count in the instance type and Vast bakes it into the offer, so a wider shape
    exists only if their live catalog lists one -- unknowable here. RunPod takes the count as a
    launch parameter, so its remedy stays. The distinction is what the message CLAIMS: RunPod gets
    "it fits on N cards" because the fit was proved offline, while a fixed-count provider is only
    offered the width to ask its catalog for. Naming a width to check beats a bare shortfall the
    user cannot act on; asserting one exists would send them to buy a shape that may not be sold.
    """
    from flash.providers.core.allocator import _resolve_exact_gpu
    from flash.providers.core.base import GPU_INFO, UnsupportedGpuError, providers_for

    assert {"lambda", "vast"} <= set(providers_for("H100"))
    need = GPU_INFO["H100"].vram_gb + 40  # fits on 2 H100s

    with pytest.raises(UnsupportedGpuError) as lambda_only:
        _resolve_exact_gpu(
            "H100",
            need=need,
            cap=1,
            max_gpu_count=1,
            provider="lambda",
            available=("lambda",),
            widest_cap=8,
        )
    lambda_message = str(lambda_only.value)
    assert f"{GPU_INFO['H100'].vram_gb} GB VRAM" in lambda_message
    # the width may be named, but only ever as a catalog check -- never as a proved fit.
    assert "it fits on" not in lambda_message
    if "--gpus" in lambda_message:
        assert "check it against their catalog" in lambda_message

    # an oversized pin gets NO width at any tier: no rentable count would help, so the honest
    # answer is the shortfall alone rather than a catalog check that cannot succeed.
    with pytest.raises(UnsupportedGpuError) as oversized:
        _resolve_exact_gpu(
            "H100",
            need=float(GPU_INFO["H100"].vram_gb) * 100,
            cap=1,
            max_gpu_count=1,
            provider="lambda",
            available=("lambda",),
            widest_cap=8,
        )
    assert "--gpus" not in str(oversized.value)

    # the SAME pin on runpod keeps the remedy: withholding it everywhere would trade a wrong
    # suggestion for a missing one.
    with pytest.raises(UnsupportedGpuError, match=r"--gpus 2"):
        _resolve_exact_gpu(
            "H100",
            need=need,
            cap=1,
            max_gpu_count=1,
            provider="runpod",
            available=("runpod",),
            widest_cap=8,
        )

    # a mixed pool still counts as purchasable: runpod alone can sell the wider shape.
    with pytest.raises(UnsupportedGpuError, match=r"--gpus 2"):
        _resolve_exact_gpu(
            "H100",
            need=need,
            cap=1,
            max_gpu_count=1,
            provider="",
            available=("lambda", "runpod"),
            widest_cap=8,
        )


def test_exact_pin_on_a_fixed_count_provider_still_names_a_width_to_check():
    """The exact-GPU path must not withhold the catalog check its non-exact sibling gives.

    An exact Lambda H100 pin that needs more than one card previously died on a bare shortfall,
    while the same run without the exact pin was told which width to try. `live_capacity` means
    the count is confirmed dynamically, not that the SKU is absent -- Lambda resolves
    `gpu_4x_h100_pcie` against its own catalog and rejects what it does not sell with a precise
    error. Naming the width to try is one flag from working; the bare shortfall is a dead end.
    """
    from flash.providers.core.allocator import _resolve_exact_gpu
    from flash.providers.core.base import GPU_INFO, UnsupportedGpuError

    need = GPU_INFO["H100"].vram_gb * 2.35  # 188 GB: needs 4 cards, not 2

    with pytest.raises(UnsupportedGpuError) as pinned:
        _resolve_exact_gpu(
            "H100",
            need=need,
            cap=1,
            max_gpu_count=1,
            provider="lambda",
            available=("lambda",),
            widest_cap=8,
            unpinned=("lambda",),
        )
    message = str(pinned.value)
    # the width is SEARCHED, not guessed: 2 cards is 160 GB and does not fit, so 4 is the answer.
    assert "`--gpus 4`" in message
    assert "check it against their catalog" in message
    # still a catalog check, never a promise -- nothing here proved the SKU is purchasable.
    assert "it fits on" not in message

    # the same shortfall reached through the multi-card branch names a width above the tried cap.
    with pytest.raises(UnsupportedGpuError) as combo:
        _resolve_exact_gpu(
            "H100",
            need=need,
            cap=2,
            max_gpu_count=4,
            provider="lambda",
            available=("lambda",),
            widest_cap=8,
            unpinned=("lambda",),
        )
    assert "`--gpus 4`" in str(combo.value)


def test_the_obstacle_never_contradicts_the_remedy_printed_after_it():
    """Whenever dropping the pin is the fix, the diagnosis must blame the PIN.

    The message states an obstacle and then a remedy in adjacent sentences. Claiming "no available
    provider is confirmed to sell" this shape while the remedy offers to drop the pin and rent
    exactly that shape is a self-contradiction: the run is one or two flags from working, not
    unsellable. The fixed-count catalog wording is only true when nothing behind the pin fits.
    """
    from flash.providers.core.fit_errors import vram_fit_error_message

    def message(need, *, requested, widenable):
        return vram_fit_error_message(
            "sft",
            need,
            requested_gpu_count=requested,
            effective_gpu_count=requested,
            max_gpu_count=8,
            gpu_names=("H100",),
            providers=("vast",),
            widenable_without_pin=widenable,
        )

    # dropping the pin unlocks a WIDER freely-rented shape: blame the pin, not the market.
    wider = message(188.0, requested=1, widenable=("H200",))
    assert "Drop the provider pin" in wider
    assert "no available provider is confirmed to sell" not in wider
    assert "the pinned provider is not confirmed to sell" in wider

    # dropping the pin unlocks a bigger card at the SAME count: no width has to change.
    same = message(100.0, requested=1, widenable=("H200",))
    assert "Drop the provider pin" in same
    assert "at 1 card:" in same
    assert "fits only on a multi-card shape" not in same

    # nothing behind the pin fits: the catalog wording is now the honest diagnosis.
    catalog = message(188.0, requested=1, widenable=())
    assert "Drop the provider pin" not in catalog
    assert "fits only on a multi-card shape" in catalog

    # the invariant itself: a drop-pin remedy never rides a claim that the shape is unsellable
    # OUTRIGHT. the same-width branch does say "no available provider is confirmed to sell at 1
    # card", which is compatible with its remedy -- it is scoped to the authored count and the very
    # next clause names the excluded providers that do sell it. what must never appear alongside a
    # drop-pin remedy is the unscoped "exists only if their live catalog lists one".
    for msg in (wider, same, catalog):
        if "Drop the provider pin" in msg:
            assert "exists only if their live catalog lists one" not in msg


def test_the_fit_message_states_capacity_the_run_will_actually_have():
    """The shapes a fit rejection PRINTS must be valued at the width the search accepted them on.

    The width search credits launched ranks, but the two shapes quoted in the message were valued
    on rented cards, so the message argued against itself: a 2-card ceiling that launches one rank
    was reported as providing 234.1 GB against a 230 GB need -- more than it needs, printed as the
    reason for rejection -- and then recommended a 4-card shape credited 252.8 GB that really
    delivers 191.6. Every number a user reads has to be memory the run will actually have.
    """
    from flash.engine.plan.steps import sft_data_parallel_cards
    from flash.providers.core.fit_errors import vram_fit_error_message

    width = lambda count: sft_data_parallel_cards(count, 3, 3)  # noqa: E731

    def message(executed_width):
        return vram_fit_error_message(
            "sft",
            230,
            requested_gpu_count=2,
            effective_gpu_count=2,
            max_gpu_count=8,
            gpu_names=("H100", "H200"),
            executed_width=executed_width,
        )

    clamped = message(width)
    assert "provides at most 141 GB" in clamped, "2 rented cards launch 1 rank, so 141 GB is all"
    assert "234.1 GB" not in clamped, "the rented-card capacity must not be quoted anywhere"
    # the rejection has to READ like a rejection: a stated capacity above the stated need does not.
    assert "needs >= 230 GB" in clamped

    # unchanged for runs that launch what they rent, whether the rule is identity or absent.
    for control in (message(None), message(lambda count: count)):
        assert "provides at most 234.1 GB" in control
        assert "`--gpus 2` (2x H200 = 234.1 GB)" in control


def test_a_shape_label_reconciles_with_the_capacity_printed_beside_it():
    """A rented COUNT next to a launched CAPACITY states a false equation unless the gap is named.

    Valuing capacity at the executed width fixed the arithmetic but left the labels reading as
    though the hardware were smaller: `2x H200` beside 141 GB says an H200 is a 70 GB card, and an
    `8-card combination` capped at 446.6 GB understates the class without saying why. A user who
    cannot reconcile the two numbers reaches for `--gpus`, which is the one knob that cannot help
    when the batch is the limiter -- so the label has to name the join count and the reason.
    """
    from flash.engine.plan.steps import sft_data_parallel_cards
    from flash.providers.core.fit_errors import vram_fit_error_message

    width = lambda count: sft_data_parallel_cards(count, 3, 3)  # noqa: E731
    reason = "sft shards by data, so the batch and retained rows bound the rank count"

    pinned = vram_fit_error_message(
        "sft",
        230,
        requested_gpu_count=2,
        effective_gpu_count=2,
        max_gpu_count=8,
        gpu_names=("H100", "H200"),
        executed_width=width,
    )
    # both shapes carry their join count, and the reason is stated once rather than per shape.
    assert "(2x H200, 1 of which joins this run)" in pinned
    assert "(4x H200 = 347.15 GB, 3 of which join this run)" in pinned
    assert pinned.count(reason) == 1

    terminal = vram_fit_error_message(
        "sft",
        9000,
        requested_gpu_count=8,
        effective_gpu_count=8,
        max_gpu_count=8,
        gpu_names=None,
        executed_width=width,
    )
    # the count attaches to the CARDS, never to the GB figure ("446.6 GB max, 3 of which join").
    assert "8-card validated GPU combination, 3 of which join this run (446.6 GB max)" in terminal
    assert terminal.count(reason) == 1

    # a run that launches every card it rents is never told about ranks at all.
    for control in (
        vram_fit_error_message(
            "sft",
            230,
            requested_gpu_count=2,
            effective_gpu_count=2,
            max_gpu_count=8,
            gpu_names=("H100", "H200"),
        ),
        vram_fit_error_message(
            "grpo",
            500,
            requested_gpu_count=4,
            effective_gpu_count=4,
            max_gpu_count=8,
            gpu_names=("H100",),
        ),
    ):
        assert "of which" not in control
        assert reason not in control


def test_pinned_class_names_the_bounding_knob_of_the_algorithm_it_rejected():
    """A pinned small-batch grpo/opd run must not be told that sft's rows bound its ranks.

    `_resolve_exact_gpu` spelled this reason itself and hardcoded "sft", so once the width clamp let
    an rl run narrow below its ceiling, a pinned rl rejection pointed at `batch_size` -- which opd
    rejects at parse time -- and at retained rows, which rl has no concept of. The reason now comes
    from the one shared formatter, so the two cannot drift apart again.
    """
    from flash.providers.core.allocator import _executed_width, _resolve_exact_gpu
    from flash.providers.core.base import UnsupportedGpuError

    def reject(algorithm, train):
        with pytest.raises(UnsupportedGpuError) as exc:
            _resolve_exact_gpu(
                "H100",
                need=400.0,
                cap=8,
                max_gpu_count=8,
                provider="",
                available=("runpod",),
                widest_cap=8,
                executed_width=_executed_width(algorithm, train, None),
                algorithm=algorithm,
            )
        return str(exc.value)

    rl = reject("opd", {"prompts_per_step": 1, "group_size": 1})
    assert "only 1 joins this run" in rl
    assert "prompts_per_step x group_size bounds the rank count" in rl
    assert "batch and retained rows" not in rl

    sft = reject("sft", {"batch_size": 1})
    assert "sft shards by data, so the batch and retained rows bound the rank count" in sft
    assert "prompts_per_step" not in sft


def test_unreachable_class_reports_the_configuration_not_the_vram_shortfall():
    """One root cause must not produce two different errors depending on the run's size.

    A class no configured provider carries is blocked by the CONFIGURATION. Deciding fit first meant
    an oversized run got a VRAM shortfall while the same fleet with a smaller need correctly got
    ``no configured active provider`` -- so the diagnostic depended on how big the run happened to
    be rather than on what actually blocked it, and the shortfall pointed at a knob that cannot help.
    """
    from flash.providers.core.allocator import _resolve_exact_gpu
    from flash.providers.core.base import GPU_INFO, UnsupportedGpuError, providers_for

    assert "lambda" not in providers_for("H200")
    over = GPU_INFO["H200"].vram_gb * 2.5  # far beyond one card
    under = GPU_INFO["H200"].vram_gb - 40  # comfortably inside one card

    for need in (over, under):
        with pytest.raises(UnsupportedGpuError, match="no configured active provider"):
            _resolve_exact_gpu(
                "H200",
                need=need,
                cap=1,
                max_gpu_count=1,
                provider="",
                available=("lambda",),
                widest_cap=8,
                unpinned=("lambda",),
            )

    # a plane that DOES carry the class still reports the fit failure, not a configuration error.
    with pytest.raises(UnsupportedGpuError, match="requires at least"):
        _resolve_exact_gpu(
            "H200",
            need=over,
            cap=1,
            max_gpu_count=1,
            provider="",
            available=("runpod",),
            widest_cap=8,
            unpinned=("runpod",),
        )


def test_catalog_check_is_withheld_when_no_configured_provider_carries_the_class():
    """A width cannot rescue a class nobody in the fleet sells.

    The catalog-check tier answers "this provider may sell a wider SKU". When NO configured
    provider carries the class at all, the obstacle is the class rather than the width, and
    `--gpus N` cannot succeed at any N -- that run belongs to the `no configured active provider`
    rejection, which names the real problem instead of sending the user to retry a dead end.
    """
    from flash.providers.core.allocator import _resolve_exact_gpu
    from flash.providers.core.base import UnsupportedGpuError, providers_for

    # H200 is runpod-only, so a lambda-only plane cannot rent it at any width.
    assert "lambda" not in providers_for("H200")

    with pytest.raises(UnsupportedGpuError) as unreachable:
        _resolve_exact_gpu(
            "H200",
            need=300.0,
            cap=1,
            max_gpu_count=1,
            provider="",
            available=("lambda",),
            widest_cap=8,
            unpinned=("lambda",),
        )
    assert "--gpus" not in str(unreachable.value)

    # the same class on a plane that DOES carry it keeps its remedy: the gate is reachability,
    # not a blanket suppression.
    with pytest.raises(UnsupportedGpuError, match=r"--gpus"):
        _resolve_exact_gpu(
            "H200",
            need=300.0,
            cap=1,
            max_gpu_count=1,
            provider="",
            available=("runpod",),
            widest_cap=8,
            unpinned=("runpod",),
        )


def test_rents_arbitrary_card_counts_splits_providers_by_how_counts_are_sold():
    """The predicate must track how a count is PURCHASED, not whether the provider is configured."""
    from flash.providers.core.fit_errors import rents_arbitrary_card_counts

    assert rents_arbitrary_card_counts(("runpod",)) is True
    assert rents_arbitrary_card_counts(("lambda",)) is False
    assert rents_arbitrary_card_counts(("vast",)) is False
    assert rents_arbitrary_card_counts(("lambda", "vast")) is False
    assert rents_arbitrary_card_counts(("lambda", "runpod")) is True
    # no provider left in play cannot promise a width either.
    assert rents_arbitrary_card_counts(()) is False
