"""Cost estimator: the ``CostEstimate`` result type (breakdown, provider). No network."""

from __future__ import annotations

import dataclasses

import pytest

from flash.cost import RunConfig, estimate_cost
from flash.cost.types import CostEstimate


@pytest.fixture
def est() -> CostEstimate:
    return estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 150))


def test_is_frozen(est):
    with pytest.raises(dataclasses.FrozenInstanceError):
        est.total_usd = 0.0  # type: ignore[misc]


def test_wall_clock_hours_derivation(est):
    assert est.wall_clock_hours == pytest.approx(est.wall_clock_seconds / 3600.0)


def test_billable_hours_derivation(est):
    assert est.billable_hours == pytest.approx(est.train_seconds / 3600.0)


def test_breakdown_lists_every_term(est):
    b = est.breakdown()
    for needle in ("GPU", "Setup", "Per step", "Train", "Wall clock", "Billable", "TOTAL"):
        assert needle in b
    assert "not billed" in b
    # GRPO estimate carries explanatory notes.
    assert "Notes" in b


def test_capped_estimate_flags_in_breakdown():
    capped = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 100_000))
    assert capped.wall_capped
    assert "CAPPED" in capped.breakdown()


def test_subhour_cap_note_renders_minutes_not_zero_hours():
    # A sub-hour wall cap (floored to 60s) must render the CAPPED duration as "1m", never a
    # confusing "0h". (The note also reports the uncapped duration, which is many hours -- so we
    # assert the cap SLOT specifically rather than scanning the whole note for "0h".)
    capped = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 100_000, max_wall_seconds=60))
    assert capped.wall_capped
    cap_note = next(n for n in capped.notes if "wall cap" in n)
    assert "fit the 1m " in cap_note  # 60s -> "1m", not "0h"


def test_fmt_duration_units():
    from flash.cost.analytical import _fmt_duration

    assert _fmt_duration(20) == "20s"  # sub-minute -> seconds, never "0m"
    assert _fmt_duration(59) == "59s"
    assert _fmt_duration(60) == "1m"  # sub-hour -> minutes, never "0h"
    assert _fmt_duration(1800) == "30m"
    assert _fmt_duration(24 * 3600) == "24h"  # whole hours stay clean
    assert _fmt_duration(int(1.5 * 3600)) == "1.5h"  # fractional multi-hour -> one decimal


def test_provider_is_normalized_and_validated():
    # Case/whitespace variants normalize to the canonical substrate; empty -> "auto".
    assert RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="RunPod").provider == "runpod"
    assert RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="").provider == "auto"
    assert RunConfig("Qwen/Qwen3.5-4B", "grpo", 10).provider == "auto"
    # An unknown substrate fails fast here (clear error) instead of as "no GPU class fits".
    with pytest.raises(ValueError, match="unknown provider"):
        RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="aws")


def test_runconfig_preserves_old_positional_constructor():
    config = RunConfig(
        "Qwen/Qwen3.5-0.8B",
        "sft",
        10,
        2048,
        None,
        4,
        None,
        8,
        False,
        None,
        "",
        3600,
        "runpod",
        "owner/environment",
        16_000,
        (2, 5),
    )

    assert config.seq_len == 2048
    assert config.batch_size == 4
    assert config.lora_rank == 8
    assert config.max_wall_seconds == 3600
    assert config.provider == "runpod"
    assert config.environment == "owner/environment"
    assert config.train_tokens == 16_000
    assert config.save_at_steps == (2, 5)
    assert config.gpu_type == ""
    assert config.opd_multi_turn is False
    assert config.opd_max_turns is None


def test_provisional_estimate_preserves_auto_provider():
    # Preparation stays offline: it cannot truthfully name a live substrate before allocation. The
    # lifecycle replaces this provisional provider/count/rate from the selected candidate.
    assert estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 10)).provider == "auto"
    assert (
        estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="runpod")).provider
        == "runpod"
    )


def test_explicit_lambda_quote_uses_lambda_offline_list_price():
    estimate = estimate_cost(
        RunConfig(
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            10,
            provider="lambda",
            gpu_type="B200",
        )
    )

    assert estimate.provider == "lambda"
    assert estimate.gpu_hourly_usd == 6.99


def test_estimate_honors_exact_gpu_instead_of_cheaper_fit():
    unconstrained = estimate_cost(RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 10))
    exact = estimate_cost(RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 10, gpu_type="H100"))

    assert unconstrained.gpu == "RTX 4090"
    assert exact.gpu == "H100"
    assert exact.gpu_hourly_usd > unconstrained.gpu_hourly_usd


def test_selected_candidate_replaces_the_provisional_quote(monkeypatch):
    from flash.providers import allocator, get_provider
    from flash.providers.base import Candidate
    from flash.providers.lambdalabs import api as lambda_api

    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod", "lambda"))
    monkeypatch.setattr(
        get_provider("runpod"),
        "live_candidates",
        lambda need, constraints: [Candidate("runpod", "H100", 3.29, 80)],
    )
    advertised: list[str] = []

    def advertised_price(instance_type, **kwargs):
        advertised.append(instance_type)
        return 2.49

    monkeypatch.setattr(lambda_api, "instance_type_price_usd_hr", advertised_price)
    monkeypatch.setattr(
        lambda_api,
        "list_instance_types",
        lambda *args, **kwargs: {"gpu_1x_h100_pcie": {}},
    )
    monkeypatch.setattr(lambda_api, "regions_with_capacity", lambda *args, **kwargs: [])

    config = RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 10, gpu_type="H100")
    allocation = allocator.allocate(
        config.model_id,
        config.method,
        train=config.train_knobs(),
        thinking=config.thinking,
        max_wall_seconds=24 * 3600,
        provider="",
        gpu_type=config.gpu_type,
        model_revision=config.model_revision,
    )
    estimate = estimate_cost(config, allocation=allocation)

    assert "gpu_1x_h100_pcie" in advertised
    assert (allocation.provider, allocation.gpu, allocation.hourly_usd) == (
        "runpod",
        "H100",
        3.29,
    )
    assert (
        estimate.provider,
        estimate.gpu,
        estimate.gpu_hourly_usd,
        estimate.required_vram_gb,
    ) == (
        allocation.provider,
        allocation.gpu,
        allocation.hourly_usd,
        allocation.min_vram_gb,
    )


def test_estimate_exact_gpu_enforces_provider_support_and_vram():
    with pytest.raises(ValueError, match="cannot provision"):
        RunConfig(
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            10,
            provider="lambda",
            gpu_type="RTX 4090",
        )
    with pytest.raises(ValueError, match="requires at least"):
        estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, gpu_type="RTX 4090"))


def test_runconfig_from_spec_preserves_gpu_constraints():
    from flash.cost.spec import runconfig_from_spec
    from flash.spec import GpuSpec, JobSpec, TrainSpec
    from tests._helpers.profile import attach_sft_profile

    # the subject is hardware passthrough, not sft pricing; the profile is attached because
    # prepare_job is the only producer of an sft spec and cannot emit one without a matching profile.
    spec = attach_sft_profile(
        JobSpec(
            model="Qwen/Qwen3.5-0.8B",
            algorithm="sft",
            train=TrainSpec(epochs=1, max_examples=8),
            gpu=GpuSpec(provider="runpod", type="H100", disk_gb=200),
        )
    )

    config = runconfig_from_spec(spec)
    assert config.provider == "runpod"
    assert config.gpu_type == "H100"
    # the run's disk floor threads through so an exact-auto quote allocates the same disk as launch
    assert config.disk_gb == 200.0


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"id": "owner/single", "params": {"multi_turn": False}}, (False, None)),
        ({"id": "owner/multi", "params": {"multi_turn": True, "max_turns": 24}}, (True, 24)),
        ({"id": "owner/unknown"}, (True, None)),
    ],
)
def test_runconfig_from_spec_preserves_conservative_opd_turn_budget(environment, expected):
    from flash.cost.spec import runconfig_from_spec
    from flash.spec import JobSpec

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "opd",
            "environment": environment,
            "train": {"max_examples": 8, "max_steps": 1},
        }
    )

    config = runconfig_from_spec(spec)

    assert (config.opd_multi_turn, config.opd_max_turns) == expected


def test_quote_preparation_never_calls_live_allocate(monkeypatch):
    """A market/API failure belongs to the retrying lifecycle, never quote preparation."""
    import flash.providers.allocator as allocator_mod

    def explode(*_args, **_kwargs):
        raise AssertionError("quote preparation called live allocate")

    monkeypatch.setattr(allocator_mod, "allocate", explode)
    estimate = estimate_cost(
        RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 10, gpu_type="H100", disk_gb=200.0)
    )
    assert estimate.gpu == "H100"


def test_vast_live_pricing_duration_mirrors_launch(monkeypatch):
    # The live pricing API remains duration-aware when called explicitly; provisional run quoting no
    # longer calls it because offer lookup is also a capacity lookup.
    from flash.cost.facts import gpu_hourly_usd
    from flash.providers.vast import jobs as vast
    from flash.providers.vast import pricing

    seen: list[float] = []

    def fake_usable(min_vram_gb, disk_gb, *a, max_wall_seconds=0, **k):
        seen.append(max_wall_seconds)
        return []

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)

    def market_walls(wall: float) -> set[float]:
        seen.clear()
        monkeypatch.setattr(pricing, "_rates_cache", {"ts": 0.0, "data": None})
        gpu_hourly_usd("H100", provider="vast", max_wall_seconds=wall)
        return set(seen)

    assert market_walls(0) == {0.0}
    assert market_walls(7200) == {7200.0}


def test_pick_gpu_vast_duration_bound_fetches_market_once(monkeypatch):
    # Copilot: pick_gpu ranks every fitting class by $/hr. A duration-bound Vast query bypasses the
    # per-call rate cache (Codex MtzrI), so pricing each candidate individually inside min(key=...)
    # would fire one identical full market fetch PER fitting class. pick_gpu must fetch the live rate
    # map ONCE and rank from it -> exactly one usable_offers call no matter how many classes fit.
    from flash.cost.facts import pick_gpu
    from flash.providers.vast import jobs as vast
    from flash.providers.vast import pricing

    calls = {"n": 0}

    def fake_usable(min_vram_gb, disk_gb, *a, max_wall_seconds=0, **k):
        calls["n"] += 1
        return []

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    monkeypatch.setattr(pricing, "_rates_cache", {"ts": 0.0, "data": None})  # isolate cache

    # A tiny VRAM floor leaves MANY fitting Vast classes -> per-candidate pricing would fetch many times.
    gpu = pick_gpu(8, provider="vast", max_wall_seconds=7200.0)
    assert gpu  # a class was chosen
    assert calls["n"] == 1  # ONE market fetch despite multiple fitting candidates (was N)


def test_pick_gpu_vast_skips_classes_without_a_live_offer(monkeypatch):
    # Codex: ranking via the static-merged map could SELECT and quote a cheaper class (e.g. RTX 4090)
    # that has NO surviving offer under the wall cap — one the launch-time usable_offers path would
    # never rent. pick_gpu(provider="vast") must restrict selection to classes that ACTUALLY have a
    # rentable offer, even when a cheaper class fits and is cheaper on its static (RunPod) rate.
    from types import SimpleNamespace

    from flash.cost.facts import pick_gpu
    from flash.providers.vast import jobs as vast

    # The live market has ONLY an A100 SXM offer (a larger/pricier class); the cheaper 24/48 GB classes
    # fit the 8 GB requirement and are cheaper statically, but have NO surviving offer.
    def fake_usable(min_vram_gb, disk_gb, *a, max_wall_seconds=0, **k):
        return [SimpleNamespace(gpu="A100 SXM", dph_total=1.20)]

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    gpu = pick_gpu(8, provider="vast", max_wall_seconds=7200.0)
    assert gpu == "A100 SXM"  # the only class with a rentable offer, NOT the cheaper-static 4090


def test_pick_gpu_vast_floors_market_search_at_required_vram(monkeypatch):
    # Codex: the live Vast offer map for a HIGH-VRAM job must be searched at the job's required VRAM, not
    # the smallest managed class. The market page is price-sorted + LIMITED, so a small-class floor lets
    # cheap 24-40 GB offers crowd the 80 GB classes off it -> they'd be omitted and the quote would fall
    # back to static pricing. pick_gpu must thread required_vram_gb into the search floor (allocator parity).
    from types import SimpleNamespace

    from flash.cost.facts import pick_gpu
    from flash.providers.vast import jobs as vast

    seen = {}

    def fake_usable(min_vram_gb, disk_gb, *a, max_wall_seconds=0, **k):
        seen["floor"] = min_vram_gb
        return [SimpleNamespace(gpu="A100 SXM", dph_total=1.20)]

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    pick_gpu(80, provider="vast", max_wall_seconds=7200.0)
    # Old behavior floored at the smallest managed class (~24 GB); the fix floors at the required 80 GB.
    assert seen["floor"] == 80


def test_gpu_hourly_usd_vast_floors_rate_at_required_vram(monkeypatch):
    # Codex 3519040487: pick_gpu floors the market at the required VRAM, but the follow-up RATE lookup
    # (gpu_hourly_usd -> vast.hourly_rate) used to search from the smallest managed class -> the same
    # crowd-off, so a high-VRAM selection missed the live map and the quote silently fell back to the
    # static catalog rate. The rate lookup must thread min_vram_gb too (selection/quote parity), and
    # then return the LIVE offer rate for the selected class, not the static one.
    from types import SimpleNamespace

    from flash.cost.facts import gpu_hourly_usd
    from flash.providers.vast import jobs as vast

    seen = {}

    def fake_usable(min_vram_gb, disk_gb, *a, max_wall_seconds=0, **k):
        seen["floor"] = min_vram_gb
        return [SimpleNamespace(gpu="A100 SXM", dph_total=1.20)]

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    rate = gpu_hourly_usd("A100 SXM", provider="vast", max_wall_seconds=7200.0, min_vram_gb=80)
    assert seen["floor"] == 80  # floored at required VRAM, not the smallest managed class
    assert rate == 1.20  # live offer rate, not the static catalog fallback


def test_gpu_hourly_usd_vast_exact_threads_class_constraint(monkeypatch):
    from types import SimpleNamespace

    from flash.cost.facts import gpu_hourly_usd
    from flash.providers.vast import jobs as vast

    seen: list[str] = []

    def fake_usable(min_vram_gb, disk_gb, *args, gpu_type="", **kwargs):
        seen.append(gpu_type)
        return [SimpleNamespace(gpu="A100 SXM 40GB", dph_total=0.77)]

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", fake_usable)

    exact = gpu_hourly_usd(
        "A100 SXM 40GB",
        provider="vast",
        min_vram_gb=40,
        gpu_type="A100 SXM 40GB",
    )
    unconstrained = gpu_hourly_usd(
        "A100 SXM 40GB",
        provider="vast",
        min_vram_gb=40,
    )

    assert exact == 0.77
    assert unconstrained == 0.77
    assert seen == ["A100 SXM 40GB", ""]


def test_explicit_vast_quote_stays_offline(monkeypatch):
    from flash.providers.base import get_gpu_info
    from flash.providers.vast import jobs as vast

    def explode(*args, **kwargs):
        raise AssertionError("provisional quote queried Vast capacity")

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", explode)
    explicit = estimate_cost(
        RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 10, provider="vast", gpu_type="H100")
    )
    assert (explicit.provider, explicit.gpu_hourly_usd) == (
        "vast",
        get_gpu_info("H100").hourly_usd,
    )


def test_auto_quote_does_not_require_live_vast_capacity(monkeypatch):
    from flash.providers.vast import jobs as vast

    def explode(*args, **kwargs):
        raise AssertionError("provisional quote queried Vast capacity")

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", explode)
    estimate = estimate_cost(RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 10, gpu_type="H100"))
    assert (estimate.provider, estimate.gpu) == ("auto", "H100")


def test_a100_sxm_40gb_has_real_tflops_not_default():
    # Codex: the 40 GB A100 SXM4 class (selectable for Lambda/Vast) must carry a real TFLOPS entry, or
    # gpu_tflops() falls to the 100-default and inflates its seconds_per_step / quoted cost ~3x. It has
    # the same SMs as the 80 GB SXM, so its compute equals the other A100 entries.
    from flash.cost.facts import _DEFAULT_TFLOPS, gpu_tflops

    assert gpu_tflops("A100 SXM 40GB") == gpu_tflops("A100 SXM")
    assert gpu_tflops("A100 SXM 40GB") != _DEFAULT_TFLOPS


def test_pick_gpu_vast_offline_falls_back_to_static(monkeypatch):
    # When the market is unreachable (no VAST_API_KEY -> live_offer_rates returns {}), selection must
    # stay offline-safe: rank ALL fitting classes by their static rate rather than crash or pick nothing.
    from flash.cost.facts import pick_gpu

    monkeypatch.delenv("VAST_API_KEY", raising=False)
    gpu = pick_gpu(8, provider="vast")
    assert gpu  # a fitting class is still chosen from the static fallback


def test_selected_live_candidate_overrides_provisional_provider_rate_and_count():
    from flash.providers.base import Candidate

    candidate = Candidate("vast", "H100", 2.17, 80, gpu_count=2)
    est = estimate_cost(
        RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 10, gpu_type="H100", gpu_count=8),
        allocation=candidate,
    )
    assert (est.provider, est.gpu, est.gpu_hourly_usd, est.gpu_count) == (
        "vast",
        "H100",
        2.17,
        2,
    )


def test_b200_not_cheaper_or_faster_than_h200_for_grpo():
    # regression: the estimator must not advertise b200 as faster/cheaper than h200 on peak flops.
    # b200/sm100 training is h200-class (portable kernels), so at its higher $/hr b200 must never
    # come out cheaper, and never faster, than h200 for the same run.
    from flash.cost.facts import gpu_hourly_usd

    h200 = estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 100, gpu_type="H200"))
    b200 = estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 100, gpu_type="B200"))

    assert gpu_hourly_usd("B200") > gpu_hourly_usd("H200")  # b200 is the pricier card
    # same effective training throughput => b200 is no faster than h200 ...
    assert b200.seconds_per_step == pytest.approx(h200.seconds_per_step)
    assert b200.train_seconds == pytest.approx(h200.train_seconds)
    # ... and at its higher $/hr, never cheaper.
    assert b200.total_usd > h200.total_usd


# ---------------------------------------------------------------------------
# multi-gpu: total scales linearly with gpu_count
# ---------------------------------------------------------------------------


def test_offline_unpinned_estimate_does_not_bill_the_ceiling():
    single = estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 150))
    wide = estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 150, gpu_count=8))
    assert single.gpu_count == 1
    assert wide.gpu_count == 1
    # pick_gpu returns a class that fits the whole run alone, so an offline estimate has no basis for
    # charging the ceiling. server-side submit uses allocate() and records the selected count instead.
    assert wide.gpu_hourly_usd == single.gpu_hourly_usd
    assert wide.train_seconds == pytest.approx(single.train_seconds)
    assert wide.total_usd == pytest.approx(single.total_usd)


def test_offline_estimate_supports_eight_card_only_runs(monkeypatch):
    """`flash train --cost` must price a run that fits eight cards but no four-card shape."""
    monkeypatch.setattr("flash.cost.analytical.required_vram_gb", lambda *a, **k: 700)
    config = RunConfig("Qwen/Qwen3.5-4B", "sft", 1, gpu_count=8)
    estimate = estimate_cost(config)
    assert estimate.required_vram_gb == 700
    assert estimate.gpu_count == 8
    with pytest.raises(ValueError, match="no GPU class fits"):
        estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "sft", 1, gpu_count=4))


def test_offline_estimate_applies_the_pinned_revision_geometry_cap(monkeypatch):
    """A pinned revision must not receive an eight-card quote allocation will never honor."""
    from flash.cost.analytical import _offline_gpu_shape

    monkeypatch.setattr("flash.cost.analytical.required_vram_gb", lambda *a, **k: 700)
    config = RunConfig(
        "Qwen/Qwen3.5-4B",
        "sft",
        1,
        gpu_count=8,
        model_revision="a" * 40,
    )

    with pytest.raises(ValueError, match="across up to 4 cards"):
        _offline_gpu_shape(config)


def test_allocator_selected_gpu_count_renders_and_applies_speedup():
    from flash.providers.base import Candidate

    config = RunConfig("Qwen/Qwen3.5-4B", "grpo", 150, gpu_count=8)
    one = estimate_cost(config, allocation=Candidate("runpod", "H100", 3.29, 80, 1))
    two = estimate_cost(config, allocation=Candidate("runpod", "H100", 3.29, 80, 2))
    # the breakdown surfaces the selected multi-gpu shape and the persisted timing credits the
    # measured sharding speedup instead of billing two cards for a one-card runtime.
    assert "2x" in two.breakdown()
    assert "per card" in two.breakdown()
    assert two.seconds_per_step < one.seconds_per_step
    assert two.train_seconds < one.train_seconds
    assert two.total_usd < 2 * one.total_usd


@pytest.mark.parametrize(("bad", "exc"), [(0, ValueError), (-1, ValueError), (True, TypeError)])
def test_runconfig_rejects_bad_gpu_count(bad, exc):
    with pytest.raises(exc):
        RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, gpu_count=bad)
