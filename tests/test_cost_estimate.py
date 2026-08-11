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
    from flash.providers.lambda_ import api as lambda_api

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
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.cost.spec import runconfig_from_spec
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
    from flash.core.spec import JobSpec
    from flash.cost.spec import runconfig_from_spec

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
    # per-call rate cache, so pricing each candidate individually inside min(key=...) would fire one
    # identical full market fetch PER fitting class. pick_gpu must fetch the live rate map ONCE and
    # rank from it -> exactly one usable_offers call no matter how many classes fit.
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


def test_b200_never_beats_h200_at_any_card_count_for_grpo_or_opd():
    """The never-faster contract must survive sharding, on both rollout algorithms.

    The single-card test above passes on equal per-step seconds. Sharding divides only the
    gpu-bound half, so a card whose floor offset differed would diverge as the count rises even
    while the 1-card numbers matched -- the tie has to hold per-count, not just at count=1. opd is
    covered too because it carries the same floor with a much larger share of its step (~94% of the
    gpu-bound half vs ~88% for grpo), so it is the more sensitive of the two, not a duplicate.
    """
    from flash.cost.analytical import multi_card_speedup, step_seconds_split

    for method in ("grpo", "opd"):
        h_bound, h_fixed = step_seconds_split(
            RunConfig("Qwen/Qwen3.5-4B", method, 100, gpu_type="H200"), "H200"
        )
        b_bound, b_fixed = step_seconds_split(
            RunConfig("Qwen/Qwen3.5-4B", method, 100, gpu_type="B200"), "B200"
        )
        for count in (1, 2, 4, 8):
            h_sps = h_bound / multi_card_speedup(count, "H200") + h_fixed
            b_sps = b_bound / multi_card_speedup(count, "B200") + b_fixed
            assert b_sps >= h_sps - 1e-9, (
                f"{method} at {count} cards quotes B200 faster than H200 "
                f"({b_sps:.3f}s vs {h_sps:.3f}s): the equivalence-class tie broke under sharding"
            )


# ---------------------------------------------------------------------------
# multi-gpu: total scales linearly with gpu_count
# ---------------------------------------------------------------------------


def test_offline_unpinned_estimate_does_not_bill_the_ceiling():
    # 40 steps, not 150: a 150-step 4B grpo run exceeds the 24h wall cap, and a clamped run
    # reports the cap's runtime on every shape, so the assertions below would pass by collision
    # rather than because the ceiling was not billed.
    single = estimate_cost(RunConfig("Qwen/Qwen3.5-2B", "grpo", 40))
    wide = estimate_cost(RunConfig("Qwen/Qwen3.5-2B", "grpo", 40, gpu_count=8))
    assert not any("wall cap" in note for note in single.notes)
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
    """The offline quote must follow the SAME pin-certification rule allocation will apply.

    Both directions matter, and asserting only the narrow one lets the test keep passing for the
    wrong reason: an uncertifiable pin fails closed at four cards, but a pin whose geometry IS
    certified gets the width its own head count allows. 3.5-4B records 16 heads, which divide 8, so
    the two cases genuinely differ here -- quoting four for a certified pin would be the very defect
    this PR removes.
    """
    import flash.engine.plan.pinned_geometry as pinned_geometry
    import flash.engine.plan.vram as vram
    from flash.core.catalog import MODELS
    from flash.cost.analytical import _offline_gpu_shape

    monkeypatch.setattr("flash.cost.analytical.required_vram_gb", lambda *a, **k: 700)
    monkeypatch.setattr("flash.cost.analytical.total_params_b", lambda *a, **k: 4.7)
    monkeypatch.setattr(pinned_geometry, "_PINNED_GEOMETRY_MEMO", {})
    config = RunConfig(
        "Qwen/Qwen3.5-4B",
        "sft",
        1,
        gpu_count=8,
        model_revision="a" * 40,
    )

    def _unreadable(*_a, **_k):
        raise RuntimeError("transient hub error")

    # uncertified: the pin keeps the unvalidated-revision ceiling, and 700 GB does not fit four.
    monkeypatch.setattr(vram, "fetch_hf_model_geometry", _unreadable)
    with pytest.raises(ValueError, match="across up to 4 cards"):
        _offline_gpu_shape(config)

    # certified: the commit's own 16 heads divide 8, so the same quote reaches an eight-card shape.
    info = MODELS["Qwen/Qwen3.5-4B"]
    monkeypatch.setattr(
        vram,
        "fetch_hf_model_geometry",
        lambda *_a, **_k: (
            info.params_b,
            info.vocab_size,
            info.hidden_size,
            info.num_layers,
            info.num_attention_heads,
        ),
    )
    # note the real order is (gpu, need, count, provider, hourly); the annotation on
    # `_offline_gpu_shape` says (gpu, count, need, ...) and is wrong, which is pre-existing.
    _gpu, need, count, _provider, _rate = _offline_gpu_shape(config)
    assert (need, count) == (700, 8)


def test_the_offline_probe_sizes_a_pinned_catalog_model_by_its_revision(monkeypatch):
    """The offline shape probe must size the PINNED commit, not the catalog's default revision.

    (This replaces an open-model version of the same test: it used an uncataloged id sized over HF,
    and uncataloged models are rejected now. The invariant it protected -- the probe passes the
    revision through rather than quoting default-revision weights -- still holds for a pinned
    catalog model, which is the only way to reach revision-specific sizing at all.)
    """
    import flash.engine.plan.pinned_geometry as pinned_geometry
    import flash.engine.plan.vram as vram
    from flash.core.catalog import MODELS
    from flash.cost.analytical import _offline_gpu_shape
    from flash.cost.facts import _PINNED_SIZE_MEMO

    model = "Qwen/Qwen3.5-9B"
    info = MODELS[model]
    expected_revision = "f" * 40
    seen_revisions = []

    def _pinned_geometry(model_id, revision="", strict=False):
        assert model_id == model
        seen_revisions.append(revision)
        return (
            info.params_b,
            info.vocab_size,
            info.hidden_size,
            info.num_layers,
            info.num_attention_heads,
        )

    monkeypatch.setattr(vram, "fetch_hf_model_geometry", _pinned_geometry)
    monkeypatch.setattr("flash.cost.facts._PINNED_SIZE_MEMO", dict(_PINNED_SIZE_MEMO))
    monkeypatch.setattr(pinned_geometry, "_PINNED_GEOMETRY_MEMO", {})
    _PINNED_SIZE_MEMO.pop((model, expected_revision), None)

    gpu, count, need, provider, rate = _offline_gpu_shape(
        RunConfig(model, "sft", 1, model_revision=expected_revision)
    )

    assert gpu
    assert count >= 1
    assert need > 0
    assert provider
    assert rate > 0
    # Several independent sites read a pinned run here -- the fail-closed params check, the VRAM
    # requirement, and the head-geometry cap that decides how wide it may be rented. Each must carry
    # the pin, and they must SHARE the one lookup: a site that re-fetched independently let a hub
    # blip between two of them narrow a just-validated pin (see
    # test_a_blip_after_sizing_cannot_narrow_an_already_validated_pin). One fetch, and the revision
    # is the one that was asked for -- a site that dropped the pin would fetch the default revision
    # under a different memo key and push this above one.
    assert len(seen_revisions) == 1, (
        f"a pinned quote must reach the hub exactly once, saw {seen_revisions}"
    )
    assert set(seen_revisions) == {expected_revision}


def test_allocator_selected_gpu_count_renders_and_applies_speedup():
    from flash.providers.base import Candidate

    # 50 steps, not 150: a long 4B grpo run exceeds the 24h wall cap, and a clamped run reports
    # the cap's runtime on every shape, so the speedup assertion below would fail on equality even
    # though sharding is working. the run has to fit under the cap for its runtime to be
    # observable at all -- the guard on the next line keeps that true if step costs change again.
    config = RunConfig("Qwen/Qwen3.5-4B", "grpo", 50, gpu_count=8)
    one = estimate_cost(config, allocation=Candidate("runpod", "H100", 3.29, 80, 1))
    two = estimate_cost(config, allocation=Candidate("runpod", "H100", 3.29, 80, 2))
    assert not any("wall cap" in note for note in one.notes)
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


def test_offline_quote_fit_failure_names_the_card_count_that_fixes_it():
    """`flash train --cost` must name the `--gpus` width when a wider shape would fit.

    A 35B-A3B GRPO run at the default single-card ceiling needs more VRAM than any one card has,
    but fits on two. Reporting only the shortfall reads as "this run is impossible" for the one
    case that is actually a one-flag fix, so the remedy is asserted here rather than the bare
    shortfall. The signature is unchanged by the fix, so this fails on the MESSAGE against
    unfixed code, not on an import.
    """
    from flash.cost.analytical import _offline_gpu_shape
    from flash.cost.types import RunConfig

    shared = {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "method": "grpo",
        "steps": 10,
        "seq_len": 2048,
        "completion_len": 512,
        "batch_size": 8,
        "group_size": 4,
        "lora_rank": 16,
    }
    with pytest.raises(ValueError, match=r"--gpus 2") as unpinned:
        _offline_gpu_shape(RunConfig(gpu_count=1, **shared))
    assert "no GPU class fits" in str(unpinned.value)

    with pytest.raises(ValueError, match=r"--gpus 2") as pinned:
        _offline_gpu_shape(RunConfig(gpu_count=1, gpu_type="H200", **shared))
    assert "cannot fit this run" in str(pinned.value)

    # and the suggested width is real: the same run quotes cleanly at it.
    gpu, _need, count, _provider, _hourly = _offline_gpu_shape(RunConfig(gpu_count=2, **shared))
    assert count == 2, (gpu, count)


def test_offline_quote_fit_failure_omits_the_remedy_when_nothing_fits(monkeypatch):
    """An unsatisfiable run must not be sent to a second dead end.

    The remedy is searched against the same fit model that rejected the run, so a need no shape
    can hold produces no suggestion at all.
    """
    monkeypatch.setattr("flash.cost.analytical.required_vram_gb", lambda *a, **k: 100_000)
    from flash.cost.analytical import _offline_gpu_shape
    from flash.cost.types import RunConfig

    with pytest.raises(ValueError, match="no GPU class fits") as exc:
        _offline_gpu_shape(RunConfig("Qwen/Qwen3.5-4B", "sft", 1, gpu_count=1))
    assert "--gpus" not in str(exc.value)
