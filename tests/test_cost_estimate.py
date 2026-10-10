"""Cost estimator: the ``CostEstimate`` result type (breakdown, provider). No network."""

from __future__ import annotations

import dataclasses

import pytest

from flash.cost import RunConfig, estimate_cost
from flash.cost.types import CostEstimate


@pytest.fixture
def est() -> CostEstimate:
    return estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 150))


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


def test_plain_analytical_total_uses_billing_round_half_up(est):
    quote = dataclasses.replace(est, total_usd=1.005)

    assert f"{quote.total_usd:.2f}" == "1.00", "the fixture must expose half-even formatting"
    assert "TOTAL      : $1.01" in quote.breakdown()


def test_styled_analytical_total_uses_billing_round_half_up(est, monkeypatch):
    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    quote = dataclasses.replace(est, total_usd=1.005)

    assert f"{quote.total_usd:.2f}" == "1.00", "the fixture must expose half-even formatting"
    assert "$1.01" in render.cost_panel(quote)


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
    assert RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, provider="RunPod").provider == "runpod"
    assert RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, provider="").provider == "auto"
    assert RunConfig("Qwen/Qwen3.5-9B", "grpo", 10).provider == "auto"
    # An unknown substrate fails fast here (clear error) instead of as "no GPU class fits".
    with pytest.raises(ValueError, match="unknown provider"):
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, provider="aws")


def test_runconfig_preserves_old_positional_constructor():
    config = RunConfig(
        "Qwen/Qwen3.5-9B",
        "sft",
        10,
        2048,
        None,
        4,
        None,
        8,
        False,
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

    # the 15 args above stop short of the opd fields, so they cannot catch a field INSERTED among
    # the later ones -- the shift only rebinds from that position on. pin the order itself, which is
    # what the positional contract actually is.
    import dataclasses

    order = [f.name for f in dataclasses.fields(RunConfig)]
    assert order.index("opd_multi_turn") < order.index("opd_max_turns")
    assert order.index("opd_max_turns") < order.index("sft_retained_examples")
    # anything added later must be APPENDED, never slotted beside a related field: an old positional
    # caller would silently bind its opd flag to the newcomer rather than fail. asserted as a
    # PREFIX rather than an exact tail so the guard keeps failing on an insertion while a correctly
    # appended field does not have to edit it -- the previous exact-tail form failed either way,
    # which makes the failure uninformative about which mistake was made.
    appended_so_far = ["sft_retained_examples", "providers", "gpu_type_fallbacks"]
    assert order[-len(appended_so_far) :] == appended_so_far, (
        "a new RunConfig field must be appended; inserting one shifts every later parameter and "
        "silently reinterprets old positional calls as different quantities"
    )


def test_a_malformed_retained_example_count_is_rejected_not_read_as_unknown():
    """A bad row count must raise, because the width rule reads it as "no constraint" instead.

    `sft_data_parallel_cards` treats a non-positive row count as "unknown, do not constrain" -- the
    quote runs before the dataset is materialized, so that default is correct there. It is exactly
    wrong for a malformed value: 0 or a negative silently credits every rented card, producing the
    understated width and cost this field exists to prevent. A guard downstream cannot recover the
    difference, since by then both cases look identical, so the type boundary has to reject it.
    """
    import pytest

    from flash.cost.analytical import executed_gpu_count
    from flash.cost.types import RunConfig

    def config(rows):
        return RunConfig("Qwen/Qwen3.8-27B", "sft", 10, batch_size=8, sft_retained_examples=rows)

    for bad in (0, -5):
        with pytest.raises(ValueError, match="sft_retained_examples must be >= 1"):
            config(bad)
    # bools are ints in python, and True would silently mean "one row" -- one rank, not a wide run.
    for wrong_type in (True, False, 2.5, "8"):
        with pytest.raises(TypeError, match="sft_retained_examples must be an integer"):
            config(wrong_type)

    # None still means UNKNOWN and must keep crediting every rented card: the quote legitimately
    # runs before the row count exists, and narrowing there would reject runs that are fine.
    assert executed_gpu_count(config(None), 4) == 4
    # a real count still narrows, so the guard did not disable the rule it protects.
    assert executed_gpu_count(config(10), 4) == 2


def test_provisional_estimate_preserves_auto_provider():
    # Preparation stays offline: it cannot truthfully name a live substrate before allocation. The
    # lifecycle replaces this provisional provider/count/rate from the selected candidate.
    assert estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10)).provider == "auto"
    assert (
        estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, provider="runpod")).provider
        == "runpod"
    )


def test_explicit_lambda_quote_uses_lambda_offline_list_price():
    estimate = estimate_cost(
        RunConfig(
            "Qwen/Qwen3.5-9B",
            "grpo",
            10,
            provider="lambda",
            gpu_type="B200",
        )
    )

    assert estimate.provider == "lambda"
    assert estimate.gpu_hourly_usd == 6.99


def test_estimate_honors_exact_gpu_instead_of_cheaper_fit():
    unconstrained = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10))
    exact = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, gpu_type="H100"))

    assert unconstrained.gpu == "A100 PCIe"
    assert exact.gpu == "H100"
    assert exact.gpu_hourly_usd > unconstrained.gpu_hourly_usd


def test_selected_candidate_replaces_the_provisional_quote(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate
    from flash.providers.core.registry import get_provider
    from flash.providers.lambda_.client import api as lambda_api

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

    config = RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, gpu_type="H100")
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
            "Qwen/Qwen3.5-9B",
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
            model="Qwen/Qwen3.5-9B",
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
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "opd",
            "environment": environment,
            "train": {"max_examples": 8, "max_steps": 1},
        }
    )

    config = runconfig_from_spec(spec)

    assert (config.opd_multi_turn, config.opd_max_turns) == expected


def test_quote_preparation_never_calls_live_allocate(monkeypatch):
    """A market/API failure belongs to the retrying lifecycle, never quote preparation."""
    import flash.providers.core.allocator as allocator_mod

    def explode(*_args, **_kwargs):
        raise AssertionError("quote preparation called live allocate")

    monkeypatch.setattr(allocator_mod, "allocate", explode)
    estimate = estimate_cost(
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, gpu_type="H100", disk_gb=200.0)
    )
    assert estimate.gpu == "H100"


def test_explicit_vast_quote_stays_offline(monkeypatch):
    from flash.providers.core.base import get_gpu_info
    from flash.providers.vast import jobs as vast

    def explode(*args, **kwargs):
        raise AssertionError("provisional quote queried Vast capacity")

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast, "usable_offers", explode)
    explicit = estimate_cost(
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, provider="vast", gpu_type="H100")
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
    estimate = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, gpu_type="H100"))
    assert (estimate.provider, estimate.gpu) == ("auto", "H100")


def test_a100_sxm_40gb_has_real_tflops_not_default():
    # Codex: the 40 GB A100 SXM4 class (selectable for Lambda/Vast) must carry a real TFLOPS entry, or
    # gpu_tflops() falls to the 100-default and inflates its seconds_per_step / quoted cost ~3x. It has
    # the same SMs as the 80 GB SXM, so its compute equals the other A100 entries.
    from flash.cost.facts import _DEFAULT_TFLOPS, gpu_tflops

    assert gpu_tflops("A100 SXM 40GB") == gpu_tflops("A100 SXM")
    assert gpu_tflops("A100 SXM 40GB") != _DEFAULT_TFLOPS


def test_selected_live_candidate_overrides_provisional_provider_rate_and_count():
    from flash.providers.core.base import Candidate

    candidate = Candidate("vast", "H100", 2.17, 80, gpu_count=2)
    est = estimate_cost(
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, gpu_type="H100", gpu_count=8),
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
    from flash.providers.core.base import GPU_INFO

    h200 = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 100, gpu_type="H200"))
    b200 = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 100, gpu_type="B200"))

    assert GPU_INFO["B200"].hourly_usd > GPU_INFO["H200"].hourly_usd
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
            RunConfig("Qwen/Qwen3.5-9B", method, 100, gpu_type="H200"), "H200"
        )
        b_bound, b_fixed = step_seconds_split(
            RunConfig("Qwen/Qwen3.5-9B", method, 100, gpu_type="B200"), "B200"
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
    single = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 40))
    wide = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 40, gpu_count=8))
    assert not any("wall cap" in note for note in single.notes)
    assert single.gpu_count == 1
    assert wide.gpu_count == 1
    # offline selection returns a class that fits the whole run alone, so the estimate has no basis
    # for charging the ceiling. server-side submit uses allocate() and records the selected count.
    assert wide.gpu_hourly_usd == single.gpu_hourly_usd
    assert wide.train_seconds == pytest.approx(single.train_seconds)
    assert wide.total_usd == pytest.approx(single.total_usd)


def test_a_pinned_gpu_class_is_never_auto_widened_by_the_quote():
    """Auto-sizing applies only when neither the class nor the count is authored.

    A pinned class with no count is a one-card pin at the parse boundary
    (`flash/schema/__init__.py`), so quoting it across eight cards would both bill eight cheap cards
    the author never asked for and name a shape submit rejects. Measured: before this guard, a 24 GB
    RTX 4090 quoted 8 cards for an 80 GB run instead of raising.
    """
    with pytest.raises(ValueError, match=r"exact GPU 'RTX 4090' cannot fit this run"):
        estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, gpu_type="RTX 4090"))


def test_auto_sizing_only_considers_cards_the_pinned_provider_can_rent():
    """A provider pin must narrow the pool BEFORE the count is sized, not just during ranking.

    The ranking loop filters per candidate, which is too late for two decisions taken up front: the
    auto-sized count and the no-fit message. Measured before this fix, a vast-pinned 119 GB run
    sized one card against another provider's H200, then ranked empty and reported "more than any
    8-card validated GPU combination (1177.6 GB max)" -- naming a capacity larger than the
    requirement it claimed could not be met, while 2x80 GB vast cards would have fit.
    """
    from flash.cost.analytical import _offline_gpu_shape
    from flash.providers.core.base import providers_for

    gpu, _need, count, _provider, _hourly = _offline_gpu_shape(
        RunConfig("Qwen/Qwen3.6-35B-A3B", "sft", 10, provider="vast")
    )
    assert "vast" in providers_for(gpu), f"quoted {gpu}, which vast cannot provision"
    assert count >= 2, "a 119 GB run does not fit one vast card"


def test_an_authored_count_still_gets_raise_count_advice_on_a_pinned_class():
    """The pinned-class message must not swallow the remedy when the COUNT is what blocks fit.

    A pinned class with no count is blocked by the class, so naming shrink knobs is right. With an
    authored count, raising it is a real remedy and `allocate()` says so -- the quote contradicting
    it would send the user to shrink a run that would fit on one more card.
    """
    with pytest.raises(ValueError, match=r"--gpus \d"):
        estimate_cost(RunConfig("Qwen/Qwen3.6-35B-A3B", "sft", 10, gpu_type="H100", gpu_count=1))


def test_offline_estimate_supports_eight_card_only_runs(monkeypatch):
    """`flash train --cost` must price a run that fits eight cards but no four-card shape."""
    monkeypatch.setattr("flash.cost.analytical.required_vram_gb", lambda *a, **k: 700)
    config = RunConfig("Qwen/Qwen3.5-9B", "sft", 1, gpu_count=8)
    estimate = estimate_cost(config)
    assert estimate.required_vram_gb == 700
    assert estimate.gpu_count == 8
    # the pinned four-card ceiling is the reason this fails, so the message must name that ceiling
    # and the count that would fit -- not just report a generic no-fit.
    with pytest.raises(ValueError, match=r"gpu\.count=4 provides at most .*`--gpus 8`"):
        estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "sft", 1, gpu_count=4))


def test_offline_estimate_applies_the_pinned_revision_geometry_cap(monkeypatch):
    """The offline quote stays OFFLINE: it caps on the catalog row and never reaches the hub.

    `_offline_gpu_shape` is documented as structural preparation that must not consume live
    failures, so it does not certify the pin. 3.5-4B records 16 heads, which divide 8, so the quote
    reaches an eight-card shape whether or not the hub is reachable.

    Asserting both hub states is the point: certifying here would let a transient hub error convert
    a quotable eight-card run into a hard "does not fit across up to 4 cards" ValueError, from a
    code path whose whole contract is that it does no network i/o. Certification belongs on the
    submission path.
    """
    import flash.engine.plan.model_config_probe as model_config_probe
    import flash.engine.plan.vram as vram
    from flash.core.catalog import MODELS
    from flash.cost.analytical import _offline_gpu_shape

    monkeypatch.setattr("flash.cost.analytical.required_vram_gb", lambda *a, **k: 700)
    monkeypatch.setattr("flash.cost.analytical.total_params_b", lambda *a, **k: 4.7)
    monkeypatch.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})
    config = RunConfig(
        "Qwen/Qwen3.5-9B",
        "sft",
        1,
        gpu_count=8,
        model_revision="a" * 40,
    )

    def _unreadable(*_a, **_k):
        raise RuntimeError("transient hub error")

    # hub down: the offline quote never calls it, so the row's 16 heads still reach eight cards.
    monkeypatch.setattr(vram, "fetch_hf_model_geometry", _unreadable)
    _gpu_d, _need_d, count_down, _provider_d, _rate_d = _offline_gpu_shape(config)
    assert count_down == 8, "a hub outage must not narrow an offline quote"

    # hub healthy: identical answer, proving the quote does not depend on hub reachability.
    info = MODELS["Qwen/Qwen3.5-9B"]
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
    import flash.engine.plan.model_config_probe as model_config_probe
    import flash.engine.plan.vram as vram
    from flash.core.catalog import MODELS
    from flash.cost.analytical import _offline_gpu_shape
    from flash.cost.facts import _PINNED_SIZE_MEMO

    model = "Qwen/Qwen3.5-9B"
    info = MODELS[model]
    expected_revision = "f" * 40
    seen_revisions = []

    def _model_config_probe(model_id, revision="", strict=False):
        assert model_id == model
        seen_revisions.append(revision)
        return (
            info.params_b,
            info.vocab_size,
            info.hidden_size,
            info.num_layers,
            info.num_attention_heads,
        )

    monkeypatch.setattr(vram, "fetch_hf_model_geometry", _model_config_probe)
    monkeypatch.setattr("flash.cost.facts._PINNED_SIZE_MEMO", dict(_PINNED_SIZE_MEMO))
    monkeypatch.setattr(model_config_probe, "_CONFIG_PROBE_MEMO", {})
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
    from flash.providers.core.base import Candidate

    # 50 steps, not 150: a long 4B grpo run exceeds the 24h wall cap, and a clamped run reports
    # the cap's runtime on every shape, so the speedup assertion below would fail on equality even
    # though sharding is working. the run has to fit under the cap for its runtime to be
    # observable at all -- the guard on the next line keeps that true if step costs change again.
    config = RunConfig("Qwen/Qwen3.5-9B", "grpo", 50, gpu_count=8)
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
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, gpu_count=bad)


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
    # the unpinned message names the authored ceiling that fell short, which is strictly more than
    # "nothing fits" -- the shared contract is that the shortfall is stated and a width is offered.
    assert "gpu.count=1 provides at most" in str(unpinned.value)

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

    with pytest.raises(ValueError, match=r"more than any .*combination") as exc:
        _offline_gpu_shape(RunConfig("Qwen/Qwen3.5-9B", "sft", 1, gpu_count=1))
    # the load-bearing half: no width is suggested, because none would work.
    assert "--gpus" not in str(exc.value)


def test_offline_quote_remedy_only_names_widths_a_provider_sells_freely():
    """An offline quote cannot promise a width whose SKU only a live catalog could confirm.

    A lambda-pinned quote may name the fitting width only as a catalog check, because lambda names
    the card count in the instance type and the offline path cannot know whether that type exists.
    It also cannot claim dropping the pin would help because it does not know the configured fleet.
    The runpod pool, where the count is a launch parameter, still gets its direct remedy.

    The catalog-check case must not fall through to "exceeds every GPU class" either: the run fits,
    so claiming it needs more than an 8-card combination is false and would send the user to shrink
    a run that is already small enough.
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
    with pytest.raises(
        ValueError, match="no available provider is confirmed to sell"
    ) as lambda_only:
        _offline_gpu_shape(RunConfig(gpu_count=1, provider="lambda", **shared))
    # the offline path cannot know an unpinned fleet, so it may name the width only as a check
    # against lambda's own live catalog rather than promise that dropping the pin makes it rentable.
    lambda_message = str(lambda_only.value)
    assert "Drop the provider pin" not in lambda_message
    assert (
        "Raise the card ceiling with `--gpus 2` to check it against their catalog" in lambda_message
    )
    assert "configure a provider that rents card counts directly (RunPod)" in lambda_message
    # the run fits; do not tell the user it exceeds every class.
    assert "more than any" not in lambda_message

    # the runpod pool still names its width: the rule is about how counts are sold, not a blanket
    # suppression of the remedy.
    with pytest.raises(ValueError, match=r"--gpus 2"):
        _offline_gpu_shape(RunConfig(gpu_count=1, provider="runpod", **shared))

    # a genuinely oversized run must STILL report exceeding every class, not the provider excuse.
    with pytest.raises(ValueError, match=r"more than any .*combination") as oversized:
        _offline_gpu_shape(
            RunConfig(
                gpu_count=1,
                provider="lambda",
                model_revision="",
                **{**shared, "seq_len": 1_000_000},
            )
        )
    assert "no available provider is confirmed to sell" not in str(oversized.value)


def test_quote_catalog_check_is_withheld_when_the_sft_width_would_not_launch():
    """Regression: the quote's catalog remedy kept crediting cards its allocator mirror does not.

    `_catalog_check_remedy` documents itself as the mirror of the allocator's `catalog_check_hint`,
    so the two must answer alike or `--cost` names a `--gpus N` that submit rejects. An unpacked sft
    run launches one rank however many cards are rented, so a width whose extra cards never join is
    not a check worth a provider round trip.

    The grpo control is the point: same pool, same pin, same shortfall, and it still names the width
    -- so this cannot pass by suppressing the remedy everywhere. Its batch has to be widened to earn
    that, because grpo bounds its width too now (see `rl_data_parallel_cards`): `RunConfig.batch_size`
    IS grpo's `prompts_per_step`, so at 1 prompt x 4 generations the control would clamp for the same
    reason as the sft arm and stop being a control at all.
    """
    from flash.cost.analytical import _offline_gpu_shape
    from flash.cost.types import RunConfig

    # the length is what makes this reachable: a batch-1 sft run needs LESS vram, so it only falls
    # through to a remedy once the shortfall survives the clamp. at 400k it exceeds every card.
    shared = {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "steps": 10,
        "seq_len": 400_000,
        "completion_len": 512,
        "batch_size": 1,
        "group_size": 4,
        "lora_rank": 16,
        "gpu_count": 1,
        "provider": "lambda",
    }
    with pytest.raises(ValueError, match="VRAM") as sft:
        _offline_gpu_shape(RunConfig(method="sft", **shared))
    assert "--gpus" not in str(sft.value), (
        "an sft run clamped to one rank gains nothing from a wider SKU, so asking lambda to "
        "confirm one is a round trip that cannot fix the quote"
    )

    # grpo whose step fills every rank still names the width, at the SAME shortfall: 8 prompts x 4
    # generations is 32 sequences, which divides every rentable count, so nothing clamps and the
    # remedy is reached exactly as before.
    with pytest.raises(ValueError, match="VRAM") as grpo:
        _offline_gpu_shape(RunConfig(method="grpo", **{**shared, "batch_size": 8}))
    assert "--gpus" in str(grpo.value)


def test_offline_exact_pin_on_a_fixed_count_provider_still_names_a_width_to_check():
    """An exact offline pin must get the same catalog check its non-exact sibling gets.

    ``_wider_shape_remedy`` drops classes whose providers name the count in the SKU, which left an
    exact lambda pin with an EMPTY remedy -- so it fell through to knob advice telling the user to
    shrink a run that already fits at a wider count. The identical run on runpod was told
    ``--gpus 2``. Same shortfall, opposite advice, decided only by how the provider sells counts.
    """
    from flash.cost.analytical import _offline_gpu_shape
    from flash.cost.types import RunConfig

    shared = {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "method": "sft",
        "steps": 10,
        "seq_len": 4096,
        "completion_len": 512,
        "batch_size": 8,
        "group_size": 4,
        "lora_rank": 16,
        "gpu_type": "H100",
        "gpu_count": 1,
    }

    with pytest.raises(ValueError, match=r"exact GPU 'H100' cannot fit this run") as pinned:
        _offline_gpu_shape(RunConfig(provider="lambda", **shared))
    message = str(pinned.value)
    assert "`--gpus 2`" in message
    assert "check it against their catalog" in message
    # a check, never a promise: nothing offline proved the 2-card SKU is sold.
    assert "it fits on" not in message
    # and it must NOT tell the user to shrink a run that fits at a width they can ask for.
    assert "Lower [train]" not in message

    # runpod keeps the PROVED remedy: the rule is about how counts are sold, not the message.
    with pytest.raises(ValueError, match=r"it fits on 2 cards"):
        _offline_gpu_shape(RunConfig(provider="runpod", **shared))

    # beyond every rentable width, knob advice is the honest answer and no width is named.
    with pytest.raises(ValueError, match=r"cannot fit this run") as unfittable:
        _offline_gpu_shape(
            RunConfig(
                provider="lambda",
                **{**shared, "seq_len": 262144, "batch_size": 1024, "lora_rank": 512},
            )
        )
    assert "--gpus" not in str(unfittable.value)
    assert "Lower [train]" in str(unfittable.value)


def test_multi_card_quote_states_pooled_vram_not_per_card():
    """A multi-card quote must not pair a per-card figure with a whole-run requirement.

    The requirement is a whole-run number and the fit gate values a shape with
    ``combined_vram_gb``, so rendering the per-card size beside it produced lines like
    "180 GB; run needs >= 199 GB" on a shape that had just PASSED the gate -- a quote that reads
    as a rejection of the hardware it recommends.
    """
    from flash.providers.core.base import Candidate
    from flash.providers.core.sharding import combined_vram_gb

    config = RunConfig("Qwen/Qwen3.5-9B", "grpo", 50, gpu_count=2)
    two = estimate_cost(config, allocation=Candidate("runpod", "H100", 3.29, 80, 2))
    pooled = int(combined_vram_gb(80, 2))
    assert two.offered_vram_gb == pooled
    assert pooled > two.gpu_vram_gb, "a 2-card shape must offer more than one of its cards"
    line = next(ln for ln in two.breakdown().splitlines() if ln.startswith("GPU"))
    assert f"{pooled} GB usable across 2x 80 GB" in line
    # the per-card number stays visible: it is the one a user checks against a provider listing.
    assert "80 GB" in line
    # whatever the requirement is, the quoted shape must not read as failing it.
    assert two.offered_vram_gb >= two.required_vram_gb


def test_single_card_quote_keeps_the_plain_vram_spelling():
    """One card offers exactly itself, so the pooled wording must not appear."""
    from flash.providers.core.base import Candidate

    one = estimate_cost(
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 50),
        allocation=Candidate("runpod", "H100", 3.29, 80, 1),
    )
    assert one.offered_vram_gb == one.gpu_vram_gb == 80
    line = next(ln for ln in one.breakdown().splitlines() if ln.startswith("GPU"))
    assert "80 GB; run needs >=" in line
    assert "usable across" not in line


def test_themed_panel_and_breakdown_agree_on_the_vram_clause():
    """The two renderers must not describe the same shape differently."""
    import os

    from flash.cli.ui import render
    from flash.providers.core.base import Candidate

    two = estimate_cost(
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 50, gpu_count=2),
        allocation=Candidate("runpod", "H100", 3.29, 80, 2),
    )
    clause = f"{two.offered_vram_gb} GB usable across 2x {two.gpu_vram_gb} GB"
    assert clause in two.breakdown()
    prior = os.environ.get("NO_COLOR")
    os.environ["NO_COLOR"] = "1"
    try:
        assert clause in render.cost_panel(two)
    finally:
        if prior is None:
            os.environ.pop("NO_COLOR", None)
        else:
            os.environ["NO_COLOR"] = prior


def test_panel_and_breakdown_agree_when_executed_width_is_below_the_billed_count():
    """The agreement must hold on the shape that can actually break it.

    The clause is computed from ``offered_vram_gb``, which values the EXECUTED width. Rendering it
    beside the billed count states a pooled figure the named cards do not add up to: three rented
    cards carrying a two-rank run read as "160 GB usable across 3x 80 GB". The twin above cannot
    catch this because both widths are 2 there, so the two spellings coincide.
    """
    import os

    from flash.cli.ui import render
    from flash.providers.core.base import Candidate

    narrow = RunConfig(
        "Qwen/Qwen3.5-9B", "sft", 10, gpu_count=3, batch_size=2, sft_retained_examples=2
    )
    est = estimate_cost(narrow, allocation=Candidate("runpod", "H100", 3.29, 80, 3))
    assert est.gpu_count == 3, "billed for all three rented cards"
    assert est.executed_gpu_count == 2, "only two ranks join the fsdp group"

    clause = f"{est.offered_vram_gb} GB usable across 2x {est.gpu_vram_gb} GB"
    assert clause in est.breakdown()
    prior = os.environ.get("NO_COLOR")
    os.environ["NO_COLOR"] = "1"
    try:
        panel = render.cost_panel(est)
    finally:
        if prior is None:
            os.environ.pop("NO_COLOR", None)
        else:
            os.environ["NO_COLOR"] = prior
    assert clause in panel, f"panel credited a width the pooled figure does not cover: {panel}"
    assert "across 3x" not in panel
    # the BILLED count still drives price: you pay for the idle card even though it holds nothing.
    assert "3x H100" in panel
    assert "per card" in panel


def test_remedy_pool_is_narrowed_by_a_hard_provider_pin_not_a_soft_preference():
    """A hard ``gpu.provider`` pin must narrow which classes can be widened.

    ``gpu.provider`` and ``gpu.providers`` are different, mutually-exclusive fields: the first is
    a hard pin, the second a soft preference. Passing only the soft list let a Lambda-pinned run
    borrow RunPod's freedom to rent arbitrary card counts and get sent to a wider SKU its pin may
    not carry.
    """
    from flash.providers.core.fit_errors import widenable_gpu_names

    assert widenable_gpu_names(("B200",), None) == ("B200",)
    assert widenable_gpu_names(("B200",), ("runpod",)) == ("B200",)
    # lambda names its card count in the instance type, so no wider B200 shape is freely rentable.
    assert widenable_gpu_names(("B200",), ("lambda",)) == ()


def test_offered_vram_credits_only_the_ranks_that_join_the_run():
    """A rented card that never enters the fsdp group contributes no memory.

    The allocator values a shape at its EXECUTED width (``_executed_width``), so quoting the
    billed count would advertise capacity the run does not have: an sft job whose one-row batch
    launches a single rank on two rented cards was told it had 130 GB when the gate valued 80.
    """
    from flash.cost.analytical import executed_gpu_count
    from flash.providers.core.base import Candidate
    from flash.providers.core.sharding import combined_vram_gb

    narrow = RunConfig("Qwen/Qwen3.5-9B", "sft", 10, batch_size=1, sft_retained_examples=1)
    est = estimate_cost(narrow, allocation=Candidate("runpod", "H100", 3.29, 80, 2))
    assert est.gpu_count == 2, "the job is still billed for both rented cards"
    assert executed_gpu_count(narrow, 2) == 1, "a one-row batch launches one rank"
    assert est.executed_gpu_count == 1
    # the quote must state the ONE card's memory the fit gate actually valued.
    assert est.offered_vram_gb == int(combined_vram_gb(80, 1)) == 80
    line = next(ln for ln in est.breakdown().splitlines() if ln.startswith("GPU"))
    assert "80 GB; run needs >=" in line
    assert "usable across" not in line, f"claimed pooled memory one rank cannot use: {line}"

    # a genuinely wide run is unaffected: both ranks join, so both cards are credited.
    wide = RunConfig("Qwen/Qwen3.5-9B", "grpo", 50, gpu_count=2)
    wide_est = estimate_cost(wide, allocation=Candidate("runpod", "H100", 3.29, 80, 2))
    assert wide_est.executed_gpu_count == 2
    assert wide_est.offered_vram_gb == int(combined_vram_gb(80, 2))
    assert "usable across 2x 80 GB" in wide_est.breakdown()


def test_unstamped_executed_count_falls_back_to_the_billed_count():
    """A hand-built estimate that never stamped the width must not read as a single rank."""
    from flash.providers.core.base import Candidate

    est = estimate_cost(
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 50, gpu_count=2),
        allocation=Candidate("runpod", "H100", 3.29, 80, 2),
    )
    unstamped = dataclasses.replace(est, executed_gpu_count=0)
    assert unstamped.joined_gpu_count == unstamped.gpu_count == 2
    assert unstamped.offered_vram_gb == est.offered_vram_gb
