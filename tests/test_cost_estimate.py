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
    assert RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, provider="Modal").provider == "modal"
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
        "modal",
        "owner/environment",
        16_000,
        (2, 5),
    )

    assert config.seq_len == 2048
    assert config.batch_size == 4
    assert config.lora_rank == 8
    assert config.max_wall_seconds == 3600
    assert config.provider == "modal"
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
        estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, provider="modal")).provider
        == "modal"
    )


def test_estimate_honors_exact_gpu_instead_of_cheaper_fit():
    unconstrained = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10))
    exact = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, gpu_type="H100"))

    assert unconstrained.gpu == "A100 SXM"
    assert exact.gpu == "H100"
    assert exact.gpu_hourly_usd > unconstrained.gpu_hourly_usd


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
            gpu=GpuSpec(provider="modal", type="H100", disk_gb=200),
        )
    )

    config = runconfig_from_spec(spec)
    assert config.provider == "modal"
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


def test_a100_sxm_40gb_has_real_tflops_not_default():
    from flash.cost.facts import _DEFAULT_TFLOPS, gpu_tflops

    assert gpu_tflops("A100 SXM 40GB") == gpu_tflops("A100 SXM")
    assert gpu_tflops("A100 SXM 40GB") != _DEFAULT_TFLOPS


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
    L4 quoted 8 cards for an 80 GB run instead of raising.
    """
    with pytest.raises(ValueError, match=r"exact GPU 'L4' cannot fit this run"):
        estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, gpu_type="L4"))


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
    one = estimate_cost(config, allocation=Candidate("modal", "H100", 3.29, 80, 1))
    two = estimate_cost(config, allocation=Candidate("modal", "H100", 3.29, 80, 2))
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
    two = estimate_cost(config, allocation=Candidate("modal", "H100", 3.29, 80, 2))
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
        allocation=Candidate("modal", "H100", 3.29, 80, 1),
    )
    assert one.offered_vram_gb == one.gpu_vram_gb == 80
    line = next(ln for ln in one.breakdown().splitlines() if ln.startswith("GPU"))
    assert "80 GB; run needs >=" in line
    assert "usable across" not in line


def test_breakdown_reports_the_joined_vram_clause():
    """The analytical breakdown describes the memory of the joined ranks."""
    from flash.providers.core.base import Candidate

    two = estimate_cost(
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 50, gpu_count=2),
        allocation=Candidate("modal", "H100", 3.29, 80, 2),
    )
    clause = f"{two.offered_vram_gb} GB usable across 2x {two.gpu_vram_gb} GB"
    assert clause in two.breakdown()


def test_breakdown_uses_executed_width_below_the_billed_count():
    """The memory clause uses the executed width even when more cards are billed.

    The clause is computed from ``offered_vram_gb``, which values the EXECUTED width. Rendering it
    beside the billed count states a pooled figure the named cards do not add up to: three rented
    cards carrying a two-rank run read as "160 GB usable across 3x 80 GB". The twin above cannot
    catch this because both widths are 2 there, so the two spellings coincide.
    """
    from flash.providers.core.base import Candidate

    narrow = RunConfig(
        "Qwen/Qwen3.5-9B", "sft", 10, gpu_count=3, batch_size=2, sft_retained_examples=2
    )
    est = estimate_cost(narrow, allocation=Candidate("modal", "H100", 3.29, 80, 3))
    assert est.gpu_count == 3, "billed for all three rented cards"
    assert est.executed_gpu_count == 2, "only two ranks join the fsdp group"

    clause = f"{est.offered_vram_gb} GB usable across 2x {est.gpu_vram_gb} GB"
    assert clause in est.breakdown()
    assert "across 3x" not in est.breakdown()


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
    est = estimate_cost(narrow, allocation=Candidate("modal", "H100", 3.29, 80, 2))
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
    wide_est = estimate_cost(wide, allocation=Candidate("modal", "H100", 3.29, 80, 2))
    assert wide_est.executed_gpu_count == 2
    assert wide_est.offered_vram_gb == int(combined_vram_gb(80, 2))
    assert "usable across 2x 80 GB" in wide_est.breakdown()


def test_unstamped_executed_count_falls_back_to_the_billed_count():
    """A hand-built estimate that never stamped the width must not read as a single rank."""
    from flash.providers.core.base import Candidate

    est = estimate_cost(
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 50, gpu_count=2),
        allocation=Candidate("modal", "H100", 3.29, 80, 2),
    )
    unstamped = dataclasses.replace(est, executed_gpu_count=0)
    assert unstamped.joined_gpu_count == unstamped.gpu_count == 2
    assert unstamped.offered_vram_gb == est.offered_vram_gb
