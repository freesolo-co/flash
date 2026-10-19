"""The per-step floor: work the FLOPs terms do not account for.

A GRPO step has three phases with no term in the cost model. Measured by verl's own `timing_s/*`
on real hardware (Qwen3.5-0.8B, RTX 4090, 8 steps, 100.0% of the step accounted):

    update_actor      62.1%   -> update_s
    old_log_prob      17.9%   -> NOTHING
    gen               15.5%   -> gen_s
    update_weights     3.0%   -> NOTHING
    save_checkpoint    1.5%   -> NOTHING
    reward             0.0%   -> reward_s (which billed 1.0s/completion)

The fake reward wall was numerically impersonating the gap -- both scale with completion count,
which is why the model looked calibrated while both halves were wrong.

Those three phases are the smaller half of it. On that arm the residual over the modelled gpu
seconds is 44.74s: 11.62s is the unmodelled phases, and 33.12s is update_actor + gen running
40.19s against a modelled 7.06s. So the floor has two real parts -- a fixed per-step overhead the
peak-FLOPs model misses at 0.8B-4B, and work that grows with the rollout batch -- which is why it
is an intercept PLUS a slope rather than either alone.

On top of that pooled line, a card carries a measured offset on the INTERCEPT only. 76% of the
variance on a matched shape is between cards and card means span 3.48x, so the card signal is
real; but per-card SLOPES collapse (0/6 on gens=256) because most cards have too few distinct
completion counts to determine one. An unmeasured card gets the pooled line and nothing else.
"""

from __future__ import annotations

import pytest

from flash.cost.analytical import (
    STEP_FLOOR_BASE_SECONDS,
    STEP_FLOOR_CARD_OFFSET_SECONDS,
    STEP_FLOOR_MIN_ARMS_FOR_OFFSET,
    STEP_FLOOR_SECONDS_PER_COMPLETION,
    seconds_per_step,
    step_floor_seconds,
    step_seconds_split,
)
from flash.cost.types import RunConfig

CARDS = ["A100 PCIe", "B200", "H100", "H200", "RTX 4090", "RTX 5090"]
# Cards the campaign measured but that did NOT earn an offset: 2 arms each, under the min-arm gate.
# RTX Pro 6000's own replicate spread on one identical config is 2.37x, so its residual is noise.
UNDER_SAMPLED = ["A100 SXM", "RTX Pro 6000"]


def _config(method="grpo", **overrides):
    base = {
        "model_id": "Qwen/Qwen3.5-9B",
        "method": method,
        "steps": 20,
        "seq_len": 2048,
        "completion_len": 1024,
        "batch_size": 8,
        "group_size": 4,
        "gpu_type": "H100",
        "provider": "runpod",
    }
    base.update(overrides)
    return RunConfig(**base)


def _pooled(completions):
    return STEP_FLOOR_BASE_SECONDS + STEP_FLOOR_SECONDS_PER_COMPLETION * completions


@pytest.mark.parametrize("gpu", [*UNDER_SAMPLED, "L40S", "some-unreleased-card", ""])
def test_an_unmeasured_card_gets_the_pooled_floor(gpu):
    """The offsets are measured corrections for cards the campaign covered, not a model of
    hardware. Leave-one-CARD-out scores 35/56 with offsets and 35/56 without -- identical, because
    a held-out card has no offset -- so guessing one for an unseen card would be invention."""
    assert step_floor_seconds(gpu, 32) == pytest.approx(_pooled(32))


def test_a_measured_card_is_actually_offset_from_the_pooled_floor():
    """Mutation guard on the table itself: if every offset were dropped (or the lookup stopped
    being applied) the floor would collapse to pooled everywhere and the suite must fail."""
    offsets = {c: step_floor_seconds(c, 32) - _pooled(32) for c in STEP_FLOOR_CARD_OFFSET_SECONDS}
    assert any(abs(v) > 1.0 for v in offsets.values()), offsets
    for card, expected in STEP_FLOOR_CARD_OFFSET_SECONDS.items():
        assert step_floor_seconds(card, 32) - _pooled(32) == pytest.approx(expected)


def test_b200_and_h200_get_the_same_floor():
    """B200 training is H200-class on portable kernels (Flash caps it at 550 of its 2250 peak
    TFLOPS) at a higher $/hr, so B200 must never be quoted faster or cheaper for the same run.
    Fitted independently these two invert: the measured H200/B200 ratio is 2.53x at 32 completions
    but 0.92x at 256 -- the ordering REVERSES -- on n=2 replicates whose own spread is 1.29x. They
    share the slower offset by construction so no fit can break the ranking."""
    for completions in (0, 16, 32, 256, 4096):
        assert step_floor_seconds("B200", completions) == step_floor_seconds("H200", completions)


@pytest.mark.parametrize("gpu", CARDS)
def test_the_offset_moves_the_intercept_and_never_the_slope(gpu):
    """Per-card slopes were measured and rejected: 6 of 8 cards have fewer than 3 distinct
    completion counts, so an intercept+slope per card is exactly determined with zero residual
    degrees of freedom -- it reproduces its training arms and scores 0/6 out of sample.

    Asserted against the pooled card's slope, not against the constant, so a mutation that zeroes
    the constant cannot satisfy this by moving both sides at once.
    """
    slope = (step_floor_seconds(gpu, 200) - step_floor_seconds(gpu, 100)) / 100
    pooled_slope = (step_floor_seconds("L40S", 200) - step_floor_seconds("L40S", 100)) / 100
    assert slope == pytest.approx(pooled_slope)
    assert slope == pytest.approx(STEP_FLOOR_SECONDS_PER_COMPLETION)
    assert slope > 0.0


def test_no_shipped_card_needs_the_negative_clamp():
    """The clamp must not be load-bearing for anything shipped: an offset that large would mean
    the card's own residual overwhelmed the pooled intercept, which is a fit to re-examine, not a
    number to silently floor at zero."""
    for card in STEP_FLOOR_CARD_OFFSET_SECONDS:
        assert step_floor_seconds(card, 0) > 0.0
        assert STEP_FLOOR_CARD_OFFSET_SECONDS[card] > -STEP_FLOOR_BASE_SECONDS


def test_an_extreme_offset_clamps_at_zero_rather_than_crediting_the_run(monkeypatch):
    """Arms the clamp, which no shipped offset reaches. A future table edit that drove the floor
    negative would credit a run for unmodelled work and could quote a big rollout cheaper than a
    small one, so the guard needs an input that actually triggers it."""
    import flash.cost.analytical as analytical

    monkeypatch.setattr(
        analytical,
        "STEP_FLOOR_CARD_OFFSET_SECONDS",
        {**STEP_FLOOR_CARD_OFFSET_SECONDS, "H100": -10_000.0},
    )
    assert analytical.step_floor_seconds("H100", 32) == 0.0
    assert analytical.step_floor_seconds("H100", 0) == 0.0


def test_under_sampled_cards_are_kept_out_of_the_table():
    """The min-arm gate is what makes the offsets hold up: including cards with 2 arms scores
    41/56 on held-out configs against 42/56 excluding them, and their residuals are replicate
    noise rather than card signal."""
    assert STEP_FLOOR_MIN_ARMS_FOR_OFFSET >= 3
    for card in UNDER_SAMPLED:
        assert card not in STEP_FLOOR_CARD_OFFSET_SECONDS


def test_the_floor_has_both_a_fixed_and_a_per_completion_part():
    """Either half alone fails a whole class of runs. A pure constant scores 0/6 on gens=256 arms
    (worst 4.23x) and 1/43 on gens=32 when it cannot fit that class; a pure per-completion slope
    with no intercept drops the fixed overhead and scores 32/56 against 38/56.

    The per-completion assertion uses the measured 0.830s/completion rather than the shipped
    constant: comparing the slope against the constant that defines it is unfalsifiable, since
    zeroing the constant zeroes both sides of the equality.
    """
    assert step_floor_seconds("L40S", 0) == pytest.approx(STEP_FLOOR_BASE_SECONDS)
    assert STEP_FLOOR_BASE_SECONDS > 1.0
    grew = step_floor_seconds("L40S", 100) - step_floor_seconds("L40S", 0)
    assert grew == pytest.approx(100 * STEP_FLOOR_SECONDS_PER_COMPLETION)
    assert grew == pytest.approx(80.5, rel=0.15)


@pytest.mark.parametrize("gpu", [*CARDS, "L40S"])
@pytest.mark.parametrize("completions", [-5, 0, 1, 32, 256, 4096])
def test_the_floor_is_never_negative_and_never_shrinks(gpu, completions):
    """A negative or shrinking floor would credit a run for generating more, which is backwards
    and would let a large rollout batch quote cheaper than a small one."""
    value = step_floor_seconds(gpu, completions)
    assert value >= 0.0
    assert value >= step_floor_seconds(gpu, max(0, completions - 1))


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_a_bigger_rollout_batch_pays_a_bigger_floor(method):
    """Measured: the floor is 77.2s at 32 completions and 230.5s at 256. It tracks the rollout
    batch because old_log_prob is a forward pass over exactly that batch."""
    small = seconds_per_step(_config(method, batch_size=2, group_size=2), "H100")
    large = seconds_per_step(_config(method, batch_size=8, group_size=8), "H100")
    assert large > small


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_the_floor_is_gpu_time_not_overhead(method):
    """old_log_prob and update_weights run ON the card, so the floor belongs to the gpu-bound half
    of the split. Filing it as fixed overhead would under-report gpu utilisation and mislead the
    allocator about how much of a step sharding can actually shorten."""
    config = _config(method)
    completions = (config.batch_size or 1) * (config.group_size or 1)
    gpu_bound, fixed = step_seconds_split(config, "H100")
    assert gpu_bound >= step_floor_seconds("H100", completions)
    assert fixed < step_floor_seconds("H100", completions)


def test_removing_the_floor_shortens_every_step(monkeypatch):
    """Mutation guard: if the floor stops being added, this test must fail. Without it the whole
    correction can be deleted and the suite stays green, which is how the original gap survived."""
    import flash.cost.analytical as analytical

    config = _config()
    completions = (config.batch_size or 1) * (config.group_size or 1)
    with_floor = seconds_per_step(config, "H100")
    monkeypatch.setattr(analytical, "step_floor_seconds", lambda _gpu, _n: 0.0)
    without = seconds_per_step(config, "H100")
    assert with_floor - without == pytest.approx(step_floor_seconds("H100", completions))


def test_the_card_reaches_the_floor_through_the_real_quote_path(monkeypatch):
    """The offsets are worthless if the quote path passes a card the lookup never sees. Two cards
    with different offsets must produce different step times end to end, not just in isolation."""
    seen = []
    real = step_floor_seconds

    def spy(gpu, completions):
        seen.append(gpu)
        return real(gpu, completions)

    import flash.cost.analytical as analytical

    monkeypatch.setattr(analytical, "step_floor_seconds", spy)
    fast = seconds_per_step(_config(gpu_type="RTX 5090"), "RTX 5090")
    slow = seconds_per_step(_config(gpu_type="H200"), "H200")
    assert "RTX 5090" in seen
    assert "H200" in seen
    assert fast != slow


# --- the floor under sharding -------------------------------------------------------------------
# Only ~80% of the floor shards. update_weights is a weight copy every rank pays and save_checkpoint
# is disk i/o; dividing them by the card count credits a wide run with time it still spends.


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_part_of_the_floor_survives_any_card_count(method):
    """The unshardable phases are a floor under the floor.

    Whatever the card count, a grpo/opd step can never drop below the part of the floor every rank
    pays. Dividing the whole floor makes the step tend to (gpu_bound/N + fixed), which understates
    a wide run without bound as N grows.
    """
    from flash.cost.analytical import (
        STEP_FLOOR_SHARDED_FRACTION,
        _step_floor_seconds_for,
        sharded_step_seconds,
        step_seconds_split,
    )

    config = _config(method)
    _gpu_bound, fixed = step_seconds_split(config, "H100")
    floor = _step_floor_seconds_for(config, "H100")
    unshardable = floor * (1.0 - STEP_FLOOR_SHARDED_FRACTION)
    assert unshardable > 0.0, "a floor that fully shards makes this test vacuous"

    for count in (1, 2, 4, 8, 64):
        sps = sharded_step_seconds(config, "H100", count)
        assert sps >= fixed + unshardable - 1e-9, (
            f"{method} at {count} cards quotes {sps:.3f}s, below the {fixed + unshardable:.3f}s "
            "every rank pays regardless of width"
        )


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_sharding_the_whole_floor_would_under_quote_a_wide_run(method, monkeypatch):
    """Mutation guard on STEP_FLOOR_SHARDED_FRACTION.

    Setting the fraction to 1.0 restores the old divide-everything behaviour. That must change the
    multi-card quote (and only the multi-card quote) or the constant is not load-bearing. Guards the
    exact defect: the whole floor sitting in the gpu-bound half and being divided entire.
    """
    import flash.cost.analytical as analytical

    config = _config(method)
    corrected = {n: analytical.sharded_step_seconds(config, "H100", n) for n in (1, 2, 4, 8)}
    monkeypatch.setattr(analytical, "STEP_FLOOR_SHARDED_FRACTION", 1.0)
    naive = {n: analytical.sharded_step_seconds(config, "H100", n) for n in (1, 2, 4, 8)}

    assert naive[1] == pytest.approx(corrected[1]), "one card must be unaffected by the split"
    for n in (2, 4, 8):
        assert naive[n] < corrected[n], (
            f"{method} at {n} cards: dividing the whole floor must under-quote against the split"
        )
    # and the error grows with width -- that is why it matters for wide runs specifically
    assert (corrected[8] - naive[8]) > (corrected[2] - naive[2])


def test_sft_is_untouched_by_the_floor_split():
    """sft runs no rollout, so it has no floor and must shard exactly as before.

    Without this, a future change could route sft through the grpo/opd correction and silently
    inflate every sft quote -- the split is a rollout-only correction.
    """
    from flash.cost.analytical import (
        _step_floor_seconds_for,
        multi_card_speedup,
        sharded_step_seconds,
        step_seconds_split,
    )

    config = _config("sft")
    assert _step_floor_seconds_for(config, "H100") == 0.0
    gpu_bound, fixed = step_seconds_split(config, "H100")
    for count in (1, 2, 4, 8):
        plain = gpu_bound / multi_card_speedup(count, "H100") + fixed
        assert sharded_step_seconds(config, "H100", count) == pytest.approx(plain)


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_multi_card_still_beats_single_card(method):
    """The correction must not overshoot into modelling extra cards as useless.

    Sharding still has to pay off: each added card must strictly shorten the step while the
    shardable part dominates. A split that swallowed the whole benefit would make the allocator
    refuse cards that genuinely help.
    """
    from itertools import pairwise

    from flash.cost.analytical import sharded_step_seconds

    config = _config(method)
    seconds = [sharded_step_seconds(config, "H100", n) for n in (1, 2, 4, 8)]
    for narrow, wide in pairwise(seconds):
        assert wide < narrow, f"{method}: adding cards must still shorten the step ({seconds})"


def test_grpo_estimate_has_no_reward_latency_allowance():
    """reward execution is customer code, so neither its identity nor latency can move the quote."""
    from dataclasses import fields, replace

    config = _config(method="grpo", environment="github:org/fast-reward")
    names = {field.name for field in fields(RunConfig)}
    assert not any("reward" in name and "model" not in name for name in names)
    fast = seconds_per_step(config, "H100")
    slow_identity = seconds_per_step(
        replace(config, environment="github:org/arbitrarily-slow-reward"),
        "H100",
    )
    assert slow_identity == fast
