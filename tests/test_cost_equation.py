"""Cost estimator invariants: the dollar figure is fully first-principles -- NO output
multiplier (the anti-reward-hacking invariant), priced at the static GPU rate, with concurrent
reward grading. No network."""

from __future__ import annotations

import dataclasses

from flash.cost import RunConfig, estimate_cost
from flash.cost.facts import gpu_hourly_usd
from flash.cost.types import CostEstimate


def test_no_output_multiplier_field():
    """CostEstimate must not carry any calibration/scaling factor -- the hack is gone."""
    names = {f.name for f in dataclasses.fields(CostEstimate)}
    assert not (names & {"calibration_factor", "breakeven_factor", "scale"})


def test_total_is_exactly_billable_train_hours_times_rate():
    """total_usd is EXACTLY training hours x $/hr -- no multiplier (assert exact, not approx,
    so any smuggled-in scaling fails)."""
    e = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 50))
    assert e.total_usd == e.billable_hours * e.gpu_hourly_usd
    assert e.billable_hours == e.train_seconds / 3600.0
    assert e.total_usd < e.wall_clock_hours * e.gpu_hourly_usd
    assert e.gpu_hourly_usd == gpu_hourly_usd(e.gpu)


def test_prices_at_static_rate():
    """Cost uses the static GPU registry rate directly."""
    from flash.providers.base import GPU_INFO

    for cls in ("RTX 5090", "A100 PCIe", "RTX 4090", "H100"):
        assert gpu_hourly_usd(cls) == GPU_INFO[cls].hourly_usd


def test_heavy_serial_reward_drives_a_small_grpo_run_into_the_wall_cap():
    """A slow grader dominates a GRPO run, and the estimate must SAY so rather than hide it.

    This previously asserted the opposite (`not heavy.wall_capped`) on the belief that graders run
    in waves of 16. They do not: both single-turn backends score a step's completions one at a
    time, and the verl bridge holds a lock to keep it that way. At the default shape that is
    64x8=512 completions, so a 3s judge costs 512*3 = 1536s of pure grading per step -- ~25min,
    and 100 of those steps cannot fit the 24h wall. Hitting the cap is the correct report; the old
    divisor made an infeasible run look comfortable.
    """
    light = estimate_cost(
        RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 100, reward_seconds_per_completion=0.05)
    )
    heavy = estimate_cost(
        RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 100, reward_seconds_per_completion=3.0)
    )
    assert heavy.total_usd > light.total_usd
    assert not light.wall_capped  # a fast grader still fits comfortably
    assert heavy.wall_capped  # a 3s judge over 512 completions/step does not
