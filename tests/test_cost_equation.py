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


def test_total_is_exactly_wall_hours_times_rate():
    """total_usd is EXACTLY wall-clock hours x $/hr -- no multiplier (assert exact, not approx,
    so any smuggled-in scaling fails)."""
    e = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 50))
    assert e.total_usd == e.wall_clock_hours * e.gpu_hourly_usd
    assert e.gpu_hourly_usd == gpu_hourly_usd(e.gpu)


def test_prices_at_static_rate():
    """Cost uses the static GPU registry rate directly."""
    from flash.providers.base import GPU_INFO

    for cls in ("RTX 5090", "A100 PCIe", "RTX 4090", "RTX A6000", "H100"):
        assert gpu_hourly_usd(cls) == GPU_INFO[cls].hourly_usd


def test_concurrent_reward_keeps_heavy_grpo_off_the_floor():
    """Heavy reward latency raises cost but doesn't explode: graders run concurrently (waves of
    16), not serially."""
    light = estimate_cost(
        RunConfig("openbmb/MiniCPM5-1B", "grpo", 100, reward_seconds_per_completion=0.05)
    )
    heavy = estimate_cost(
        RunConfig("openbmb/MiniCPM5-1B", "grpo", 100, reward_seconds_per_completion=3.0)
    )
    assert heavy.total_usd > light.total_usd
    assert not heavy.wall_capped  # concurrent grading keeps a small run under the wall cap
