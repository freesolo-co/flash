"""Cost estimator invariants: the dollar figure is fully first-principles -- NO output
multiplier (the anti-reward-hacking invariant), priced at the realized market rate, with
concurrent reward grading. No network."""

from __future__ import annotations

import dataclasses

from flash.cost import RunConfig, estimate_cost
from flash.cost.facts import gpu_hourly_usd, realized_hourly_usd
from flash.cost.types import CostEstimate


def test_no_output_multiplier_field():
    """CostEstimate must not carry any calibration/scaling factor -- the hack is gone."""
    names = {f.name for f in dataclasses.fields(CostEstimate)}
    assert not (names & {"calibration_factor", "breakeven_factor", "scale"})


def test_total_is_exactly_wall_hours_times_rate():
    """The dollar figure is wall-clock hours x market $/hr -- nothing else, no factor.

    This is the anti-reward-hacking invariant, and it is EXACT by construction: ``total_usd``
    is computed as ``wall_clock_seconds / 3600 * hourly`` and ``wall_clock_hours`` divides the
    same seconds by the same 3600, so re-deriving the product reproduces ``total_usd`` to the
    bit. Assert exact equality (not ``approx``) so any smuggled-in multiplier/scaling — however
    small — fails the test instead of hiding under a tolerance.
    """
    rate = realized_hourly_usd("RTX 5090")
    cfg = RunConfig("Qwen/Qwen3.5-9B", "grpo", 50, gpu="RTX 5090")
    e = estimate_cost(cfg)
    assert e.total_usd == e.wall_clock_hours * rate
    assert e.gpu_hourly_usd == rate


def test_prices_at_realized_rate():
    """Observed classes price at the realized (spot/queue) rate, usually below list; an
    unobserved class falls back to the list price (no rate invented)."""
    for cls in ("RTX 5090", "A100 PCIe", "RTX 3090"):
        assert realized_hourly_usd(cls) < gpu_hourly_usd(cls)
    assert realized_hourly_usd("A40") == gpu_hourly_usd("A40")  # unobserved -> list
    # H100 has no clean realized rate (the lone sample was an anomalous ~$10/hr surge run), so it
    # falls back to its list price -- never the discarded outlier.
    assert realized_hourly_usd("H100") == gpu_hourly_usd("H100") == 3.29


def test_concurrent_reward_keeps_heavy_grpo_off_the_floor():
    """A heavy reward latency raises cost but must not explode: graders run CONCURRENTLY, so
    512 completions cost ceil(512/16)=32 waves x latency, not 512 x latency (which a serial
    model would push past the 24h wall cap)."""
    light = estimate_cost(
        RunConfig("openbmb/MiniCPM5-1B", "grpo", 100, reward_seconds_per_completion=0.05)
    )
    heavy = estimate_cost(
        RunConfig("openbmb/MiniCPM5-1B", "grpo", 100, reward_seconds_per_completion=3.0)
    )
    assert heavy.total_usd > light.total_usd
    assert not heavy.wall_capped  # concurrent grading keeps a small run under the wall cap
