"""Cost estimator invariants: the dollar figure is fully first-principles, with no output
multiplier, priced at the static GPU rate. No network."""

from __future__ import annotations

import dataclasses

from flash.cost import RunConfig, estimate_cost
from flash.cost.types import CostEstimate
from flash.providers.core.base import GPU_INFO


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
    assert e.gpu_hourly_usd == GPU_INFO[e.gpu].hourly_usd
