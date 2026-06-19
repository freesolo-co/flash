"""Cost estimator: the equation is fully first-principles -- NO output multiplier.

These tests pin the anti-reward-hacking invariant (the dollar figure is wall x rate, with
no factor scaling the result), verify the equation prices at the realized market rate +
concurrent reward, and exercise the accuracy grader against measured runs. No network.
"""

from __future__ import annotations

import dataclasses

import pytest

from flash.cost import (
    RunConfig,
    environment_cost_sweep,
    estimate_cost,
    fit_constants,
    verify_accuracy,
)
from flash.cost.estimate import CostEstimate
from flash.cost.hardware import gpu_hourly_usd, realized_hourly_usd


def test_no_output_multiplier_field():
    """CostEstimate must not carry any calibration/scaling factor -- the hack is gone."""
    names = {f.name for f in dataclasses.fields(CostEstimate)}
    assert not (names & {"calibration_factor", "breakeven_factor", "scale"})


def test_total_is_exactly_wall_hours_times_rate():
    """The dollar figure is wall-clock hours x market $/hr -- nothing else, no factor."""
    cfg = RunConfig("Qwen/Qwen3.5-9B", "grpo", 50, environment="gsm8k", gpu="RTX 5090")
    e = estimate_cost(cfg)
    assert e.total_usd == pytest.approx(e.wall_clock_hours * realized_hourly_usd("RTX 5090"))
    assert e.gpu_hourly_usd == pytest.approx(realized_hourly_usd("RTX 5090"))


def test_prices_at_realized_rate_below_list():
    """Observed classes are priced at the spot/queue rate, which is below on-demand list."""
    for cls in ("RTX 5090", "A100 PCIe", "RTX 3090"):
        assert realized_hourly_usd(cls) < gpu_hourly_usd(cls)
    # an unobserved class falls back to the list price (no spot discount invented)
    assert realized_hourly_usd("H100") == gpu_hourly_usd("H100")


def test_concurrent_reward_keeps_heavy_grpo_off_the_floor():
    """A heavy-reward env must cost more than a trivial one but not explode (serial model
    would put 512 completions x 3s/step past the cap)."""
    trivial = estimate_cost(RunConfig("openbmb/MiniCPM5-1B", "grpo", 100, environment="gsm8k"))
    heavy = estimate_cost(RunConfig("openbmb/MiniCPM5-1B", "grpo", 100, environment="swe/code-exec"))
    assert heavy.total_usd > trivial.total_usd
    assert not heavy.wall_capped  # concurrent grading keeps a small run under the wall cap


def test_verify_accuracy_reports_unforced_bias():
    acc = verify_accuracy()
    assert acc["all"]["n"] >= 20
    # honest scorecard: bias is a real number near (not pinned to) 1.0, error is positive
    assert 0.5 < acc["all"]["agg_bias"] < 1.5
    assert acc["all"]["median_ape_pct"] > 0
    assert 0.0 <= acc["all"]["within_33pct"] <= 1.0


def test_fit_constants_realized_rates_match_hardcoded():
    """The realized $/hr the equation prices at must track the measured billing data."""
    from flash.cost.hardware import REALIZED_HOURLY_USD

    fit = fit_constants()["realized_hourly_usd"]
    for cls, rate in REALIZED_HOURLY_USD.items():
        if cls in fit:
            assert rate == pytest.approx(fit[cls], abs=0.05), cls


def test_environment_sweep_reward_tier_orders_cost():
    rows = [r for r in environment_cost_sweep() if r["gpu"]]
    by = {(r["model"], r["environment"]): r for r in rows}
    m = "openbmb/MiniCPM5-1B"
    assert by[(m, "openai/gsm8k")]["usd"] < by[(m, "primeintellect/web-search")]["usd"]


def test_environment_sweep_has_grand_average():
    rows = environment_cost_sweep()
    assert rows[-1]["environment"] == "GRAND AVERAGE"
    assert rows[-1]["usd"] > 0
