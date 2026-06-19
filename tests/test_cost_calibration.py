"""Cost estimator: break-even calibration.

The analytical model runs ~1.4x high vs measured cost; ``calibration`` scales it so the
summed quote breaks even. These tests pin the published factors to the committed real-run
data, prove the portfolio centers, and check the wrapper/sweep wiring. No network.
"""

from __future__ import annotations

import pytest

from flash.cost import (
    BREAKEVEN_FACTORS,
    RunConfig,
    breakeven_estimate,
    breakeven_factor_from_real_runs,
    environment_cost_sweep,
    estimate_cost,
    verify_centering,
)
from flash.cost.calibration import BREAKEVEN_FACTOR_GLOBAL, breakeven_factor


def test_published_factors_match_committed_data():
    """The hardcoded factors must equal Σmeasured/Σanalytical over the committed runs."""
    recomputed = breakeven_factor_from_real_runs()
    assert recomputed["sft"] == pytest.approx(BREAKEVEN_FACTORS["sft"], abs=1e-3)
    assert recomputed["grpo"] == pytest.approx(BREAKEVEN_FACTORS["grpo"], abs=1e-3)
    assert recomputed["global"] == pytest.approx(BREAKEVEN_FACTOR_GLOBAL, abs=1e-3)


def test_grpo_is_overestimated_more_than_sft():
    """GRPO's extra rollout/reward/vLLM-init is over-estimated harder -> smaller factor."""
    assert BREAKEVEN_FACTORS["grpo"] < BREAKEVEN_FACTORS["sft"] < 1.0


def test_portfolio_centers_to_break_even():
    """Per-method calibration makes Σ quote == Σ measured (ratio ~1.0), per method + total."""
    cen = verify_centering()
    for group in ("sft", "grpo", "all"):
        assert cen[group]["ratio"] == pytest.approx(1.0, abs=0.01), group


def test_breakeven_estimate_applies_factor():
    cfg = RunConfig("Qwen/Qwen3.5-9B", "grpo", 150, environment="gsm8k")
    raw = estimate_cost(cfg)
    cal = breakeven_estimate(cfg)
    factor = breakeven_factor("grpo")
    assert cal.calibration_factor == pytest.approx(factor)
    assert cal.total_usd == pytest.approx(raw.total_usd * factor)
    # the raw reference is recoverable, and the un-calibrated model is unchanged.
    assert cal.total_usd / cal.calibration_factor == pytest.approx(raw.total_usd)
    assert raw.calibration_factor == 1.0


def test_breakeven_quote_is_cheaper_than_raw():
    cfg = RunConfig("Qwen/Qwen3.5-4B", "sft", 200)
    assert breakeven_estimate(cfg).total_usd < estimate_cost(cfg).total_usd


def test_breakdown_shows_calibration_only_when_applied():
    cfg = RunConfig("Qwen/Qwen3.5-4B", "grpo", 150)
    assert "Break-even" in breakeven_estimate(cfg).breakdown()
    assert "Break-even" not in estimate_cost(cfg).breakdown()


def test_unknown_method_falls_back_to_global():
    # method is normalized to sft|grpo, so an unmapped key uses the global fallback.
    assert breakeven_factor("nope") == BREAKEVEN_FACTOR_GLOBAL


def test_environment_sweep_reward_tier_orders_cost():
    """A heavier reward grader costs more (until the wall cap binds)."""
    rows = [r for r in environment_cost_sweep() if r["raw_usd"] is not None]
    by_env = {(r["model"], r["environment"]): r for r in rows}
    model = "openbmb/MiniCPM5-1B"
    trivial = by_env[(model, "openai/gsm8k")]["raw_usd"]
    medium = by_env[(model, "primeintellect/web-search")]["raw_usd"]
    assert trivial < medium  # trivial (0.01s) cheaper than medium (0.6s) reward


def test_environment_sweep_has_averages():
    rows = environment_cost_sweep()
    assert rows[-1]["environment"] == "GRAND AVERAGE"
    assert rows[-1]["breakeven_usd"] > 0
