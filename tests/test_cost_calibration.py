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
from flash.cost.facts import gpu_hourly_usd, realized_hourly_usd
from flash.cost.types import CostEstimate


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
    """Observed classes are priced at the realized rate (usually below list -- the spot
    discount -- but not always: a surge-priced class can bill ABOVE its static list)."""
    for cls in ("RTX 5090", "A100 PCIe", "RTX 3090"):
        assert realized_hourly_usd(cls) < gpu_hourly_usd(cls)
    # the realized median is an empirical observation, not list +/- a fixed discount, so an
    # observed class CAN bill above its static list when the market is tight (the H100 GRPO
    # run billed ~$10/hr against a $3.29 list -- so the equation must price it at ~$10, not
    # ~$3.29, or it underquotes ~3x).
    assert realized_hourly_usd("H100") > gpu_hourly_usd("H100")
    # an unobserved class (no measured runs) falls back to the list price (no rate invented)
    assert realized_hourly_usd("A40") == gpu_hourly_usd("A40")


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


def test_measured_rows_graded_on_the_gpu_they_actually_ran_on():
    # The grader must price each measured run on its RECORDED card -- the run demonstrably ran
    # there. The offline VRAM heuristic over-estimates some real GRPO rows (e.g. Qwen3.5-4B
    # 64x8 completions sizes to ~35 GB > the RTX 5090's 32 GB), and the FORWARD pick would drop
    # the 5090 pin for a cheaper/larger card -- which would grade the measured 5090 bill against
    # a DIFFERENT GPU's price. verify_accuracy passes pin_must_fit=False so that can't happen.
    from flash.cost.analytical import select_gpu
    from flash.cost.calibration import _config_of, _load_runs, _ran_its_work

    affected = []  # real runs whose forward pick would diverge from the recorded card
    for r in _load_runs():
        if not _ran_its_work(r):
            continue
        cfg = _config_of(r)
        forward, _ = select_gpu(cfg, pin_must_fit=True)
        graded, _ = select_gpu(cfg, pin_must_fit=False)
        # The graded card is ALWAYS the one the run actually ran on (a record of fact).
        assert graded == r["gpu"], f"{r['run_id']} graded on {graded}, ran on {r['gpu']}"
        if forward != r["gpu"]:
            affected.append(r["run_id"])
    # The two real RTX-5090 GRPO rows whose VRAM estimate exceeds 32 GB are exactly the rows
    # the forward pick would have mis-graded -- the regression this fix guards against.
    assert {"autoslm-1781832449-942dc159", "autoslm-1781845796-30891f05"} <= set(affected)


def test_fit_constants_realized_rates_match_hardcoded():
    """The realized $/hr the equation prices at must track the measured billing data."""
    from flash.cost.facts import REALIZED_HOURLY_USD

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
