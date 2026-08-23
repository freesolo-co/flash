"""Cost estimator invariants: the dollar figure is fully first-principles, with no output
multiplier, priced at the static GPU rate. No network."""

from __future__ import annotations

import dataclasses
import json

import pytest

from flash import runner
from flash.cost import RunConfig, estimate_cost
from flash.cost.types import CostEstimate
from flash.providers.base import GPU_INFO


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


def _quoted_grpo_spec_and_status():
    from flash.cost.spec import estimate_for_spec
    from flash.providers.base import Allocation
    from flash.schema import spec_from_dict

    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
            "train": {
                "epochs": 1,
                "max_examples": 16,
                "prompts_per_step": 8,
                "group_size": 4,
            },
            "gpu": {"type": "H100", "count": 2},
        },
        run_id="billing-window",
    )
    allocation = Allocation("runpod", "H100", 3.25, 0, (), gpu_count=2)
    estimate = estimate_for_spec(spec, allocation=allocation)
    status = runner.RunStatus(
        run_id=spec.run_id,
        state="done",
        spec=spec.to_dict(),
        estimated_cost_usd=estimate.total_usd,
        remote={
            "provider": "runpod",
            "allocated_gpu": "H100",
            "allocated_gpu_count": 2,
        },
    )
    return spec, status, estimate


def _write_settle_metrics(tmp_path, monkeypatch, spec, **metrics) -> None:
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path))
    dest = runner.artifacts_dir(spec)
    import os

    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "metrics.json"), "w") as handle:
        json.dump(
            {
                "step": 1,
                "allocated_provider": "runpod",
                "allocated_gpu": "H100",
                "allocated_gpu_count": 2,
                **metrics,
            },
            handle,
        )


def test_completed_charge_excludes_measured_init_and_reward_exactly(tmp_path, monkeypatch):
    spec, status, estimate = _quoted_grpo_spec_and_status()
    _write_settle_metrics(
        tmp_path,
        monkeypatch,
        spec,
        framework_init_seconds=12.0,
        reward_seconds=3.0,
    )

    charge = runner._status_estimated_charge(status, spec)

    expected = estimate.total_usd - (12.0 + 3.0) / 3600.0 * estimate.gpu_hourly_usd * 2
    assert charge == pytest.approx(expected)
    assert 0.0 <= charge <= estimate.total_usd


def test_completed_charge_is_zero_for_zero_steps_and_clamped_at_zero(tmp_path, monkeypatch):
    spec, status, estimate = _quoted_grpo_spec_and_status()
    _write_settle_metrics(
        tmp_path,
        monkeypatch,
        spec,
        step=0,
        framework_init_seconds=1.0,
        reward_seconds=0.0,
    )
    assert runner._status_estimated_charge(status, spec) == 0.0

    _write_settle_metrics(
        tmp_path,
        monkeypatch,
        spec,
        framework_init_seconds=estimate.train_seconds * 2,
        reward_seconds=estimate.train_seconds * 2,
    )
    assert runner._status_estimated_charge(status, spec) == 0.0

    _write_settle_metrics(
        tmp_path,
        monkeypatch,
        spec,
        framework_init_seconds=-100.0,
        reward_seconds=-25.0,
    )
    assert runner._status_estimated_charge(status, spec) == estimate.total_usd


def test_completed_charge_without_measurements_preserves_quote(tmp_path, monkeypatch):
    spec, status, estimate = _quoted_grpo_spec_and_status()
    _write_settle_metrics(tmp_path, monkeypatch, spec, wall_seconds=123.0)

    assert runner._status_estimated_charge(status, spec) == estimate.total_usd


def test_completed_charge_uses_live_whole_instance_rate(tmp_path, monkeypatch):
    spec, status, _estimate = _quoted_grpo_spec_and_status()
    status.estimated_cost_usd = 10.0
    status.remote = {
        "provider": "vast",
        "allocated_gpu": "H100",
        "allocated_gpu_count": 2,
        "hourly_usd": 7.0,
    }
    _write_settle_metrics(
        tmp_path,
        monkeypatch,
        spec,
        allocated_provider="vast",
        framework_init_seconds=36.0,
        reward_seconds=0.0,
    )
    monkeypatch.setattr(
        runner,
        "_gpu_rate",
        lambda *_args: pytest.fail("live instance rate must not be replaced by a static rate"),
    )

    assert runner._status_estimated_charge(status, spec) == pytest.approx(10.0 - 36.0 / 3600 * 7.0)


def test_near_zero_reward_latency_is_materially_neutral(tmp_path, monkeypatch):
    spec, status, estimate = _quoted_grpo_spec_and_status()
    _write_settle_metrics(
        tmp_path,
        monkeypatch,
        spec,
        framework_init_seconds=60.0,
        reward_seconds=0.0,
    )
    zero_reward = runner._status_estimated_charge(status, spec)
    _write_settle_metrics(
        tmp_path,
        monkeypatch,
        spec,
        framework_init_seconds=60.0,
        reward_seconds=1e-6,
    )
    fast_reward = runner._status_estimated_charge(status, spec)

    assert fast_reward <= zero_reward <= estimate.total_usd
    assert zero_reward - fast_reward == pytest.approx(1e-6 / 3600.0 * estimate.gpu_hourly_usd * 2)
