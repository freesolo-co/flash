"""Charge fallbacks for workers that predate the training meter, and the cancel integration.

Metered runs are covered in tests/test_metered_billing.py. A worker without a meter (a run in
flight across the deploy that introduced metering) pays, when cancelled, the quote's share of its
completed steps, and when completed, the measured wall-clock cost of its terminal metrics.
"""

from __future__ import annotations

import json
import math
import os

import pytest

import flash.runner.accounting.costs as runner_costs
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.supervise.recovery as runner_recovery

SPEC = {
    "model": "Qwen/Qwen3.5-9B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {"epochs": 20, "max_examples": 20, "prompts_per_step": 20},
    "gpu": {},
}


def _spec():
    from flash.schema import spec_from_dict

    return spec_from_dict(SPEC, run_id="run-1")


def _write_terminal_steps(
    tmp_path,
    monkeypatch,
    spec,
    step,
    *,
    wall_seconds=1.0,
    allocated_provider=None,
    allocated_gpu=None,
    allocated_gpu_count=None,
):
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path))
    dest = runner_state.artifacts_dir(spec)
    os.makedirs(dest, exist_ok=True)
    metrics = {"step": step, "wall_seconds": wall_seconds}
    if allocated_provider is not None:
        metrics["allocated_provider"] = allocated_provider
    if allocated_gpu is not None:
        metrics["allocated_gpu"] = allocated_gpu
    if allocated_gpu_count is not None:
        metrics["allocated_gpu_count"] = allocated_gpu_count
    with open(os.path.join(dest, "metrics.json"), "w") as handle:
        json.dump(metrics, handle)


# --------------------------------------------------------------------------- legacy fallbacks


def _legacy_status(quote, *, state="cancelled"):
    # no training_timing in the heartbeat: a worker that predates the meter.
    return runner_state.RunStatus(
        run_id="r", state=state, spec={}, estimated_cost_usd=quote, last_heartbeat=None
    )


def test_a_legacy_cancel_pays_the_quotes_share_of_completed_steps():
    spec = _spec()  # 20 planned steps
    st = _legacy_status(8.0)
    assert runner_costs.cancelled_charge_usd(st, spec, steps=0) == 0.0
    assert runner_costs.cancelled_charge_usd(st, spec, steps=5) == pytest.approx(2.0)
    assert runner_costs.cancelled_charge_usd(st, spec, steps=20) == pytest.approx(8.0)
    # a step count beyond the plan never bills above the quote.
    assert runner_costs.cancelled_charge_usd(st, spec, steps=25) == pytest.approx(8.0)


@pytest.mark.parametrize("bad_quote", ["not-a-number", True, -1.0, math.inf, math.nan])
def test_a_legacy_cancel_with_an_invalid_quote_is_a_pricing_failure(bad_quote):
    st = _legacy_status(bad_quote)
    assert runner_costs.cancelled_charge_usd(st, _spec(), steps=10, fallback=3.25) == 3.25


def test_a_legacy_completion_pays_its_measured_wall_cost(tmp_path, monkeypatch):
    spec = _spec()
    _write_terminal_steps(tmp_path, monkeypatch, spec, 20, wall_seconds=3600.0)
    st = _legacy_status(105.6, state="done")
    # the fallback is the measured wall-clock cost the completion path computed, never the quote.
    assert runner_costs._status_estimated_charge(st, spec, fallback=4.43) == pytest.approx(4.43)


def test_actual_steps_run_reads_last_heartbeat_step():
    def st(hb):
        return runner_state.RunStatus(run_id="r", state="cancelled", spec={}, last_heartbeat=hb)

    assert runner_costs.actual_steps_run(st({"stage": "rl_step", "step": 7})) == 7
    # no heartbeat / setup stage -> 0 (cancelled during cold-start, no GPU training yet)
    assert runner_costs.actual_steps_run(st(None)) == 0
    assert runner_costs.actual_steps_run(st({"stage": "setup"})) == 0
    # training started but no step completed yet (the ~17-min first GRPO rollout emits no `step`) ->
    # floor to 1 so real GPU time isn't billed as $0.
    assert runner_costs.actual_steps_run(st({"stage": "rl_step", "step": 0})) == 1
    assert runner_costs.actual_steps_run(st({"stage": "rl_step"})) == 1
    assert runner_costs.actual_steps_run(st({"stage": "sft_step"})) == 1
    # Terminal `done` heartbeat: a cancel racing the DONE upload (done recorded, run not yet
    # transitioned) reads a STEPLESS done and floors a fully-trained run to 0. _publish_training_completion carries
    # opt_steps onto `done` so the true count bills.
    assert (
        runner_costs.actual_steps_run(st({"stage": "done"})) == 0
    )  # the bug the step guards against
    assert runner_costs.actual_steps_run(st({"stage": "done", "step": 12})) == 12


# --------------------------------------------------------------------------- cancel integration


def test_cancel_run_prorates_a_legacy_runs_quote_by_steps(monkeypatch, tmp_path):
    from flash.runner.supervise import deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_recovery, "_gc_run_sandboxes", lambda spec: None)
    spec = _spec()  # 20 planned steps
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="run-1",
            state="running",
            spec=spec.to_dict(),
            billing_context={"org_id": "o"},
            billing_state="pending",
            estimated_cost_usd=8.0,
            last_heartbeat={"stage": "rl_step", "step": 10},
        )
    )

    deploy.cancel_run("run-1")

    st = runner_status.get_status("run-1")
    assert st.state == "cancelled"
    assert st.cost_usd == pytest.approx(4.0)


def test_cancel_run_with_malformed_quote_still_settles_as_a_billing_failure(monkeypatch, tmp_path):
    # a nonnumeric persisted quote survives the tolerant status loader; the cancel must still reach
    # the cancelled transition and record the $0 pricing-failure diagnostic instead of raising.
    from flash.runner.supervise import deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_recovery, "_gc_run_sandboxes", lambda spec: None)
    spec = _spec()
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="run-1",
            state="running",
            spec=spec.to_dict(),
            billing_context={"org_id": "o"},
            billing_state="pending",
            estimated_cost_usd="not-a-number",
            last_heartbeat={"stage": "rl_step", "step": 10},
        )
    )

    deploy.cancel_run("run-1")

    st = runner_status.get_status("run-1")
    assert st.state == "cancelled"
    assert st.cost_usd == 0.0
    assert st.billing_state == "failed"


def test_cancel_run_before_any_step_is_free(monkeypatch, tmp_path):
    from flash.runner.supervise import deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_recovery, "_gc_run_sandboxes", lambda spec: None)
    spec = _spec()
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="run-1",
            state="running",
            spec=spec.to_dict(),
            billing_context={"org_id": "o"},
            billing_state="pending",
            last_heartbeat=None,  # cancelled during cold-start/setup, no training step reported
        )
    )

    deploy.cancel_run("run-1")

    st = runner_status.get_status("run-1")
    assert st.state == "cancelled"
    assert st.cost_usd == 0.0  # 0 steps -> $0
