"""Customer-charge pricing: a completed run is charged its QUOTE (the flash.cost estimate at planned
steps); a run cancelled mid-training is re-priced to the SAME estimate at the steps it actually ran."""

from __future__ import annotations

import flash.runner as runner

SPEC = {
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {"steps": 20, "hf_repo": "org/test-runs"},
    "gpu": {"type": "RTX 5090"},
}


def _spec():
    from flash.schema import spec_from_dict

    return spec_from_dict(SPEC, run_id="run-1")


# --------------------------------------------------------------------------- pricing helper


def test_charge_usd_for_spec_is_the_submit_quote():
    from flash.cost.spec import estimate_for_spec

    spec = _spec()
    quote = runner.charge_usd_for_spec(spec)
    assert quote > 0
    # pricing the spec's planned steps == exactly the submit quote
    assert quote == float(estimate_for_spec(spec).total_usd)


def test_charge_usd_for_spec_scales_with_actual_steps():
    spec = _spec()  # 20 planned steps
    full = runner.charge_usd_for_spec(spec)  # the quote
    half = runner.charge_usd_for_spec(spec, steps=10)  # cancelled at 10 steps
    assert 0 < half < full
    # cancelled before any step -> $0
    assert runner.charge_usd_for_spec(spec, steps=0) == 0.0


def test_charge_usd_for_spec_prorates_sft_cancel_by_tokens(monkeypatch):
    # SFT is priced from train_tokens, not steps, so a cancel must scale the token count to the
    # fraction of steps that ran -- lowering steps alone would leave the full-run token estimate.
    from dataclasses import replace

    from flash.cost import spec as cost_spec
    from flash.cost.analytical import estimate_cost
    from flash.cost.types import RunConfig

    cfg = RunConfig(model_id="Qwen/Qwen3.5-4B", method="sft", steps=20, train_tokens=4_000_000)
    monkeypatch.setattr(cost_spec, "runconfig_from_spec", lambda spec: cfg)

    full = float(estimate_cost(cfg).total_usd)  # the 20-step / full-token quote
    naive = float(estimate_cost(replace(cfg, steps=10)).total_usd)  # steps lowered, tokens NOT scaled
    half = runner.charge_usd_for_spec(object(), steps=10)  # cancelled at 10 of 20 steps
    assert 0 < half < full
    # the token scaling is what prorates SFT: a steps-only replace (naive) barely moves the price.
    assert half < naive


def test_charge_usd_for_spec_falls_back_when_unpriceable():
    # a spec that can't be priced returns the fallback rather than raising (a charge is never blocked)
    assert runner.charge_usd_for_spec(object(), fallback=1.5) == 1.5
    assert runner.charge_usd_for_spec(object(), steps=5, fallback=2.0) == 2.0


# --------------------------------------------------------------------------- actual steps


def test_actual_steps_run_reads_last_heartbeat_step():
    def st(hb):
        return runner.RunStatus(run_id="r", state="cancelled", spec={}, last_heartbeat=hb)

    assert runner.actual_steps_run(st({"stage": "rl_step", "step": 7})) == 7
    # no heartbeat / setup stage -> 0 (cancelled during cold-start, no GPU training yet)
    assert runner.actual_steps_run(st(None)) == 0
    assert runner.actual_steps_run(st({"stage": "setup"})) == 0
    # training started but no step completed yet (the ~17-min first GRPO rollout emits no `step`) ->
    # floor to 1 so real GPU time isn't billed as $0.
    assert runner.actual_steps_run(st({"stage": "rl_step", "step": 0})) == 1
    assert runner.actual_steps_run(st({"stage": "rl_step"})) == 1
    assert runner.actual_steps_run(st({"stage": "sft_step"})) == 1
    # A completed OPD run's final pre-DONE heartbeats (opd_trained / opd_train_done) are NOT training
    # stages, so a STEPLESS one floors a cancel-between-publish-and-DONE to 0 -- re-pricing a fully
    # trained run as $0. opd.py/finalize.py attach step=opt_steps so the true count bills (codex[bot]).
    assert runner.actual_steps_run(st({"stage": "opd_trained"})) == 0  # the bug the step guards against
    assert runner.actual_steps_run(st({"stage": "opd_train_done"})) == 0
    assert runner.actual_steps_run(st({"stage": "opd_trained", "step": 12})) == 12
    assert runner.actual_steps_run(st({"stage": "opd_train_done", "step": 12})) == 12
    # Terminal `done` heartbeat: a cancel racing the DONE upload (done recorded, run not yet
    # transitioned) reads a STEPLESS done and floors a fully-trained run to 0. _finalize carries
    # opt_steps onto `done` so the true count bills (codex[bot]).
    assert runner.actual_steps_run(st({"stage": "done"})) == 0  # the bug the step guards against
    assert runner.actual_steps_run(st({"stage": "done", "step": 12})) == 12


# --------------------------------------------------------------------------- cancel re-pricing


def test_cancel_run_prices_mid_training_cancel_at_actual_steps(monkeypatch, tmp_path):
    from flash.runner import deploy

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda spec: None)
    spec = _spec()
    runner._save_status(
        runner.RunStatus(
            run_id="run-1",
            state="running",
            spec=spec.to_dict(),
            billing_context={"org_id": "o"},
            billing_state="pending",
            last_heartbeat={"stage": "rl_step", "step": 10},
        )
    )

    deploy.cancel_run("run-1")

    st = runner.get_status("run-1")
    assert st.state == "cancelled"
    # re-priced to the estimate at the 10 steps it ran: > 0 and less than the full 20-step quote.
    assert st.cost_usd == runner.charge_usd_for_spec(spec, steps=10)
    assert 0 < st.cost_usd < runner.charge_usd_for_spec(spec)


def test_cancel_run_before_any_step_is_free(monkeypatch, tmp_path):
    from flash.runner import deploy

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda spec: None)
    spec = _spec()
    runner._save_status(
        runner.RunStatus(
            run_id="run-1",
            state="running",
            spec=spec.to_dict(),
            billing_context={"org_id": "o"},
            billing_state="pending",
            last_heartbeat=None,  # cancelled during cold-start/setup, no training step reported
        )
    )

    deploy.cancel_run("run-1")

    st = runner.get_status("run-1")
    assert st.state == "cancelled"
    assert st.cost_usd == 0.0  # 0 steps -> $0
