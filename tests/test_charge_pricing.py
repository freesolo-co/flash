"""Customer-charge pricing: a completed run is charged its QUOTE (the flash.cost estimate at planned
steps); a run cancelled mid-training bills the persisted quote prorated by the steps it actually
ran (never above the quote), falling back to a spec reprice only when no quote was persisted."""

from __future__ import annotations

import flash.runner as runner

SPEC = {
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {"epochs": 20, "max_examples": 20, "batch_size": 20},
    "gpu": {},
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
    naive = float(
        estimate_cost(replace(cfg, steps=10)).total_usd
    )  # steps lowered, tokens NOT scaled
    half = runner.charge_usd_for_spec(object(), steps=10)  # cancelled at 10 of 20 steps
    assert 0 < half < full
    # the token scaling is what prorates SFT: a steps-only replace (naive) barely moves the price.
    assert half < naive


def test_cancelled_profile_is_billed_all_or_nothing():
    """A profile that never started is free; one that started owes its whole bounded wall.

    A profile has no optimizer steps, so its quote is a wall cap rather than a per-step price and
    cannot be prorated. Both directions are defects: charging the cap for a profile cancelled while
    still queued bills a gpu that never ran, and since the id is derived from the workload rather
    than the account that lands on whichever submitter won the claim; charging $0 for one that ran
    gives the rented wall away.
    """
    from dataclasses import replace

    from flash.engine.profiling.workload_profile import SFT_PROFILE_KIND

    profile = replace(
        _spec(), run_id="profile-sft-abc", workload_profile_kind=SFT_PROFILE_KIND, algorithm="sft"
    )
    quote = runner.charge_usd_for_spec(profile)
    assert quote > 0
    # never started -> nothing rented -> nothing owed.
    assert runner.charge_usd_for_spec(profile, steps=0) == 0.0
    # started at all -> the whole bounded wall, not a fraction of it.
    assert runner.charge_usd_for_spec(profile, steps=1) == quote


def test_profile_steps_run_reads_started_not_optimizer_steps():
    """The profile cancel signal is "did the worker ever speak", not a step count.

    ``actual_steps_run`` looks for rl_step/sft_step/opd_step heartbeats, which a profile never
    emits -- so reusing it here would read 0 for a profile that ran to completion and hand back
    its rented wall for free. The last assertion is the one that makes the split load-bearing.
    """

    def st(hb):
        return runner.RunStatus(run_id="r", state="cancelled", spec={}, last_heartbeat=hb)

    assert runner.profile_steps_run(st(None)) == 0
    assert runner.profile_steps_run(st({})) == 0
    assert runner.profile_steps_run(st({"stage": "profile_start"})) == 1
    assert runner.profile_steps_run(st({"stage": "setup"})) == 1
    assert runner.profile_steps_run(st({"stage": "done"})) == 1
    # the reason profile_steps_run exists rather than reusing actual_steps_run.
    assert runner.actual_steps_run(st({"stage": "profile_start"})) == 0


def test_charge_usd_for_spec_falls_back_when_unpriceable():
    # a spec that can't be priced returns the fallback rather than raising (a charge is never blocked)
    assert runner.charge_usd_for_spec(object(), fallback=1.5) == 1.5
    assert runner.charge_usd_for_spec(object(), steps=5, fallback=2.0) == 2.0


def test_cancelled_charge_usd_prorates_and_clamps_to_the_quote():
    # the persisted quote carries the accepted live rate, so a cancel bills a linear share of it
    # by completed steps and can never exceed it.
    spec = _spec()  # 20 planned steps
    st = runner.RunStatus(run_id="r", state="cancelled", spec={}, estimated_cost_usd=8.0)
    assert runner.cancelled_charge_usd(st, spec, steps=0) == 0.0
    assert runner.cancelled_charge_usd(st, spec, steps=10) == 4.0
    assert runner.cancelled_charge_usd(st, spec, steps=20) == 8.0
    # a step count beyond the plan still caps at the full quote.
    assert runner.cancelled_charge_usd(st, spec, steps=25) == 8.0


def test_cancelled_charge_usd_falls_back_to_reprice_without_a_quote():
    # a run persisted before quotes existed has nothing to prorate; the spec reprice still bills it.
    spec = _spec()
    st = runner.RunStatus(run_id="r", state="cancelled", spec={})
    assert runner.cancelled_charge_usd(st, spec, steps=10) == runner.charge_usd_for_spec(
        spec, steps=10
    )
    assert runner.cancelled_charge_usd(st, spec, steps=0) == 0.0


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
    # A completed OPD run's final pre-DONE heartbeats (opd_trained / opd_train_done) are NOT
    # training stages, so a STEPLESS one floors a cancel-between-publish-and-DONE to 0 -- re-pricing
    # a fully trained run as $0. opd.py/finalize.py attach step=opt_steps so the true count bills.
    assert (
        runner.actual_steps_run(st({"stage": "opd_trained"})) == 0
    )  # the bug the step guards against
    assert runner.actual_steps_run(st({"stage": "opd_train_done"})) == 0
    assert runner.actual_steps_run(st({"stage": "opd_trained", "step": 12})) == 12
    assert runner.actual_steps_run(st({"stage": "opd_train_done", "step": 12})) == 12
    # Terminal `done` heartbeat: a cancel racing the DONE upload (done recorded, run not yet
    # transitioned) reads a STEPLESS done and floors a fully-trained run to 0. _finalize carries
    # opt_steps onto `done` so the true count bills.
    assert runner.actual_steps_run(st({"stage": "done"})) == 0  # the bug the step guards against
    assert runner.actual_steps_run(st({"stage": "done", "step": 12})) == 12


# --------------------------------------------------------------------------- cancel re-pricing


def test_cancel_run_prices_mid_training_cancel_at_actual_steps(monkeypatch, tmp_path):
    from flash.runner.supervise import deploy

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
    # no persisted quote on this status, so the cancel falls back to re-pricing the spec at the
    # 10 steps it ran: > 0 and less than the full 20-step quote.
    assert st.cost_usd == runner.charge_usd_for_spec(spec, steps=10)
    assert 0 < st.cost_usd < runner.charge_usd_for_spec(spec)


def test_cancel_near_completion_never_bills_above_the_accepted_quote(monkeypatch, tmp_path):
    """when the accepted live rate is below today's static rate, a near-complete cancel must bill
    a prorated share of the persisted quote, not the (higher) offline static-rate reprice."""
    from flash.runner.supervise import deploy

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda spec: None)
    spec = _spec()  # 20 planned steps
    static_reprice = runner.charge_usd_for_spec(spec, steps=19)
    # the user accepted a live-market quote at half the static rate for the full run.
    accepted_quote = runner.charge_usd_for_spec(spec) / 2
    assert static_reprice > accepted_quote  # the overbilling hazard the proration guards against
    runner._save_status(
        runner.RunStatus(
            run_id="run-1",
            state="running",
            spec=spec.to_dict(),
            billing_context={"org_id": "o"},
            billing_state="pending",
            estimated_cost_usd=accepted_quote,
            last_heartbeat={"stage": "rl_step", "step": 19},
        )
    )

    deploy.cancel_run("run-1")

    st = runner.get_status("run-1")
    assert st.state == "cancelled"
    # prorated share of the accepted quote, strictly under both the quote and the static reprice.
    assert st.cost_usd == accepted_quote * 19 / 20
    assert st.cost_usd < accepted_quote
    assert st.cost_usd < static_reprice


def test_cancel_run_prorates_the_persisted_quote(monkeypatch, tmp_path):
    from flash.runner.supervise import deploy

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda spec: None)
    spec = _spec()  # 20 planned steps
    runner._save_status(
        runner.RunStatus(
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

    st = runner.get_status("run-1")
    assert st.state == "cancelled"
    # half the steps -> half the accepted quote.
    assert st.cost_usd == 4.0


def test_cancel_run_before_any_step_is_free(monkeypatch, tmp_path):
    from flash.runner.supervise import deploy

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
