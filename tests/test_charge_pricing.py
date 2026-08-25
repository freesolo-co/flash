"""Customer-charge pricing from the accepted quote and completed estimated work.

A full planned run pays the quote exactly. A cancelled or successfully shortened run pays the
completed estimated-work fraction, never measured elapsed wall and never more than the quote.
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


# --------------------------------------------------------------------------- pricing helper


def test_charge_usd_for_spec_is_the_submit_quote():
    from flash.cost.spec import estimate_for_spec

    spec = _spec()
    quote = runner_costs.charge_usd_for_spec(spec)
    assert quote > 0
    # pricing the spec's planned steps == exactly the submit quote
    assert quote == float(estimate_for_spec(spec).total_usd)


def test_charge_usd_for_spec_scales_with_actual_steps():
    spec = _spec()  # 20 planned steps
    full = runner_costs.charge_usd_for_spec(spec)  # the quote
    half = runner_costs.charge_usd_for_spec(spec, steps=10)  # cancelled at 10 steps
    assert 0 < half < full
    # cancelled before any step -> $0
    assert runner_costs.charge_usd_for_spec(spec, steps=0) == 0.0


def test_charge_usd_for_spec_prorates_sft_cancel_by_tokens(monkeypatch):
    # SFT is priced from train_tokens, not steps, so a cancel must scale the token count to the
    # fraction of steps that ran -- lowering steps alone would leave the full-run token estimate.
    from dataclasses import replace

    from flash.cost import spec as cost_spec
    from flash.cost.analytical import estimate_cost
    from flash.cost.types import RunConfig

    cfg = RunConfig(model_id="Qwen/Qwen3.5-9B", method="sft", steps=20, train_tokens=4_000_000)
    monkeypatch.setattr(cost_spec, "runconfig_from_spec", lambda spec: cfg)

    full = float(estimate_cost(cfg).total_usd)  # the 20-step / full-token quote
    naive = float(
        estimate_cost(replace(cfg, steps=10)).total_usd
    )  # steps lowered, tokens NOT scaled
    half = runner_costs.charge_usd_for_spec(object(), steps=10)  # cancelled at 10 of 20 steps
    assert 0 < half < full
    # the token scaling is what prorates SFT: a steps-only replace (naive) barely moves the price.
    assert half < naive


def test_charge_usd_for_spec_falls_back_when_unpriceable():
    # a spec that can't be priced returns the fallback rather than raising (a charge is never blocked)
    assert runner_costs.charge_usd_for_spec(object(), fallback=1.5) == 1.5
    assert runner_costs.charge_usd_for_spec(object(), steps=5, fallback=2.0) == 2.0


def _unbounded_on_policy_spec(algorithm: str):
    """A grpo/opd spec that states no prompt-pool size and no horizon -- the shape a quote refuses.

    Neither ``max_examples`` nor ``max_steps`` is set, so ``spec_steps`` cannot derive an update
    horizon and raises rather than inventing one. Submitting this is blocked; a run that predates
    the check, or one whose environment supplies the pool at load time, still reaches cancellation.
    """
    from flash.schema import spec_from_dict

    return spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": algorithm,
            "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
            "train": {"epochs": 1, "prompts_per_step": 8, "group_size": 4},
            "gpu": {},
        },
        run_id="run-unbounded",
    )


def test_cancel_prices_an_unbounded_on_policy_run_from_its_completed_steps():
    # refusing to GUESS a horizon must not spread to a run that no longer needs one guessed: a
    # cancel states the steps it ran, which is a horizon, so the missing pool size is irrelevant.
    # regression -- the raise propagated through the blanket except here and returned the fallback,
    # so a cancelled unbounded run billed $0 and settled as a pricing failure while having really
    # rented the GPU. both algorithms derive the same way, so both regressed.
    for algorithm in ("grpo", "opd"):
        spec = _unbounded_on_policy_spec(algorithm)
        charge = runner_costs.charge_usd_for_spec(spec, steps=7, fallback=float("nan"))
        assert math.isfinite(charge), f"{algorithm} cancel was unpriceable"
        assert charge > 0
        # priced from the completed count, not a guess: more steps run == more owed, and a cancel
        # before the first step is still free.
        assert runner_costs.charge_usd_for_spec(spec, steps=14, fallback=float("nan")) > charge
        assert runner_costs.charge_usd_for_spec(spec, steps=0, fallback=float("nan")) == 0.0
        # and the quote-anchored path reaches the same estimate instead of discarding the quote.
        st = runner_state.RunStatus(run_id="r", state="cancelled", spec={}, estimated_cost_usd=12.0)
        assert runner_costs.cancelled_charge_usd(st, spec, steps=7, fallback=float("nan")) == charge


def test_cancelled_unbounded_on_policy_run_never_exceeds_its_quote():
    # when the full-work horizon is unknowable, cancellation falls back to the completed-step
    # reprice. that fallback still cannot exceed the price the customer accepted at submission.
    for algorithm in ("grpo", "opd"):
        spec = _unbounded_on_policy_spec(algorithm)
        repriced = runner_costs.charge_usd_for_spec(spec, steps=7, fallback=float("nan"))
        accepted_quote = repriced / 2
        st = runner_state.RunStatus(
            run_id="r",
            state="cancelled",
            spec={},
            estimated_cost_usd=accepted_quote,
        )
        assert (
            runner_costs.cancelled_charge_usd(st, spec, steps=7, fallback=float("nan"))
            == accepted_quote
        )


def test_quoting_an_unbounded_on_policy_run_still_refuses_to_guess():
    # the paired control: with no steps to state there is still no horizon, so a PRE-run quote must
    # keep refusing. without this, a fix for the cancel path could silently restore the guess the
    # raise exists to prevent, and the test above would pass just as well.
    from flash.cost.spec import UnknownPromptPoolSize, spec_steps

    for algorithm in ("grpo", "opd"):
        spec = _unbounded_on_policy_spec(algorithm)
        with pytest.raises(UnknownPromptPoolSize):
            spec_steps(spec)
        assert math.isnan(runner_costs.charge_usd_for_spec(spec, fallback=float("nan")))


def test_cancelled_charge_usd_prorates_against_the_whole_cent_quote():
    spec = _spec()
    st = runner_state.RunStatus(run_id="r", state="cancelled", spec={}, estimated_cost_usd=1.01)

    assert runner_costs.cancelled_charge_usd(st, spec, steps=10) == 0.505
    assert runner_costs.cancelled_charge_usd(st, spec, steps=20) == 1.01


def test_cancelled_charge_usd_prorates_and_clamps_to_the_quote():
    # the persisted quote carries the accepted live rate, so a cancel bills the completed share of
    # it and can never exceed it. this plan is uniform (no saves, dense model, no wall cap), so the
    # work share reduces to the bare step fraction.
    spec = _spec()  # 20 planned steps
    st = runner_state.RunStatus(run_id="r", state="cancelled", spec={}, estimated_cost_usd=8.0)
    assert runner_costs.cancelled_charge_usd(st, spec, steps=0) == 0.0
    assert runner_costs.cancelled_charge_usd(st, spec, steps=10) == 4.0
    assert runner_costs.cancelled_charge_usd(st, spec, steps=20) == 8.0
    # a step count beyond the plan still caps at the full quote.
    assert runner_costs.cancelled_charge_usd(st, spec, steps=25) == 8.0


@pytest.mark.parametrize(
    "bad_quote",
    ["not-a-number", True, float("nan"), float("inf"), float("-inf")],
)
def test_cancelled_charge_usd_treats_a_malformed_quote_as_a_pricing_failure(bad_quote):
    # malformed persisted quotes must not raise out of the cancel path, and repricing the spec
    # instead could bill above the unknowable accepted rate, so the measured fallback propagates.
    spec = _spec()
    st = runner_state.RunStatus(
        run_id="r", state="cancelled", spec={}, estimated_cost_usd=bad_quote
    )
    assert runner_costs.cancelled_charge_usd(st, spec, steps=10, fallback=3.25) == 3.25


def test_cancelled_charge_usd_treats_a_negative_quote_as_a_pricing_failure():
    # a negative persisted quote cannot represent the accepted whole-cent amount, so the measured
    # fallback propagates rather than persisting a negative customer charge.
    spec = _spec()
    st = runner_state.RunStatus(run_id="r", state="cancelled", spec={}, estimated_cost_usd=-8.0)
    assert runner_costs.cancelled_charge_usd(st, spec, steps=10, fallback=3.25) == 3.25


def test_cancelled_charge_usd_preserves_a_zero_quote_for_incomplete_work(monkeypatch):
    spec = _spec()
    st = runner_state.RunStatus(run_id="r", state="cancelled", spec={}, estimated_cost_usd=0.0)

    def unexpected_reprice(*_args, **_kwargs):
        raise AssertionError("a zero accepted quote must not be repriced")

    monkeypatch.setattr(runner_costs, "charge_usd_for_spec", unexpected_reprice)
    assert runner_costs.cancelled_charge_usd(st, spec, steps=10, fallback=4.5) == 0.0


def _patched_cfg_spec(monkeypatch, cfg):
    """Route both the partial and full spec estimates through a fixed RunConfig."""
    from types import SimpleNamespace

    from flash.cost import spec as cost_spec

    monkeypatch.setattr(cost_spec, "runconfig_from_spec", lambda spec: cfg)
    return SimpleNamespace()


def test_cancelled_charge_usd_prices_against_the_capped_horizon(monkeypatch):
    # a wall-capped quote pays for cap_s of training, not the uncapped step count, so the first
    # step of a capped 100k-step plan owes its share of the capped horizon, not 1/100000 of the
    # quote.
    from flash.cost.analytical import estimate_cost
    from flash.cost.types import RunConfig

    cfg = RunConfig("Qwen/Qwen3.5-9B", "grpo", 100_000, max_wall_seconds=3600)
    assert estimate_cost(cfg).wall_capped
    spec = _patched_cfg_spec(monkeypatch, cfg)
    st = runner_state.RunStatus(run_id="r", state="cancelled", spec={}, estimated_cost_usd=8.0)
    charge = runner_costs.cancelled_charge_usd(st, spec, steps=1)
    expected = (
        8.0
        * runner_costs.charge_usd_for_spec(spec, steps=1)
        / runner_costs.charge_usd_for_spec(spec)
    )
    assert charge == expected
    # far above the bare-step fraction, at or under the quote.
    assert 8.0 / 100_000 < charge <= 8.0


def test_cancelled_charge_usd_excludes_unreached_required_saves(monkeypatch):
    # the quote includes the synchronous save at step 20, but a cancel at step 19 never ran it, so
    # the charge stays strictly under the bare 19/20 share of the quote.
    from flash.cost.analytical import required_save_overhead_seconds
    from flash.cost.types import RunConfig

    cfg = RunConfig("Qwen/Qwen3.5-9B", "grpo", 20, save_at_steps=(20,))
    assert required_save_overhead_seconds(cfg) > 0
    spec = _patched_cfg_spec(monkeypatch, cfg)
    st = runner_state.RunStatus(run_id="r", state="cancelled", spec={}, estimated_cost_usd=8.0)
    charge = runner_costs.cancelled_charge_usd(st, spec, steps=19)
    assert 0 < charge < 8.0 * 19 / 20


def test_cancelled_charge_usd_bills_the_one_time_compile_with_the_first_step(monkeypatch):
    # the moe compile is paid whole during the first step, so cancelling right after it owes more
    # than the bare 1/1000 step share of the quote.
    from flash.cost.analytical import compile_seconds
    from flash.cost.types import RunConfig

    cfg = RunConfig("Qwen/Qwen3.6-35B-A3B", "sft", 1000)
    assert compile_seconds(cfg, "H100") > 0
    spec = _patched_cfg_spec(monkeypatch, cfg)
    st = runner_state.RunStatus(run_id="r", state="cancelled", spec={}, estimated_cost_usd=8.0)
    charge = runner_costs.cancelled_charge_usd(st, spec, steps=1)
    assert 8.0 / 1000 < charge <= 8.0


def test_cancelled_charge_usd_prices_the_rented_topology(monkeypatch):
    # an auto-allocated worker spec persists no provider, so the reprice would credit nvlink
    # scaling on a vast rental that cannot deliver it. the moe compile is fixed while step time
    # scales with topology, so the wrong speedup shifts the work fraction instead of cancelling
    # out of it; the durable handle names the rented substrate and both estimates must use it.
    from flash.cost.types import RunConfig

    cfg = RunConfig("Qwen/Qwen3.6-35B-A3B", "sft", 1000, gpu_type="H100", gpu_count=2)
    spec = _patched_cfg_spec(monkeypatch, cfg)
    st = runner_state.RunStatus(
        run_id="r",
        state="cancelled",
        spec={},
        estimated_cost_usd=8.0,
        remote={"provider": "vast"},
    )
    charge = runner_costs.cancelled_charge_usd(st, spec, steps=1)
    partial = runner_costs.charge_usd_for_spec(spec, steps=1, provider="vast")
    full = runner_costs.charge_usd_for_spec(spec, provider="vast")
    assert charge == 8.0 * partial / full
    # the auto reprice credits nvlink scaling, so its step time, fixed-cost weighting, and wall-cap
    # decision all differ from the pcie substrate the run rented: on this plan the slower pcie
    # steps trip the wall cap, shrinking the paid horizon and raising the first-step share.
    auto = (
        8.0
        * runner_costs.charge_usd_for_spec(spec, steps=1)
        / runner_costs.charge_usd_for_spec(spec)
    )
    assert charge != auto
    assert 0 < charge <= 8.0
    # an unknown handle provider must degrade to the spec's own pricing, never fail the charge.
    stale = runner_state.RunStatus(
        run_id="r",
        state="cancelled",
        spec={},
        estimated_cost_usd=8.0,
        remote={"provider": "gone-provider"},
    )
    assert runner_costs.cancelled_charge_usd(stale, spec, steps=1) == auto


def test_cancelled_charge_usd_pins_the_rented_card_count(monkeypatch):
    # the live allocator rented 4 cards, but the offline shape search treats the spec's count as a
    # ceiling and re-optimizes under it, repricing the run on the cheaper 1-card shape. the moe
    # compile is fixed while step time scales with the card shape, so the wrong geometry shifts
    # the work fraction instead of cancelling out of it; the durable handle stamps the rented
    # count and both estimates must pin it.
    from flash.cost.types import RunConfig

    cfg = RunConfig("Qwen/Qwen3.6-35B-A3B", "sft", 1000, gpu_type="H100", gpu_count=4)
    spec = _patched_cfg_spec(monkeypatch, cfg)
    st = runner_state.RunStatus(
        run_id="r",
        state="cancelled",
        spec={},
        estimated_cost_usd=8.0,
        remote={"provider": "runpod", "allocated_gpu": "H100", "allocated_gpu_count": 4},
    )
    charge = runner_costs.cancelled_charge_usd(st, spec, steps=1)
    # the ceiling search picks a different (cheaper) shape, so its fraction is not the rented one.
    ceiling = (
        8.0
        * runner_costs.charge_usd_for_spec(spec, steps=1, provider="runpod")
        / runner_costs.charge_usd_for_spec(spec, provider="runpod")
    )
    assert charge != ceiling
    partial = runner_costs.charge_usd_for_spec(
        spec, steps=1, provider="runpod", gpu_type="H100", gpu_count=4
    )
    full = runner_costs.charge_usd_for_spec(spec, provider="runpod", gpu_type="H100", gpu_count=4)
    assert charge == 8.0 * partial / full
    assert 0 < charge <= 8.0


def test_cancelled_charge_usd_degrades_without_an_allocation_stamp(monkeypatch):
    # a legacy handle predating the allocation stamp names only the provider: the shape falls back
    # to the spec-derived search (today's behavior) and the charge stays clamped to the quote. a
    # half stamp (count without card) cannot name a geometry and must degrade the same way.
    from flash.cost.types import RunConfig

    cfg = RunConfig("Qwen/Qwen3.6-35B-A3B", "sft", 1000, gpu_type="H100", gpu_count=4)
    spec = _patched_cfg_spec(monkeypatch, cfg)
    expected = (
        8.0
        * runner_costs.charge_usd_for_spec(spec, steps=1, provider="runpod")
        / runner_costs.charge_usd_for_spec(spec, provider="runpod")
    )
    for legacy_remote in (
        {"provider": "runpod"},
        {"provider": "runpod", "allocated_gpu_count": 4},
    ):
        st = runner_state.RunStatus(
            run_id="r",
            state="cancelled",
            spec={},
            estimated_cost_usd=8.0,
            remote=legacy_remote,
        )
        charge = runner_costs.cancelled_charge_usd(st, spec, steps=1)
        assert charge == expected
        assert 0 < charge <= 8.0
    # no handle at all (never-allocated run): the spec's own pricing, still clamped.
    bare = runner_state.RunStatus(run_id="r", state="cancelled", spec={}, estimated_cost_usd=8.0)
    charge = runner_costs.cancelled_charge_usd(bare, spec, steps=1)
    assert charge == 8.0 * runner_costs.charge_usd_for_spec(
        spec, steps=1
    ) / runner_costs.charge_usd_for_spec(spec)
    assert 0 < charge <= 8.0


def test_cancelled_charge_usd_falls_back_to_reprice_without_a_quote():
    # a run persisted before quotes existed has nothing to prorate; the spec reprice still bills it.
    spec = _spec()
    st = runner_state.RunStatus(run_id="r", state="cancelled", spec={})
    assert runner_costs.cancelled_charge_usd(
        st, spec, steps=10
    ) == runner_costs.charge_usd_for_spec(spec, steps=10)
    assert runner_costs.cancelled_charge_usd(st, spec, steps=0) == 0.0


# --------------------------------------------------------------------------- actual steps


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
    # A completed OPD run's final pre-DONE heartbeats (opd_trained / opd_train_done) are NOT
    # training stages, so a STEPLESS one floors a cancel-between-publish-and-DONE to 0 -- re-pricing
    # a fully trained run as $0. opd.py/finalize.py attach step=opt_steps so the true count bills.
    assert (
        runner_costs.actual_steps_run(st({"stage": "opd_trained"})) == 0
    )  # the bug the step guards against
    assert runner_costs.actual_steps_run(st({"stage": "opd_train_done"})) == 0
    assert runner_costs.actual_steps_run(st({"stage": "opd_trained", "step": 12})) == 12
    assert runner_costs.actual_steps_run(st({"stage": "opd_train_done", "step": 12})) == 12
    # Terminal `done` heartbeat: a cancel racing the DONE upload (done recorded, run not yet
    # transitioned) reads a STEPLESS done and floors a fully-trained run to 0. _finalize carries
    # opt_steps onto `done` so the true count bills.
    assert (
        runner_costs.actual_steps_run(st({"stage": "done"})) == 0
    )  # the bug the step guards against
    assert runner_costs.actual_steps_run(st({"stage": "done", "step": 12})) == 12


# --------------------------------------------------------------------------- cancel re-pricing


def test_cancel_run_prices_mid_training_cancel_at_actual_steps(monkeypatch, tmp_path):
    from flash.runner.supervise import deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda spec: None)
    spec = _spec()
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="run-1",
            state="running",
            spec=spec.to_dict(),
            billing_context={"org_id": "o"},
            billing_state="pending",
            last_heartbeat={"stage": "rl_step", "step": 10},
        )
    )

    deploy.cancel_run("run-1")

    st = runner_status.get_status("run-1")
    assert st.state == "cancelled"
    # no persisted quote on this status, so the cancel falls back to re-pricing the spec at the
    # 10 steps it ran: > 0 and less than the full 20-step quote.
    assert st.cost_usd == runner_costs.charge_usd_for_spec(spec, steps=10)
    assert 0 < st.cost_usd < runner_costs.charge_usd_for_spec(spec)


def test_cancel_near_completion_never_bills_above_the_accepted_quote(monkeypatch, tmp_path):
    """when the accepted live rate is below today's static rate, a near-complete cancel must bill
    a prorated share of the persisted quote, not the (higher) offline static-rate reprice."""
    from flash.runner.supervise import deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda spec: None)
    spec = _spec()  # 20 planned steps
    static_reprice = runner_costs.charge_usd_for_spec(spec, steps=19)
    # the user accepted a live-market quote at half the static rate for the full run.
    accepted_quote = runner_costs.charge_usd_for_spec(spec) / 2
    assert static_reprice > accepted_quote  # the overbilling hazard the proration guards against
    runner_state._save_status(
        runner_state.RunStatus(
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

    st = runner_status.get_status("run-1")
    assert st.state == "cancelled"
    # the completed share of the accepted quote, strictly under both the quote and the static
    # reprice.
    assert st.cost_usd == accepted_quote * static_reprice / runner_costs.charge_usd_for_spec(spec)
    assert st.cost_usd < accepted_quote
    assert st.cost_usd < static_reprice


def test_cancel_run_prorates_the_persisted_quote(monkeypatch, tmp_path):
    from flash.runner.supervise import deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda spec: None)
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
    # half the steps -> half the accepted quote.
    assert st.cost_usd == 4.0


def test_cancel_run_prices_the_rented_basis_after_teardown_clears_the_handle(monkeypatch, tmp_path):
    """a confirmed teardown clears status.remote before billing runs, and the handle is the only
    durable record of the rented substrate and card shape -- so cancel must capture it while it
    still holds the handle, or the charge silently reprices on the offline auto topology."""
    from flash.cost.types import RunConfig
    from flash.runner.supervise import deploy, lifecycle

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    cfg = RunConfig("Qwen/Qwen3.6-35B-A3B", "sft", 1000, gpu_type="H100", gpu_count=2)
    spec_stub = _patched_cfg_spec(monkeypatch, cfg)
    torn_down = []

    def teardown(handle, _run_id):
        torn_down.append(handle.provider)
        return True

    monkeypatch.setattr(lifecycle, "_strict_teardown_handle", teardown)
    spec = _spec()
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="run-1",
            state="running",
            spec=spec.to_dict(),
            billing_context={"org_id": "o"},
            billing_state="pending",
            estimated_cost_usd=8.0,
            remote={
                "provider": "vast",
                "instance_id": 101,
                "offer_id": 7,
                "machine_id": 3,
                "label": "flash-run-1",
                "gpu": "H100",
                "hourly_usd": 1.0,
                "attempt": 1,
                "started_ts": 1.0,
                "allocated_gpu": "H100",
                "allocated_gpu_count": 2,
            },
            last_heartbeat={"stage": "sft_step", "step": 1},
        )
    )

    deploy.cancel_run("run-1")

    st = runner_status.get_status("run-1")
    assert st.state == "cancelled"
    # the scenario under test: the teardown really cleared the handle before billing ran.
    assert torn_down == ["vast"]
    assert st.remote is None
    # not the auto/offline-shape fraction the cleared handle would degrade to.
    auto = (
        8.0
        * runner_costs.charge_usd_for_spec(spec_stub, steps=1)
        / runner_costs.charge_usd_for_spec(spec_stub)
    )
    assert st.cost_usd != auto
    partial = runner_costs.charge_usd_for_spec(
        spec_stub, steps=1, provider="vast", gpu_type="H100", gpu_count=2
    )
    full = runner_costs.charge_usd_for_spec(
        spec_stub, provider="vast", gpu_type="H100", gpu_count=2
    )
    assert st.cost_usd == 8.0 * partial / full
    assert 0 < st.cost_usd <= 8.0


def test_cancel_run_with_malformed_quote_still_settles_as_a_billing_failure(monkeypatch, tmp_path):
    # a nonnumeric persisted quote survives the tolerant status loader; the cancel must still reach
    # the cancelled transition and record the $0 pricing-failure diagnostic instead of raising.
    from flash.runner.supervise import deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda spec: None)
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
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda spec: None)
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


def test_completed_full_work_charges_exactly_the_quote_despite_faster_wall(tmp_path, monkeypatch):
    spec = _spec()
    accepted_quote = 8.0
    status = runner_state.RunStatus(
        run_id="r",
        state="done",
        spec={},
        estimated_cost_usd=accepted_quote,
    )
    _write_terminal_steps(tmp_path, monkeypatch, spec, 20, wall_seconds=0.001)

    assert runner_costs._status_estimated_charge(status, spec) == accepted_quote


@pytest.mark.parametrize("completed_steps", [20, 10], ids=["full-work", "incomplete-work"])
def test_completed_work_preserves_a_zero_quote_despite_positive_measured_fallback(
    tmp_path, monkeypatch, completed_steps
):
    spec = _spec()
    status = runner_state.RunStatus(
        run_id="r",
        state="done",
        spec={},
        estimated_cost_usd=0.0,
    )
    _write_terminal_steps(tmp_path, monkeypatch, spec, completed_steps, wall_seconds=99_999.0)

    def unexpected_reprice(*_args, **_kwargs):
        raise AssertionError("a zero accepted quote must not be repriced")

    monkeypatch.setattr(runner_costs, "charge_usd_for_spec", unexpected_reprice)
    assert runner_costs._status_estimated_charge(status, spec, fallback=6.75) == 0.0


@pytest.mark.parametrize("completed_steps", [21, 10**1000])
def test_completed_steps_beyond_the_horizon_still_charge_exactly_the_quote(
    tmp_path, monkeypatch, completed_steps
):
    spec = _spec()
    accepted_quote = 8.0
    status = runner_state.RunStatus(
        run_id="r",
        state="done",
        spec={},
        estimated_cost_usd=accepted_quote,
    )
    _write_terminal_steps(tmp_path, monkeypatch, spec, completed_steps, wall_seconds=0.001)

    assert runner_costs._status_estimated_charge(status, spec) == accepted_quote


def test_completed_early_work_uses_the_same_estimated_fraction_as_cancellation(
    tmp_path, monkeypatch
):
    spec = _spec()
    accepted_quote = 8.0
    status = runner_state.RunStatus(
        run_id="r",
        state="done",
        spec={},
        estimated_cost_usd=accepted_quote,
    )
    _write_terminal_steps(tmp_path, monkeypatch, spec, 10, wall_seconds=99_999.0)

    expected = runner_costs.cancelled_charge_usd(status, spec, steps=10)
    assert 0.0 < expected < accepted_quote
    assert runner_costs._status_estimated_charge(status, spec) == expected


def test_completed_early_work_uses_the_persisted_rented_topology(tmp_path, monkeypatch):
    from flash.cost.types import RunConfig

    cfg = RunConfig("Qwen/Qwen3.6-35B-A3B", "sft", 1000, gpu_type="H100", gpu_count=4)
    spec = _patched_cfg_spec(monkeypatch, cfg)
    accepted_quote = 8.0
    status = runner_state.RunStatus(
        run_id="r",
        state="done",
        spec={},
        estimated_cost_usd=accepted_quote,
        remote=None,
    )
    monkeypatch.setattr(runner_state, "artifacts_dir", lambda _spec: str(tmp_path / "metrics"))
    _write_terminal_steps(
        tmp_path,
        monkeypatch,
        spec,
        1,
        allocated_provider="runpod",
        allocated_gpu="H100",
        allocated_gpu_count=4,
    )

    charge = runner_costs._status_estimated_charge(status, spec)
    partial = runner_costs.charge_usd_for_spec(
        spec, steps=1, provider="runpod", gpu_type="H100", gpu_count=4
    )
    full = runner_costs.charge_usd_for_spec(spec, provider="runpod", gpu_type="H100", gpu_count=4)
    expected = accepted_quote * partial / full
    ceiling = (
        accepted_quote
        * runner_costs.charge_usd_for_spec(spec, steps=1)
        / runner_costs.charge_usd_for_spec(spec)
    )

    assert charge == expected
    assert charge != ceiling


def test_completed_zero_work_charges_zero(tmp_path, monkeypatch):
    spec = _spec()
    status = runner_state.RunStatus(run_id="r", state="done", spec={}, estimated_cost_usd=8.0)
    _write_terminal_steps(tmp_path, monkeypatch, spec, 0)

    assert runner_costs._status_estimated_charge(status, spec) == 0.0


@pytest.mark.parametrize("step", [None, True, -1, 1.5, float("nan"), float("inf"), "10"])
def test_completed_run_without_a_valid_step_preserves_the_quote(tmp_path, monkeypatch, step):
    spec = _spec()
    accepted_quote = 8.0
    status = runner_state.RunStatus(
        run_id="r",
        state="done",
        spec={},
        estimated_cost_usd=accepted_quote,
    )
    _write_terminal_steps(tmp_path, monkeypatch, spec, step)

    assert runner_costs._status_estimated_charge(status, spec) == accepted_quote
