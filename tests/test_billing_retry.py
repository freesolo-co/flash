"""Retry/recovery of completion-time CUSTOMER charges (flash/server/billing_retry.py).

Offline: the only network boundary (flash.server.billing.charge_completed_run) is stubbed, so a
transient blip and a successful retry can be simulated without touching the backend. The point of
the feature is that a finished-but-uncharged run is eventually charged exactly once -- the backend
route is idempotent by runId, so a retry can never double-charge.
"""

from __future__ import annotations

import importlib
import io

import pytest

from flash.server import billing_retry

SPEC = {
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {"steps": 1, "seeds": [0], "hf_repo": "org/test-runs"},
    "gpu": {"type": "RTX 5090"},
}

_USER_PREFIX = "fslo-user-"


def _spec():
    from flash.schema import spec_from_dict

    return spec_from_dict(SPEC, run_id="run-1")


def _save_run(runner, tmp_path, *, state="done", billing_state="pending", billing_context=None):
    spec = _spec()
    status = runner.RunStatus(
        run_id=spec.run_id,
        state=state,
        spec=spec.to_dict(),
        cost_usd=1.23,
        remote={"provider": "runpod", "allocated_gpu": "RTX 5090"},
        billing_context={"org_id": "org-A"} if billing_context is None else billing_context,
        billing_state=billing_state,
    )
    runner._save_status(status)
    return spec


# --------------------------------------------------------------------------- predicate


def test_needs_charge_predicate():
    from flash.runner import RunStatus

    def st(**kw):
        base = {"run_id": "r", "state": "done", "spec": {}}
        base.update(kw)
        return RunStatus(**base)

    ctx = {"org_id": "o"}
    # a completed external run not yet charged -> needs a charge
    assert billing_retry._needs_charge(st(state="done", billing_context=ctx, billing_state="pending"))
    assert billing_retry._needs_charge(st(state="done", billing_context=ctx, billing_state="failed"))
    assert billing_retry._needs_charge(
        st(state="done", billing_context=ctx, billing_state="charging")
    )
    # a deployed run still bills (done-then-deployed; its done-time charge may have failed)
    assert billing_retry._needs_charge(
        st(state="deployed", billing_context=ctx, billing_state="failed")
    )
    # already charged -> no
    assert not billing_retry._needs_charge(
        st(state="done", billing_context=ctx, billing_state="charged")
    )
    # never completed (failed/cancelled) -> never charged, even if billing_state is still pending
    # (the submit-time default: the charge machine never ran, so the run never trained)
    assert not billing_retry._needs_charge(
        st(state="failed", billing_context=ctx, billing_state="pending")
    )
    assert not billing_retry._needs_charge(
        st(state="cancelled", billing_context=ctx, billing_state="pending")
    )
    # completed-THEN-cancelled: a run that reached `done`/`deployed` (so the charge machine ran and
    # left `charging`/`failed`) and was later cancelled still owes its completion charge -> needs one,
    # even though `cancelled` is not a billable state. This is the deployed-then-cancelled leak.
    assert billing_retry._needs_charge(
        st(state="cancelled", billing_context=ctx, billing_state="failed")
    )
    assert billing_retry._needs_charge(
        st(state="cancelled", billing_context=ctx, billing_state="charging")
    )
    # PARTIAL MULTI-SEED ATTACH (must NOT charge): attach_run sets `artifacts_dir` + partial
    # `cost_usd` for an intermediate seed while the run is still `running` (deploy.py:232-238). Such a
    # run -- still running, or later cancelled -- has NOT finished all training, so the predicate must
    # leave it alone. `artifacts_dir`/`cost_usd` are deliberately NOT eligibility signals here: only
    # `state in {done,deployed}` or a started charge (`charging`/`failed`) qualifies, neither of which
    # this partial run has. Charging it would bill a partial amount and pre-empt the real final charge.
    assert not billing_retry._needs_charge(
        st(
            state="running",
            billing_context=ctx,
            billing_state="pending",
            artifacts_dir="/results/runpod/rl/run-1",
            cost_usd=0.5,
            resume_seed_index=1,
        )
    )
    assert not billing_retry._needs_charge(
        st(
            state="cancelled",
            billing_context=ctx,
            billing_state="pending",
            artifacts_dir="/results/runpod/rl/run-1",
            cost_usd=0.5,
        )
    )
    # internal/non-external run (no billing context) -> nothing to charge
    assert not billing_retry._needs_charge(
        st(state="done", billing_context=None, billing_state=None)
    )


# --------------------------------------------------------------------------- sweep


def test_sweep_disabled_without_internal_key(monkeypatch, tmp_path):
    import flash.runner as runner

    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    _save_run(runner, tmp_path)
    monkeypatch.setattr(
        "flash.server.billing.charge_completed_run",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not charge without internal key")),
    )
    assert billing_retry.retry_completion_charges_once() == 0


def test_sweep_charges_crashed_pending_run(monkeypatch, tmp_path):
    """A run that went `done` but whose charge never ran (crash between the done write and the
    charge -> billing_state stuck `pending`) is recovered by the sweep and charged exactly once."""
    import flash.runner as runner

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    _save_run(runner, tmp_path, billing_state="pending")

    calls = []
    monkeypatch.setattr(
        "flash.server.billing.charge_completed_run",
        lambda *, internal_key, status: calls.append(status.run_id)
        or {"amountCents": 123, "replay": False},
    )

    assert billing_retry.retry_completion_charges_once() == 1
    assert calls == ["run-1"]
    st = runner.get_status("run-1")
    assert st.billing_state == "charged"
    assert st.billing_charge == {"amountCents": 123, "replay": False}


def test_transient_failure_then_retry_charges_exactly_once(monkeypatch, tmp_path):
    """The core guarantee: a transient BillingError leaves the run `failed`, the next sweep charges
    it, and once `charged` no further sweep re-invokes the backend -> the successful charge happens
    exactly once. (The backend route is also idempotent by runId, so even a racing duplicate replays
    rather than double-charging.)"""
    import flash.runner as runner
    from flash.server.billing import BillingError

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    _save_run(runner, tmp_path, billing_state="pending")

    calls = []

    def flaky_charge(*, internal_key, status):
        calls.append(status.run_id)
        if len(calls) == 1:
            raise BillingError(503, "billing service unavailable: transient")
        return {"amountCents": 123, "replay": False}

    monkeypatch.setattr("flash.server.billing.charge_completed_run", flaky_charge)

    # sweep #1: the transient blip -> run marked failed, nothing charged
    assert billing_retry.retry_completion_charges_once() == 0
    assert runner.get_status("run-1").billing_state == "failed"
    assert len(calls) == 1

    # sweep #2: the blip cleared -> the run is charged
    assert billing_retry.retry_completion_charges_once() == 1
    assert runner.get_status("run-1").billing_state == "charged"
    assert len(calls) == 2

    # sweep #3: already charged -> the backend is NOT hit again (no second/double charge)
    assert billing_retry.retry_completion_charges_once() == 0
    assert len(calls) == 2


def test_sweep_skips_charged_failed_and_internal_runs(monkeypatch, tmp_path):
    """Only completed external runs needing a charge are swept: a charged run, a non-completed
    (failed) run, and an internal run with no billing context are all left untouched."""
    import flash.runner as runner
    from flash.schema import spec_from_dict

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))

    def save(run_id, *, state, billing_state, billing_context):
        spec = spec_from_dict(SPEC, run_id=run_id)
        runner._save_status(
            runner.RunStatus(
                run_id=run_id,
                state=state,
                spec=spec.to_dict(),
                cost_usd=1.0,
                billing_context=billing_context,
                billing_state=billing_state,
            )
        )

    save("already", state="done", billing_state="charged", billing_context={"org_id": "o"})
    save("failed-run", state="failed", billing_state="pending", billing_context={"org_id": "o"})
    save("internal", state="done", billing_state=None, billing_context=None)

    monkeypatch.setattr(
        "flash.server.billing.charge_completed_run",
        lambda **_: (_ for _ in ()).throw(AssertionError("no eligible run should be charged")),
    )
    assert billing_retry.retry_completion_charges_once() == 0


def test_sweep_never_charges_partial_multiseed_attach_run(monkeypatch, tmp_path):
    """Regression for the over-charge the artifacts_dir anchor caused: attach_run sets `artifacts_dir`
    and a partial per-seed `cost_usd` for a multi-seed run while it is still `running` after recovering
    ONE intermediate seed (deploy.py:232-238). That run has NOT finished all training, so the sweep
    must NEVER charge it -- charging would bill a partial amount and pre-empt the real final charge.
    The same holds if that partial run is then cancelled mid-recovery (still `pending`, no charge
    attempt). The conservative predicate (state in {done,deployed} OR billing_state in
    {charging,failed}) leaves both alone."""
    import flash.runner as runner

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))

    # mid-recovery: one seed done so artifacts_dir + partial cost are set, but still running
    runner._save_status(
        runner.RunStatus(
            run_id="partial-running",
            state="running",
            spec=_spec().to_dict(),
            cost_usd=0.5,
            artifacts_dir="/results/runpod/rl/partial-running",
            resume_seed_index=1,
            billing_context={"org_id": "org-A"},
            billing_state="pending",
        )
    )
    # the same partial run cancelled mid-recovery: artifacts_dir set, charge never attempted
    runner._save_status(
        runner.RunStatus(
            run_id="partial-cancelled",
            state="cancelled",
            spec=_spec().to_dict(),
            cost_usd=0.5,
            artifacts_dir="/results/runpod/rl/partial-cancelled",
            billing_context={"org_id": "org-A"},
            billing_state="pending",
        )
    )

    monkeypatch.setattr(
        "flash.server.billing.charge_completed_run",
        lambda **_: (_ for _ in ()).throw(AssertionError("a partial multi-seed run must not be charged")),
    )
    assert billing_retry.retry_completion_charges_once() == 0
    assert runner.get_status("partial-running").billing_state == "pending"
    assert runner.get_status("partial-cancelled").billing_state == "pending"


def test_sweep_charges_completed_then_cancelled_run(monkeypatch, tmp_path):
    """A run that completed training (its charge machine ran and left `failed`) and was LATER
    cancelled -- e.g. a deployed run gets cancelled (deploy.py flips it to `cancelled`) -- still owes
    its completion charge. The widened predicate keeps it eligible even though `cancelled` left
    `_BILLABLE_STATES`, so the sweep recovers the otherwise-leaked revenue and charges it once.

    A run cancelled BEFORE completing (submit-time `pending`, never charge-machine-touched) stays
    ineligible in the same sweep -- it never trained, so it must never be charged."""
    import flash.runner as runner

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    # completed-then-cancelled: charge machine already ran (`failed`) -> still owes a charge
    _save_run(runner, tmp_path, state="cancelled", billing_state="failed")

    calls = []
    monkeypatch.setattr(
        "flash.server.billing.charge_completed_run",
        lambda *, internal_key, status: calls.append(status.run_id)
        or {"amountCents": 123, "replay": False},
    )

    assert billing_retry.retry_completion_charges_once() == 1
    assert calls == ["run-1"]
    assert runner.get_status("run-1").billing_state == "charged"


def test_sweep_never_charges_cancelled_before_completion(monkeypatch, tmp_path):
    """Guard against over-correcting finding (1): a run cancelled BEFORE it completed carries the
    submit-time `pending` (the charge machine never ran) and a non-billable state, so the sweep must
    leave it alone. A run that never trained must never be charged."""
    import flash.runner as runner

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    _save_run(runner, tmp_path, state="cancelled", billing_state="pending")

    monkeypatch.setattr(
        "flash.server.billing.charge_completed_run",
        lambda **_: (_ for _ in ()).throw(AssertionError("a never-completed run must not be charged")),
    )
    assert billing_retry.retry_completion_charges_once() == 0
    assert runner.get_status("run-1").billing_state == "pending"


def test_sweep_charges_run_with_stale_unparseable_spec(monkeypatch, tmp_path):
    """A completed run whose persisted spec is legacy/stale and rejected by `JobSpec.from_dict` must
    still be charged: the charge reads org/cost from the persisted RunStatus, not a reparsed spec, so
    requiring a full reparse just to pass the run id would silently leak that run's revenue forever."""
    import flash.runner as runner

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))

    # a finished, uncharged run whose persisted spec no longer round-trips through JobSpec.from_dict.
    # A legacy payload carrying a local `environment.path` is rejected by from_dict (local env paths
    # are no longer supported) -- the exact "stale persisted spec" finding (3) is about.
    stale_spec = {**SPEC, "environment": {"path": "./local/environment.py"}}
    spec = _spec()
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="done",
            spec=stale_spec,
            cost_usd=1.23,
            remote={"provider": "runpod", "allocated_gpu": "RTX 5090"},
            billing_context={"org_id": "org-A"},
            billing_state="pending",
        )
    )

    # belt-and-suspenders: prove this spec really would have aborted the old reparse path
    from flash.spec import JobSpec

    with pytest.raises(ValueError, match="local environment paths"):
        JobSpec.from_dict(stale_spec)

    calls = []
    monkeypatch.setattr(
        "flash.server.billing.charge_completed_run",
        lambda *, internal_key, status: calls.append(status.run_id) or {"amountCents": 123},
    )

    assert billing_retry.retry_completion_charges_once() == 1
    assert calls == ["run-1"]
    assert runner.get_status("run-1").billing_state == "charged"


def test_record_billing_state_preserves_run_state(monkeypatch, tmp_path):
    """codex P2 race fix (unit): record_billing_state writes ONLY the billing fields, re-reading the
    run UNDER the lock, so it never re-asserts (and so never clobbers) the run's state. This is what
    lets the completion charge run concurrently with a /deploy without the stale `done` overwriting a
    just-written `deployed`. It also rejects any non-billing field so it can't be misused to move
    state."""
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    _save_run(runner, tmp_path, state="deployed", billing_state="charging")

    # a billing-fields-only write must leave `deployed` intact (the race the old _update lost)
    runner.record_billing_state("run-1", billing_state="charged", billing_charge={"amountCents": 9})
    st = runner.get_status("run-1")
    assert st.state == "deployed"
    assert st.billing_state == "charged"
    assert st.billing_charge == {"amountCents": 9}

    # vanished run -> no-op (no raise)
    runner.record_billing_state("nope", billing_state="charged")

    # guard: only billing fields are writable, so it can never be used to change `state`
    with pytest.raises(ValueError, match="only writes billing fields"):
        runner.record_billing_state("run-1", state="done")


def test_sweep_charge_does_not_downgrade_deployed_state(monkeypatch, tmp_path):
    """End to end: charging a run that is `deployed` (a done-then-deployed run whose inline charge
    failed) records the charge WITHOUT moving the run off `deployed`."""
    import flash.runner as runner

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    _save_run(runner, tmp_path, state="deployed", billing_state="failed")

    monkeypatch.setattr(
        "flash.server.billing.charge_completed_run",
        lambda *, internal_key, status: {"amountCents": 123},
    )

    assert billing_retry.retry_completion_charges_once() == 1
    st = runner.get_status("run-1")
    assert st.billing_state == "charged"
    assert st.state == "deployed"  # charge never downgraded the live deployment


def test_record_billing_state_preserves_legacy_terminal_timestamp(monkeypatch, tmp_path):
    """cursor Medium / codex P2: a legacy ALREADY-terminal run with finished_at=None must keep its
    prior terminal time when a billing retry writes its fields. record_billing_state backfills
    finished_at from the PRE-update updated_at (the persisted teardown time) before bumping
    updated_at, mirroring _update -- otherwise reconcile._terminal_ts (which falls back to updated_at)
    would treat the retry time as the run end and over-bill flat-rate remotes."""
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    teardown_ts = 1_000_000.0  # the real terminal/teardown time, frozen in updated_at
    runner._save_status(
        runner.RunStatus(
            run_id="legacy",
            state="done",
            spec=_spec().to_dict(),
            cost_usd=1.0,
            updated_at=teardown_ts,
            finished_at=None,  # legacy run: never stamped
            billing_context={"org_id": "org-A"},
            billing_state="pending",
        )
    )

    runner.record_billing_state("legacy", billing_state="failed", billing_error="blip")
    st = runner.get_status("legacy")
    # finished_at backfilled from the prior teardown time, NOT the (now-advanced) updated_at
    assert st.finished_at == teardown_ts
    assert st.updated_at > teardown_ts  # the write did advance updated_at
    assert st.billing_state == "failed"


def test_record_billing_state_preserves_legacy_deployed_timestamp(monkeypatch, tmp_path):
    """codex P2: `deployed` is non-terminal but reconcile treats it as reconcilable
    (reconcile._RECONCILABLE_STATES = TERMINAL_STATES | {deployed}), and _terminal_ts falls back to
    updated_at when finished_at is None. So a legacy `deployed` run with finished_at=None must ALSO
    have its teardown time preserved when a billing retry writes its fields -- otherwise the retry
    time becomes the effective run end, delaying reconciliation and inflating flat-rate cost."""
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    teardown_ts = 2_000_000.0  # training teardown, frozen in updated_at before the deploy/retry
    runner._save_status(
        runner.RunStatus(
            run_id="legacy-dep",
            state="deployed",
            spec=_spec().to_dict(),
            cost_usd=1.0,
            updated_at=teardown_ts,
            finished_at=None,  # legacy: never stamped, and `deployed` isn't in TERMINAL_STATES
            billing_context={"org_id": "org-A"},
            billing_state="failed",
        )
    )

    runner.record_billing_state("legacy-dep", billing_state="charged", billing_charge={"a": 1})
    st = runner.get_status("legacy-dep")
    assert st.state == "deployed"  # state untouched
    assert st.finished_at == teardown_ts  # teardown preserved, not advanced to the retry time
    assert st.updated_at > teardown_ts
    assert st.billing_state == "charged"


def test_record_billing_state_skips_backfill_once_reconciled(monkeypatch, tmp_path):
    """A run already reconciled has updated_at == reconcile time (not teardown), so backfilling
    finished_at from it would freeze a wrong, later run end. Match mark_deployed: skip the backfill
    once reconciled_at is set (a reconciled run is never re-billed, so leaving finished_at None is
    harmless)."""
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    reconcile_ts = 3_000_000.0  # updated_at was moved here by record_realized_cost, NOT teardown
    runner._save_status(
        runner.RunStatus(
            run_id="reconciled",
            state="done",
            spec=_spec().to_dict(),
            cost_usd=1.0,
            updated_at=reconcile_ts,
            finished_at=None,
            reconciled_at=reconcile_ts,
            billing_context={"org_id": "org-A"},
            billing_state="pending",
        )
    )

    runner.record_billing_state("reconciled", billing_state="failed", billing_error="blip")
    st = runner.get_status("reconciled")
    assert st.finished_at is None  # NOT backfilled from the reconcile-time updated_at
    assert st.billing_state == "failed"


def test_record_billing_state_never_downgrades_charged(monkeypatch, tmp_path):
    """codex P2: a racing duplicate attempt that times out / hits a transient BillingError AFTER
    another attempt already landed `charged` must NOT flip the local state back to `failed`/`charging`
    (which would re-retry an already-charged run). record_billing_state no-ops any non-`charged` write
    once the run is `charged`."""
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    _save_run(runner, tmp_path, state="done", billing_state="charged")
    runner.record_billing_state("run-1", billing_charge={"amountCents": 42})  # establish the charge

    # a late failure write from a racing attempt -> must be ignored, charge stands
    runner.record_billing_state("run-1", billing_state="failed", billing_error="late blip")
    assert runner.get_status("run-1").billing_state == "charged"
    # a late charging write (racing re-entry) is likewise ignored
    runner.record_billing_state("run-1", billing_state="charging")
    assert runner.get_status("run-1").billing_state == "charged"
    # re-writing `charged` itself is still allowed (idempotent)
    runner.record_billing_state("run-1", billing_state="charged", billing_charge={"amountCents": 42})
    assert runner.get_status("run-1").billing_state == "charged"


# --------------------------------------------------------------------------- background loop
# Same contract as the sibling reconcile loop: re-raise CancelledError (so shutdown's task.cancel()
# stops it), swallow a real Exception and continue to the next cycle.


class _StopLoop(Exception):
    pass


def _run_loop_once(monkeypatch, sweep_impl):
    import asyncio

    from flash.server import _runtime

    monkeypatch.setattr(billing_retry, "retry_completion_charges_once", sweep_impl)
    calls = {"sleep": 0}

    async def fake_sleep(_interval):
        calls["sleep"] += 1
        if calls["sleep"] >= 2:  # 1st sleep enters the body; 2nd breaks out post-sweep
            raise _StopLoop

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return asyncio.run(_runtime._charge_retry_loop())


def test_loop_swallows_sweep_error_and_continues(monkeypatch):
    swept = {"n": 0}

    def boom(*a, **k):
        swept["n"] += 1
        raise RuntimeError("backend blip")

    with pytest.raises(_StopLoop):
        _run_loop_once(monkeypatch, boom)
    assert swept["n"] == 1  # the sweep ran and its failure was swallowed (loop kept going)


def test_loop_reraises_cancelled(monkeypatch):
    import asyncio

    def cancel(*a, **k):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        _run_loop_once(monkeypatch, cancel)


# --------------------------------------------------------------------------- startup wiring


def _identity_for_token(token: str) -> dict[str, str]:
    if not token.startswith(_USER_PREFIX):
        return {}
    suffix = token.removeprefix(_USER_PREFIX)
    return {"email": f"u-{suffix}@x", "key_prefix": "fslo_test", "org_id": f"org-{suffix}"}


def test_startup_runs_completion_charge_sweep(monkeypatch, tmp_path):
    """The lifespan must run the completion-charge recovery sweep at startup: a crash between the
    `done` write and the charge leaves a terminal `done` run that recover_runs cannot reach, so the
    startup sweep is what charges it on the next boot. The sweep is scheduled as a BACKGROUND task
    (off the critical path), so wait for it to land before asserting / leaving the context."""
    pytest.importorskip("fastapi")
    import time

    from fastapi.testclient import TestClient

    # Full operator config so the app's startup preflight passes (>= 2 RunPod accounts + Lambda +
    # Hyperstack + the shared tokens + the internal key); mirrors tests/test_server_api.py.
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test,rp-test-2")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam-test")
    monkeypatch.setenv("HYPERSTACK_API_KEY", "hyp-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    # runpod.keys caches the parsed pool on first read; reset so the startup preflight reads THIS
    # RUNPOD_API_KEY (the autouse _offline fixture also resets, but make this self-contained).
    import flash.providers.runpod.keys as runpod_keys

    runpod_keys.reset()

    import flash.runner as runner
    import flash.server.auth as auth_mod
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setattr(runner, "_run_job", lambda *a, **k: None)

    # a finished run whose completion charge never landed (billing_state stuck pending)
    _save_run(runner, tmp_path, billing_state="pending")

    charged = []
    monkeypatch.setattr(
        "flash.server.billing.charge_completed_run",
        lambda *, internal_key, status: charged.append(status.run_id) or {"amountCents": 123},
    )

    import flash.server.app as app_mod

    importlib.reload(app_mod)
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(_USER_PREFIX))
    monkeypatch.setattr(auth_mod, "_cached_identity", _identity_for_token)

    with TestClient(app_mod.create_app()):
        # entering the context runs the lifespan startup, which SCHEDULES the recovery sweep as a
        # background task; poll until it lands before leaving (shutdown cancels the task).
        deadline = time.time() + 5.0
        while runner.get_status("run-1").billing_state != "charged" and time.time() < deadline:
            time.sleep(0.02)

    assert charged == ["run-1"]
    assert runner.get_status("run-1").billing_state == "charged"


def test_startup_does_not_block_on_slow_charge_backlog(monkeypatch, tmp_path):
    """Finding (2): a slow/down billing backend with a backlog of pending charges must NOT delay the
    server from accepting traffic. The startup recovery is scheduled off the critical path, so the
    lifespan must reach `yield` (TestClient.__enter__ returns) WITHOUT waiting for the sweep -- even
    when the sweep is blocked mid-charge."""
    pytest.importorskip("fastapi")
    import threading
    import time

    from fastapi.testclient import TestClient

    # Full operator config so the app's startup preflight passes (>= 2 RunPod accounts + Lambda +
    # Hyperstack + the shared tokens + the internal key); mirrors tests/test_server_api.py.
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test,rp-test-2")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam-test")
    monkeypatch.setenv("HYPERSTACK_API_KEY", "hyp-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    # runpod.keys caches the parsed pool on first read; reset so the startup preflight reads THIS
    # RUNPOD_API_KEY (the autouse _offline fixture also resets, but make this self-contained).
    import flash.providers.runpod.keys as runpod_keys

    runpod_keys.reset()

    import flash.runner as runner
    import flash.server.auth as auth_mod
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setattr(runner, "_run_job", lambda *a, **k: None)

    _save_run(runner, tmp_path, billing_state="pending")

    in_charge = threading.Event()  # set once the sweep is actually inside the (blocked) charge
    release = threading.Event()  # held until the test lets the slow backend respond

    def slow_charge(*, internal_key, status):
        in_charge.set()
        # simulate a slow/down billing backend: block until the test releases us
        if not release.wait(timeout=10.0):
            raise AssertionError("slow charge was never released")
        return {"amountCents": 123}

    monkeypatch.setattr("flash.server.billing.charge_completed_run", slow_charge)

    import flash.server.app as app_mod

    importlib.reload(app_mod)
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(_USER_PREFIX))
    monkeypatch.setattr(auth_mod, "_cached_identity", _identity_for_token)

    started = time.time()
    with TestClient(app_mod.create_app()):
        # __enter__ has returned -> the lifespan reached `yield` and the server accepts traffic.
        # The startup sweep must already be mid-charge (proving it was scheduled, not skipped) while
        # NOT having blocked __enter__ on the 10s backend.
        elapsed = time.time() - started
        assert elapsed < 5.0, f"startup blocked on the slow charge backlog ({elapsed:.1f}s)"
        assert in_charge.wait(timeout=5.0), "startup sweep was not scheduled at all"
        release.set()  # let the slow charge finish so shutdown is clean


def test_completion_hook_failure_is_recoverable_by_sweep(monkeypatch, tmp_path):
    """End to end across the two paths: the inline hook hits a transient BillingError and marks the
    run failed; the sweep then recovers it. This is the exact silent-revenue-leak scenario."""
    import flash.runner as runner
    from flash.runner import lifecycle
    from flash.server.billing import BillingError

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _save_run(runner, tmp_path, billing_state="pending")

    state = {"fail": True}

    def charge(*, internal_key, status):
        if state["fail"]:
            raise BillingError(503, "transient")
        return {"amountCents": 123}

    monkeypatch.setattr("flash.server.billing.charge_completed_run", charge)

    # inline completion hook (runs right after `done`) hits the blip and records failure
    lifecycle._charge_completed_run_best_effort(spec, io.StringIO())
    assert runner.get_status("run-1").billing_state == "failed"

    # blip clears; the background sweep recovers the uncharged run
    state["fail"] = False
    assert billing_retry.retry_completion_charges_once() == 1
    assert runner.get_status("run-1").billing_state == "charged"
