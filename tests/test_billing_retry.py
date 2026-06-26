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
    assert not billing_retry._needs_charge(
        st(state="failed", billing_context=ctx, billing_state="pending")
    )
    assert not billing_retry._needs_charge(
        st(state="cancelled", billing_context=ctx, billing_state="pending")
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
    startup sweep is what charges it on the next boot."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")

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
        pass  # entering the context runs the lifespan startup, which sweeps pending charges

    assert charged == ["run-1"]
    assert runner.get_status("run-1").billing_state == "charged"


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
