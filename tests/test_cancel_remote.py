"""Regression test: `flash cancel` must reliably stop the REMOTE Flash worker.

Bug: ``cancel_run`` called ``stop_endpoint``, which only scales endpoints found in the
*current process's* in-memory cache. In a fresh ``flash cancel`` invocation that cache is empty,
so the remote RunPod worker kept running (and billing) until the wall-clock cap. Fix:
``cancel_run`` uses ``terminate_endpoint`` to look the run's uniquely-named endpoint up in
runpod_flash's persisted registry and delete it via the RunPod API (cross-process).
"""

from __future__ import annotations

import types

import flash.providers.runpod.train as ftrain
from flash.providers.runpod.train import _run_suffix, _select_endpoint_resources, endpoint_name


def _res(name):
    return types.SimpleNamespace(name=name)


def test_select_matches_live_prefixed_endpoint():
    target = endpoint_name("RTX 5090", _run_suffix("flash-123-c220526e"))  # flash-5090-c220526e
    resources = {
        "u1": _res(f"live-{target}"),  # the live-provisioned resource for this run
        "u2": _res("flash-5090-deadbeef"),  # a different run
        "u3": _res("live-flash-4090-c220526e"),  # different GPU class
    }
    assert _select_endpoint_resources(resources, target) == ["u1"]


def test_select_empty_target_matches_nothing():
    assert _select_endpoint_resources({"u1": _res("live-flash-5090-x")}, "") == []


def test_terminate_endpoint_never_raises_when_sdk_missing(monkeypatch):
    # ensure_auth raises (no key) -> terminate_endpoint must swallow and return a result list
    import flash.providers.runpod.auth as auth

    monkeypatch.setattr(auth, "ensure_auth", lambda: (_ for _ in ()).throw(RuntimeError("no key")))
    out = ftrain.terminate_endpoint("RTX 5090", "flash-1-abcd1234")
    assert isinstance(out, list)
    assert out
    assert out[0]["success"] is False


def test_cancel_run_calls_terminate_and_marks_cancelled(tmp_path, monkeypatch):
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "gpu": {"type": "RTX 5090"},
            "run_id": "flash-9-feedface",
        }
    )
    st = orch.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
    orch._save_status(st)

    calls = {}

    def fake_terminate(gpu, run_id):
        calls["gpu"] = gpu
        calls["run_id"] = run_id
        return [{"success": True}]

    monkeypatch.setattr(ftrain, "terminate_endpoint", fake_terminate)

    out = orch.cancel_run(spec.run_id)
    assert calls == {"gpu": "RTX 5090", "run_id": "flash-9-feedface"}, (
        "must terminate the remote endpoint"
    )
    assert out.state == "cancelled"


def test_cancel_deployed_run_marks_deployment_inactive(tmp_path, monkeypatch):
    # Cancelling a deployed run tears down its serve endpoint; the deployment record
    # must flip to "undeployed" so /v1/deployments and /chat stop treating the
    # cancelled run as active (and can't recreate the endpoint).
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-dep-1"})
    st = orch.RunStatus(
        run_id=spec.run_id,
        state="deployed",
        spec=spec.to_dict(),
        deployment={"state": "ready", "gpu": "RTX 5090"},
    )
    orch._save_status(st)

    monkeypatch.setattr(deploy, "undeploy_adapter", lambda *a, **k: ["flash-serve-5090-x"])
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"


def test_cancel_deployed_run_undeploy_goes_through_lock_guarded_path(tmp_path, monkeypatch):
    # Regression: the deployed branch used a bare _save_status OUTSIDE _STATUS_LOCK, which
    # persisted a stale pre-teardown snapshot and bypassed serialization. It must instead
    # mark the deployment inactive through the lock-guarded mark_deployment_undeployed
    # helper, and that write must happen while _STATUS_LOCK is held.
    import inspect

    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-dep-lock"})
    st = orch.RunStatus(
        run_id=spec.run_id,
        state="deployed",
        spec=spec.to_dict(),
        deployment={"state": "ready", "gpu": "RTX 5090"},
    )
    orch._save_status(st)

    monkeypatch.setattr(deploy, "undeploy_adapter", lambda *a, **k: ["flash-serve-5090-x"])
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    # The undeploy write must route through the lock-guarded helper (not a bare _save_status
    # outside _STATUS_LOCK, the old racy path); that helper holds _STATUS_LOCK.
    assert "with _STATUS_LOCK" in inspect.getsource(orch.mark_deployment_undeployed)

    called = []
    real_helper = orch.mark_deployment_undeployed

    def spy(run_id):
        called.append(run_id)
        return real_helper(run_id)

    monkeypatch.setattr(orch, "mark_deployment_undeployed", spy)

    out = orch.cancel_run(spec.run_id)
    assert called == [spec.run_id], "undeploy must go through mark_deployment_undeployed"
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"


def test_cancel_deployed_run_undeployed_even_when_raced_to_terminal(tmp_path, monkeypatch):
    # Race: while cancel_run tears down a `deployed` run, a concurrent mark_undeployed moves
    # the run to terminal `done` on disk. The deployment-field write must NOT re-assert a
    # non-terminal state (the old _update(run_id, "deployed", deployment=...) path no-ops
    # against the terminal `done` CAS, leaving the deployment advertised as `ready`). It must
    # mark the deployment undeployed regardless of the terminal race.
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-dep-race"})
    st = orch.RunStatus(
        run_id=spec.run_id,
        state="deployed",
        spec=spec.to_dict(),
        deployment={"state": "ready", "gpu": "RTX 5090"},
    )
    orch._save_status(st)

    monkeypatch.setattr(deploy, "undeploy_adapter", lambda *a, **k: ["flash-serve-5090-x"])
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    # Inject the race: a concurrent mark_undeployed flips the run to terminal `done` AFTER
    # cancel_run's initial get_status (state="deployed") but BEFORE the deployment is retired.
    def racing_undeploy(*a, **k):
        # mark_undeployed moves a live `deployed` run to terminal `done`.
        orch.mark_undeployed(spec.run_id)
        return ["flash-serve-5090-x"]

    monkeypatch.setattr(deploy, "undeploy_adapter", racing_undeploy)

    out = orch.cancel_run(spec.run_id)
    # Explicit cancel WINS over the racing undeploy: even though mark_undeployed flipped the
    # run to terminal `done`, cancel_run's final transition (allow_from_terminal) overrides it
    # so the run ends `cancelled`, and the deployment is reliably retired regardless of the race.
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed", (
        "deployment must end undeployed even when the run raced to terminal `done`"
    )


def test_cancel_wins_over_racing_undeploy_done(tmp_path, monkeypatch):
    # Cancel-race regression: while cancel_run tears down a `deployed` run, a concurrent
    # mark_undeployed() moves the run to terminal `done`. The user explicitly asked to cancel,
    # so the final transition must OVERRIDE the racing `done` — the run must end `cancelled`,
    # not `done`. (Mirrors test_cancel_deployed_run_undeployed_even_when_raced_to_terminal but
    # asserts the state verdict specifically.)
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancel-wins"})
    st = orch.RunStatus(
        run_id=spec.run_id,
        state="deployed",
        spec=spec.to_dict(),
        deployment={"state": "ready", "gpu": "RTX 5090"},
    )
    orch._save_status(st)

    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    # The racing undeploy flips the run to terminal `done` mid-cancel (after cancel_run's
    # initial non-terminal read, before its final `cancelled` write).
    def racing_undeploy(*a, **k):
        orch.mark_undeployed(spec.run_id)
        assert orch.get_status(spec.run_id).state == "done"  # the race landed
        return ["flash-serve-5090-x"]

    monkeypatch.setattr(deploy, "undeploy_adapter", racing_undeploy)

    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled", "explicit cancel must win over a racing undeploy `done`"
    assert orch.get_status(spec.run_id).state == "cancelled"
    assert out.deployment["state"] == "undeployed"


def test_cancel_loses_to_racing_genuine_completion_done(tmp_path, monkeypatch):
    # Cancel-race regression (the mirror of test_cancel_wins_over_racing_undeploy_done): while
    # cancel_run tears down a NON-deployed `running` run, the run's OWN training thread finishes
    # and writes a GENUINE training-completion `done` (real metrics: cost_usd + artifacts_dir).
    # The run actually finished, so its result MUST be preserved — cancel must NOT clobber it to
    # `cancelled`. (A blunt allow_from_terminal=True would discard the real result here; the
    # override is scoped to runs that were `deployed` at entry, which a `running` run is not.)
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-finish-race"})
    # A `running` run (NOT deployed): no deployment, an in-flight training thread.
    st = orch.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
    orch._save_status(st)

    # Inject the race: the training thread completes mid-teardown and writes the terminal
    # `done` with a real result (mirrors _run_job_inner's finish path) AFTER cancel_run's
    # initial non-terminal read but BEFORE its final `cancelled` write.
    def racing_completion(*a, **k):
        orch._update(spec.run_id, "done", cost_usd=1.23, artifacts_dir="/runs/finished")
        assert orch.get_status(spec.run_id).state == "done"  # the genuine finish landed
        return [{"success": True}]

    monkeypatch.setattr(ftrain, "terminate_endpoint", racing_completion)

    out = orch.cancel_run(spec.run_id)
    assert out.state == "done", (
        "a genuine training-completion `done` must NOT be clobbered by cancel"
    )
    assert orch.get_status(spec.run_id).state == "done"
    assert out.cost_usd == 1.23, "the finished run's real result (cost) must be preserved"
    assert out.artifacts_dir == "/runs/finished"


def test_terminate_endpoint_holds_lock_across_isolation(monkeypatch):
    """Regression (6 bot threads): isolate_flash_state() + the ResourceManager lookup must run
    UNDER FLASH_SDK_LOCK, not just the undeploy. isolate_flash_state swaps runpod_flash's
    process-wide registry globals, so a concurrent deploy could swap the scope mid-teardown.
    Asserts the lock is held when isolate_flash_state runs (and released afterward)."""
    import flash.providers.runpod.auth as auth

    monkeypatch.setattr(auth, "ensure_auth", lambda: None)
    held = {}

    def rec_isolate(scope=None):
        held["locked"] = ftrain.FLASH_SDK_LOCK.locked()
        raise RuntimeError("short-circuit before the real SDK lookup")

    monkeypatch.setattr(ftrain.endpoints, "isolate_flash_state", rec_isolate)
    out = ftrain.terminate_endpoint("RTX 5090", "flash-1-abcd1234")
    assert held.get("locked") is True, "isolate_flash_state must run while holding FLASH_SDK_LOCK"
    assert ftrain.FLASH_SDK_LOCK.locked() is False, "lock must be released after terminate"
    assert isinstance(out, list)  # still never raises
    assert out
    assert out[0]["success"] is False


def test_cancel_run_noop_when_terminal(tmp_path, monkeypatch):
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-done-1"})
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="done", spec=spec.to_dict()))

    called = {"v": False}
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: called.__setitem__("v", True))
    out = orch.cancel_run(spec.run_id)
    assert out.state == "done"
    assert called["v"] is False, "must not tear down endpoints for an already-terminal run"


# ---------------------------------------------------------------------------
# Recovery TOCTOU: a run flipped terminal mid-recovery must not submit paid work
# ---------------------------------------------------------------------------
def _make_poll_provider(monkeypatch, *, on_poll):
    """Wire flash.providers.get_provider to a stub provider whose poll() runs ``on_poll``.

    Also no-ops _gc_run_endpoints so attach_run's teardown doesn't reach the real SDK.
    """
    import flash.providers as providers
    import flash.runner as orch

    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda *a, **k: None)

    class _StubProvider:
        def poll(self, handle, spec, seed, *, log=None):
            return on_poll(handle, spec, seed)

    monkeypatch.setattr(providers, "get_provider", lambda name: _StubProvider())


def test_attach_run_recovery_skips_seed_loop_when_raced_terminal(tmp_path, monkeypatch):
    """Recovery-path TOCTOU regression (the reviewed bug). attach_run checks the terminal state
    ONCE up front, then runs a long poll. If a concurrent thread/process flips the run terminal
    (e.g. another attach_run marks it `failed`) DURING that poll, the not-ok recovery path must
    NOT resume `_run_seed_loop` — doing so would submit PAID GPU work for an already-terminal run.
    The atomic guard: _update(.., "running", ..)'s sticky CAS rejects the write (returns False),
    so the resume is skipped. Here we flip the run to `failed` from inside the (mocked) poll, then
    return a not-ok result, and assert the seed loop is never entered."""
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-race-terminal"})
    st = orch.RunStatus(
        run_id=spec.run_id,
        state="running",
        spec=spec.to_dict(),
        remote={"provider": "runpod", "endpoint_id": "ep-1", "job_id": "job-1", "seed": 0},
    )
    orch._save_status(st)

    # The seed loop is the PAID-work entry point; it must never be called for a terminal run.
    seed_loop_calls = {"n": 0}
    monkeypatch.setattr(
        orch,
        "_run_seed_loop",
        lambda *a, **k: seed_loop_calls.__setitem__("n", seed_loop_calls["n"] + 1),
    )

    from flash.providers.base import PollResult

    def racing_poll(handle, spec, seed):
        # A concurrent recovery/cancel flips the run terminal AFTER attach_run's initial check
        # (top of attach_run) but BEFORE the not-ok recovery resume below.
        orch._update(spec.run_id, "failed", error="raced terminal by another thread")
        assert orch.get_status(spec.run_id).state == "failed"  # the race landed
        return PollResult(False, failure="stalled", detail="control plane was down")

    _make_poll_provider(monkeypatch, on_poll=racing_poll)

    out = orch.attach_run(spec.run_id)
    assert seed_loop_calls["n"] == 0, (
        "must NOT submit paid work (resume seed loop) for a run raced to terminal"
    )
    assert out.state == "failed", "the authoritative terminal state must be preserved"
    assert orch.get_status(spec.run_id).state == "failed"


def test_attach_run_recovery_resumes_seed_loop_when_still_active(tmp_path, monkeypatch):
    """Happy-path guard: the TOCTOU fix must NOT regress a genuine recovery. A not-ok poll on a
    run that is STILL active (no terminal race) must resume `_run_seed_loop` exactly as before."""
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-recover-active"})
    st = orch.RunStatus(
        run_id=spec.run_id,
        state="running",
        spec=spec.to_dict(),
        remote={"provider": "runpod", "endpoint_id": "ep-1", "job_id": "job-1", "seed": 0},
    )
    orch._save_status(st)

    seed_loop_calls = {"n": 0}
    monkeypatch.setattr(
        orch,
        "_run_seed_loop",
        lambda *a, **k: seed_loop_calls.__setitem__("n", seed_loop_calls["n"] + 1),
    )

    from flash.providers.base import PollResult

    _make_poll_provider(
        monkeypatch,
        on_poll=lambda h, s, seed: PollResult(False, failure="stalled", detail="redeploy"),
    )

    out = orch.attach_run(spec.run_id)
    assert seed_loop_calls["n"] == 1, "a still-active run must resume the seed loop (no regression)"
    assert out.state == "running"


def test_run_seed_loop_bails_on_terminal_before_paid_work(tmp_path, monkeypatch):
    """Defense in depth: _run_seed_loop's own pre-submit guard now bails on ANY terminal state
    (not just `cancelled`). If the run is terminal when the loop is entered — e.g. a concurrent
    thread marked it `done`/`failed` after the caller decided to resume — it must raise
    _RunCancelled and never call _submit_seed_supervised (the paid GPU submit)."""
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict(
        {"gpu": {"type": "RTX 5090"}, "run_id": "flash-loop-terminal", "train": {"seeds": [0, 1]}}
    )
    # The run is already terminal (failed) before the loop runs.
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="failed", spec=spec.to_dict()))

    submitted = {"n": 0}
    monkeypatch.setattr(
        orch,
        "_submit_seed_supervised",
        lambda *a, **k: submitted.__setitem__("n", submitted["n"] + 1),
    )

    import io

    import pytest

    with pytest.raises(orch._RunCancelled):
        orch._run_seed_loop(spec, io.StringIO(), start_index=0, prior_cost=0.0)
    assert submitted["n"] == 0, "no paid GPU work may be submitted for an already-terminal run"
    assert orch.get_status(spec.run_id).state == "failed", "the terminal state must be untouched"


def test_update_returns_false_when_terminal_sticky(tmp_path, monkeypatch):
    """_update now reports whether the transition applied: True normally, False when the sticky
    terminal CAS rejects it. The recovery guard relies on this signal."""
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-update-ret"})
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()))
    assert orch._update(spec.run_id, "running", cost_usd=1.0) is True
    assert orch._update(spec.run_id, "failed", error="boom") is True  # terminal write applies
    # Now terminal: a non-terminal transition is rejected and reported False.
    assert orch._update(spec.run_id, "running") is False
    assert orch.get_status(spec.run_id).state == "failed"
