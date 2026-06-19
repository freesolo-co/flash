"""Regression test: `slm cancel` must reliably stop the REMOTE Flash worker.

Bug: ``cancel_run`` called ``stop_endpoint``, which only scales endpoints found in the
*current process's* in-memory cache. In a fresh ``slm cancel`` invocation that cache is empty,
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
    target = endpoint_name(
        "RTX 5090", _run_suffix("flash-123-c220526e")
    )  # flash-5090-c220526e
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
        deployment={"state": "ready", "gpu": "RTX 5090", "mode": "dev"},
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
        deployment={"state": "ready", "gpu": "RTX 5090", "mode": "dev"},
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
        deployment={"state": "ready", "gpu": "RTX 5090", "mode": "dev"},
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
        deployment={"state": "ready", "gpu": "RTX 5090", "mode": "dev"},
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
    assert out.state == "done", "a genuine training-completion `done` must NOT be clobbered by cancel"
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

    monkeypatch.setattr(ftrain, "isolate_flash_state", rec_isolate)
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
