"""Regression test: `slm cancel` must reliably stop the REMOTE Flash worker.

Bug: ``cancel_run`` called ``stop_endpoint``, which only scales endpoints found in the
*current process's* in-memory cache. In a fresh ``slm cancel`` invocation that cache is empty,
so the remote RunPod worker kept running (and billing) until the wall-clock cap. Fix:
``cancel_run`` uses ``terminate_endpoint`` to look the run's uniquely-named endpoint up in
runpod_flash's persisted registry and delete it via the RunPod API (cross-process).
"""

import types

import autoslm.providers.runpod.train as ftrain
from autoslm.providers.runpod.train import _run_suffix, _select_endpoint_resources, endpoint_name


def _res(name):
    return types.SimpleNamespace(name=name)


def test_select_matches_live_prefixed_endpoint():
    target = endpoint_name(
        "RTX 5090", _run_suffix("autoslm-123-c220526e")
    )  # autoslm-train-5090-c220526e
    resources = {
        "u1": _res(f"live-{target}"),  # the live-provisioned resource for this run
        "u2": _res("autoslm-train-5090-deadbeef"),  # a different run
        "u3": _res("live-autoslm-train-4090-c220526e"),  # different GPU class
    }
    assert _select_endpoint_resources(resources, target) == ["u1"]


def test_select_empty_target_matches_nothing():
    assert _select_endpoint_resources({"u1": _res("live-autoslm-train-5090-x")}, "") == []


def test_terminate_endpoint_never_raises_when_sdk_missing(monkeypatch):
    # ensure_auth raises (no key) -> terminate_endpoint must swallow and return a result list
    import autoslm.providers.runpod.auth as auth

    monkeypatch.setattr(auth, "ensure_auth", lambda: (_ for _ in ()).throw(RuntimeError("no key")))
    out = ftrain.terminate_endpoint("RTX 5090", "autoslm-1-abcd1234")
    assert isinstance(out, list)
    assert out
    assert out[0]["success"] is False


def test_cancel_run_calls_terminate_and_marks_cancelled(tmp_path, monkeypatch):
    import autoslm.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from autoslm.spec import JobSpec

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "gpu": {"type": "RTX 5090"},
            "run_id": "autoslm-9-feedface",
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
    assert calls == {"gpu": "RTX 5090", "run_id": "autoslm-9-feedface"}, (
        "must terminate the remote endpoint"
    )
    assert out.state == "cancelled"


def test_cancel_deployed_run_marks_deployment_inactive(tmp_path, monkeypatch):
    # Cancelling a deployed run tears down its serve endpoint; the deployment record
    # must flip to "undeployed" so /v1/deployments and /chat stop treating the
    # cancelled run as active (and can't recreate the endpoint).
    import autoslm.runner as orch
    import autoslm.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from autoslm.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "autoslm-dep-1"})
    st = orch.RunStatus(
        run_id=spec.run_id,
        state="deployed",
        spec=spec.to_dict(),
        deployment={"state": "ready", "gpu": "RTX 5090", "mode": "dev"},
    )
    orch._save_status(st)

    monkeypatch.setattr(deploy, "undeploy_adapter", lambda *a, **k: ["autoslm-serve-5090-x"])
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"


def test_terminate_endpoint_holds_lock_across_isolation(monkeypatch):
    """Regression (6 bot threads): isolate_flash_state() + the ResourceManager lookup must run
    UNDER FLASH_SDK_LOCK, not just the undeploy. isolate_flash_state swaps runpod_flash's
    process-wide registry globals, so a concurrent deploy could swap the scope mid-teardown.
    Asserts the lock is held when isolate_flash_state runs (and released afterward)."""
    import autoslm.providers.runpod.auth as auth

    monkeypatch.setattr(auth, "ensure_auth", lambda: None)
    held = {}

    def rec_isolate(scope=None):
        held["locked"] = ftrain.FLASH_SDK_LOCK.locked()
        raise RuntimeError("short-circuit before the real SDK lookup")

    monkeypatch.setattr(ftrain, "isolate_flash_state", rec_isolate)
    out = ftrain.terminate_endpoint("RTX 5090", "autoslm-1-abcd1234")
    assert held.get("locked") is True, "isolate_flash_state must run while holding FLASH_SDK_LOCK"
    assert ftrain.FLASH_SDK_LOCK.locked() is False, "lock must be released after terminate"
    assert isinstance(out, list)  # still never raises
    assert out
    assert out[0]["success"] is False


def test_cancel_run_noop_when_terminal(tmp_path, monkeypatch):
    import autoslm.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from autoslm.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "autoslm-done-1"})
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="done", spec=spec.to_dict()))

    called = {"v": False}
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: called.__setitem__("v", True))
    out = orch.cancel_run(spec.run_id)
    assert out.state == "done"
    assert called["v"] is False, "must not tear down endpoints for an already-terminal run"
