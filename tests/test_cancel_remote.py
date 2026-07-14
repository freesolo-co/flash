"""Regression test: `flash cancel` must reliably stop the REMOTE Flash worker.

Bug: ``cancel_run`` called ``stop_endpoint``, which only scales endpoints found in the
*current process's* in-memory cache. In a fresh ``flash cancel`` invocation that cache is empty,
so the remote RunPod worker kept running (and billing) until the wall-clock cap. Fix:
``cancel_run`` uses ``terminate_endpoint`` to look the run's uniquely-named endpoint up in
runpod_flash's persisted registry and delete it via the RunPod API (cross-process).
"""

from __future__ import annotations

import sys
import types

import pytest

import flash.providers.runpod.train as ftrain
from flash.providers.runpod.train import _run_suffix, _select_endpoint_resources, endpoint_name


def _res(name):
    return types.SimpleNamespace(name=name)


def test_isolate_flash_state_resets_runpod_flash_manager_on_scope_change(tmp_path, monkeypatch):
    import flash.providers.runpod.train.endpoints as ep_mod

    for mod_name in (
        "runpod_flash",
        "runpod_flash.core",
        "runpod_flash.core.resources",
    ):
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, mod_name, stub)

    class FakeRM:
        pass

    FakeRM._resources = {"old-class": object()}
    FakeRM._resource_configs = {"old-class": "hash"}
    FakeRM._deployment_locks = {"old-class": object()}
    FakeRM._resources_initialized = True

    fake_instance = types.SimpleNamespace(
        _resources={"old-instance": object()},
        _resource_configs={"old-instance": "hash"},
        _deployment_locks={"old-instance": object()},
    )
    FakeRM._instances = {FakeRM: fake_instance}

    rm_mod = types.ModuleType("runpod_flash.core.resources.resource_manager")
    rm_mod.ResourceManager = FakeRM
    rm_mod.FLASH_STATE_DIR = tmp_path / "old"
    rm_mod.RESOURCE_STATE_FILE = tmp_path / "old" / "resources.pkl"
    rm_mod.RUNPOD_FLASH_DIR = tmp_path / "old"
    monkeypatch.setitem(sys.modules, "runpod_flash.core.resources.resource_manager", rm_mod)
    monkeypatch.setenv("HOME", str(tmp_path))

    ep_mod.isolate_flash_state("run-a")

    state_dir = tmp_path / ".flash" / "flash-state" / "run-a"
    assert state_dir == rm_mod.FLASH_STATE_DIR
    assert state_dir / "resources.pkl" == rm_mod.RESOURCE_STATE_FILE
    assert state_dir == rm_mod.RUNPOD_FLASH_DIR
    assert FakeRM._resources == {}
    assert FakeRM._resource_configs == {}
    assert FakeRM._deployment_locks == {}
    assert fake_instance._resources == {}
    assert fake_instance._resource_configs == {}
    assert fake_instance._deployment_locks == {}
    assert FakeRM._resources_initialized is False

    FakeRM._resources["kept"] = object()
    fake_instance._resources["kept"] = object()
    FakeRM._resources_initialized = True
    ep_mod.isolate_flash_state("run-a")

    assert "kept" in FakeRM._resources
    assert "kept" in fake_instance._resources
    assert FakeRM._resources_initialized is True


def test_get_train_endpoint_locks_sdk_state_and_does_not_cache_run_scoped_handlers(monkeypatch):
    import flash.providers.runpod.auth as auth
    import flash.providers.runpod.jobs as jobs
    import flash.providers.runpod.train.endpoints as ep_mod

    locked_events = []

    class FakeEndpoint:
        def __init__(self, **kwargs):
            assert ep_mod.FLASH_SDK_LOCK.locked()
            self.kwargs = kwargs
            locked_events.append(("endpoint", kwargs["name"]))

        def __call__(self, fn):
            assert ep_mod.FLASH_SDK_LOCK.locked()
            locked_events.append(("handler", self.kwargs["name"]))
            return types.SimpleNamespace(endpoint=self, fn=fn)

        def _build_resource_config(self):
            assert ep_mod.FLASH_SDK_LOCK.locked()
            locked_events.append(("config", self.kwargs["name"]))
            return {}

    runpod_flash = types.ModuleType("runpod_flash")
    runpod_flash.Endpoint = FakeEndpoint
    monkeypatch.setitem(sys.modules, "runpod_flash", runpod_flash)

    monkeypatch.setattr(auth, "ensure_auth", lambda: None)
    monkeypatch.setattr(ep_mod, "_patch_runpod_backoff", lambda: None)

    def rec_isolate(scope):
        assert ep_mod.FLASH_SDK_LOCK.locked()
        locked_events.append(("isolate", scope))

    acquired = []
    monkeypatch.setattr(ep_mod, "isolate_flash_state", rec_isolate)
    monkeypatch.setattr(ep_mod, "_acquire_endpoint_slot", lambda name: acquired.append(name))
    monkeypatch.setattr(ep_mod, "_release_endpoint_slot", lambda _name: None)
    monkeypatch.setattr(ep_mod, "canonical_gpu", lambda gpu: gpu)
    monkeypatch.setattr(ep_mod, "flash_gpu", lambda gpu: gpu)
    monkeypatch.setattr(ep_mod, "gpu_short", lambda gpu: gpu.lower().replace(" ", ""))
    monkeypatch.setattr(ep_mod, "worker_image_for_gpu", lambda *_args, **_kwargs: "image")
    monkeypatch.setattr(jobs, "weight_cache_endpoint_kwargs", lambda _spec: {})
    monkeypatch.setattr(jobs, "apply_disk_gb", lambda _cfg, _disk_gb: None)
    monkeypatch.setattr(ep_mod, "_ENDPOINT_CACHE", {})

    run_handler = ep_mod.get_train_endpoint("RTX 5090", name_suffix="run-a")
    assert run_handler.endpoint.kwargs["name"] == "flash-rtx5090-run-a"
    assert ep_mod._ENDPOINT_CACHE == {}
    assert acquired == ["flash-rtx5090-run-a"]
    assert ("isolate", "run-a") in locked_events

    default_handler = ep_mod.get_train_endpoint("RTX 5090")
    assert default_handler.endpoint.kwargs["name"] == "flash-rtx5090"
    assert {"flash-rtx5090": default_handler} == ep_mod._ENDPOINT_CACHE
    assert acquired == ["flash-rtx5090-run-a", "flash-rtx5090"]
    assert ("isolate", None) in locked_events

    cached_handler = ep_mod.get_train_endpoint("RTX 5090")
    assert cached_handler is default_handler
    assert acquired == ["flash-rtx5090-run-a", "flash-rtx5090"]


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


def _run_spec(run_id: str):
    from flash.spec import JobSpec

    return JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})


def _checkpoint_revision(run_id: str, step: int, sha: str = "a") -> str:
    return f"{run_id}@step-{step}." + sha * 40


class _AcquirableTestLock:
    def acquire(self, blocking: bool = True) -> bool:
        self.__enter__()
        return True

    def release(self) -> None:
        return None


def _ready_checkpoint(orch, run_id: str, step: int, *, remote: dict | None = None) -> dict:
    spec = _run_spec(run_id)
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            remote=remote,
        )
    )
    deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "adapter_revision": _checkpoint_revision(run_id, step),
        "checkpoint_step": step,
    }
    orch.mark_checkpoint_deployed(
        run_id,
        deployment,
        verification_generation=orch.verified_adapter_revision_generation(run_id),
    )
    return deployment


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


def test_cancel_run_tears_down_initial_remote_before_waiting_for_deploy_lock(
    tmp_path, monkeypatch
):
    import flash.providers as providers
    import flash.runner as orch
    import flash.server._locks as locks
    from flash.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-initial-remote"
    remote = {"provider": "stub", "job_id": "initial-job"}
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})
    orch._save_status(
        orch.RunStatus(run_id=run_id, state="running", spec=spec.to_dict(), remote=remote)
    )
    calls = []

    class StubProvider:
        def cancel(self, handle):
            calls.append(("cancel", handle.to_dict()))

        def destroy(self, handle):
            calls.append(("destroy", handle.to_dict()))

    class ObservedLock(_AcquirableTestLock):
        def __enter__(self):
            assert calls == [("cancel", remote), ("destroy", remote)]

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(providers, "get_provider", lambda provider: StubProvider())
    monkeypatch.setattr(locks, "_deploy_lock", lambda target: ObservedLock())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda spec: None)

    out = orch.cancel_run(run_id)

    assert out.state == "cancelled"
    assert calls == [("cancel", remote), ("destroy", remote)]


def test_cancel_tears_down_training_before_checkpoint_serving_decision(tmp_path, monkeypatch):
    import flash.providers as providers
    import flash.runner as orch
    import flash.runner.verified_revisions as verified_revisions
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-order"
    remote = {"provider": "stub", "job_id": "training-job"}
    deployment = _ready_checkpoint(orch, run_id, 40, remote=remote)
    events = []

    class StubProvider:
        def cancel(self, handle):
            events.append("provider-cancel")

        def destroy(self, handle):
            events.append("provider-destroy")

    real_read = verified_revisions.read_verified_adapter_revisions

    def read_verified(target):
        events.append("serving-decision")
        return real_read(target)

    monkeypatch.setattr(providers, "get_provider", lambda _provider: StubProvider())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: events.append("endpoint-gc"))
    monkeypatch.setattr(verified_revisions, "read_verified_adapter_revisions", read_verified)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("a valid checkpoint deployment must remain serving"),
    )

    out = orch.cancel_run(run_id)

    assert events == [
        "provider-cancel",
        "provider-destroy",
        "endpoint-gc",
        "serving-decision",
    ]
    assert out.state == "cancelled"
    assert out.deployment == deployment


def test_cancel_reruns_endpoint_gc_after_contended_deploy_lock(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.server._locks as locks

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-contended-endpoint-gc"
    spec = _run_spec(run_id)
    orch._save_status(orch.RunStatus(run_id=run_id, state="running", spec=spec.to_dict()))
    resource = {"present": False}
    events = []

    class ContendedLock:
        held = False

        def acquire(self, blocking: bool = True) -> bool:
            if not blocking:
                events.append("contended")
                return False
            self.held = True
            resource["present"] = True
            events.append("resource-materialized")
            return True

        def release(self) -> None:
            assert self.held is True
            self.held = False
            events.append("released")

    def gc_endpoint(_spec):
        if resource["present"]:
            resource["present"] = False
            events.append("gc-removed")
        else:
            events.append("gc-empty")

    lock = ContendedLock()
    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: lock)
    monkeypatch.setattr(orch, "_gc_run_endpoints", gc_endpoint)

    out = orch.cancel_run(run_id)

    assert out.state == "cancelled"
    assert resource["present"] is False
    assert events == [
        "gc-empty",
        "contended",
        "resource-materialized",
        "gc-removed",
        "released",
    ]


def test_contended_cancel_process_death_leaves_revocation_retryable(tmp_path, monkeypatch):
    import threading

    import flash.runner as orch
    import flash.serve.deploy as deploy
    import flash.server._locks as locks

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-contended-cancel-death"
    spec = _run_spec(run_id)
    orch._save_status(orch.RunStatus(run_id=run_id, state="done", spec=spec.to_dict()))
    revision = f"{run_id}@final." + "a" * 40
    orch.mark_deployed(
        run_id,
        {"state": "ready", "adapter_revision": revision, "endpoint_name": "final"},
        verification_generation=orch.verified_adapter_revision_generation(run_id),
    )

    class SimulatedProcessDeath(BaseException):
        pass

    class DeathWhileWaitingForDeployLock:
        def acquire(self, blocking: bool = True) -> bool:
            if not blocking:
                return False
            pending = orch.get_status(run_id)
            assert pending.deployment["state"] == "revocation_failed"
            assert pending.deployment["retryable"] is True
            raise SimulatedProcessDeath

        def release(self) -> None:
            pytest.fail("an unacquired lock must not be released")

    lock_attempts = iter([DeathWhileWaitingForDeployLock(), threading.Lock()])
    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: next(lock_attempts))
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    attempts = []
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: attempts.append(target) or [])

    with pytest.raises(SimulatedProcessDeath):
        orch.cancel_run(run_id)

    interrupted = orch.get_status(run_id)
    assert interrupted.state == "deployed"
    assert interrupted.deployment["state"] == "revocation_failed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()
    assert attempts == []

    retried = orch.cancel_run(run_id)

    assert attempts == [run_id]
    assert retried.state == "cancelled"
    assert retried.deployment["state"] == "undeployed"


def test_cancel_preserves_ready_verified_same_step_checkpoint(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-preserve"
    deployment = _ready_checkpoint(orch, run_id, 80)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("a valid checkpoint deployment must remain serving"),
    )
    monkeypatch.setattr(
        orch,
        "mark_deployment_undeployed",
        lambda _target: pytest.fail("checkpoint authorization must remain intact"),
    )

    out = orch.cancel_run(run_id)

    assert out.state == "cancelled"
    assert out.deployment == deployment
    assert orch.read_verified_adapter_revisions(run_id) == frozenset(
        {deployment["adapter_revision"]}
    )


def test_cancel_revokes_final_deployment(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-final-revoke"
    spec = _run_spec(run_id)
    orch._save_status(orch.RunStatus(run_id=run_id, state="done", spec=spec.to_dict()))
    revision = f"{run_id}@final." + "f" * 40
    orch.mark_deployed(
        run_id,
        {"state": "ready", "adapter_revision": revision, "endpoint_name": "final"},
        verification_generation=orch.verified_adapter_revision_generation(run_id),
    )
    undeployed = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeployed.append(target))

    out = orch.cancel_run(run_id)

    assert undeployed == [run_id]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


def test_cancel_revokes_inflight_checkpoint_deployment(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-inflight"
    spec = _run_spec(run_id)
    revision = _checkpoint_revision(run_id, 40)
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            deployment={
                "state": "deploying",
                "adapter_revision": revision,
                "checkpoint_step": 40,
            },
        )
    )
    orch.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=orch.verified_adapter_revision_generation(run_id),
    )
    undeployed = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeployed.append(target))

    out = orch.cancel_run(run_id)

    assert undeployed == [run_id]
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


@pytest.mark.parametrize(
    ("checkpoint_step", "revision", "verified"),
    [
        (40, "not-an-immutable-revision", True),
        (40, " " + _checkpoint_revision("flash-checkpoint-invalid", 40), True),
        (40, _checkpoint_revision("flash-other-run", 40), True),
        (40, _checkpoint_revision("flash-checkpoint-invalid", 80), True),
        (True, _checkpoint_revision("flash-checkpoint-invalid", 1), True),
        (40, _checkpoint_revision("flash-checkpoint-invalid", 40), False),
    ],
    ids=[
        "malformed",
        "noncanonical",
        "wrong-run",
        "wrong-step",
        "invalid-step",
        "unverified",
    ],
)
def test_cancel_revokes_invalid_checkpoint_records(
    tmp_path, monkeypatch, checkpoint_step, revision, verified
):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-invalid"
    spec = _run_spec(run_id)
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            deployment={
                "state": "ready",
                "adapter_revision": revision,
                "checkpoint_step": checkpoint_step,
            },
        )
    )
    if verified and revision.strip().startswith(f"{run_id}@"):
        orch.add_verified_adapter_revision(
            run_id,
            revision,
            expected_generation=orch.verified_adapter_revision_generation(run_id),
        )
    undeployed = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeployed.append(target))

    out = orch.cancel_run(run_id)

    assert undeployed == [run_id]
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


@pytest.mark.parametrize(
    ("run_state", "deployment", "expected_state"),
    [
        ("running", {}, "cancelled"),
        ("done", {}, "done"),
        ("dry_run", {}, "dry_run"),
        ("running", [], "cancelled"),
        ("running", {"state": 1}, "cancelled"),
        ("running", {"state": []}, "cancelled"),
    ],
    ids=[
        "nonterminal-missing-state",
        "terminal-missing-state",
        "dry-run-missing-state",
        "non-dict",
        "non-string-state",
        "unhashable-state",
    ],
)
def test_cancel_revokes_malformed_deployment_containers(
    tmp_path, monkeypatch, run_state, deployment, expected_state
):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = f"flash-malformed-{run_state}"
    spec = _run_spec(run_id)
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state=run_state,
            spec=spec.to_dict(),
            deployment=deployment,
        )
    )
    undeployed = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeployed.append(target))

    out = orch.cancel_run(run_id)

    assert undeployed == [run_id]
    assert out.state == expected_state
    assert out.deployment == {"state": "undeployed"}


@pytest.mark.parametrize("deployment_state", ["ready", "deploying", "revocation_failed"])
def test_cancel_dry_run_revokes_active_deployment_state(
    tmp_path, monkeypatch, deployment_state
):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = f"flash-dry-active-{deployment_state}"
    spec = _run_spec(run_id)
    revision = _checkpoint_revision(run_id, 40)
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="dry_run",
            spec=spec.to_dict(),
            deployment={
                "state": deployment_state,
                "adapter_revision": revision,
                "checkpoint_step": 40,
            },
        )
    )
    orch.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=orch.verified_adapter_revision_generation(run_id),
    )
    undeployed = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeployed.append(target))

    out = orch.cancel_run(run_id)

    assert undeployed == [run_id]
    assert out.state == "dry_run"
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


def test_repeated_cancel_preserves_checkpoint_serving(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-repeat"
    deployment = _ready_checkpoint(orch, run_id, 40)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("repeated cancel must not revoke the checkpoint"),
    )

    first = orch.cancel_run(run_id)
    second = orch.cancel_run(run_id)

    assert first.state == second.state == "cancelled"
    assert first.deployment == second.deployment == deployment
    assert orch.read_verified_adapter_revisions(run_id) == frozenset(
        {deployment["adapter_revision"]}
    )


def test_cancel_preserves_checkpoint_that_becomes_ready_before_locked_reread(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    import flash.server._locks as locks

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-race"
    spec = _run_spec(run_id)
    orch._save_status(orch.RunStatus(run_id=run_id, state="running", spec=spec.to_dict()))
    deployment = {
        "state": "ready",
        "adapter_revision": _checkpoint_revision(run_id, 40),
        "checkpoint_step": 40,
        "endpoint_name": "checkpoint",
    }

    class RacingLock(_AcquirableTestLock):
        def __enter__(self):
            orch.mark_checkpoint_deployed(
                run_id,
                deployment,
                verification_generation=orch.verified_adapter_revision_generation(run_id),
            )

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: RacingLock())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("the locked reread must preserve the ready checkpoint"),
    )

    out = orch.cancel_run(run_id)

    assert out.state == "cancelled"
    assert out.deployment == deployment
    assert orch.read_verified_adapter_revisions(run_id) == frozenset(
        {deployment["adapter_revision"]}
    )


def test_cancel_revokes_final_deployment_that_races_before_locked_reread(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    import flash.server._locks as locks

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-final-race"
    spec = _run_spec(run_id)
    orch._save_status(orch.RunStatus(run_id=run_id, state="running", spec=spec.to_dict()))
    revision = f"{run_id}@final." + "e" * 40

    class RacingLock(_AcquirableTestLock):
        def __enter__(self):
            orch._update(run_id, "done", cost_usd=1.0, artifacts_dir="/runs/finished")
            orch.mark_deployed(
                run_id,
                {"state": "ready", "adapter_revision": revision, "endpoint_name": "final"},
                verification_generation=orch.verified_adapter_revision_generation(run_id),
            )

        def __exit__(self, exc_type, exc, traceback):
            return False

    undeployed = []
    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: RacingLock())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeployed.append(target))

    out = orch.cancel_run(run_id)

    assert undeployed == [run_id]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


def test_cancel_run_tears_down_remote_handle_observed_after_entry(tmp_path, monkeypatch):
    import flash.providers as providers
    import flash.runner as orch
    from flash.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-late-remote"
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})
    orch._save_status(orch.RunStatus(run_id=run_id, state="running", spec=spec.to_dict()))
    remote = {"provider": "stub", "job_id": "late-job"}
    calls = []

    class StubProvider:
        def cancel(self, handle):
            calls.append(("cancel", handle.to_dict()))

        def destroy(self, handle):
            calls.append(("destroy", handle.to_dict()))

    def observe_remote(_spec):
        status = orch.get_status(run_id)
        status.remote = remote
        orch._save_status(status)

    monkeypatch.setattr(providers, "get_provider", lambda provider: StubProvider())
    monkeypatch.setattr(orch, "_gc_run_endpoints", observe_remote)

    out = orch.cancel_run(run_id)

    assert out.state == "cancelled"
    assert calls == [("cancel", remote), ("destroy", remote)]


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


def test_cancel_double_undeploy_failure_revokes_authority_and_is_retryable(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    run_id = "flash-dep-revoke"
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})
    orch._save_status(orch.RunStatus(run_id=run_id, state="done", spec=spec.to_dict()))
    revision = f"{run_id}@final." + "a" * 40
    generation = orch.verified_adapter_revision_generation(run_id)
    ready = orch.mark_deployed(
        run_id,
        {"state": "ready", "adapter_revision": revision, "endpoint_name": "https://serve.example"},
        verification_generation=generation,
    )
    assert ready.state == "deployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset({revision})

    attempts = []

    def fail_undeploy(target):
        attempts.append(target)
        raise deploy.ServingError("backend unavailable")

    monkeypatch.setattr(deploy, "undeploy_adapter", fail_undeploy)
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    with pytest.raises(orch.DeploymentRevocationError) as excinfo:
        orch.cancel_run(run_id)

    assert excinfo.value.retryable is True
    assert attempts == [run_id, run_id]
    failed = orch.get_status(run_id)
    assert failed.state == "cancelled"
    assert failed.deployment["state"] == "revocation_failed"
    assert failed.deployment["retryable"] is True
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()

    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: attempts.append(target) or {})
    retried = orch.cancel_run(run_id)

    assert attempts == [run_id, run_id, run_id]
    assert retried.state == "cancelled"
    assert retried.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


def test_cancel_undeploys_deployment_that_raced_in_after_entry_snapshot(tmp_path, monkeypatch):
    # Race: cancel_run enters on a non-`deployed` snapshot (state="running"), but a deploy lands during
    # teardown (running -> done -> deployed) before the terminal `cancelled` write. `deployed` is
    # non-terminal so `cancelled` still wins, but the entry-gated undeploy never ran. cancel_run must
    # re-read post-write and tear down the raced-in deployment so it is never orphaned.
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-dep-racein"})
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()))

    undeployed: list[str] = []
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda rid, *a, **k: undeployed.append(rid) or ["x"]
    )
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    # Inject the deploy race at the last step before the terminal write (after the entry snapshot).
    real_gc = orch._gc_run_endpoints

    def gc_then_deploy(s):
        real_gc(s)
        orch.mark_deployed(
            spec.run_id,
            {
                "state": "ready",
                "gpu": "RTX 5090",
                "adapter_revision": f"{spec.run_id}@final." + "a" * 40,
            },
            verification_generation=orch.verified_adapter_revision_generation(spec.run_id),
        )

    monkeypatch.setattr(orch, "_gc_run_endpoints", gc_then_deploy)

    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert undeployed == [spec.run_id], "the raced-in deployment must be torn down, not orphaned"
    assert (out.deployment or {}).get("state") == "undeployed"


def test_cancel_undeploys_active_deployment_after_terminal_transition_at_lock(
    tmp_path, monkeypatch
):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    import flash.server._locks as locks
    from flash.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-terminal-at-lock"
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            deployment={"state": "ready", "gpu": "RTX 5090"},
        )
    )
    undeployed = []

    class RacingLock(_AcquirableTestLock):
        def __enter__(self):
            orch._update(run_id, "done", cost_usd=1.0, artifacts_dir="/runs/finished")

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: RacingLock())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda target: undeployed.append(target) or ["endpoint"]
    )

    out = orch.cancel_run(run_id)

    assert out.state == "done"
    assert undeployed == [run_id]
    assert out.deployment["state"] == "undeployed"


def test_cancel_revocation_retry_transitions_deployed_run_to_cancelled(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    from flash.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-deployed-revocation-retry"
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="deployed",
            spec=spec.to_dict(),
            deployment={"state": "revocation_failed", "error": "backend unavailable"},
        )
    )
    attempts = []

    def undeploy(target):
        attempts.append(target)
        if len(attempts) == 1:
            raise deploy.ServingError("backend still unavailable")
        return ["endpoint"]

    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)

    retried = orch.cancel_run(run_id)

    assert attempts == [run_id, run_id]
    assert retried.state == "cancelled"
    assert retried.deployment["state"] == "undeployed"


def test_cancel_revocation_retry_reprices_running_run(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    from flash.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-running-revocation-retry"
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            cost_usd=9.0,
            billing_context={"org_id": "org-test"},
            deployment={"state": "revocation_failed", "error": "backend unavailable"},
        )
    )
    attempts = []

    def undeploy(target):
        attempts.append(target)
        if len(attempts) == 1:
            raise deploy.ServingError("backend still unavailable")
        return ["endpoint"]

    charges = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(orch, "effective_spec_from_status", lambda _status: spec)
    monkeypatch.setattr(orch, "actual_steps_run", lambda _status: 3)
    monkeypatch.setattr(
        orch,
        "charge_usd_for_spec",
        lambda *_args, **_kwargs: charges.append(1.25) or 1.25,
    )
    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)

    retried = orch.cancel_run(run_id)

    assert attempts == [run_id, run_id]
    assert charges == [1.25]
    assert retried.state == "cancelled"
    assert retried.cost_usd == 1.25
    assert retried.deployment["state"] == "undeployed"


def test_cancel_retries_revocation_after_terminal_transition_at_lock(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    import flash.server._locks as locks
    from flash.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-terminal-revocation-retry"
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            deployment={"state": "ready", "verification_race": True},
        )
    )

    class RacingLock(_AcquirableTestLock):
        raced = False

        def __enter__(self):
            if not self.raced:
                self.raced = True
                orch._update(run_id, "done", cost_usd=1.0, artifacts_dir="/runs/finished")

        def __exit__(self, exc_type, exc, traceback):
            return False

    lock = RacingLock()
    attempts = []

    def undeploy(target):
        attempts.append(target)
        if len(attempts) == 1:
            raise deploy.ServingError("backend unavailable")
        return ["endpoint"]

    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: lock)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)

    with pytest.raises(orch.DeploymentRevocationError):
        orch.cancel_run(run_id)

    failed = orch.get_status(run_id)
    assert failed.state == "done"
    assert failed.deployment["state"] == "revocation_failed"

    retried = orch.cancel_run(run_id)

    assert attempts == [run_id, run_id]
    assert retried.state == "done"
    assert retried.deployment["state"] == "undeployed"


def test_concurrent_cancel_does_not_rewrite_terminal_billing(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.server._locks as locks
    from flash.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-double-cancel-billing"
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            cost_usd=9.0,
            billing_context={"org_id": "org-test"},
        )
    )

    class ConcurrentCancelLock(_AcquirableTestLock):
        def __enter__(self):
            orch._update(run_id, "cancelled", cost_usd=9.0)

        def __exit__(self, exc_type, exc, traceback):
            return False

    charges = []
    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: ConcurrentCancelLock())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(orch, "effective_spec_from_status", lambda _status: spec)
    monkeypatch.setattr(orch, "actual_steps_run", lambda _status: 1)
    monkeypatch.setattr(
        orch,
        "charge_usd_for_spec",
        lambda *_args, **_kwargs: charges.append(1.0) or 1.0,
    )

    out = orch.cancel_run(run_id)

    assert out.state == "cancelled"
    assert out.cost_usd == 9.0
    assert charges == []


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


def test_cancel_preserves_ready_checkpoint_without_overwriting_genuine_completion(
    tmp_path, monkeypatch
):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-finish-race"
    spec = _run_spec(run_id)
    orch._save_status(orch.RunStatus(run_id=run_id, state="running", spec=spec.to_dict()))
    deployment = {
        "state": "ready",
        "adapter_revision": _checkpoint_revision(run_id, 40),
        "checkpoint_step": 40,
        "endpoint_name": "checkpoint",
    }

    def checkpoint_then_finish(_spec):
        orch.mark_checkpoint_deployed(
            run_id,
            deployment,
            verification_generation=orch.verified_adapter_revision_generation(run_id),
        )
        orch._update(run_id, "done", cost_usd=1.23, artifacts_dir="/runs/finished")

    monkeypatch.setattr(orch, "_gc_run_endpoints", checkpoint_then_finish)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("the verified checkpoint must remain serving"),
    )

    out = orch.cancel_run(run_id)

    assert out.state == "done"
    assert out.cost_usd == 1.23
    assert out.artifacts_dir == "/runs/finished"
    assert out.deployment == deployment
    assert orch.read_verified_adapter_revisions(run_id) == frozenset(
        {deployment["adapter_revision"]}
    )


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


def test_terminate_endpoint_from_async_context_does_not_raise(monkeypatch):
    """Regression: terminate_endpoint must not raise when called from a running event loop.

    The original code called asyncio.run(_undeploy_all()) directly, which raises
    RuntimeError('cannot be called when another event loop is running') from FastAPI/Uvicorn
    lifespan shutdown or any other async context. The fix detects a running loop and falls
    back to a ThreadPoolExecutor thread where asyncio.run() always succeeds.
    """
    import asyncio
    import sys
    import types as _types

    import flash.providers.runpod.auth as auth
    import flash.providers.runpod.train.endpoints as ep_mod
    from flash.providers.base import canonical_gpu

    run_id = "flash-1-abcd1234"
    friendly = canonical_gpu("RTX 5090")
    target = endpoint_name(friendly, _run_suffix(run_id))
    resource_name = f"live-{target}"

    monkeypatch.setattr(auth, "ensure_auth", lambda: None)
    monkeypatch.setattr(ep_mod, "isolate_flash_state", lambda _: None)

    async def fake_undeploy(uid, **_):
        return {"success": True, "name": resource_name}

    fake_resource = _types.SimpleNamespace(name=resource_name)
    fake_rm = _types.SimpleNamespace(
        list_all_resources=lambda: {"uid-1": fake_resource},
        undeploy_resource=fake_undeploy,
    )

    fake_rm_mod = _types.ModuleType("runpod_flash.core.resources.resource_manager")
    fake_rm_mod.ResourceManager = lambda: fake_rm

    for mod_name in (
        "runpod_flash",
        "runpod_flash.core",
        "runpod_flash.core.resources",
        "runpod_flash.core.resources.resource_manager",
    ):
        if mod_name not in sys.modules:
            stub = _types.ModuleType(mod_name)
            stub.__path__ = []  # mark as package so sub-imports don't raise
            monkeypatch.setitem(sys.modules, mod_name, stub)
    monkeypatch.setitem(
        sys.modules, "runpod_flash.core.resources.resource_manager", fake_rm_mod
    )

    async def _call():
        return ftrain.terminate_endpoint("RTX 5090", run_id)

    result = asyncio.run(_call())
    assert isinstance(result, list)
    assert result == [{"success": True, "name": resource_name}]


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


def test_attach_run_recovery_skips_training_when_raced_terminal(tmp_path, monkeypatch):
    """Recovery-path TOCTOU regression (the reviewed bug). attach_run checks the terminal state
    ONCE up front, then runs a long poll. If a concurrent thread/process flips the run terminal
    (e.g. another attach_run marks it `failed`) DURING that poll, the not-ok recovery path must
    NOT resume `_run_training` — doing so would submit PAID GPU work for an already-terminal run.
    The atomic guard: _update(.., "running", ..)'s sticky CAS rejects the write (returns False),
    so the resume is skipped. Here we flip the run to `failed` from inside the (mocked) poll, then
    return a not-ok result, and assert training is never resumed."""
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-race-terminal"})
    st = orch.RunStatus(
        run_id=spec.run_id,
        state="running",
        spec=spec.to_dict(),
        remote={
            "provider": "runpod",
            "endpoint_id": "ep-1",
            "endpoint_name": "n",
            "job_id": "job-1",
            "attempt": 0,
        },
    )
    orch._save_status(st)

    # _run_training is the PAID-work entry point; it must never be called for a terminal run.
    training_calls = {"n": 0}
    monkeypatch.setattr(
        orch,
        "_run_training",
        lambda *a, **k: training_calls.__setitem__("n", training_calls["n"] + 1),
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
    assert training_calls["n"] == 0, (
        "must NOT submit paid work (resume training) for a run raced to terminal"
    )
    assert out.state == "failed", "the authoritative terminal state must be preserved"
    assert orch.get_status(spec.run_id).state == "failed"


def test_attach_run_recovery_resumes_training_when_still_active(tmp_path, monkeypatch):
    """Happy-path guard: the TOCTOU fix must NOT regress a genuine recovery. A not-ok poll on a
    run that is STILL active (no terminal race) must resume `_run_training` exactly as before."""
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-recover-active"})
    st = orch.RunStatus(
        run_id=spec.run_id,
        state="running",
        spec=spec.to_dict(),
        remote={
            "provider": "runpod",
            "endpoint_id": "ep-1",
            "endpoint_name": "n",
            "job_id": "job-1",
            "attempt": 0,
        },
    )
    orch._save_status(st)

    training_calls = {"n": 0}
    monkeypatch.setattr(
        orch,
        "_run_training",
        lambda *a, **k: training_calls.__setitem__("n", training_calls["n"] + 1),
    )
    monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo, *, code_prefix: repo)

    from flash.providers.base import PollResult

    _make_poll_provider(
        monkeypatch,
        on_poll=lambda h, s, seed: PollResult(False, failure="stalled", detail="redeploy"),
    )

    out = orch.attach_run(spec.run_id)
    assert training_calls["n"] == 1, "a still-active run must resume training (no regression)"
    assert out.state == "running"


def test_run_training_bails_on_terminal_before_paid_work(tmp_path, monkeypatch):
    """Defense in depth: _run_training's own pre-submit guard bails on ANY terminal state
    (not just `cancelled`). If the run is terminal when training is entered — e.g. a concurrent
    thread marked it `done`/`failed` after the caller decided to resume — it must raise
    _RunCancelled and never call _submit_seed_supervised (the paid GPU submit)."""
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-loop-terminal"})
    # The run is already terminal (failed) before training runs.
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
        orch._run_training(spec, io.StringIO(), prior_cost=0.0)
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


# ---------------------------------------------------------------------------
# Quota-slot release in terminate_endpoint: release ONLY when the remote endpoint is
# provably gone — never on an undeploy failure (would oversubscribe RunPod's quota), but
# DO release when we positively verify nothing exists (else the slot leaks → queue deadlock).
# ---------------------------------------------------------------------------
def target_for(run_id):
    """The endpoint name terminate_endpoint reconstructs for ``run_id`` on RTX 5090."""
    from flash.providers.base import canonical_gpu

    return endpoint_name(canonical_gpu("RTX 5090"), _run_suffix(run_id))


def _install_fake_sdk(monkeypatch, *, resources, undeploy, rest_find, rest_delete=lambda _id: True):
    """Stub the runpod_flash ResourceManager + the RunPod REST API for terminate_endpoint.

    ``resources`` is the registry dict returned by list_all_resources(); ``undeploy`` is an
    async fn called per uid; ``rest_find``/``rest_delete`` back the registry-less REST fallback
    (``rest_find`` may raise to simulate an unreachable API). Also drops the quota semaphore +
    tracking set to a fresh pair (monkeypatch restores them) so the test owns one acquired slot.
    Returns ``(ep_mod, target)``.
    """
    import sys
    import threading
    import types as _types

    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.auth as auth
    import flash.providers.runpod.train.endpoints as ep_mod
    from flash.providers.base import canonical_gpu

    monkeypatch.setattr(auth, "ensure_auth", lambda: None)
    monkeypatch.setattr(ep_mod, "isolate_flash_state", lambda *a, **k: None)

    fake_rm = _types.SimpleNamespace(
        list_all_resources=lambda: resources,
        undeploy_resource=undeploy,
    )
    fake_rm_mod = _types.ModuleType("runpod_flash.core.resources.resource_manager")
    fake_rm_mod.ResourceManager = lambda: fake_rm
    for mod_name in (
        "runpod_flash",
        "runpod_flash.core",
        "runpod_flash.core.resources",
        "runpod_flash.core.resources.resource_manager",
    ):
        if mod_name not in sys.modules:
            stub = _types.ModuleType(mod_name)
            stub.__path__ = []
            monkeypatch.setitem(sys.modules, mod_name, stub)
    monkeypatch.setitem(sys.modules, "runpod_flash.core.resources.resource_manager", fake_rm_mod)

    monkeypatch.setattr(runpod_api, "find_endpoints_by_name", rest_find)
    monkeypatch.setattr(runpod_api, "delete_endpoint", rest_delete)

    # Fresh local semaphore + tracking map, with one slot already "acquired" (local mode, as a
    # no-internal-key get_train_endpoint would have done). monkeypatch restores the globals after.
    target = endpoint_name(canonical_gpu("RTX 5090"), _run_suffix("flash-q-1"))
    monkeypatch.setattr(ep_mod, "_LOCAL_SLOTS", threading.Semaphore(28))
    monkeypatch.setattr(ep_mod, "_ACQUIRED", {})
    ep_mod._ACQUIRED[target] = "local"
    ep_mod._LOCAL_SLOTS.acquire()  # 27 free; releasing puts it back to 28
    return ep_mod, target


async def _undeploy_ok(uid, **_):
    return {"success": True}


async def _undeploy_fail(uid, **_):
    raise RuntimeError("undeploy boom")


def test_terminate_releases_slot_when_undeploy_succeeds(monkeypatch):
    # (a) at least one undeploy succeeded → the endpoint is gone → release the slot.
    ep_mod, target = _install_fake_sdk(
        monkeypatch,
        resources={"u1": types.SimpleNamespace(name=f"live-{target_for('flash-q-1')}")},
        undeploy=_undeploy_ok,
        rest_find=lambda _s: [],
    )
    ftrain.terminate_endpoint("RTX 5090", "flash-q-1")
    assert target not in ep_mod._ACQUIRED, "a successful undeploy must release the slot"


def test_terminate_does_not_release_slot_on_undeploy_failure(monkeypatch):
    # The endpoint exists (a uid was found) but undeploy FAILED — it may still be alive and
    # counting against the RunPod quota. Releasing here would oversubscribe the quota, so the
    # slot MUST stay held until a later teardown confirms the endpoint is actually gone.
    ep_mod, target = _install_fake_sdk(
        monkeypatch,
        resources={"u1": types.SimpleNamespace(name=f"live-{target_for('flash-q-1')}")},
        undeploy=_undeploy_fail,
        rest_find=lambda _s: [],  # not consulted: uids was non-empty
    )
    ftrain.terminate_endpoint("RTX 5090", "flash-q-1")
    assert target in ep_mod._ACQUIRED, "a failed undeploy must NOT release the slot"


def test_terminate_releases_slot_when_no_remote_endpoint_exists(monkeypatch):
    # (b) registry returned no uids AND the REST lookup confirms no endpoint of this name —
    # the endpoint provably does not exist (e.g. it never finished deploying). Release the slot;
    # otherwise it leaks forever and eventually deadlocks the queue.
    ep_mod, target = _install_fake_sdk(
        monkeypatch,
        resources={},  # registry finds nothing
        undeploy=_undeploy_ok,  # never called
        rest_find=lambda _s: [],  # REST confirms nothing remote
    )
    ftrain.terminate_endpoint("RTX 5090", "flash-q-1")
    assert target not in ep_mod._ACQUIRED, (
        "a positively-verified-absent endpoint must release the slot (else the queue deadlocks)"
    )


def test_terminate_does_not_release_slot_when_rest_lookup_unreachable(monkeypatch):
    # No uids in the registry, but the REST lookup RAISES (API unreachable) — we cannot prove
    # the endpoint is gone. Releasing on an unverified absence risks oversubscribing the quota,
    # so the slot stays held.
    def _boom(_s):
        raise RuntimeError("REST API down")

    ep_mod, target = _install_fake_sdk(
        monkeypatch,
        resources={},
        undeploy=_undeploy_ok,
        rest_find=_boom,
    )
    out = ftrain.terminate_endpoint("RTX 5090", "flash-q-1")
    assert target in ep_mod._ACQUIRED, (
        "an unverifiable absence (REST unreachable) must NOT release the slot"
    )
    assert isinstance(out, list)  # still never raises


def test_terminate_releases_slot_when_rest_deletes_orphan(monkeypatch):
    # No uids in the registry, but the REST fallback FINDS and deletes a live orphan endpoint
    # (e.g. the registry entry was lost across a container restart). That delete succeeds →
    # the endpoint is gone → release the slot.
    ep_mod, target = _install_fake_sdk(
        monkeypatch,
        resources={},
        undeploy=_undeploy_ok,
        rest_find=lambda _s: [{"id": "ep-orphan", "name": target_for("flash-q-1")}],
        rest_delete=lambda _id: True,
    )
    ftrain.terminate_endpoint("RTX 5090", "flash-q-1")
    assert target not in ep_mod._ACQUIRED, "a REST-deleted orphan must release the slot"
