"""verify cancellation stops a remote flash worker across processes.

``stop_endpoint`` only sees the current process cache. cancellation must use ``terminate_endpoint`` to
find the persisted runpod resource and delete it through the api before billing continues.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

import flash.providers.runpod.serverless as ftrain
from flash.providers.runpod.serverless import _run_suffix, _select_endpoint_resources, endpoint_name
from tests._helpers.runner import provisioned_status

_RUNPOD_FINGERPRINT = "rpk-0123456789ab"


def _remote(endpoint_id, job_id, attempt):
    return {
        "provider": "runpod",
        "endpoint_id": endpoint_id,
        "endpoint_name": f"flash-{endpoint_id}",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "job_id": job_id,
        "attempt": attempt,
        "started_ts": float(attempt + 1),
    }


def _res(name):
    return types.SimpleNamespace(name=name)


def test_isolate_flash_state_resets_runpod_flash_manager_on_scope_change(tmp_path, monkeypatch):
    import flash.providers.runpod.serverless.endpoints as ep_mod

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
    import flash.providers.runpod.serverless.endpoints as ep_mod

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

    monkeypatch.setattr(ep_mod, "isolate_flash_state", rec_isolate)
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
    assert ("isolate", "run-a") in locked_events

    default_handler = ep_mod.get_train_endpoint("RTX 5090")
    assert default_handler.endpoint.kwargs["name"] == "flash-rtx5090"
    assert {"flash-rtx5090": default_handler} == ep_mod._ENDPOINT_CACHE
    assert ("isolate", None) in locked_events

    cached_handler = ep_mod.get_train_endpoint("RTX 5090")
    assert cached_handler is default_handler


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


@pytest.mark.parametrize("failed_revocation_call", [1, 2])
def test_cancel_run_revocation_failure_defers_until_after_fence_and_teardown(
    tmp_path, monkeypatch, failed_revocation_call
):
    import flash.runner as orch
    from flash.core.spec import JobSpec
    from flash.runner.supervise import lifecycle
    from flash.server.platform import db as server_db

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "gpu": {"type": "RTX 5090"},
            "run_id": f"flash-revoke-failure-{failed_revocation_call}",
        }
    )
    status = provisioned_status(
        orch,
        spec,
        state="running",
        remote=_remote("endpoint-1", "job-1", 1),
    )
    orch._save_status(status)
    revocation_calls = 0
    teardown_calls = []

    def revoke(_run_id):
        nonlocal revocation_calls
        revocation_calls += 1
        if revocation_calls == failed_revocation_call:
            raise RuntimeError(f"revocation failure {failed_revocation_call}")
        return 1

    monkeypatch.setattr(server_db, "revoke_teacher_capabilities_for_run", revoke)

    def teardown(handle, run_id):
        teardown_calls.append((handle.provider, run_id, orch.get_status(run_id).state))
        return True

    monkeypatch.setattr(lifecycle, "_strict_teardown_handle", teardown)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)

    with pytest.raises(RuntimeError, match=f"revocation failure {failed_revocation_call}"):
        orch.cancel_run(spec.run_id)

    persisted = orch.get_status(spec.run_id)
    assert persisted.state == "cancelled"
    assert persisted.remote is None
    assert teardown_calls == [("runpod", spec.run_id, "cancelled")]
    assert revocation_calls == 2


def test_cancel_run_calls_terminate_and_marks_cancelled(tmp_path, monkeypatch):
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "gpu": {"type": "RTX 5090"},
            "run_id": "flash-9-feedface",
        }
    )
    st = provisioned_status(orch, spec, state="running")
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
    from flash.core.spec import JobSpec

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


def test_cancel_undeploys_deployment_that_raced_in_after_entry_snapshot(tmp_path, monkeypatch):
    # Race: cancel_run enters on a non-`deployed` snapshot (state="running"), but a deploy lands during
    # teardown (running -> done -> deployed) before the terminal `cancelled` write. `deployed` is
    # non-terminal so `cancelled` still wins, but the entry-gated undeploy never ran. cancel_run must
    # re-read post-write and tear down the raced-in deployment so it is never orphaned.
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

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
        revision = f"{spec.run_id}@final." + "a" * 40
        orch.mark_deployed(
            spec.run_id,
            {"state": "ready", "gpu": "RTX 5090", "adapter_revision": revision},
            verification_generation=orch.verified_adapter_revision_generation(spec.run_id),
        )

    monkeypatch.setattr(orch, "_gc_run_endpoints", gc_then_deploy)

    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert undeployed == [spec.run_id], "the raced-in deployment must be torn down, not orphaned"
    assert (out.deployment or {}).get("state") == "undeployed"


def test_cancel_deployed_run_undeploy_goes_through_lock_guarded_path(tmp_path, monkeypatch):
    # Regression: the deployed branch used a bare _save_status OUTSIDE _STATUS_LOCK, which
    # persisted a stale pre-teardown snapshot and bypassed serialization. It must instead
    # mark the deployment inactive through the lock-guarded mark_deployment_undeployed
    # helper, and that write must happen while _STATUS_LOCK is held.
    import inspect

    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

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
    assert "with _status_guard(run_id)" in inspect.getsource(orch.mark_deployment_undeployed)

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
    from flash.core.spec import JobSpec

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
    from flash.core.spec import JobSpec

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
    # if a running job genuinely finishes while cancellation tears it down, preserve its done metrics
    # and artifacts. only runs deployed at cancellation entry allow the terminal override; a blanket
    # override would clobber real training results.
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

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


def test_terminate_endpoint_from_async_context_does_not_raise(monkeypatch):
    """verify terminate_endpoint works inside an active event loop.

    it must move ``asyncio.run`` to another thread because calling it directly from fastapi/uvicorn or
    any async context raises RuntimeError.
    """
    import asyncio
    import sys
    import types as _types

    import flash.providers.runpod.auth as auth
    import flash.providers.runpod.serverless.endpoints as ep_mod
    from flash.providers.base import canonical_gpu

    run_id = "flash-1-abcd1234"
    friendly = canonical_gpu("RTX 5090")
    target = endpoint_name(friendly, _run_suffix(run_id))
    resource_name = f"live-{target}"

    monkeypatch.setattr(auth, "ensure_auth", lambda: None)
    monkeypatch.setattr(ep_mod, "isolate_flash_state", lambda _: None)

    # The registry-less REST sweep runs after the undeploy and would report every configured
    # account as unreachable (offline suite), appending a failure row this assertion would then
    # have to spell out. Stub it clear: this test is about the event loop, not the sweep.
    import flash.providers.runpod.api as runpod_api

    monkeypatch.setattr(runpod_api, "list_endpoints_by_key", lambda **_: ({}, []))

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
    monkeypatch.setitem(sys.modules, "runpod_flash.core.resources.resource_manager", fake_rm_mod)

    async def _call():
        return ftrain.terminate_endpoint("RTX 5090", run_id)

    result = asyncio.run(_call())
    assert isinstance(result, list)
    assert result == [{"success": True, "name": resource_name}]


def test_cancel_run_noop_when_terminal(tmp_path, monkeypatch):
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-done-1"})
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="done", spec=spec.to_dict()))

    called = {"v": False}
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: called.__setitem__("v", True))
    out = orch.cancel_run(spec.run_id)
    assert out.state == "done"
    assert called["v"] is False, "must not tear down endpoints for an already-terminal run"


def test_cancel_run_retries_durable_cleanup_for_cancelled_run(tmp_path, monkeypatch):
    import flash.providers as providers
    import flash.runner as orch
    from flash.core.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancelled-1"})
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="cancelled", spec=spec.to_dict()))
    remote = _remote("endpoint-cleanup", "job-cleanup", 1)
    assert orch._preserve_cleanup_remote(spec.run_id, remote) is True
    events = []

    class Provider:
        def cancel(self, handle):
            events.append(("cancel", handle.to_dict()["job_id"]))

        def destroy(self, handle):
            events.append(("destroy", handle.to_dict()["endpoint_id"]))

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())

    out = orch.cancel_run(spec.run_id)

    assert out.state == "cancelled"
    assert events == [("cancel", "job-cleanup"), ("destroy", "endpoint-cleanup")]
    assert orch._CLEANUP_REMOTES_KEY not in orch._load_status_json(spec.run_id)


def test_cancel_run_accepts_confirmed_endpoint_delete_after_cancel_ack_failure(
    tmp_path, monkeypatch
):
    import flash.providers as providers
    import flash.runner as orch
    from flash.core.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancel-retry"})
    remote = {**_remote("endpoint-exact", "job-exact", 7), "seed": 42}
    orch._save_status(provisioned_status(orch, spec, state="running", remote=remote))
    events = []

    class Provider:
        def cancel(self, handle):
            data = handle.to_dict()
            events.append(("cancel", data["endpoint_id"], data["job_id"], data["attempt"]))
            raise RuntimeError("cancellation acknowledgement failed")

        def destroy(self, handle):
            data = handle.to_dict()
            events.append(("destroy", data["endpoint_id"], data.get("job_id"), data["attempt"]))

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    gc_calls = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda value: gc_calls.append(value.run_id))

    result = orch.cancel_run(spec.run_id)
    raw = orch._load_status_json(spec.run_id)

    assert result.state == "cancelled"
    assert raw["remote"] is None
    assert orch._CLEANUP_REMOTES_KEY not in raw
    assert gc_calls == [spec.run_id]
    assert events == [
        ("cancel", "endpoint-exact", "job-exact", 7),
        ("destroy", "endpoint-exact", "job-exact", 7),
    ]


def test_cancel_run_failed_teardown_does_not_replace_racing_public_remote(tmp_path, monkeypatch):
    import flash.providers as providers
    import flash.runner as orch
    from flash.core.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancel-race"})
    original_remote = _remote("endpoint-original", "job-original", 2)
    replacement_remote = _remote("endpoint-replacement", "job-replacement", 3)
    orch._save_status(
        orch.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            remote=original_remote,
        )
    )

    class Provider:
        def cancel(self, _handle):
            current = orch.get_status(spec.run_id)
            current.remote = replacement_remote
            orch._save_status(current)
            raise RuntimeError("cancellation acknowledgement failed")

        def destroy(self, _handle):
            raise RuntimeError("endpoint deletion failed")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)

    out = orch.cancel_run(spec.run_id)
    raw = orch._load_status_json(spec.run_id)

    assert out.state == "cancelled"
    assert raw["remote"] == replacement_remote
    assert raw[orch._CLEANUP_REMOTES_KEY] == [original_remote, replacement_remote]


def test_cancel_run_marks_billing_failed_when_pricing_falls_back(tmp_path, monkeypatch):
    import flash.runner as orch
    from flash.core.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancel-price"})
    orch._save_status(
        orch.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            billing_context={"org_id": "org-a"},
        )
    )
    monkeypatch.setattr(orch, "actual_steps_run", lambda _status: 1)
    monkeypatch.setattr(orch, "charge_usd_for_spec", lambda *a, **kw: kw["fallback"])
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)

    status = orch.cancel_run(spec.run_id)

    assert status.state == "cancelled"
    assert status.cost_usd == 0.0
    assert status.billing_state == "failed"
    assert "pricing failed" in (status.billing_error or "")


def test_cancel_run_bills_a_profile_on_started_not_on_optimizer_steps(tmp_path, monkeypatch):
    """charge cancelled profiles only when they started, using the persisted quote.

    profiles emit no training-step heartbeat, so ``actual_steps_run`` would incorrectly make all
    cancellations free. started profiles must use the submitted quote, matching successful settlement;
    never-started profiles cost zero.

    assert ``billing_state`` because pricing fails closed to cost 0.0 on exceptions, which otherwise
    looks identical to the expected never-started charge.
    """
    import flash.runner as orch
    from flash.core.spec import JobSpec
    from flash.engine.profiling.workload_profile import SFT_PROFILE_KIND

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)

    def _cancel(run_id: str, heartbeat: dict | None, quote: float | None) -> tuple:
        spec = JobSpec.from_dict(
            {
                "gpu": {"type": "RTX 5090"},
                "run_id": run_id,
                "workload_profile_kind": SFT_PROFILE_KIND,
            }
        )
        # built the way submit_job builds it: to_dict() strips workload_profile_kind as
        # platform-managed, so the private snapshot (to_internal_dict, which retains it) is the
        # only carrier of the kind into the rebuilt effective spec.
        orch._save_status(
            orch.RunStatus(
                run_id=run_id,
                state="running",
                spec=spec.to_dict(),
                billing_context={"org_id": "org-a"},
                last_heartbeat=heartbeat,
                workload_profile_kind=SFT_PROFILE_KIND,
                estimated_cost_usd=quote,
                effective_preparation={
                    "worker_spec": spec.to_internal_dict(),
                    "workload_profile": spec.workload_profile or None,
                    "adapter_identity": None,
                    "preparation_digest": orch._preparation_digest(
                        JobSpec.from_dict(spec.to_dict()), spec, None
                    ),
                    "backend": orch.TRAINER_BACKEND,
                },
            )
        )
        assert orch.cancel_run(run_id).state == "cancelled"
        final = orch.get_status(run_id)
        return final.cost_usd, getattr(final, "billing_state", None)

    # asserted on the CHARGE, not on the internal steps kwarg: the charge is the contract, and
    # pinning the call shape made this test fail on a change that preserved every billed amount.
    #
    # never started: no heartbeat at all -> nothing was rented -> $0, and priced successfully.
    assert _cancel("profile-sft-never", None, 7.0) == (0.0, None)
    # started: the profile worker's own first heartbeat, which is NOT a training stage. it owes the
    # bounded wall it rented, priced at the quote it was submitted under -- not a fresh offline
    # re-derivation, which could differ from both the number the user was shown and the number the
    # same profile would bill on success.
    assert _cancel("profile-sft-started", {"stage": "profile_start", "ts": 1.0}, 7.0) == (7.0, None)
    # no persisted quote: it still must price, falling back to the spec estimate rather than
    # collapsing to the swallowed-failure $0.
    charged, billing_state = _cancel(
        "profile-sft-noquote", {"stage": "profile_start", "ts": 1.0}, None
    )
    assert billing_state is None, f"pricing must not fail closed to $0; got {billing_state!r}"
    assert charged > 0.0, f"a started profile without a stored quote must still bill; got {charged}"


def test_cancel_run_successful_exact_teardown_leaves_no_cleanup_remote(tmp_path, monkeypatch):
    import flash.providers as providers
    import flash.runner as orch
    from flash.core.spec import JobSpec

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancel-clean"})
    remote = _remote("endpoint-clean", "job-clean", 3)
    orch._save_status(
        orch.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            remote=remote,
        )
    )
    events = []

    class Provider:
        def cancel(self, handle):
            events.append(("cancel", handle.to_dict()["job_id"]))

        def destroy(self, handle):
            events.append(("destroy", handle.to_dict()["endpoint_id"]))

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)

    out = orch.cancel_run(spec.run_id)

    assert out.state == "cancelled"
    assert events == [("cancel", "job-clean"), ("destroy", "endpoint-clean")]
    assert orch._CLEANUP_REMOTES_KEY not in orch._load_status_json(spec.run_id)


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
        def poll(self, handle, spec, seed, *, log=None, _deadline_at=None):
            return on_poll(handle, spec, seed)

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            return None

    monkeypatch.setattr(providers, "get_provider", lambda name: _StubProvider())


def test_attach_run_recovery_skips_training_when_raced_terminal(tmp_path, monkeypatch):
    """verify recovery cannot restart paid work after a concurrent terminal transition.

    flip the run to failed during polling; the sticky ``_update(..., "running", ...)`` cas must reject
    resume so ``_run_training`` is never called.
    """
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-race-terminal"})
    st = orch.RunStatus(
        run_id=spec.run_id,
        state="running",
        spec=spec.to_dict(),
        remote=_remote("ep-1", "job-1", 0),
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
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-recover-active"})
    st = provisioned_status(orch, spec, state="running", remote=_remote("ep-1", "job-1", 0))
    orch._save_status(st)

    training_calls = {"n": 0}
    monkeypatch.setattr(
        orch,
        "_run_training",
        lambda *a, **k: training_calls.__setitem__("n", training_calls["n"] + 1),
    )
    monkeypatch.setattr(
        "flash.providers._lifecycle.worker.upload_code", lambda repo, *, code_prefix: repo
    )

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
    from flash.core.spec import JobSpec

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
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-update-ret"})
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()))
    assert orch._update(spec.run_id, "running", cost_usd=1.0) is True
    assert orch._update(spec.run_id, "failed", error="boom") is True  # terminal write applies
    # Now terminal: a non-terminal transition is rejected and reported False.
    assert orch._update(spec.run_id, "running") is False
    assert orch.get_status(spec.run_id).state == "failed"


def _run_spec(run_id: str):
    from flash.core.spec import JobSpec

    return JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})


def _checkpoint_revision(run_id: str, step: int, sha: str = "a") -> str:
    return f"{run_id}@step-{step}." + sha * 40


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


def test_cancel_tears_down_training_before_checkpoint_serving_decision(tmp_path, monkeypatch):
    import flash.providers as providers
    import flash.runner as orch
    import flash.runner.results.verified_revisions as verified_revisions
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-order"
    remote = _remote("endpoint-training", "training-job", 0)
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


def test_cancel_preserves_ready_verified_same_step_checkpoint(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-preserve"
    deployment = _ready_checkpoint(orch, run_id, 80)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "adapter_alias_target",
        lambda _target: pytest.fail("non-contended verified cancellation must not read the alias"),
    )
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


@pytest.mark.parametrize("state", ["queued", "smoke_testing"])
def test_cancel_preserves_busy_attempt_previous_checkpoint_without_alias_lookup(
    tmp_path, monkeypatch, state
):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = f"flash-checkpoint-{state}"
    previous = _ready_checkpoint(orch, run_id, 40)
    status = orch.get_status(run_id)
    status.deployment = {
        "state": state,
        "requested_at": 123.0,
        "previous_deployment": previous,
    }
    orch._save_status(status)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "adapter_alias_target",
        lambda _run_id: pytest.fail("ordinary cancellation must not read the serving alias"),
    )
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _run_id: pytest.fail("the verified previous checkpoint must remain serving"),
    )

    out = orch.cancel_run(run_id)

    assert out.state == "cancelled"
    assert out.deployment == previous
    assert orch.read_verified_adapter_revisions(run_id) == frozenset({previous["adapter_revision"]})


def test_cancel_contended_deploy_fences_previous_checkpoint_before_wait(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    import flash.server.platform.locks as locks

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-contended-fence"
    previous = _ready_checkpoint(orch, run_id, 40)
    status = orch.get_status(run_id)
    status.state = "deployed"
    attempted = {
        "state": "smoke_testing",
        "requested_at": 456.0,
        "adapter_revision": f"{run_id}@final." + "b" * 40,
        "previous_deployment": previous,
        "verification_generation": orch.verified_adapter_revision_generation(run_id),
    }
    status.deployment = attempted
    orch._save_status(status)
    stale_commit = {**attempted, "state": "ready"}
    lock_events = []

    class ContendedLock:
        held = False

        def acquire(self, blocking: bool = True) -> bool:
            if not blocking:
                lock_events.append("contended")
                return False
            lock_events.append("waiting")
            fenced = orch.get_status(run_id)
            assert fenced.deployment == previous
            assert orch.verified_adapter_revision_generation(run_id) == (
                attempted["verification_generation"] + 1
            )
            stale_pending = orch.mark_deployment_pending(
                run_id,
                {**attempted, "state": "reconciling"},
                owner_deployment=attempted,
            )
            assert stale_pending.deployment == previous
            stale = orch.mark_deployed(
                run_id,
                stale_commit,
                expect_state="deployed",
                verification_generation=attempted["verification_generation"],
            )
            assert stale.deployment == previous
            self.held = True
            return True

        def release(self) -> None:
            assert self.held is True
            self.held = False
            lock_events.append("released")

    alias_reads = []
    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: ContendedLock())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "adapter_alias_target",
        lambda target: alias_reads.append(target) or previous["adapter_revision"],
    )
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("the verified predecessor must remain serving"),
    )

    out = orch.cancel_run(run_id)

    assert alias_reads == [run_id]
    assert lock_events == ["contended", "waiting", "released"]
    assert out.state == "cancelled"
    assert out.deployment == previous
    assert orch.read_verified_adapter_revisions(run_id) == frozenset({previous["adapter_revision"]})


def test_cancel_contended_unknown_activation_is_fenced_before_wait(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    import flash.server.platform.locks as locks

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-contended-unknown"
    previous = _ready_checkpoint(orch, run_id, 40)
    generation = orch.verified_adapter_revision_generation(run_id)
    attempted = {
        "state": "reconciling",
        "requested_at": 456.0,
        "adapter_revision": _checkpoint_revision(run_id, 80, "b"),
        "checkpoint_step": 80,
        "activation_outcome_unknown": True,
        "previous_deployment": previous,
        "verification_generation": generation,
    }
    status = orch.get_status(run_id)
    status.deployment = attempted
    orch._save_status(status)

    class ContendedLock:
        held = False

        def acquire(self, blocking: bool = True) -> bool:
            if not blocking:
                return False
            assert orch.verified_adapter_revision_generation(run_id) == generation + 1
            assert orch.get_status(run_id).deployment == previous
            self.held = True
            return True

        def release(self) -> None:
            assert self.held is True
            self.held = False

    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: ContendedLock())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy, "adapter_alias_target", lambda _target: previous["adapter_revision"]
    )
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("the safely recommitted predecessor must remain serving"),
    )

    out = orch.cancel_run(run_id)

    assert out.state == "cancelled"
    assert out.deployment == previous
    assert orch.read_verified_adapter_revisions(run_id) == frozenset({previous["adapter_revision"]})


@pytest.mark.parametrize("restore_failure", ["miss", "raise"])
def test_cancel_contended_predecessor_recommit_failure_stays_fenced_and_revokes(
    tmp_path, monkeypatch, restore_failure
):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    import flash.server.platform.locks as locks

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-contended-restore-miss"
    previous = _ready_checkpoint(orch, run_id, 40)
    attempted = {
        "state": "smoke_testing",
        "requested_at": 456.0,
        "adapter_revision": _checkpoint_revision(run_id, 80, "b"),
        "checkpoint_step": 80,
        "previous_deployment": previous,
        "verification_generation": orch.verified_adapter_revision_generation(run_id),
    }
    status = orch.get_status(run_id)
    status.deployment = attempted
    orch._save_status(status)
    real_mark_checkpoint_deployed = orch.mark_checkpoint_deployed

    def fail_predecessor_restore(*args, **kwargs):
        owner = kwargs.get("owner_deployment")
        if isinstance(owner, dict) and owner.get("state") == "revocation_failed":
            if restore_failure == "raise":
                real_mark_checkpoint_deployed(*args, **kwargs)
                raise OSError("checkpoint restoration acknowledgement lost")
            return orch.get_status(run_id)
        return real_mark_checkpoint_deployed(*args, **kwargs)

    class ContendedLock:
        held = False

        def acquire(self, blocking: bool = True) -> bool:
            if not blocking:
                return False
            fenced = orch.get_status(run_id)
            assert fenced.deployment["state"] == "revocation_failed"
            assert orch.read_verified_adapter_revisions(run_id) == frozenset()
            self.held = True
            return True

        def release(self) -> None:
            assert self.held is True
            self.held = False

    undeploys = []
    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: ContendedLock())
    monkeypatch.setattr(orch, "mark_checkpoint_deployed", fail_predecessor_restore)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy, "adapter_alias_target", lambda _target: previous["adapter_revision"]
    )
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeploys.append(target))

    out = orch.cancel_run(run_id)

    assert undeploys == [run_id]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


@pytest.mark.parametrize("attempted_step", [None, 80])
def test_cancel_contended_fence_revokes_when_alias_changes_before_lock_release(
    tmp_path, monkeypatch, attempted_step
):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    import flash.server.platform.locks as locks

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = f"flash-checkpoint-contended-alias-race-{attempted_step}"
    previous = _ready_checkpoint(orch, run_id, 40)
    status = orch.get_status(run_id)
    status.state = "deployed"
    attempted_revision = (
        f"{run_id}@final." + "b" * 40
        if attempted_step is None
        else _checkpoint_revision(run_id, attempted_step, "b")
    )
    attempted = {
        "state": "smoke_testing",
        "requested_at": 456.0,
        "adapter_revision": attempted_revision,
        "previous_deployment": previous,
        "verification_generation": orch.verified_adapter_revision_generation(run_id),
    }
    if attempted_step is not None:
        attempted["checkpoint_step"] = attempted_step
    status.deployment = attempted
    orch._save_status(status)
    stale_commit = {**attempted, "state": "ready"}
    alias_target = [previous["adapter_revision"]]

    class ContendedLock:
        held = False

        def acquire(self, blocking: bool = True) -> bool:
            if not blocking:
                return False
            fenced = orch.get_status(run_id)
            assert fenced.deployment == previous
            assert orch.verified_adapter_revision_generation(run_id) == (
                attempted["verification_generation"] + 1
            )
            alias_target[0] = attempted["adapter_revision"]
            if attempted_step is None:
                stale = orch.mark_deployed(
                    run_id,
                    stale_commit,
                    expect_state="deployed",
                    verification_generation=attempted["verification_generation"],
                )
            else:
                stale = orch.mark_checkpoint_deployed(
                    run_id,
                    stale_commit,
                    verification_generation=attempted["verification_generation"],
                    owner_deployment=attempted,
                )
            assert stale.deployment == previous
            self.held = True
            return True

        def release(self) -> None:
            assert self.held is True
            self.held = False

    undeploys = []
    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: ContendedLock())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "adapter_alias_target", lambda _target: alias_target[0])

    def undeploy(target):
        undeploys.append(target)
        assert orch.get_status(run_id).deployment["state"] == "revocation_failed"

    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)

    out = orch.cancel_run(run_id)

    assert undeploys == [run_id]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


def test_cancel_unknown_outcome_restores_live_verified_previous_checkpoint(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-unknown-previous"
    previous = _ready_checkpoint(orch, run_id, 40)
    busy = {
        "state": "reconciling",
        "requested_at": 456.0,
        "adapter_revision": f"{run_id}@final." + "b" * 40,
        "activation_outcome_unknown": True,
        "previous_deployment": previous,
    }
    status = orch.get_status(run_id)
    status.deployment = busy
    orch._save_status(status)
    alias_reads = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "adapter_alias_target",
        lambda target: alias_reads.append(target) or previous["adapter_revision"],
    )
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _run_id: pytest.fail("the authoritative previous checkpoint must remain serving"),
    )

    out = orch.cancel_run(run_id)

    assert alias_reads == [run_id]
    assert out.state == "cancelled"
    assert out.deployment == previous
    assert out.deployment != busy
    assert orch.read_verified_adapter_revisions(run_id) == frozenset({previous["adapter_revision"]})


@pytest.mark.parametrize(
    ("raced_state", "expected_state"),
    [("done", "cancelled"), ("failed", "failed")],
)
def test_cancel_checkpoint_restore_survives_owned_run_state_race(
    tmp_path, monkeypatch, raced_state, expected_state
):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = f"flash-checkpoint-state-race-{raced_state}"
    previous = _ready_checkpoint(orch, run_id, 40)
    busy = {
        "state": "reconciling",
        "requested_at": 456.0,
        "adapter_revision": f"{run_id}@final." + "b" * 40,
        "activation_outcome_unknown": True,
        "previous_deployment": previous,
    }
    status = orch.get_status(run_id)
    status.deployment = busy
    orch._save_status(status)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy, "adapter_alias_target", lambda _run_id: previous["adapter_revision"]
    )
    real_mark_checkpoint_deployed = orch.mark_checkpoint_deployed

    def race_run_state(*args, **kwargs):
        if kwargs.get("retain_only_revision"):
            assert kwargs["owner_deployment"] == previous
            return real_mark_checkpoint_deployed(*args, **kwargs)
        assert kwargs["owner_deployment"] == busy
        assert "expect_state" not in kwargs
        raced = orch.get_status(run_id)
        assert raced.deployment == busy
        raced.state = raced_state
        orch._save_status(raced)
        return real_mark_checkpoint_deployed(*args, **kwargs)

    monkeypatch.setattr(orch, "mark_checkpoint_deployed", race_run_state)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _run_id: pytest.fail("the still-owned checkpoint must remain serving"),
    )

    out = orch.cancel_run(run_id)

    assert out.state == expected_state
    assert out.deployment == previous
    assert orch.read_verified_adapter_revisions(run_id) == frozenset({previous["adapter_revision"]})


def test_checkpoint_restore_owner_fence_rejects_newer_attempt(tmp_path, monkeypatch):
    import flash.runner as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-stale-restore"
    previous = _ready_checkpoint(orch, run_id, 40)
    stale_owner = {
        "state": "reconciling",
        "requested_at": 100.0,
        "activation_outcome_unknown": True,
        "previous_deployment": previous,
    }
    newer_attempt = {
        "state": "queued",
        "requested_at": 200.0,
        "previous_deployment": previous,
    }
    status = orch.get_status(run_id)
    status.deployment = newer_attempt
    orch._save_status(status)

    out = orch.mark_checkpoint_deployed(
        run_id,
        previous,
        owner_deployment=stale_owner,
        verification_generation=orch.verified_adapter_revision_generation(run_id),
    )

    assert out.deployment == newer_attempt
    assert orch.get_status(run_id).deployment == newer_attempt


def test_cancel_restore_failure_revokes_instead_of_leaving_reconciling_authority(
    tmp_path, monkeypatch
):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-restore-failure"
    previous = _ready_checkpoint(orch, run_id, 40)
    busy = {
        "state": "reconciling",
        "requested_at": 456.0,
        "adapter_revision": f"{run_id}@final." + "b" * 40,
        "activation_outcome_unknown": True,
        "previous_deployment": previous,
    }
    status = orch.get_status(run_id)
    status.deployment = busy
    orch._save_status(status)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy, "adapter_alias_target", lambda _run_id: previous["adapter_revision"]
    )

    def fail_restore(*_args, **kwargs):
        assert kwargs["owner_deployment"] == busy
        raise OSError("checkpoint status store unavailable")

    monkeypatch.setattr(orch, "mark_checkpoint_deployed", fail_restore)
    undeploys = []

    def undeploy(target):
        undeploys.append(target)
        fenced = orch.get_status(run_id)
        assert fenced.deployment["state"] == "revocation_failed"
        assert fenced.deployment != busy
        assert orch.read_verified_adapter_revisions(run_id) == frozenset()

    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)

    out = orch.cancel_run(run_id)

    assert undeploys == [run_id]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"
    assert out.deployment != previous
    assert out.deployment != busy
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


def test_cancel_restore_ack_failure_preserves_persisted_verified_checkpoint(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-restore-ack-failure"
    previous = _ready_checkpoint(orch, run_id, 40)
    busy = {
        "state": "reconciling",
        "requested_at": 789.0,
        "adapter_revision": f"{run_id}@final." + "c" * 40,
        "activation_outcome_unknown": True,
        "previous_deployment": previous,
    }
    status = orch.get_status(run_id)
    status.deployment = busy
    orch._save_status(status)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy, "adapter_alias_target", lambda _run_id: previous["adapter_revision"]
    )
    real_mark_checkpoint_deployed = orch.mark_checkpoint_deployed

    def persist_then_raise(*args, **kwargs):
        if kwargs.get("retain_only_revision"):
            assert kwargs["owner_deployment"] == previous
            return real_mark_checkpoint_deployed(*args, **kwargs)
        assert kwargs["owner_deployment"] == busy
        real_mark_checkpoint_deployed(*args, **kwargs)
        raise OSError("checkpoint status write acknowledgement lost")

    monkeypatch.setattr(orch, "mark_checkpoint_deployed", persist_then_raise)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _run_id: pytest.fail("the authoritative restored checkpoint must remain serving"),
    )

    out = orch.cancel_run(run_id)

    assert out.state == "cancelled"
    assert out.deployment == previous
    assert out.deployment != busy
    assert orch.read_verified_adapter_revisions(run_id) == frozenset({previous["adapter_revision"]})


@pytest.mark.parametrize("live_verified", [False, True])
def test_cancel_unknown_outcome_revokes_attempted_live_checkpoint(
    tmp_path, monkeypatch, live_verified
):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = f"flash-checkpoint-live-{live_verified}"
    stale_previous = _ready_checkpoint(orch, run_id, 10)
    live_revision = _checkpoint_revision(run_id, 20, "c")
    if live_verified:
        orch.add_verified_adapter_revision(
            run_id,
            live_revision,
            expected_generation=orch.verified_adapter_revision_generation(run_id),
        )
    status = orch.get_status(run_id)
    status.deployment = {
        "state": "reconciling",
        "requested_at": 789.0,
        "adapter_revision": live_revision,
        "checkpoint_step": 20,
        "activation_outcome_unknown": True,
        "previous_deployment": stale_previous,
        "verified_at": 123.0,
        "verify_kind": "fixed_prompt",
        "verify_turns": 1,
        "verify_latency_s": 0.1,
        "verify_finish_reason": "stop",
        "thinking_tag": False,
        "verify_sample": "4",
    }
    orch._save_status(status)
    undeploys = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "adapter_alias_target", lambda _run_id: live_revision)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeploys.append(target))

    out = orch.cancel_run(run_id)

    assert undeploys == [run_id]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


def test_cancel_unknown_outcome_rejects_unverified_divergent_checkpoint(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-live-divergent"
    previous = _ready_checkpoint(orch, run_id, 10)
    attempted_revision = _checkpoint_revision(run_id, 20, "b")
    divergent_revision = _checkpoint_revision(run_id, 30, "c")
    status = orch.get_status(run_id)
    status.deployment = {
        "state": "reconciling",
        "requested_at": 789.0,
        "adapter_revision": attempted_revision,
        "checkpoint_step": 20,
        "activation_outcome_unknown": True,
        "previous_deployment": previous,
    }
    orch._save_status(status)
    undeploys = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "adapter_alias_target", lambda _run_id: divergent_revision)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeploys.append(target))

    out = orch.cancel_run(run_id)

    assert undeploys == [run_id]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


@pytest.mark.parametrize("alias_result", ["missing", "disabled", "error"])
def test_cancel_unknown_outcome_alias_failure_revokes_fail_closed(
    tmp_path, monkeypatch, alias_result
):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = f"flash-checkpoint-alias-{alias_result}"
    previous = _ready_checkpoint(orch, run_id, 30)
    status = orch.get_status(run_id)
    status.deployment = {
        "state": "reconciling",
        "activation_outcome_unknown": True,
        "previous_deployment": previous,
    }
    orch._save_status(status)
    alias_reads = []

    def alias_target(target):
        alias_reads.append(target)
        if alias_result == "error":
            raise deploy.ServingError("alias read failed")

    undeploys = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "adapter_alias_target", alias_target)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeploys.append(target))

    out = orch.cancel_run(run_id)

    assert alias_reads == [run_id]
    assert undeploys == [run_id]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


def test_cancel_unknown_outcome_never_preserves_verified_final_alias(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-final-alias-unknown"
    final_revision = f"{run_id}@final." + "f" * 40
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="done",
            spec=_run_spec(run_id).to_dict(),
            deployment={"state": "reconciling", "activation_outcome_unknown": True},
        )
    )
    orch.add_verified_adapter_revision(
        run_id,
        final_revision,
        expected_generation=orch.verified_adapter_revision_generation(run_id),
    )
    undeploys = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "adapter_alias_target", lambda _run_id: final_revision)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: undeploys.append(target))

    out = orch.cancel_run(run_id)

    assert undeploys == [run_id]
    assert out.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


def test_activation_unknown_final_predecessor_is_not_preservable_checkpoint(tmp_path, monkeypatch):
    import flash.runner as orch
    from flash.runner.supervise.deploy import _preservable_checkpoint_deployment

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-final-predecessor"
    revision = f"{run_id}@final." + "f" * 40
    orch.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=orch.verified_adapter_revision_generation(run_id),
    )

    assert (
        _preservable_checkpoint_deployment(
            run_id,
            {
                "state": "failed",
                "activation_outcome_unknown": True,
                "previous_deployment": {
                    "state": "ready",
                    "adapter_revision": revision,
                    "checkpoint_step": None,
                },
            },
            live_alias_target=revision,
        )
        is None
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


def test_cancel_active_deployment_with_malformed_spec_still_revokes(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    import flash.server.platform.locks as locks

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-malformed-spec-revoke"
    spec = _run_spec(run_id)
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            deployment={"state": "deploying"},
        )
    )
    raw = orch._load_status_json(run_id)
    raw["spec"] = ["malformed-spec"]
    with open(orch.runs_file_path(run_id, ".json"), "w") as file:
        json.dump(raw, file)

    class ContendedLock:
        held = False

        def acquire(self, blocking: bool = True) -> bool:
            if not blocking:
                return False
            self.held = True
            return True

        def release(self) -> None:
            assert self.held is True
            self.held = False

    backend_calls = []
    gc_calls = []
    monkeypatch.setattr(locks, "_deploy_lock", lambda _target: ContendedLock())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: gc_calls.append(_spec))
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: backend_calls.append(target))

    out = orch.cancel_run(run_id)

    assert gc_calls == []
    assert backend_calls == [run_id]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"


def test_cancel_backend_success_local_commit_failure_is_not_backend_uncertainty(
    tmp_path, monkeypatch
):
    import flash.runner as orch
    import flash.serve.deploy as deploy
    from flash.runner.supervise.deploy import DeploymentStatePersistenceError

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-local-persistence-failure"
    spec = _run_spec(run_id)
    revision = f"{run_id}@final." + "d" * 40
    orch._save_status(
        orch.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            deployment={"state": "ready", "adapter_revision": revision},
        )
    )
    orch.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=orch.verified_adapter_revision_generation(run_id),
    )
    backend_calls = []
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: backend_calls.append(target))
    monkeypatch.setattr(
        orch,
        "mark_deployment_undeployed",
        lambda _target: (_ for _ in ()).throw(OSError("status store unavailable")),
    )

    with pytest.raises(DeploymentStatePersistenceError) as excinfo:
        orch.cancel_run(run_id)

    assert not isinstance(excinfo.value, orch.DeploymentRevocationError)
    assert excinfo.value.backend_outcome == "confirmed"
    assert "backend disablement was confirmed" in str(excinfo.value)
    assert backend_calls == [run_id]
    failed = orch.get_status(run_id)
    assert failed.state == "cancelled"
    assert failed.deployment["state"] == "revocation_failed"
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


def test_cancel_preserved_checkpoint_prunes_other_verified_revisions(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-prune"
    preserved = _ready_checkpoint(orch, run_id, 40, remote=None)
    older_revisions = {
        _checkpoint_revision(run_id, 20, "b"),
        f"{run_id}@final." + "c" * 40,
    }
    for revision in older_revisions:
        orch.add_verified_adapter_revision(
            run_id,
            revision,
            expected_generation=orch.verified_adapter_revision_generation(run_id),
        )
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("the preserved checkpoint must remain serving"),
    )

    out = orch.cancel_run(run_id)

    assert out.state == "cancelled"
    assert out.deployment == preserved
    assert orch.read_verified_adapter_revisions(run_id) == frozenset(
        {preserved["adapter_revision"]}
    )


def test_cancel_checkpoint_prune_failure_is_retryable_without_revocation(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.runner.results.verified_revisions as verified_revisions
    import flash.serve.deploy as deploy
    from flash.runner.supervise.deploy import DeploymentStatePersistenceError

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-prune-retry"
    preserved = _ready_checkpoint(orch, run_id, 40, remote=None)
    older_revision = _checkpoint_revision(run_id, 20, "b")
    orch.add_verified_adapter_revision(
        run_id,
        older_revision,
        expected_generation=orch.verified_adapter_revision_generation(run_id),
    )
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("ledger persistence failure must not revoke serving"),
    )
    real_write = verified_revisions._write_unlocked

    def fail_prune(runs_dir, path, generation, revisions):
        if revisions == [preserved["adapter_revision"]]:
            raise OSError("verified revision ledger unavailable")
        return real_write(runs_dir, path, generation, revisions)

    monkeypatch.setattr(verified_revisions, "_write_unlocked", fail_prune)

    with pytest.raises(DeploymentStatePersistenceError) as excinfo:
        orch.cancel_run(run_id)

    assert excinfo.value.backend_outcome == "not_required"
    failed = orch.get_status(run_id)
    assert failed.state == "running"
    assert failed.deployment == preserved
    assert orch.read_verified_adapter_revisions(run_id) == frozenset(
        {older_revision, preserved["adapter_revision"]}
    )

    monkeypatch.setattr(verified_revisions, "_write_unlocked", real_write)
    retried = orch.cancel_run(run_id)

    assert retried.state == "cancelled"
    assert retried.deployment == preserved
    assert orch.read_verified_adapter_revisions(run_id) == frozenset(
        {preserved["adapter_revision"]}
    )


def test_cancel_double_undeploy_failure_revokes_authority_and_is_retryable(tmp_path, monkeypatch):
    import flash.runner as orch
    import flash.serve.deploy as deploy

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

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
    assert attempts == [run_id]
    failed = orch.get_status(run_id)
    assert failed.state == "cancelled"
    assert failed.deployment["state"] == "revocation_failed"
    assert failed.deployment["retryable"] is True
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()

    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target: attempts.append(target) or {})
    retried = orch.cancel_run(run_id)

    assert attempts == [run_id, run_id]
    assert retried.state == "cancelled"
    assert retried.deployment["state"] == "undeployed"
    assert orch.read_verified_adapter_revisions(run_id) == frozenset()


# ---------------------------------------------------------------------------
# registry-less REST fallback in terminate_endpoint: when the resource registry has lost the
# endpoint (e.g. across a container restart), cleanup still has to find the orphan by name on
# every configured account and delete it. an unconfirmed delete must surface as a failed result
# rather than a silent success, or a live endpoint keeps billing while the run looks cleaned up.
# ---------------------------------------------------------------------------
def _fake_sdk_with_orphan(monkeypatch, *, rest_find, rest_delete, resources=None, undeploy=None):
    """Stub auth + a resource registry so terminate_endpoint reaches the REST fallback.

    ``resources`` defaults to an empty registry (nothing to undeploy, straight to REST). Pass one
    plus ``undeploy`` to drive the registry leg too. ``rest_find`` may raise to model an
    unreachable account-enumeration API.
    """
    import types as _types

    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.auth as auth
    import flash.providers.runpod.serverless.endpoints as ep_mod
    from flash.providers.base import canonical_gpu

    monkeypatch.setattr(auth, "ensure_auth", lambda: None)
    monkeypatch.setattr(ep_mod, "isolate_flash_state", lambda *a, **k: None)

    registry = resources or {}
    fake_rm = _types.SimpleNamespace(
        list_all_resources=lambda: registry,
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
            stub = types.ModuleType(mod_name)
            stub.__path__ = []
            monkeypatch.setitem(sys.modules, mod_name, stub)
    monkeypatch.setitem(sys.modules, "runpod_flash.core.resources.resource_manager", fake_rm_mod)

    target = endpoint_name(canonical_gpu("RTX 5090"), _run_suffix("flash-q-1"))
    monkeypatch.setattr(
        runpod_api, "list_endpoints_by_key", lambda: ({_RUNPOD_FINGERPRINT: rest_find(target)}, [])
    )
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, _fingerprint: rest_delete(endpoint_id),
    )
    return target


def test_terminate_deletes_a_rest_discovered_orphan(monkeypatch):
    deleted = []
    target = _fake_sdk_with_orphan(
        monkeypatch,
        rest_find=lambda t: [{"id": "ep-orphan", "name": t}],
        rest_delete=lambda eid: deleted.append(eid) or True,
    )

    out = ftrain.terminate_endpoint("RTX 5090", "flash-q-1")

    assert deleted == ["ep-orphan"], "an orphan matching the run must be deleted on its account"
    assert {"success": True, "name": target, "message": "deleted via REST API"} in out


def test_terminate_reports_an_unconfirmed_rest_delete_as_failure(monkeypatch):
    # a delete the API would not confirm must NOT be reported as success: the endpoint may still
    # be live and billing, and cancellation is what the caller believes just happened.
    target = _fake_sdk_with_orphan(
        monkeypatch,
        rest_find=lambda t: [{"id": "ep-orphan", "name": t}],
        rest_delete=lambda _eid: False,
    )

    out = ftrain.terminate_endpoint("RTX 5090", "flash-q-1")

    assert {
        "success": False,
        "name": target,
        "message": "REST endpoint deletion was unconfirmed",
    } in out


def test_terminate_keeps_undeploy_failures_when_rest_enumeration_is_unreachable(monkeypatch):
    """terminate_endpoint is best-effort: it must never raise, and never lose what it learned.

    An undeploy that raises becomes a failure row rather than escaping, and a REST enumeration that
    is unreachable is swallowed -- but the earlier undeploy failure must survive that swallow. A
    caller that saw an empty list here would read 'nothing to clean up' from what was really 'the
    API could not tell us', and stop chasing an endpoint that is still billing.
    """

    async def _undeploy_boom(_uid, **_):
        raise RuntimeError("undeploy boom")

    def _enumeration_down(_target):
        raise RuntimeError("REST API down")

    _fake_sdk_with_orphan(
        monkeypatch,
        resources={"u1": _res(f"live-{endpoint_name('RTX 5090', _run_suffix('flash-q-1'))}")},
        undeploy=_undeploy_boom,
        rest_find=_enumeration_down,
        rest_delete=lambda _eid: True,
    )

    out = ftrain.terminate_endpoint("RTX 5090", "flash-q-1")

    assert isinstance(out, list), "an unreachable REST API must be swallowed, not raised"
    assert any(
        r.get("success") is False and "undeploy boom" in str(r.get("message")) for r in out
    ), "the undeploy failure must survive the swallowed enumeration error"
