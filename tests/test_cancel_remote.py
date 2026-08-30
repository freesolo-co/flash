"""verify cancellation stops a remote flash worker across processes.

the deploying process may be gone by the time a cancellation arrives, so cancellation must use
``terminate_endpoint`` to find the persisted runpod resource and delete it through the api before
billing continues.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

import flash.providers.runpod.serverless.endpoints as ftrain
import flash.runner.accounting.costs as runner_costs
import flash.runner.accounting.reconciliation as runner_reconciliation
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.results.verified_revisions as runner_verified_revisions
import flash.runner.supervise.attach as runner_attach
import flash.runner.supervise.deploy as runner_deploy
import flash.runner.supervise.errors as runner_errors
import flash.runner.supervise.lifecycle as runner_lifecycle
import flash.runner.supervise.recovery as runner_recovery
import flash.runner.supervise.transitions as runner_transitions
import flash.serve.contract.errors as serving_errors
from flash.providers.runpod.serverless.naming import (
    attempt_suffix,
    endpoint_name,
    run_suffix,
    select_endpoint_resources,
)
from tests._helpers.runner import provisioned_status, save_provisioned_status
from tests._helpers.source_snapshot import valid_source_snapshot

_RUNPOD_FINGERPRINT = "rpk-" + "0" * 64
_SOURCE_SNAPSHOT = valid_source_snapshot()


def _remote(endpoint_id, job_id, attempt):
    return {
        "provider": "runpod",
        "endpoint_id": endpoint_id,
        "endpoint_name": f"flash-{endpoint_id}",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "job_id": job_id,
        "attempt": attempt,
        "started_ts": float(attempt + 1),
        # a live persisted handle carries the token that authorized its attempt and the allocation
        # stamp retry reconstructs its candidate from. both are written by the same persist.
        "launch_claim_token": f"token-{endpoint_id}-{attempt}",
        "allocated_gpu": "RTX 5090",
        "allocated_gpu_count": 1,
        "allocated_usable_vram_gb": 32.0,
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


def test_deploy_and_terminate_isolate_the_same_registry_scope(monkeypatch):
    """Deploy must write the SDK registry where teardown reads it.

    The endpoint *name* is attempt-scoped (``<digest>-aN``) but ``terminate_endpoint`` is
    run-scoped: it isolates on the bare run digest and reaps every attempt in one call. If deploy
    isolates under the attempt instead, it writes a ``resources.pkl`` teardown never opens, so the
    undeploy leg reads an empty registry and cleanup silently rests on the REST sweep alone.
    Attempt zero is the tell: it used to share a scope with teardown, and an explicit ``-a0``
    breaks that unless the scope is derived from the run.
    """
    import inspect

    import flash.providers.runpod.execution.job_execution as je

    # read the scope expression straight out of deploy_train_endpoint rather than reimplementing
    # it: a hand-recomputed scope agrees with itself no matter what the production line says.
    src = inspect.getsource(je.deploy_train_endpoint)
    scope_lines = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("registry_scope")]
    assert len(scope_lines) == 1, f"expected one registry_scope assignment, got {scope_lines}"
    scope_expr = scope_lines[0].split("=", 1)[1].strip()
    assert "isolate_flash_state(registry_scope)" in src, (
        "deploy must isolate on registry_scope; it now passes something else"
    )

    run_id = "flash-scope-1"
    terminate_scope = run_suffix(run_id)
    for attempt in (0, 1, 7):
        name_suffix = attempt_suffix(run_id, attempt)
        deploy_scope = eval(
            scope_expr, {"runpod_naming": je.runpod_naming}, {"name_suffix": name_suffix}
        )
        assert deploy_scope == terminate_scope, (
            f"attempt {attempt}: deploy isolates {deploy_scope!r} but teardown reads "
            f"{terminate_scope!r}; the undeploy leg would find an empty registry"
        )
        # the name itself stays attempt-scoped - the fix must not collapse attempt identity
        assert endpoint_name("b200", name_suffix).endswith(f"-a{attempt}")


def test_select_matches_live_prefixed_endpoint():
    run_id = "flash-123-c220526e"
    target = endpoint_name("RTX 5090", run_suffix(run_id))  # flash-5090-<digest>
    attempt = endpoint_name("RTX 5090", attempt_suffix(run_id, 0))
    resources = {
        "u1": _res(f"live-{attempt}"),  # the live-provisioned resource for this run's attempt
        "u2": _res("flash-5090-deadbeef-a0"),  # a different run
        "u3": _res("live-flash-4090-c220526e-a0"),  # different GPU class
        "u4": _res(f"live-{target}"),  # the bare run target names no attempt
    }
    assert select_endpoint_resources(resources, target) == ["u1"]


def test_select_empty_target_matches_nothing():
    assert select_endpoint_resources({"u1": _res("live-flash-5090-x")}, "") == []


def test_terminate_endpoint_never_raises_when_sdk_missing(monkeypatch):
    # ensure_auth raises (no key) -> terminate_endpoint must swallow and return a result list
    import flash.providers.runpod.client.auth as auth

    monkeypatch.setattr(auth, "ensure_auth", lambda: (_ for _ in ()).throw(RuntimeError("no key")))
    out = ftrain.terminate_endpoint("RTX 5090", "flash-1-abcd1234")
    assert isinstance(out, list)
    assert out
    assert out[0]["success"] is False


@pytest.mark.parametrize("failed_revocation_call", [1, 2])
def test_cancel_run_revocation_failure_defers_until_after_fence_and_teardown(
    tmp_path, monkeypatch, failed_revocation_call
):
    from flash.core.spec import JobSpec
    from flash.runner.supervise import lifecycle
    from flash.server.platform import db as server_db

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "gpu": {"type": "RTX 5090"},
            "run_id": f"flash-revoke-failure-{failed_revocation_call}",
        }
    )
    status = provisioned_status(
        spec,
        state="running",
        remote=_remote("endpoint-1", "job-1", 1),
    )
    runner_state._save_status(status)
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
        teardown_calls.append((handle.provider, run_id, runner_status.get_status(run_id).state))
        return True

    monkeypatch.setattr(lifecycle, "_strict_teardown_handle", teardown)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)

    with pytest.raises(RuntimeError, match=f"revocation failure {failed_revocation_call}"):
        runner_deploy.cancel_run(spec.run_id)

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.state == "cancelled"
    assert persisted.remote is None
    assert teardown_calls == [("runpod", spec.run_id, "cancelled")]
    assert revocation_calls == 2


def test_cancel_run_calls_terminate_and_marks_cancelled(tmp_path, monkeypatch):

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "gpu": {"type": "RTX 5090"},
            "run_id": "flash-9-feedface",
        }
    )
    st = provisioned_status(spec, state="running")
    runner_state._save_status(st)

    calls = {}

    def fake_terminate(gpu, run_id):
        calls["gpu"] = gpu
        calls["run_id"] = run_id
        return [{"success": True}]

    monkeypatch.setattr(ftrain, "terminate_endpoint", fake_terminate)

    out = runner_deploy.cancel_run(spec.run_id)
    assert calls == {"gpu": "RTX 5090", "run_id": "flash-9-feedface"}, (
        "must terminate the remote endpoint"
    )
    assert out.state == "cancelled"


def test_cancel_tears_down_every_acceptable_class_of_an_ordered_pin(tmp_path, monkeypatch):
    """Cancellation tears down every endpoint name an ordered pin could select."""

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "gpu": {"type": ["A100 PCIe", "A100 SXM"]},
            "run_id": "flash-9-feedface",
        }
    )
    runner_state._save_status(provisioned_status(spec, state="running"))

    terminated: list[str] = []
    monkeypatch.setattr(
        ftrain,
        "terminate_endpoint",
        lambda gpu, run_id: terminated.append(gpu) or [{"success": True}],
    )

    assert runner_deploy.cancel_run(spec.run_id).state == "cancelled"
    assert terminated == ["A100 PCIe", "A100 SXM"]


def test_terminal_charge_uses_the_selected_fallback_after_remote_cleanup(monkeypatch):
    from flash.runner.lifecycle.state import RunStatus
    from flash.server.billing import charges

    captured: dict[str, object] = {}

    def post_billing(*, token, path, body):
        captured.update(token=token, path=path, body=body)
        return {"ok": True}

    monkeypatch.setattr(charges, "_post_billing", post_billing)
    status = RunStatus(
        run_id="flash-billed-fallback",
        state="cancelled",
        spec={"algorithm": "sft", "model": "m", "gpu": {"type": ["RTX 5090", "A100 PCIe"]}},
        effective_preparation={"worker_spec": {"gpu": {"type": "A100 PCIe"}}},
        billing_context={"org_id": "org-1"},
        cost_usd=1.25,
    )

    charges.charge_completed_run(internal_key="internal", status=status)

    assert captured["body"]["gpu"] == "A100 PCIe"


def test_late_cancellation_uses_retained_rented_basis(tmp_path, monkeypatch):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "gpu": {"type": "RTX 5090"},
            "run_id": "flash-late-cancel-basis",
        }
    )
    retained = {
        **_remote("endpoint-finished", "job-finished", 0),
        "allocated_gpu": "A100 PCIe",
        "allocated_gpu_count": 4,
    }
    status = provisioned_status(spec, state="running", remote=None)
    status.billing_context = {"org_id": "org-1"}
    status.realized_cost_remote = retained
    runner_state._save_status(status)
    captured = []

    def cancellation_billing(run_id, effective_spec, *, bill_cancel, rented_remote):
        captured.append((run_id, bill_cancel, rented_remote))
        return 0.5, {}

    monkeypatch.setattr(runner_deploy, "_cancellation_billing", cancellation_billing)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)

    cancelled = runner_deploy.cancel_run(spec.run_id)

    assert cancelled.state == "cancelled"
    assert captured == [(spec.run_id, True, retained)]


def test_terminal_charge_uses_retained_remote_after_confirmed_cleanup(monkeypatch):
    from flash.runner.lifecycle.state import RunStatus
    from flash.server.billing import charges

    captured: dict[str, object] = {}

    def post_billing(*, token, path, body):
        captured.update(token=token, path=path, body=body)
        return {"ok": True}

    monkeypatch.setattr(charges, "_post_billing", post_billing)
    status = RunStatus(
        run_id="flash-billed-cleanup",
        state="done",
        spec={"algorithm": "sft", "model": "m", "gpu": {"type": "RTX 5090"}},
        remote=None,
        realized_cost_remote={"provider": "runpod", "allocated_gpu": "A100 PCIe"},
        billing_context={"org_id": "org-1"},
        cost_usd=1.25,
    )

    charges.charge_completed_run(internal_key="internal", status=status)

    assert captured["body"]["provider"] == "runpod"
    assert captured["body"]["gpu"] == "A100 PCIe"


def test_cancel_deployed_run_marks_deployment_inactive(tmp_path, monkeypatch):
    # Cancelling a deployed run tears down its serve endpoint; the deployment record
    # must flip to "undeployed" so /v1/deployments and /chat stop treating the
    # cancelled run as active (and can't recreate the endpoint).
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-dep-1"})
    st = runner_state.RunStatus(
        run_id=spec.run_id,
        state="deployed",
        spec=spec.to_dict(),
        platform_context={"org_id": "org-1"},
        deployment={
            "state": "ready",
            "gpu": "RTX 5090",
            "checkpoint_id": f"{spec.run_id}/final",
        },
    )
    runner_state._save_status(st)

    monkeypatch.setattr(deploy, "undeploy_adapter", lambda *a, **k: ["flash-serve-5090-x"])
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    out = runner_deploy.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"


def test_cancel_undeploys_deployment_that_raced_in_after_entry_snapshot(tmp_path, monkeypatch):
    # Race: cancel_run enters on a non-`deployed` snapshot (state="running"), but a deploy lands during
    # teardown (running -> done -> deployed) before the terminal `cancelled` write. `deployed` is
    # non-terminal so `cancelled` still wins, but the entry-gated undeploy never ran. cancel_run must
    # re-read post-write and tear down the raced-in deployment so it is never orphaned.
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-dep-racein"})
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            platform_context={"org_id": "org-1"},
        )
    )

    undeployed: list[str] = []
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda rid, *a, **k: undeployed.append(rid) or ["x"]
    )
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    # Inject the deploy race at the last step before the terminal write (after the entry snapshot).
    real_gc = runner_recovery._gc_run_endpoints

    def gc_then_deploy(s):
        real_gc(s)
        revision = f"{spec.run_id}/final"
        runner_transitions.mark_deployed(
            spec.run_id,
            {"state": "ready", "gpu": "RTX 5090", "checkpoint_id": revision},
            verification_generation=runner_verified_revisions.verified_checkpoint_generation(
                spec.run_id
            ),
        )

    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", gc_then_deploy)

    out = runner_deploy.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert undeployed == [f"{spec.run_id}/final"], (
        "the raced-in deployment must be torn down, not orphaned"
    )
    assert (out.deployment or {}).get("state") == "undeployed"


def test_cancel_deployed_run_undeploy_goes_through_lock_guarded_path(tmp_path, monkeypatch):
    # Regression: the deployed branch used a bare _save_status OUTSIDE _STATUS_LOCK, which
    # persisted a stale pre-teardown snapshot and bypassed serialization. It must instead
    # mark the exact checkpoint inactive through the lock-guarded mark_undeployed helper.
    import inspect

    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-dep-lock"})
    st = runner_state.RunStatus(
        run_id=spec.run_id,
        state="deployed",
        spec=spec.to_dict(),
        platform_context={"org_id": "org-1"},
        deployment={
            "state": "ready",
            "gpu": "RTX 5090",
            "checkpoint_id": f"{spec.run_id}/final",
        },
    )
    runner_state._save_status(st)

    monkeypatch.setattr(deploy, "undeploy_adapter", lambda *a, **k: ["flash-serve-5090-x"])
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    # The undeploy write must route through the lock-guarded helper (not a bare _save_status
    # outside _STATUS_LOCK, the old racy path); that helper holds _STATUS_LOCK.
    assert "with _status_guard(run_id)" in inspect.getsource(runner_transitions.mark_undeployed)

    called = []
    real_helper = runner_transitions.mark_undeployed

    def spy(run_id, checkpoint_id=None):
        called.append((run_id, checkpoint_id))
        return real_helper(run_id, checkpoint_id)

    monkeypatch.setattr(runner_transitions, "mark_undeployed", spy)

    out = runner_deploy.cancel_run(spec.run_id)
    assert called == [(spec.run_id, f"{spec.run_id}/final")]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"


def test_cancel_deployed_run_undeployed_even_when_raced_to_terminal(tmp_path, monkeypatch):
    # Race: while cancel_run tears down a `deployed` run, a concurrent mark_undeployed moves
    # the run to terminal `done` on disk. The deployment-field write must NOT re-assert a
    # non-terminal state (the old _update(run_id, "deployed", deployment=...) path no-ops
    # against the terminal `done` CAS, leaving the deployment advertised as `ready`). It must
    # mark the deployment undeployed regardless of the terminal race.
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-dep-race"})
    st = runner_state.RunStatus(
        run_id=spec.run_id,
        state="deployed",
        spec=spec.to_dict(),
        platform_context={"org_id": "org-1"},
        deployment={
            "state": "ready",
            "gpu": "RTX 5090",
            "checkpoint_id": f"{spec.run_id}/final",
        },
    )
    runner_state._save_status(st)

    monkeypatch.setattr(deploy, "undeploy_adapter", lambda *a, **k: ["flash-serve-5090-x"])
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    # Inject the race: a concurrent mark_undeployed flips the run to terminal `done` AFTER
    # cancel_run's initial get_status (state="deployed") but BEFORE the deployment is retired.
    def racing_undeploy(*a, **k):
        # mark_undeployed moves a live `deployed` run to terminal `done`.
        runner_transitions.mark_undeployed(spec.run_id, f"{spec.run_id}/final")
        return ["flash-serve-5090-x"]

    monkeypatch.setattr(deploy, "undeploy_adapter", racing_undeploy)

    out = runner_deploy.cancel_run(spec.run_id)
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
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancel-wins"})
    st = runner_state.RunStatus(
        run_id=spec.run_id,
        state="deployed",
        spec=spec.to_dict(),
        platform_context={"org_id": "org-1"},
        deployment={
            "state": "ready",
            "gpu": "RTX 5090",
            "checkpoint_id": f"{spec.run_id}/final",
        },
    )
    runner_state._save_status(st)

    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    # The racing undeploy flips the run to terminal `done` mid-cancel (after cancel_run's
    # initial non-terminal read, before its final `cancelled` write).
    def racing_undeploy(*a, **k):
        runner_transitions.mark_undeployed(spec.run_id, f"{spec.run_id}/final")
        assert runner_status.get_status(spec.run_id).state == "done"  # the race landed
        return ["flash-serve-5090-x"]

    monkeypatch.setattr(deploy, "undeploy_adapter", racing_undeploy)

    out = runner_deploy.cancel_run(spec.run_id)
    assert out.state == "cancelled", "explicit cancel must win over a racing undeploy `done`"
    assert runner_status.get_status(spec.run_id).state == "cancelled"
    assert out.deployment["state"] == "undeployed"


def test_cancel_loses_to_racing_genuine_completion_done(tmp_path, monkeypatch):
    # if a running job genuinely finishes while cancellation tears it down, preserve its done metrics
    # and artifacts. only runs deployed at cancellation entry allow the terminal override; a blanket
    # override would clobber real training results.

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-finish-race"})
    # A `running` run (NOT deployed): no deployment, an in-flight training thread.
    st = runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
    runner_state._save_status(st)

    # Inject the race: the training thread completes mid-teardown and writes the terminal
    # `done` with a real result (mirrors _run_job_inner's finish path) AFTER cancel_run's
    # initial non-terminal read but BEFORE its final `cancelled` write.
    def racing_completion(*a, **k):
        runner_status._update(spec.run_id, "done", cost_usd=1.23, artifacts_dir="/runs/finished")
        assert runner_status.get_status(spec.run_id).state == "done"  # the genuine finish landed
        return [{"success": True}]

    monkeypatch.setattr(ftrain, "terminate_endpoint", racing_completion)

    out = runner_deploy.cancel_run(spec.run_id)
    assert out.state == "done", (
        "a genuine training-completion `done` must NOT be clobbered by cancel"
    )
    assert runner_status.get_status(spec.run_id).state == "done"
    assert out.cost_usd == 1.23, "the finished run's real result (cost) must be preserved"
    assert out.artifacts_dir == "/runs/finished"


def test_terminate_endpoint_holds_lock_across_isolation(monkeypatch):
    """Regression (6 bot threads): isolate_flash_state() + the ResourceManager lookup must run
    UNDER FLASH_SDK_LOCK, not just the undeploy. isolate_flash_state swaps runpod_flash's
    process-wide registry globals, so a concurrent deploy could swap the scope mid-teardown.
    Asserts the lock is held when isolate_flash_state runs (and released afterward)."""
    import flash.providers.runpod.client.auth as auth

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


def test_terminate_endpoint_from_async_context_does_not_raise(monkeypatch):
    """verify terminate_endpoint works inside an active event loop.

    it must move ``asyncio.run`` to another thread because calling it directly from fastapi/uvicorn or
    any async context raises RuntimeError.
    """
    import asyncio
    import sys
    import types as _types

    import flash.providers.runpod.client.auth as auth
    import flash.providers.runpod.serverless.endpoints as ep_mod
    from flash.providers.core.base import canonical_gpu

    run_id = "flash-1-abcd1234"
    friendly = canonical_gpu("RTX 5090")
    target = endpoint_name(friendly, attempt_suffix(run_id, 0))
    resource_name = f"live-{target}"

    monkeypatch.setattr(auth, "ensure_auth", lambda: None)
    monkeypatch.setattr(ep_mod, "isolate_flash_state", lambda _: None)

    # The registry-less REST sweep runs after the undeploy and would report every configured
    # account as unreachable (offline suite), appending a failure row this assertion would then
    # have to spell out. Stub it clear: this test is about the event loop, not the sweep.
    import flash.providers.runpod.client.api as runpod_api

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

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-done-1"})
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="done", spec=spec.to_dict())
    )

    called = {"v": False}
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: called.__setitem__("v", True))
    out = runner_deploy.cancel_run(spec.run_id)
    assert out.state == "done"
    assert called["v"] is False, "must not tear down endpoints for an already-terminal run"


def test_cancel_run_retries_durable_cleanup_for_cancelled_run(tmp_path, monkeypatch):
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancelled-1"})
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="cancelled", spec=spec.to_dict())
    )
    remote = _remote("endpoint-cleanup", "job-cleanup", 1)
    assert runner_reconciliation._preserve_cleanup_remote(spec.run_id, remote) is True
    events = []

    class Provider:
        def cancel(self, handle):
            events.append(("cancel", handle.to_dict()["job_id"]))

        def destroy(self, handle):
            events.append(("destroy", handle.to_dict()["endpoint_id"]))

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())

    out = runner_deploy.cancel_run(spec.run_id)

    assert out.state == "cancelled"
    assert events == [("cancel", "job-cleanup"), ("destroy", "endpoint-cleanup")]
    assert runner_state._CLEANUP_REMOTES_KEY not in runner_status._load_status_json(spec.run_id)


def test_cancel_run_accepts_confirmed_endpoint_delete_after_cancel_ack_failure(
    tmp_path, monkeypatch
):
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancel-retry"})
    remote = {**_remote("endpoint-exact", "job-exact", 7), "seed": 42}
    runner_state._save_status(provisioned_status(spec, state="running", remote=remote))
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
    monkeypatch.setattr(
        runner_recovery, "_gc_run_endpoints", lambda value: gc_calls.append(value.run_id)
    )

    result = runner_deploy.cancel_run(spec.run_id)
    raw = runner_status._load_status_json(spec.run_id)

    assert result.state == "cancelled"
    assert raw["remote"] is None
    assert runner_state._CLEANUP_REMOTES_KEY not in raw
    assert gc_calls == [spec.run_id]
    assert events == [
        ("cancel", "endpoint-exact", "job-exact", 7),
        ("destroy", "endpoint-exact", "job-exact", 7),
    ]


def test_cancel_run_failed_teardown_does_not_replace_racing_public_remote(tmp_path, monkeypatch):
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancel-race"})
    original_remote = _remote("endpoint-original", "job-original", 2)
    replacement_remote = _remote("endpoint-replacement", "job-replacement", 3)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            remote=original_remote,
        )
    )

    class Provider:
        def cancel(self, _handle):
            current = runner_status.get_status(spec.run_id)
            current.remote = replacement_remote
            runner_state._save_status(current)
            raise RuntimeError("cancellation acknowledgement failed")

        def destroy(self, _handle):
            raise RuntimeError("endpoint deletion failed")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)

    out = runner_deploy.cancel_run(spec.run_id)
    raw = runner_status._load_status_json(spec.run_id)

    assert out.state == "cancelled"
    assert raw["remote"] == replacement_remote
    # cleanup records are canonical teardown identities, not launch authorizations: the launch
    # token and the allocation stamp are both dropped by the provider handle canonicalization.
    from flash.providers.runpod.execution.jobs import JobHandle as RunpodJobHandle

    assert raw[runner_state._CLEANUP_REMOTES_KEY] == [
        RunpodJobHandle.from_dict(remote).to_dict()
        for remote in (original_remote, replacement_remote)
    ]
    for record in raw[runner_state._CLEANUP_REMOTES_KEY]:
        assert "launch_claim_token" not in record


def test_cancel_run_marks_billing_failed_when_pricing_falls_back(tmp_path, monkeypatch):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancel-price"})
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            billing_context={"org_id": "org-a"},
        )
    )
    monkeypatch.setattr(runner_costs, "actual_steps_run", lambda _status: 1)
    monkeypatch.setattr(runner_costs, "charge_usd_for_spec", lambda *a, **kw: kw["fallback"])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)

    status = runner_deploy.cancel_run(spec.run_id)

    assert status.state == "cancelled"
    assert status.cost_usd == 0.0
    assert status.billing_state == "failed"
    assert "pricing failed" in (status.billing_error or "")


def test_cancel_run_successful_exact_teardown_leaves_no_cleanup_remote(tmp_path, monkeypatch):
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-cancel-clean"})
    remote = _remote("endpoint-clean", "job-clean", 3)
    runner_state._save_status(
        runner_state.RunStatus(
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
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)

    out = runner_deploy.cancel_run(spec.run_id)

    assert out.state == "cancelled"
    assert events == [("cancel", "job-clean"), ("destroy", "endpoint-clean")]
    assert runner_state._CLEANUP_REMOTES_KEY not in runner_status._load_status_json(spec.run_id)


# ---------------------------------------------------------------------------
# Recovery TOCTOU: a run flipped terminal mid-recovery must not submit paid work
# ---------------------------------------------------------------------------
def _make_poll_provider(monkeypatch, *, on_poll):
    """Wire flash.providers.get_provider to a stub provider whose poll_attempt() runs ``on_poll``.

    Also no-ops _gc_run_endpoints so attach_run's teardown doesn't reach the real SDK.
    """
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda *a, **k: None)

    class _StubProvider:
        def poll_attempt(self, handle, spec, *, log=None, _deadline_at=None):
            return on_poll(handle, spec)

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

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-race-terminal"})
    st = runner_state.RunStatus(
        run_id=spec.run_id,
        state="running",
        spec=spec.to_dict(),
        remote=_remote("ep-1", "job-1", 0),
    )
    runner_state._save_status(st)

    # _run_training is the PAID-work entry point; it must never be called for a terminal run.
    training_calls = {"n": 0}
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_training",
        lambda *a, **k: training_calls.__setitem__("n", training_calls["n"] + 1),
    )

    from flash.providers.core.base import PollResult

    def racing_poll(handle, spec):
        # A concurrent recovery/cancel flips the run terminal AFTER attach_run's initial check
        # (top of attach_run) but BEFORE the not-ok recovery resume below.
        runner_status._update(spec.run_id, "failed", error="raced terminal by another thread")
        assert runner_status.get_status(spec.run_id).state == "failed"  # the race landed
        return PollResult(False, failure="stalled", detail="control plane was down")

    _make_poll_provider(monkeypatch, on_poll=racing_poll)

    out = runner_attach.attach_run(spec.run_id)
    assert training_calls["n"] == 0, (
        "must NOT submit paid work (resume training) for a run raced to terminal"
    )
    assert out.state == "failed", "the authoritative terminal state must be preserved"
    assert runner_status.get_status(spec.run_id).state == "failed"


def test_attach_run_recovery_resumes_training_when_still_active(tmp_path, monkeypatch):
    """Happy-path guard: the TOCTOU fix must NOT regress a genuine recovery. A not-ok poll on a
    run that is STILL active (no terminal race) must resume `_run_training` exactly as before."""

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-recover-active"})
    st = provisioned_status(spec, state="running", remote=_remote("ep-1", "job-1", 0))
    st.source_snapshot = _SOURCE_SNAPSHOT
    save_provisioned_status(st)

    training_calls = {"n": 0}
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_training",
        lambda *a, **k: training_calls.__setitem__("n", training_calls["n"] + 1),
    )
    from flash.providers.core.base import PollResult

    _make_poll_provider(
        monkeypatch,
        on_poll=lambda h, s: PollResult(False, failure="stalled", detail="redeploy"),
    )

    out = runner_attach.attach_run(spec.run_id)
    assert training_calls["n"] == 1, "a still-active run must resume training (no regression)"
    # the replacement attempt is reserved as `provisioning`; the real `_run_training` (stubbed
    # here) is what flips it back to `running`. what matters is that the run stays live.
    assert out.state not in runner_state.TERMINAL_STATES


def test_run_training_bails_on_terminal_before_paid_work(tmp_path, monkeypatch):
    """Defense in depth: _run_training's own pre-submit guard bails on ANY terminal state
    (not just `cancelled`). If the run is terminal when training is entered — e.g. a concurrent
    thread marked it `done`/`failed` after the caller decided to resume — it must raise
    _RunCancelled and never call _run_attempts_supervised (the paid GPU submit)."""

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-loop-terminal"})
    # The run is already terminal (failed) before training runs.
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="failed", spec=spec.to_dict())
    )

    submitted = {"n": 0}
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_attempts_supervised",
        lambda *a, **k: submitted.__setitem__("n", submitted["n"] + 1),
    )

    import io

    import pytest

    with pytest.raises(runner_errors._RunCancelled):
        runner_lifecycle._run_training(
            spec,
            io.StringIO(),
            prior_cost=0.0,
            source_snapshot=_SOURCE_SNAPSHOT,
        )
    assert submitted["n"] == 0, "no paid GPU work may be submitted for an already-terminal run"
    assert runner_status.get_status(spec.run_id).state == "failed", (
        "the terminal state must be untouched"
    )


def test_update_returns_false_when_terminal_sticky(tmp_path, monkeypatch):
    """_update now reports whether the transition applied: True normally, False when the sticky
    terminal CAS rejects it. The recovery guard relies on this signal."""

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": "flash-update-ret"})
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
    )
    assert runner_status._update(spec.run_id, "running", cost_usd=1.0) is True
    assert (
        runner_status._update(spec.run_id, "failed", error="boom") is True
    )  # terminal write applies
    # Now terminal: a non-terminal transition is rejected and reported False.
    assert runner_status._update(spec.run_id, "running") is False
    assert runner_status.get_status(spec.run_id).state == "failed"


def _run_spec(run_id: str):
    from flash.core.spec import JobSpec

    return JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})


def _checkpoint_id(run_id: str, step: int, sha: str = "a") -> str:
    del sha
    return f"{run_id}/step-{step}"


def _ready_checkpoint(run_id: str, step: int, *, remote: dict | None = None) -> dict:
    spec = _run_spec(run_id)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            remote=remote,
        )
    )
    deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "checkpoint_id": _checkpoint_id(run_id, step),
        "checkpoint_step": step,
    }
    runner_transitions.mark_checkpoint_deployed(
        run_id,
        deployment,
        verification_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
    )
    return deployment


def test_cancel_tears_down_training_before_checkpoint_serving_decision(tmp_path, monkeypatch):
    import flash.runner.results.verified_revisions as verified_revisions
    import flash.serve.deployment.deploy as deploy
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-order"
    remote = _remote("endpoint-training", "training-job", 0)
    deployment = _ready_checkpoint(run_id, 40, remote=remote)
    events = []

    class StubProvider:
        def cancel(self, handle):
            events.append("provider-cancel")

        def destroy(self, handle):
            events.append("provider-destroy")

    real_read = verified_revisions.read_verified_checkpoints

    def read_verified(target):
        events.append("serving-decision")
        return real_read(target)

    monkeypatch.setattr(providers, "get_provider", lambda _provider: StubProvider())
    monkeypatch.setattr(
        runner_recovery, "_gc_run_endpoints", lambda _spec: events.append("endpoint-gc")
    )
    monkeypatch.setattr(verified_revisions, "read_verified_checkpoints", read_verified)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("a valid checkpoint deployment must remain serving"),
    )

    out = runner_deploy.cancel_run(run_id)

    assert events == [
        "provider-cancel",
        "provider-destroy",
        "endpoint-gc",
        "serving-decision",
        "serving-decision",
    ]
    assert out.state == "cancelled"
    assert out.deployment == deployment


def test_checkpoint_restore_owner_fence_rejects_newer_attempt(tmp_path, monkeypatch):

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-stale-restore"
    previous = _ready_checkpoint(run_id, 40)
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
    status = runner_status.get_status(run_id)
    status.deployment = newer_attempt
    runner_state._save_status(status)

    out = runner_transitions.mark_checkpoint_deployed(
        run_id,
        previous,
        owner_deployment=stale_owner,
        verification_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
    )

    assert out.deployment == newer_attempt
    assert runner_status.get_status(run_id).deployment == newer_attempt


def test_cancel_revokes_inflight_checkpoint_deployment(tmp_path, monkeypatch):
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-inflight"
    spec = _run_spec(run_id)
    revision = _checkpoint_id(run_id, 40)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            platform_context={"org_id": "org-1"},
            deployment={
                "state": "deploying",
                "checkpoint_id": revision,
                "checkpoint_step": 40,
            },
            billing_context={"org_id": "org-a"},
        )
    )
    runner_verified_revisions.add_verified_checkpoint(
        run_id,
        revision,
        expected_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
    )
    undeployed = []
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda target, *, org_id: undeployed.append((org_id, target)),
    )

    out = runner_deploy.cancel_run(run_id)

    assert undeployed == [("org-a", revision)]
    assert out.deployment["state"] == "undeployed"
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset()


@pytest.mark.parametrize(
    "retired_model",
    ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B"],
)
def test_cancel_active_removed_model_still_cleans_up_and_revokes(
    tmp_path, monkeypatch, retired_model
):
    import flash.serve.deployment.deploy as deploy
    import flash.server.platform.locks as locks

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-malformed-spec-revoke"
    spec = _run_spec(run_id)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            platform_context={"org_id": "org-1"},
            deployment={
                "state": "deploying",
                "checkpoint_id": f"{run_id}/final",
                "checkpoint_step": None,
            },
        )
    )
    raw = runner_status._load_status_json(run_id)
    raw["spec"]["model"] = retired_model
    with open(runner_state.runs_file_path(run_id, ".json"), "w") as file:
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
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: gc_calls.append(_spec))
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda target, **_: backend_calls.append(target)
    )

    out = runner_deploy.cancel_run(run_id)

    assert [spec.model for spec in gc_calls] == [retired_model]
    assert backend_calls == [f"{run_id}/final"]
    assert out.state == "cancelled"
    assert out.deployment["state"] == "undeployed"


def test_cancel_backend_success_local_commit_failure_is_not_backend_uncertainty(
    tmp_path, monkeypatch
):
    import flash.serve.deployment.deploy as deploy
    from flash.runner.supervise.deploy import DeploymentStatePersistenceError

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-local-persistence-failure"
    spec = _run_spec(run_id)
    revision = f"{run_id}/final"
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            platform_context={"org_id": "org-1"},
            deployment={"state": "ready", "checkpoint_id": revision},
        )
    )
    backend_calls = []
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda target, **_: backend_calls.append(target)
    )
    monkeypatch.setattr(
        runner_transitions,
        "mark_undeployed",
        lambda _run_id, _checkpoint_id=None: (_ for _ in ()).throw(
            OSError("status store unavailable")
        ),
    )

    with pytest.raises(DeploymentStatePersistenceError) as excinfo:
        runner_deploy.cancel_run(run_id)

    assert not isinstance(excinfo.value, runner_deploy.DeploymentRevocationError)
    assert excinfo.value.backend_outcome == "confirmed"
    assert "backend disablement was confirmed" in str(excinfo.value)
    assert backend_calls == [f"{run_id}/final"]
    failed = runner_status.get_status(run_id)
    assert failed.state == "cancelled"
    assert failed.deployment["state"] == "revocation_failed"
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset()


def test_repeated_cancel_preserves_checkpoint_serving(tmp_path, monkeypatch):
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-repeat"
    deployment = _ready_checkpoint(run_id, 40)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("repeated cancel must not revoke the checkpoint"),
    )

    first = runner_deploy.cancel_run(run_id)
    second = runner_deploy.cancel_run(run_id)

    assert first.state == second.state == "cancelled"
    assert first.deployment == second.deployment == deployment
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset(
        {deployment["checkpoint_id"]}
    )


def test_cancel_preserved_checkpoint_keeps_verified_ready_siblings(tmp_path, monkeypatch):
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-prune"
    preserved = _ready_checkpoint(run_id, 40, remote=None)
    older_revisions = {
        _checkpoint_id(run_id, 20, "b"),
        f"{run_id}/final",
    }
    for revision in older_revisions:
        runner_verified_revisions.add_verified_checkpoint(
            run_id,
            revision,
            expected_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
        )
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("the preserved checkpoint must remain serving"),
    )

    out = runner_deploy.cancel_run(run_id)

    assert out.state == "cancelled"
    assert out.deployment == preserved
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset(
        {*older_revisions, preserved["checkpoint_id"]}
    )


def test_cancel_double_undeploy_failure_revokes_authority_and_is_retryable(tmp_path, monkeypatch):
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    run_id = "flash-dep-revoke"
    spec = JobSpec.from_dict({"gpu": {"type": "RTX 5090"}, "run_id": run_id})
    revision = f"{run_id}/final"
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=run_id,
            state="deployed",
            spec=spec.to_dict(),
            platform_context={"org_id": "org-1"},
            deployment={
                "state": "ready",
                "checkpoint_id": revision,
                "endpoint_name": "https://serve.example",
            },
        )
    )
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset()

    attempts = []

    def fail_undeploy(target, **_):
        attempts.append(target)
        raise serving_errors.ServingError("backend unavailable")

    monkeypatch.setattr(deploy, "undeploy_adapter", fail_undeploy)
    monkeypatch.setattr(ftrain, "terminate_endpoint", lambda *a, **k: [{"success": True}])

    with pytest.raises(runner_deploy.DeploymentRevocationError) as excinfo:
        runner_deploy.cancel_run(run_id)

    assert excinfo.value.retryable is True
    assert attempts == [revision]
    failed = runner_status.get_status(run_id)
    assert failed.state == "cancelled"
    assert failed.deployment["state"] == "revocation_failed"
    assert failed.deployment["retryable"] is True
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset()

    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda target, **_: attempts.append(target) or {}
    )
    retried = runner_deploy.cancel_run(run_id)

    assert attempts == [revision, revision]
    assert retried.state == "cancelled"
    assert retried.deployment["state"] == "undeployed"
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset()


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

    import flash.providers.runpod.client.api as runpod_api
    import flash.providers.runpod.client.auth as auth
    import flash.providers.runpod.serverless.endpoints as ep_mod
    from flash.providers.core.base import canonical_gpu

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

    target = endpoint_name(canonical_gpu("RTX 5090"), run_suffix("flash-q-1"))
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
        rest_find=lambda t: [{"id": "ep-orphan", "name": f"{t}-a0"}],
        rest_delete=lambda eid: deleted.append(eid) or True,
    )

    out = ftrain.terminate_endpoint("RTX 5090", "flash-q-1")

    assert deleted == ["ep-orphan"], "an orphan matching the run must be deleted on its account"
    assert {"success": True, "name": f"{target}-a0", "message": "deleted via REST API"} in out


def test_terminate_reports_an_unconfirmed_rest_delete_as_failure(monkeypatch):
    # a delete the API would not confirm must NOT be reported as success: the endpoint may still
    # be live and billing, and cancellation is what the caller believes just happened.
    target = _fake_sdk_with_orphan(
        monkeypatch,
        rest_find=lambda t: [{"id": "ep-orphan", "name": f"{t}-a0"}],
        rest_delete=lambda _eid: False,
    )

    out = ftrain.terminate_endpoint("RTX 5090", "flash-q-1")

    assert {
        "success": False,
        "name": f"{target}-a0",
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
        resources={"u1": _res(f"live-{endpoint_name('RTX 5090', attempt_suffix('flash-q-1', 0))}")},
        undeploy=_undeploy_boom,
        rest_find=_enumeration_down,
        rest_delete=lambda _eid: True,
    )

    out = ftrain.terminate_endpoint("RTX 5090", "flash-q-1")

    assert isinstance(out, list), "an unreachable REST API must be swallowed, not raised"
    assert any(
        r.get("success") is False and "undeploy boom" in str(r.get("message")) for r in out
    ), "the undeploy failure must survive the swallowed enumeration error"
