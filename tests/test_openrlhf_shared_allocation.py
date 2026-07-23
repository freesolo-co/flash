"""cpu-only tests for shared allocation and per-run control isolation."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from flash.providers.base import JobHandle, UnreconciledCreateError
from flash.runner.openrlhf_shared_allocation import (
    SharedAllocationRequest,
    SharedAllocationState,
    SharedAllocationStateError,
    SharedBundleAllocationSession,
    SharedQueueAllocationBackend,
    SharedRunAuthenticationError,
    SharedRunCommandKind,
)
from flash.runner.openrlhf_shared_bundle import (
    BundleAdmissionOutcome,
    LogicalRunStatus,
    SharedEngineBundle,
)
from flash.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec


@dataclass(frozen=True)
class _Handle:
    allocation_id: str


class _FakeBackend:
    def __init__(self) -> None:
        self.requests: list[SharedAllocationRequest] = []
        self.releases: list[tuple[object, str]] = []

    def allocate(self, request: SharedAllocationRequest) -> object:
        self.requests.append(request)
        return _Handle(f"allocation-{len(self.requests)}")

    def release(self, handle: object, bundle_id: str) -> None:
        self.releases.append((handle, bundle_id))


class _UnreconciledBackend(_FakeBackend):
    def allocate(self, request: SharedAllocationRequest) -> object:
        self.requests.append(request)
        raise UnreconciledCreateError("endpoint cleanup pending")


class _FlakyReleaseBackend(_FakeBackend):
    def release(self, handle: object, bundle_id: str) -> None:
        self.releases.append((handle, bundle_id))
        if len(self.releases) == 1:
            raise RuntimeError("release failed")


class _Clock:
    def __init__(self) -> None:
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _spec(
    run_id: str,
    *,
    algorithm: str = "grpo",
    max_wall_seconds: int = 120,
    environment: EnvironmentSpec | None = None,
) -> JobSpec:
    return JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm=algorithm,
        run_id=run_id,
        environment=environment or EnvironmentSpec(),
        train=TrainSpec(
            lora_rank=64,
            lora_alpha=128,
            max_context_tokens=8192,
            max_completion_tokens=512,
            batch_size=8,
            group_size=4,
            max_steps=2,
        ),
        gpu=GpuSpec(
            type="H100",
            exact_type="H100",
            max_wall_seconds=max_wall_seconds,
        ),
    )


def _bundle(*run_ids: str) -> SharedEngineBundle:
    bundle = SharedEngineBundle("bundle-one", _spec("seed"))
    for index, run_id in enumerate(run_ids):
        algorithm = "opd" if index % 2 else "grpo"
        decision = bundle.try_admit(_spec(run_id, algorithm=algorithm))
        assert decision.outcome is BundleAdmissionOutcome.ADMITTED
    bundle.seal()
    return bundle


def _bundle_from_specs(*specs: JobSpec) -> SharedEngineBundle:
    bundle = SharedEngineBundle("bundle-custom", specs[0])
    for spec in specs:
        decision = bundle.try_admit(spec)
        assert decision.outcome is BundleAdmissionOutcome.ADMITTED
    bundle.seal()
    return bundle


def _session(*run_ids: str):
    tokens = iter(f"token-{index}" for index in range(len(run_ids)))
    backend = _FakeBackend()
    clock = _Clock()
    session = SharedBundleAllocationSession(
        _bundle(*run_ids),
        backend,
        token_factory=lambda: next(tokens),
        clock=clock,
    )
    return session, backend, clock


def test_one_sealed_bundle_creates_one_allocation_sized_for_admitted_runs():
    session, backend, clock = _session("run-a", "run-b", "run-c")

    handle = session.allocate()

    assert handle == _Handle("allocation-1")
    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.bundle_id == "bundle-one"
    assert request.admitted_run_count == 3
    assert request.engine_adapter_capacity == 4
    assert request.compatibility_key.gpu_type == "H100"
    assert request.compatibility_key.gpu_count == 1
    assert tuple(run.run_id for run in request.runs) == ("run-a", "run-b", "run-c")
    assert request.deadline_at == clock.value + 120
    assert request.worker_manifest()["admitted_run_count"] == 3
    assert request.worker_manifest()["engine_adapter_capacity"] == 4
    assert session.allocation_state is SharedAllocationState.ACTIVE

    with pytest.raises(RuntimeError, match="may be created only once"):
        session.allocate()
    assert len(backend.requests) == 1


def test_session_keeps_one_absolute_deadline_across_delay_and_retry():
    session, backend, clock = _session("run-a")
    original_deadline = session.allocation_request().deadline_at

    clock.advance(30)
    delayed_request = session.allocation_request()
    session.allocate()

    assert original_deadline == 1120.0
    assert delayed_request.deadline_at == original_deadline
    assert backend.requests[0].deadline_at == original_deadline


def test_unreconciled_create_cannot_allocate_a_second_endpoint():
    backend = _UnreconciledBackend()
    session = SharedBundleAllocationSession(_bundle("run-a"), backend)

    with pytest.raises(UnreconciledCreateError, match="cleanup pending"):
        session.allocate()

    assert session.allocation_state is SharedAllocationState.RELEASED
    assert session.allocation_handle is None
    with pytest.raises(SharedAllocationStateError, match="may be created only once"):
        session.allocate()
    assert len(backend.requests) == 1


def test_session_rejects_mixed_wall_limits_before_allocation():
    bundle = _bundle_from_specs(
        _spec("run-a", max_wall_seconds=60),
        _spec("run-b", max_wall_seconds=120),
    )

    with pytest.raises(ValueError, match="identical wall-clock limits"):
        SharedBundleAllocationSession(bundle, _FakeBackend())


def test_session_rejects_runtime_secret_environments_until_isolated_delivery_lands():
    bundle = _bundle_from_specs(
        _spec(
            "run-a",
            environment=EnvironmentSpec(secrets=("PRIVATE_TOKEN",)),
        )
    )

    with pytest.raises(ValueError, match="isolated secret delivery from PR9"):
        SharedBundleAllocationSession(bundle, _FakeBackend())


def test_queue_backend_fails_closed_until_shared_worker_runtime_lands():
    session, _backend, _clock = _session("run-a")
    backend = SharedQueueAllocationBackend(persist_cleanup_handle=lambda _handle: None)

    with pytest.raises(SharedAllocationStateError, match="runtime is not yet available"):
        backend.allocate(session.allocation_request())


def test_queue_backend_rejects_conflicting_worker_dependencies():
    bundle = _bundle_from_specs(
        _spec("run-a", environment=EnvironmentSpec(pip=("package-a==1",))),
        _spec("run-b", environment=EnvironmentSpec(pip=("package-a==2",))),
    )
    tokens = iter(("token-a", "token-b"))
    session = SharedBundleAllocationSession(
        bundle,
        _FakeBackend(),
        token_factory=lambda: next(tokens),
    )
    backend = SharedQueueAllocationBackend(
        persist_cleanup_handle=lambda _handle: None,
        code_prefix="code/0123456789abcdef0123456789abcdef/flash",
        worker_runtime_available=True,
    )

    with pytest.raises(ValueError, match="identical worker Python dependencies"):
        backend.allocate(session.allocation_request())


def test_queue_backend_reuses_existing_endpoint_submit_and_teardown(monkeypatch):
    session, _backend, _clock = _session("run-a", "run-b")
    request = session.allocation_request()
    calls: dict[str, object] = {}

    def deploy(gpu_type, **kwargs):
        calls["deploy"] = (gpu_type, kwargs)
        return "endpoint-id", "endpoint-name", "rpk-0123456789ab"

    def submit(endpoint_id, function_input, *, key_fingerprint, deadline_at):
        calls["submit"] = (endpoint_id, function_input, key_fingerprint, deadline_at)
        return "job-id"

    def teardown(handle, bundle_id):
        calls["teardown"] = (handle.to_dict(), bundle_id)
        return True

    def upload(repo, *, code_prefix, deadline_at):
        calls["upload"] = (repo, code_prefix, deadline_at)

    def require_allowance(deadline_at):
        calls["allowance"] = deadline_at
        return 75

    monkeypatch.setattr("flash.providers.runpod.jobs.deploy_train_endpoint", deploy)
    monkeypatch.setattr("flash.providers.runpod.jobs.build_function_input", lambda payload: payload)
    monkeypatch.setattr("flash.providers.runpod.api.submit_job", submit)
    monkeypatch.setattr("flash.envs.registry.worker_pip_for_env", lambda _env_id: ["env-dep"])
    monkeypatch.setattr("flash.providers._worker.chalk_extra_pip", lambda _spec: ["chalk-dep"])
    monkeypatch.setattr("flash.providers._worker.upload_code", upload)
    monkeypatch.setattr("flash.providers._deadline.require_create_allowance", require_allowance)
    monkeypatch.setattr(
        "flash.runner.flash_code_prefix",
        lambda: "code/0123456789abcdef0123456789abcdef/flash",
    )
    monkeypatch.setattr("flash.runner.lifecycle._strict_teardown_handle", teardown)

    backend = SharedQueueAllocationBackend(
        persist_cleanup_handle=lambda _handle: None,
        worker_runtime_available=True,
    )
    handle = backend.allocate(request)
    backend.release(handle, request.bundle_id)

    assert set(calls) == {"upload", "deploy", "allowance", "submit", "teardown"}
    assert calls["allowance"] == request.deadline_at
    upload_repo, upload_prefix, upload_deadline = calls["upload"]
    assert upload_repo == request.seed_spec.train.hf_repo
    assert upload_prefix == "code/0123456789abcdef0123456789abcdef/flash"
    assert upload_deadline == request.deadline_at
    gpu_type, deploy_kwargs = calls["deploy"]
    assert gpu_type == "H100"
    assert deploy_kwargs["execution_timeout_ms"] == 75_000
    assert deploy_kwargs["disk_gb"] == 60
    assert deploy_kwargs["spec"].run_id == "run-a"
    assert deploy_kwargs["deadline_at"] == request.deadline_at

    endpoint_id, payload, key_fingerprint, deadline_at = calls["submit"]
    assert endpoint_id == "endpoint-id"
    assert key_fingerprint == "rpk-0123456789ab"
    assert deadline_at == request.deadline_at
    assert payload["phase"] == "shared"
    assert payload["extra_pip"] == ["env-dep", "chalk-dep"]
    manifest = json.loads(payload["shared_bundle_manifest_json"])
    assert manifest["bundle_id"] == "bundle-one"
    assert manifest["admitted_run_count"] == 2
    assert manifest["engine_adapter_capacity"] == 3
    assert [run["run_id"] for run in manifest["runs"]] == ["run-a", "run-b"]
    assert manifest["runs"][0]["control_token"] != manifest["runs"][1]["control_token"]
    teardown_handle, teardown_bundle_id = calls["teardown"]
    assert teardown_handle["endpoint_id"] == "endpoint-id"
    assert teardown_handle["job_id"] == "job-id"
    assert teardown_bundle_id == "bundle-one"


def test_unconfirmed_submit_failure_persists_exact_cleanup_handle(monkeypatch):
    session, _backend, _clock = _session("run-a")
    request = session.allocation_request()
    cleanup_handles: list[dict] = []

    monkeypatch.setattr(
        "flash.providers.runpod.jobs.deploy_train_endpoint",
        lambda *_args, **_kwargs: ("endpoint-id", "endpoint-name", "rpk-0123456789ab"),
    )

    def fail_submit(*_args, **_kwargs):
        raise RuntimeError("submit failed")

    monkeypatch.setattr("flash.providers.runpod.jobs.build_function_input", lambda payload: payload)
    monkeypatch.setattr(
        "flash.providers._deadline.require_create_allowance", lambda _deadline: 75
    )
    monkeypatch.setattr("flash.providers.runpod.api.submit_job", fail_submit)
    monkeypatch.setattr(
        "flash.providers.runpod.api.delete_endpoint_for_fingerprint",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr("flash.envs.registry.worker_pip_for_env", lambda _env_id: [])
    monkeypatch.setattr("flash.providers._worker.chalk_extra_pip", lambda _spec: [])

    backend = SharedQueueAllocationBackend(
        persist_cleanup_handle=lambda handle: cleanup_handles.append(handle),
        code_prefix="code/0123456789abcdef0123456789abcdef/flash",
        worker_runtime_available=True,
    )

    with pytest.raises(UnreconciledCreateError, match="could not be reconciled"):
        backend.allocate(request)

    assert len(cleanup_handles) == 1
    assert cleanup_handles[0]["provider"] == "runpod"
    assert cleanup_handles[0]["endpoint_id"] == "endpoint-id"
    assert cleanup_handles[0]["endpoint_name"] == "endpoint-name"
    assert cleanup_handles[0]["key_fingerprint"] == "rpk-0123456789ab"
    assert "job_id" not in cleanup_handles[0]


def test_unconfirmed_terminal_release_persists_exact_cleanup_handle_once(monkeypatch):
    cleanup_handles: list[dict] = []
    queue_backend = SharedQueueAllocationBackend(
        persist_cleanup_handle=lambda handle: cleanup_handles.append(handle),
    )
    handle = JobHandle(
        provider="runpod",
        data={
            "endpoint_id": "endpoint-id",
            "endpoint_name": "endpoint-name",
            "key_fingerprint": "rpk-0123456789ab",
            "job_id": "job-id",
            "attempt": 0,
            "started_ts": 1000.0,
        },
    )
    teardown_calls: list[tuple[dict, str]] = []

    def terminal_without_endpoint_delete(canonical, bundle_id):
        teardown_calls.append((canonical.to_dict(), bundle_id))
        return False

    class _SessionBackend:
        def allocate(self, _request):
            return handle

        def release(self, release_handle, bundle_id):
            queue_backend.release(release_handle, bundle_id)

    monkeypatch.setattr(
        "flash.runner.lifecycle._strict_teardown_handle",
        terminal_without_endpoint_delete,
    )
    session = SharedBundleAllocationSession(
        _bundle("run-a"),
        _SessionBackend(),
        token_factory=lambda: "token-a",
    )
    session.allocate()
    capability = session.capability("run-a")
    session.submit("run-a", capability.token)

    session.complete("run-a", capability.token)
    session.complete("run-a", capability.token)

    assert session.allocation_state is SharedAllocationState.RELEASED
    assert teardown_calls == [(handle.to_dict(), "bundle-one")]
    assert cleanup_handles == [handle.to_dict()]


def test_cancel_isolated_to_authorized_run_and_frees_only_its_slot():
    session, backend, clock = _session("run-a", "run-b")
    session.allocate()
    run_a = session.capability("run-a")
    run_b = session.capability("run-b")
    session.submit("run-a", run_a.token)
    session.submit("run-b", run_b.token)

    clock.advance(5)
    heartbeat = session.heartbeat("run-b", run_b.token)
    cancelled = session.cancel("run-a", run_a.token)

    assert cancelled.status is LogicalRunStatus.CANCELLED
    assert session.status("run-b", run_b.token).status is LogicalRunStatus.ACTIVE
    assert heartbeat.heartbeat_sequence == 1
    assert heartbeat.last_heartbeat_at == 1005.0
    assert session.occupied_slots == 1
    assert session.available_slots == 1
    assert backend.releases == []

    run_a_commands = session.poll_commands("run-a", run_a.token)
    run_b_commands = session.poll_commands("run-b", run_b.token)
    assert [command.kind for command in run_a_commands] == [
        SharedRunCommandKind.START,
        SharedRunCommandKind.CANCEL,
    ]
    assert [command.kind for command in run_b_commands] == [SharedRunCommandKind.START]


def test_run_capability_cannot_control_or_read_a_sibling():
    session, backend, _clock = _session("run-a", "run-b")
    session.allocate()
    run_a = session.capability("run-a")
    run_b = session.capability("run-b")
    session.submit("run-a", run_a.token)

    with pytest.raises(SharedRunAuthenticationError, match="capability rejected"):
        session.cancel("run-b", run_a.token)
    with pytest.raises(SharedRunAuthenticationError, match="capability rejected"):
        session.status("run-b", run_a.token)
    with pytest.raises(SharedRunAuthenticationError, match="capability rejected"):
        session.poll_commands("run-b", "wrong-token")

    assert session.status("run-a", run_a.token).status is LogicalRunStatus.ACTIVE
    assert session.status("run-b", run_b.token).status is LogicalRunStatus.QUEUED
    assert session.occupied_slots == 2
    assert backend.releases == []


def test_allocation_releases_only_after_every_run_is_terminal():
    session, backend, _clock = _session("run-a", "run-b", "run-c")
    handle = session.allocate()
    capabilities = {run_id: session.capability(run_id) for run_id in ("run-a", "run-b", "run-c")}
    for run_id, capability in capabilities.items():
        session.submit(run_id, capability.token)

    failed = session.fail("run-a", capabilities["run-a"].token, "reward bridge failed")

    assert failed.status is LogicalRunStatus.FAILED
    assert failed.error == "reward bridge failed"
    assert session.status("run-b", capabilities["run-b"].token).status is LogicalRunStatus.ACTIVE
    assert [
        command.kind for command in session.poll_commands("run-a", capabilities["run-a"].token)
    ] == [SharedRunCommandKind.START, SharedRunCommandKind.CANCEL]
    assert backend.releases == []
    assert session.available_slots == 1

    session.complete("run-b", capabilities["run-b"].token)
    assert backend.releases == []
    assert session.available_slots == 2

    session.cancel("run-c", capabilities["run-c"].token)
    assert backend.releases == [(handle, "bundle-one")]
    assert session.allocation_state is SharedAllocationState.RELEASED
    assert session.available_slots == 3
    assert session.release_if_terminal() is False
    assert backend.releases == [(handle, "bundle-one")]


@pytest.mark.parametrize(
    ("terminal_action", "expected_status"),
    [
        ("complete", LogicalRunStatus.DONE),
        ("fail", LogicalRunStatus.FAILED),
        ("cancel", LogicalRunStatus.CANCELLED),
    ],
)
def test_terminal_retry_retries_failed_release(terminal_action, expected_status):
    backend = _FlakyReleaseBackend()
    session = SharedBundleAllocationSession(
        _bundle("run-a"),
        backend,
        token_factory=lambda: "token-a",
    )
    handle = session.allocate()
    capability = session.capability("run-a")
    session.submit("run-a", capability.token)

    def invoke_terminal_action():
        if terminal_action == "complete":
            return session.complete("run-a", capability.token)
        if terminal_action == "fail":
            return session.fail("run-a", capability.token, "worker failed")
        return session.cancel("run-a", capability.token)

    with pytest.raises(RuntimeError, match="release failed"):
        invoke_terminal_action()

    assert session.status("run-a", capability.token).status is expected_status
    assert session.allocation_state is SharedAllocationState.ACTIVE
    assert backend.releases == [(handle, "bundle-one")]

    retried = invoke_terminal_action()

    assert retried.status is expected_status
    assert session.allocation_state is SharedAllocationState.RELEASED
    assert backend.releases == [(handle, "bundle-one"), (handle, "bundle-one")]

    invoke_terminal_action()
    assert backend.releases == [(handle, "bundle-one"), (handle, "bundle-one")]


def test_drain_before_allocation_does_not_mutate_session_state():
    session, backend, _clock = _session("run-a")

    with pytest.raises(RuntimeError, match="allocation is not active"):
        session.drain()

    assert session.drained is False
    assert session.allocation_state is SharedAllocationState.NEW
    handle = session.allocate()
    run_a = session.capability("run-a")
    session.submit("run-a", run_a.token)
    session.complete("run-a", run_a.token)
    assert backend.releases == [(handle, "bundle-one")]


def test_draining_bundle_cancels_remaining_runs_and_releases_once():
    session, backend, _clock = _session("run-a", "run-b")
    handle = session.allocate()
    run_a = session.capability("run-a")
    run_b = session.capability("run-b")
    session.submit("run-a", run_a.token)
    session.submit("run-b", run_b.token)
    session.complete("run-a", run_a.token)

    session.drain()

    assert session.drained is True
    assert session.status("run-a", run_a.token).status is LogicalRunStatus.DONE
    assert session.status("run-b", run_b.token).status is LogicalRunStatus.CANCELLED
    assert backend.releases == [(handle, "bundle-one")]
    assert session.allocation_state is SharedAllocationState.RELEASED

    session.drain()
    assert backend.releases == [(handle, "bundle-one")]
