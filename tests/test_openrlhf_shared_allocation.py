"""cpu-only tests for shared allocation and per-run control isolation."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from flash.runner.openrlhf_shared_allocation import (
    SharedAllocationRequest,
    SharedAllocationState,
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
from flash.spec import GpuSpec, JobSpec, TrainSpec


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


class _Clock:
    def __init__(self) -> None:
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _spec(run_id: str, *, algorithm: str = "grpo") -> JobSpec:
    return JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm=algorithm,
        run_id=run_id,
        train=TrainSpec(
            lora_rank=64,
            lora_alpha=128,
            max_context_tokens=8192,
            max_completion_tokens=512,
            group_size=4,
            max_steps=2,
        ),
        gpu=GpuSpec(type="H100", exact_type="H100", max_wall_seconds=120),
    )


def _bundle(*run_ids: str) -> SharedEngineBundle:
    bundle = SharedEngineBundle("bundle-one", _spec("seed"))
    for index, run_id in enumerate(run_ids):
        algorithm = "opd" if index % 2 else "grpo"
        decision = bundle.try_admit(_spec(run_id, algorithm=algorithm))
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


def test_queue_backend_reuses_existing_endpoint_submit_and_teardown(monkeypatch):
    session, _backend, _clock = _session("run-a", "run-b")
    request = session.allocation_request()
    calls: dict[str, object] = {}

    def deploy(gpu_type, **kwargs):
        calls["deploy"] = (gpu_type, kwargs)
        return "endpoint-id", "endpoint-name", "rpk-0123456789ab"

    def submit(endpoint_id, function_input, *, key_fingerprint):
        calls["submit"] = (endpoint_id, function_input, key_fingerprint)
        return "job-id"

    def teardown(handle, bundle_id):
        calls["teardown"] = (handle.to_dict(), bundle_id)
        return True

    monkeypatch.setattr("flash.providers.runpod.jobs.deploy_train_endpoint", deploy)
    monkeypatch.setattr("flash.providers.runpod.jobs.build_function_input", lambda payload: payload)
    monkeypatch.setattr("flash.providers.runpod.api.submit_job", submit)
    monkeypatch.setattr("flash.envs.registry.worker_pip_for_env", lambda _env_id: ["env-dep"])
    monkeypatch.setattr("flash.providers._worker.chalk_extra_pip", lambda _spec: ["chalk-dep"])
    monkeypatch.setattr("flash.runner.lifecycle._strict_teardown_handle", teardown)

    backend = SharedQueueAllocationBackend(
        code_prefix="code/0123456789abcdef0123456789abcdef/flash"
    )
    handle = backend.allocate(request)
    backend.release(handle, request.bundle_id)

    assert set(calls) == {"deploy", "submit", "teardown"}
    gpu_type, deploy_kwargs = calls["deploy"]
    assert gpu_type == "H100"
    assert deploy_kwargs["execution_timeout_ms"] == 120_000
    assert deploy_kwargs["disk_gb"] == 60
    assert deploy_kwargs["spec"].run_id == "run-a"

    endpoint_id, payload, key_fingerprint = calls["submit"]
    assert endpoint_id == "endpoint-id"
    assert key_fingerprint == "rpk-0123456789ab"
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
