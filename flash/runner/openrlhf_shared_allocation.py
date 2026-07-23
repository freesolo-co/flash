"""one shared allocation with authenticated per-run control and status."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol

from flash.runner.openrlhf_shared_bundle import (
    BundleCompatibilityKey,
    BundleRunSnapshot,
    LogicalRunStatus,
    SharedEngineBundle,
)
from flash.spec import JobSpec

_TERMINAL_RUN_STATUSES = frozenset(
    {LogicalRunStatus.DONE, LogicalRunStatus.FAILED, LogicalRunStatus.CANCELLED}
)
_DUMMY_CAPABILITY_DIGEST = hashlib.sha256(b"invalid-shared-run-capability").digest()


class SharedRunAuthenticationError(PermissionError):
    """raised when a capability does not authorize the requested logical run."""


class SharedAllocationStateError(RuntimeError):
    """raised when a bundle allocation or control transition is invalid."""


class SharedRunCommandKind(StrEnum):
    """commands sent from the control plane to one logical worker run."""

    START = "start"
    CANCEL = "cancel"


class SharedAllocationState(StrEnum):
    """physical lifecycle for the bundle's single allocation."""

    NEW = "new"
    ACTIVE = "active"
    RELEASING = "releasing"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class SharedRunCapability:
    """unguessable authority for exactly one logical run."""

    run_id: str
    token: str

    def __repr__(self) -> str:
        return f"SharedRunCapability(run_id={self.run_id!r}, token=<redacted>)"


@dataclass(frozen=True, slots=True)
class SharedRunCommand:
    """one ordered control-plane command for a logical run."""

    sequence: int
    kind: SharedRunCommandKind


@dataclass(frozen=True, slots=True)
class SharedRunControlSnapshot:
    """authenticated status view for one logical run only."""

    run_id: str
    status: LogicalRunStatus
    error: str | None
    heartbeat_sequence: int
    last_heartbeat_at: float | None


@dataclass(frozen=True, slots=True)
class SharedAllocationRun:
    """one sealed run carried inside the shared worker manifest."""

    run_id: str
    spec_json: str
    capability: SharedRunCapability

    @property
    def spec(self) -> JobSpec:
        return JobSpec.from_json(self.spec_json)

    def worker_manifest(self) -> dict:
        return {
            "run_id": self.run_id,
            "job_spec_json": self.spec_json,
            "control_token": self.capability.token,
        }


@dataclass(frozen=True, slots=True)
class SharedAllocationRequest:
    """one sealed bundle request submitted as one provider queue job."""

    bundle_id: str
    compatibility_key: BundleCompatibilityKey
    runs: tuple[SharedAllocationRun, ...]
    admitted_run_count: int
    engine_adapter_capacity: int
    deadline_at: float

    @property
    def seed_spec(self) -> JobSpec:
        return self.runs[0].spec

    def worker_manifest(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "compatibility_key": asdict(self.compatibility_key),
            "admitted_run_count": self.admitted_run_count,
            "engine_adapter_capacity": self.engine_adapter_capacity,
            "runs": [run.worker_manifest() for run in self.runs],
        }


class SharedAllocationBackend(Protocol):
    """physical allocation boundary used by the bundle session."""

    def allocate(self, request: SharedAllocationRequest) -> object:
        """create and submit exactly one physical allocation."""

    def release(self, handle: object, bundle_id: str) -> None:
        """release the exact physical allocation."""


class SharedQueueAllocationBackend:
    """reuse Flash's existing RunPod endpoint and queue-job lifecycle."""

    def __init__(
        self,
        *,
        persist_cleanup_handle: Callable[[dict], None],
        code_prefix: str | None = None,
        process_env: dict[str, str] | None = None,
    ) -> None:
        self._persist_cleanup_handle = persist_cleanup_handle
        self._code_prefix = code_prefix
        self._process_env = dict(process_env or {})

    def allocate(self, request: SharedAllocationRequest) -> object:
        from flash.envs.registry import worker_pip_for_env
        from flash.providers._worker import chalk_extra_pip, weight_cache_env
        from flash.providers.base import UnreconciledCreateError
        from flash.providers.runpod import api as queue_api
        from flash.providers.runpod.jobs import (
            JobHandle,
            build_function_input,
            deploy_train_endpoint,
        )
        from flash.providers.runpod.train import _run_suffix
        from flash.runner import flash_code_prefix

        seed_spec = request.seed_spec
        execution_timeout_ms = (
            max(int(run.spec.gpu.max_wall_seconds) for run in request.runs) * 1000
        )
        extra_pip: list[str] = []
        seen_dependencies: set[str] = set()
        for run in request.runs:
            dependencies = list(run.spec.environment.pip) or worker_pip_for_env(
                run.spec.environment.id
            )
            dependencies += chalk_extra_pip(run.spec)
            for dependency in dependencies:
                if dependency not in seen_dependencies:
                    seen_dependencies.add(dependency)
                    extra_pip.append(dependency)

        env = {"FLASH_ARM": "runpod", **self._process_env}
        for key in ("HF_TOKEN", "GITHUB_TOKEN"):
            if os.environ.get(key):
                env[key] = os.environ[key]
        if seed_spec.gpu.network_volume:
            env.update(weight_cache_env())

        payload = {
            "hf_repo": seed_spec.train.hf_repo,
            "job_spec_json": seed_spec.to_json(),
            "phase": "shared",
            "seed": seed_spec.seed,
            "env": env,
            "extra_pip": extra_pip,
            "code_prefix": self._code_prefix or flash_code_prefix(),
            "deadline_at": request.deadline_at,
            "shared_bundle_manifest_json": json.dumps(
                request.worker_manifest(), sort_keys=True, separators=(",", ":")
            ),
        }
        endpoint_id, endpoint_name, key_fingerprint = deploy_train_endpoint(
            request.compatibility_key.gpu_type,
            execution_timeout_ms=execution_timeout_ms,
            name_suffix=_run_suffix(request.bundle_id),
            disk_gb=max(int(run.spec.gpu.disk_gb) for run in request.runs),
            spec=seed_spec,
            deadline_at=request.deadline_at,
        )
        submitted_at = time.time()
        try:
            job_id = queue_api.submit_job(
                endpoint_id,
                build_function_input(payload),
                key_fingerprint=key_fingerprint,
                deadline_at=request.deadline_at,
            )
        except Exception as exc:
            deletion_confirmed = False
            with contextlib.suppress(Exception):
                deletion_confirmed = (
                    queue_api.delete_endpoint_for_fingerprint(endpoint_id, key_fingerprint) is True
                )
            if deletion_confirmed:
                raise
            cleanup_handle = JobHandle(
                endpoint_id=endpoint_id,
                endpoint_name=endpoint_name,
                key_fingerprint=key_fingerprint,
                job_id=None,
                attempt=0,
                started_ts=submitted_at,
            )
            self._persist_cleanup_handle(cleanup_handle.to_dict())
            raise UnreconciledCreateError(
                "shared queue submission could not be reconciled and endpoint deletion was unconfirmed"
            ) from exc
        return JobHandle(
            endpoint_id=endpoint_id,
            endpoint_name=endpoint_name,
            key_fingerprint=key_fingerprint,
            job_id=job_id,
            attempt=0,
            started_ts=submitted_at,
        )

    def release(self, handle: object, bundle_id: str) -> None:
        from flash.providers.base import JobHandle
        from flash.runner.lifecycle import _strict_teardown_handle

        if not hasattr(handle, "to_dict"):
            raise TypeError("shared allocation handle must provide to_dict")
        canonical = JobHandle.from_dict(handle.to_dict())
        if not _strict_teardown_handle(canonical, bundle_id):
            raise RuntimeError("shared allocation endpoint deletion was not confirmed")


class SharedBundleAllocationSession:
    """control one sealed bundle, one allocation, and isolated logical runs."""

    def __init__(
        self,
        bundle: SharedEngineBundle,
        backend: SharedAllocationBackend,
        *,
        token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not bundle.sealed:
            raise ValueError("shared allocation requires a sealed bundle")
        snapshots = bundle.run_snapshots()
        if not snapshots:
            raise ValueError("shared allocation requires at least one run")
        self._bundle = bundle
        self._backend = backend
        self._clock = clock
        self._lock = threading.RLock()
        self._allocation_state = SharedAllocationState.NEW
        self._handle: object | None = None
        self._drained = False
        self._command_sequence = 0
        self._commands: dict[str, deque[SharedRunCommand]] = {
            snapshot.run_id: deque() for snapshot in snapshots
        }
        self._heartbeat_sequence = {snapshot.run_id: 0 for snapshot in snapshots}
        self._last_heartbeat_at: dict[str, float | None] = {
            snapshot.run_id: None for snapshot in snapshots
        }
        self._capabilities: dict[str, SharedRunCapability] = {}
        self._capability_digests: dict[str, bytes] = {}
        used_digests: set[bytes] = set()
        for snapshot in snapshots:
            token = token_factory()
            if not isinstance(token, str) or not token:
                raise ValueError("shared run capability must be a nonempty string")
            digest = self._capability_digest(token)
            if digest in used_digests:
                raise ValueError("shared run capabilities must be unique")
            used_digests.add(digest)
            capability = SharedRunCapability(snapshot.run_id, token)
            self._capabilities[snapshot.run_id] = capability
            self._capability_digests[snapshot.run_id] = digest

    @property
    def bundle_id(self) -> str:
        return self._bundle.bundle_id

    @property
    def allocation_state(self) -> SharedAllocationState:
        with self._lock:
            return self._allocation_state

    @property
    def allocation_handle(self) -> object | None:
        with self._lock:
            return self._handle

    @property
    def admitted_run_count(self) -> int:
        return len(self._capabilities)

    @property
    def occupied_slots(self) -> int:
        with self._lock:
            return sum(
                snapshot.status not in _TERMINAL_RUN_STATUSES
                for snapshot in self._bundle.run_snapshots()
            )

    @property
    def available_slots(self) -> int:
        return self.admitted_run_count - self.occupied_slots

    @property
    def drained(self) -> bool:
        with self._lock:
            return self._drained

    def capability(self, run_id: str) -> SharedRunCapability:
        with self._lock:
            try:
                return self._capabilities[str(run_id).strip()]
            except KeyError as exc:
                raise KeyError("unknown shared bundle run") from exc

    def allocation_request(self) -> SharedAllocationRequest:
        with self._lock:
            runs = tuple(
                SharedAllocationRun(
                    snapshot.run_id,
                    snapshot.spec_json,
                    self._capabilities[snapshot.run_id],
                )
                for snapshot in self._bundle.run_snapshots()
            )
            max_wall_seconds = max(int(run.spec.gpu.max_wall_seconds) for run in runs)
            return SharedAllocationRequest(
                bundle_id=self.bundle_id,
                compatibility_key=self._bundle.compatibility_key,
                runs=runs,
                admitted_run_count=len(runs),
                engine_adapter_capacity=len(runs) + 1,
                deadline_at=float(self._clock()) + max_wall_seconds,
            )

    def allocate(self) -> object:
        with self._lock:
            if self._drained:
                raise SharedAllocationStateError("a drained shared bundle cannot be allocated")
            if self._allocation_state is not SharedAllocationState.NEW:
                raise SharedAllocationStateError(
                    "shared bundle allocation may be created only once"
                )
            request = self.allocation_request()
            handle = self._backend.allocate(request)
            self._handle = handle
            self._allocation_state = SharedAllocationState.ACTIVE
            return handle

    def submit(self, run_id: str, token: str) -> SharedRunControlSnapshot:
        """admit one logical run into the already-created shared worker."""

        with self._lock:
            normalized = self._authorize(run_id, token)
            self._require_active_allocation()
            snapshot = self._bundle.run_snapshot(normalized)
            if snapshot.status is LogicalRunStatus.QUEUED:
                self._bundle.transition_run(normalized, LogicalRunStatus.ACTIVE)
                self._enqueue(normalized, SharedRunCommandKind.START)
            elif snapshot.status is not LogicalRunStatus.ACTIVE:
                raise SharedAllocationStateError("only a queued run can be submitted")
            return self._control_snapshot(normalized)

    def heartbeat(self, run_id: str, token: str) -> SharedRunControlSnapshot:
        """record liveness for one active run without touching siblings."""

        with self._lock:
            normalized = self._authorize(run_id, token)
            self._require_active_allocation()
            snapshot = self._bundle.run_snapshot(normalized)
            if snapshot.status is not LogicalRunStatus.ACTIVE:
                raise SharedAllocationStateError("heartbeats require an active logical run")
            self._heartbeat_sequence[normalized] += 1
            self._last_heartbeat_at[normalized] = float(self._clock())
            return self._control_snapshot(normalized)

    def complete(self, run_id: str, token: str) -> SharedRunControlSnapshot:
        """complete one run and release the allocation only when it was last."""

        with self._lock:
            normalized = self._authorize(run_id, token)
            self._require_active_allocation()
            snapshot = self._bundle.run_snapshot(normalized)
            if snapshot.status is LogicalRunStatus.DONE:
                return self._control_snapshot(normalized)
            if snapshot.status is LogicalRunStatus.ACTIVE:
                self._bundle.transition_run(normalized, LogicalRunStatus.FINISHING)
            elif snapshot.status is not LogicalRunStatus.FINISHING:
                raise SharedAllocationStateError("only an active run can complete")
            self._bundle.transition_run(normalized, LogicalRunStatus.DONE)
            result = self._control_snapshot(normalized)
            self._release_if_terminal()
            return result

    def fail(self, run_id: str, token: str, error: str) -> SharedRunControlSnapshot:
        """fail one run, free its slot, and keep live siblings on the allocation."""

        with self._lock:
            normalized = self._authorize(run_id, token)
            self._require_active_allocation()
            snapshot = self._bundle.run_snapshot(normalized)
            if snapshot.status is LogicalRunStatus.FAILED:
                return self._control_snapshot(normalized)
            if snapshot.status in _TERMINAL_RUN_STATUSES:
                raise SharedAllocationStateError("a terminal logical run cannot fail again")
            was_started = snapshot.status in {
                LogicalRunStatus.ACTIVE,
                LogicalRunStatus.FINISHING,
            }
            self._bundle.transition_run(normalized, LogicalRunStatus.FAILED, error=error)
            if was_started:
                self._enqueue(normalized, SharedRunCommandKind.CANCEL)
            result = self._control_snapshot(normalized)
            self._release_if_terminal()
            return result

    def cancel(self, run_id: str, token: str) -> SharedRunControlSnapshot:
        """cancel exactly one authorized run and enqueue only its worker command."""

        with self._lock:
            normalized = self._authorize(run_id, token)
            self._require_active_allocation()
            snapshot = self._bundle.run_snapshot(normalized)
            if snapshot.status is LogicalRunStatus.CANCELLED:
                return self._control_snapshot(normalized)
            if snapshot.status in _TERMINAL_RUN_STATUSES:
                raise SharedAllocationStateError("a terminal logical run cannot be cancelled")
            self._bundle.transition_run(normalized, LogicalRunStatus.CANCELLED)
            self._enqueue(normalized, SharedRunCommandKind.CANCEL)
            result = self._control_snapshot(normalized)
            self._release_if_terminal()
            return result

    def status(self, run_id: str, token: str) -> SharedRunControlSnapshot:
        with self._lock:
            normalized = self._authorize(run_id, token)
            return self._control_snapshot(normalized)

    def poll_commands(self, run_id: str, token: str) -> tuple[SharedRunCommand, ...]:
        """return and acknowledge commands for exactly one authenticated run."""

        with self._lock:
            normalized = self._authorize(run_id, token)
            commands = tuple(self._commands[normalized])
            self._commands[normalized].clear()
            return commands

    def drain(self) -> None:
        """cancel every remaining run and release the one allocation."""

        with self._lock:
            if self._allocation_state is SharedAllocationState.RELEASED:
                self._drained = True
                return
            self._require_active_allocation()
            self._drained = True
            for snapshot in self._bundle.run_snapshots():
                if snapshot.status in _TERMINAL_RUN_STATUSES:
                    continue
                self._bundle.transition_run(snapshot.run_id, LogicalRunStatus.CANCELLED)
                self._enqueue(snapshot.run_id, SharedRunCommandKind.CANCEL)
            self._release_if_terminal()

    def release_if_terminal(self) -> bool:
        """retry a release after all runs are terminal."""

        with self._lock:
            return self._release_if_terminal()

    @staticmethod
    def _capability_digest(token: str) -> bytes:
        return hashlib.sha256(token.encode()).digest()

    def _authorize(self, run_id: str, token: str) -> str:
        normalized = str(run_id).strip()
        candidate = (
            self._capability_digest(token) if isinstance(token, str) else _DUMMY_CAPABILITY_DIGEST
        )
        expected = self._capability_digests.get(normalized, _DUMMY_CAPABILITY_DIGEST)
        if not secrets.compare_digest(candidate, expected) or normalized not in self._capabilities:
            raise SharedRunAuthenticationError("shared run capability rejected")
        return normalized

    def _require_active_allocation(self) -> None:
        if self._allocation_state is not SharedAllocationState.ACTIVE:
            raise SharedAllocationStateError("shared bundle allocation is not active")

    def _enqueue(self, run_id: str, kind: SharedRunCommandKind) -> None:
        self._command_sequence += 1
        self._commands[run_id].append(SharedRunCommand(self._command_sequence, kind))

    def _control_snapshot(self, run_id: str) -> SharedRunControlSnapshot:
        snapshot: BundleRunSnapshot = self._bundle.run_snapshot(run_id)
        return SharedRunControlSnapshot(
            run_id=run_id,
            status=snapshot.status,
            error=snapshot.error,
            heartbeat_sequence=self._heartbeat_sequence[run_id],
            last_heartbeat_at=self._last_heartbeat_at[run_id],
        )

    def _release_if_terminal(self) -> bool:
        if self._allocation_state is SharedAllocationState.RELEASED:
            return False
        if self._allocation_state is not SharedAllocationState.ACTIVE:
            return False
        if any(
            snapshot.status not in _TERMINAL_RUN_STATUSES
            for snapshot in self._bundle.run_snapshots()
        ):
            return False
        handle = self._handle
        if handle is None:
            raise SharedAllocationStateError("active shared allocation has no handle")
        self._allocation_state = SharedAllocationState.RELEASING
        try:
            self._backend.release(handle, self.bundle_id)
        except Exception:
            self._allocation_state = SharedAllocationState.ACTIVE
            raise
        self._allocation_state = SharedAllocationState.RELEASED
        return True
