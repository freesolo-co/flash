"""Fake collaborators for driving `DeploymentService` without a serving backend or a real store.

These are deliberately literal: they record what they were asked to do and return what they were
told to return, so a test can assert on call ORDER and on the exact arguments a real collaborator
would have received. Nothing here re-implements the service's decisions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from flash.server.domain.deployments import CallerContext, DeploymentService
from flash.server.platform.deployment_jobs import DeploymentJobStartError

CALLER = CallerContext(key={"id": "key-1", "org_id": "org-test"}, org_id=None, project_id=None)


class FakeLock:
    """The per-run mutex, with the acquire/release history the ownership tests assert on."""

    def __init__(self, *, acquirable: bool = True) -> None:
        self.acquirable = acquirable
        self.held = False
        self.events: list[str] = []
        self._lock = threading.Lock()

    def acquire(self, blocking: bool = True) -> bool:
        if not self.acquirable:
            self.events.append("acquire-failed")
            return False
        self._lock.acquire()
        self.held = True
        self.events.append("acquire")
        return True

    def release(self) -> None:
        if not self.held:
            raise RuntimeError("deploy lock is not held")
        self.held = False
        self.events.append("release")
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class FakeLockProvider:
    """Hands out one lock per run id, so every caller for a run contends on the same object."""

    def __init__(self, *, acquirable: bool = True) -> None:
        self.acquirable = acquirable
        self.locks: dict[str, FakeLock] = {}

    def __call__(self, run_id: str) -> FakeLock:
        if run_id not in self.locks:
            self.locks[run_id] = FakeLock(acquirable=self.acquirable)
        return self.locks[run_id]


def FakeStatus(
    run_id: str = "flash-1-abcd",
    state: str = "done",
    spec: dict | None = None,
    deployment: Any = None,
    **extra: Any,
):
    """A real ``RunStatus``.

    Deliberately not a hand-built double: the service reads fields (``effective_preparation``,
    ``to_dict``) that a stand-in would have to re-implement, and a stand-in that drifts from the
    real record passes while production breaks.
    """
    from flash.runner import RunStatus

    return RunStatus(run_id=run_id, state=state, spec=spec or {}, deployment=deployment, **extra)


class FakeRunRepository:
    deployable_states = frozenset({"done", "deployed"})

    def __init__(self, status: Any = None) -> None:
        self.status = status if status is not None else FakeStatus()
        # each get_status returns the NEXT queued status when one is queued, so a test can walk a
        # record through the states a real store would have moved it through.
        self.status_queue: list[Any] = []
        self.calls: list[str] = []
        self.rows: list[dict] = []

    def manageable_run(self, run_id, key, org_id, project_id):
        self.calls.append("manageable_run")
        return self.status

    def owned_run(self, run_id, key):
        self.calls.append("owned_run")
        return self.status

    def get_status(self, run_id):
        self.calls.append("get_status")
        if self.status_queue:
            return self.status_queue.pop(0)
        return self.status

    def all_runs(self):
        return list(self.rows)

    def runs_for_key(self, key_id):
        return list(self.rows)


class FakeDeploymentRepository:
    """Records every write and echoes it back as the persisted record by default."""

    def __init__(self, *, generation: object = 1) -> None:
        self.writes: list[tuple[str, dict]] = []
        self.generation = generation
        self.revisions: list[str] = []
        # when set, the next write of this kind returns this status instead of an echo, which is
        # how a test simulates a lost CAS.
        self.results: dict[str, Any] = {}

    def first_write(self, kind: str) -> dict:
        """The first record written under `kind`. Fails the lookup loudly if there was none."""
        return next(record for written, record in self.writes if written == kind)

    def _echo(self, kind: str, run_id: str, deployment: dict, state: str = "done"):
        self.writes.append((kind, deployment))
        if kind in self.results:
            return self.results[kind]
        return FakeStatus(run_id=run_id, state=state, deployment=deployment)

    def mark_pending(self, run_id, deployment, *, expect_state=None, owner_deployment=None):
        self.writes.append(
            (
                "mark_pending_args",
                {"expect_state": expect_state, "owner_deployment": owner_deployment},
            )
        )
        return self._echo("mark_pending", run_id, deployment)

    def mark_deployed(self, run_id, deployment, *, expect_state, verification_generation):
        self.writes.append(
            (
                "mark_deployed_args",
                {"expect_state": expect_state, "verification_generation": verification_generation},
            )
        )
        return self._echo("mark_deployed", run_id, deployment, state="deployed")

    def mark_checkpoint_deployed(
        self, run_id, deployment, *, expect_state=None, verification_generation
    ):
        self.writes.append(
            (
                "mark_checkpoint_deployed_args",
                {"expect_state": expect_state, "verification_generation": verification_generation},
            )
        )
        return self._echo("mark_checkpoint_deployed", run_id, deployment)

    def mark_failed(self, run_id, deployment):
        return self._echo("mark_failed", run_id, deployment)

    def mark_revocation_failed(self, run_id, error):
        return self._echo(
            "mark_revocation_failed", run_id, {"state": "revocation_failed", "error": error}
        )

    def mark_undeployed(self, run_id):
        return self._echo("mark_undeployed", run_id, {"state": "undeployed"})

    def verification_generation(self, run_id):
        return self.generation

    def verified_revisions(self, run_id):
        return list(self.revisions)

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.writes if not kind.endswith("_args")]


class FakeArtifactRepository:
    def __init__(
        self, checkpoints: list[dict] | None = None, export_url: str = "https://hf/x"
    ) -> None:
        self.checkpoints = checkpoints or []
        self.export_url = export_url
        self.exports: list[dict] = []

    def list_checkpoints(self, spec):
        return list(self.checkpoints)

    def adapter_prefix(self, spec):
        return "sft/run/seed0"

    def checkpoint_adapter_prefix(self, spec, step):
        return f"sft/run/seed0/step-{step}"

    def export_adapter(self, **kwargs):
        self.exports.append(kwargs)
        return self.export_url


@dataclass
class FakeDeployment:
    """Stands in for `flash.serve.deploy.Deployment`."""

    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.data)


class FakeServingGateway:
    """Records the deployment calls and lets a test drive activation and smoke outcomes."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.deploy_kwargs: list[dict] = []
        self.chat_calls: list[dict] = []
        self.alias_target: str | None = None
        # raised BEFORE the activation hook: registration and artifact failures, which happen
        # while the previous alias is still the live one.
        self.deploy_error: Exception | None = None
        # raised AFTER the hook: the smoke ran and the activation itself failed or was ambiguous.
        self.activation_error: Exception | None = None
        self.undeploy_error: Exception | None = None
        self.chat_result: dict = {"choices": [{"message": {"content": "4"}}]}
        self.activation: tuple[str, str] = ("flash-1-abcd@final." + "a" * 40, "flash-1-abcd")
        self.record_state: dict = {}

    def deployment_record(self, *, run_id, model, adapter_prefix, state, checkpoint_step):
        self.calls.append("deployment_record")
        return FakeDeployment(
            {
                "run_id": run_id,
                "model": model,
                "state": state,
                "checkpoint_step": checkpoint_step,
                "adapter_hf_prefix": f"{adapter_prefix}/adapter",
                "openai_model": run_id,
                **self.record_state,
            }
        )

    def deploy_adapter(self, before_activate=None, **kwargs):
        self.calls.append("deploy_adapter")
        self.deploy_kwargs.append(kwargs)
        if self.deploy_error is not None:
            raise self.deploy_error
        if kwargs.get("dry_run"):
            # mirrors the real gateway: a dry run returns the record and never activates, so it
            # must not run the caller's before_activate hook either.
            return FakeDeployment({"run_id": kwargs.get("run_id"), "state": "dry_run"})
        if before_activate is not None:
            before_activate(*self.activation)
        if self.activation_error is not None:
            raise self.activation_error
        return FakeDeployment({"adapter_revision": self.activation[0], "state": "ready"})

    def undeploy_adapter(self, run_id):
        self.calls.append("undeploy_adapter")
        if self.undeploy_error is not None:
            raise self.undeploy_error
        return {"disabled_aliases": 1, "disabled_revisions": 2, "serving_deregistered": True}

    def adapter_alias_target(self, run_id):
        self.calls.append("adapter_alias_target")
        return self.alias_target

    def chat(self, **kwargs):
        self.calls.append("chat")
        self.chat_calls.append(kwargs)
        return self.chat_result

    def chat_stream(self, **kwargs):
        self.calls.append("chat_stream")
        self.chat_calls.append(kwargs)
        return iter([b"chunk"])


class FakeJobRunner:
    """Captures the job instead of running it, unless told to run it inline."""

    def __init__(self, *, run_inline: bool = False, start_error: Exception | None = None) -> None:
        self.run_inline = run_inline
        self.start_error = start_error
        self.started: list[dict] = []

    def start(self, target, *args, **kwargs) -> bool:
        if self.start_error is not None:
            raise self.start_error
        self.started.append(kwargs)
        if self.run_inline:
            target(*args, **kwargs)
            return True
        return False


class FakeReporter:
    def __init__(self) -> None:
        self.transitions: list[tuple[Any, Any, bool]] = []
        self.reports: list[Any] = []
        self.sequential_reports: list[Any] = []

    def report_transition(self, previous, current, *, persisted: bool) -> None:
        self.transitions.append((previous, current, persisted))
        if persisted:
            self.reports.append(current)

    def report(self, status) -> None:
        self.reports.append(status)

    def report_sequentially(self, status) -> None:
        self.sequential_reports.append(status)
        self.reports.append(status)

    def persisted_reports(self) -> list[Any]:
        return list(self.reports)


@dataclass
class Harness:
    """A service plus every fake it was built from."""

    service: DeploymentService
    runs: FakeRunRepository
    deployments: FakeDeploymentRepository
    artifacts: FakeArtifactRepository
    serving: FakeServingGateway
    jobs: FakeJobRunner
    reporter: FakeReporter
    locks: FakeLockProvider

    def lock_for(self, run_id: str) -> FakeLock:
        return self.locks(run_id)


def build_harness(
    *,
    status: Any = None,
    checkpoints: list[dict] | None = None,
    run_inline: bool = False,
    start_error: Exception | None = None,
    acquirable: bool = True,
) -> Harness:
    runs = FakeRunRepository(status)
    deployments = FakeDeploymentRepository()
    artifacts = FakeArtifactRepository(checkpoints)
    serving = FakeServingGateway()
    jobs = FakeJobRunner(run_inline=run_inline, start_error=start_error)
    reporter = FakeReporter()
    locks = FakeLockProvider(acquirable=acquirable)
    service = DeploymentService(
        runs=runs,
        deployments=deployments,
        artifacts=artifacts,
        serving=serving,
        jobs=jobs,
        reporter=reporter,
        lock_provider=locks,
    )
    return Harness(
        service=service,
        runs=runs,
        deployments=deployments,
        artifacts=artifacts,
        serving=serving,
        jobs=jobs,
        reporter=reporter,
        locks=locks,
    )


__all__ = [
    "CALLER",
    "DeploymentJobStartError",
    "FakeArtifactRepository",
    "FakeDeployment",
    "FakeDeploymentRepository",
    "FakeJobRunner",
    "FakeLock",
    "FakeLockProvider",
    "FakeReporter",
    "FakeRunRepository",
    "FakeServingGateway",
    "FakeStatus",
    "Harness",
    "build_harness",
]
