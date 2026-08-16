"""What the deployment service is allowed to talk to, and how it refuses.

The service orchestrates a deployment without knowing it is being driven over HTTP, so it raises
the errors below instead of `HTTPException` and reaches its collaborators through the protocols
below instead of importing them. Nothing in `flash.server.domain.deployment*` may import FastAPI:
the route module is the only place a status code is chosen.

`detail` is carried verbatim onto the HTTP response. Most are plain strings; three of them
(`alias_reconciliation_failed`, `deployment_job_unavailable`, `deployment_revocation_failed`) are
structured dicts the CLI matches on, so the type stays open.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Protocol


class DeploymentError(Exception):
    """A refusal the route translates into a status code."""

    def __init__(self, detail: str | dict) -> None:
        super().__init__(detail if isinstance(detail, str) else str(detail))
        self.detail = detail


class InvalidDeploymentRequest(DeploymentError):
    """The request could not be honoured as written. -> 400"""


class DeploymentNotFound(DeploymentError):
    """The run, checkpoint, or artifact named does not exist. -> 404"""


class DeploymentConflict(DeploymentError):
    """The run or deployment is not in a state that allows this. -> 409"""


class DeploymentUpstreamError(DeploymentError):
    """Serving or the Hub answered, and the answer was a failure. -> 502"""


class DeploymentUnavailable(DeploymentError):
    """This plane cannot run the operation right now; retry is meaningful. -> 503"""


class RunRepository(Protocol):
    """Loading run status, with the authorization checks that must run inside the deploy lock."""

    deployable_states: frozenset[str]

    def manageable_run(
        self, run_id: str, key: dict, org_id: str | None, project_id: str | None
    ) -> Any: ...

    def owned_run(self, run_id: str, key: dict) -> Any: ...

    def get_status(self, run_id: str) -> Any: ...

    def all_runs(self) -> Iterable[dict]: ...

    def runs_for_key(self, key_id: str) -> Iterable[dict]: ...


class DeploymentRepository(Protocol):
    """The persisted deployment record and the verified-revision ledger it commits against."""

    def mark_pending(
        self,
        run_id: str,
        deployment: dict,
        *,
        expect_state: str | None = None,
        owner_deployment: dict | None = None,
    ) -> Any: ...

    def mark_deployed(
        self,
        run_id: str,
        deployment: dict,
        *,
        expect_state: str | None,
        verification_generation: Any,
    ) -> Any: ...

    def mark_checkpoint_deployed(
        self,
        run_id: str,
        deployment: dict,
        *,
        expect_state: str | None = None,
        verification_generation: Any,
    ) -> Any: ...

    def mark_failed(self, run_id: str, deployment: dict) -> Any: ...

    def mark_revocation_failed(self, run_id: str, error: str) -> Any: ...

    def mark_undeployed(self, run_id: str) -> Any: ...

    def verification_generation(self, run_id: str) -> Any: ...

    def verified_revisions(self, run_id: str) -> Sequence[str]: ...


class ArtifactRepository(Protocol):
    """Where a run's adapter artifacts live and which of them are deployable."""

    def list_checkpoints(self, spec: Any) -> Sequence[dict]: ...

    def adapter_prefix(self, spec: Any) -> str: ...

    def checkpoint_adapter_prefix(self, spec: Any, step: int) -> str: ...

    def export_adapter(
        self,
        *,
        source_repo: str,
        source_subfolder: str,
        dest_repo: str,
        dest_token: str,
        private: bool,
        base_model: str,
        base_model_revision: str | None,
    ) -> str: ...


class ServingGateway(Protocol):
    """The serving backend: revision registration, alias activation, revocation, inference."""

    def deployment_record(
        self,
        *,
        run_id: str,
        model: str,
        adapter_prefix: str,
        state: str,
        checkpoint_step: int | None,
    ) -> Any: ...

    def deploy_adapter(
        self, before_activate: Callable[[str, str], None] | None = None, **kwargs: Any
    ) -> Any: ...

    def undeploy_adapter(self, run_id: str) -> dict: ...

    def adapter_alias_target(self, run_id: str) -> str | None: ...

    def chat(self, **kwargs: Any) -> dict: ...

    def chat_stream(self, **kwargs: Any) -> Iterator[bytes]: ...


class DeploymentJobRunner(Protocol):
    """Runs the part of a deployment that outlives the HTTP response."""

    def start(self, target: Callable[..., object], *args: Any, **kwargs: Any) -> bool:
        """True when it ran synchronously; raises `DeploymentJobStartError` when it never ran."""
        ...


class DeploymentReporter(Protocol):
    """Mirrors a persisted status to the platform backend. Never called before the write lands."""

    def report_transition(self, previous: Any, current: Any, *, persisted: bool) -> None: ...

    def report(self, status: Any) -> None: ...

    def report_sequentially(self, status: Any) -> None:
        """Mirror one status and do not return until it has been handed over.

        Startup replay uses this instead of `report`. Delivering a historical backlog through the
        shared asynchronous reporter would queue it ahead of live updates, so the replay must block
        per item even where the live path does not.
        """
        ...


class DeploymentLockProvider(Protocol):
    """Returns the per-run mutex shared with undeploy, export, cancellation, and recovery.

    It must return the SAME object those paths take, so a deploy in flight actually blocks them.
    """

    def __call__(self, run_id: str) -> Any: ...
