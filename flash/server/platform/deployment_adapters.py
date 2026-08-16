"""Concrete collaborators for the deployment service, over the real repositories and serving.

Each adapter is a thin binding onto an existing functional boundary.

Most bind straight to the module that owns the function. The exceptions go through
`flash.server.app` at call time (``_app.<name>``) because the control plane's own tests and
operators patch `app.<name>` to stand in a fake backend, and a by-value import would not see that
patch. Route a call through `_app` only when something actually patches it there; otherwise import
from the canonical module, so the indirection marks a real seam instead of decorating every call.

This is the only place in the deployment path that knows those module-level seams exist; the
service and the lifecycle see nothing but the protocols.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any

from flash.server.domain.deployment_ports import (
    ArtifactRepository,
    DeploymentReporter,
    DeploymentRepository,
    RunRepository,
    ServingGateway,
)
from flash.server.domain.deployments import DeploymentService
from flash.server.platform.deployment_jobs import ThreadDeploymentJobRunner


class AppRunRepository(RunRepository):
    """Run status plus the authorization loads that must happen inside the deploy lock."""

    @property
    def deployable_states(self) -> frozenset[str]:
        from flash.server.app import _DEPLOYABLE_STATES

        return frozenset(_DEPLOYABLE_STATES)

    def manageable_run(self, run_id: str, key: dict, org_id, project_id) -> Any:
        from flash.server.platform.deps import manageable_run

        return manageable_run(run_id, key, org_id, project_id)

    def owned_run(self, run_id: str, key: dict) -> Any:
        from flash.server.platform.deps import owned_run

        return owned_run(run_id, key)

    def get_status(self, run_id: str) -> Any:
        from flash.server import app as _app

        return _app.get_status(run_id)

    def all_runs(self) -> Iterable[dict]:
        from flash.server.platform import db

        return db.all_runs()

    def runs_for_key(self, key_id: str) -> Iterable[dict]:
        from flash.server.platform import db

        return db.runs_for_key(key_id)


class RunnerDeploymentRepository(DeploymentRepository):
    """The persisted deployment record and the verified-revision ledger it commits against."""

    def mark_pending(
        self, run_id: str, deployment: dict, *, expect_state=None, owner_deployment=None
    ) -> Any:
        from flash.runner import mark_deployment_pending

        return mark_deployment_pending(
            run_id, deployment, expect_state=expect_state, owner_deployment=owner_deployment
        )

    def mark_deployed(
        self, run_id: str, deployment: dict, *, expect_state, verification_generation
    ) -> Any:
        from flash.runner import mark_deployed

        return mark_deployed(
            run_id,
            deployment,
            expect_state=expect_state,
            verification_generation=verification_generation,
        )

    def mark_checkpoint_deployed(
        self, run_id: str, deployment: dict, *, expect_state=None, verification_generation
    ) -> Any:
        from flash.runner import mark_checkpoint_deployed

        return mark_checkpoint_deployed(
            run_id,
            deployment,
            expect_state=expect_state,
            verification_generation=verification_generation,
        )

    def mark_failed(self, run_id: str, deployment: dict) -> Any:
        from flash.runner import mark_deployment_failed

        return mark_deployment_failed(run_id, deployment)

    def mark_revocation_failed(self, run_id: str, error: str) -> Any:
        from flash.runner import mark_deployment_revocation_failed

        return mark_deployment_revocation_failed(run_id, error)

    def mark_undeployed(self, run_id: str) -> Any:
        from flash.runner import mark_undeployed

        return mark_undeployed(run_id)

    def verification_generation(self, run_id: str) -> Any:
        from flash.runner import verified_adapter_revision_generation

        return verified_adapter_revision_generation(run_id)

    def verified_revisions(self, run_id: str) -> Sequence[str]:
        from flash.runner import read_verified_adapter_revisions

        return read_verified_adapter_revisions(run_id)


class HubArtifactRepository(ArtifactRepository):
    """Where a run's adapter artifacts live, and copying one out to a user-owned repo."""

    def list_checkpoints(self, spec: Any) -> Sequence[dict]:
        from flash.server import app as _app

        return _app.list_checkpoints(spec)

    def adapter_prefix(self, spec: Any) -> str:
        from flash.runner import adapter_prefix

        return adapter_prefix(spec)

    def checkpoint_adapter_prefix(self, spec: Any, step: int) -> str:
        from flash.runner.results.checkpoints import checkpoint_adapter_prefix

        return checkpoint_adapter_prefix(spec, step)

    def export_adapter(self, **kwargs: Any) -> str:
        from flash.server import app as _app

        return _app.export_adapter(**kwargs)


class RemoteServingGateway(ServingGateway):
    """The serving backend: revision registration, alias activation, revocation, inference."""

    def deployment_record(self, *, run_id, model, adapter_prefix, state, checkpoint_step) -> Any:
        from flash.serve.deploy import deployment_record

        return deployment_record(
            run_id=run_id,
            model=model,
            adapter_prefix=adapter_prefix,
            state=state,
            checkpoint_step=checkpoint_step,
        )

    def deploy_adapter(self, before_activate=None, **kwargs: Any) -> Any:
        from flash.server import app as _app

        if before_activate is None:
            return _app.deploy_adapter(**kwargs)
        return _app.deploy_adapter(**kwargs, before_activate=before_activate)

    def undeploy_adapter(self, run_id: str) -> dict:
        from flash.server import app as _app

        return _app.undeploy_adapter(run_id)

    def adapter_alias_target(self, run_id: str) -> str | None:
        from flash.server import app as _app

        return _app.adapter_alias_target(run_id)

    def chat(self, **kwargs: Any) -> dict:
        from flash.server import app as _app

        return _app.serve_chat(**kwargs)

    def chat_stream(self, **kwargs: Any):
        from flash.server import app as _app

        return _app.serve_chat_stream(**kwargs)


class StatusReporter(DeploymentReporter):
    """Mirrors a persisted status to the platform backend, never before the write landed."""

    def report_transition(self, previous, current, *, persisted: bool) -> None:
        if not persisted or (
            previous.state == current.state and previous.deployment == current.deployment
        ):
            return
        self.report(current)

    def report(self, status) -> None:
        from flash.runner import _report_status, _report_status_async

        if os.environ.get("FLASH_DEPLOY_SYNC") == "1":
            _report_status(status)
        else:
            _report_status_async(status)

    def report_sequentially(self, status) -> None:
        from flash.runner import _report_status

        _report_status(status)


def build_deployment_service(*, jobs: ThreadDeploymentJobRunner | None = None) -> DeploymentService:
    """The production service: real repositories, real serving, the shared per-run lock."""
    from flash.server.platform.locks import _deploy_lock

    return DeploymentService(
        runs=AppRunRepository(),
        deployments=RunnerDeploymentRepository(),
        artifacts=HubArtifactRepository(),
        serving=RemoteServingGateway(),
        jobs=jobs or ThreadDeploymentJobRunner(),
        reporter=StatusReporter(),
        # MUST be this exact function: undeploy, export, cancellation teardown, and startup
        # recovery all contend on the object it returns. A private lock here would let a deploy
        # and a cancel run at the same time.
        lock_provider=_deploy_lock,
    )
