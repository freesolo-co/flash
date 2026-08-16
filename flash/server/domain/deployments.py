"""The deployment application service.

Every deployment operation enters here as a command and leaves as data. The service owns the
per-run lock, the authorization loads that must happen inside it, validation, persistence, and the
decision to hand the rest of the attempt to a background job. It knows nothing about HTTP: it
raises the domain errors in `deployment_ports` and the route chooses a status code.

The lock is the subtle part. `deploy` acquires it non-blocking and, once the background job has
started, TRANSFERS ownership to that job, which releases it when the lifecycle ends. A job that
never started leaves ownership here, and this method releases it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, NoReturn

from flash._internal.channel import CLI_NAME
from flash.core.spec import JobSpec
from flash.serve.deploy import ServingError
from flash.server.domain import deployment_revisions as revisions
from flash.server.domain.deployment_lifecycle import DeploymentLifecycle
from flash.server.domain.deployment_ports import (
    ArtifactRepository,
    DeploymentConflict,
    DeploymentJobRunner,
    DeploymentLockProvider,
    DeploymentNotFound,
    DeploymentReporter,
    DeploymentRepository,
    DeploymentUnavailable,
    DeploymentUpstreamError,
    InvalidDeploymentRequest,
    RunRepository,
    ServingGateway,
)
from flash.server.domain.deployment_records import (
    deployment_attempt_is_stale,
    deployment_failure_persisted,
    deployment_state,
    public_deployment_view,
)
from flash.server.platform.deployment_jobs import DeploymentJobStartError

DEPLOYMENT_STALE_SECONDS = 40 * 60


@dataclass(frozen=True)
class CallerContext:
    """Who is asking, and on whose behalf."""

    key: dict
    org_id: str | None = None
    project_id: str | None = None


@dataclass(frozen=True)
class DeployCommand:
    run_id: str
    caller: CallerContext
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExportCommand:
    run_id: str
    caller: CallerContext
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChatCommand:
    run_id: str
    caller: CallerContext
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExportResult:
    run_id: str
    repository: str
    url: str
    checkpoint_step: int | None
    is_checkpoint: bool
    status: Any

    def to_dict(self) -> dict:
        result = {
            "run_id": self.run_id,
            "adapter_id": self.run_id,
            "repository": self.repository,
            "url": self.url,
            "source": (
                f"{self.run_id}/step-{self.checkpoint_step}" if self.is_checkpoint else self.run_id
            ),
        }
        if self.is_checkpoint:
            result["step"] = self.checkpoint_step
        return result


@dataclass(frozen=True)
class ChatPlan:
    """A validated chat request, resolved to the exact revision that will serve it."""

    serving_model: str
    messages: list[dict]
    temperature: float
    max_tokens: int
    thinking: bool
    stop: list[str] | None
    stream: bool


class DeploymentService:
    """Commands over a run's deployment: read, deploy, undeploy, export, list, chat."""

    def __init__(
        self,
        *,
        runs: RunRepository,
        deployments: DeploymentRepository,
        artifacts: ArtifactRepository,
        serving: ServingGateway,
        jobs: DeploymentJobRunner,
        reporter: DeploymentReporter,
        lock_provider: DeploymentLockProvider,
        lifecycle: DeploymentLifecycle | None = None,
        stale_seconds: float = DEPLOYMENT_STALE_SECONDS,
    ) -> None:
        self._runs = runs
        self._deployments = deployments
        self._artifacts = artifacts
        self._serving = serving
        self._jobs = jobs
        self._reporter = reporter
        self._lock_provider = lock_provider
        self._stale_seconds = stale_seconds
        self.lifecycle = lifecycle or DeploymentLifecycle(
            runs=runs,
            deployments=deployments,
            artifacts=artifacts,
            serving=serving,
            reporter=reporter,
            lock_provider=lock_provider,
        )

    # ---- reads -------------------------------------------------------------------------

    def get_deployment(self, run_id: str, caller: CallerContext) -> dict:
        status = self._manageable(run_id, caller)
        persisted = (
            status.deployment if isinstance(status.deployment, dict) else {"state": "undeployed"}
        )
        return public_deployment_view({**persisted, "run_id": run_id})

    def list_deployments(self, caller: CallerContext, *, scope: tuple[str, str] | None) -> list:
        out = []
        for row in self._runs.runs_for_key(caller.key["id"]):
            try:
                status = self._runs.get_status(row["run_id"])
            except FileNotFoundError:
                continue
            if scope is not None and not self._in_listing_scope(status, *scope):
                continue
            if status.deployment and status.deployment.get("state") not in (
                "undeployed",
                "dry_run",
            ):
                data = status.to_dict()
                data["deployment"] = public_deployment_view(data["deployment"])
                out.append(data)
        return out

    @staticmethod
    def _in_listing_scope(status, org: str, project: str) -> bool:
        """Whether a run belongs to the requested org AND project (manageable_run's predicate)."""
        from flash.core.spec import require_project_id
        from flash.runner import _status_org_id

        if _status_org_id(status) != org:
            return False
        persisted_project = status.spec.get("project") if isinstance(status.spec, dict) else None
        try:
            return require_project_id(persisted_project) == project
        except (TypeError, ValueError):
            return False

    # ---- deploy ------------------------------------------------------------------------

    def deploy(self, command: DeployCommand) -> dict:
        run_id = command.run_id
        deploy_lock = self._lock_provider(run_id)
        if not deploy_lock.acquire(blocking=False):
            self._reject_contended_deploy(run_id, command.caller)
        # a list, not a return value: ownership must already have moved when an UNEXPECTED
        # exception unwinds out of the body. the job may have started, and releasing a lock it
        # is about to release again would corrupt the mutex for every later deploy.
        ownership: list[bool] = [False]
        try:
            return self._deploy_locked(command, deploy_lock, ownership)
        finally:
            if not ownership[0]:
                deploy_lock.release()

    def _deploy_locked(self, command: DeployCommand, deploy_lock, ownership: list[bool]) -> dict:
        """The deploy body, under the lock. Sets ``ownership[0]`` when the job takes the lock."""
        run_id = command.run_id
        payload = command.payload
        status = self._manageable(run_id, command.caller)
        spec = JobSpec.from_dict(status.spec)
        dry_run = _require_bool(payload, "dry_run", False)
        effective_spec, current_deployment = self._validate_deploy_request(
            run_id, status, spec, payload, dry_run
        )
        checkpoint_step, is_checkpoint, deploy_prefix = revisions.resolve_deployable_target(
            run_id,
            effective_spec,
            status,
            payload.get("step"),
            action="deploy",
            enforce_state=not dry_run,
            artifacts=self._artifacts,
            runs=self._runs,
        )
        if not dry_run and not effective_spec.train.hf_repo:
            raise DeploymentConflict(
                f"run {run_id} has no [train].hf_repo; its adapter artifacts "
                "cannot be located, so it cannot be deployed"
            )
        # CAS guard: /cancel's worker + provider teardown runs outside this lock (only its status
        # write is lock-serialized), so capture state before deploy and re-verify it on the write.
        prev_state = status.state
        deploy_org_id = self._deploy_org_id(status, command.caller.key)
        self._require_deploy_org(run_id, deploy_org_id)
        previous_deployment = None
        expected_adapter_revision = None
        if not dry_run:
            try:
                expected_adapter_revision, previous_deployment = revisions.activation_predecessor(
                    run_id,
                    current_deployment,
                    serving=self._serving,
                    deployments=self._deployments,
                )
            except ServingError as exc:
                raise DeploymentUpstreamError(
                    {
                        "code": "alias_reconciliation_failed",
                        "run_id": run_id,
                        "retryable": True,
                        "message": str(exc),
                    }
                ) from exc
        deploy_kwargs = {
            "run_id": run_id,
            "model": spec.model,
            "hf_repo": effective_spec.train.hf_repo,
            "adapter_prefix": deploy_prefix,
            "dry_run": dry_run,
            "lora_rank": effective_spec.train.lora_rank,
            # a run trained with thinking serves with thinking (per-run parity)
            "thinking": spec.thinking,
            # a run trained with structured_outputs serves under the same grammar. thinking runs
            # are registered only after serving advertises deferred post-reasoning constraints.
            "structured_outputs": spec.train.structured_outputs,
            "org_id": deploy_org_id,
            "checkpoint_step": checkpoint_step,
            "expected_adapter_revision": expected_adapter_revision,
        }
        if dry_run:
            return self._dry_run(deploy_kwargs)

        dep_dict = self._queued_deployment_record(
            run_id,
            spec,
            effective_spec,
            deploy_prefix,
            checkpoint_step,
            is_checkpoint,
            current_deployment,
            previous_deployment,
        )
        marked = self._deployments.mark_pending(run_id, dep_dict, expect_state=prev_state)
        if marked.deployment != dep_dict:
            raise DeploymentConflict(f"run {run_id} became {marked.state!r} during deploy; aborted")
        self._reporter.report_transition(status, marked, persisted=True)

        job_kwargs = {
            "run_id": run_id,
            "spec_dict": status.spec,
            "is_checkpoint": is_checkpoint,
            "deploy_kwargs": deploy_kwargs,
            "deployment": dep_dict,
            "prev_state": prev_state,
            "deploy_lock": deploy_lock,
        }
        # the job owns the lock from here on and releases it when the lifecycle ends, so a
        # booting replica's non-blocking probe keeps failing for the whole deploy. a start
        # failure means the job never ran, so ownership reverts to this request.
        ownership[0] = True
        try:
            ran_sync = self._jobs.start(self.lifecycle.finish_locked, **job_kwargs)
        except DeploymentJobStartError as exc:
            ownership[0] = False
            self._record_unstarted_job(run_id, dep_dict, exc)
        if ran_sync:
            return public_deployment_view(self._runs.get_status(run_id).deployment or dep_dict)
        return public_deployment_view(dep_dict)

    def _dry_run(self, deploy_kwargs: dict) -> dict:
        """Validate and ask the gateway, but persist nothing, report nothing, activate nothing."""
        try:
            dep = self._serving.deploy_adapter(**deploy_kwargs)
        except ValueError as exc:
            raise InvalidDeploymentRequest(str(exc)) from exc
        return dep.to_dict()

    def _record_unstarted_job(self, run_id: str, dep_dict: dict, exc: Exception) -> NoReturn:
        error = f"deployment job could not start: {exc}"
        failed = deployment_state(
            dep_dict,
            "failed",
            error=error,
            detail="deployment was not started; retry when the control plane is available",
            retryable=True,
        )
        previous = self._runs.get_status(run_id)
        marked = self._deployments.mark_failed(run_id, failed)
        self._reporter.report_transition(
            previous, marked, persisted=deployment_failure_persisted(marked, failed)
        )
        raise DeploymentUnavailable(
            {
                "code": "deployment_job_unavailable",
                "run_id": run_id,
                "retryable": True,
                "message": error,
            }
        ) from exc

    def _reject_contended_deploy(self, run_id: str, caller: CallerContext) -> NoReturn:
        """Raise the conflict for a deploy that could not take the per-run lock.

        Always raises. The lock was NOT acquired on this path, so the caller must not release it.
        """
        status = self._manageable(run_id, caller)
        current_deployment = status.deployment or {}
        current_deployment_state = current_deployment.get("state")
        if current_deployment_state in revisions.DEPLOYMENT_BUSY_STATES:
            detail = (
                f"run {run_id} already has a deployment in "
                f"{current_deployment_state} state; run `flash models deployments` "
                "to check progress"
            )
        else:
            # the per-run lock is shared with undeploy, export, and startup recovery, so a
            # contended acquire cannot claim a deployment is running unless the state says so.
            detail = f"another operation is in progress for run {run_id}; retry shortly"
        raise DeploymentConflict(detail)

    def _validate_deploy_request(
        self, run_id: str, status, spec: JobSpec, payload: dict, dry_run: bool
    ) -> tuple[JobSpec, dict]:
        """Reject a deploy that cannot proceed, before anything is queued or registered.

        Returns the effective spec and the current deployment record for the caller to work from.
        """
        from flash.runner import _internal_spec_from_status, effective_spec_from_status

        # A pin the AUTHOR wrote before the config key was removed is a request serving cannot
        # honour, so it stays refused for persisted runs.
        #
        # A pin the RUNNER assigned is different. SFT is force-pinned by
        # `runner.submit.prepare_job` -> `_resolve_model_revision(required=True)` so workload
        # profiling keys on an immutable commit. Rejecting those made every SFT run, and every
        # adapter warm-started from one, permanently undeployable, which also blocks
        # `flash models chat` and `flash env eval`, since both require a deployment.
        #
        # provenance and the pin are read from the INTERNAL worker spec, not `spec`: to_dict()
        # strips both from new public specs. that includes a new child inheriting a legacy authored
        # source pin, which must remain rejected just like its parent. `_internal_spec_from_status`
        # falls back to the public spec for pre-upgrade runs without a worker snapshot, so
        # historical authored pins also fail closed.
        internal_spec = _internal_spec_from_status(status)
        if (
            spec.model_revision or internal_spec.model_revision
        ) and not internal_spec.model_revision_auto:
            raise InvalidDeploymentRequest(
                "deployment does not support this run's legacy revision-pinned base model; "
                f"submit a new run without the removed model_revision key, or use `{CLI_NAME} "
                "models export` to load the adapter from Hugging Face"
            )
        try:
            effective_spec = effective_spec_from_status(status)
        except ValueError as exc:
            raise DeploymentConflict(str(exc)) from exc
        # smoke verification is mandatory for every real deployment: a loadable-but-broken
        # revision must never become the bare-run alias target. reject an explicit opt-out
        # before anything is queued or registered (dry runs never register or activate, so
        # the flag is meaningless there too).
        if _require_bool(payload, "verify", True) is False:
            raise InvalidDeploymentRequest(
                "verify=false is not supported: deployment smoke verification is "
                "mandatory before alias activation"
            )
        current_deployment = status.deployment or {}
        current_deployment_state = current_deployment.get("state")
        completed_unknown_activation = (
            current_deployment_state == "reconciling"
            and current_deployment.get("activation_outcome_unknown") is True
        )
        if (
            not dry_run
            and current_deployment_state in revisions.DEPLOYMENT_BUSY_STATES
            and not completed_unknown_activation
            and not deployment_attempt_is_stale(
                current_deployment,
                self._stale_seconds,
                busy_states=revisions.DEPLOYMENT_BUSY_STATES,
            )
        ):
            raise DeploymentConflict(
                f"run {run_id} already has a deployment in "
                f"{current_deployment.get('state')} state; run `flash models deployments` "
                "to check progress"
            )
        return effective_spec, current_deployment

    def _queued_deployment_record(
        self,
        run_id: str,
        spec: JobSpec,
        effective_spec: JobSpec,
        deploy_prefix: str,
        checkpoint_step,
        is_checkpoint: bool,
        current_deployment: dict,
        previous_deployment,
    ) -> dict:
        """Build the ``queued`` deployment record that gets persisted before the job starts."""
        # Validate the cheap configured-rank part synchronously so obvious spec errors are refused
        # here instead of becoming background deployment failures.
        try:
            from flash.serve.deploy import validate_serving_lora_rank

            validate_serving_lora_rank(
                spec.model,
                effective_spec.train.lora_rank,
                rank_source="effective prepared LoRA rank",
            )
        except ValueError as exc:
            raise InvalidDeploymentRequest(str(exc)) from exc

        try:
            dep_dict = self._serving.deployment_record(
                run_id=run_id,
                model=spec.model,
                adapter_prefix=deploy_prefix,
                state="queued",
                checkpoint_step=checkpoint_step,
            ).to_dict()
        except ValueError as exc:
            raise InvalidDeploymentRequest(str(exc)) from exc
        dep_dict = deployment_state(
            dep_dict,
            "queued",
            detail="deployment queued",
            verify=True,
            requested_at=time.time(),
        )
        dep_dict["verification_generation"] = self._deployments.verification_generation(run_id)
        if current_deployment.get("activation_outcome_unknown"):
            dep_dict["activation_outcome_unknown"] = True
        if is_checkpoint:
            dep_dict["checkpoint_step"] = checkpoint_step
        if previous_deployment:
            dep_dict["previous_deployment"] = previous_deployment
        return dep_dict

    @staticmethod
    def _deploy_org_id(status, key: dict) -> str | None:
        """Prefer the run's own org over the caller's key (operator deploys land on run's owner)."""
        from flash.server.platform.internal_client import run_org_id

        return run_org_id(status) or str(key.get("org_id") or "").strip() or None

    @staticmethod
    def _require_deploy_org(run_id: str, deploy_org_id: str | None) -> None:
        """Fail closed when a managed-plane deploy would register an adapter with no owning org.

        Serving authorizes external chat requests against the org that owns the adapter (see
        platform docs), so registering a revision without an org would leave the field's
        enforcement to whatever the serving backend does with an unowned adapter. A standalone
        plane is single-tenant by definition and has no organization directory, so it keeps
        deploying without one (same escape hatch as ``deps.manageable_run``).
        """
        from flash.server.platform import auth

        if deploy_org_id is not None or auth.standalone():
            return
        raise DeploymentConflict(
            f"run {run_id} has no owning organization on record and the caller's key carries "
            "none; refusing to register an adapter without an org, because serving authorizes "
            "chat requests against the org that owns it. deploy with a key that belongs to the "
            "run's organization, or re-submit the run through the platform so it carries an "
            "org context."
        )

    # ---- undeploy ----------------------------------------------------------------------

    def undeploy(self, run_id: str, caller: CallerContext) -> dict:
        with self._lock_provider(run_id):
            status = self._manageable(run_id, caller)
            try:
                result = self._serving.undeploy_adapter(run_id)
            except ServingError as exc:
                self._record_revocation_failure(run_id, status, exc)
            marked = self._deployments.mark_undeployed(run_id)
            persisted = isinstance(marked.deployment, dict) and (
                marked.deployment.get("state") == "undeployed"
            )
            self._reporter.report_transition(status, marked, persisted=persisted)
            deployment = (
                marked.deployment
                if isinstance(marked.deployment, dict)
                else {"state": "undeployed"}
            )
            response = public_deployment_view({**deployment, "run_id": run_id})
            response.update(
                {
                    field_name: result[field_name]
                    for field_name in (
                        "disabled_aliases",
                        "disabled_revisions",
                        "serving_deregistered",
                    )
                    if field_name in result
                }
            )
            return response

    def _record_revocation_failure(self, run_id: str, status, exc: ServingError):
        marked = self._deployments.mark_revocation_failed(run_id, str(exc))
        persisted = isinstance(marked.deployment, dict) and (
            marked.deployment.get("state") == "revocation_failed"
            and marked.deployment.get("error") == str(exc)
        )
        self._reporter.report_transition(status, marked, persisted=persisted)
        raise DeploymentUpstreamError(
            {
                "code": "deployment_revocation_failed",
                "run_id": run_id,
                "retryable": True,
                "message": str(exc),
            }
        ) from exc

    # ---- export ------------------------------------------------------------------------

    def export(self, command: ExportCommand) -> ExportResult:
        from flash.runner import effective_spec_from_status

        run_id = command.run_id
        payload = command.payload
        with self._lock_provider(run_id):
            repository = _validated_repository(payload)
            hf_token = str(payload.get("hf_token") or "").strip()
            if not hf_token:
                raise InvalidDeploymentRequest(
                    "hf_token is required: a HuggingFace token with write access to the "
                    "destination repo"
                )
            private = _require_bool(payload, "private", True)

            status = self._runs.owned_run(run_id, command.caller.key)
            spec = JobSpec.from_dict(status.spec)
            try:
                effective_spec = effective_spec_from_status(status)
            except ValueError as exc:
                raise DeploymentConflict(str(exc)) from exc
            if not effective_spec.train.hf_repo:
                raise DeploymentConflict(
                    f"run {run_id} has no [train].hf_repo; its adapter artifacts "
                    "cannot be located, so it cannot be exported"
                )
            checkpoint_step, is_checkpoint, prefix = revisions.resolve_deployable_target(
                run_id,
                effective_spec,
                status,
                payload.get("step"),
                action="export",
                enforce_state=True,
                artifacts=self._artifacts,
                runs=self._runs,
            )
            try:
                url = self._artifacts.export_adapter(
                    source_repo=effective_spec.train.hf_repo,
                    source_subfolder=f"{prefix}/adapter",
                    dest_repo=repository,
                    dest_token=hf_token,
                    private=private,
                    base_model=spec.model,
                    # the effective half, not the public one: a runner-assigned pin is stripped
                    # from the public spec (it cannot carry the marker that labels it), so `spec`
                    # reads "". the worker stamps the real sha into adapter_config.json from its
                    # internal spec, and export refuses a stamped revision that disagrees with what
                    # it is handed -- so reading the public half here 404s every auto-pinned sft
                    # run and every warm start that inherited its pin.
                    base_model_revision=effective_spec.model_revision,
                )
            except ValueError as exc:
                raise DeploymentNotFound(str(exc)) from exc
            except ServingError as exc:
                raise DeploymentUpstreamError(str(exc)) from exc
        return ExportResult(
            run_id=run_id,
            repository=repository,
            url=url,
            checkpoint_step=checkpoint_step,
            is_checkpoint=is_checkpoint,
            status=status,
        )

    # ---- chat --------------------------------------------------------------------------

    def plan_chat(self, command: ChatCommand) -> ChatPlan:
        """Authorize and resolve a chat request without performing it.

        Authorization comes from verified-ledger membership, not the status record alone: a record
        can claim ready for a revision the ledger never vouched for.
        """
        from flash.runner import effective_spec_from_status
        from flash.schema import parse_adapter_revision

        run_id = command.run_id
        payload = command.payload
        messages = _chat_messages_from_payload(payload)
        status = self._runs.owned_run(run_id, command.caller.key)
        verified_revisions = set(self._deployments.verified_revisions(run_id))
        deployment = status.deployment or {}
        ready_deployment = revisions.previous_ready_deployment(deployment)
        ready_revision = (
            ready_deployment.get("adapter_revision") if ready_deployment is not None else None
        )
        pinned_revision = revisions.resolve_explicit_chat_revision(
            run_id,
            payload.get("adapter_revision"),
            payload.get("step"),
            verified_revisions,
            preferred_revision=ready_revision if isinstance(ready_revision, str) else None,
        )
        spec = JobSpec.from_dict(status.spec)
        try:
            effective_spec = effective_spec_from_status(status)
        except ValueError as exc:
            raise DeploymentConflict(str(exc)) from exc
        has_ready_deploy = pinned_revision is not None or ready_deployment is not None
        if pinned_revision is None and ready_deployment is not None:
            ready_revision = ready_deployment.get("adapter_revision")
            parsed_ready_revision = (
                parse_adapter_revision(ready_revision) if isinstance(ready_revision, str) else None
            )
            has_ready_deploy = bool(
                parsed_ready_revision is not None
                and parsed_ready_revision[0] == run_id
                and ready_revision in verified_revisions
            )
        if not has_ready_deploy:
            self._reject_unservable_chat(run_id, status, deployment)
        if not effective_spec.train.hf_repo:
            raise DeploymentConflict(
                f"run {run_id} has no [train].hf_repo; its adapter cannot be served"
            )
        temperature, max_tokens = _chat_sampling(payload)
        # the same stops the deployment smoke verified with: a run trained to terminate on a
        # delimiter rather than EOS would otherwise pass verification and then run to max_tokens,
        # or emit trailing text past its answer, on every real request.
        stop_sequences = [
            str(value) for value in (getattr(spec.train, "stop_sequences", ()) or ())
        ] or None
        return ChatPlan(
            serving_model=pinned_revision or run_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=spec.thinking,
            stop=stop_sequences,
            stream=payload.get("stream") is True,
        )

    @staticmethod
    def _reject_unservable_chat(run_id: str, status, deployment: dict):
        # A cancelled run can still serve a per-step checkpoint it deployed: checkpoint deploy
        # records a live adapter that /v1/deployments lists as active without requiring a final
        # adapter. Only block chat when there's no active deployment to serve.
        deployment_state_name = deployment.get("state")
        if deployment_state_name in revisions.DEPLOYMENT_BUSY_STATES:
            raise DeploymentConflict(
                f"run {run_id} deployment is {deployment_state_name}; run "
                "`flash models deployments` to check progress"
            )
        if deployment_state_name == "failed":
            raise DeploymentConflict(
                f"run {run_id} deployment failed: {deployment.get('error') or 'unknown error'}"
            )
        if status.state == "cancelled":
            raise DeploymentConflict(
                f"run {run_id} was cancelled; deploy a checkpoint with "
                f"`flash models deploy {run_id}/step-<N>` first"
            )
        raise DeploymentConflict(
            f"run {run_id} has no active deployment; `flash models deploy {run_id}` first"
        )

    def chat(self, plan: ChatPlan) -> dict:
        return self._serving.chat(
            run_id=plan.serving_model,
            messages=plan.messages,
            temperature=plan.temperature,
            max_tokens=plan.max_tokens,
            thinking=plan.thinking,
            stop=plan.stop,
        )

    def chat_stream(self, plan: ChatPlan):
        return self._serving.chat_stream(
            run_id=plan.serving_model,
            messages=plan.messages,
            temperature=plan.temperature,
            max_tokens=plan.max_tokens,
            thinking=plan.thinking,
            stop=plan.stop,
        )

    # ---- startup -----------------------------------------------------------------------

    def recover_deployments(self) -> int:
        return self.lifecycle.recover_deployments()

    def replay_status_reports(self, stop=None) -> int:
        return self.lifecycle.replay_status_reports(stop)

    # ---- helpers -----------------------------------------------------------------------

    def _manageable(self, run_id: str, caller: CallerContext):
        return self._runs.manageable_run(run_id, caller.key, caller.org_id, caller.project_id)


def _require_bool(payload: dict, field_name: str, default: bool) -> bool:
    """Read a real JSON boolean; never coerce -- "false" (str) is truthy."""
    value = payload.get(field_name, default)
    if not isinstance(value, bool):
        raise InvalidDeploymentRequest(f"{field_name} must be a boolean")
    return value


def _chat_messages_from_payload(payload: dict) -> list[dict]:
    raw = payload.get("messages")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InvalidDeploymentRequest("messages must be a list")
    for index, message in enumerate(raw):
        if not isinstance(message, dict):
            raise InvalidDeploymentRequest(f"messages[{index}] must be a chat message object")
    return raw


def _chat_sampling(payload: dict) -> tuple[float, int]:
    """Parse sampling params up front so bad values are refused before serving is reached."""
    import math

    try:
        temperature = float(payload.get("temperature") or 0.0)
        # Avoid `or 512`: that silently coerces an explicit 0 to 512.
        raw_max_tokens = payload.get("max_tokens")
        # OverflowError (int(inf), an ArithmeticError) is NOT a TypeError/ValueError -- catch it
        # too so a JSON `Infinity`/`1e400` max_tokens is a clean refusal, not an uncaught 500.
        max_tokens = 512 if raw_max_tokens is None else int(raw_max_tokens)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidDeploymentRequest(f"invalid temperature/max_tokens: {exc}") from exc
    if not math.isfinite(temperature):
        raise InvalidDeploymentRequest(f"temperature must be a finite number, got {temperature}")
    if max_tokens <= 0:
        raise InvalidDeploymentRequest(f"max_tokens must be a positive integer, got {max_tokens}")
    return temperature, max_tokens


def _validated_repository(payload: dict) -> str:
    repository = str(payload.get("repository") or "").strip()
    if not repository:
        raise InvalidDeploymentRequest(
            "repository is required: the destination HuggingFace repo 'owner/name'"
        )
    if len(parts := repository.strip("/").split("/")) != 2 or not all(parts):
        raise InvalidDeploymentRequest(
            f"repository must be a HuggingFace repo of the form 'owner/name', got {repository!r}"
        )
    repository = "/".join(parts)
    _validate_hf_repo_id(repository)
    return repository


def _validate_hf_repo_id(repository: str) -> None:
    """Validate HF repo id grammar early -- malformed ids only fail AFTER downloading the source."""
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id
    except ModuleNotFoundError:
        return
    try:
        validate_repo_id(repository)
    except HFValidationError as exc:
        raise InvalidDeploymentRequest(
            f"repository is not a valid HuggingFace repo id: {exc}"
        ) from exc
