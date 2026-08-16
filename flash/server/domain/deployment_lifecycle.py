"""The half of a deployment that outlives the HTTP response.

`DeploymentLifecycle.finish` runs after the request has persisted its `queued` record and handed
over the deploy lock. It smokes the new revision, activates the alias, and records the answer.
Everything here is about recording that answer safely -- fencing a stale attempt, committing the
ready state, reconciling a commit that raced, and sweeping deployments a restart left mid-flight.

Ordering that must not be rearranged:
  fence -> smoke -> fence again -> activate -> coordinated ready commit
The second fence exists because a cancel or a newer deploy can land while smoke is blocked.
"""

from __future__ import annotations

import time
from threading import Event

from flash.core.spec import JobSpec
from flash.serve.deploy import (
    ActivationOutcomeUnknown,
    AdapterConfigMissing,
    AliasThinkingSilent,
    ServingError,
)
from flash.server.domain import deployment_smoke
from flash.server.domain.deployment_ports import (
    ArtifactRepository,
    DeploymentLockProvider,
    DeploymentReporter,
    DeploymentRepository,
    RunRepository,
    ServingGateway,
)
from flash.server.domain.deployment_records import (
    deployment_failure_persisted,
    deployment_state,
    public_deployment_view,
)
from flash.server.domain.deployment_revisions import (
    DEPLOYMENT_BUSY_STATES,
    DEPLOYMENT_READY_STATES,
    spec_is_unservable,
)


class DeploymentLifecycle:
    """Runs one deployment attempt to its recorded conclusion, and sweeps interrupted ones."""

    def __init__(
        self,
        *,
        runs: RunRepository,
        deployments: DeploymentRepository,
        artifacts: ArtifactRepository,
        serving: ServingGateway,
        reporter: DeploymentReporter,
        lock_provider: DeploymentLockProvider,
    ) -> None:
        self._runs = runs
        self._deployments = deployments
        self._artifacts = artifacts
        self._serving = serving
        self._reporter = reporter
        self._lock_provider = lock_provider

    # the job runner calls this; it owns the lock the request handed over.
    def finish_locked(self, *, deploy_lock, **kwargs) -> None:
        try:
            self.finish(**kwargs)
        finally:
            deploy_lock.release()

    def finish(
        self,
        *,
        run_id: str,
        spec_dict: dict,
        is_checkpoint: bool,
        deploy_kwargs: dict,
        deployment: dict,
        prev_state: str,
    ) -> None:
        spec = JobSpec.from_dict(spec_dict)
        active = self._runs.get_status(run_id).deployment or {}
        if (
            active.get("requested_at") != deployment.get("requested_at")
            or active.get("state") not in DEPLOYMENT_BUSY_STATES
        ):
            return
        attempt = _Attempt(
            lifecycle=self,
            run_id=run_id,
            spec=spec,
            is_checkpoint=is_checkpoint,
            deployment=deployment,
            prev_state=prev_state,
        )
        attempt.run(deploy_kwargs)

    def assert_activation_fence(
        self, run_id: str, deployment: dict, is_checkpoint: bool, prev_state: str
    ) -> None:
        """Raise ``ServingError`` unless this attempt still owns the record and may activate.

        Re-read on every call: the point of the fence is that a cancel or a newer deploy can land
        while smoke is blocked, so a cached status would defeat it.
        """
        latest = self._runs.get_status(run_id)
        latest_deployment = latest.deployment or {}
        if (
            latest_deployment.get("requested_at") != deployment.get("requested_at")
            or latest_deployment.get("state") not in DEPLOYMENT_BUSY_STATES
        ):
            raise ServingError("deployment attempt was superseded before alias activation")
        if is_checkpoint:
            if prev_state in self._runs.deployable_states and latest.state != prev_state:
                raise ServingError(
                    f"run state changed from {prev_state!r} to {latest.state!r} "
                    "before alias activation"
                )
            if latest.state in {"cancelled", "failed", "dry_run"} and latest.state != prev_state:
                raise ServingError(
                    f"run became {latest.state!r} before checkpoint alias activation"
                )
        elif latest.state != prev_state:
            raise ServingError(
                f"run state changed from {prev_state!r} to {latest.state!r} before alias activation"
            )
        expected_generation = deployment.get("verification_generation")
        if self._deployments.verification_generation(run_id) != expected_generation:
            raise ServingError("deployment verification generation changed before alias activation")

    def commit_ready(
        self,
        run_id: str,
        current: dict,
        verification_generation,
        is_checkpoint: bool,
        prev_state: str,
    ) -> bool:
        """Persist the ready record. Returns whether the guarded write landed.

        The verification generation travels WITH the write so ledger membership and the status
        commit under the ledger lock together. Splitting them would publish a ready deployment the
        ledger does not vouch for.
        """
        previous = self._runs.get_status(run_id)
        if is_checkpoint:
            state_guard = prev_state if prev_state in self._runs.deployable_states else None
            marked = self._deployments.mark_checkpoint_deployed(
                run_id,
                current,
                expect_state=state_guard,
                verification_generation=verification_generation,
            )
            persisted = marked.deployment == current
        else:
            marked = self._deployments.mark_deployed(
                run_id,
                current,
                expect_state=prev_state,
                verification_generation=verification_generation,
            )
            persisted = marked.state == "deployed" and marked.deployment == current
        if persisted:
            self._reporter.report_transition(
                previous, marked, persisted=marked.deployment == current
            )
        return persisted

    def reconcile_ready_commit_miss(
        self,
        run_id: str,
        current: dict,
        verification_generation,
        is_checkpoint: bool,
        deployment: dict,
    ) -> None:
        # deploy_adapter already flipped the serving alias when this runs, so a lost
        # control-plane cas must never be dropped silently -- and the alias is never
        # reverted here (post-promotion recovery reads the authoritative alias; a revert
        # could clobber a newer deployment).
        latest = self._runs.get_status(run_id)
        latest_deployment = latest.deployment or {}
        owned = latest_deployment.get("requested_at") == deployment.get("requested_at")
        if owned and latest_deployment.get("state") in DEPLOYMENT_BUSY_STATES:
            # this attempt still owns the record; only the run state moved under the
            # guard. retry the write once against the fresh state.
            previous = latest
            if is_checkpoint:
                marked = self._deployments.mark_checkpoint_deployed(
                    run_id,
                    current,
                    verification_generation=verification_generation,
                )
            else:
                marked = self._deployments.mark_deployed(
                    run_id,
                    current,
                    expect_state=latest.state,
                    verification_generation=verification_generation,
                )
            if marked.deployment == current:
                self._reporter.report_transition(
                    previous, marked, persisted=marked.deployment == current
                )
                return
            latest = marked
            latest_deployment = latest.deployment or {}
        # superseded, undeployed, or uncommittable: a newer actor owns the record now, so
        # log the divergence loudly but never write over what that actor recorded -- a
        # clobber here would erase a concurrent final deploy's ready record or resurrect
        # an explicit undeploy.
        divergence = (
            "deployment_record_diverged: serving alias targets "
            f"{current.get('adapter_revision')} but the deployment record moved to "
            f"{latest_deployment.get('state')!r} (run state {latest.state!r}) during "
            "activation; serving alias left as activated"
        )
        print(f"deploy[{run_id}]: {divergence}", flush=True)

    def record_failure(
        self,
        run_id: str,
        spec: JobSpec,
        exc: Exception,
        current: dict,
        deployment: dict,
        is_checkpoint: bool,
    ) -> None:
        """Persist the failed record for an attempt that never activated the alias."""
        error = str(exc)
        if not is_checkpoint and isinstance(exc, AdapterConfigMissing):
            steps = [c["step"] for c in self._artifacts.list_checkpoints(spec)]
            if steps:
                error = (
                    f"run {run_id} has no run-level adapter at "
                    f"{deployment.get('adapter_hf_prefix')} (the run likely never finalized); "
                    f"deploy a saved checkpoint instead, e.g. `flash models deploy "
                    f"{run_id}/step-{steps[-1]}` (available steps: "
                    f"{', '.join(str(step) for step in steps)})"
                )
        failed_source = dict(current)
        if not deployment.get("activation_outcome_unknown"):
            failed_source.pop("activation_outcome_unknown", None)
        failed = deployment_state(
            failed_source,
            "failed",
            error=error,
            detail="deployment failed; previous working alias was preserved",
        )
        previous = self._runs.get_status(run_id)
        marked = self._deployments.mark_failed(run_id, failed)
        self._reporter.report_transition(
            previous, marked, persisted=deployment_failure_persisted(marked, failed)
        )

    def record_post_activation_failure(self, run_id: str, exc: Exception, current: dict) -> None:
        """The alias IS live and stays live: a revert could clobber a newer deployment."""
        failed_source = dict(current)
        failed_source.pop("activation_outcome_unknown", None)
        failed_source.pop("previous_deployment", None)
        fields = {"alias_activation_confirmed": True}
        if isinstance(exc, AliasThinkingSilent):
            fields["alias_thinking_tag"] = False
        failed = deployment_state(
            failed_source,
            "failed",
            error=str(exc),
            detail="alias activated but post-activation verification failed; redeploy to retry",
            **fields,
        )
        try:
            previous = self._runs.get_status(run_id)
            marked = self._deployments.mark_failed(run_id, failed)
            self._reporter.report_transition(
                previous, marked, persisted=deployment_failure_persisted(marked, failed)
            )
        except Exception as persistence_exc:
            divergence = (
                "deployment_record_diverged: serving alias was activated for "
                f"{failed.get('adapter_revision')} but failure-state recovery did not complete "
                f"after {exc!r}: {persistence_exc!r}"
            )
            print(f"deploy[{run_id}]: {divergence}", flush=True)

    def recover_deployments(self) -> int:
        """Clear deployment lifecycle records left busy by a control-plane restart."""
        recovered = 0
        for row in self._runs.all_runs():
            try:
                status = self._runs.get_status(row["run_id"])
            except FileNotFoundError:
                continue
            state = (status.deployment or {}).get("state")
            if state not in DEPLOYMENT_BUSY_STATES and state not in DEPLOYMENT_READY_STATES:
                continue
            lock = self._lock_provider(row["run_id"])
            # another replica mid-deploy holds the flock, so a non-blocking miss proves live
            # ownership.
            if not lock.acquire(blocking=False):
                continue
            try:
                recovered += self._recover_one_locked(row["run_id"])
            finally:
                lock.release()
        return recovered

    def _recover_one_locked(self, run_id: str) -> int:
        try:
            status = self._runs.get_status(run_id)
        except FileNotFoundError:
            return 0
        deployment = status.deployment or {}
        state = deployment.get("state")
        if state in DEPLOYMENT_BUSY_STATES:
            # No freshness test, deliberately: a live lifecycle holds the flock for its whole
            # duration, so ACQUIRING it proves no owner survives however recent the timestamp
            # looks. Recovery runs only at startup, so skipping fresh records left them busy with
            # nothing to revisit them, answering retries with 409 until aged out.
            error = "deployment lifecycle interrupted by control-plane restart"
            detail = "deployment interrupted; retry `flash models deploy`"
        elif state in DEPLOYMENT_READY_STATES and spec_is_unservable(status):
            # a ready record with an unparseable spec is unservable, so fail it during startup.
            # handle both readiness spellings from persisted builds.
            error = "deployment spec is no longer supported by this control plane"
            detail = "deployment retired: its algorithm was removed; submit a new run to deploy"
        else:
            return 0
        recovered_state = (
            "reconciling"
            if state == "reconciling" and deployment.get("activation_outcome_unknown") is True
            else "failed"
        )
        failed = deployment_state(
            deployment,
            recovered_state,
            error=error,
            detail=detail,
            recovered_at=time.time(),
        )
        marked = self._deployments.mark_failed(status.run_id, failed)
        self._reporter.report_transition(
            status, marked, persisted=deployment_failure_persisted(marked, failed)
        )
        return 1

    def run_smoke(
        self, run_id: str, spec: JobSpec, *, serving_model: str, expected_checkpoint: str
    ) -> dict:
        return deployment_smoke.run_deployment_smoke(
            run_id,
            spec,
            serving=self._serving,
            serving_model=serving_model,
            expected_checkpoint=expected_checkpoint,
        )

    def verify_alias_thinking(
        self, run_id: str, spec: JobSpec, adapter_revision: str, expected_checkpoint: str
    ) -> dict:
        return deployment_smoke.verify_alias_thinking(
            run_id,
            spec,
            adapter_revision,
            expected_checkpoint,
            serving=self._serving,
        )

    def replay_status_reports(self, stop: Event | None = None) -> int:
        """Sequentially mirror persisted statuses that may have been dropped during shutdown."""
        replayed = 0
        for row in self._runs.all_runs():
            if stop is not None and stop.is_set():
                break
            try:
                status = self._runs.get_status(row["run_id"])
                self._reporter.report_sequentially(status)
            except (OSError, TypeError, ValueError):
                continue
            replayed += 1
        return replayed


class _Attempt:
    """One deployment attempt's mutable progress through smoke, activation, and commit."""

    def __init__(
        self,
        *,
        lifecycle: DeploymentLifecycle,
        run_id: str,
        spec: JobSpec,
        is_checkpoint: bool,
        deployment: dict,
        prev_state: str,
    ) -> None:
        self._lifecycle = lifecycle
        self.run_id = run_id
        self.spec = spec
        self.is_checkpoint = is_checkpoint
        self.deployment = deployment
        self.prev_state = prev_state
        self.current = dict(deployment)
        self.smoke_result: dict = {}
        self.activation_target: tuple[str, str] | None = None

    def run(self, deploy_kwargs: dict) -> None:
        lifecycle = self._lifecycle
        try:
            dep = lifecycle._serving.deploy_adapter(
                **deploy_kwargs, before_activate=self._before_activate
            )
        except ActivationOutcomeUnknown as exc:
            self._record_unknown_activation(exc)
            return
        except Exception as exc:
            lifecycle.record_failure(
                self.run_id, self.spec, exc, self.current, self.deployment, self.is_checkpoint
            )
            return

        activated_current = {**self.current, **dep.to_dict()}
        activated_current.pop("activation_outcome_unknown", None)
        try:
            if self.spec.thinking and self.smoke_result.get("thinking_tag"):
                if self.activation_target is None:
                    raise ServingError(
                        "deploy_adapter returned without reporting its activation target"
                    )
                self._verify_activated_alias_thinking()
        except Exception as exc:
            lifecycle.record_post_activation_failure(self.run_id, exc, activated_current)
            return

        self._commit_ready(activated_current)

    def _assert_fence(self) -> None:
        self._lifecycle.assert_activation_fence(
            self.run_id, self.deployment, self.is_checkpoint, self.prev_state
        )

    def _persist_progress(self, current: dict) -> None:
        lifecycle = self._lifecycle
        previous = lifecycle._runs.get_status(self.run_id)
        marked = lifecycle._deployments.mark_pending(
            self.run_id, current, owner_deployment=self.deployment
        )
        lifecycle._reporter.report_transition(
            previous, marked, persisted=marked.deployment == current
        )

    def _before_activate(self, adapter_revision: str, checkpoint: str) -> None:
        self._assert_fence()
        self.activation_target = (adapter_revision, checkpoint)
        self.current = deployment_state(
            {**self.current, "adapter_revision": adapter_revision},
            "smoke_testing",
            detail="running bounded fixed-prompt smoke",
        )
        self._persist_progress(self.current)
        self.smoke_result.update(
            self._lifecycle.run_smoke(
                self.run_id,
                self.spec,
                serving_model=adapter_revision,
                expected_checkpoint=checkpoint,
            )
        )
        self.current = deployment_state(
            self.current,
            "reconciling",
            detail="activating alias and reconciling the authoritative target",
            activation_outcome_unknown=True,
        )
        self._persist_progress(self.current)
        # cancellation can revoke the ledger while smoke is blocked, so fence again immediately
        # before deploy_adapter issues the activation request.
        self._assert_fence()

    def _verify_activated_alias_thinking(self) -> None:
        """Prove the freshly activated alias kept the reasoning channel the revision smoked with.

        Raises `ServingError` when it did not, which reaches the caller's failure path with the
        alias already live. That is the correct direction: the alias serves a real adapter and
        answers normally, so tearing it down would replace a degraded deployment with none, but
        committing `ready` would state that a thinking deployment thinks when it demonstrably
        does not.

        Gated on the smoke's own `thinking_tag` so this only ever fires on a genuine regression: if
        the pinned revision produced no reasoning either, the smoke has already judged that (it
        raises for a catalog model), and the difference this check exists to catch is not present.
        """
        revision, expected_checkpoint = self.activation_target
        self.smoke_result.update(
            self._lifecycle.verify_alias_thinking(
                self.run_id, self.spec, revision, expected_checkpoint
            )
        )

    def _record_unknown_activation(self, exc: Exception) -> None:
        lifecycle = self._lifecycle
        reconciling = deployment_state(
            self.current,
            "reconciling",
            error=str(exc),
            detail="alias activation outcome is unknown; authoritative reconciliation required",
            activation_outcome_unknown=True,
        )
        previous = lifecycle._runs.get_status(self.run_id)
        marked = lifecycle._deployments.mark_failed(self.run_id, reconciling)
        lifecycle._reporter.report_transition(
            previous, marked, persisted=marked.deployment == reconciling
        )

    def _commit_ready(self, activated_current: dict) -> None:
        lifecycle = self._lifecycle
        current = dict(activated_current)
        current["verify"] = True
        current = deployment_state(
            current,
            "ready",
            detail="immutable revision verified and alias activated",
            **self.smoke_result,
        )
        verification_generation = current.get("verification_generation")
        current = public_deployment_view(current)
        try:
            if not lifecycle.commit_ready(
                self.run_id, current, verification_generation, self.is_checkpoint, self.prev_state
            ):
                lifecycle.reconcile_ready_commit_miss(
                    self.run_id,
                    current,
                    verification_generation,
                    self.is_checkpoint,
                    self.deployment,
                )
        except Exception as exc:
            self._recover_ready_commit(current, verification_generation, exc)

    def _recover_ready_commit(self, current: dict, verification_generation, exc: Exception) -> None:
        lifecycle = self._lifecycle
        try:
            latest = lifecycle._runs.get_status(self.run_id)
            latest_deployment = latest.deployment or {}
            if (
                latest_deployment.get("adapter_revision") == current.get("adapter_revision")
                and latest_deployment.get("state") in DEPLOYMENT_READY_STATES
            ):
                return
            if not lifecycle.commit_ready(
                self.run_id, current, verification_generation, self.is_checkpoint, self.prev_state
            ):
                lifecycle.reconcile_ready_commit_miss(
                    self.run_id,
                    current,
                    verification_generation,
                    self.is_checkpoint,
                    self.deployment,
                )
        except Exception as recovery_exc:
            divergence = (
                "deployment_record_diverged: serving alias was activated for "
                f"{current.get('adapter_revision')} but ready-state recovery failed after "
                f"{exc!r}: {recovery_exc!r}"
            )
            print(f"deploy[{self.run_id}]: {divergence}", flush=True)
