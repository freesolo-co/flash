"""Completing a deployment: activation fence, ready commit, failure record, and recovery.

`_finish_deployment_unlocked` runs after the smoke test has already answered whether the new
revision serves. Everything here is about recording that answer safely -- fencing a stale attempt,
committing the ready state, reconciling a commit that raced, and sweeping deployments a restart
left mid-flight. The HTTP routes stay in `serving.py`; this module never defines one.

Split out of `flash.server.routes.serving` to keep that module under the file-size limit.
"""

from __future__ import annotations

import time
from threading import Event

from flash.core.spec import JobSpec
from flash.runner import (
    mark_checkpoint_deployed,
    verified_adapter_revision_generation,
)
from flash.serve.deploy import ActivationOutcomeUnknown, AdapterConfigMissing, ServingError
from flash.server import app as _app
from flash.server.platform import db

# resolved through `serving` rather than imported from `flash.runner`: a serving test patches
# `serving.mark_deployed` to fail after persistence, and a direct import would bind the original
# and never see that patch.
# every name reached through `_serving` below is patched as an attribute of that module by the
# deploy and recovery tests. a `from ... import` would capture the original at import time, so the
# patch would rebind the parent's attribute while this module kept calling the real function.
from flash.server.routes import serving as _serving
from flash.server.routes.serving import (
    _deployment_attempt_is_stale,
    _deployment_failure_persisted,
    _deployment_state,
    _public_deployment,
)
from flash.server.routes.serving_revisions import (
    _DEPLOYMENT_BUSY_STATES,
    _DEPLOYMENT_READY_STATES,
    _spec_is_unservable,
)


def recover_deployments() -> int:
    """Clear deployment lifecycle records left busy by a control-plane restart."""
    recovered = 0
    for row in db.all_runs():
        try:
            status = _app.get_status(row["run_id"])
        except FileNotFoundError:
            continue
        state = (status.deployment or {}).get("state")
        if state not in _DEPLOYMENT_BUSY_STATES and state not in _DEPLOYMENT_READY_STATES:
            continue
        lock = _app._deploy_lock(row["run_id"])
        # another replica mid-deploy holds the flock, so a non-blocking miss proves live ownership.
        if not lock.acquire(blocking=False):
            continue
        try:
            try:
                status = _app.get_status(row["run_id"])
            except FileNotFoundError:
                continue
            deployment = status.deployment or {}
            state = deployment.get("state")
            if state in _DEPLOYMENT_BUSY_STATES:
                if not _deployment_attempt_is_stale(deployment):
                    continue
                error = "deployment lifecycle interrupted by control-plane restart"
                detail = "deployment interrupted; retry `flash models deploy`"
            elif state in _DEPLOYMENT_READY_STATES and _spec_is_unservable(status):
                # a ready record with an unparseable spec is unservable, so fail it during startup.
                # handle both readiness spellings from persisted builds; staleness applies only to
                # busy states.
                error = "deployment spec is no longer supported by this control plane"
                detail = "deployment retired: its algorithm was removed; submit a new run to deploy"
            else:
                continue
            failed = _deployment_state(
                deployment,
                "failed",
                error=error,
                detail=detail,
                recovered_at=time.time(),
            )
            marked = _serving.mark_deployment_failed(status.run_id, failed)
            _serving._report_persisted_transition(
                status,
                marked,
                persisted=_deployment_failure_persisted(marked, failed),
            )
            recovered += 1
        finally:
            lock.release()
    return recovered


def replay_status_reports(stop: Event | None = None) -> int:
    """Sequentially mirror persisted statuses that may have been dropped during shutdown."""
    from flash.runner import _report_status

    replayed = 0
    for row in db.all_runs():
        if stop is not None and stop.is_set():
            break
        try:
            status = _app.get_status(row["run_id"])
            _report_status(status)
        except (OSError, TypeError, ValueError):
            continue
        replayed += 1
    return replayed


def _assert_deployment_activation_fence(
    run_id: str, deployment: dict, is_checkpoint: bool, prev_state: str
) -> None:
    """Raise ``ServingError`` unless this attempt still owns the record and may activate the alias.

    Re-read on every call: the point of the fence is that a cancel or a newer deploy can land while
    smoke is blocked, so a cached status would defeat it.
    """
    latest = _app.get_status(run_id)
    latest_deployment = latest.deployment or {}
    if (
        latest_deployment.get("requested_at") != deployment.get("requested_at")
        or latest_deployment.get("state") not in _DEPLOYMENT_BUSY_STATES
    ):
        raise ServingError("deployment attempt was superseded before alias activation")
    if is_checkpoint:
        if prev_state in _app._DEPLOYABLE_STATES and latest.state != prev_state:
            raise ServingError(
                f"run state changed from {prev_state!r} to {latest.state!r} before alias activation"
            )
        if latest.state in {"cancelled", "failed", "dry_run"} and latest.state != prev_state:
            raise ServingError(f"run became {latest.state!r} before checkpoint alias activation")
    elif latest.state != prev_state:
        raise ServingError(
            f"run state changed from {prev_state!r} to {latest.state!r} before alias activation"
        )
    expected_generation = deployment.get("verification_generation")
    if verified_adapter_revision_generation(run_id) != expected_generation:
        raise ServingError("deployment verification generation changed before alias activation")


def _commit_ready_deployment(
    run_id: str,
    current: dict,
    verification_generation,
    is_checkpoint: bool,
    prev_state: str,
) -> bool:
    """Persist the ready deployment record. Returns whether the guarded write landed."""
    previous = _app.get_status(run_id)
    if is_checkpoint:
        state_guard = prev_state if prev_state in _app._DEPLOYABLE_STATES else None
        marked = mark_checkpoint_deployed(
            run_id,
            current,
            expect_state=state_guard,
            verification_generation=verification_generation,
        )
        persisted = marked.deployment == current
    else:
        marked = _serving.mark_deployed(
            run_id,
            current,
            expect_state=prev_state,
            verification_generation=verification_generation,
        )
        persisted = marked.state == "deployed" and marked.deployment == current
    if persisted:
        _serving._report_persisted_transition(
            previous, marked, persisted=marked.deployment == current
        )
    return persisted


def _reconcile_ready_commit_miss(
    run_id: str,
    current: dict,
    verification_generation,
    is_checkpoint: bool,
    deployment: dict,
) -> None:
    # deploy_adapter already flipped the serving alias when this runs, so a lost
    # control-plane cas must never be dropped silently — and the alias is never
    # reverted here (post-promotion recovery reads the authoritative alias; a revert
    # could clobber a newer deployment).
    latest = _app.get_status(run_id)
    latest_deployment = latest.deployment or {}
    owned = latest_deployment.get("requested_at") == deployment.get("requested_at")
    if owned and latest_deployment.get("state") in _DEPLOYMENT_BUSY_STATES:
        # this attempt still owns the record; only the run state moved under the
        # guard. retry the write once against the fresh state.
        previous = latest
        if is_checkpoint:
            marked = mark_checkpoint_deployed(
                run_id,
                current,
                verification_generation=verification_generation,
            )
        else:
            marked = _serving.mark_deployed(
                run_id,
                current,
                expect_state=latest.state,
                verification_generation=verification_generation,
            )
        if marked.deployment == current:
            _serving._report_persisted_transition(
                previous, marked, persisted=marked.deployment == current
            )
            return
        latest = marked
        latest_deployment = latest.deployment or {}
    # superseded, undeployed, or uncommittable: a newer actor owns the record now, so
    # log the divergence loudly but never write over what that actor recorded — a
    # clobber here would erase a concurrent final deploy's ready record or resurrect
    # an explicit undeploy.
    divergence = (
        "deployment_record_diverged: serving alias targets "
        f"{current.get('adapter_revision')} but the deployment record moved to "
        f"{latest_deployment.get('state')!r} (run state {latest.state!r}) during "
        "activation; serving alias left as activated"
    )
    print(f"deploy[{run_id}]: {divergence}", flush=True)


def _finish_deployment_unlocked(
    *,
    run_id: str,
    spec_dict: dict,
    checkpoint_step: int | None,
    is_checkpoint: bool,
    deploy_kwargs: dict,
    deployment: dict,
    prev_state: str,
) -> None:
    spec = JobSpec.from_dict(spec_dict)
    active = _app.get_status(run_id).deployment or {}
    if (
        active.get("requested_at") != deployment.get("requested_at")
        or active.get("state") not in _DEPLOYMENT_BUSY_STATES
    ):
        return
    current = dict(deployment)
    smoke_result: dict = {}
    activated = False

    def _assert_activation_fence() -> None:
        _assert_deployment_activation_fence(run_id, deployment, is_checkpoint, prev_state)

    def _before_activate(adapter_revision: str, checkpoint: str) -> None:
        nonlocal current
        _assert_activation_fence()
        # smoke is unconditional for real deployments: alias activation only ever follows a
        # verified generation against the immutable revision.
        current = _deployment_state(
            {**current, "adapter_revision": adapter_revision},
            "smoke_testing",
            detail="running bounded fixed-prompt smoke",
        )
        previous = _app.get_status(run_id)
        marked = _serving.mark_deployment_pending(run_id, current, owner_deployment=deployment)
        _serving._report_persisted_transition(
            previous, marked, persisted=marked.deployment == current
        )
        smoke_result.update(
            _serving._run_deployment_smoke(
                run_id,
                spec,
                serving_model=adapter_revision,
                expected_checkpoint=checkpoint,
            )
        )
        current = _deployment_state(
            current,
            "reconciling",
            detail="activating alias and reconciling the authoritative target",
            activation_outcome_unknown=True,
        )
        previous = _app.get_status(run_id)
        marked = _serving.mark_deployment_pending(run_id, current, owner_deployment=deployment)
        _serving._report_persisted_transition(
            previous, marked, persisted=marked.deployment == current
        )
        # cancellation can revoke the ledger while smoke is blocked, so fence again immediately
        # before deploy_adapter issues the activation request.
        _assert_activation_fence()

    try:
        dep = _app.deploy_adapter(**deploy_kwargs, before_activate=_before_activate)
        activated = True
        current = {**current, **dep.to_dict()}
        current.pop("activation_outcome_unknown", None)
        current["verify"] = True
        current = _deployment_state(
            current,
            "ready",
            detail="immutable revision verified and alias activated",
            **smoke_result,
        )
        verification_generation = current.get("verification_generation")
        current = _public_deployment(current)

        def _commit_ready() -> bool:
            return _commit_ready_deployment(
                run_id, current, verification_generation, is_checkpoint, prev_state
            )

        def _reconcile_commit_miss() -> None:
            _reconcile_ready_commit_miss(
                run_id, current, verification_generation, is_checkpoint, deployment
            )

        if not _commit_ready():
            _reconcile_commit_miss()
            return
    except Exception as exc:
        if isinstance(exc, ActivationOutcomeUnknown):
            reconciling = _deployment_state(
                current,
                "reconciling",
                error=str(exc),
                detail="alias activation outcome is unknown; authoritative reconciliation required",
                activation_outcome_unknown=True,
            )
            previous = _app.get_status(run_id)
            marked = _serving.mark_deployment_failed(run_id, reconciling)
            _serving._report_persisted_transition(
                previous, marked, persisted=marked.deployment == reconciling
            )
            return
        if activated:
            try:
                latest = _app.get_status(run_id)
                latest_deployment = latest.deployment or {}
                if (
                    latest_deployment.get("adapter_revision") == current.get("adapter_revision")
                    and latest_deployment.get("state") in _DEPLOYMENT_READY_STATES
                ):
                    return
                if not _commit_ready():
                    _reconcile_commit_miss()
            except Exception as recovery_exc:
                divergence = (
                    "deployment_record_diverged: serving alias was activated for "
                    f"{current.get('adapter_revision')} but ready-state recovery failed after "
                    f"{exc!r}: {recovery_exc!r}"
                )
                print(f"deploy[{run_id}]: {divergence}", flush=True)
            return
        _record_deployment_failure(run_id, spec, exc, current, deployment, is_checkpoint)


def _record_deployment_failure(
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
        steps = [c["step"] for c in _app.list_checkpoints(spec)]
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
    failed = _deployment_state(
        failed_source,
        "failed",
        error=error,
        detail="deployment failed; previous working alias was preserved",
    )
    previous = _app.get_status(run_id)
    marked = _serving.mark_deployment_failed(run_id, failed)
    _serving._report_persisted_transition(
        previous, marked, persisted=_deployment_failure_persisted(marked, failed)
    )
