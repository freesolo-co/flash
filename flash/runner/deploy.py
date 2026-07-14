"""Deploy / cancel / recover state transitions for a run.

Store helpers and the lifecycle functions (``_run_training`` / ``_gc_run_endpoints``) are
pulled in via FUNCTION-LOCAL lazy ``from flash.runner import ...`` imports — never at module
level — for the same two reasons as ``lifecycle.py``: avoid a partially-initialized-package
import cycle, and keep the test monkeypatches (e.g. ``flash.runner._gc_run_endpoints``)
reachable through the package global rather than a statically-bound copy.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from flash.schema import parse_adapter_revision
from flash.spec import JobSpec

if TYPE_CHECKING:
    from flash.runner import RunStatus

_FINAL_DEPLOYMENT_STATES = frozenset({"done", "deployed"})
_RESTORABLE_DEPLOYMENT_STATES = frozenset({"ready", "deployed"})
_DEPLOYMENT_BUSY_STATES = frozenset(
    {
        "deploying",
        "queued",
        "registered",
        "downloading",
        "loading",
        "smoke_testing",
        "registering",
        "verifying",
        "reconciling",
    }
)
_REVOCATION_RETRY_STATE = "revocation_failed"


class DeploymentRevocationError(RuntimeError):
    """Backend revocation is unconfirmed after local serving authority was removed."""

    def __init__(self, run_id: str, error: str):
        super().__init__(
            f"deployment_revocation_failed: local authorization for {run_id} was revoked, but "
            f"backend disablement is unconfirmed: {error}; retry cancellation"
        )
        self.run_id = run_id
        self.retryable = True


def cancel_run(run_id: str) -> RunStatus:
    """Cancel a run and fail closed if serving backend revocation is unconfirmed."""
    from flash.runner import (
        TERMINAL_STATES,
        _gc_run_endpoints,
        _update,
        actual_steps_run,
        charge_usd_for_spec,
        effective_spec_from_status,
        get_status,
        mark_deployment_revocation_failed,
        mark_deployment_undeployed,
    )
    from flash.server._locks import _deploy_lock

    def _teardown_remote(remote: dict, *, skip_cancel: bool = False) -> tuple[bool, bool]:
        if not remote:
            return False, False
        try:
            from flash.providers import get_provider
            from flash.providers.base import JobHandle

            handle = JobHandle.from_dict(remote)
            provider = get_provider(handle.provider)
        except Exception:
            return False, False
        cancelled = skip_cancel
        if not skip_cancel:
            try:
                provider.cancel(handle)
                cancelled = True
            except Exception:
                return False, False
        try:
            provider.destroy(handle)
            return cancelled, True
        except Exception:
            return cancelled, False

    initial_status = get_status(run_id)
    initial_deployment_state = (initial_status.deployment or {}).get("state")
    initial_retry_revocation = initial_deployment_state == _REVOCATION_RETRY_STATE
    initial_active_deployment = initial_deployment_state not in (None, "undeployed", "dry_run")
    if initial_status.state == "dry_run" or (
        initial_status.state in TERMINAL_STATES
        and not initial_retry_revocation
        and not initial_active_deployment
    ):
        return initial_status
    entered_deployed = initial_status.state == "deployed"

    # revoke serving authority and stop any durable remote before waiting on the deploy lock.
    teardown_error: Exception | None = None
    initial_teardown_confirmed = False
    if initial_active_deployment:
        try:
            from flash.serve.deploy import undeploy_adapter

            undeploy_adapter(run_id)
            mark_deployment_undeployed(run_id)
            initial_teardown_confirmed = True
        except Exception as exc:
            teardown_error = exc
    initial_remote = initial_status.remote or {}
    remote_cancelled, remote_destroyed = _teardown_remote(initial_remote)

    with _deploy_lock(run_id):
        # reread after the submission handshake releases the lock so a newly durable provider
        # handle is cancelled and destroyed rather than leaving an untracked paid resource.
        status = get_status(run_id)
        deployment_state = (status.deployment or {}).get("state")
        retry_revocation = deployment_state == _REVOCATION_RETRY_STATE
        has_active_deployment = deployment_state not in (None, "undeployed", "dry_run")
        if status.state == "dry_run" or (
            status.state in TERMINAL_STATES
            and not retry_revocation
            and not entered_deployed
            and not has_active_deployment
        ):
            return status
        # only a deployed run can have a racing undeploy write `done`; a training `done` is genuine.
        entered_deployed = entered_deployed or status.state == "deployed"
        public_spec = JobSpec.from_dict(status.spec)
        effective_spec = None
        with contextlib.suppress(Exception):
            effective_spec = effective_spec_from_status(status)
        cleanup_spec = effective_spec or public_spec
        # a mid-training cancel is re-priced from the authoritative prepared spec, including
        # revocation retries that are still non-terminal. deployed and terminal runs retain billing.
        bill_cancel = (
            bool(status.billing_context)
            and status.state not in TERMINAL_STATES
            and not entered_deployed
        )
        remote = status.remote or {}
        same_remote = remote == initial_remote
        if remote and (not same_remote or not remote_destroyed):
            _teardown_remote(remote, skip_cancel=same_remote and remote_cancelled)
        with contextlib.suppress(Exception):
            _gc_run_endpoints(cleanup_spec)
        if not remote:
            _teardown_remote(get_status(run_id).remote or {})
        cancel_charge_usd: float | None = None
        billing_diagnostic: dict = {}
        if bill_cancel:
            if effective_spec is not None:
                cancel_charge_usd = charge_usd_for_spec(
                    effective_spec,
                    steps=actual_steps_run(get_status(run_id)),
                    fallback=0.0,
                )
            else:
                cancel_charge_usd = 0.0
                billing_diagnostic = {
                    "billing_state": "failed",
                    "billing_error": (
                        "cancellation charge was not computed because the private preparation "
                        "snapshot was unavailable or invalid; teardown was still attempted"
                    ),
                }
        cancel_updates = {} if cancel_charge_usd is None else {"cost_usd": cancel_charge_usd}
        cancel_updates.update(billing_diagnostic)
        preserve_terminal_retry = (
            retry_revocation and status.state in TERMINAL_STATES and not entered_deployed
        )
        if status.state != "cancelled" and not preserve_terminal_retry:
            _update(run_id, "cancelled", allow_from_terminal=entered_deployed, **cancel_updates)
        final = get_status(run_id)
        deployment_state = (final.deployment or {}).get("state")
        if deployment_state not in (None, "undeployed", "dry_run"):
            if teardown_error is not None and status.state in TERMINAL_STATES and not entered_deployed:
                mark_deployment_revocation_failed(run_id, str(teardown_error))
            else:
                try:
                    from flash.serve.deploy import undeploy_adapter

                    undeploy_adapter(run_id)
                    mark_deployment_undeployed(run_id)
                    teardown_error = None
                except Exception as exc:
                    teardown_error = exc
                    mark_deployment_revocation_failed(run_id, str(exc))
        elif teardown_error is not None:
            mark_deployment_revocation_failed(run_id, str(teardown_error))
        elif not initial_teardown_confirmed:
            # cancel always advances the local generation, including stale status projections.
            mark_deployment_undeployed(run_id)
        with contextlib.suppress(Exception):
            from flash.server.checkpoints import register_checkpoints_best_effort

            register_checkpoints_best_effort(get_status(run_id))
    if teardown_error is not None:
        raise DeploymentRevocationError(run_id, str(teardown_error)) from teardown_error
    return get_status(run_id)


def attach_run(run_id: str, log_stream=None) -> RunStatus:
    """Re-attach to a run's remote job from any process (after a client crash/restart)."""
    import sys

    from flash.runner import (
        FIXED_SEED,
        TERMINAL_STATES,
        _gc_run_endpoints,
        _persist_metrics,
        _run_training,
        _RunCancelled,
        _status_estimated_charge,
        _update,
        artifacts_dir,
        effective_spec_from_status,
        get_status,
    )

    cleanup_terminal = False

    def status_for_return() -> RunStatus:
        nonlocal cleanup_terminal
        current = get_status(run_id)
        cleanup_terminal = current.state in TERMINAL_STATES
        return current

    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    if not status.remote:
        raise ValueError(f"run {run_id} has no persisted job handle; cannot reattach")

    public_spec = JobSpec.from_dict(status.spec)
    worker_spec = public_spec
    log = log_stream or sys.stderr
    from flash.providers import get_provider
    from flash.providers.base import JobHandle

    try:
        worker_spec = effective_spec_from_status(status)
        remote = dict(status.remote)
        seed = int(remote.pop("seed", FIXED_SEED))
        code_prefix = remote.pop("code_prefix", None)
        # The class the run actually provisioned (a policy retry may have walked past the
        # provisional spec.gpu.type). The in-process success path stamps this into metrics;
        # on recovery the worker output carries no such field, so recover it from the handle
        # to cost the right card.
        allocated_gpu = remote.pop("allocated_gpu", None)
        handle = JobHandle.from_dict(remote)
        print(f"attaching to {run_id}: provider={handle.provider} {handle.data}", file=log)
        res = get_provider(handle.provider).poll(handle, worker_spec, seed, log=log)
        if get_status(run_id).state == "cancelled":
            return status_for_return()
        if not res.ok:
            # job ended not-ok, so any replacement must revalidate the pinned source before paid work.
            worker_spec = effective_spec_from_status(get_status(run_id), verify_source=True)
            # Job ended not-ok — usually because it was abandoned during the redeploy. Resume from
            # the last HF checkpoint (fresh allocation, worker resumes mid-training) instead of
            # failing; _run_training still terminates a genuinely broken run when it re-fails.
            print(
                f"attach: {run_id} ended ({res.failure}); resuming from checkpoint",
                file=log,
            )
            # Before resuming, the in-flight instance MUST be CONFIRMED torn down. Resubmitting while
            # it may still be alive runs TWO workers against this run's shared HF artifacts
            # (DONE/metrics/checkpoints) — double bill AND corrupted state. An instance provider's
            # destroy() raises only on an UNCONFIRMED teardown (Vast: DELETE success:false / network
            # breakdown — a real 404 is now treated as confirmed-gone). The poll loop's own finally
            # already best-effort-destroyed the box; re-confirm here. On an unconfirmed result, GC by
            # label (run-scoped, not orphan-sweep-shielded) and BAIL with the handle intact + the run
            # left non-terminal, so a later recovery/sweep reconciles instead of racing a live box.
            from flash.providers import INSTANCE_PROVIDERS

            teardown_confirmed = True
            if handle.provider in INSTANCE_PROVIDERS:
                try:
                    get_provider(handle.provider).destroy(handle)
                except Exception as exc:
                    teardown_confirmed = False
                    print(
                        f"attach: {run_id} {handle.provider} instance teardown UNCONFIRMED ({exc}); "
                        "not resuming over a possibly-live box",
                        file=log,
                    )
            # GC the dead endpoint / any label-named instances (a second force-reap attempt when the
            # teardown above was unconfirmed), then clear the stale handle.
            with contextlib.suppress(Exception):
                _gc_run_endpoints(worker_spec)
            if not teardown_confirmed:
                # Keep ``remote`` so the still-billing box stays reachable for the next recovery/sweep,
                # and leave the run non-terminal (do not _update) so a future re-attach re-polls it.
                return status_for_return()
            # Bail if the run was raced to terminal during the long poll above: _update's CAS
            # returns False, and resuming would submit paid work for a dead run.
            if not _update(run_id, "running", remote=None):
                print(f"attach: {run_id} went terminal during recovery; not resuming", file=log)
                return status_for_return()
            if code_prefix is None:
                from flash.providers._worker import upload_code
                from flash.runner import flash_code_prefix

                code_prefix = flash_code_prefix()
                upload_code(worker_spec.train.hf_repo, code_prefix=code_prefix)
            _run_training(
                worker_spec,
                log,
                prior_cost=float(status.cost_usd or 0.0),
                code_prefix=code_prefix,
            )
            return status_for_return()
        if allocated_gpu and isinstance(res.metrics, dict):
            res.metrics.setdefault("allocated_gpu", allocated_gpu)
        # Add the recovered run's cost to any already booked before the restart so recovery
        # doesn't underreport spend.
        measured = float(status.cost_usd or 0.0) + _persist_metrics(public_spec, res.metrics)
        # Charge the submit-time QUOTE, not measured wall; recovery doesn't change the quote.
        # Legacy runs without a persisted quote are re-priced from the spec.
        charge_usd = _status_estimated_charge(get_status(run_id), public_spec, fallback=measured)
        # A cancel can land while this thread persists the recovered metrics (after the late-cancel
        # check above). Re-read before the terminal "done" so a late worker success can't resurrect
        # a user-cancelled run. _RunCancelled is caught below, leaving the cancellation intact.
        if get_status(run_id).state == "cancelled":
            cleanup_terminal = True
            raise _RunCancelled(f"run {run_id} was cancelled")
        _update(run_id, "done", cost_usd=charge_usd, artifacts_dir=artifacts_dir(public_spec))
        cleanup_terminal = True
    except _RunCancelled:
        # this also signals a non-terminal duplicate-supervisor refusal, so only a readable terminal
        # status may authorize cleanup; otherwise retain prior positive terminal knowledge.
        with contextlib.suppress(Exception):
            cleanup_terminal = get_status(run_id).state in TERMINAL_STATES
    except Exception as exc:
        if get_status(run_id).state != "cancelled":
            _update(run_id, "failed", error=str(exc))
        cleanup_terminal = True
    finally:
        # only reap a positively observed terminal run. if the final status read is transiently
        # unavailable, retain prior terminal knowledge without risking a live duplicate supervisor.
        with contextlib.suppress(Exception):
            cleanup_terminal = get_status(run_id).state in TERMINAL_STATES
        if cleanup_terminal:
            with contextlib.suppress(Exception):
                _gc_run_endpoints(worker_spec)
    return status_for_return()


def _deployment_attempt_is_owned(status: RunStatus, deployment: dict) -> bool:
    requested_at = deployment.get("requested_at")
    if requested_at is None:
        return True
    current = status.deployment or {}
    if current == deployment:
        return True
    return (
        current.get("requested_at") == requested_at
        and current.get("state") in _DEPLOYMENT_BUSY_STATES
    )


def _commit_verified_deployment(
    run_id: str,
    deployment: dict,
    *,
    verification_generation: int | None,
    commit: Callable[[], None],
) -> bool:
    if deployment.get("state") not in _RESTORABLE_DEPLOYMENT_STATES:
        raise ValueError("immutable deployment commit requires ready or deployed state")
    revision = deployment.get("adapter_revision")
    parsed_revision = parse_adapter_revision(revision) if isinstance(revision, str) else None
    if parsed_revision is None or parsed_revision[0] != run_id:
        raise ValueError(
            f"immutable deployment commit requires a full same-run adapter revision for {run_id}"
        )
    if verification_generation is None:
        raise ValueError("immutable deployment commit requires a verification generation")
    from flash.runner.verified_revisions import commit_verified_adapter_revision

    return commit_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=verification_generation,
        commit=commit,
    )


def _promote_final_deployment(status: RunStatus, deployment: dict) -> None:
    """Apply the lifecycle state for a final-adapter deployment."""
    # Preserve teardown time for legacy `done` runs (finished_at=None) before deploy bumps updated_at.
    if status.state == "done" and status.finished_at is None and not status.reconciled_at:
        status.finished_at = status.updated_at
    status.deployment = deployment
    status.state = "deployed"


def mark_deployed(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
    *,
    verification_generation: int | None = None,
) -> RunStatus:
    from flash.runner import _STATUS_LOCK, _UNDEPLOYABLE_STATES, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state in _UNDEPLOYABLE_STATES:
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        if not _deployment_attempt_is_owned(status, deployment):
            return status

        def _commit() -> None:
            _promote_final_deployment(status, deployment)
            status.updated_at = time.time()
            _save_status(status)

        if not _commit_verified_deployment(
            run_id,
            deployment,
            verification_generation=verification_generation,
            commit=_commit,
        ):
            return get_status(run_id)
        return status


def mark_checkpoint_deployed(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
    *,
    verification_generation: int | None = None,
) -> RunStatus:
    """Record a checkpoint deployment using the run's current lifecycle state.

    If training has finished by the time serving registration completes, the run behaves like any
    finished deployed run. Otherwise, keep the training state and only attach the deployment record.
    """
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state == "dry_run":
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        if not _deployment_attempt_is_owned(status, deployment):
            return status

        def _commit() -> None:
            if status.state in _FINAL_DEPLOYMENT_STATES:
                _promote_final_deployment(status, deployment)
            else:
                status.deployment = deployment
            status.updated_at = time.time()
            _save_status(status)

        if not _commit_verified_deployment(
            run_id,
            deployment,
            verification_generation=verification_generation,
            commit=_commit,
        ):
            return get_status(run_id)
        return status


def mark_deployment_pending(
    run_id: str, deployment: dict, expect_state: str | None = None
) -> RunStatus:
    """Attach an in-progress deployment record without changing the run lifecycle state."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state == "dry_run":
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        status.deployment = deployment
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_failed(run_id: str, deployment: dict) -> RunStatus:
    """Record a failed deployment attempt while preserving the run lifecycle state."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        current = status.deployment or {}
        # Don't clobber a newer deployment attempt or an explicit undeploy.
        if current.get("state") == "undeployed":
            return status
        if (
            current.get("requested_at") is not None
            and deployment.get("requested_at") is not None
            and current.get("requested_at") != deployment.get("requested_at")
        ):
            return status
        previous = deployment.get("previous_deployment")
        if (
            not deployment.get("activation_outcome_unknown")
            and isinstance(previous, dict)
            and previous.get("state") in _RESTORABLE_DEPLOYMENT_STATES
        ):
            status.deployment = {
                **previous,
                "last_deploy_error": deployment.get("error") or "deployment failed",
                "last_deploy_failed_at": time.time(),
            }
        else:
            failed = dict(deployment)
            if not failed.get("activation_outcome_unknown"):
                failed.pop("previous_deployment", None)
            state = (
                "reconciling"
                if failed.get("activation_outcome_unknown")
                and failed.get("state") == "reconciling"
                else "failed"
            )
            status.deployment = {**failed, "state": state}
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_revocation_failed(run_id: str, error: str) -> RunStatus:
    """Revoke local serving authority while retaining retryable backend cleanup state."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status
    from flash.runner.verified_revisions import invalidate_verified_adapter_revisions

    with _STATUS_LOCK:
        status = get_status(run_id)

        def _commit() -> None:
            status.deployment = {
                **(status.deployment or {}),
                "state": "revocation_failed",
                "error": error,
                "retryable": True,
                "updated_at": time.time(),
            }
            status.updated_at = time.time()
            _save_status(status)

        invalidate_verified_adapter_revisions(run_id, commit=_commit)
        return status


def mark_undeployed(run_id: str) -> RunStatus:
    """Record an explicit undeploy; live final-adapter deployments return to `done`."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status
    from flash.runner.verified_revisions import invalidate_verified_adapter_revisions

    with _STATUS_LOCK:
        status = get_status(run_id)

        def _commit() -> None:
            if status.deployment:
                status.deployment = {**status.deployment, "state": "undeployed"}
            if status.state == "deployed":
                status.state = "done"
            status.updated_at = time.time()
            _save_status(status)

        invalidate_verified_adapter_revisions(run_id, commit=_commit)
        return status


def mark_deployment_undeployed(run_id: str) -> RunStatus:
    """Flip ONLY the deployment field to ``undeployed``, leaving the run's state untouched.

    Used by cancel_run — unlike mark_undeployed, never asserts or changes the run state,
    so it works even after a racing mark_undeployed has already written terminal `done`.
    """
    from flash.runner import _STATUS_LOCK, _save_status, get_status
    from flash.runner.verified_revisions import invalidate_verified_adapter_revisions

    with _STATUS_LOCK:
        status = get_status(run_id)

        def _commit() -> None:
            if status.deployment:
                status.deployment = {**status.deployment, "state": "undeployed"}
                status.updated_at = time.time()
                _save_status(status)

        invalidate_verified_adapter_revisions(run_id, commit=_commit)
        return status
