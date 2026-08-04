"""Deploy / cancel / recover state transitions for a run.

Store helpers and the lifecycle functions (``_run_training`` / ``_gc_run_endpoints``) are
pulled in via FUNCTION-LOCAL lazy ``from flash.runner import ...`` imports — never at module
level — for the same two reasons as ``lifecycle.py``: avoid a partially-initialized-package
import cycle, and keep the test monkeypatches (e.g. ``flash.runner._gc_run_endpoints``)
reachable through the package global rather than a statically-bound copy.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from flash.providers._deadline import deadline_kwargs
from flash.providers._poll import _attempt_int
from flash.schema import format_adapter_revision, parse_adapter_revision
from flash.spec import JobSpec

if TYPE_CHECKING:
    from flash.runner import RunStatus

_FINAL_DEPLOYMENT_STATES = frozenset({"done", "deployed"})
_RESTORABLE_DEPLOYMENT_STATES = frozenset({"ready", "deployed"})
_DEPLOYMENT_BUSY_STATES = frozenset({"queued", "smoke_testing", "reconciling"})
_REVOCATION_RETRY_STATE = "revocation_failed"
_INACTIVE_DEPLOYMENT_STATES = frozenset({"undeployed", "dry_run"})
_ATTACH_RECONCILE_INTERVAL_S = 120.0
_ATTACH_RECONCILING: set[str] = set()
_ATTACH_RECONCILING_LOCK = threading.Lock()


class DeploymentRevocationError(RuntimeError):
    """Backend revocation is unconfirmed after local serving authority was removed."""

    def __init__(self, run_id: str, error: str):
        super().__init__(
            f"deployment_revocation_failed: local authorization for {run_id} was revoked, but "
            f"backend disablement is unconfirmed: {error}; retry cancellation"
        )
        self.run_id = run_id
        self.retryable = True


_BackendOutcome = Literal["confirmed", "not_required", "not_attempted"]
_BACKEND_OUTCOMES = frozenset({"confirmed", "not_required", "not_attempted"})
_UNKNOWN_ALIAS_UNCHECKED = object()


class DeploymentStatePersistenceError(RuntimeError):
    """Serving cleanup was resolved, but local deployment state was not persisted."""

    def __init__(self, run_id: str, error: str, *, backend_outcome: _BackendOutcome):
        if backend_outcome not in _BACKEND_OUTCOMES:
            raise ValueError(f"invalid backend outcome: {backend_outcome!r}")
        if backend_outcome == "confirmed":
            detail = (
                "backend disablement was confirmed, but the local inactive state could not be "
                f"persisted: {error}; retry cancellation to reconcile local state"
            )
        elif backend_outcome == "not_required":
            detail = (
                "backend revocation was not required, but generation-only local persistence "
                f"failed: {error}; retry cancellation to reconcile local state"
            )
        else:
            detail = (
                "local serving authority could not be fenced before backend cleanup: "
                f"{error}; backend disablement was not attempted"
            )
        super().__init__(f"deployment_state_persistence_failed: {run_id}: {detail}")
        self.run_id = run_id
        self.backend_outcome = backend_outcome
        self.retryable = True


def _deployment_state_and_requires_revocation(
    deployment: object,
) -> tuple[str | None, bool]:
    if deployment is None:
        return None, False
    if not isinstance(deployment, dict):
        return None, True
    state = deployment.get("state")
    if not isinstance(state, str):
        return None, True
    return state, state not in _INACTIVE_DEPLOYMENT_STATES


def _is_preservable_checkpoint_deployment(run_id: str, deployment: object) -> bool:
    if not isinstance(deployment, dict):
        return False
    state, _ = _deployment_state_and_requires_revocation(deployment)
    if state not in _RESTORABLE_DEPLOYMENT_STATES:
        return False
    checkpoint_step = deployment.get("checkpoint_step")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        return False
    revision = deployment.get("adapter_revision")
    if not isinstance(revision, str):
        return False
    parsed_revision = parse_adapter_revision(revision)
    if parsed_revision is None:
        return False
    revision_run_id, revision_step, hf_revision = parsed_revision
    if revision_step is None:
        return False
    canonical_revision = format_adapter_revision(revision_run_id, revision_step, hf_revision)
    if (
        revision != canonical_revision
        or revision_run_id != run_id
        or revision_step != checkpoint_step
    ):
        return False
    from flash.runner.verified_revisions import read_verified_adapter_revisions

    try:
        return revision in read_verified_adapter_revisions(run_id)
    except Exception:
        return False


def _preservable_checkpoint_deployment(
    run_id: str,
    deployment: object,
    *,
    live_alias_target: object = _UNKNOWN_ALIAS_UNCHECKED,
) -> dict | None:
    if not isinstance(deployment, dict):
        return None
    if deployment.get("activation_outcome_unknown"):
        if live_alias_target is _UNKNOWN_ALIAS_UNCHECKED:
            return None
        predecessor = deployment.get("previous_deployment")
        if (
            isinstance(predecessor, dict)
            and predecessor.get("adapter_revision") == live_alias_target
            and _is_preservable_checkpoint_deployment(run_id, predecessor)
        ):
            return dict(predecessor)
        return None
    if _is_preservable_checkpoint_deployment(run_id, deployment):
        return dict(deployment)
    if deployment.get("state") in {"queued", "smoke_testing"}:
        predecessor = deployment.get("previous_deployment")
        if _is_preservable_checkpoint_deployment(run_id, predecessor):
            return dict(predecessor)
    return None


def cancel_run(run_id: str) -> RunStatus:
    """Cancel training while preserving verified serving and durable cleanup targets."""
    from flash.runner import (
        TERMINAL_STATES,
        _cleanup_remote_key,
        _drain_cleanup_remotes,
        _gc_run_endpoints,
        _record_cleanup_remote,
        _remote_resource_identity,
        _report_status,
        _save_status_unlocked,
        _snapshot_cleanup_remotes,
        _status_guard,
        _update,
        actual_steps_run,
        charge_usd_for_spec,
        effective_spec_from_status,
        get_status,
        mark_checkpoint_deployed,
        mark_deployment_revocation_failed,
        mark_deployment_undeployed,
        read_verified_adapter_revisions,
        verified_adapter_revision_generation,
    )
    from flash.server import db as server_db
    from flash.server._locks import _deploy_lock

    # fence teacher authority before waiting on deployment locks or slow provider teardown.
    server_db.revoke_teacher_capabilities_for_run(run_id)

    def _clear_exact_remote(expected_remote: dict) -> bool:
        expected_identity = _remote_resource_identity(expected_remote)
        if expected_identity is None:
            return False
        report_status = None
        with _status_guard(run_id):
            current = get_status(run_id)
            if _remote_resource_identity(current.remote) != expected_identity:
                return False
            current.remote = None
            current.updated_at = time.time()
            _save_status_unlocked(current)
            report_status = current
        if report_status is not None:
            _report_status(report_status)
        return True

    def _drain_confirmed_cleanup() -> set[tuple]:
        try:
            before = _snapshot_cleanup_remotes(run_id)
            _drain_cleanup_remotes(run_id)
            remaining_keys = {
                key
                for record in _snapshot_cleanup_remotes(run_id)
                if (key := _cleanup_remote_key(record)) is not None
            }
        except Exception:
            return set()
        confirmed_identities = set()
        for record in before:
            key = _cleanup_remote_key(record)
            if key is None or key in remaining_keys:
                continue
            identity = _remote_resource_identity(record)
            if identity is not None:
                confirmed_identities.add(identity)
            _clear_exact_remote(record)
        return confirmed_identities

    initial_status = get_status(run_id)
    confirmed_cleanup_identities = set()
    if initial_status.state == "cancelled":
        confirmed_cleanup_identities = _drain_confirmed_cleanup()
        initial_status = get_status(run_id)
    _, initial_active = _deployment_state_and_requires_revocation(initial_status.deployment)
    if initial_status.state in TERMINAL_STATES and not initial_active:
        return initial_status

    entered_deployed = initial_status.state == "deployed"
    effective_spec = None
    with contextlib.suppress(Exception):
        effective_spec = effective_spec_from_status(initial_status)
    cleanup_spec = effective_spec
    if cleanup_spec is None:
        with contextlib.suppress(Exception):
            cleanup_spec = JobSpec.from_dict(initial_status.spec)
    bill_cancel = (
        bool(initial_status.billing_context)
        and initial_status.state not in TERMINAL_STATES
        and not entered_deployed
    )

    deploy_lock = _deploy_lock(run_id)
    captured_contended_attempt: dict | None = None
    contended_active_attempt = False
    contended_predecessor: dict | None = None
    contended_predecessor_recommitted = False
    lock_acquired = deploy_lock.acquire(blocking=False)
    try:
        if not lock_acquired:
            prelock_status = get_status(run_id)
            prelock_raw_deployment = prelock_status.deployment
            prelock_deployment = (
                dict(prelock_raw_deployment) if isinstance(prelock_raw_deployment, dict) else None
            )
            _, prelock_active = _deployment_state_and_requires_revocation(prelock_raw_deployment)
            must_fence = (
                prelock_status.state != "dry_run"
                and prelock_active
                and not _is_preservable_checkpoint_deployment(run_id, prelock_deployment)
            )
            if must_fence:
                contended_active_attempt = True
                captured_contended_attempt = prelock_deployment
                if prelock_deployment is not None:
                    predecessor = prelock_deployment.get("previous_deployment")
                    if _is_preservable_checkpoint_deployment(run_id, predecessor):
                        contended_predecessor = dict(predecessor)
                try:
                    fenced_status = mark_deployment_revocation_failed(
                        run_id,
                        "backend revocation pending: cancellation fenced an in-progress deployment",
                    )
                except Exception as exc:
                    raise DeploymentStatePersistenceError(
                        run_id, str(exc), backend_outcome="not_attempted"
                    ) from exc
                fenced_deployment = (
                    dict(fenced_status.deployment)
                    if isinstance(fenced_status.deployment, dict)
                    else None
                )
                if contended_predecessor is not None:
                    if fenced_deployment is not None:
                        try:
                            restored_status = mark_checkpoint_deployed(
                                run_id,
                                contended_predecessor,
                                verification_generation=verified_adapter_revision_generation(
                                    run_id
                                ),
                                owner_deployment=fenced_deployment,
                                retain_only_revision=True,
                            )
                            predecessor_revision = contended_predecessor.get("adapter_revision")
                            contended_predecessor_recommitted = (
                                restored_status.deployment == contended_predecessor
                                and isinstance(predecessor_revision, str)
                                and _is_preservable_checkpoint_deployment(
                                    run_id, restored_status.deployment
                                )
                                and read_verified_adapter_revisions(run_id)
                                == frozenset({predecessor_revision})
                            )
                        except Exception:
                            contended_predecessor_recommitted = False
                    if not contended_predecessor_recommitted:
                        current = get_status(run_id)
                        current_deployment = (
                            current.deployment if isinstance(current.deployment, dict) else {}
                        )
                        still_fenced = (
                            current_deployment.get("state") == _REVOCATION_RETRY_STATE
                            and read_verified_adapter_revisions(run_id) == frozenset()
                        )
                        if not still_fenced:
                            try:
                                mark_deployment_revocation_failed(
                                    run_id,
                                    "backend revocation pending: verified predecessor restoration failed",
                                )
                            except Exception as exc:
                                raise DeploymentStatePersistenceError(
                                    run_id, str(exc), backend_outcome="not_attempted"
                                ) from exc
            deploy_lock.acquire()
            lock_acquired = True

        # close the race where submission held the deploy lock, minted a capability after the
        # pre-lock fence, and released the lock only after persisting its provider handle.
        server_db.revoke_teacher_capabilities_for_run(run_id)
        status = get_status(run_id)
        entered_deployed = entered_deployed or status.state == "deployed"

        from flash.providers.base import JobHandle
        from flash.runner.lifecycle import _strict_teardown_handle

        def _teardown_or_preserve_remote(remote: dict) -> bool:
            try:
                resource_deleted = _strict_teardown_handle(JobHandle.from_dict(remote), run_id)
            except Exception:
                if not _record_cleanup_remote(run_id, remote):
                    raise RuntimeError(
                        f"run {run_id} teardown was unconfirmed and its exact cleanup target "
                        "could not be preserved"
                    ) from None
                return False
            if not resource_deleted and not _record_cleanup_remote(run_id, remote):
                raise RuntimeError(
                    f"run {run_id} leaked endpoint cleanup target could not be preserved"
                )
            return True

        processed_remote_identities = set()
        while True:
            status = get_status(run_id)
            remote = status.remote or {}
            identity = _remote_resource_identity(remote)
            if not remote or identity in processed_remote_identities:
                break
            processed_remote_identities.add(identity)
            if identity in confirmed_cleanup_identities:
                _clear_exact_remote(remote)
                continue
            remote_safe_to_clear = _teardown_or_preserve_remote(remote)
            if remote_safe_to_clear:
                _clear_exact_remote(remote)
                continue
            latest_remote = get_status(run_id).remote or {}
            if _remote_resource_identity(latest_remote) != identity:
                continue
            break
        if cleanup_spec is not None:
            with contextlib.suppress(Exception):
                _gc_run_endpoints(cleanup_spec)

        status = get_status(run_id)
        entered_deployed = entered_deployed or status.state == "deployed"
        _, active_deployment = _deployment_state_and_requires_revocation(status.deployment)
        live_alias_target: object = _UNKNOWN_ALIAS_UNCHECKED
        unknown_activation = isinstance(status.deployment, dict) and status.deployment.get(
            "activation_outcome_unknown"
        )
        if status.state != "dry_run" and (contended_active_attempt or unknown_activation):
            try:
                from flash.serve.deploy import adapter_alias_target

                live_alias_target = adapter_alias_target(run_id)
            except Exception:
                live_alias_target = None
        if status.state == "dry_run":
            preserved_checkpoint = None
        elif contended_active_attempt:
            predecessor_revision = (
                contended_predecessor.get("adapter_revision")
                if contended_predecessor is not None
                else None
            )
            if (
                contended_predecessor_recommitted
                and isinstance(captured_contended_attempt, dict)
                and isinstance(predecessor_revision, str)
                and live_alias_target == predecessor_revision
                and status.deployment == contended_predecessor
                and _is_preservable_checkpoint_deployment(run_id, status.deployment)
            ):
                preserved_checkpoint = dict(contended_predecessor)
            else:
                preserved_checkpoint = None
        else:
            preserved_checkpoint = _preservable_checkpoint_deployment(
                run_id,
                status.deployment,
                live_alias_target=live_alias_target,
            )
        if preserved_checkpoint is not None and status.deployment != preserved_checkpoint:
            intended_revision = preserved_checkpoint.get("adapter_revision")
            owner_deployment = status.deployment if isinstance(status.deployment, dict) else None
            try:
                status = mark_checkpoint_deployed(
                    run_id,
                    preserved_checkpoint,
                    verification_generation=verified_adapter_revision_generation(run_id),
                    owner_deployment=owner_deployment,
                )
            except Exception:
                try:
                    status = get_status(run_id)
                    stored_deployment = status.deployment
                    if (
                        not isinstance(stored_deployment, dict)
                        or stored_deployment.get("adapter_revision") != intended_revision
                        or not _is_preservable_checkpoint_deployment(run_id, stored_deployment)
                    ):
                        preserved_checkpoint = None
                    else:
                        preserved_checkpoint = dict(stored_deployment)
                except Exception:
                    preserved_checkpoint = None
            else:
                if (
                    status.deployment != preserved_checkpoint
                    or not _is_preservable_checkpoint_deployment(run_id, status.deployment)
                ):
                    preserved_checkpoint = None
            entered_deployed = entered_deployed or status.state == "deployed"
        if preserved_checkpoint is not None:
            preserved_revision = preserved_checkpoint.get("adapter_revision")
            owner_deployment = status.deployment if isinstance(status.deployment, dict) else None
            try:
                status = mark_checkpoint_deployed(
                    run_id,
                    preserved_checkpoint,
                    verification_generation=verified_adapter_revision_generation(run_id),
                    owner_deployment=owner_deployment,
                    retain_only_revision=True,
                )
                if (
                    status.deployment != preserved_checkpoint
                    or not isinstance(preserved_revision, str)
                    or read_verified_adapter_revisions(run_id) != frozenset({preserved_revision})
                ):
                    raise RuntimeError(
                        "authoritative checkpoint preservation did not prune verified revisions"
                    )
            except Exception as exc:
                raise DeploymentStatePersistenceError(
                    run_id, str(exc), backend_outcome="not_required"
                ) from exc
            entered_deployed = entered_deployed or status.state == "deployed"
        preserve_checkpoint = preserved_checkpoint is not None
        backend_error: Exception | None = None
        persistence_error: tuple[Exception, _BackendOutcome] | None = None

        if not preserve_checkpoint:
            if active_deployment:
                already_fenced = (
                    isinstance(status.deployment, dict)
                    and status.deployment.get("state") == _REVOCATION_RETRY_STATE
                )
                if not already_fenced:
                    try:
                        mark_deployment_revocation_failed(
                            run_id,
                            "backend revocation pending: local serving authority was revoked before "
                            "backend disablement",
                        )
                    except Exception as exc:
                        raise DeploymentStatePersistenceError(
                            run_id, str(exc), backend_outcome="not_attempted"
                        ) from exc
                try:
                    from flash.serve.deploy import undeploy_adapter

                    undeploy_adapter(run_id)
                except Exception as exc:
                    backend_error = exc
                    with contextlib.suppress(Exception):
                        mark_deployment_revocation_failed(run_id, str(exc))
                else:
                    try:
                        mark_deployment_undeployed(run_id)
                    except Exception as exc:
                        persistence_error = (exc, "confirmed")
            else:
                try:
                    mark_deployment_undeployed(run_id)
                except Exception as exc:
                    persistence_error = (exc, "not_required")

        cancel_charge_usd: float | None = None
        billing_diagnostic: dict = {}
        if bill_cancel:
            if effective_spec is not None:
                estimated_charge = charge_usd_for_spec(
                    effective_spec,
                    steps=actual_steps_run(get_status(run_id)),
                    fallback=float("nan"),
                )
                if math.isfinite(estimated_charge):
                    cancel_charge_usd = estimated_charge
                else:
                    cancel_charge_usd = 0.0
                    billing_diagnostic = {
                        "billing_state": "failed",
                        "billing_error": (
                            "cancellation charge was not computed because pricing failed; "
                            "teardown was still attempted"
                        ),
                    }
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
        preserve_retry_state = (
            persistence_error is not None and persistence_error[1] == "not_required"
        )
        if not preserve_retry_state:
            update_context = (
                contextlib.suppress(Exception)
                if persistence_error is not None or backend_error is not None
                else contextlib.nullcontext()
            )
            with update_context:
                _update(run_id, "cancelled", allow_from_terminal=entered_deployed, **cancel_updates)

        with contextlib.suppress(Exception):
            from flash.server.checkpoints import register_checkpoints_best_effort

            register_checkpoints_best_effort(get_status(run_id))

        if backend_error is not None:
            raise DeploymentRevocationError(run_id, str(backend_error)) from backend_error
        if persistence_error is not None:
            error, backend_outcome = persistence_error
            raise DeploymentStatePersistenceError(
                run_id, str(error), backend_outcome=backend_outcome
            ) from error
        return get_status(run_id)
    finally:
        if lock_acquired:
            deploy_lock.release()


def _resume_after_confirmed_teardown(
    run_id: str,
    worker_spec: JobSpec,
    persisted_remote: dict,
    next_attempt: int,
    code_prefix: str | None,
    log,
    *,
    failure: str,
) -> RunStatus:
    """CAS-clear one captured remote, then resume its next attempt exactly once."""
    from flash.runner import (
        _compare_and_clear_remote,
        _compare_and_fail_remote,
        _load_run_deadline_at,
        _record_cleanup_remote,
        _run_training,
        _RunCancelled,
        _spec_with_remaining_wall,
        _update,
        _verified_opd_next_attempt,
        flash_code_prefix,
        get_status,
        reallocation_spec_from_status,
    )

    if int(worker_spec.gpu.max_retries) == 0:
        _compare_and_fail_remote(run_id, persisted_remote, failure)
        print(
            f"attach: {run_id} exhausted its one-shot retry budget; not resubmitting",
            file=log,
        )
        return get_status(run_id)
    if worker_spec.algorithm == "opd":
        verified_next_attempt = _verified_opd_next_attempt(run_id)
        if verified_next_attempt != next_attempt:
            raise RuntimeError(
                "persisted opd attempt identity does not match the attached worker; "
                "replacement is blocked"
            )
        next_attempt = verified_next_attempt
    try:
        _spec_with_remaining_wall(worker_spec, require_provider_minimum=True)
    except RuntimeError as exc:
        _compare_and_fail_remote(run_id, persisted_remote, str(exc))
        print(f"attach: {run_id} {exc}", file=log)
        return get_status(run_id)
    worker_spec = reallocation_spec_from_status(get_status(run_id), verify_source=True)
    if not _compare_and_clear_remote(run_id, persisted_remote):
        print(
            f"attach: {run_id} persisted remote changed before clear; not resuming",
            file=log,
        )
        return get_status(run_id)
    if not _update(run_id, "running"):
        print(f"attach: {run_id} went terminal during recovery; not resuming", file=log)
        return get_status(run_id)
    print(
        f"attach: {run_id} resubmitting from the latest checkpoint before the "
        "run-global wall deadline",
        file=log,
    )
    if code_prefix is None:
        from flash.providers._worker import upload_code

        code_prefix = flash_code_prefix()
        try:
            upload_code(
                worker_spec.train.hf_repo,
                code_prefix=code_prefix,
                **deadline_kwargs(upload_code, _load_run_deadline_at(run_id)),
            )
        except Exception as exc:
            _compare_and_fail_remote(run_id, None, str(exc))
            raise
    try:
        _run_training(
            worker_spec,
            log,
            prior_cost=float(get_status(run_id).cost_usd or 0.0),
            code_prefix=code_prefix,
            attempt_start=next_attempt,
        )
    except _RunCancelled:
        raise
    except Exception as exc:
        current = get_status(run_id)
        current_remote = current.remote
        current_attempt = (
            _attempt_int(current_remote.get("attempt"))
            if isinstance(current_remote, dict)
            else None
        )
        if current_remote is None or (
            current_attempt is not None and current_attempt >= next_attempt
        ):
            if current_remote is not None:
                with contextlib.suppress(Exception):
                    _record_cleanup_remote(run_id, current_remote)
            _compare_and_fail_remote(run_id, current_remote, str(exc))
        raise
    return get_status(run_id)


def _reconcile_attached_remote(
    run_id: str,
    expected_remote: dict,
    worker_spec: JobSpec,
    next_attempt: int,
    code_prefix: str | None,
    log,
    failure: str,
) -> None:
    """Reconcile one exact maybe-live attempt until it is gone or the wall deadline expires."""
    from flash.providers.base import JobHandle
    from flash.runner import (
        TERMINAL_STATES,
        _compare_and_fail_remote,
        _load_run_deadline_at,
        _record_cleanup_remote,
        _remote_resource_identity,
        _spec_with_remaining_wall,
        get_status,
    )
    from flash.runner.lifecycle import (
        _RECOVERY_MARKER_GRACE_S,
        _adopt_completed_attempt,
        _completed_attempt_metrics,
        _CompletedAttemptPending,
        _runpod_completed_metrics,
        _strict_teardown_handle,
        _worker_provably_gone,
    )

    expected_identity = _remote_resource_identity(expected_remote)
    handle = JobHandle.from_dict(expected_remote)
    while True:
        try:
            status = get_status(run_id)
        except Exception:
            time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
            continue
        if status.state in TERMINAL_STATES:
            return
        if _remote_resource_identity(status.remote) != expected_identity:
            return
        try:
            deadline_at = _load_run_deadline_at(run_id)
        except Exception as exc:
            try:
                if _compare_and_fail_remote(run_id, expected_remote, str(exc)):
                    return
            except Exception:
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            return
        try:
            completed_metrics = _runpod_completed_metrics(
                expected_remote,
                deadline_at=deadline_at,
            )
        except _CompletedAttemptPending:
            # the queue job already completed but its metrics are still landing; keep
            # reconciling (do not tear it down) until they are readable -- but do not sleep
            # past the grace cutoff, or a run that fails at the cutoff keeps reconciling a
            # full interval beyond it before terminating.
            pending_interval = _ATTACH_RECONCILE_INTERVAL_S
            if deadline_at is not None:
                pending_interval = min(
                    pending_interval,
                    max(0.0, deadline_at + _RECOVERY_MARKER_GRACE_S - time.time()),
                )
            time.sleep(pending_interval)
            continue
        if completed_metrics is not None:
            try:
                if _adopt_completed_attempt(
                    run_id,
                    worker_spec,
                    expected_remote,
                    completed_metrics,
                    log=log,
                ):
                    return
            except Exception:
                pass
            # the job completed with metrics; keep retrying adoption (e.g. across a
            # transient cleanup failure) until it sticks -- but never past the wall
            # deadline plus the recovery grace, or a persistently failing adoption
            # would leave the run non-terminal forever. past the grace, preserve the
            # remote for cost reconciliation and fail the run.
            if time.time() >= deadline_at + _RECOVERY_MARKER_GRACE_S:
                # best-effort: preserve the remote for cost reconciliation, but do NOT
                # gate termination on it -- a persistently failing cleanup-persist must
                # not leave the run non-terminal forever (the whole point of the grace
                # cutoff). attempt the cleanup record, then fail the run regardless.
                with contextlib.suppress(Exception):
                    _record_cleanup_remote(run_id, expected_remote)
                try:
                    if _compare_and_fail_remote(
                        run_id,
                        expected_remote,
                        "completed attempt could not be adopted within the recovery grace window",
                    ):
                        return
                except Exception:
                    pass
                # past the grace window the terminal CAS is the only exit; if it raised or
                # lost the compare-and-swap, rate-limit the retry at the full reconcile
                # interval. remaining grace is <= 0 here, so falling through to the shared
                # sleep below would sleep 0 and busy-spin the reconciler.
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            remaining_grace = deadline_at + _RECOVERY_MARKER_GRACE_S - time.time()
            time.sleep(min(_ATTACH_RECONCILE_INTERVAL_S, max(0.0, remaining_grace)))
            continue
        if time.time() >= deadline_at:
            try:
                metrics = _completed_attempt_metrics(
                    worker_spec,
                    provider=handle.provider,
                    attempt=next_attempt - 1,
                    launch_floor=float(expected_remote["started_ts"]),
                    deadline_at=deadline_at,
                    log=log,
                )
            except Exception:
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            if metrics is not None:
                try:
                    cleanup_preserved = _record_cleanup_remote(run_id, expected_remote)
                    adopted = cleanup_preserved and _adopt_completed_attempt(
                        run_id,
                        worker_spec,
                        expected_remote,
                        metrics,
                        log=log,
                    )
                except Exception:
                    time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                    continue
                if adopted:
                    return
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            try:
                cleanup_preserved = _record_cleanup_remote(run_id, expected_remote)
            except Exception:
                cleanup_preserved = False
            if not cleanup_preserved:
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            try:
                if _compare_and_fail_remote(run_id, expected_remote, failure):
                    return
            except Exception:
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            return
        delay = min(_ATTACH_RECONCILE_INTERVAL_S, max(0.0, deadline_at - time.time()))
        if delay > 0:
            time.sleep(delay)
            if time.time() >= deadline_at:
                continue
        if time.time() < deadline_at:
            # if a replacement cannot meet the 60-second provider minimum yet the run
            # wall deadline is still open, keep reconciling (probe for completion) rather
            # than tearing down and failing early — mirror handle-less recovery, which
            # waits until the wall deadline.
            try:
                _spec_with_remaining_wall(worker_spec, require_provider_minimum=True)
            except RuntimeError:
                # cap the reconcile wait at the wall deadline so a near-deadline wake does
                # not overshoot the run's wall deadline by a full interval.
                time.sleep(min(_ATTACH_RECONCILE_INTERVAL_S, max(0.0, deadline_at - time.time())))
                continue
        try:
            resource_deleted = _strict_teardown_handle(handle, run_id)
            worker_gone = True
        except Exception:
            resource_deleted = False
            worker_gone = _worker_provably_gone(run_id, handle)
        if not worker_gone:
            continue
        if handle.provider == "runpod" and not resource_deleted:
            try:
                cleanup_preserved = _record_cleanup_remote(run_id, expected_remote)
            except Exception:
                cleanup_preserved = False
            if not cleanup_preserved:
                continue
        try:
            _resume_after_confirmed_teardown(
                run_id,
                worker_spec,
                expected_remote,
                next_attempt,
                code_prefix,
                log,
                failure=failure,
            )
        except Exception as exc:
            try:
                current = get_status(run_id)
            except Exception:
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            if current.state in TERMINAL_STATES:
                return
            if current.remote is None:
                try:
                    if _compare_and_fail_remote(run_id, None, str(exc)):
                        return
                except Exception:
                    time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                    continue
                return
            if _remote_resource_identity(current.remote) != expected_identity:
                return
            time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
            continue
        return


def _schedule_attach_reconciliation(
    run_id: str,
    expected_remote: dict,
    worker_spec: JobSpec,
    next_attempt: int,
    code_prefix: str | None,
    log,
    failure: str,
) -> bool:
    """Schedule one in-process reconciler for a captured remote identity."""
    with _ATTACH_RECONCILING_LOCK:
        if run_id in _ATTACH_RECONCILING:
            return False
        _ATTACH_RECONCILING.add(run_id)

    def run() -> None:
        try:
            _reconcile_attached_remote(
                run_id,
                expected_remote,
                worker_spec,
                next_attempt,
                code_prefix,
                log,
                failure,
            )
        finally:
            try:
                from flash.runner import TERMINAL_STATES, _gc_run_endpoints, get_status

                if get_status(run_id).state in TERMINAL_STATES:
                    _gc_run_endpoints(worker_spec)
            except Exception:
                pass
            with _ATTACH_RECONCILING_LOCK:
                _ATTACH_RECONCILING.discard(run_id)

    try:
        threading.Thread(target=run, daemon=True).start()
    except Exception:
        with _ATTACH_RECONCILING_LOCK:
            _ATTACH_RECONCILING.discard(run_id)
        raise
    return True


def attach_run(run_id: str, log_stream=None) -> RunStatus:
    """Re-attach to a run's remote job from any process (after a client crash/restart)."""
    import sys

    from flash.runner import (
        TERMINAL_STATES,
        _compare_and_fail_remote,
        _gc_run_endpoints,
        _load_run_deadline_at,
        _record_cleanup_remote,
        _RunCancelled,
        _spec_with_remaining_wall,
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

    # Seed from the lossy public view so the except/finally handlers always have a spec, then
    # upgrade to the authoritative worker spec (real run_id + managed fields) inside the try.
    try:
        worker_spec = JobSpec.from_dict(status.spec)
    except Exception as exc:
        # This parse is above the try below, so a raise here escapes every handler -- and attach_run
        # is dispatched on a daemon thread, so the failure is silent: the run stays nonterminal with
        # a live handle and its worker keeps billing. A spec stops parsing when the plane upgrades
        # past an algorithm a still-in-flight run was accepted under (codex[bot]). It cannot be
        # resumed, so fail it closed and tear the worker down. `_gc_run_endpoints` needs a parsed
        # spec we do not have; the endpoint name is derived from the run id plus GPU class, both
        # readable from the raw persisted status, which is the same route recover_runs takes for its
        # own unparseable-spec branch.
        from flash.providers.base import JobHandle
        from flash.runner.lifecycle import _strict_teardown_handle

        detail = f"unrecoverable: persisted spec is malformed: {exc}"
        persisted_remote = dict(status.remote)
        try:
            resource_deleted = _strict_teardown_handle(
                JobHandle.from_dict(persisted_remote), run_id
            )
        except Exception:
            resource_deleted = False
        # Tear down BEFORE the state write and hand an unconfirmed deletion to the cleanup drainer:
        # failing the run first would drop the last record of a worker we have not proven gone.
        if not resource_deleted:
            _record_cleanup_remote(run_id, persisted_remote)
        _compare_and_fail_remote(run_id, persisted_remote, detail)
        with contextlib.suppress(Exception):
            gpu_type = (status.spec.get("gpu") or {}).get("type")
            if gpu_type:
                from flash.providers.runpod.train import terminate_endpoint

                terminate_endpoint(gpu_type, run_id)
        print(f"attach: {run_id} {detail}", file=log_stream or sys.stderr)
        return get_status(run_id)
    persisted_remote = dict(status.remote)
    next_attempt = 0
    code_prefix = None
    failure = "provider attempt failed"
    log = log_stream or sys.stderr
    from flash.providers import get_provider
    from flash.providers.base import JobHandle
    from flash.runner.lifecycle import (
        _adopt_completed_attempt,
        _completed_attempt_metrics,
        _CompletedAttemptPending,
        _runpod_completed_metrics,
        _strict_teardown_handle,
        _worker_provably_gone,
    )

    try:
        worker_spec = effective_spec_from_status(status)
        remote = dict(persisted_remote)
        seed = int(remote.pop("seed", worker_spec.seed))
        code_prefix = remote.pop("code_prefix", None)
        provider_name = remote.get("provider")
        if not isinstance(provider_name, str) or not provider_name:
            raise ValueError("persisted provider identity is missing or invalid")
        recovered_attempt = _attempt_int(remote.get("attempt"))
        if recovered_attempt is None:
            raise ValueError("persisted attempt identity is missing or invalid")
        next_attempt = recovered_attempt + 1
        allocated_gpu = remote.pop("allocated_gpu", None)
        handle = JobHandle.from_dict(remote)
        try:
            poll_spec = _spec_with_remaining_wall(worker_spec, require_provider_minimum=False)
        except RuntimeError as exc:
            deadline_at = _load_run_deadline_at(run_id)
            metrics = _runpod_completed_metrics(
                persisted_remote,
                deadline_at=deadline_at,
            )
            started_ts = persisted_remote.get("started_ts")
            if metrics is None and started_ts is not None:
                metrics = _completed_attempt_metrics(
                    worker_spec,
                    provider=handle.provider,
                    attempt=recovered_attempt,
                    launch_floor=float(started_ts),
                    deadline_at=deadline_at,
                    log=log,
                )
            if metrics is not None:
                try:
                    adopted = _adopt_completed_attempt(
                        run_id,
                        worker_spec,
                        persisted_remote,
                        metrics,
                        log=log,
                    )
                except Exception:
                    adopted = False
                if adopted:
                    print(
                        f"attach: {run_id} adopted a completed attempt at the wall deadline",
                        file=log,
                    )
                    return status_for_return()
                # completed work whose adoption is a transient defer (e.g. a cleanup blip) must NEVER be
                # torn down at the wall deadline; defer to background reconciliation, which retries
                # adoption until the deadline like the in-loop completion path.
                _schedule_attach_reconciliation(
                    run_id, persisted_remote, worker_spec, next_attempt, code_prefix, log, str(exc)
                )
                print(
                    f"attach: {run_id} completed RunPod work at the wall deadline; deferring adoption to reconciliation",
                    file=log,
                )
                return status_for_return()
            try:
                resource_deleted = _strict_teardown_handle(handle, run_id)
            except Exception:
                resource_deleted = False
            if not resource_deleted:
                _record_cleanup_remote(run_id, persisted_remote)
            _compare_and_fail_remote(run_id, persisted_remote, str(exc))
            print(f"attach: {run_id} {exc}", file=log)
            return status_for_return()
        print(f"attaching to {run_id}: provider={handle.provider} {handle.data}", file=log)
        res = get_provider(handle.provider).poll(
            handle,
            poll_spec,
            seed,
            log=log,
            _deadline_at=_load_run_deadline_at(run_id),
        )
        if get_status(run_id).state == "cancelled":
            return status_for_return()
        if not res.ok:
            failure = f"{res.failure or 'job_failed'}: {res.detail or 'provider attempt failed'}"
            print(f"attach: {run_id} ended ({res.failure}); evaluating recovery", file=log)
            completed_metrics = _runpod_completed_metrics(
                persisted_remote,
                deadline_at=_load_run_deadline_at(run_id),
            )
            if completed_metrics is not None:
                # the job completed. adoption may return False (a transient defer, e.g. a
                # cleanup-remote CAS lost) OR raise (e.g. a durable-confirmation exception);
                # treat BOTH the same -- never tear down completed work, defer to background
                # reconciliation, which retries adoption until the deadline like
                # _reconcile_attached_remote.
                try:
                    adopted = _adopt_completed_attempt(
                        run_id,
                        worker_spec,
                        persisted_remote,
                        completed_metrics,
                        log=log,
                    )
                except Exception:
                    adopted = False
                if adopted:
                    print(f"attach: {run_id} adopted completed RunPod work", file=log)
                    return status_for_return()
                _schedule_attach_reconciliation(
                    run_id,
                    persisted_remote,
                    worker_spec,
                    next_attempt,
                    code_prefix,
                    log,
                    failure,
                )
                print(
                    f"attach: {run_id} completed RunPod work; deferring adoption to reconciliation",
                    file=log,
                )
                return status_for_return()
            try:
                resource_deleted = _strict_teardown_handle(handle, run_id)
                worker_gone = True
            except Exception:
                resource_deleted = False
                worker_gone = _worker_provably_gone(run_id, handle)
            if (
                worker_gone
                and handle.provider == "runpod"
                and not resource_deleted
                and not _record_cleanup_remote(run_id, persisted_remote)
            ):
                raise RuntimeError("leaked endpoint cleanup target could not be persisted")
            if worker_gone:
                return _resume_after_confirmed_teardown(
                    run_id,
                    worker_spec,
                    persisted_remote,
                    next_attempt,
                    code_prefix,
                    log,
                    failure=failure,
                )
            _schedule_attach_reconciliation(
                run_id,
                persisted_remote,
                worker_spec,
                next_attempt,
                code_prefix,
                log,
                failure,
            )
            print(
                f"attach: {run_id} {handle.provider} teardown unconfirmed; "
                "reconciling the captured remote without resuming over a possibly-live resource",
                file=log,
            )
            return status_for_return()
        if allocated_gpu and isinstance(res.metrics, dict):
            res.metrics.setdefault("allocated_gpu", allocated_gpu)
        if not _adopt_completed_attempt(
            run_id,
            worker_spec,
            persisted_remote,
            res.metrics,
            log=log,
        ):
            print(
                f"attach: {run_id} persisted remote changed before completion adoption",
                file=log,
            )
        return status_for_return()
    except _CompletedAttemptPending as exc:
        _schedule_attach_reconciliation(
            run_id,
            persisted_remote,
            worker_spec,
            next_attempt,
            code_prefix,
            log,
            str(exc),
        )
        print(
            f"attach: {run_id} completed successfully; waiting for metrics.json visibility",
            file=log,
        )
        return status_for_return()
    except _RunCancelled:
        with contextlib.suppress(Exception):
            cleanup_terminal = get_status(run_id).state in TERMINAL_STATES
    except Exception as exc:
        try:
            _record_cleanup_remote(run_id, persisted_remote)
            # cas-only fail: no-ops if a concurrent cancel already cleared this remote (so a user
            # cancel is never overwritten as failed); the terminal gc below reaps the box by run label.
            _compare_and_fail_remote(run_id, persisted_remote, str(exc))
        except Exception:
            if next_attempt > 0:
                _schedule_attach_reconciliation(
                    run_id,
                    persisted_remote,
                    worker_spec,
                    next_attempt,
                    code_prefix,
                    log,
                    str(exc),
                )
        with contextlib.suppress(Exception):
            cleanup_terminal = get_status(run_id).state in TERMINAL_STATES
    finally:
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
    retain_only_revision: bool = False,
    advance_generation: bool = False,
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
        retain_only_revision=retain_only_revision,
        advance_generation=advance_generation,
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
    from flash.runner import _UNDEPLOYABLE_STATES, _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
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
            _save_status_unlocked(status)

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
    owner_deployment: dict | None = None,
    retain_only_revision: bool = False,
    advance_generation: bool = False,
) -> RunStatus:
    """Record a checkpoint deployment using the run's current lifecycle state.

    If training has finished by the time serving registration completes, the run behaves like any
    finished deployed run. Otherwise, keep the training state and only attach the deployment record.
    """
    from flash.runner import _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state == "dry_run":
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        ownership_token = deployment if owner_deployment is None else owner_deployment
        if not _deployment_attempt_is_owned(status, ownership_token):
            return status

        def _commit() -> None:
            if status.state in _FINAL_DEPLOYMENT_STATES:
                _promote_final_deployment(status, deployment)
            else:
                status.deployment = deployment
            status.updated_at = time.time()
            _save_status_unlocked(status)

        if not _commit_verified_deployment(
            run_id,
            deployment,
            verification_generation=verification_generation,
            commit=_commit,
            retain_only_revision=retain_only_revision,
            advance_generation=advance_generation,
        ):
            return get_status(run_id)
        return status


def mark_deployment_pending(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
    *,
    owner_deployment: dict | None = None,
) -> RunStatus:
    """Attach an in-progress deployment record without changing the run lifecycle state."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state == "dry_run":
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        ownership_token = deployment if owner_deployment is None else owner_deployment
        expected_generation = ownership_token.get("verification_generation")
        if expected_generation is not None:
            from flash.runner.verified_revisions import verified_adapter_revision_generation

            if verified_adapter_revision_generation(run_id) != expected_generation:
                return status
        current = status.deployment if isinstance(status.deployment, dict) else {}
        same_attempt = current.get("requested_at") == deployment.get("requested_at")
        if same_attempt and current.get("state") in {"undeployed", _REVOCATION_RETRY_STATE}:
            return status
        if owner_deployment is not None and not _deployment_attempt_is_owned(
            status, owner_deployment
        ):
            return status
        status.deployment = deployment
        status.updated_at = time.time()
        _save_status_unlocked(status)
        return status


def mark_deployment_failed(run_id: str, deployment: dict) -> RunStatus:
    """Record a failed deployment attempt while preserving the run lifecycle state."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        current = status.deployment or {}
        # don't clobber a newer deployment attempt, explicit undeploy, or pending revocation.
        if current.get("state") in {"undeployed", _REVOCATION_RETRY_STATE}:
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
                if failed.get("activation_outcome_unknown") and failed.get("state") == "reconciling"
                else "failed"
            )
            status.deployment = {**failed, "state": state}
        status.updated_at = time.time()
        _save_status_unlocked(status)
        return status


def mark_deployment_revocation_failed(run_id: str, error: str) -> RunStatus:
    """Revoke local serving authority while retaining retryable backend cleanup state."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status
    from flash.runner.verified_revisions import invalidate_verified_adapter_revisions

    with _status_guard(run_id):
        status = get_status(run_id)

        def _commit() -> None:
            deployment = status.deployment if isinstance(status.deployment, dict) else {}
            status.deployment = {
                **deployment,
                "state": "revocation_failed",
                "error": error,
                "retryable": True,
                "updated_at": time.time(),
            }
            status.updated_at = time.time()
            _save_status_unlocked(status)

        invalidate_verified_adapter_revisions(run_id, commit=_commit)
        return status


def mark_undeployed(run_id: str) -> RunStatus:
    """Record an explicit undeploy; live final-adapter deployments return to `done`."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status
    from flash.runner.verified_revisions import invalidate_verified_adapter_revisions

    with _status_guard(run_id):
        status = get_status(run_id)

        def _commit() -> None:
            if status.deployment:
                deployment = dict(status.deployment)
                for field in ("error", "retryable", "updated_at"):
                    deployment.pop(field, None)
                status.deployment = {**deployment, "state": "undeployed"}
            if status.state == "deployed":
                status.state = "done"
            status.updated_at = time.time()
            _save_status_unlocked(status)

        invalidate_verified_adapter_revisions(run_id, commit=_commit)
        return status


def mark_deployment_undeployed(run_id: str) -> RunStatus:
    """Flip ONLY the deployment field to ``undeployed``, leaving the run's state untouched.

    Used by cancel_run, unlike mark_undeployed, never asserts or changes the run state,
    so it works even after a racing mark_undeployed has already written terminal `done`.
    """
    from flash.runner import _save_status_unlocked, _status_guard, get_status
    from flash.runner.verified_revisions import invalidate_verified_adapter_revisions

    with _status_guard(run_id):
        status = get_status(run_id)

        def _commit() -> None:
            if status.deployment is not None:
                deployment = status.deployment if isinstance(status.deployment, dict) else {}
                deployment = dict(deployment)
                for field in ("error", "retryable", "updated_at"):
                    deployment.pop(field, None)
                status.deployment = {**deployment, "state": "undeployed"}
                status.updated_at = time.time()
                _save_status_unlocked(status)

        invalidate_verified_adapter_revisions(run_id, commit=_commit)
        return status
