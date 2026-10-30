"""deploy, cancel, and recover run state transitions."""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from flash.core.spec import JobSpec
from flash.runner.results.deployments import is_ready

if TYPE_CHECKING:
    from flash.runner.lifecycle.state import RunStatus

_UNDEPLOYING_STATE = "undeploying"
_INACTIVE_DEPLOYMENT_STATES = frozenset({"undeployed", "dry_run"})


class DeploymentCleanupError(RuntimeError):
    """Backend cleanup is unconfirmed after local serving authority was removed."""

    def __init__(self, run_id: str, error: str):
        super().__init__(
            f"deployment_cleanup_failed: serving for {run_id} is disabled locally, but "
            f"backend cleanup is unconfirmed: {error}; retry cancellation"
        )
        self.run_id = run_id
        self.retryable = True


_BackendOutcome = Literal["confirmed", "not_required", "not_attempted"]
_BACKEND_OUTCOMES = frozenset({"confirmed", "not_required", "not_attempted"})


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
                "backend cleanup was not required, but generation-only local persistence "
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


def _deployment_state_and_requires_cleanup(
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


class _CancellationFence:
    """Persist the ``cancelled`` state once, deferring any failure to the caller's error list.

    Idempotent by construction: the fence is attempted at most once per cancellation, so a later
    call after the deploy lock is held cannot re-run it. Failures are collected rather than raised
    because teardown must still be attempted.
    """

    def __init__(
        self,
        run_id: str,
        deferred_errors: list[Exception],
        *,
        entered_deployed: bool,
    ):
        self.run_id = run_id
        self.deferred_errors = deferred_errors
        self.entered_deployed = entered_deployed
        self.attempted = False

    def persist(self) -> None:
        from flash.runner.lifecycle.state import TERMINAL_STATES
        from flash.runner.lifecycle.status import _update, get_status

        if self.attempted:
            return
        self.attempted = True
        try:
            fence_applied = _update(
                self.run_id, "cancelled", allow_from_terminal=self.entered_deployed
            )
        except Exception as exc:
            self.deferred_errors.append(exc)
            return
        if fence_applied:
            return
        try:
            fenced_status = get_status(self.run_id)
        except Exception as exc:
            self.deferred_errors.append(exc)
            return
        if fenced_status.state not in TERMINAL_STATES:
            self.deferred_errors.append(
                RuntimeError(f"run {self.run_id} cancellation fence was not persisted")
            )


def _clear_remote_if_unchanged(run_id: str, expected_remote: dict) -> bool:
    """Drop the status's remote, but only while it is still the exact one that was torn down.

    Compare-and-clear under the status guard: a racing write that installed a DIFFERENT remote must
    keep it, or the run loses the only handle to a live billing resource.
    """
    from flash.runner.lifecycle.cleanup import _remote_resource_identity
    from flash.runner.lifecycle.reporting import _report_status
    from flash.runner.lifecycle.state import _save_status_unlocked, _status_guard
    from flash.runner.lifecycle.status import get_status

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


def _drain_confirmed_cleanup(run_id: str) -> set[tuple]:
    """Drain deferred cleanup targets and return the identities confirmed gone.

    A record still present after the drain was NOT confirmed, so it is left for a later attempt;
    only records that disappeared are reported, and each has its status entry cleared.
    """
    from flash.runner.lifecycle.cleanup import (
        _cleanup_remote_key,
        _drain_cleanup_remotes,
        _remote_resource_identity,
        _snapshot_cleanup_remotes,
    )

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
        _clear_remote_if_unchanged(run_id, record)
    return confirmed_identities


def _teardown_or_preserve_remote(run_id: str, remote: dict) -> bool:
    """cancellation clears a resource only after confirmed deletion."""
    from flash.runner.lifecycle.cleanup import reconcile_cleanup

    cleanup = reconcile_cleanup(run_id, remote)
    if not cleanup.resource_deleted and not cleanup.obligation_recorded:
        raise RuntimeError(f"run {run_id} unconfirmed cleanup target could not be preserved")
    return cleanup.resource_deleted


def _teardown_persisted_remotes(
    run_id: str,
    *,
    confirmed_cleanup_identities: set,
    clear_exact_remote: Callable[[dict], bool],
) -> None:
    """Tear down every remote the status names, re-reading after each so a racing write is seen.

    Stops on the first remote that could not be confirmed torn down, unless the status has since
    moved to a different one -- an already-confirmed identity is cleared without a second teardown.
    """
    from flash.runner.lifecycle.cleanup import _remote_resource_identity
    from flash.runner.lifecycle.status import get_status

    processed_remote_identities = set()
    while True:
        status = get_status(run_id)
        remote = status.remote or {}
        identity = _remote_resource_identity(remote)
        if not remote or identity in processed_remote_identities:
            return
        processed_remote_identities.add(identity)
        if identity in confirmed_cleanup_identities:
            clear_exact_remote(remote)
            continue
        if _teardown_or_preserve_remote(run_id, remote):
            clear_exact_remote(remote)
            continue
        latest_remote = get_status(run_id).remote or {}
        if _remote_resource_identity(latest_remote) != identity:
            continue
        return


def _fence_contended_deployment(run_id: str) -> None:
    """Fence every unfinished checkpoint before waiting for the deployment job."""
    from flash.runner.lifecycle.state import _save_status_unlocked, _status_guard
    from flash.runner.lifecycle.status import get_status
    from flash.runner.results.deployments import generation

    with _status_guard(run_id):
        status = get_status(run_id)
        changed = False
        for checkpoint_id, record in status.deployments.items():
            _, active = _deployment_state_and_requires_cleanup(record)
            if not active or is_ready(run_id, record):
                continue
            status.deployments[checkpoint_id] = {
                **record,
                "state": "undeploying",
                "retryable": True,
                "generation": generation(record) + 1,
                "error": "backend cleanup pending: cancellation fenced an in-progress deployment",
                "updated_at": time.time(),
            }
            changed = True
        if changed:
            status.updated_at = time.time()
            try:
                _save_status_unlocked(status)
            except Exception as exc:
                raise DeploymentStatePersistenceError(
                    run_id, str(exc), backend_outcome="not_attempted"
                ) from exc


def _cleanup_serving(
    run_id: str,
    status,
    *,
    deployment: dict,
) -> tuple[Exception | None, tuple[Exception, _BackendOutcome] | None]:
    """Clean up one unfinished checkpoint while preserving ready siblings.

    Returns ``(backend_error, persistence_error)`` rather than raising, so the caller can still bill
    and persist the cancellation before surfacing either. Local authority is fenced BEFORE the
    backend call, so a crash between the two leaves the run visibly un-serveable, not silently live.
    """
    from flash.runner.supervise.transitions import (
        mark_deployment_undeploying,
        mark_undeployed,
    )

    checkpoint_id = deployment["checkpoint_id"]
    already_fenced = deployment.get("state") == _UNDEPLOYING_STATE
    if not already_fenced:
        try:
            mark_deployment_undeploying(
                run_id,
                "backend cleanup pending: checkpoint serving was disabled locally before "
                "backend cleanup",
                checkpoint_id=checkpoint_id,
            )
        except Exception as exc:
            raise DeploymentStatePersistenceError(
                run_id, str(exc), backend_outcome="not_attempted"
            ) from exc
    try:
        from flash.serve.deployment.deploy import undeploy_adapter
        from flash.server.platform.internal_client import run_serving_org_id

        org_id = run_serving_org_id(status)
        if not org_id:
            raise ValueError(f"run {run_id} has no organization scope")
        undeploy_adapter(checkpoint_id, org_id=org_id)
    except Exception as exc:
        with contextlib.suppress(Exception):
            mark_deployment_undeploying(run_id, str(exc), checkpoint_id)
        return exc, None
    try:
        mark_undeployed(run_id, checkpoint_id)
    except Exception as exc:
        return None, (exc, "confirmed")
    return None, None


def _cancellation_billing(
    run_id: str,
    effective_spec,
    *,
    bill_cancel: bool,
    rented_remote: dict | None = None,
) -> tuple[float | None, dict]:
    """Price a cancellation. Returns ``(charge_usd, billing_diagnostic)``.

    ``None`` means this cancellation is not billable at all and must not write ``cost_usd``; a
    pricing failure bills 0.0 and reports why, because teardown is attempted either way.
    ``rented_remote`` is the provider handle snapshotted before teardown: the status reloaded here
    no longer carries it after a confirmed teardown, and it is the only durable record of the
    provider and card shape the run rented, which the cancel price must be computed on.
    """
    from flash.runner.accounting.costs import actual_steps_run, cancelled_charge_usd
    from flash.runner.lifecycle.status import get_status

    if not bill_cancel:
        return None, {}
    if effective_spec is None:
        return 0.0, {
            "billing_state": "failed",
            "billing_error": (
                "cancellation charge was not computed because the private preparation "
                "snapshot was unavailable or invalid; teardown was still attempted"
            ),
        }
    cancel_status = get_status(run_id)
    steps_billed = actual_steps_run(cancel_status)
    # a fresh spec estimate uses offline static rates, which on live-market providers can
    # exceed the accepted quote's rate, so a mid-training cancel is priced from the persisted
    # quote (scaled by the completed share of the estimated work) to keep a near-complete
    # cancel at or under what the run would have cost on success.
    estimated_charge = cancelled_charge_usd(
        cancel_status,
        effective_spec,
        steps=steps_billed,
        fallback=float("nan"),
        rented_remote=rented_remote,
    )
    if math.isfinite(estimated_charge):
        return estimated_charge, {}
    return 0.0, {
        "billing_state": "failed",
        "billing_error": (
            "cancellation charge was not computed because pricing failed; "
            "teardown was still attempted"
        ),
    }


def cancel_run(run_id: str) -> RunStatus:
    """Cancel training while preserving verified serving and durable cleanup targets."""
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import _update, effective_spec_from_status, get_status
    from flash.runner.supervise.recovery import _gc_run_sandboxes
    from flash.server.platform.locks import _deploy_lock

    deferred_fencing_errors: list[Exception] = []

    def _clear_exact_remote(expected_remote: dict) -> bool:
        return _clear_remote_if_unchanged(run_id, expected_remote)

    initial_status = get_status(run_id)
    confirmed_cleanup_identities = set()
    if initial_status.state == "cancelled":
        confirmed_cleanup_identities = _drain_confirmed_cleanup(run_id)
        initial_status = get_status(run_id)
    initial_active = any(
        _deployment_state_and_requires_cleanup(record)[1]
        for record in initial_status.deployments.values()
    )
    if initial_status.state in TERMINAL_STATES and not initial_active:
        if deferred_fencing_errors:
            raise deferred_fencing_errors[0]
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

    fence = _CancellationFence(run_id, deferred_fencing_errors, entered_deployed=entered_deployed)
    _persist_cancellation_fence = fence.persist

    deploy_lock = _deploy_lock(run_id)
    lock_acquired = deploy_lock.acquire(blocking=False)
    requires_early_cancellation_fence = bool(initial_status.remote) or not lock_acquired
    try:
        if not lock_acquired:
            _fence_contended_deployment(run_id)
            deploy_lock.acquire()
            lock_acquired = True

        if requires_early_cancellation_fence:
            _persist_cancellation_fence()
        status = get_status(run_id)
        entered_deployed = entered_deployed or status.state == "deployed"

        # teardown clears the active handle before cancellation pricing. successful work may already
        # have moved that exact rented basis into the retained private accounting identity.
        rented_basis = (
            status.remote
            or getattr(status, "billing_remote", None)
            or getattr(status, "cleanup_confirmed_remote", None)
        )
        rented_remote = dict(rented_basis) if isinstance(rented_basis, dict) else None

        _teardown_persisted_remotes(
            run_id,
            confirmed_cleanup_identities=confirmed_cleanup_identities,
            clear_exact_remote=_clear_exact_remote,
        )
        if cleanup_spec is not None:
            with contextlib.suppress(Exception):
                _gc_run_sandboxes(cleanup_spec)

        status = get_status(run_id)
        entered_deployed = entered_deployed or status.state == "deployed"
        backend_error: Exception | None = None
        persistence_error: tuple[Exception, _BackendOutcome] | None = None
        for record in status.deployments.values():
            _, active = _deployment_state_and_requires_cleanup(record)
            if not active or is_ready(run_id, record):
                continue
            try:
                cleanup_error, persist_error = _cleanup_serving(run_id, status, deployment=record)
                backend_error = backend_error or cleanup_error
                persistence_error = persistence_error or persist_error
            except DeploymentStatePersistenceError as exc:
                persistence_error = persistence_error or (exc, exc.backend_outcome)

        cancel_charge_usd, billing_diagnostic = _cancellation_billing(
            run_id, effective_spec, bill_cancel=bill_cancel, rented_remote=rented_remote
        )
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
            from flash.server.domain.registry.checkpoints import register_checkpoints_best_effort

            register_checkpoints_best_effort(get_status(run_id))

        if backend_error is not None:
            raise DeploymentCleanupError(run_id, str(backend_error)) from backend_error
        if persistence_error is not None:
            error, backend_outcome = persistence_error
            raise DeploymentStatePersistenceError(
                run_id, str(error), backend_outcome=backend_outcome
            ) from error
        if deferred_fencing_errors:
            raise deferred_fencing_errors[0]
        return get_status(run_id)
    finally:
        if lock_acquired:
            deploy_lock.release()
