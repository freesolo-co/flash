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
        _preserve_cleanup_remote,
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
    from flash.server._locks import _deploy_lock

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

        status = get_status(run_id)
        entered_deployed = entered_deployed or status.state == "deployed"

        from flash.providers.base import JobHandle
        from flash.runner.lifecycle import _strict_teardown_handle

        def _teardown_remote(remote: dict) -> bool:
            try:
                _strict_teardown_handle(JobHandle.from_dict(remote))
                return True
            except Exception:
                if not _preserve_cleanup_remote(run_id, remote):
                    raise RuntimeError(
                        f"run {run_id} teardown was unconfirmed and its exact cleanup target "
                        "could not be preserved"
                    ) from None
                return False

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
            teardown_confirmed = _teardown_remote(remote)
            if teardown_confirmed:
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


def attach_run(run_id: str, log_stream=None) -> RunStatus:
    """Re-attach to a run's remote job from any process (after a client crash/restart)."""
    import sys

    from flash.runner import (
        TERMINAL_STATES,
        _compare_and_clear_remote,
        _gc_run_endpoints,
        _load_run_deadline_at,
        _persist_metrics,
        _run_training,
        _RunCancelled,
        _spec_with_remaining_wall,
        _status_estimated_charge,
        _update,
        _verified_opd_next_attempt,
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
    from flash.runner.lifecycle import _strict_teardown_handle

    try:
        worker_spec = effective_spec_from_status(status)
        persisted_remote = dict(status.remote)
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
        # The class the run actually provisioned (a policy retry may have walked past the
        # provisional spec.gpu.type). The in-process success path stamps this into metrics;
        # on recovery the worker output carries no such field, so recover it from the handle
        # to cost the right card.
        allocated_gpu = remote.pop("allocated_gpu", None)
        handle = JobHandle.from_dict(remote)
        try:
            poll_spec = _spec_with_remaining_wall(worker_spec, require_provider_minimum=False)
        except RuntimeError as exc:
            # expiry is terminal: stop the persisted worker before clearing its handle, and retain the
            # handle when teardown is unconfirmed so later cleanup can still target the paid resource.
            teardown_confirmed = True
            try:
                _strict_teardown_handle(handle)
            except Exception:
                teardown_confirmed = False
            if teardown_confirmed:
                _compare_and_clear_remote(run_id, persisted_remote)
            _update(run_id, "failed", error=str(exc))
            print(f"attach: {run_id} {exc}", file=log)
            return get_status(run_id)
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
            print(f"attach: {run_id} ended ({res.failure}); evaluating recovery", file=log)
            try:
                _strict_teardown_handle(handle)
            except Exception:
                with contextlib.suppress(Exception):
                    _gc_run_endpoints(worker_spec)
                print(
                    f"attach: {run_id} {handle.provider} teardown unconfirmed; "
                    "not resuming over a possibly-live resource",
                    file=log,
                )
                return status_for_return()
            with contextlib.suppress(Exception):
                _gc_run_endpoints(worker_spec)
            if not _compare_and_clear_remote(run_id, persisted_remote):
                print(
                    f"attach: {run_id} persisted remote changed before clear; not resuming",
                    file=log,
                )
                return status_for_return()
            if int(worker_spec.gpu.max_retries) == 0:
                detail = f"{res.failure or 'job_failed'}: {res.detail or 'provider attempt failed'}"
                _update(run_id, "failed", error=detail)
                print(
                    f"attach: {run_id} exhausted its one-shot retry budget; not resubmitting",
                    file=log,
                )
                return status_for_return()
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
                _update(run_id, "failed", error=str(exc))
                print(f"attach: {run_id} {exc}", file=log)
                return status_for_return()
            worker_spec = effective_spec_from_status(get_status(run_id), verify_source=True)
            if not _update(run_id, "running"):
                print(f"attach: {run_id} went terminal during recovery; not resuming", file=log)
                return status_for_return()
            print(
                f"attach: {run_id} resubmitting from the latest checkpoint before the "
                "run-global wall deadline",
                file=log,
            )
            if code_prefix is None:
                from flash.providers._worker import upload_code
                from flash.runner import flash_code_prefix

                code_prefix = flash_code_prefix()
                upload_code(
                    worker_spec.train.hf_repo,
                    code_prefix=code_prefix,
                    **deadline_kwargs(upload_code, _load_run_deadline_at(run_id)),
                )
            _run_training(
                worker_spec,
                log,
                prior_cost=float(status.cost_usd or 0.0),
                code_prefix=code_prefix,
                attempt_start=next_attempt,
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
                status.deployment = {**status.deployment, "state": "undeployed"}
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
