"""Deploy, cancel, and recover run state transitions.

Keep ``flash.runner`` imports function-local to avoid its import cycle and preserve package-level
monkeypatch seams such as ``flash.runner._gc_run_endpoints``.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from flash.core.spec import JobSpec
from flash.schema import format_adapter_revision, parse_adapter_revision

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


def _carry_allocation_stamp(metrics: dict, remote: dict | None) -> None:
    """Carry the persisted allocation stamp onto adopted metrics.

    Workers do not know the allocator's card, count, or provider. Recovery must restore all three or
    multi-card Vast/Lambda runs are priced as one RunPod card. ``setdefault`` preserves worker data.
    """
    if not isinstance(metrics, dict) or not isinstance(remote, dict):
        return
    allocated_gpu = remote.get("allocated_gpu")
    if allocated_gpu:
        metrics.setdefault("allocated_gpu", allocated_gpu)
    allocated_count = remote.get("allocated_gpu_count")
    if allocated_count:
        metrics.setdefault("allocated_gpu_count", int(allocated_count))
    # the substrate that billed the run. `_gpu_rate` falls back to whichever configured provider
    # offers the class, which on a multi-provider plane is normally RunPod -- so an adopted lambda
    # or vast run is otherwise priced at RunPod's rate and its notes name the wrong provider.
    # `provider` is required on a persisted JobHandle, so it is always present here.
    provider = remote.get("provider")
    if provider:
        metrics.setdefault("allocated_provider", provider)


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
    from flash.runner.results.verified_revisions import read_verified_adapter_revisions

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
        from flash.runner import TERMINAL_STATES, _update, get_status

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
    from flash.runner import (
        _remote_resource_identity,
        _report_status,
        _save_status_unlocked,
        _status_guard,
        get_status,
    )

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
    from flash.runner import (
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
    """Tear one remote down. Returns whether it is safe to clear from the status.

    An unconfirmed teardown is NOT safe to clear: the target is recorded for the cleanup drain
    instead, so a leaked box stays addressable. Failing to record either is a hard error, since
    that would lose the only handle to a billing resource.
    """
    from flash.providers.base import JobHandle
    from flash.runner import _record_cleanup_remote
    from flash.runner.supervise.lifecycle import _strict_teardown_handle

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
        raise RuntimeError(f"run {run_id} leaked endpoint cleanup target could not be preserved")
    return True


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
    from flash.runner import _remote_resource_identity, get_status

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


@dataclass(frozen=True)
class _ContendedFence:
    """What fencing an in-progress deployment observed, for the preservation decision later."""

    active_attempt: bool = False
    captured_attempt: dict | None = None
    predecessor: dict | None = None
    predecessor_recommitted: bool = False


def _restore_contended_predecessor(
    run_id: str,
    predecessor: dict,
    fenced_deployment: dict | None,
) -> bool:
    """Try to recommit the verified predecessor over the fenced attempt.

    Returns whether the predecessor is now the authoritative deployment AND the only verified
    revision. On failure, re-fence so no unverified attempt is left looking serveable.
    """
    from flash.runner import (
        get_status,
        mark_checkpoint_deployed,
        mark_deployment_revocation_failed,
        read_verified_adapter_revisions,
        verified_adapter_revision_generation,
    )

    recommitted = False
    if fenced_deployment is not None:
        try:
            restored_status = mark_checkpoint_deployed(
                run_id,
                predecessor,
                verification_generation=verified_adapter_revision_generation(run_id),
                owner_deployment=fenced_deployment,
                retain_only_revision=True,
            )
            predecessor_revision = predecessor.get("adapter_revision")
            recommitted = (
                restored_status.deployment == predecessor
                and isinstance(predecessor_revision, str)
                and _is_preservable_checkpoint_deployment(run_id, restored_status.deployment)
                and read_verified_adapter_revisions(run_id) == frozenset({predecessor_revision})
            )
        except Exception:
            recommitted = False
    if recommitted:
        return True
    current = get_status(run_id)
    current_deployment = current.deployment if isinstance(current.deployment, dict) else {}
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
    return False


def _fence_contended_deployment(run_id: str) -> _ContendedFence:
    """Fence a deployment that holds the deploy lock, before blocking on it.

    Runs BEFORE the blocking acquire so an in-progress deployment cannot finish and present itself
    as serveable after the cancel decided to tear it down. A preservable checkpoint is left alone.
    """
    from flash.runner import get_status, mark_deployment_revocation_failed

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
    if not must_fence:
        return _ContendedFence()

    predecessor: dict | None = None
    if prelock_deployment is not None:
        candidate = prelock_deployment.get("previous_deployment")
        if _is_preservable_checkpoint_deployment(run_id, candidate):
            predecessor = dict(candidate)
    try:
        fenced_status = mark_deployment_revocation_failed(
            run_id,
            "backend revocation pending: cancellation fenced an in-progress deployment",
        )
    except Exception as exc:
        raise DeploymentStatePersistenceError(
            run_id, str(exc), backend_outcome="not_attempted"
        ) from exc
    if predecessor is None:
        return _ContendedFence(active_attempt=True, captured_attempt=prelock_deployment)
    fenced_deployment = (
        dict(fenced_status.deployment) if isinstance(fenced_status.deployment, dict) else None
    )
    return _ContendedFence(
        active_attempt=True,
        captured_attempt=prelock_deployment,
        predecessor=predecessor,
        predecessor_recommitted=_restore_contended_predecessor(
            run_id, predecessor, fenced_deployment
        ),
    )


def _checkpoint_to_preserve(
    run_id: str,
    status,
    *,
    contended: _ContendedFence,
) -> dict | None:
    """Decide which deployment, if any, survives this cancellation.

    A dry run preserves nothing. A contended attempt preserves ONLY its predecessor, and only when
    every check agrees the predecessor is what is actually live: recommitted, still the stored
    deployment, still verified, and the alias really points at it.
    """
    live_alias_target: object = _UNKNOWN_ALIAS_UNCHECKED
    unknown_activation = isinstance(status.deployment, dict) and status.deployment.get(
        "activation_outcome_unknown"
    )
    if status.state != "dry_run" and (contended.active_attempt or unknown_activation):
        try:
            from flash.serve.deploy import adapter_alias_target

            live_alias_target = adapter_alias_target(run_id)
        except Exception:
            live_alias_target = None
    if status.state == "dry_run":
        return None
    if not contended.active_attempt:
        return _preservable_checkpoint_deployment(
            run_id,
            status.deployment,
            live_alias_target=live_alias_target,
        )
    predecessor = contended.predecessor
    predecessor_revision = predecessor.get("adapter_revision") if predecessor is not None else None
    if (
        contended.predecessor_recommitted
        and isinstance(contended.captured_attempt, dict)
        and isinstance(predecessor_revision, str)
        and live_alias_target == predecessor_revision
        and status.deployment == predecessor
        and _is_preservable_checkpoint_deployment(run_id, status.deployment)
    ):
        return dict(predecessor)
    return None


def _commit_preserved_checkpoint(run_id: str, status, preserved_checkpoint: dict | None):
    """Make the preserved checkpoint authoritative and the only verified revision.

    Returns ``(status, preserved_checkpoint)``; the checkpoint comes back ``None`` when the first
    commit could not be confirmed, which drops the run through to normal revocation. The second
    commit is the authoritative one, so its failure is fatal rather than a downgrade.
    """
    from flash.runner import (
        get_status,
        mark_checkpoint_deployed,
        read_verified_adapter_revisions,
        verified_adapter_revision_generation,
    )

    if preserved_checkpoint is None:
        return status, None
    if status.deployment != preserved_checkpoint:
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
            # the write may still have landed, so re-read rather than assume it did not.
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
            if status.deployment != preserved_checkpoint or not (
                _is_preservable_checkpoint_deployment(run_id, status.deployment)
            ):
                preserved_checkpoint = None
        if preserved_checkpoint is None:
            return status, None

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
    return status, preserved_checkpoint


def _revoke_serving(
    run_id: str,
    status,
    *,
    active_deployment: bool,
) -> tuple[Exception | None, tuple[Exception, _BackendOutcome] | None]:
    """Revoke serving for a run that preserves no checkpoint.

    Returns ``(backend_error, persistence_error)`` rather than raising, so the caller can still bill
    and persist the cancellation before surfacing either. Local authority is fenced BEFORE the
    backend call, so a crash between the two leaves the run visibly un-serveable, not silently live.
    """
    from flash.runner import mark_deployment_revocation_failed, mark_deployment_undeployed

    if not active_deployment:
        try:
            mark_deployment_undeployed(run_id)
        except Exception as exc:
            return None, (exc, "not_required")
        return None, None

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
        with contextlib.suppress(Exception):
            mark_deployment_revocation_failed(run_id, str(exc))
        return exc, None
    try:
        mark_deployment_undeployed(run_id)
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
    from flash.runner import (
        _status_estimated_charge,
        actual_steps_run,
        cancelled_charge_usd,
        get_status,
        profile_steps_run,
    )

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
    # a profile has no optimizer steps, so actual_steps_run reads 0 for every one and
    # would price a profile that ran to completion at $0. profile_steps_run answers the
    # question that actually applies to one: did it start at all.
    #
    # the profile marker is read from the STATUS, not from effective_spec: to_dict()
    # strips workload_profile_kind as a platform-managed field, so a spec rebuilt from
    # persisted public status always reports "" and the branch below would never be
    # taken. the status carries the kind explicitly for exactly this reason.
    steps_billed = (
        profile_steps_run(cancel_status)
        if cancel_status.workload_profile_kind
        else actual_steps_run(cancel_status)
    )
    if cancel_status.workload_profile_kind and steps_billed > 0:
        # a STARTED profile owes exactly the quote it was submitted under. re-deriving
        # it here would re-price against today's offline rate table, so a cancel could
        # bill a different number than the one the user was shown and than the same
        # profile would have billed on success. it never started -> steps_billed is 0
        # and the branch below correctly charges nothing.
        estimated_charge = _status_estimated_charge(
            cancel_status,
            effective_spec,
            fallback=float("nan"),
        )
    else:
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
    from flash.runner import (
        TERMINAL_STATES,
        _gc_run_endpoints,
        _update,
        effective_spec_from_status,
        get_status,
    )
    from flash.server.platform import db as server_db
    from flash.server.platform.locks import _deploy_lock

    # fence teacher authority before waiting on deployment locks or slow provider teardown.
    deferred_fencing_errors: list[Exception] = []
    try:
        server_db.revoke_teacher_capabilities_for_run(run_id)
    except Exception as exc:
        deferred_fencing_errors.append(exc)

    def _clear_exact_remote(expected_remote: dict) -> bool:
        return _clear_remote_if_unchanged(run_id, expected_remote)

    initial_status = get_status(run_id)
    confirmed_cleanup_identities = set()
    if initial_status.state == "cancelled":
        confirmed_cleanup_identities = _drain_confirmed_cleanup(run_id)
        initial_status = get_status(run_id)
    _, initial_active = _deployment_state_and_requires_revocation(initial_status.deployment)
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
    captured_contended_attempt: dict | None = None
    contended_active_attempt = False
    contended_predecessor: dict | None = None
    contended_predecessor_recommitted = False
    lock_acquired = deploy_lock.acquire(blocking=False)
    requires_early_cancellation_fence = bool(initial_status.remote) or not lock_acquired
    try:
        if not lock_acquired:
            contended = _fence_contended_deployment(run_id)
            contended_active_attempt = contended.active_attempt
            captured_contended_attempt = contended.captured_attempt
            contended_predecessor = contended.predecessor
            contended_predecessor_recommitted = contended.predecessor_recommitted
            deploy_lock.acquire()
            lock_acquired = True

        # close the race where submission held the deploy lock, minted a capability after the
        # pre-lock fence, and released the lock only after persisting its provider handle.
        try:
            server_db.revoke_teacher_capabilities_for_run(run_id)
        except Exception as exc:
            deferred_fencing_errors.append(exc)
        if requires_early_cancellation_fence:
            _persist_cancellation_fence()
        status = get_status(run_id)
        entered_deployed = entered_deployed or status.state == "deployed"

        # teardown clears the durable handle on success and it is the only record of the rented
        # basis (provider, card, count), so capture it now for billing (see _cancellation_billing).
        rented_remote = dict(status.remote) if isinstance(status.remote, dict) else None

        _teardown_persisted_remotes(
            run_id,
            confirmed_cleanup_identities=confirmed_cleanup_identities,
            clear_exact_remote=_clear_exact_remote,
        )
        if cleanup_spec is not None:
            with contextlib.suppress(Exception):
                _gc_run_endpoints(cleanup_spec)

        status = get_status(run_id)
        entered_deployed = entered_deployed or status.state == "deployed"
        _, active_deployment = _deployment_state_and_requires_revocation(status.deployment)
        preserved_checkpoint = _checkpoint_to_preserve(
            run_id,
            status,
            contended=_ContendedFence(
                active_attempt=contended_active_attempt,
                captured_attempt=captured_contended_attempt,
                predecessor=contended_predecessor,
                predecessor_recommitted=contended_predecessor_recommitted,
            ),
        )
        status, preserved_checkpoint = _commit_preserved_checkpoint(
            run_id, status, preserved_checkpoint
        )
        entered_deployed = entered_deployed or status.state == "deployed"
        backend_error: Exception | None = None
        persistence_error: tuple[Exception, _BackendOutcome] | None = None
        if preserved_checkpoint is None:
            backend_error, persistence_error = _revoke_serving(
                run_id, status, active_deployment=active_deployment
            )

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
            from flash.server.domain.checkpoints import register_checkpoints_best_effort

            register_checkpoints_best_effort(get_status(run_id))

        if backend_error is not None:
            raise DeploymentRevocationError(run_id, str(backend_error)) from backend_error
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


# re-exported at the bottom rather than imported at the top: `attach` resolves the patched
# reconciliation names back through this module, so a top import would be circular. `flash.runner`
# and the attach tests both address these as attributes of THIS module.
from flash.runner.supervise.attach import (  # noqa: E402,F401
    _reconcile_attached_remote,
    _resume_after_confirmed_teardown,
    _schedule_attach_reconciliation,
    attach_run,
)
