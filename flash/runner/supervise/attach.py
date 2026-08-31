"""Reattaching to a run the control plane restarted away from, and reconciling what it finds.

`attach_run` is called for every non-terminal run at startup: the supervisor thread that owned the
run is gone, but the remote worker may still be alive, already finished, or already torn down.
Deciding which -- without double-billing, double-tearing-down, or resuming a run whose teardown was
never confirmed -- is what the reconciliation loop here does, on its own background thread.

Split out of `flash.runner.supervise.deploy` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from flash.core.spec import JobSpec
from flash.envs.loading.staged import StagedEnvironmentTransientError
from flash.providers._lifecycle.instances.poll import _attempt_int
from flash.runner.supervise.attach_reconcile import (
    carry_allocation_stamp as _carry_allocation_stamp,
)
from flash.runner.supervise.attach_reconcile import (
    completed_metrics_for_remote as _completed_metrics_for_remote,
)
from flash.runner.supervise.attach_reconcile import (
    teardown_reconciled_remote as _teardown_reconciled_remote,
)

_ATTACH_RECONCILE_INTERVAL_S = 120.0
_ATTACH_RECONCILING: set[str] = set()
_ATTACH_RECONCILING_LOCK = threading.Lock()


if TYPE_CHECKING:
    from flash.providers.core.base import JobHandle, PollResult
    from flash.runner.lifecycle.state import RunStatus


def _resume_after_confirmed_teardown(
    run_id: str,
    worker_spec: JobSpec,
    persisted_remote: dict,
    next_attempt: int,
    source_snapshot: dict | None,
    log,
    *,
    failure: str,
) -> RunStatus:
    """CAS-clear one captured remote, then resume its next attempt exactly once."""
    from flash.runner.accounting.artifacts import stage_environment_package
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle import state as lifecycle_state
    from flash.runner.lifecycle.attempts import reserve_verified_attempt_launch
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at, _spec_with_remaining_wall
    from flash.runner.lifecycle.status import (
        _load_status_json,
        get_status,
        reallocation_spec_from_status,
        source_snapshot_from_status,
    )
    from flash.runner.lifecycle.submit import _persist_effective_worker_spec
    from flash.runner.supervise.errors import _RunCancelled
    from flash.runner.supervise.lifecycle import _run_training
    from flash.runner.supervise.retry_decision import RetryState, _drop_weight_cache

    raw = _load_status_json(run_id)
    retry_snapshot = raw[lifecycle_state._RETRY_STATE_KEY]
    retry_state = RetryState.from_snapshot(worker_spec, retry_snapshot)
    try:
        from flash.snapshot.archive import parse_descriptor

        source_snapshot = parse_descriptor(
            source_snapshot or source_snapshot_from_status(get_status(run_id), required=True)
        ).to_dict()
    except Exception as exc:
        _compare_and_fail_remote(run_id, persisted_remote, str(exc))
        return get_status(run_id)
    try:
        _spec_with_remaining_wall(worker_spec, require_provider_minimum=True)
    except RuntimeError as exc:
        _compare_and_fail_remote(run_id, persisted_remote, str(exc))
        print(f"attach: {run_id} {exc}", file=log)
        return get_status(run_id)
    worker_spec = reallocation_spec_from_status(get_status(run_id), verify_source=True)
    if retry_state.drop_weight_cache:
        worker_spec = _drop_weight_cache(worker_spec)
    if worker_spec.run_id != run_id:
        worker_spec = replace(worker_spec, run_id=run_id)
    deadline_at = _load_run_deadline_at(run_id)
    worker_spec = stage_environment_package(worker_spec, deadline_at=deadline_at)
    if not _persist_effective_worker_spec(worker_spec):
        raise _RunCancelled(f"run {run_id} went terminal before environment staging")
    claim = reserve_verified_attempt_launch(
        run_id,
        expected_remote=persisted_remote,
        expected_next_attempt=next_attempt,
        transition_state="provisioning",
    )
    if claim is None:
        print(
            f"attach: {run_id} persisted ownership changed before replacement reservation",
            file=log,
        )
        return get_status(run_id)
    print(
        f"attach: {run_id} resubmitting from the latest checkpoint before the "
        "run-global wall deadline",
        file=log,
    )
    try:
        _run_training(
            worker_spec,
            log,
            prior_cost=float(get_status(run_id).cost_usd or 0.0),
            source_snapshot=source_snapshot,
            reserved_claim=claim,
        )
    except _RunCancelled:
        raise
    except Exception as exc:
        from flash.runner.lifecycle.attempts import attempt_is_this_callers_to_fail

        if attempt_is_this_callers_to_fail(run_id, claim):
            current_remote = get_status(run_id).remote
            if current_remote is not None:
                with contextlib.suppress(Exception):
                    _record_cleanup_remote(run_id, current_remote)
            _compare_and_fail_remote(run_id, current_remote, str(exc))
        raise
    return get_status(run_id)


def _reconcile_completed_remote(
    run_id: str,
    worker_spec: JobSpec,
    expected_remote: dict,
    completed_metrics: dict,
    deadline_at: float,
    log,
) -> bool:
    """Retry completed-attempt adoption and decide whether reconciliation is finished."""
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.supervise.lifecycle import (
        _RECOVERY_MARKER_GRACE_S,
        _adopt_completed_attempt,
    )

    try:
        if _adopt_completed_attempt(
            run_id,
            worker_spec,
            expected_remote,
            completed_metrics,
            log=log,
        ):
            return True
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
                return True
        except Exception:
            pass
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    remaining_grace = deadline_at + _RECOVERY_MARKER_GRACE_S - time.time()
    time.sleep(min(_ATTACH_RECONCILE_INTERVAL_S, max(0.0, remaining_grace)))
    return False


def _reconcile_expired_remote(
    run_id: str,
    worker_spec: JobSpec,
    expected_remote: dict,
    handle: JobHandle,
    next_attempt: int,
    deadline_at: float,
    log,
    failure: str,
) -> bool:
    """Adopt late metrics or fail an attempt whose wall deadline has elapsed."""
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.supervise.lifecycle import (
        _adopt_completed_attempt,
        _completed_attempt_metrics,
    )

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
        return False
    if metrics is not None:
        _carry_allocation_stamp(metrics, expected_remote)
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
            return False
        if adopted:
            return True
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    try:
        cleanup_preserved = _record_cleanup_remote(run_id, expected_remote)
    except Exception:
        cleanup_preserved = False
    if not cleanup_preserved:
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    try:
        if _compare_and_fail_remote(run_id, expected_remote, failure):
            return True
    except Exception:
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    return True


def _wait_for_replacement_window(worker_spec: JobSpec, deadline_at: float) -> bool:
    """Wait within the wall deadline and decide whether teardown should be deferred."""
    from flash.runner.lifecycle.deadlines import _spec_with_remaining_wall

    delay = min(_ATTACH_RECONCILE_INTERVAL_S, max(0.0, deadline_at - time.time()))
    if delay > 0:
        time.sleep(delay)
        if time.time() >= deadline_at:
            return True
    if time.time() < deadline_at:
        # if a replacement cannot meet the 60-second provider minimum yet the run
        # wall deadline is still open, keep reconciling (probe for completion) rather
        # than tearing down and failing early - mirror handle-less recovery, which
        # waits until the wall deadline.
        try:
            _spec_with_remaining_wall(worker_spec, require_provider_minimum=True)
        except RuntimeError:
            # cap the reconcile wait at the wall deadline so a near-deadline wake does
            # not overshoot the run's wall deadline by a full interval.
            time.sleep(min(_ATTACH_RECONCILE_INTERVAL_S, max(0.0, deadline_at - time.time())))
            return True
    return False


def _reconcile_attached_remote(
    run_id: str,
    expected_remote: dict,
    worker_spec: JobSpec,
    next_attempt: int,
    source_snapshot: dict | None,
    log,
    failure: str,
) -> None:
    from flash.providers.core.base import JobHandle
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _remote_resource_identity,
    )
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import (
        _RECOVERY_MARKER_GRACE_S,
        _CompletedAttemptPending,
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
        active_remote_matches = _remote_resource_identity(status.remote) == expected_identity
        confirmed_teardown = (
            status.remote is None
            and _remote_resource_identity(status.cleanup_confirmed_remote) == expected_identity
        )
        if not active_remote_matches and not confirmed_teardown:
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
            completed_metrics = _completed_metrics_for_remote(
                worker_spec,
                expected_remote,
                provider=handle.provider,
                attempt=next_attempt - 1,
                deadline_at=deadline_at,
                log=log,
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
            if _reconcile_completed_remote(
                run_id,
                worker_spec,
                expected_remote,
                completed_metrics,
                deadline_at,
                log,
            ):
                return
            continue
        if time.time() >= deadline_at:
            if _reconcile_expired_remote(
                run_id,
                worker_spec,
                expected_remote,
                handle,
                next_attempt,
                deadline_at,
                log,
                failure,
            ):
                return
            continue
        if _wait_for_replacement_window(worker_spec, deadline_at):
            continue
        may_continue, resource_deleted = _teardown_reconciled_remote(
            run_id,
            expected_remote,
            handle,
            confirmed_teardown=confirmed_teardown,
        )
        if not may_continue:
            continue
        if resource_deleted and not confirmed_teardown:
            from flash.runner.accounting.reconciliation import (
                _compare_and_confirm_remote_teardown,
            )

            try:
                confirmed_teardown = _compare_and_confirm_remote_teardown(run_id, expected_remote)
            except Exception:
                confirmed_teardown = False
            if not confirmed_teardown:
                continue
        try:
            _resume_after_confirmed_teardown(
                run_id,
                worker_spec,
                expected_remote,
                next_attempt,
                source_snapshot,
                log,
                failure=failure,
            )
        except StagedEnvironmentTransientError:
            time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
            continue
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
    source_snapshot: dict | None,
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
                source_snapshot,
                log,
                failure,
            )
        finally:
            try:
                from flash.runner.lifecycle.state import TERMINAL_STATES
                from flash.runner.lifecycle.status import get_status
                from flash.runner.supervise.recovery import _gc_run_endpoints

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


@dataclass(frozen=True)
class _AttachContext:
    worker_spec: JobSpec
    persisted_remote: dict
    handle: JobHandle
    recovered_attempt: int
    next_attempt: int
    source_snapshot: dict | None
    launch_claim_token: str


def _build_attach_context(
    worker_spec: JobSpec,
    persisted_remote: dict,
) -> _AttachContext:
    """Validate the persisted handle and collect the inputs needed to poll it."""
    from flash.providers.core.base import JobHandle
    from flash.runner.lifecycle.status import get_status, source_snapshot_from_status

    remote = dict(persisted_remote)
    remote.pop("code_prefix", None)
    source_snapshot = source_snapshot_from_status(get_status(worker_spec.run_id))
    provider_name = remote.get("provider")
    if not isinstance(provider_name, str) or not provider_name:
        raise ValueError("persisted provider identity is missing or invalid")
    launch_claim_token = remote.pop("launch_claim_token", None)
    if not isinstance(launch_claim_token, str) or not launch_claim_token:
        raise ValueError("persisted launch claim token is missing or invalid")
    recovered_attempt = _attempt_int(remote.get("attempt"))
    if recovered_attempt is None:
        raise ValueError("persisted attempt identity is missing or invalid")
    # strip the allocation stamp off the HANDLE copy only: `JobHandle.from_dict` round-trips
    # unknown keys, and the stamp belongs to `persisted_remote`, which `_carry_allocation_stamp`
    # reads whole when adopting metrics.
    remote.pop("allocated_gpu", None)
    remote.pop("allocated_gpu_count", None)
    remote.pop("allocated_usable_vram_gb", None)
    return _AttachContext(
        worker_spec=worker_spec,
        persisted_remote=persisted_remote,
        handle=JobHandle.from_dict(remote),
        recovered_attempt=recovered_attempt,
        next_attempt=recovered_attempt + 1,
        source_snapshot=source_snapshot,
        launch_claim_token=launch_claim_token,
    )


def _fail_unparseable_attach(run_id: str, status: RunStatus, exc: Exception, log) -> RunStatus:
    """Tear down and fail a run whose persisted public spec cannot be parsed."""
    from flash.providers.core.base import JobHandle
    from flash.runner.accounting.reconciliation import (
        _compare_and_confirm_remote_teardown,
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import _strict_teardown_handle

    # a parse failure cannot escape the daemon recovery thread while a worker may still bill.
    # fail closed using the exact persisted handle, without requiring a parsed spec.
    detail = f"unrecoverable: persisted spec is malformed: {exc}"
    confirmed_teardown = status.remote is None and status.cleanup_confirmed_remote is not None
    persisted_remote = dict(status.remote or status.cleanup_confirmed_remote or {})
    resource_deleted = confirmed_teardown
    if not confirmed_teardown:
        try:
            resource_deleted = _strict_teardown_handle(
                JobHandle.from_dict(persisted_remote), run_id
            )
        except Exception:
            resource_deleted = False
    # tear down before the state write and hand an unconfirmed deletion to the cleanup drainer:
    # failing the run first would drop the last record of a worker we have not proven gone.
    if resource_deleted and not confirmed_teardown:
        _compare_and_confirm_remote_teardown(run_id, persisted_remote)
    elif not resource_deleted:
        _record_cleanup_remote(run_id, persisted_remote)
    _compare_and_fail_remote(run_id, persisted_remote, detail)
    if not confirmed_teardown:
        from flash.providers.runpod.execution.provider import terminate_persisted_endpoints

        terminate_persisted_endpoints(status.spec, run_id)
    print(f"attach: {run_id} {detail}", file=log)
    return get_status(run_id)


def _handle_attach_wall_deadline(
    run_id: str,
    context: _AttachContext,
    log,
    exc: RuntimeError,
) -> RunStatus:
    """Adopt finished work or fail and tear down an attempt whose wall time is exhausted."""
    from flash.runner.accounting.reconciliation import (
        _compare_and_confirm_remote_teardown,
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import (
        _adopt_completed_attempt,
        _completed_attempt_metrics,
        _runpod_completed_metrics,
        _strict_teardown_handle,
    )

    deadline_at = _load_run_deadline_at(run_id)
    metrics = _runpod_completed_metrics(
        context.persisted_remote,
        deadline_at=deadline_at,
    )
    started_ts = context.persisted_remote.get("started_ts")
    if metrics is None and started_ts is not None:
        metrics = _completed_attempt_metrics(
            context.worker_spec,
            provider=context.handle.provider,
            attempt=context.recovered_attempt,
            launch_floor=float(started_ts),
            deadline_at=deadline_at,
            log=log,
        )
    if metrics is not None:
        _carry_allocation_stamp(metrics, context.persisted_remote)
        try:
            adopted = _adopt_completed_attempt(
                run_id,
                context.worker_spec,
                context.persisted_remote,
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
            return get_status(run_id)
        # completed work whose adoption is a transient defer (e.g. a cleanup blip) must NEVER be
        # torn down at the wall deadline; defer to background reconciliation, which retries
        # adoption until the deadline like the in-loop completion path.
        _schedule_attach_reconciliation(
            run_id,
            context.persisted_remote,
            context.worker_spec,
            context.next_attempt,
            context.source_snapshot,
            log,
            str(exc),
        )
        print(
            f"attach: {run_id} completed RunPod work at the wall deadline; "
            "deferring adoption to reconciliation",
            file=log,
        )
        return get_status(run_id)
    try:
        resource_deleted = _strict_teardown_handle(context.handle, run_id)
    except Exception:
        resource_deleted = False
    if resource_deleted:
        _compare_and_confirm_remote_teardown(run_id, context.persisted_remote)
    else:
        _record_cleanup_remote(run_id, context.persisted_remote)
    _compare_and_fail_remote(run_id, context.persisted_remote, str(exc))
    print(f"attach: {run_id} {exc}", file=log)
    return get_status(run_id)


def _handle_failed_attach_poll(
    run_id: str,
    context: _AttachContext,
    result: PollResult,
    log,
) -> RunStatus:
    """Adopt completed work or safely recover from an unsuccessful provider poll."""
    from flash.runner.accounting.reconciliation import (
        _compare_and_confirm_remote_teardown,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.attempts import decide_attempt_failure
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import (
        _adopt_completed_attempt,
        _runpod_completed_metrics,
        _strict_teardown_handle,
        _worker_provably_gone,
    )
    from flash.runner.supervise.retry_decision import (
        FailureObservation,
        _managed_cache_mounted,
        retry_candidate_from_remote,
    )

    failure = f"{result.failure or 'job_failed'}: {result.detail or 'provider attempt failed'}"
    print(f"attach: {run_id} ended ({result.failure}); evaluating recovery", file=log)
    completed_metrics = _runpod_completed_metrics(
        context.persisted_remote,
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
                context.worker_spec,
                context.persisted_remote,
                completed_metrics,
                log=log,
            )
        except Exception:
            adopted = False
        if adopted:
            print(f"attach: {run_id} adopted completed RunPod work", file=log)
            return get_status(run_id)
        _schedule_attach_reconciliation(
            run_id,
            context.persisted_remote,
            context.worker_spec,
            context.next_attempt,
            context.source_snapshot,
            log,
            failure,
        )
        print(
            f"attach: {run_id} completed RunPod work; deferring adoption to reconciliation",
            file=log,
        )
        return get_status(run_id)
    chosen = retry_candidate_from_remote(context.persisted_remote)
    plan = decide_attempt_failure(
        run_id,
        claim_token=context.launch_claim_token,
        expected_remote=context.persisted_remote,
        observation=FailureObservation(
            result.failure,
            chosen=chosen,
            candidates=None,
            managed_cache_mounted=_managed_cache_mounted(context.worker_spec, chosen),
        ),
        attempt=context.recovered_attempt,
    )
    if plan is None:
        return get_status(run_id)
    print(f"attach: {run_id} {plan.action}", file=log)
    try:
        resource_deleted = _strict_teardown_handle(context.handle, run_id)
        worker_gone = True
    except Exception:
        resource_deleted = False
        worker_gone = _worker_provably_gone(run_id, context.handle)
    if (
        worker_gone
        and context.handle.provider == "runpod"
        and not resource_deleted
        and not _record_cleanup_remote(run_id, context.persisted_remote)
    ):
        raise RuntimeError("leaked endpoint cleanup target could not be persisted")
    if resource_deleted:
        _compare_and_confirm_remote_teardown(run_id, context.persisted_remote)
    if worker_gone:
        if not plan.retry:
            from flash.runner.accounting.reconciliation import _compare_and_fail_remote

            _compare_and_fail_remote(run_id, context.persisted_remote, failure)
            return get_status(run_id)
        return _resume_after_confirmed_teardown(
            run_id,
            context.worker_spec,
            context.persisted_remote,
            context.next_attempt,
            context.source_snapshot,
            log,
            failure=failure,
        )
    if not plan.retry:
        print(
            f"attach: {run_id} teardown unconfirmed; retaining the terminal failure for reconciliation",
            file=log,
        )
    _schedule_attach_reconciliation(
        run_id,
        context.persisted_remote,
        context.worker_spec,
        context.next_attempt,
        context.source_snapshot,
        log,
        failure,
    )
    print(
        f"attach: {run_id} {context.handle.provider} teardown unconfirmed; "
        "reconciling the captured remote without resuming over a possibly-live resource",
        file=log,
    )
    return get_status(run_id)


def _adopt_attached_poll_result(
    run_id: str,
    context: _AttachContext,
    result: PollResult,
    log,
) -> None:
    """Restore allocation metadata and adopt one successful provider result."""
    from flash.runner.supervise.lifecycle import _adopt_completed_attempt

    # the shared carrier, not a local copy of two of its three fields: `_build_attach_context` pops
    # gpu and count off its handle copy but leaves `persisted_remote` whole, so the provider is
    # available here too. dropping it left `_gpu_rate` to fall back to whichever configured provider
    # offers the class -- normally RunPod -- so an attached vast or lambda run was priced at the
    # wrong substrate's rate and its notes named a provider that never ran it.
    _carry_allocation_stamp(result.metrics, context.persisted_remote)
    if not _adopt_completed_attempt(
        run_id,
        context.worker_spec,
        context.persisted_remote,
        result.metrics,
        log=log,
    ):
        print(
            f"attach: {run_id} persisted remote changed before completion adoption",
            file=log,
        )


def _recover_confirmed_remote(
    run_id: str,
    context: _AttachContext,
    source_snapshot: dict | None,
    log,
) -> RunStatus:
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.supervise.lifecycle import (
        _adopt_completed_attempt,
        _CompletedAttemptPending,
    )

    metrics = _completed_metrics_for_remote(
        context.worker_spec,
        context.persisted_remote,
        provider=context.handle.provider,
        attempt=context.recovered_attempt,
        deadline_at=_load_run_deadline_at(run_id),
        log=log,
    )
    if metrics is not None:
        if _adopt_completed_attempt(
            run_id,
            context.worker_spec,
            context.persisted_remote,
            metrics,
            log=log,
        ):
            from flash.runner.lifecycle.status import get_status

            return get_status(run_id)
        raise _CompletedAttemptPending("completed attempt adoption is pending durable confirmation")
    return _resume_after_confirmed_teardown(
        run_id,
        context.worker_spec,
        context.persisted_remote,
        context.next_attempt,
        source_snapshot,
        log,
        failure="persisted cleanup confirmed the worker was torn down",
    )


def attach_run(run_id: str, log_stream=None) -> RunStatus:
    """Re-attach to a run's remote job from any process (after a client crash/restart)."""
    import sys

    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at, _spec_with_remaining_wall
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import effective_spec_from_status, get_status
    from flash.runner.supervise.errors import _RunCancelled
    from flash.runner.supervise.lifecycle import _CompletedAttemptPending
    from flash.runner.supervise.recovery import _gc_run_endpoints

    cleanup_terminal = False

    def status_for_return() -> RunStatus:
        nonlocal cleanup_terminal
        current = get_status(run_id)
        cleanup_terminal = current.state in TERMINAL_STATES
        return current

    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    confirmed_teardown = status.remote is None and status.cleanup_confirmed_remote is not None
    if not status.remote and not confirmed_teardown:
        raise ValueError(f"run {run_id} has no persisted job handle; cannot reattach")

    # start from the lossy public view so the except/finally handlers always have a spec, then
    # upgrade to the authoritative worker spec (real run_id + managed fields) inside the try.
    try:
        worker_spec = JobSpec.from_dict(status.spec)
    except Exception as exc:
        # this parse is above the try below, so a raise here escapes every handler.
        return _fail_unparseable_attach(run_id, status, exc, log_stream or sys.stderr)
    persisted_remote = dict(status.remote or status.cleanup_confirmed_remote)
    next_attempt = 0
    source_snapshot = None
    log = log_stream or sys.stderr

    from flash.providers.core.registry import get_provider

    try:
        worker_spec = effective_spec_from_status(status)
        context = _build_attach_context(worker_spec, persisted_remote)
        next_attempt = context.next_attempt
        source_snapshot = context.source_snapshot
        if confirmed_teardown:
            recovered = _recover_confirmed_remote(run_id, context, source_snapshot, log)
            cleanup_terminal = recovered.state in TERMINAL_STATES
            return recovered
        try:
            poll_spec = _spec_with_remaining_wall(worker_spec, require_provider_minimum=False)
        except RuntimeError as exc:
            return _handle_attach_wall_deadline(run_id, context, log, exc)
        print(
            f"attaching to {run_id}: provider={context.handle.provider} {context.handle.data}",
            file=log,
        )
        result = get_provider(context.handle.provider).poll_attempt(
            context.handle,
            poll_spec,
            log=log,
            _deadline_at=_load_run_deadline_at(run_id),
        )
        if get_status(run_id).state == "cancelled":
            return status_for_return()
        if not result.ok:
            return _handle_failed_attach_poll(run_id, context, result, log)
        _adopt_attached_poll_result(run_id, context, result, log)
        return status_for_return()
    except _CompletedAttemptPending as exc:
        _schedule_attach_reconciliation(
            run_id,
            persisted_remote,
            worker_spec,
            next_attempt,
            source_snapshot,
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
    except StagedEnvironmentTransientError as exc:
        _schedule_attach_reconciliation(
            run_id,
            persisted_remote,
            worker_spec,
            next_attempt,
            source_snapshot,
            log,
            str(exc),
        )
        print(
            f"attach: {run_id} staged environment verification is temporarily unavailable; "
            "deferring replacement",
            file=log,
        )
        return status_for_return()
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
                    source_snapshot,
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
