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
    teardown_reconciled_remote as _teardown_reconciled_remote,
)

_ATTACH_RECONCILE_INTERVAL_S = 120.0
_ATTACH_RECONCILING: set[str] = set()
_ATTACH_RECONCILING_LOCK = threading.Lock()


class _CompletedAttemptPending(RuntimeError):
    """A verified terminal result is waiting for durable adoption."""


if TYPE_CHECKING:
    from flash.providers.core.base import JobHandle, PollResult
    from flash.runner.lifecycle.state import RunStatus


def _verified_attached_opd_next_attempt(
    worker_spec: JobSpec, run_id: str, next_attempt: int
) -> int:
    """verify mutation safety before replacing or terminalizing an attached opd attempt."""
    if worker_spec.algorithm != "opd":
        return next_attempt
    from flash.runner.lifecycle.attempts import _verified_opd_next_attempt

    verified = _verified_opd_next_attempt(run_id)
    if verified != next_attempt:
        raise RuntimeError(
            "persisted opd attempt identity does not match the attached worker; "
            "replacement is blocked"
        )
    return verified


def _resume_after_confirmed_teardown(
    run_id: str,
    worker_spec: JobSpec,
    persisted_remote: dict,
    next_attempt: int,
    source_snapshot: dict | None,
    log,
    *,
    failure: str,
    expected_attempt: tuple[int, int],
    retry_failure: str | None = None,
) -> RunStatus:
    """CAS-clear one captured remote, then resume its next attempt exactly once."""
    from flash.runner.accounting.artifacts import stage_environment_package
    from flash.runner.accounting.reconciliation import (
        _compare_and_clear_remote,
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at, _spec_with_remaining_wall
    from flash.runner.lifecycle.status import (
        _update,
        get_status,
        reallocation_spec_from_status,
        source_snapshot_from_status,
    )
    from flash.runner.lifecycle.submit import _persist_effective_worker_spec
    from flash.runner.supervise.errors import _RunCancelled
    from flash.runner.supervise.lifecycle import (
        _consume_recovered_retry_state,
        _run_training,
    )

    if retry_failure is not None:
        retry_policy = _consume_recovered_retry_state(
            worker_spec,
            get_status(run_id),
            retry_failure,
            expected_attempt,
            persisted_remote,
        )
        if retry_policy is None:
            _verified_attached_opd_next_attempt(worker_spec, run_id, next_attempt)
            _compare_and_fail_remote(
                run_id,
                persisted_remote,
                failure,
                expected_attempt=expected_attempt,
            )
            return get_status(run_id)
    if int(worker_spec.gpu.max_retries) == 0:
        _compare_and_fail_remote(
            run_id,
            persisted_remote,
            failure,
            expected_attempt=expected_attempt,
        )
        print(
            f"attach: {run_id} exhausted its one-shot retry budget; not resubmitting",
            file=log,
        )
        return get_status(run_id)
    try:
        from flash.snapshot.archive import parse_descriptor

        source_snapshot = parse_descriptor(
            source_snapshot or source_snapshot_from_status(get_status(run_id), required=True)
        ).to_dict()
    except Exception as exc:
        _compare_and_fail_remote(
            run_id,
            persisted_remote,
            str(exc),
            expected_attempt=expected_attempt,
        )
        return get_status(run_id)
    next_attempt = _verified_attached_opd_next_attempt(worker_spec, run_id, next_attempt)
    try:
        _spec_with_remaining_wall(worker_spec, require_provider_minimum=True)
    except RuntimeError as exc:
        _compare_and_fail_remote(
            run_id,
            persisted_remote,
            str(exc),
            expected_attempt=expected_attempt,
        )
        print(f"attach: {run_id} {exc}", file=log)
        return get_status(run_id)
    worker_spec = reallocation_spec_from_status(get_status(run_id), verify_source=True)
    if worker_spec.run_id != run_id:
        worker_spec = replace(worker_spec, run_id=run_id)
    deadline_at = _load_run_deadline_at(run_id)
    worker_spec = stage_environment_package(worker_spec, deadline_at=deadline_at)
    if not _persist_effective_worker_spec(worker_spec):
        raise _RunCancelled(f"run {run_id} went terminal before environment staging")
    if not _compare_and_clear_remote(
        run_id,
        persisted_remote,
        expected_attempt=expected_attempt,
    ):
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
    try:
        _run_training(
            worker_spec,
            log,
            prior_cost=float(get_status(run_id).cost_usd or 0.0),
            source_snapshot=source_snapshot,
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
    from flash.runner.lifecycle.protocol import AttemptRecord
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import _adopt_completed_attempt

    result_deadline_at = AttemptRecord.from_dict(get_status(run_id).attempt).result_deadline_at

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
    if time.time() >= result_deadline_at:
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
        # past the grace window the terminal CAS is the only exit; if it raised or
        # lost the compare-and-swap, rate-limit the retry at the full reconcile
        # interval. remaining grace is <= 0 here, so falling through to the shared
        # sleep below would sleep 0 and busy-spin the reconciler.
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    remaining_grace = result_deadline_at - time.time()
    time.sleep(min(_ATTACH_RECONCILE_INTERVAL_S, max(0.0, remaining_grace)))
    return False


def _reconcile_attached_remote(
    run_id: str,
    expected_remote: dict,
    worker_spec: JobSpec,
    next_attempt: int,
    source_snapshot: dict | None,
    log,
    failure: str,
    retry_failure: str | None = None,
) -> None:
    from flash.providers.core.base import JobHandle
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _remote_attempt_identity,
        _remote_resource_identity,
    )
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.protocol import AttemptRecord
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import _attempt_result

    expected_identity = _remote_resource_identity(expected_remote)
    expected_attempt = _remote_attempt_identity(expected_remote)
    if expected_attempt is None:
        return
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
                if _compare_and_fail_remote(
                    run_id,
                    expected_remote,
                    str(exc),
                    expected_attempt=expected_attempt,
                ):
                    return
            except Exception:
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            return
        attempt_record = AttemptRecord.from_dict(status.attempt)
        if (attempt_record.attempt_id, attempt_record.fence) != expected_attempt:
            return
        result_deadline_at = attempt_record.result_deadline_at
        if confirmed_teardown:
            try:
                _recover_confirmed_remote(
                    run_id, worker_spec, expected_remote, source_snapshot, log
                )
            except StagedEnvironmentTransientError:
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            except Exception:
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            return
        try:
            terminal_result = _attempt_result(run_id, expected_remote)
        except Exception:
            terminal_result = None
        if terminal_result is not None:
            from flash.runner.supervise.attach_retry import reconcile_terminal_result

            if reconcile_terminal_result(
                run_id,
                worker_spec,
                expected_remote,
                expected_attempt,
                handle,
                terminal_result,
                deadline_at,
                next_attempt,
                source_snapshot,
                status,
                log,
            ):
                return
            continue
        now = time.time()
        if now < result_deadline_at:
            time.sleep(min(_ATTACH_RECONCILE_INTERVAL_S, result_deadline_at - now))
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
                expected_attempt=expected_attempt,
                retry_failure=retry_failure,
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
    retry_failure: str | None = None,
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
                retry_failure,
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

    @property
    def expected_attempt(self) -> tuple[int, int]:
        fence = _attempt_int(self.persisted_remote.get("fence"))
        if fence is None:
            raise ValueError("persisted attempt fence is missing or invalid")
        return self.recovered_attempt, fence


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
    recovered_attempt = _attempt_int(remote.get("attempt"))
    recovered_fence = _attempt_int(remote.get("fence"))
    if recovered_attempt is None or recovered_fence is None:
        raise ValueError("persisted attempt identity is missing or invalid")
    # strip the allocation stamp off the HANDLE copy only: `JobHandle.from_dict` round-trips
    # unknown keys, and the stamp belongs to `persisted_remote`, which `_carry_allocation_stamp`
    # reads whole when adopting metrics.
    remote.pop("allocated_gpu", None)
    remote.pop("allocated_gpu_count", None)
    return _AttachContext(
        worker_spec=worker_spec,
        persisted_remote=persisted_remote,
        handle=JobHandle.from_dict(remote),
        recovered_attempt=recovered_attempt,
        next_attempt=recovered_attempt + 1,
        source_snapshot=source_snapshot,
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


def _adopt_persisted_attach_result(
    run_id: str,
    context: _AttachContext,
    attempt,
    log,
    failure: str,
) -> tuple[PollResult | None, RunStatus | None]:
    """decode and adopt the current fenced persisted success, or defer it safely."""
    from flash.providers.artifacts.attempts import AttemptArtifactError, poll_result_from_manifest
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import _adopt_completed_attempt

    status = get_status(run_id)
    persisted_result = status.result if isinstance(status.result, dict) else None
    if persisted_result is None:
        return None, None
    if (
        persisted_result.get("attempt_id") != attempt.attempt_id
        or persisted_result.get("fence") != attempt.fence
    ):
        raise AttemptArtifactError("persisted result does not match the current fenced attempt")
    terminal_result = poll_result_from_manifest(persisted_result)
    if not terminal_result.ok:
        return terminal_result, None
    completed_metrics = terminal_result.metrics
    _carry_allocation_stamp(completed_metrics, context.persisted_remote)
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
        print(
            f"attach: {run_id} adopted a persisted completed attempt at the wall deadline",
            file=log,
        )
        return terminal_result, get_status(run_id)
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
        f"attach: {run_id} persisted completed work at the wall deadline; "
        "deferring adoption to reconciliation",
        file=log,
    )
    return terminal_result, get_status(run_id)


def _handle_attach_wall_deadline(
    run_id: str,
    context: _AttachContext,
    log,
    exc: RuntimeError,
) -> RunStatus:
    """Adopt finished work or fail and tear down an attempt whose wall time is exhausted."""
    from flash.providers.artifacts.attempts import AttemptArtifactError
    from flash.runner.accounting.reconciliation import (
        _compare_and_confirm_remote_teardown,
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.protocol import AttemptRecord
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import (
        _adopt_completed_attempt,
        _attempt_result_metrics,
        _strict_teardown_handle,
    )

    _load_run_deadline_at(run_id)
    status = get_status(run_id)
    attempt = AttemptRecord.from_dict(status.attempt)
    result_deadline_at = attempt.result_deadline_at
    try:
        metrics = _attempt_result_metrics(run_id, context.persisted_remote)
    except AttemptArtifactError:
        raise
    except Exception:
        metrics = None
        if time.time() < result_deadline_at:
            _schedule_attach_reconciliation(
                run_id,
                context.persisted_remote,
                context.worker_spec,
                context.next_attempt,
                context.source_snapshot,
                log,
                str(exc),
            )
            return get_status(run_id)
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
    terminal_result, persisted_status = _adopt_persisted_attach_result(
        run_id,
        context,
        attempt,
        log,
        str(exc),
    )
    if persisted_status is not None:
        return persisted_status
    terminal_failure = terminal_result is not None and not terminal_result.ok
    if terminal_failure:
        return _handle_failed_attach_poll(run_id, context, terminal_result, log)
    failure = str(exc)
    if time.time() < result_deadline_at:
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
            f"attach: {run_id} work deadline passed; awaiting terminal result visibility",
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
    _compare_and_fail_remote(
        run_id,
        context.persisted_remote,
        failure,
        expected_attempt=context.expected_attempt,
    )
    print(f"attach: {run_id} {failure}", file=log)
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
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import (
        _adopt_completed_attempt,
        _attempt_result_metrics,
        _strict_teardown_handle,
        _worker_provably_gone,
    )

    failure = f"{result.failure or 'job_failed'}: {result.detail or 'provider attempt failed'}"
    print(f"attach: {run_id} ended ({result.failure}); evaluating recovery", file=log)
    completed_metrics = _attempt_result_metrics(run_id, context.persisted_remote)
    if completed_metrics is not None:
        _carry_allocation_stamp(completed_metrics, context.persisted_remote)
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
        return _resume_after_confirmed_teardown(
            run_id,
            context.worker_spec,
            context.persisted_remote,
            context.next_attempt,
            context.source_snapshot,
            log,
            failure=failure,
            expected_attempt=context.expected_attempt,
            retry_failure=result.failure,
        )
    _schedule_attach_reconciliation(
        run_id,
        context.persisted_remote,
        context.worker_spec,
        context.next_attempt,
        context.source_snapshot,
        log,
        failure,
        result.failure,
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
    worker_spec: JobSpec,
    persisted_remote: dict,
    source_snapshot: dict | None,
    log,
) -> RunStatus:
    """Adopt immutable terminal evidence or resume after an exact confirmed teardown."""
    from flash.runner.accounting.reconciliation import _remote_attempt_identity
    from flash.runner.supervise.lifecycle import _adopt_completed_attempt, _attempt_result

    expected_attempt = _remote_attempt_identity(persisted_remote)
    if expected_attempt is None:
        raise ValueError("persisted attempt identity is missing or invalid")
    terminal_result = _attempt_result(run_id, persisted_remote)
    if terminal_result is not None and terminal_result.ok:
        _carry_allocation_stamp(terminal_result.metrics, persisted_remote)
        if _adopt_completed_attempt(
            run_id,
            worker_spec,
            persisted_remote,
            terminal_result.metrics,
            log=log,
        ):
            from flash.runner.lifecycle.status import get_status

            return get_status(run_id)
        raise _CompletedAttemptPending("completed attempt adoption is pending durable confirmation")
    failure = "persisted cleanup confirmed the worker was torn down"
    retry_failure = None
    if terminal_result is not None:
        retry_failure = terminal_result.failure
        failure = (
            f"{terminal_result.failure or 'job_failed'}: "
            f"{terminal_result.detail or 'worker attempt failed'}"
        )
    return _resume_after_confirmed_teardown(
        run_id,
        worker_spec,
        persisted_remote,
        expected_attempt[0] + 1,
        source_snapshot,
        log,
        failure=failure,
        expected_attempt=expected_attempt,
        retry_failure=retry_failure,
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

    # seed from the lossy public view so the except/finally handlers always have a spec, then
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
            recovered = _recover_confirmed_remote(
                run_id,
                context.worker_spec,
                context.persisted_remote,
                source_snapshot,
                log,
            )
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
    except _RunCancelled:
        with contextlib.suppress(Exception):
            cleanup_terminal = get_status(run_id).state in TERMINAL_STATES
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
        return status_for_return()
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
