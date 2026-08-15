"""Reattaching to a run the control plane restarted away from, and reconciling what it finds.

`attach_run` is called for every non-terminal run at startup: the supervisor thread that owned the
run is gone, but the remote worker may still be alive, already finished, or already torn down.
Deciding which -- without double-billing, double-tearing-down, or resuming a run whose teardown was
never confirmed -- is what the reconciliation loop here does, on its own background thread.

Split out of `flash.runner.supervise.deploy` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from flash.core.spec import JobSpec, persisted_gpu_types
from flash.providers._lifecycle.deadline import deadline_kwargs
from flash.providers._lifecycle.poll import _attempt_int

# imported by value rather than through `_deploy()`: the set and its lock are mutated in place and
# never rebound, so both modules share the one object, and the tests that assert on membership read
# that same object through `deploy`. `_carry_allocation_stamp` is a plain unpatched helper.
from flash.runner.supervise.deploy import (
    _ATTACH_RECONCILING,
    _ATTACH_RECONCILING_LOCK,
    _carry_allocation_stamp,
)

if TYPE_CHECKING:
    from flash.providers.base import JobHandle, PollResult
    from flash.runner import RunStatus


def _deploy():
    """The deploy module, imported lazily because it re-exports this one.

    Five names here are patched as attributes of `flash.runner.supervise.deploy` by the attach
    tests -- `_ATTACH_RECONCILE_INTERVAL_S` (driven to 0.0 so a reconcile loop runs without real
    sleeps), the three reconciliation entry points, and `threading` (to run the scheduled thread
    inline) -- and every caller of theirs lives in this file. Resolving them through the parent is
    what keeps those patches effective; a direct reference would bind this module's own copy, and
    for the interval that copy is a float, so the patch could never reach it.
    """
    from flash.runner.supervise import deploy

    return deploy


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
        from flash.providers._lifecycle.worker import upload_code

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


def _reconcile_completed_remote(
    run_id: str,
    worker_spec: JobSpec,
    expected_remote: dict,
    completed_metrics: dict,
    deadline_at: float,
    log,
) -> bool:
    """Retry completed-attempt adoption and decide whether reconciliation is finished."""
    from flash.runner import _compare_and_fail_remote, _record_cleanup_remote
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
        # past the grace window the terminal CAS is the only exit; if it raised or
        # lost the compare-and-swap, rate-limit the retry at the full reconcile
        # interval. remaining grace is <= 0 here, so falling through to the shared
        # sleep below would sleep 0 and busy-spin the reconciler.
        time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
        return False
    remaining_grace = deadline_at + _RECOVERY_MARKER_GRACE_S - time.time()
    time.sleep(min(_deploy()._ATTACH_RECONCILE_INTERVAL_S, max(0.0, remaining_grace)))
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
    from flash.runner import _compare_and_fail_remote, _record_cleanup_remote
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
        time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
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
            time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
            return False
        if adopted:
            return True
        time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
        return False
    try:
        cleanup_preserved = _record_cleanup_remote(run_id, expected_remote)
    except Exception:
        cleanup_preserved = False
    if not cleanup_preserved:
        time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
        return False
    try:
        if _compare_and_fail_remote(run_id, expected_remote, failure):
            return True
    except Exception:
        time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
        return False
    return True


def _wait_for_replacement_window(worker_spec: JobSpec, deadline_at: float) -> bool:
    """Wait within the wall deadline and decide whether teardown should be deferred."""
    from flash.runner import _spec_with_remaining_wall

    delay = min(_deploy()._ATTACH_RECONCILE_INTERVAL_S, max(0.0, deadline_at - time.time()))
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
            time.sleep(
                min(_deploy()._ATTACH_RECONCILE_INTERVAL_S, max(0.0, deadline_at - time.time()))
            )
            return True
    return False


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
        get_status,
    )
    from flash.runner.supervise.lifecycle import (
        _RECOVERY_MARKER_GRACE_S,
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
            time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
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
                time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
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
            pending_interval = _deploy()._ATTACH_RECONCILE_INTERVAL_S
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
            _deploy()._resume_after_confirmed_teardown(
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
                time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
                continue
            if current.state in TERMINAL_STATES:
                return
            if current.remote is None:
                try:
                    if _compare_and_fail_remote(run_id, None, str(exc)):
                        return
                except Exception:
                    time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
                    continue
                return
            if _remote_resource_identity(current.remote) != expected_identity:
                return
            time.sleep(_deploy()._ATTACH_RECONCILE_INTERVAL_S)
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
            _deploy()._reconcile_attached_remote(
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
        _deploy().threading.Thread(target=run, daemon=True).start()
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
    seed: int
    recovered_attempt: int
    next_attempt: int
    code_prefix: str | None
    allocated_gpu: object | None
    allocated_gpu_count: object | None


def _build_attach_context(
    worker_spec: JobSpec,
    persisted_remote: dict,
) -> _AttachContext:
    """Validate the persisted handle and collect the inputs needed to poll it."""
    from flash.providers.base import JobHandle

    remote = dict(persisted_remote)
    seed = int(remote.pop("seed", worker_spec.seed))
    code_prefix = remote.pop("code_prefix", None)
    provider_name = remote.get("provider")
    if not isinstance(provider_name, str) or not provider_name:
        raise ValueError("persisted provider identity is missing or invalid")
    recovered_attempt = _attempt_int(remote.get("attempt"))
    if recovered_attempt is None:
        raise ValueError("persisted attempt identity is missing or invalid")
    allocated_gpu = remote.pop("allocated_gpu", None)
    allocated_gpu_count = remote.pop("allocated_gpu_count", None)
    return _AttachContext(
        worker_spec=worker_spec,
        persisted_remote=persisted_remote,
        handle=JobHandle.from_dict(remote),
        seed=seed,
        recovered_attempt=recovered_attempt,
        next_attempt=recovered_attempt + 1,
        code_prefix=code_prefix,
        allocated_gpu=allocated_gpu,
        allocated_gpu_count=allocated_gpu_count,
    )


def _fail_unparseable_attach(run_id: str, status: RunStatus, exc: Exception, log) -> RunStatus:
    """Tear down and fail a run whose persisted public spec cannot be parsed."""
    from flash.providers.base import JobHandle
    from flash.runner import _compare_and_fail_remote, _record_cleanup_remote, get_status
    from flash.runner.supervise.lifecycle import _strict_teardown_handle

    # attach_run is dispatched on a daemon thread, so an escaped parse failure is silent: the run
    # stays nonterminal with a live handle and its worker keeps billing. A spec stops parsing when
    # the plane upgrades past an algorithm a still-in-flight run was accepted under. It cannot be
    # resumed, so fail it closed and tear the worker down. `_gc_run_endpoints` needs a parsed spec
    # we do not have; the endpoint name is derived from the run id plus GPU class, both readable
    # from the raw persisted status, which is the same route recover_runs takes for its own
    # unparseable-spec branch.
    detail = f"unrecoverable: persisted spec is malformed: {exc}"
    persisted_remote = dict(status.remote)
    try:
        resource_deleted = _strict_teardown_handle(JobHandle.from_dict(persisted_remote), run_id)
    except Exception:
        resource_deleted = False
    # Tear down BEFORE the state write and hand an unconfirmed deletion to the cleanup drainer:
    # failing the run first would drop the last record of a worker we have not proven gone.
    if not resource_deleted:
        _record_cleanup_remote(run_id, persisted_remote)
    _compare_and_fail_remote(run_id, persisted_remote, detail)
    # every acceptable class, not just the head: allocation may have rented a fallback, and the
    # endpoint name is reconstructed from the class rather than read back, so naming only the head
    # leaves a fallback endpoint running. isolate each name so one malformed entry cannot prevent
    # cleanup of the remaining valid classes.
    for gpu_type in persisted_gpu_types(status.spec):
        with contextlib.suppress(Exception):
            from flash.providers.runpod.serverless import terminate_endpoint

            terminate_endpoint(gpu_type, run_id)
    print(f"attach: {run_id} {detail}", file=log)
    return get_status(run_id)


def _handle_attach_wall_deadline(
    run_id: str,
    context: _AttachContext,
    log,
    exc: RuntimeError,
) -> RunStatus:
    """Adopt finished work or fail and tear down an attempt whose wall time is exhausted."""
    from flash.runner import (
        _compare_and_fail_remote,
        _load_run_deadline_at,
        _record_cleanup_remote,
        get_status,
    )
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
        _deploy()._schedule_attach_reconciliation(
            run_id,
            context.persisted_remote,
            context.worker_spec,
            context.next_attempt,
            context.code_prefix,
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
    if not resource_deleted:
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
    from flash.runner import _load_run_deadline_at, _record_cleanup_remote, get_status
    from flash.runner.supervise.lifecycle import (
        _adopt_completed_attempt,
        _runpod_completed_metrics,
        _strict_teardown_handle,
        _worker_provably_gone,
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
        _deploy()._schedule_attach_reconciliation(
            run_id,
            context.persisted_remote,
            context.worker_spec,
            context.next_attempt,
            context.code_prefix,
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
    if worker_gone:
        return _deploy()._resume_after_confirmed_teardown(
            run_id,
            context.worker_spec,
            context.persisted_remote,
            context.next_attempt,
            context.code_prefix,
            log,
            failure=failure,
        )
    _deploy()._schedule_attach_reconciliation(
        run_id,
        context.persisted_remote,
        context.worker_spec,
        context.next_attempt,
        context.code_prefix,
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

    if context.allocated_gpu and isinstance(result.metrics, dict):
        result.metrics.setdefault("allocated_gpu", context.allocated_gpu)
    if context.allocated_gpu_count and isinstance(result.metrics, dict):
        result.metrics.setdefault("allocated_gpu_count", int(context.allocated_gpu_count))
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
    from flash.runner.supervise.lifecycle import _CompletedAttemptPending

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

    # seed from the lossy public view so the except/finally handlers always have a spec, then
    # upgrade to the authoritative worker spec (real run_id + managed fields) inside the try.
    try:
        worker_spec = JobSpec.from_dict(status.spec)
    except Exception as exc:
        # this parse is above the try below, so a raise here escapes every handler.
        return _fail_unparseable_attach(run_id, status, exc, log_stream or sys.stderr)
    persisted_remote = dict(status.remote)
    next_attempt = 0
    code_prefix = None
    log = log_stream or sys.stderr

    from flash.providers import get_provider

    try:
        worker_spec = effective_spec_from_status(status)
        context = _build_attach_context(worker_spec, persisted_remote)
        next_attempt = context.next_attempt
        code_prefix = context.code_prefix
        try:
            poll_spec = _spec_with_remaining_wall(worker_spec, require_provider_minimum=False)
        except RuntimeError as exc:
            return _handle_attach_wall_deadline(run_id, context, log, exc)
        print(
            f"attaching to {run_id}: provider={context.handle.provider} {context.handle.data}",
            file=log,
        )
        result = get_provider(context.handle.provider).poll(
            context.handle,
            poll_spec,
            context.seed,
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
        _deploy()._schedule_attach_reconciliation(
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
                _deploy()._schedule_attach_reconciliation(
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
