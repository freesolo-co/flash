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

_ATTACH_RECONCILE_INTERVAL_S = 120.0
_ATTACH_RECONCILING: set[str] = set()
_ATTACH_RECONCILING_LOCK = threading.Lock()


def _carry_allocation_stamp(metrics: dict, remote: dict | None) -> None:
    """carry the persisted allocation stamp onto adopted metrics."""
    if not isinstance(metrics, dict) or not isinstance(remote, dict):
        return
    allocated_gpu = remote.get("allocated_gpu")
    if allocated_gpu:
        metrics.setdefault("allocated_gpu", allocated_gpu)
    allocated_count = remote.get("allocated_gpu_count")
    if allocated_count:
        metrics.setdefault("allocated_gpu_count", int(allocated_count))
    provider = remote.get("provider")
    if provider:
        metrics.setdefault("allocated_provider", provider)


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
        _compare_and_clear_remote,
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.attempts import _verified_opd_next_attempt
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at, _spec_with_remaining_wall
    from flash.runner.lifecycle.status import (
        _update,
        get_status,
        reallocation_spec_from_status,
        source_snapshot_from_status,
    )
    from flash.runner.lifecycle.submit import _persist_effective_worker_spec
    from flash.runner.supervise.errors import _RunCancelled
    from flash.runner.supervise.lifecycle import _run_training

    if int(worker_spec.gpu.max_retries) == 0:
        _compare_and_fail_remote(run_id, persisted_remote, failure)
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
        _compare_and_fail_remote(run_id, persisted_remote, str(exc))
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
    if worker_spec.run_id != run_id:
        worker_spec = replace(worker_spec, run_id=run_id)
    deadline_at = _load_run_deadline_at(run_id)
    worker_spec = stage_environment_package(worker_spec, deadline_at=deadline_at)
    if not _persist_effective_worker_spec(worker_spec):
        raise _RunCancelled(f"run {run_id} went terminal before environment staging")
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
        _attempt_result_metrics,
    )

    try:
        metrics = _attempt_result_metrics(run_id, expected_remote)
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
    """Reconcile one exact maybe-live attempt until it is gone or the wall deadline expires."""
    from flash.providers.core.base import JobHandle
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
        _remote_resource_identity,
    )
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.protocol import AttemptRecord
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import (
        _attempt_result_metrics,
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
        attempt_record = AttemptRecord.from_dict(status.attempt)
        result_deadline_at = attempt_record.result_deadline_at
        try:
            completed_metrics = _attempt_result_metrics(run_id, expected_remote)
        except Exception:
            completed_metrics = None
        if completed_metrics is not None:
            _carry_allocation_stamp(completed_metrics, expected_remote)
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
        if time.time() >= result_deadline_at:
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
            _resume_after_confirmed_teardown(
                run_id,
                worker_spec,
                expected_remote,
                next_attempt,
                source_snapshot,
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
    seed: int
    recovered_attempt: int
    next_attempt: int
    source_snapshot: dict | None


def _build_attach_context(
    worker_spec: JobSpec,
    persisted_remote: dict,
) -> _AttachContext:
    """Validate the persisted handle and collect the inputs needed to poll it."""
    from flash.providers.core.base import JobHandle
    from flash.runner.lifecycle.status import get_status, source_snapshot_from_status

    remote = dict(persisted_remote)
    seed = int(remote.pop("seed", worker_spec.seed))
    remote.pop("code_prefix", None)
    source_snapshot = source_snapshot_from_status(get_status(worker_spec.run_id))
    provider_name = remote.get("provider")
    if not isinstance(provider_name, str) or not provider_name:
        raise ValueError("persisted provider identity is missing or invalid")
    recovered_attempt = _attempt_int(remote.get("attempt"))
    if recovered_attempt is None:
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
        seed=seed,
        recovered_attempt=recovered_attempt,
        next_attempt=recovered_attempt + 1,
        source_snapshot=source_snapshot,
    )


def _fail_unparseable_attach(run_id: str, status: RunStatus, exc: Exception, log) -> RunStatus:
    """Tear down and fail a run whose persisted public spec cannot be parsed."""
    from flash.providers.core.base import JobHandle
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.status import get_status
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
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import (
        _adopt_completed_attempt,
        _attempt_result_metrics,
        _strict_teardown_handle,
    )

    _load_run_deadline_at(run_id)
    metrics = _attempt_result_metrics(run_id, context.persisted_remote)
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
    from flash.runner.accounting.reconciliation import _record_cleanup_remote
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
    if worker_gone:
        return _resume_after_confirmed_teardown(
            run_id,
            context.worker_spec,
            context.persisted_remote,
            context.next_attempt,
            context.source_snapshot,
            log,
            failure=failure,
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
    source_snapshot = None
    log = log_stream or sys.stderr

    from flash.providers.core.registry import get_provider

    try:
        worker_spec = effective_spec_from_status(status)
        context = _build_attach_context(worker_spec, persisted_remote)
        next_attempt = context.next_attempt
        source_snapshot = context.source_snapshot
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
