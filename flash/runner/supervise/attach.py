"""reattach and reconcile a run whose original supervisor exited."""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
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


def _resume_after_confirmed_teardown(*args, **kwargs):
    from flash.runner.supervise.attach_replacement import resume_after_confirmed_teardown

    return resume_after_confirmed_teardown(*args, **kwargs)


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


def _defer_transient_result(
    run_id: str,
    expected_remote: dict,
    handle: JobHandle,
    result_deadline_at: float,
) -> bool:
    """defer one transient read only inside the persisted visibility window."""
    remaining = result_deadline_at - time.time()
    if remaining > 0:
        time.sleep(min(_ATTACH_RECONCILE_INTERVAL_S, remaining))
        return False
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.supervise.lifecycle import _strict_teardown_handle, _worker_provably_gone

    try:
        _strict_teardown_handle(handle, run_id)
        worker_gone = True
    except Exception:
        worker_gone = _worker_provably_gone(run_id, handle)
    if not _record_cleanup_remote(run_id, expected_remote):
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    if not worker_gone:
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    settled = _compare_and_fail_remote(
        run_id,
        expected_remote,
        "current-fence result observation remained unavailable through its deadline",
    )
    if not settled:
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
    return settled


def _fail_verified_result(run_id: str, expected_remote: dict, result: PollResult) -> bool:
    """persist one authoritative current-fence worker failure exactly."""
    from flash.runner.accounting.reconciliation import _compare_and_fail_remote
    from flash.runner.supervise.lifecycle import _result_failure_detail

    return _compare_and_fail_remote(run_id, expected_remote, _result_failure_detail(result))


def _oom_floor_from_remote(remote: dict, worker_spec: JobSpec) -> float:
    from flash.runner.supervise.attach_replacement import oom_floor_from_remote

    return oom_floor_from_remote(remote, worker_spec)


def _attach_retry_plan(context: _AttachContext, failure: str | None):
    """consume the persisted recovery budget and return its replacement inputs."""
    from flash.runner.lifecycle.deadlines import _spec_with_remaining_wall
    from flash.runner.supervise.lifecycle import (
        _failure_disposition,
        _reconstructed_retry_budget,
    )

    try:
        _spec_with_remaining_wall(context.worker_spec, require_provider_minimum=True)
    except RuntimeError:
        return None
    from flash.providers.core.registry import get_provider
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    max_retries = int(context.worker_spec.gpu.max_retries)
    cache_fallbacks = int(
        max_retries > 0
        and getattr(context.worker_spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME
        and getattr(get_provider(context.handle.provider), "supports_weight_cache", False)
    )
    budget = _reconstructed_retry_budget(
        max_retries,
        counters=context.retry_counters,
        cache_fallbacks=cache_fallbacks,
    )
    oom_vram_floor = (
        _oom_floor_from_remote(context.persisted_remote, context.worker_spec)
        if failure == "oom"
        else 0.0
    )
    cache_drop = bool(
        cache_fallbacks
        and budget.cache_used < budget.cache_fallbacks
        and failure in {"no_capacity", "poll_error"}
    )
    disposition = _failure_disposition(
        budget,
        failure,
        cache_drop=cache_drop,
        allow_retry=failure != "oom" or oom_vram_floor > 0,
    )
    if not disposition.retry:
        return None
    return budget, oom_vram_floor, cache_drop


def _reconcile_expired_remote(
    run_id: str,
    worker_spec: JobSpec,
    expected_remote: dict,
    handle: JobHandle,
    next_attempt: int,
    source_snapshot: dict | None,
    deadline_at: float,
    log,
    failure: str,
) -> bool:
    """Adopt late metrics or fail an attempt whose wall deadline has elapsed."""
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.protocol import AttemptRecord
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import (
        _adopt_completed_attempt,
        _attempt_result,
        _fail_permanent_result_artifact,
    )

    result_deadline_at = AttemptRecord.from_dict(get_status(run_id).attempt).result_deadline_at
    try:
        result = _attempt_result(run_id, expected_remote)
    except Exception as exc:
        if _result_transport_is_transient(exc):
            return _defer_transient_result(run_id, expected_remote, handle, result_deadline_at)
        return _fail_permanent_result_artifact(run_id, expected_remote, exc)
    if result is not None and not result.ok:
        context = _AttachContext(
            worker_spec=worker_spec,
            persisted_remote=expected_remote,
            handle=handle,
            seed=int(expected_remote.get("seed", worker_spec.seed)),
            recovered_attempt=next_attempt - 1,
            next_attempt=next_attempt,
            source_snapshot=source_snapshot,
            retry_counters=get_status(run_id).retry_counters,
        )
        return _dispose_authoritative_failure(run_id, context, result, log) is not None
    if result is not None:
        metrics = result.metrics
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
        _remote_resource_identity,
    )
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.protocol import AttemptRecord
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import (
        _attempt_result,
        _fail_permanent_result_artifact,
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
            observed_result = _attempt_result(run_id, expected_remote)
        except Exception as exc:
            if _result_transport_is_transient(exc):
                if _defer_transient_result(run_id, expected_remote, handle, result_deadline_at):
                    return
                continue
            if _fail_permanent_result_artifact(run_id, expected_remote, exc):
                return
            continue
        if observed_result is not None and not observed_result.ok:
            context = _AttachContext(
                worker_spec=worker_spec,
                persisted_remote=expected_remote,
                handle=handle,
                seed=int(expected_remote.get("seed", worker_spec.seed)),
                recovered_attempt=next_attempt - 1,
                next_attempt=next_attempt,
                source_snapshot=source_snapshot,
                retry_counters=status.retry_counters,
            )
            try:
                disposed = _dispose_authoritative_failure(run_id, context, observed_result, log)
            except Exception:
                time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
                continue
            if disposed is not None:
                return
            time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
            continue
        if observed_result is not None:
            completed_metrics = observed_result.metrics
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
                source_snapshot,
                deadline_at,
                log,
                failure,
            ):
                return
            continue
        if _wait_for_replacement_window(worker_spec, deadline_at):
            continue
        from flash.runner.supervise.attach_replacement import reconcile_absent_remote

        if reconcile_absent_remote(
            run_id,
            worker_spec,
            expected_remote,
            handle,
            next_attempt,
            source_snapshot,
            log,
            failure,
            expected_identity,
            _ATTACH_RECONCILE_INTERVAL_S,
        ):
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
    retry_counters: dict | None = None


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
    status = get_status(worker_spec.run_id)
    source_snapshot = source_snapshot_from_status(status)
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
    remote.pop("executed_gpu_count", None)
    return _AttachContext(
        worker_spec=worker_spec,
        persisted_remote=persisted_remote,
        handle=JobHandle.from_dict(remote),
        seed=seed,
        recovered_attempt=recovered_attempt,
        next_attempt=recovered_attempt + 1,
        source_snapshot=source_snapshot,
        retry_counters=status.retry_counters,
    )


def _dispose_authoritative_failure(
    run_id: str,
    context: _AttachContext,
    result: PollResult,
    log,
) -> RunStatus | None:
    """settle or replace one verified failure after strict resource disposition."""
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.lifecycle import (
        _result_failure_detail,
        _strict_teardown_handle,
        _worker_provably_gone,
    )

    plan = _attach_retry_plan(context, result.failure)
    try:
        _strict_teardown_handle(context.handle, run_id)
        worker_gone = True
    except Exception:
        worker_gone = _worker_provably_gone(run_id, context.handle)
    if not _record_cleanup_remote(run_id, context.persisted_remote):
        raise RuntimeError("authoritative failure cleanup identity could not be persisted")
    if not worker_gone:
        return None
    if plan is None:
        _compare_and_fail_remote(run_id, context.persisted_remote, _result_failure_detail(result))
        return get_status(run_id)
    retry_budget, oom_vram_floor, drop_weight_cache = plan
    return _resume_after_confirmed_teardown(
        run_id,
        context.worker_spec,
        context.persisted_remote,
        context.next_attempt,
        context.source_snapshot,
        log,
        failure=_result_failure_detail(result),
        retry_budget=retry_budget,
        oom_vram_floor=oom_vram_floor,
        drop_weight_cache=drop_weight_cache,
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


def _result_transport_is_transient(error: BaseException) -> bool:
    from flash.runner.supervise.lifecycle import _result_transport_is_transient as classify

    return classify(error)


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
        _attempt_result,
        _result_failure_detail,
        _strict_teardown_handle,
    )

    _load_run_deadline_at(run_id)
    try:
        observed_result = _attempt_result(run_id, context.persisted_remote)
    except Exception as artifact_error:
        if not _result_transport_is_transient(artifact_error):
            raise
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
            f"attach: {run_id} current-fence result observation is temporarily unavailable; "
            "deferring recovery",
            file=log,
        )
        return get_status(run_id)
    if observed_result is not None and not observed_result.ok:
        disposed = _dispose_authoritative_failure(run_id, context, observed_result, log)
        if disposed is not None:
            return disposed
        _schedule_attach_reconciliation(
            run_id,
            context.persisted_remote,
            context.worker_spec,
            context.next_attempt,
            context.source_snapshot,
            log,
            _result_failure_detail(observed_result),
        )
        return get_status(run_id)
    if observed_result is not None:
        metrics = observed_result.metrics
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
        _attempt_result,
        _result_failure_detail,
        _strict_teardown_handle,
        _worker_provably_gone,
    )

    failure = f"{result.failure or 'job_failed'}: {result.detail or 'provider attempt failed'}"
    print(f"attach: {run_id} ended ({result.failure}); evaluating recovery", file=log)
    try:
        observed_result = _attempt_result(run_id, context.persisted_remote)
    except Exception as artifact_error:
        if not _result_transport_is_transient(artifact_error):
            raise
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
            f"attach: {run_id} current-fence result observation is temporarily unavailable; "
            "deferring recovery",
            file=log,
        )
        return get_status(run_id)
    if observed_result is not None and not observed_result.ok:
        disposed = _dispose_authoritative_failure(run_id, context, observed_result, log)
        if disposed is not None:
            return disposed
        _schedule_attach_reconciliation(
            run_id,
            context.persisted_remote,
            context.worker_spec,
            context.next_attempt,
            context.source_snapshot,
            log,
            _result_failure_detail(observed_result),
        )
        return get_status(run_id)
    if observed_result is not None:
        completed_metrics = observed_result.metrics
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
    retry_plan = _attach_retry_plan(
        context,
        result.failure if result.failure in {"job_preempted", "poll_error"} else "job_preempted",
    )
    try:
        resource_deleted = _strict_teardown_handle(context.handle, run_id)
        worker_gone = True
    except Exception:
        resource_deleted = False
        worker_gone = _worker_provably_gone(run_id, context.handle)
    if (
        worker_gone
        and not resource_deleted
        and not _record_cleanup_remote(run_id, context.persisted_remote)
    ):
        raise RuntimeError("cleanup target could not be persisted")
    if worker_gone:
        if retry_plan is None:
            from flash.runner.accounting.reconciliation import _compare_and_fail_remote

            _compare_and_fail_remote(run_id, context.persisted_remote, failure)
            return get_status(run_id)
        retry_budget, oom_vram_floor, drop_weight_cache = retry_plan
        return _resume_after_confirmed_teardown(
            run_id,
            context.worker_spec,
            context.persisted_remote,
            context.next_attempt,
            context.source_snapshot,
            log,
            failure=failure,
            retry_budget=retry_budget,
            oom_vram_floor=oom_vram_floor,
            drop_weight_cache=drop_weight_cache,
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
