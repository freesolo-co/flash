"""execute one bounded replacement after attach proves the prior worker is gone."""

from __future__ import annotations

import contextlib
from dataclasses import replace

from flash.core.spec import JobSpec
from flash.providers._lifecycle.instances.poll import _attempt_int


def _oom_floor(worker_spec: JobSpec, gpu: object, count: object, executed: object = None) -> float:
    from flash.cost.spec import sft_ranking_overrides
    from flash.providers.core.allocator import _executed_gpu_count
    from flash.providers.core.base import GPU_INFO, Candidate
    from flash.runner.supervise.lifecycle import _candidate_usable_vram_gb

    gpu_class = GPU_INFO.get(gpu)
    if gpu_class is None:
        return 0.0
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return 0.0
    if isinstance(executed, bool) or not isinstance(executed, int) or not 1 <= executed <= count:
        executed = _executed_gpu_count(
            worker_spec.algorithm,
            worker_spec.train,
            sft_ranking_overrides(worker_spec),
            count,
        )
    if isinstance(executed, bool) or not isinstance(executed, int) or not 1 <= executed <= count:
        return 0.0
    return _candidate_usable_vram_gb(
        Candidate(
            "",
            gpu_class.name,
            0.0,
            gpu_class.vram_gb,
            count,
            executed_gpu_count=executed,
        )
    )


def oom_floor_from_remote(remote: dict, worker_spec: JobSpec) -> float:
    gpu = remote.get("allocated_gpu") or remote.get("gpu")
    count = remote.get("allocated_gpu_count", 1)
    return _oom_floor(worker_spec, gpu, count, remote.get("executed_gpu_count"))


def oom_floor_from_effective_spec(worker_spec: JobSpec) -> float:
    """derive the executed allocation floor only from the persisted selected worker spec."""
    return _oom_floor(worker_spec, worker_spec.gpu.type, worker_spec.gpu.count)


def resume_after_confirmed_teardown(
    run_id: str,
    worker_spec: JobSpec,
    persisted_remote: dict,
    next_attempt: int,
    source_snapshot: dict | None,
    log,
    *,
    failure: str,
    retry_budget,
    oom_vram_floor: float = 0.0,
    drop_weight_cache: bool = False,
):
    """cas-clear one captured remote, then resume its next attempt exactly once."""
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
    if drop_weight_cache:
        from flash.runner.supervise.lifecycle import _drop_weight_cache

        worker_spec = _drop_weight_cache(worker_spec)
    deadline_at = _load_run_deadline_at(run_id)
    worker_spec = stage_environment_package(worker_spec, deadline_at=deadline_at)
    if not _persist_effective_worker_spec(worker_spec):
        raise _RunCancelled(f"run {run_id} went terminal before environment staging")
    if not _compare_and_clear_remote(
        run_id,
        persisted_remote,
        retry_counters=retry_budget.counters(),
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
            retry_budget=retry_budget,
            oom_vram_floor=oom_vram_floor,
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
            identity_kwargs = {}
            if current_remote is None:
                from flash.runner.lifecycle.protocol import AttemptRecord

                attempt = AttemptRecord.from_dict(current.attempt)
                identity_kwargs = {
                    "expected_attempt_id": attempt.attempt_id,
                    "expected_fence": attempt.fence,
                }
            _compare_and_fail_remote(
                run_id,
                current_remote,
                str(exc),
                **identity_kwargs,
            )
        raise
    return get_status(run_id)


def reconcile_absent_remote(
    run_id: str,
    worker_spec: JobSpec,
    expected_remote: dict,
    handle,
    next_attempt: int,
    source_snapshot: dict | None,
    log,
    failure: str,
    expected_identity: tuple,
    reconcile_interval_s: float,
) -> bool:
    """settle or replace one result-absent remote after strict teardown."""
    import time

    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
        _remote_resource_identity,
    )
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.attach import (
        _attach_retry_plan,
        _AttachContext,
        _resume_after_confirmed_teardown,
    )
    from flash.runner.supervise.lifecycle import (
        _strict_teardown_handle,
        _worker_provably_gone,
    )

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
    retry_plan = _attach_retry_plan(context, failure.partition(":")[0])
    try:
        resource_deleted = _strict_teardown_handle(handle, run_id)
        worker_gone = True
    except Exception:
        resource_deleted = False
        worker_gone = _worker_provably_gone(run_id, handle)
    if not worker_gone:
        return False
    if not resource_deleted:
        try:
            cleanup_preserved = _record_cleanup_remote(run_id, expected_remote)
        except Exception:
            cleanup_preserved = False
        if not cleanup_preserved:
            return False
    if retry_plan is None:
        try:
            return bool(_compare_and_fail_remote(run_id, expected_remote, failure))
        except Exception:
            time.sleep(reconcile_interval_s)
            return False
    retry_budget, oom_vram_floor, drop_weight_cache = retry_plan
    try:
        _resume_after_confirmed_teardown(
            run_id,
            worker_spec,
            expected_remote,
            next_attempt,
            source_snapshot,
            log,
            failure=failure,
            retry_budget=retry_budget,
            oom_vram_floor=oom_vram_floor,
            drop_weight_cache=drop_weight_cache,
        )
    except Exception as exc:
        try:
            current = get_status(run_id)
        except Exception:
            time.sleep(reconcile_interval_s)
            return False
        if current.state in TERMINAL_STATES:
            return True
        if current.remote is None:
            try:
                from flash.runner.lifecycle.protocol import AttemptRecord

                attempt = AttemptRecord.from_dict(current.attempt)
                return bool(
                    _compare_and_fail_remote(
                        run_id,
                        None,
                        str(exc),
                        expected_attempt_id=attempt.attempt_id,
                        expected_fence=attempt.fence,
                    )
                )
            except Exception:
                time.sleep(reconcile_interval_s)
                return False
        return _remote_resource_identity(current.remote) != expected_identity
    return True
