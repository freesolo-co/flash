"""run-execution machinery for submission, supervision, and cleanup."""

from __future__ import annotations

import contextlib
import os
import time

from flash._internal.background import checkpoint, pause
from flash.core.spec import JobSpec, gpu_count_of
from flash.runner.lifecycle.attempts import AttemptLaunchClaim
from flash.runner.supervise import recovery as attempt_recovery

_STAGED_ENVIRONMENT_RETRY_S = 5.0


def _run_job_background(
    spec: JobSpec,
    runtime_secrets: dict[str, str] | None = None,
    reserved_claim: AttemptLaunchClaim | None = None,
) -> None:
    """run a supervised job without leaking a daemon-thread traceback."""
    import logging

    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import _update, get_status

    try:
        recovery = {"reserved_claim": reserved_claim} if reserved_claim is not None else {}
        if runtime_secrets is not None:
            _run_job(spec, runtime_secrets=runtime_secrets, **recovery)
        else:
            _run_job(spec, **recovery)
    except Exception as exc:
        detail = f"{type(exc).__name__}: background run failed"
        with contextlib.suppress(Exception):
            if get_status(spec.run_id).state not in TERMINAL_STATES:
                _update(spec.run_id, "failed", error=detail)
        logging.getLogger(__name__).warning(
            "background run %s ended in error: %s", spec.run_id, detail
        )


def _consume_reserved_claim(run_id: str, claim: AttemptLaunchClaim | None) -> None:
    """Release a reserved claim this process will not launch. Idempotent and never raises."""
    if claim is None:
        return
    from flash.runner.lifecycle.attempts import consume_active_launch_claim

    with contextlib.suppress(Exception):
        consume_active_launch_claim(run_id, claim)


def _run_job(
    spec: JobSpec,
    runtime_secrets: dict[str, str] | None = None,
    reserved_claim: AttemptLaunchClaim | None = None,
) -> None:
    from flash.runner.lifecycle.state import RUNS_DIR, TERMINAL_STATES
    from flash.runner.lifecycle.status import _update, get_status
    from flash.runner.supervise.lifecycle import _run_job_inner
    from flash.runner.supervise.recovery import _gc_run_sandboxes

    try:
        # Cancel can land before this thread starts; don't overwrite a terminal state with provisioning.
        if get_status(spec.run_id).state in TERMINAL_STATES:
            return
        _update(spec.run_id, "provisioning")
        log_path = os.path.join(RUNS_DIR, f"{spec.run_id}.log")
        while True:
            checkpoint()
            try:
                _run_job_inner(
                    spec,
                    log_path,
                    runtime_secrets=runtime_secrets,
                    reserved_claim=reserved_claim,
                )
                break
            except Exception as exc:
                from flash.envs.loading.staged import StagedEnvironmentTransientError
                from flash.runner.lifecycle.deadlines import _load_run_deadline_at

                if not isinstance(exc, StagedEnvironmentTransientError):
                    raise
                if get_status(spec.run_id).state in TERMINAL_STATES:
                    return
                remaining = _load_run_deadline_at(spec.run_id) - time.time()
                if remaining <= 0:
                    _update(
                        spec.run_id,
                        "failed",
                        error="RuntimeError: run wall deadline exhausted during environment staging",
                    )
                    raise RuntimeError(
                        "run wall deadline exhausted during environment staging"
                    ) from exc
                with open(log_path, "a") as log:
                    print(
                        "environment staging is temporarily unavailable; deferring before retry",
                        file=log,
                        flush=True,
                    )
                pause(min(_STAGED_ENVIRONMENT_RETRY_S, remaining))
    finally:
        # every exit before the supervised submit -- staging deadline exhaustion, a transient staging
        # failure that turned terminal, or any raise out of _run_job_inner -- leaves the reserved claim
        # unconsumed. consuming is idempotent, so the submission path's own release stays authoritative.
        _consume_reserved_claim(spec.run_id, reserved_claim)
        # gc registered endpoints because undeleted endpoints count against the account-wide worker quota.
        # skip when the run is still non-terminal: another live supervisor then owns the durable handle,
        # and reaping here would tear down its still-active provider resources.
        if get_status(spec.run_id).state in TERMINAL_STATES:
            _gc_run_sandboxes(spec)


def _spec_with_gpu(spec: JobSpec, gpu_type: str, gpu_count: int = 0) -> JobSpec:
    """The spec the workers/loggers see for THIS attempt's allocated class and card count.

    The allocator may satisfy the run with a multi-card combination of a smaller class than the spec
    named, so the count it CHOSE has to land on the spec too: the worker sizes its rank count from
    gpu.count, and the provider payload rents gpu.count cards. Letting those diverge would either
    strand rented cards or launch more ranks than were rented.
    """
    count = gpu_count if gpu_count >= 1 else gpu_count_of(spec)
    if spec.gpu.type == gpu_type and gpu_count_of(spec) == count:
        return spec
    d = spec.to_internal_dict()
    acceptable = spec.gpu.acceptable_types
    gpu = {**d["gpu"], "type": gpu_type, "count": count}
    if acceptable:
        gpu["type_fallbacks"] = tuple(name for name in acceptable if name != gpu_type)
    else:
        gpu.pop("type_fallbacks", None)
    d["gpu"] = gpu
    # the auto marker records that gpu.count was omitted and survives shape resolution.
    return JobSpec.from_dict(d)


def _run_attempts_supervised(
    spec: JobSpec,
    log,
    runtime_secrets: dict[str, str] | None = None,
    source_snapshot: dict | None = None,
    reserved_claim: AttemptLaunchClaim | None = None,
) -> dict:
    """Run one run's attempts with bounded auto-retry on infra-shaped failures.

    Retries resume from the latest HF checkpoint on a fresh host. Genuine worker errors fail fast.
    ``attempt_start`` offsets persisted identities without expanding this invocation's retry budget.
    """
    from flash.runner.supervise.execution import run_attempts_supervised

    return run_attempts_supervised(
        spec,
        log,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        reserved_claim=reserved_claim,
    )


def _terminal_failure_detail(exc: BaseException) -> str:
    """Render a run's persisted `error` from the exception that ended it.

    `RunStatus.error` is shown to the submitter, so an arbitrary exception's text is NOT safe to
    put there: it can carry internal storage paths, provider payloads, and upstream bodies. The
    """
    return f"{type(exc).__name__}: run failed"


def _run_job_inner(
    spec: JobSpec,
    log_path: str,
    runtime_secrets: dict[str, str] | None = None,
    reserved_claim: AttemptLaunchClaim | None = None,
) -> None:
    from flash.runner.accounting.artifacts import stage_environment_package
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.status import _update, get_status
    from flash.runner.supervise.errors import _RunCancelled
    from flash.runner.supervise.lifecycle import _run_training

    try:
        # dev replaced the explicit code upload with managed source snapshots, so staging only has
        # to pin the environment package before the provider is allocated. the staged package rides
        # into the persisted snapshot at the per-attempt persist in `_run_attempts_supervised`,
        # which already runs after this with the fully planned spec -- persisting a second time
        # here would
        # hash a half-planned spec no later integrity check can reproduce.
        deadline_at = _load_run_deadline_at(spec.run_id)
        spec = stage_environment_package(spec, deadline_at=deadline_at)
        with open(log_path, "a") as log:
            _run_training(
                spec,
                log,
                prior_cost=0.0,
                runtime_secrets=runtime_secrets,
                reserved_claim=reserved_claim,
            )
    except _RunCancelled:
        return  # cancel_run already set the terminal state
    except Exception as exc:
        from flash.envs.loading.staged import StagedEnvironmentTransientError

        if isinstance(exc, StagedEnvironmentTransientError):
            raise
        if get_status(spec.run_id).state != "cancelled":
            _update(spec.run_id, "failed", error=_terminal_failure_detail(exc))
        raise


def _run_training(
    spec: JobSpec,
    log,
    *,
    prior_cost: float,
    runtime_secrets: dict[str, str] | None = None,
    source_snapshot: dict | None = None,
    reserved_claim: AttemptLaunchClaim | None = None,
) -> None:
    """Train one adapter while preserving run-global attempt authorization and cost."""
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import (
        _update,
        get_status,
        source_snapshot_from_status,
    )
    from flash.runner.supervise.errors import _RunCancelled
    from flash.runner.supervise.lifecycle import _run_attempts_supervised

    status = get_status(spec.run_id)
    if source_snapshot is None:
        source_snapshot = source_snapshot_from_status(status, required=True)
    # defense in depth against the recovery toctou (see attach_run): a run can be flipped into any
    # terminal state — not just `cancelled` — by a concurrent thread/process between the resume
    # decision and here. Bail before _update + the supervised submit so we never submit PAID GPU
    # work for an already-terminal run. _RunCancelled is the terminal signal; callers swallow it.
    if get_status(spec.run_id).state in TERMINAL_STATES:
        raise _RunCancelled(f"run {spec.run_id} is already terminal; not submitting")
    # The pre-check above closes most of the window, but a concurrent flip can still land between
    # it and this transition. _update is a compare-and-set: it returns False when the run is already
    # terminal and leaves the state untouched. Gate the PAID supervised submit on that result so a
    # run cancelled in this last instant is never resumed onto a GPU.
    if not _update(spec.run_id, "running"):
        raise _RunCancelled(f"run {spec.run_id} went terminal before submit; not submitting")
    print(
        f"starting phase={spec.phase} model={spec.model} gpu={spec.gpu.type}",
        file=log,
        flush=True,
    )
    metrics = _run_attempts_supervised(
        spec,
        log,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        reserved_claim=reserved_claim,
    )
    current = get_status(spec.run_id)
    expected_remote = current.remote or current.cleanup_confirmed_remote
    if not attempt_recovery._adopt_completed_attempt(
        spec.run_id, spec, expected_remote, metrics, log=log
    ):
        raise _RunCancelled(f"run {spec.run_id} lost completion ownership")
    print(f"done: train_wall={metrics.get('wall_seconds')}", file=log, flush=True)
