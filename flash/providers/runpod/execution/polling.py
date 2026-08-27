"""poll RunPod resource state and fenced immutable attempt artifacts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from flash.providers._lifecycle.instances.poll import PollErrorTracker, make_say
from flash.providers.artifacts.attempts import (
    AttemptArtifactError,
    persist_attempt_artifacts,
    poll_result_from_manifest,
    read_attempt_artifacts,
)
from flash.providers.core.base import PollResult
from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.execution.jobs import (
    TERMINAL_FAIL,
    TERMINAL_OK,
    GraceTimer,
    capacity_escalation_note,
)
from flash.runner.lifecycle.deadlines import RESULT_VISIBILITY_ALLOWANCE_S
from flash.runner.lifecycle.protocol import AttemptRecord

if TYPE_CHECKING:
    from collections.abc import Callable

_GRANT_PROVING_STATUSES = frozenset(
    {"IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
)


@dataclass(frozen=True)
class _PollContext:
    handle: Any
    spec: Any
    say: Callable[[str], None]
    interval_s: float
    attempt: AttemptRecord
    source_snapshot: dict
    on_last_gpu: bool
    queue_grace_s: float
    unhealthy_grace_s: float
    throttled_grace_s: float


@dataclass
class _PollState:
    last_status: str | None
    granted: bool
    queued: GraceTimer
    unhealthy: GraceTimer
    throttled: GraceTimer


def _current_attempt(run_id: str, handle) -> tuple[AttemptRecord, dict]:
    from flash.runner.lifecycle.status import get_status, source_snapshot_from_status

    status = get_status(run_id)
    attempt = AttemptRecord.from_dict(status.attempt)
    if attempt.attempt_id != handle.attempt or attempt.fence != handle.fence:
        raise RuntimeError("RunPod handle does not match the current fenced attempt")
    return attempt, source_snapshot_from_status(status, required=True)


def _record_resource(context: _PollContext, status: str, *, transport: str = "ok") -> None:
    from flash.runner.accounting.reconciliation import _remote_resource_identity
    from flash.runner.lifecycle.status import record_resource

    normalized = (
        "queued"
        if status == "IN_QUEUE"
        else "running"
        if status == "IN_PROGRESS"
        else "terminal"
        if status in TERMINAL_OK | TERMINAL_FAIL
        else "unknown"
    )
    record_resource(
        context.spec.run_id,
        {
            "attempt_id": context.handle.attempt,
            "fence": context.handle.fence,
            "provider": "runpod",
            "state": normalized,
            "provider_state": status,
            "transport": transport,
            "observed_at": time.time(),
            "endpoint_id": context.handle.endpoint_id,
            "job_id": context.handle.job_id,
        },
        attempt_id=context.handle.attempt,
        fence=context.handle.fence,
        resource_identity=_remote_resource_identity(context.handle.to_dict()),
    )


def _observe_artifacts(context: _PollContext) -> PollResult | None:
    artifacts = read_attempt_artifacts(
        context.spec.train.hf_repo,
        phase=context.spec.phase,
        run_id=context.spec.run_id,
        attempt_id=context.handle.attempt,
        fence=context.handle.fence,
        source_snapshot=context.source_snapshot,
    )
    persist_attempt_artifacts(context.spec.run_id, artifacts)
    return poll_result_from_manifest(artifacts.result) if artifacts.result is not None else None


def _queued_too_long(context: _PollContext, state: _PollState, now: float) -> PollResult | None:
    if not state.queued.expired(True, now, context.queue_grace_s):
        return None
    return PollResult(
        False,
        failure="no_capacity",
        detail=(
            f"never scheduled: job remained IN_QUEUE for {int(now - state.queued.since)}s; "
            f"{capacity_escalation_note(context.on_last_gpu)}"
        ),
    )


def _queue_failure(
    context: _PollContext, state: _PollState, status: str, now: float
) -> PollResult | None:
    if status != "IN_QUEUE" or state.granted:
        state.queued.expired(False, now, context.queue_grace_s)
        state.unhealthy.expired(False, now, context.unhealthy_grace_s)
        state.throttled.expired(False, now, context.throttled_grace_s)
        return None
    if now >= context.attempt.grant_deadline_at:
        unhealthy_since = state.unhealthy.since
        if (
            unhealthy_since is not None
            and unhealthy_since + context.unhealthy_grace_s < context.attempt.grant_deadline_at
            and now - unhealthy_since > context.unhealthy_grace_s
        ):
            return PollResult(
                False,
                failure="job_preempted",
                detail="RunPod worker remained unhealthy",
            )
        return PollResult(
            False,
            failure="no_capacity",
            detail=(
                "never scheduled: job remained IN_QUEUE until its immutable grant deadline; "
                f"{capacity_escalation_note(context.on_last_gpu)}"
            ),
        )
    try:
        health = runpod_api.endpoint_health_for_fingerprint(
            context.handle.endpoint_id,
            context.handle.key_fingerprint,
            # bound the probe by the same immutable deadline that decides no_capacity above. an
            # unbounded probe can burn its full retry ladder past the grant deadline, delaying the
            # capacity verdict, the retry, and endpoint cleanup by minutes.
            deadline_at=context.attempt.grant_deadline_at,
        )
    except Exception:
        return _queued_too_long(context, state, now)
    workers = health.get("workers") or {}
    usable = workers.get("running") or workers.get("ready") or workers.get("idle")
    initializing = workers.get("initializing")
    if usable or initializing:
        state.granted = True
        return None
    unhealthy = bool(workers.get("unhealthy"))
    if state.unhealthy.expired(unhealthy, now, context.unhealthy_grace_s):
        return PollResult(False, failure="job_preempted", detail="RunPod worker remained unhealthy")
    if unhealthy:
        state.queued.expired(False, now, context.queue_grace_s)
        state.throttled.expired(False, now, context.throttled_grace_s)
        return None
    if state.throttled.expired(bool(workers.get("throttled")), now, context.throttled_grace_s):
        return PollResult(False, failure="no_capacity", detail="RunPod worker remained throttled")
    return _queued_too_long(context, state, now)


def poll_job(
    handle,
    spec,
    log=None,
    interval_s: float = 10.0,
    unhealthy_grace_s: float = 240.0,
    throttled_grace_s: float = 300.0,
    queue_grace_s: float = 300.0,
    deadline_at: float | None = None,
    on_last_gpu: bool = False,
    **_ignored,
) -> PollResult:
    """poll resource state without treating sparse progress as failure authority."""
    del deadline_at
    if not handle.job_id:
        raise ValueError("endpoint-only RunPod handles cannot be polled")
    attempt, source_snapshot = _current_attempt(spec.run_id, handle)
    context = _PollContext(
        handle=handle,
        spec=spec,
        say=make_say(log),
        interval_s=interval_s,
        attempt=attempt,
        source_snapshot=source_snapshot,
        on_last_gpu=on_last_gpu,
        queue_grace_s=min(queue_grace_s, max(0.0, attempt.grant_deadline_at - time.time())),
        unhealthy_grace_s=unhealthy_grace_s,
        throttled_grace_s=throttled_grace_s,
    )
    state = _PollState(None, False, GraceTimer(), GraceTimer(), GraceTimer())
    poll_errors = PollErrorTracker(context.say, interval_s)
    terminal_status: str | None = None
    terminal_result_deadline_at: float | None = None
    artifact_error: str | None = None
    while True:
        now = time.time()
        try:
            result = _observe_artifacts(context)
            artifact_error = None
        except AttemptArtifactError as exc:
            return PollResult(False, failure="job_failed", detail=str(exc))
        except Exception as exc:
            result = None
            artifact_error = type(exc).__name__
        if result is not None:
            return result
        result_deadline = terminal_result_deadline_at or attempt.result_deadline_at
        if now >= result_deadline:
            state_detail = (
                f"resource ended with {terminal_status}"
                if terminal_status is not None
                else "work deadline expired"
            )
            return PollResult(
                False,
                failure="job_preempted",
                detail=(
                    f"RunPod {state_detail} without a result manifest"
                    + (f"; artifact read failed with {artifact_error}" if artifact_error else "")
                ),
            )
        if now >= attempt.work_deadline_at:
            delay = min(interval_s, max(0.0, result_deadline - time.time()))
            if delay > 0:
                time.sleep(delay)
            continue
        try:
            provider_status = runpod_api.job_status(
                handle.endpoint_id,
                handle.job_id,
                key_fingerprint=handle.key_fingerprint,
                deadline_at=attempt.work_deadline_at,
            )
            poll_errors.reset()
            status = str(provider_status.get("status") or "UNKNOWN")
            _record_resource(context, status)
        except runpod_api.RunpodApiError as exc:
            _record_resource(context, state.last_status or "UNKNOWN", transport="unavailable")
            if poll_errors.record(exc, deadline_at=attempt.work_deadline_at):
                return PollResult(
                    False, failure="poll_error", detail="RunPod status transport failed"
                )
            time.sleep(min(interval_s, max(0.0, attempt.work_deadline_at - time.time())))
            continue
        if status != state.last_status:
            context.say(f"job {handle.job_id}: {status}")
            state.last_status = status
        if status in _GRANT_PROVING_STATUSES:
            state.granted = True
        if status in TERMINAL_OK | TERMINAL_FAIL:
            if terminal_status is None:
                terminal_result_deadline_at = min(
                    attempt.result_deadline_at,
                    now + RESULT_VISIBILITY_ALLOWANCE_S,
                )
            terminal_status = status
        failure = _queue_failure(context, state, status, now)
        if failure is not None:
            return failure
        sleep_until = (
            terminal_result_deadline_at or attempt.result_deadline_at
            if terminal_status
            else attempt.work_deadline_at
        )
        delay = min(interval_s, max(0.0, sleep_until - time.time()))
        if delay > 0:
            time.sleep(delay)
