"""poll instance resource state and fenced immutable attempt artifacts."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from flash.providers._lifecycle.instances.poll import PollErrorTracker, make_say
from flash.providers.artifacts.attempts import (
    AttemptArtifactError,
    persist_attempt_artifacts,
    poll_result_from_manifest,
    read_attempt_artifacts,
)
from flash.providers.core.base import PollResult
from flash.runner.lifecycle.deadlines import _RESULT_VISIBILITY_ALLOWANCE_S
from flash.runner.lifecycle.protocol import AttemptRecord


@dataclass
class InstancePollAdapter:
    """provider-specific resource observation seams for one fenced instance attempt."""

    provider: str
    instance_id: object
    run_id: str
    current_attempt: int
    fence: int
    launch_ts: float
    hf_repo: str
    phase: str
    source_snapshot: dict
    fetch_instance: Callable[[], dict | None]
    poll_error_exceptions: tuple
    status_field: str
    running_status: str
    dead_states: frozenset
    missing_dead_threshold: int
    stamp_cost_and_notes: Callable[..., None]
    record_resource_loss: Callable[[str], None] | None = None


def _current_attempt(adapter: InstancePollAdapter) -> AttemptRecord:
    from flash.runner.lifecycle.status import get_status

    attempt = AttemptRecord.from_dict(get_status(adapter.run_id).attempt)
    if attempt.attempt_id != adapter.current_attempt or attempt.fence != adapter.fence:
        raise RuntimeError("instance handle does not match the current fenced attempt")
    return attempt


def _record_resource(adapter: InstancePollAdapter, status: str, *, transport: str = "ok") -> None:
    from flash.runner.accounting.reconciliation import _remote_resource_identity
    from flash.runner.lifecycle.status import get_status, record_resource

    current = get_status(adapter.run_id)
    normalized = (
        "running"
        if status == adapter.running_status
        else "terminal"
        if status in adapter.dead_states or status == "missing"
        else "provisioning"
    )
    remote = current.remote if isinstance(current.remote, dict) else {}
    record_resource(
        adapter.run_id,
        {
            "attempt_id": adapter.current_attempt,
            "fence": adapter.fence,
            "provider": adapter.provider,
            "state": normalized,
            "provider_state": status,
            "transport": transport,
            "observed_at": time.time(),
            "instance_id": adapter.instance_id,
        },
        attempt_id=adapter.current_attempt,
        fence=adapter.fence,
        resource_identity=_remote_resource_identity(remote),
    )


def _observe_result(adapter: InstancePollAdapter) -> PollResult | None:
    artifacts = read_attempt_artifacts(
        adapter.hf_repo,
        phase=adapter.phase,
        run_id=adapter.run_id,
        attempt_id=adapter.current_attempt,
        fence=adapter.fence,
        source_snapshot=adapter.source_snapshot,
    )
    persist_attempt_artifacts(adapter.run_id, artifacts)
    if artifacts.result is None:
        return None
    result = poll_result_from_manifest(artifacts.result)
    if result.ok and result.metrics is not None:
        adapter.stamp_cost_and_notes(
            result.metrics,
            end_ts=float(artifacts.result["finished_at"]),
            launch_ts=adapter.launch_ts,
            observed_at=float(artifacts.observed_at),
        )
    return result


def _missing_result(adapter: InstancePollAdapter, terminal_status: str | None) -> PollResult:
    if terminal_status is not None and adapter.record_resource_loss is not None:
        adapter.record_resource_loss(terminal_status)
    state_detail = (
        f"resource ended with {terminal_status}"
        if terminal_status is not None
        else "work deadline expired"
    )
    return PollResult(
        False,
        failure="job_preempted",
        detail=f"{adapter.provider} {state_detail} without a result manifest",
    )


def poll_instance_job(
    adapter: InstancePollAdapter,
    *,
    log=None,
    interval_s: float = 15.0,
    deadline_at: float | None = None,
    **_ignored,
) -> PollResult:
    """poll provider resource state without deriving lifecycle from progress age."""
    del deadline_at
    attempt = _current_attempt(adapter)
    say = make_say(log)
    poll_errors = PollErrorTracker(say, interval_s)
    last_status: str | None = None
    terminal_status: str | None = None
    terminal_result_deadline_at: float | None = None
    terminal_final_probe_pending = False
    missing_streak = 0
    while True:
        if terminal_result_deadline_at is not None:
            if terminal_final_probe_pending:
                try:
                    result = _observe_result(adapter)
                except AttemptArtifactError as exc:
                    return PollResult(False, failure="job_failed", detail=str(exc))
                except Exception:
                    result = None
                return result if result is not None else _missing_result(adapter, terminal_status)
            if time.time() >= terminal_result_deadline_at:
                return _missing_result(adapter, terminal_status)
            try:
                result = _observe_result(adapter)
            except AttemptArtifactError as exc:
                return PollResult(False, failure="job_failed", detail=str(exc))
            except Exception:
                result = None
            observed_at = time.time()
            if observed_at >= terminal_result_deadline_at:
                return _missing_result(adapter, terminal_status)
            if result is not None:
                return result
            remaining = terminal_result_deadline_at - observed_at
            if interval_s >= remaining:
                terminal_final_probe_pending = True
                time.sleep(remaining)
            elif interval_s > 0:
                time.sleep(interval_s)
            continue
        try:
            result = _observe_result(adapter)
        except AttemptArtifactError as exc:
            return PollResult(False, failure="job_failed", detail=str(exc))
        except Exception:
            result = None
        if result is not None:
            return result
        now = time.time()
        if now >= attempt.result_deadline_at:
            return _missing_result(adapter, None)
        if now >= attempt.work_deadline_at:
            delay = min(interval_s, attempt.result_deadline_at - now)
            if delay > 0:
                time.sleep(delay)
            continue
        try:
            instance = adapter.fetch_instance()
            poll_errors.reset()
        except adapter.poll_error_exceptions as exc:
            _record_resource(adapter, last_status or "unknown", transport="unavailable")
            if poll_errors.record(exc, deadline_at=attempt.work_deadline_at):
                return PollResult(
                    False, failure="poll_error", detail="provider status transport failed"
                )
            delay = min(interval_s, max(0.0, attempt.work_deadline_at - time.time()))
            if delay > 0:
                time.sleep(delay)
            continue
        missing_streak = missing_streak + 1 if instance is None else 0
        status = str(
            (instance or {}).get(adapter.status_field)
            or ("missing" if instance is None else "unknown")
        )
        _record_resource(adapter, status)
        if status != last_status:
            say(f"instance {adapter.instance_id}: {status}")
            last_status = status
        if status in adapter.dead_states or missing_streak >= adapter.missing_dead_threshold:
            terminal_status = status
            terminal_observed_at = time.time()
            terminal_result_deadline_at = min(
                attempt.result_deadline_at,
                terminal_observed_at + _RESULT_VISIBILITY_ALLOWANCE_S,
            )
            remaining = max(0.0, terminal_result_deadline_at - terminal_observed_at)
            if interval_s >= remaining and remaining > 0:
                terminal_final_probe_pending = True
                time.sleep(remaining)
            elif remaining > 0 and interval_s > 0:
                time.sleep(interval_s)
            continue
        now = time.time()
        if status != adapter.running_status and now >= attempt.grant_deadline_at:
            return PollResult(
                False,
                failure="job_preempted",
                detail=f"{adapter.provider} resource did not become active before its grant deadline",
            )
        delay = min(interval_s, max(0.0, attempt.work_deadline_at - now))
        if delay > 0:
            time.sleep(delay)
