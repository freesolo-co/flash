"""shared retry disposition for confirmed-absent handleless attempts."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class HandlelessResubmit:
    attempt_id: int | None = None
    fence: int | None = None
    attempt_state: str | None = None
    attempt_start: int = 0
    retry_counters: dict | None = None
    oom_vram_floor: float = 0.0


class ResubmitLaunch:
    """hold one cross-process launch lease until its replacement supervisor exits."""

    def __init__(self, run_id: str) -> None:
        from flash.server.platform.locks import _resubmit_lock

        self._lock = _resubmit_lock(run_id)
        self._started = False

    def acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def start(self, target) -> None:
        def run() -> None:
            try:
                target()
            finally:
                self._lock.release()

        thread = threading.Thread(target=run, daemon=True)
        self._started = True
        try:
            thread.start()
        except BaseException:
            self._started = False
            raise

    def release_unstarted(self) -> None:
        if not self._started:
            self._lock.release()


def _attempt_identity_kwargs(attempt: object) -> dict[str, int]:
    from flash.runner.lifecycle.protocol import AttemptRecord

    if attempt is None:
        return {}
    record = attempt if isinstance(attempt, AttemptRecord) else AttemptRecord.from_dict(attempt)
    return {"expected_attempt_id": record.attempt_id, "expected_fence": record.fence}


def _oom_floor(status) -> float:
    from flash.runner.lifecycle.status import effective_spec_from_status
    from flash.runner.supervise.attach_replacement import oom_floor_from_effective_spec

    try:
        return oom_floor_from_effective_spec(effective_spec_from_status(status))
    except Exception:
        return 0.0


def failure_retry(spec, status, result) -> HandlelessResubmit | None:
    """consume one shared retry only after the current worker is confirmed absent."""
    from flash.runner.supervise.lifecycle import (
        _failure_disposition,
        _reconstructed_retry_budget,
    )

    budget = _reconstructed_retry_budget(
        int(spec.gpu.max_retries),
        counters=status.retry_counters,
    )
    oom_vram_floor = _oom_floor(status) if result.failure == "oom" else 0.0
    if result.failure == "oom" and oom_vram_floor <= 0:
        return None
    disposition = _failure_disposition(
        budget,
        result.failure,
        allow_retry=result.failure != "oom" or oom_vram_floor > 0,
    )
    if not disposition.retry:
        return None
    attempt = status.attempt if isinstance(status.attempt, dict) else {}
    attempt_id = attempt.get("attempt_id")
    fence = attempt.get("fence")
    if (
        isinstance(attempt_id, bool)
        or not isinstance(attempt_id, int)
        or attempt_id < 0
        or isinstance(fence, bool)
        or not isinstance(fence, int)
        or fence < 1
    ):
        return None
    return HandlelessResubmit(
        attempt_id=attempt_id,
        fence=fence,
        attempt_state=attempt.get("state"),
        attempt_start=attempt_id + 1,
        retry_counters=budget.counters(),
        oom_vram_floor=oom_vram_floor,
    )


def claimed_retry(status, result) -> HandlelessResubmit | None:
    """resume one already-consumed replacement claim without spending its budget twice."""
    from flash.runner.supervise.lifecycle import _retry_category

    if _retry_category(result.failure) is None:
        return None
    attempt = status.attempt if isinstance(status.attempt, dict) else {}
    if attempt.get("state") != "settling":
        return None
    attempt_id = attempt.get("attempt_id")
    fence = attempt.get("fence")
    if (
        isinstance(attempt_id, bool)
        or not isinstance(attempt_id, int)
        or attempt_id < 0
        or isinstance(fence, bool)
        or not isinstance(fence, int)
        or fence < 1
    ):
        return None
    oom_vram_floor = _oom_floor(status) if result.failure == "oom" else 0.0
    if result.failure == "oom" and oom_vram_floor <= 0:
        return None
    return HandlelessResubmit(
        attempt_id=attempt_id,
        fence=fence,
        attempt_state=attempt.get("state"),
        attempt_start=attempt_id + 1,
        oom_vram_floor=oom_vram_floor,
    )
