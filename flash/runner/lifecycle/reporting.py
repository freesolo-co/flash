"""ordered delivery of run status reports."""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

from flash.runner.lifecycle.state import RunStatus

_STATUS_LOCK = threading.Lock()
_STATUS_REPORT_LOCK = threading.RLock()
_STATUS_REPORT_CONDITION = threading.Condition(_STATUS_REPORT_LOCK)
_STATUS_REPORT_EXECUTOR: ThreadPoolExecutor | None = None
_STATUS_REPORT_QUEUES: dict[str, deque[tuple[RunStatus, int, threading.Event, int]]] = {}
_STATUS_REPORT_WORKERS: dict[str, Future[None]] = {}
_STATUS_REPORT_ACTIVE: set[str] = set()
_STATUS_REPORT_DRAINING: set[str] = set()
_STATUS_REPORT_PENDING = 0
_STATUS_REPORT_ACCEPTING = True
_STATUS_REPORT_LAST_SENT: dict[str, int] = {}
_STATUS_REPORT_LAST_ATTEMPTED: dict[str, int] = {}
_STATUS_REPORT_LAST_QUEUED: dict[str, int] = {}


def _send_status_report(status: RunStatus) -> bool:
    from flash.server.domain.registry.runs import record_training_run

    return record_training_run(status=status)


def _valid_status_report_sequence(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _status_report_sequence_unlocked(status: RunStatus) -> int:
    run_id = status.run_id
    persisted = _valid_status_report_sequence(getattr(status, "report_sequence", 0))
    if persisted > 0:
        return persisted
    return _STATUS_REPORT_LAST_QUEUED.get(run_id, 0) + 1


def _deliver_status_report(status: RunStatus, sequence: int, attempt_budget: int) -> bool:
    with _STATUS_REPORT_LOCK:
        if sequence <= _STATUS_REPORT_LAST_SENT.get(status.run_id, 0):
            return True
        if sequence < _STATUS_REPORT_LAST_ATTEMPTED.get(status.run_id, 0):
            return True
        _STATUS_REPORT_LAST_ATTEMPTED[status.run_id] = sequence
    for attempt in range(attempt_budget):
        try:
            if _send_status_report(status) is not False:
                with _STATUS_REPORT_LOCK:
                    _STATUS_REPORT_LAST_SENT[status.run_id] = sequence
                return True
        except Exception:
            pass
        if attempt + 1 < attempt_budget:
            with _STATUS_REPORT_LOCK:
                if not _STATUS_REPORT_ACCEPTING:
                    return False
    return False


def _finish_status_report(done: threading.Event) -> None:
    global _STATUS_REPORT_PENDING
    with _STATUS_REPORT_CONDITION:
        _STATUS_REPORT_PENDING -= 1
        done.set()
        _STATUS_REPORT_CONDITION.notify_all()


def _drain_status_report_run(run_id: str) -> None:
    with _STATUS_REPORT_CONDITION:
        if run_id in _STATUS_REPORT_DRAINING:
            return
        _STATUS_REPORT_DRAINING.add(run_id)
    while True:
        with _STATUS_REPORT_CONDITION:
            queue = _STATUS_REPORT_QUEUES.get(run_id)
            if not queue:
                _STATUS_REPORT_QUEUES.pop(run_id, None)
                _STATUS_REPORT_ACTIVE.discard(run_id)
                _STATUS_REPORT_DRAINING.discard(run_id)
                _STATUS_REPORT_CONDITION.notify_all()
                return
            status, sequence, done, attempt_budget = queue.popleft()
        delivered = False
        try:
            delivered = _deliver_status_report(status, sequence, attempt_budget)
        finally:
            if not delivered:
                with _STATUS_REPORT_CONDITION:
                    if _STATUS_REPORT_LAST_QUEUED.get(run_id) == sequence:
                        sent = _STATUS_REPORT_LAST_SENT.get(run_id, 0)
                        if sent:
                            _STATUS_REPORT_LAST_QUEUED[run_id] = sent
                        else:
                            _STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
            _finish_status_report(done)


def _discard_status_report_worker(run_id: str, future: Future[None]) -> None:
    fallback = False
    with _STATUS_REPORT_CONDITION:
        if _STATUS_REPORT_WORKERS.get(run_id) is not future:
            return
        _STATUS_REPORT_WORKERS.pop(run_id, None)
        if run_id not in _STATUS_REPORT_DRAINING:
            _STATUS_REPORT_ACTIVE.discard(run_id)
            if _STATUS_REPORT_QUEUES.get(run_id):
                fallback = not _start_status_report_worker_unlocked(run_id)
        _STATUS_REPORT_CONDITION.notify_all()
    if fallback:
        _drain_status_report_run(run_id)


def _status_report_executor_unlocked() -> ThreadPoolExecutor:
    global _STATUS_REPORT_EXECUTOR
    if _STATUS_REPORT_EXECUTOR is None:
        _STATUS_REPORT_EXECUTOR = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="flash-status-reporter",
        )
    return _STATUS_REPORT_EXECUTOR


def _start_status_report_worker_unlocked(run_id: str) -> bool:
    global _STATUS_REPORT_EXECUTOR
    _STATUS_REPORT_ACTIVE.add(run_id)
    executor = _status_report_executor_unlocked()
    try:
        future = executor.submit(_drain_status_report_run, run_id)
    except RuntimeError:
        can_drain_queued_work = (
            not getattr(executor, "_shutdown", False)
            and not getattr(executor, "_broken", False)
            and any(thread.is_alive() for thread in getattr(executor, "_threads", ()))
        )
        if can_drain_queued_work:
            return True
        if _STATUS_REPORT_EXECUTOR is executor:
            _STATUS_REPORT_EXECUTOR = None
        return False
    _STATUS_REPORT_WORKERS[run_id] = future
    future.add_done_callback(
        lambda completed, current_run_id=run_id: _discard_status_report_worker(
            current_run_id, completed
        )
    )
    return True


def _cancel_status_report_run_unlocked(run_id: str, *, forget_sequence: bool) -> None:
    global _STATUS_REPORT_PENDING
    queue = _STATUS_REPORT_QUEUES.pop(run_id, None)
    if queue is not None:
        while queue:
            _, _, done, _attempt_budget = queue.popleft()
            _STATUS_REPORT_PENDING -= 1
            done.set()
    worker = _STATUS_REPORT_WORKERS.get(run_id)
    if (worker is None or worker.done()) and run_id not in _STATUS_REPORT_DRAINING:
        _STATUS_REPORT_ACTIVE.discard(run_id)
    if forget_sequence:
        attempted = _STATUS_REPORT_LAST_ATTEMPTED.get(run_id, 0)
        if attempted:
            _STATUS_REPORT_LAST_QUEUED[run_id] = attempted
        else:
            _STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    _STATUS_REPORT_CONDITION.notify_all()


def _open_status_reporter() -> None:
    global _STATUS_REPORT_ACCEPTING
    with _STATUS_REPORT_CONDITION:
        _STATUS_REPORT_ACCEPTING = True


def _queue_status_report(status: RunStatus, *, wait: bool) -> None:
    global _STATUS_REPORT_PENDING
    snapshot = copy.deepcopy(status)
    run_id = snapshot.run_id
    done = threading.Event()
    fallback = False
    attempt_budget = 1 if wait else 2
    with _STATUS_REPORT_CONDITION:
        if not _STATUS_REPORT_ACCEPTING:
            return
        sequence = _status_report_sequence_unlocked(snapshot)
        while (
            wait
            and sequence == _STATUS_REPORT_LAST_QUEUED.get(run_id, 0)
            and sequence > _STATUS_REPORT_LAST_SENT.get(run_id, 0)
        ):
            _STATUS_REPORT_CONDITION.wait()
            if not _STATUS_REPORT_ACCEPTING:
                return
        if sequence <= _STATUS_REPORT_LAST_QUEUED.get(run_id, 0):
            return
        _STATUS_REPORT_LAST_QUEUED[run_id] = sequence
        _STATUS_REPORT_QUEUES.setdefault(run_id, deque()).append(
            (snapshot, sequence, done, attempt_budget)
        )
        _STATUS_REPORT_PENDING += 1
        if run_id not in _STATUS_REPORT_ACTIVE:
            fallback = not _start_status_report_worker_unlocked(run_id)
        if fallback and not wait:
            _cancel_status_report_run_unlocked(run_id, forget_sequence=True)
    if fallback and wait:
        _drain_status_report_run(run_id)
    if wait:
        done.wait()


def _report_status(status: RunStatus) -> None:
    _queue_status_report(status, wait=True)


def _report_status_async(status: RunStatus) -> None:
    _queue_status_report(status, wait=False)


def _wait_for_status_reports(timeout: float | None = None) -> bool:
    deadline = None if timeout is None else time.monotonic() + timeout
    with _STATUS_REPORT_CONDITION:
        while _STATUS_REPORT_PENDING:
            if deadline is None:
                _STATUS_REPORT_CONDITION.wait()
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _STATUS_REPORT_CONDITION.wait(remaining)
        return True


def _cancel_queued_status_reports_unlocked(*, forget_sequences: bool) -> None:
    for run_id in tuple(_STATUS_REPORT_QUEUES):
        _cancel_status_report_run_unlocked(run_id, forget_sequence=forget_sequences)


def _shutdown_status_reporter(timeout: float = 15.0, *, close: bool = False) -> bool:
    global _STATUS_REPORT_ACCEPTING, _STATUS_REPORT_EXECUTOR
    with _STATUS_REPORT_CONDITION:
        _STATUS_REPORT_ACCEPTING = not close
    flushed = _wait_for_status_reports(timeout)
    with _STATUS_REPORT_CONDITION:
        executor = _STATUS_REPORT_EXECUTOR
        _STATUS_REPORT_EXECUTOR = None
        if not flushed:
            _cancel_queued_status_reports_unlocked(forget_sequences=close)
            for future in tuple(_STATUS_REPORT_WORKERS.values()):
                future.cancel()
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=not flushed)
    return flushed
