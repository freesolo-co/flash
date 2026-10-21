"""Bounded background tasks, scheduled retries, and cooperative process shutdown.

The application owns a Supervisor. Its task-local binding lets nested runner/provider code
observe cancellation and dispatch follow-up work without a process-global executor.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
import time
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)
_current: contextvars.ContextVar[Supervisor | None] = contextvars.ContextVar(
    "supervisor", default=None
)
_stop: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "task_stop", default=None
)


class TaskAdmissionError(RuntimeError):
    """The owner is closing or has no capacity for another outstanding task."""


class TaskStopped(BaseException):
    """Detach local supervision without treating application shutdown as run failure."""


@dataclass(frozen=True)
class Retry:
    delay: float


@dataclass(frozen=True)
class Limits:
    active: int
    outstanding: int

    def __post_init__(self) -> None:
        if self.active < 1 or self.outstanding < self.active:
            raise ValueError("task limits require outstanding >= active >= 1")


@dataclass
class _Task:
    lane: str
    target: Callable[[], object] | None
    due: float
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


def current_supervisor() -> Supervisor:
    owner = _current.get()
    if owner is None:
        raise TaskAdmissionError("background work requires an application-owned supervisor")
    return owner


def checkpoint() -> None:
    stop = _stop.get()
    if stop is not None and stop.is_set():
        raise TaskStopped


def pause(seconds: float) -> None:
    """Interruptible wait for active monitors; scheduled retries should return Retry instead."""
    stop = _stop.get()
    if stop is None:
        time.sleep(seconds)
    elif stop.wait(seconds):
        raise TaskStopped


class Supervisor:
    def __init__(
        self,
        limits: dict[str, Limits] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._limits = (
            dict(limits)
            if limits is not None
            else {
                "training": Limits(64, 192),
                "recovery": Limits(4, 132),
                "cleanup": Limits(4, 132),
            }
        )
        self._condition = threading.Condition()
        self._tasks: dict[Hashable, _Task] = {}
        self._accepting = True
        self._scheduler: threading.Thread | None = None

    @contextlib.contextmanager
    def bind(self) -> Iterator[Supervisor]:
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def contains(self, key: Hashable) -> bool:
        with self._condition:
            return key in self._tasks

    def submit(
        self, key: Hashable, lane: str, target: Callable[[], object], *, delay: float = 0.0
    ) -> bool:
        """Admit once per key; delayed and running tasks share the outstanding-work bound."""
        with self._condition:
            if not self._accepting:
                raise TaskAdmissionError("background tasks are shutting down")
            if key in self._tasks:
                return False
            with self.reserve(key, lane, delay=delay) as dispatch:
                dispatch(target)
            return True

    @contextlib.contextmanager
    def reserve(
        self, key: Hashable, lane: str, *, delay: float = 0.0
    ) -> Iterator[Callable[[Callable[[], object]], None]]:
        """Hold bounded admission while a submitter persists inputs, before dispatch is possible."""
        with self._condition:
            if not self._accepting:
                raise TaskAdmissionError("background tasks are shutting down")
            if key in self._tasks:
                raise TaskAdmissionError("background task is already admitted")
            limit = self._limits[lane]
            if sum(t.lane == lane for t in self._tasks.values()) >= limit.outstanding:
                raise TaskAdmissionError(f"{lane} background task capacity reached")
            task = _Task(lane, None, 0.0)
            self._tasks[key] = task

        def dispatch(target: Callable[[], object]) -> None:
            with self._condition:
                if not self._accepting or self._tasks.get(key) is not task:
                    raise TaskAdmissionError("background tasks are shutting down")
                task.target = target
                task.due = self._clock() + max(0.0, delay)
                try:
                    if self._scheduler is None:
                        scheduler = threading.Thread(
                            target=self._dispatch, name="flash-tasks", daemon=True
                        )
                        scheduler.start()
                        self._scheduler = scheduler
                except Exception:
                    task.target = None
                    raise
                self._condition.notify_all()

        try:
            yield dispatch
        finally:
            with self._condition:
                if task.target is None and self._tasks.get(key) is task:
                    del self._tasks[key]
                self._condition.notify_all()

    def cancel(self, key: Hashable) -> bool:
        """Cancel queued work or signal a running task at its next cooperative boundary."""
        with self._condition:
            task = self._tasks.get(key)
            if task is None:
                return False
            task.stop.set()
            if task.thread is None:
                del self._tasks[key]
            self._condition.notify_all()
            return True

    def close(self, timeout: float = 0.0) -> bool:
        """Stop dispatch, discard pending work, signal active tasks, and wait within one budget."""
        deadline = self._clock() + timeout
        with self._condition:
            self._accepting = False
            for key, task in list(self._tasks.items()):
                task.stop.set()
                if task.thread is None:
                    del self._tasks[key]
            self._condition.notify_all()
            while self._tasks:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def _dispatch(self) -> None:
        with self._condition:
            while self._accepting:
                active = dict.fromkeys(self._limits, 0)
                for task in self._tasks.values():
                    if task.thread is not None:
                        active[task.lane] += 1
                wait: float | None = None
                for key, task in list(self._tasks.items()):
                    if (
                        task.target is None
                        or task.thread is not None
                        or active[task.lane] >= self._limits[task.lane].active
                    ):
                        continue
                    delay = task.due - self._clock()
                    if delay > 0:
                        wait = delay if wait is None else min(wait, delay)
                        continue
                    thread = threading.Thread(
                        target=self._run, args=(key, task), name=f"flash-{task.lane}", daemon=True
                    )
                    task.thread = thread
                    try:
                        thread.start()
                    except Exception:
                        task.thread = None
                        task.due = self._clock() + 1.0
                        wait = 1.0 if wait is None else min(wait, 1.0)
                        _log.exception("could not start background task %r; retrying", key)
                    else:
                        active[task.lane] += 1
                self._condition.wait(wait)

    def _run(self, key: Hashable, task: _Task) -> None:
        result = None
        token = _stop.set(task.stop)
        try:
            with self.bind():
                checkpoint()
                assert task.target is not None
                result = task.target()
        except TaskStopped:
            pass
        except Exception:
            _log.exception("background task %r failed", key)
        finally:
            _stop.reset(token)
            with self._condition:
                if isinstance(result, Retry) and self._accepting and not task.stop.is_set():
                    task.due = self._clock() + max(0.0, result.delay)
                    task.thread = None
                else:
                    self._tasks.pop(key, None)
                self._condition.notify_all()
