"""Is the parent inside a unit of work the child is blocked on, right now?

The silence watchdog reads two parent-side signals. A COUNT answers "did another unit finish
since the last tick", which is enough while units are short. A single long unit defeats it: one
coalesced grading call can hold every completion-side counter flat for longer than the whole
silence budget, and a healthy run gets torn down mid-grade. This module supplies the other
signal -- the parent is inside that call -- which exists from the moment the call starts.

Both the GRPO reward buffer and the OPD bridge need it, so it lives here rather than being
reimplemented per caller with its own lock and its own subtly different reset rules.

Split out of `flash.engine.worker.io.heartbeat` and `flash.engine.worker.train.opd.bridge` so one
definition serves both.
"""

from __future__ import annotations

import contextlib
import threading


class ParentWorkGauge:
    """Counts the parent-side calls currently in flight, for the silence watchdog to read.

    A depth rather than a flag: calls overlap across the bridge's request threads, and a flag
    cleared by the first to finish would report idle while the rest are still running.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0

    @contextlib.contextmanager
    def working(self):
        """Mark the parent busy for the duration of one call.

        Wraps the call ITSELF, not its result, so the watchdog sees the work start rather than only
        its completion -- which is the whole point, since a long call has no completion to see.
        """
        with self._lock:
            self._depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1

    def in_flight(self) -> bool:
        """Is the parent inside at least one such call right now?"""
        with self._lock:
            return self._depth > 0


class OPDParentActivity:
    """The OPD bridge's side of the silence-watchdog contract, kept together in one place.

    The child blocks on this bridge in two distinct ways and neither signal alone is sufficient.
    Teacher scoring is many short calls, so it reports a COUNT. Environment hooks are arbitrary
    user code of unbounded duration, so they report a GAUGE. Both are read every tick by
    `VerlChildSilenceWatchdog`; supplying one without the other reopens the blind spot the other
    closes, which is why they are defined as a pair rather than scattered across the bridge.

    Mixed into the bridge, which owns the state below -- annotated, not assigned, so this class
    adds no `__init__` to the bridge's MRO and cannot race its initialisation order.
    """

    _stats_lock: threading.Lock
    _env_work: ParentWorkGauge
    teacher_ok: int
    teacher_transient: int
    teacher_error: int

    def _count_teacher_ok(self) -> None:
        """Record ONE completed teacher request, as it completes.

        The multi-turn path hands its whole turn list to ``score_many``, which runs
        OPD_TEACHER_SCORING_CONCURRENCY at a time. Counting the batch after it returns leaves this
        total frozen for as long as the batch takes -- 3 waves of 434s worst-case retries is 1302s,
        past the silence threshold -- so a healthy child blocked on scoring would look wedged.
        """
        with self._stats_lock:
            self.teacher_ok += 1

    def teacher_activity_count(self) -> int:
        """Completed teacher interactions, including transient and permanent failures."""
        with self._stats_lock:
            return self.teacher_ok + self.teacher_transient + self.teacher_error

    def env_work_in_flight(self) -> bool:
        """Is the parent inside an environment call right now?"""
        return self._env_work.in_flight()
