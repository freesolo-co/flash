"""shared parent-side work accounting for verl child liveness."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class ParentWorkSnapshot:
    completed: int
    depth: int


class ParentWorkGauge:
    """track completed work and the depth of calls currently executing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._completed = 0
        self._depth = 0

    @contextlib.contextmanager
    def busy(self):
        with self._lock:
            self._depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1

    def complete(self) -> None:
        with self._lock:
            self._completed += 1

    def snapshot(self) -> ParentWorkSnapshot:
        with self._lock:
            return ParentWorkSnapshot(self._completed, self._depth)
