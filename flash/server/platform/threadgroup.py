"""Lifespan-scoped ownership for background threads."""

import contextlib
import threading
import time
from collections.abc import Callable, Iterator


class OwnedThreadGroup:
    """One generation of background threads owned by a single server lifespan.

    Three properties make the group safe to shut down, and each is a defect the plain
    "set of threads plus a stop flag" shape does not have:

    - Admission and registration are one critical section, so a thread can neither start
      after the group closed nor be missed by a join that ran between the two steps.
    - Members discard themselves on completion, so the join observes membership rather
      than liveness: a registered thread that has not yet reached its first instruction is
      still owned, where an ``is_alive()`` filter would read it as already gone.
    - A generation is an object. Arming a new lifespan means constructing a new group, so a
      thread that outlived its own shutdown deadline keeps its own stop signal set forever.
      Reusing one registry would instead forget that thread and re-arm its loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: set[threading.Thread] = set()
        self._accepting = True
        self.stopped = threading.Event()

    def close(self) -> None:
        """Refuse further admissions and signal every member to unwind.

        Closing and admission share one linearization point: work already admitted completes its
        short launch decision before the stop signal, while later work is refused. Admission bodies
        must therefore never contain an unbounded wait.
        """
        with self._lock:
            self._accepting = False
            self.stopped.set()

    @contextlib.contextmanager
    def admit(self) -> Iterator["Admission | None"]:
        """Hold admission open across the caller's own critical section.

        Yields ``None`` once the group has closed. Keep the body short: it blocks every
        other admission and the shutdown bookkeeping.
        """
        with self._lock:
            yield Admission(self) if self._accepting else None

    def start(self, target: Callable[..., object], *args: object) -> bool:
        """Start an owned daemon thread. False when the group has already closed."""
        with self.admit() as slot:
            if slot is None:
                return False
            slot.start(target, *args)
        return True

    def wait(self, timeout: float) -> bool:
        """Join the members. False when one is still registered at the deadline."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                threads = tuple(self._threads)
            if not threads:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            threads[0].join(remaining)

    def sleep(self, seconds: float) -> None:
        """Wait between a member's retries, returning early once the group closes."""
        if seconds > 0:
            self.stopped.wait(seconds)

    def _start_locked(self, target: Callable[..., object], args: tuple[object, ...]) -> None:
        thread = threading.Thread(target=self._run, args=(target, args), daemon=True)
        self._threads.add(thread)
        try:
            thread.start()
        except Exception:
            self._threads.discard(thread)
            raise

    def _run(self, target: Callable[..., object], args: tuple[object, ...]) -> None:
        _CURRENT.group = self
        try:
            target(*args)
        finally:
            with self._lock:
                self._threads.discard(threading.current_thread())


_CURRENT = threading.local()


def current_group() -> OwnedThreadGroup | None:
    """The group that owns the calling thread, if it was started by one.

    A member must consult its own generation rather than whichever group is current: one that
    outlived its shutdown deadline would otherwise read the next lifespan's group and revive.
    """
    return getattr(_CURRENT, "group", None)


def adopt_group(group: OwnedThreadGroup | None) -> None:
    """Answer to ``group``'s stop signal on this thread without joining its membership.

    For a thread a member spawns whose work has its own lifetime: it must still see its parent's
    generation shut down, but the bounded lifespan join has no business waiting for it.
    """
    _CURRENT.group = group


class Admission:
    """A held admission slot: the group cannot finish closing while this is open."""

    def __init__(self, group: OwnedThreadGroup) -> None:
        self._group = group

    def start(self, target: Callable[..., object], *args: object) -> None:
        """Register and start an owned thread without releasing the admission."""
        self._group._start_locked(target, args)
