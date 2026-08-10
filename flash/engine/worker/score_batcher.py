"""Coalesce concurrent scoring requests into one ordered batched call.

Both RL paths need the same thing: verl's child process fires many scoring requests at once, and the
thing that answers them (a flash env's judge for GRPO, a managed teacher for OPD) is far cheaper per
item when handed a batch. A single daemon thread takes whatever is pending once the oldest waiter's
grace period expires, scores it in one call, and scatters the results back in request order. One
thread means the scorer still sees exactly one top-level call at a time; concurrency lives inside the
scorer's own batched implementation.

Parent-side only. The batchers live on the flash side of the verl process boundary (the child
reaches them over the reward/teacher rpc bridges), so this module must never be imported by the
copied child modules -- it is not part of what ``sitecustomize.py`` delivers.

Callers differ in three ways, each a constructor argument rather than a fork in the code:

* ``make_error`` builds the errors this module raises itself -- shutdown, a stopped thread, a waiter
  that completed with nothing. GRPO wants a plain ``RuntimeError``; OPD wants a permanent
  ``TeacherError`` so the caller does not retry a batcher that is gone.
* ``wrap_batch_error`` re-shapes an exception raised by ``score_batch`` before it reaches the
  waiters. GRPO propagates as-is; OPD narrows to its own error type, preserving permanence.
* ``recheck_closed_after_wait`` decides whether a batch assembled during the flush window is still
  scored once ``close`` has been called. Leaving it off scores that final partial batch; turning it
  on abandons it to the stranded-waiter path.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class _Waiter:
    """one request waiting on a batched scoring call."""

    def __init__(self, request: Any, enqueued_at: float, *, label: str) -> None:
        self.request = request
        self.enqueued_at = enqueued_at
        self.label = label
        self.done = threading.Event()
        self.result: Any = None
        self.error: Exception | None = None
        self._lock = threading.Lock()

    def complete(self, *, result: Any = None, error: Exception | None = None) -> None:
        """Settle this waiter. Idempotent: the first writer wins.

        Shutdown races the consumer thread by design -- ``close`` may complete a waiter the thread is
        about to complete itself -- so the loser of that race must be a no-op rather than an
        overwrite, or a successfully scored request could be replaced by a shutdown error.
        """
        with self._lock:
            if self.done.is_set():
                return
            self.result = result
            self.error = error
            self.done.set()

    def wait(self, make_error: Callable[[str], Exception]) -> Any:
        # no deadline: the wait is the scorer's own time plus however long the batch ahead of it
        # takes, and the caller's stall watchdog is what catches a genuinely wedged scorer.
        self.done.wait()
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise make_error(f"{self.label} waiter completed without a result")
        return self.result


class ScoreBatcher:
    """Coalesce concurrent requests into one ordered batched scoring call.

    ``score_batch(requests)`` must return one result per request, in request order. The full vector
    is validated before any waiter is completed: a strict zip checked while scattering would resolve
    a prefix before discovering a length mismatch, leaving some requests answered and the rest
    failed off the same batch.
    """

    def __init__(
        self,
        score_batch: Callable[[list[Any]], list[Any]],
        *,
        max_batch_size: int,
        flush_wait_s: float,
        label: str,
        thread_name: str,
        make_error: Callable[[str], Exception] = RuntimeError,
        wrap_batch_error: Callable[[Exception], Exception] | None = None,
        recheck_closed_after_wait: bool = False,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"{label} batch size must be positive")
        if flush_wait_s <= 0:
            raise ValueError(f"{label} flush wait must be positive")
        self._score_batch = score_batch
        self.max_batch_size = int(max_batch_size)
        self.flush_wait_s = float(flush_wait_s)
        self.label = label
        self.thread_name = thread_name
        self._make_error = make_error
        self._wrap_batch_error = wrap_batch_error
        self._recheck_closed_after_wait = bool(recheck_closed_after_wait)
        self._condition = threading.Condition()
        self._pending: list[_Waiter] = []
        self._in_flight: list[_Waiter] = []
        self._closed = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the consumer thread; idempotent and safe to race."""
        with self._condition:
            if self._closed:
                raise self._make_error(f"{self.label} shut down")
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name=self.thread_name, daemon=True)
            thread = self._thread
        thread.start()

    def submit(self, request: Any) -> Any:
        """Enqueue one request and block until its batch has been scored."""
        self.start()
        with self._condition:
            if self._closed:
                raise self._make_error(f"{self.label} shut down")
            waiter = _Waiter(request, enqueued_at=time.monotonic(), label=self.label)
            self._pending.append(waiter)
            self._condition.notify_all()
        return waiter.wait(self._make_error)

    def _take_batch(self) -> list[_Waiter] | None:
        with self._condition:
            while not self._pending:
                if self._closed:
                    return None
                self._condition.wait()
            deadline = self._pending[0].enqueued_at + self.flush_wait_s
            while len(self._pending) < self.max_batch_size and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._recheck_closed_after_wait and self._closed:
                return None
            batch = self._pending[: self.max_batch_size]
            del self._pending[: len(batch)]
            self._in_flight = batch
            return batch

    def _run(self) -> None:
        try:
            while True:
                batch = self._take_batch()
                if batch is None:
                    return
                try:
                    results = self._score_batch([waiter.request for waiter in batch])
                    scattered = list(zip(batch, results, strict=True))
                    for waiter, result in scattered:
                        waiter.complete(result=result)
                except Exception as error:
                    wrapped = self._wrap_batch_error(error) if self._wrap_batch_error else error
                    for waiter in batch:
                        waiter.complete(error=wrapped)
                finally:
                    with self._condition:
                        self._in_flight = []
                        self._condition.notify_all()
        finally:
            # the thread is leaving for good, so anything still queued or in flight will never be
            # answered. fail them here rather than leaving their callers blocked forever.
            error = self._make_error(f"{self.label} stopped")
            with self._condition:
                stranded = [*self._pending, *self._in_flight]
                self._pending.clear()
                self._in_flight = []
                self._closed = True
                self._condition.notify_all()
            for waiter in stranded:
                waiter.complete(error=error)

    def close(self, timeout_s: float) -> None:
        """Stop accepting requests, fail what is queued, and wait out the batch in flight.

        The join is bounded because the scorer is a network call that can hang. A batch that lands
        within the bound scatters its real results -- ``complete`` is first-writer-wins, so the
        drain below cannot overwrite them. One that does not is failed so its callers unblock.

        The two callers this replaced spelled the final drain differently -- one gated it on
        ``thread.is_alive()``, the other ran it unconditionally -- but the forms are equivalent, so
        there is no flag for it: a thread that exited has already cleared ``_in_flight`` and settled
        its own stranded waiters in ``_run``'s ``finally``, leaving nothing for the drain to find.
        """
        error = self._make_error(f"{self.label} shut down")
        with self._condition:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()
            thread = self._thread
        for waiter in pending:
            waiter.complete(error=error)
        if thread is not None:
            thread.join(timeout=max(0.0, timeout_s))
        # unconditional: a thread that exited already emptied `_in_flight` and settled its own
        # stranded waiters, so this is a no-op in that case rather than a second verdict.
        with self._condition:
            in_flight = list(self._in_flight)
        for waiter in in_flight:
            waiter.complete(error=error)
