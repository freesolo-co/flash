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
* ``cancel_undispatched_on_close`` decides whether ``close`` immediately cancels work that has not
  yet won the condition-protected dispatch claim. Leaving it off gives the consumer the bounded
  shutdown window to claim the final partial batch; turning it on fails that work without scoring it.
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
        cancel_undispatched_on_close: bool = False,
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
        self._cancel_undispatched_on_close = bool(cancel_undispatched_on_close)
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
            thread = threading.Thread(target=self._run, name=self.thread_name, daemon=True)
            thread.start()
            self._thread = thread

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

    def _claim_batch(self) -> list[_Waiter] | None:
        """Wait for work, then atomically cancel it or claim it for dispatch."""
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
            batch = self._pending[: self.max_batch_size]
            if not batch:
                # `close` cancelled the queue while this held no claim: the only way the queue can
                # empty between the outer loop and here. dispatching now would score an empty batch.
                return None
            del self._pending[: len(batch)]
            self._in_flight = batch
            self._condition.notify_all()
            return batch

    def _run(self) -> None:
        try:
            while True:
                batch = self._claim_batch()
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
        """Stop accepting requests and settle queued or claimed work within the shutdown bound.

        There is exactly one cancellation point, and ``_pending`` is what it cancels: work that has
        not won the condition-protected claim in ``_claim_batch``. The flag only decides whether the
        consumer is given the bounded window to win that claim first, so an unbillable dispatch is
        prevented by never yielding the condition rather than by a second racing check.

        ``_in_flight`` is the claimed batch, so shutdown retains it through the join and the final
        drain. A batch that lands within the bound scatters its real results; ``complete`` is
        first-writer-wins, so the drain below cannot overwrite them. One that does not is failed so
        its callers unblock.

        The two callers this replaced spelled the final drain differently -- one gated it on
        ``thread.is_alive()``, the other ran it unconditionally -- but the forms are equivalent, so
        there is no flag for it: a thread that exited has already cleared ``_in_flight`` and settled
        its own stranded waiters in ``_run``'s ``finally``, leaving nothing for the drain to find.
        """
        error = self._make_error(f"{self.label} shut down")
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
            # the consumer cannot run while this holds the condition, so skipping the window below
            # means it never gets to claim -- which is exactly what an unbillable dispatch needs.
            while (
                not self._cancel_undispatched_on_close
                and self._pending
                and thread is not None
                and thread.is_alive()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            cancelled = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()
        for waiter in cancelled:
            waiter.complete(error=error)
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        # unconditional: a thread that exited already emptied `_in_flight` and settled its own
        # stranded waiters, so this is a no-op in that case rather than a second verdict.
        with self._condition:
            in_flight = list(self._in_flight)
        for waiter in in_flight:
            waiter.complete(error=error)
