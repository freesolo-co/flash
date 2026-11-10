"""Coalesce concurrent GRPO reward requests into ordered batches in the parent worker.

Daemon consumer threads (one by default) gather requests until the oldest waiter's grace period
expires, score the batch, and scatter results in request order. With ``max_concurrent_batches``
above one, a slow batch no longer holds up the requests that arrive behind it. Shutdown gives pending work a
bounded drain window and propagates scoring failures to all affected waiters.
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

    def wait(self) -> Any:
        # no deadline: the wait is the scorer's own time plus however long the batch ahead of it
        # takes, and the caller's stall watchdog is what catches a genuinely wedged scorer.
        self.done.wait()
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError(f"{self.label} waiter completed without a result")
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
        max_concurrent_batches: int = 1,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"{label} batch size must be positive")
        if max_concurrent_batches <= 0:
            raise ValueError(f"{label} concurrent batch count must be positive")
        if flush_wait_s <= 0:
            raise ValueError(f"{label} flush wait must be positive")
        self._score_batch = score_batch
        self.max_batch_size = int(max_batch_size)
        self.flush_wait_s = float(flush_wait_s)
        self.label = label
        self.thread_name = thread_name
        self.max_concurrent_batches = int(max_concurrent_batches)
        self._condition = threading.Condition()
        self._pending: list[_Waiter] = []
        # every claimed batch not yet settled, across all consumer threads.
        self._in_flight: list[_Waiter] = []
        self._closed = False
        # `_thread` is the first consumer; `_threads` holds all of them.
        self._thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []
        self._live = 0

    def start(self) -> None:
        """Start the consumer threads; idempotent and safe to race."""
        with self._condition:
            if self._closed:
                raise RuntimeError(f"{self.label} shut down")
            if self._thread is not None:
                return
            # registered as each one starts: a failed first start leaves the batcher unstarted (a
            # retry starts it again), and a later failure leaves the consumers already running.
            for index in range(self.max_concurrent_batches):
                thread = threading.Thread(
                    target=self._run,
                    name=self.thread_name if index == 0 else f"{self.thread_name}-{index}",
                    daemon=True,
                )
                self._live += 1
                try:
                    thread.start()
                except BaseException:
                    self._live -= 1
                    raise
                self._threads.append(thread)
                if self._thread is None:
                    self._thread = thread

    def submit(self, request: Any) -> Any:
        """Enqueue one request and block until its batch has been scored."""
        self.start()
        with self._condition:
            if self._closed:
                raise RuntimeError(f"{self.label} shut down")
            waiter = _Waiter(request, enqueued_at=time.monotonic(), label=self.label)
            self._pending.append(waiter)
            self._condition.notify_all()
        return waiter.wait()

    def _claim_batch(self) -> list[_Waiter] | None:
        """Wait for work, then atomically cancel it or claim it for dispatch."""
        with self._condition:
            while True:
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
                if batch:
                    break
                # the queue emptied while this held no claim: `close` cancelled it, or another
                # consumer claimed it. dispatching now would score an empty batch.
                if self._closed:
                    return None
            del self._pending[: len(batch)]
            self._in_flight.extend(batch)
            self._condition.notify_all()
            return batch

    def _settle(self, batch: list[_Waiter]) -> None:
        with self._condition:
            claimed = {id(waiter) for waiter in batch}
            self._in_flight = [w for w in self._in_flight if id(w) not in claimed]
            self._condition.notify_all()

    def _run(self) -> None:
        batch: list[_Waiter] = []
        error = RuntimeError(f"{self.label} stopped")
        try:
            while True:
                batch = self._claim_batch() or []
                if not batch:
                    return
                try:
                    results = self._score_batch([waiter.request for waiter in batch])
                    scattered = list(zip(batch, results, strict=True))
                    for waiter, result in scattered:
                        waiter.complete(result=result)
                except Exception as failure:
                    for waiter in batch:
                        waiter.complete(error=failure)
                self._settle(batch)
                batch = []
        finally:
            # this thread's own claimed batch is never answered if it left mid-score; fail it
            # (a no-op for waiters it already settled).
            for waiter in batch:
                waiter.complete(error=error)
            self._settle(batch)
            # the LAST consumer is leaving for good, so anything still queued or in flight will never
            # be answered. fail them here rather than leaving their callers blocked forever. while
            # other consumers live they still own their batches and the queue.
            with self._condition:
                self._live -= 1
                stranded: list[_Waiter] = []
                if self._live == 0:
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
        error = RuntimeError(f"{self.label} shut down")
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            threads = list(self._threads)
            # Release the condition during each wait so a consumer can claim the final batch.
            while self._pending and any(thread.is_alive() for thread in threads):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            cancelled = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()
        for waiter in cancelled:
            waiter.complete(error=error)
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        # unconditional: a thread that exited already emptied `_in_flight` and settled its own
        # stranded waiters, so this is a no-op in that case rather than a second verdict.
        with self._condition:
            in_flight = list(self._in_flight)
        for waiter in in_flight:
            waiter.complete(error=error)
