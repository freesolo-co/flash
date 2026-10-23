"""Bounded, ordered disk-snapshot persistence beside the adapter publisher.

The watcher owns admission and completion callbacks. Only storage runs on the
worker thread, so checkpoint ledgers remain confined to their owning watcher.
"""

from __future__ import annotations

import errno
import os
import queue
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future

from flash.engine.worker.io.heartbeat import join_while_draining
from flash.engine.worker.io.hf import PublicationResult
from flash.engine.worker.runtime.state import WorkerContext


def pin_checkpoint(source: str, destination: str) -> str:
    """Retain completed, immutable checkpoint files outside trainer retention.

    Cross-device staging copies files; other errors fail closed. An incomplete
    snapshot must never release a retention gate or enter the persistence queue.
    """

    def link_or_copy(src, dst):
        try:
            os.link(src, dst)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(src, dst)

    if os.path.lexists(destination):
        raise FileExistsError(destination)
    try:
        shutil.copytree(source, destination, copy_function=link_or_copy)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


class CheckpointPersistence:
    """One serial uploader, at most two admitted snapshots including active work.

    Call ``wait_for_capacity`` before capturing a snapshot. Direct calls before
    ``start`` persist synchronously; the watch loop enables overlap explicitly.
    ``close`` drains even after a publication error, preserving snapshot ownership.
    """

    def __init__(self, *, context: WorkerContext) -> None:
        self.context = context
        self._jobs: queue.Queue = queue.Queue()
        self._pending: deque[tuple[Future, Callable[[PublicationResult], None]]] = deque()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("checkpoint persistence is already running")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def reap(self) -> None:
        failure = None
        while self._pending and self._pending[0][0].done():
            future, complete = self._pending.popleft()
            try:
                complete(future.result())
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    def wait_for_capacity(self) -> None:
        self.reap()
        if len(self._pending) >= 2:
            started = time.monotonic()
            self._pending[0][0].result()
            print(f"[ckpt] persistence backpressure_s={time.monotonic() - started:.3f}", flush=True)
            self.reap()

    def submit(self, persist: Callable[[], PublicationResult], complete) -> None:
        self.wait_for_capacity()
        if self._thread is None:
            complete(persist())
            return
        future: Future = Future()
        self._pending.append((future, complete))
        self._jobs.put((future, persist))

    def _run(self) -> None:
        while (job := self._jobs.get()) is not None:
            future, persist = job
            try:
                future.set_result(persist())
            except BaseException as error:
                future.set_exception(error)

    def close(self) -> None:
        if self._thread is None:
            return
        self._jobs.put(None)
        join_while_draining(self._thread, "checkpoint persistence", context=self.context)
        self._thread = None
        self.reap()
