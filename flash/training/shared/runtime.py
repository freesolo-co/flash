"""Exclusive GPU execution ownership and crash-safe command completion."""

from __future__ import annotations

import fcntl
import logging
import threading
import time
from contextlib import ExitStack

from flash.training.shared.contracts import ConflictError, UnavailableError

_LOG = logging.getLogger(__name__)


class Runtime:
    def __init__(self, journal, backend, *, lock_path, clock=time.monotonic_ns):
        self.journal, self.backend, self.clock = journal, backend, clock
        self.lock_path = lock_path
        self.lock_file = None
        self.resources = ExitStack()
        self.thread = None
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.failed = False
        self.last_end = clock()

    def accept(self, operation, *args):
        # check after HTTP body consumption, under the same lock as recovery.
        # a request arriving during a device failure cannot leave fresh work orphaned.
        with self.journal.lock:
            if self.failed or self.stopping.is_set():
                raise UnavailableError("shared runtime unavailable")
            result = getattr(self.journal, operation)(*args)
        self.wake.set()
        return result

    def start(self):
        if self.lock_file is not None:
            raise RuntimeError("runtime already started")
        handle = self.resources.enter_context(open(self.lock_path, "a+"))  # noqa: SIM115
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            handle.close()
            raise RuntimeError("another runtime owns this pool journal") from None
        self.lock_file = handle
        started = self.clock()
        try:
            if callable(self.backend):
                self.backend = self.backend()
            self.journal.recover()
            self.journal.record_overhead("startup", max(0, self.clock() - started))
        except BaseException:
            self.resources.close()
            self.lock_file = None
            raise
        self.last_end = self.clock()
        self.thread = threading.Thread(target=self._loop, name="flash-shared-training", daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while not self.stopping.is_set() and not self.failed:
                self.wake.clear()
                if not self.run_one():
                    self.wake.wait(0.25)
        except BaseException:
            # leave the claimed command uncertain if persistence itself failed.
            self.failed = True
            _LOG.exception("shared runtime stopped; replacement and explicit recovery required")

    def run_one(self):
        command = self.journal.claim()
        if command is None:
            return False
        started = self.clock()
        self.journal.record_overhead("between_commands", max(0, started - self.last_end))
        failed = False
        fatal = False
        try:
            result = self.backend.execute(command)
            self.backend.synchronize()
        except ConflictError as error:
            result, failed = {"error": str(error)}, True
        except BaseException:
            # an arbitrary device error may affect shared base state or the CUDA context.
            # do not let another tenant execute against an uncertain runtime.
            _LOG.exception("shared backend failed during %s", command["operation"])
            result, failed, fatal = (
                {"error": "shared runtime lost; restore a completed checkpoint"},
                True,
                True,
            )
        ended = self.clock()
        self.journal.finish(
            command, result, execution_ns=max(0, ended - started), failed=failed, lose_session=fatal
        )
        self.last_end = ended
        if fatal:
            self.failed = True
            self.journal.recover()
        return True

    def stop(self):
        self.stopping.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("shared operation still executing; process termination required")
        self.journal.record_overhead("shutdown", max(0, self.clock() - self.last_end))
        if self.lock_file is not None:
            self.resources.close()
            self.lock_file = None
