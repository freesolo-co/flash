"""One serialized GPU session with fail-closed sequencing and retry receipts."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import deque

from flash._internal.diagnostics import sanitize_diagnostic
from flash.custom.types import integer


class SessionConflict(ValueError):
    """The caller must inspect/reconcile session state before doing more work."""


class TrainingSession:
    def __init__(self, backend, *, max_steps=1000, idle_seconds=900):
        self.backend = backend
        self.id = uuid.uuid4().hex
        self.max_steps = integer(max_steps, "max_steps", 1, 1_000_000)
        self.idle_seconds = integer(idle_seconds, "idle_seconds", 30, 86400)
        self.version = 0
        self.step = 0
        self.sequence = 0
        self.failed = False
        self.closed = False
        self.last_activity = time.monotonic()
        self.lock = threading.Lock()
        self.receipt = None
        self.events = deque(maxlen=1000)
        self.event_cursor = 0
        self.on_commit = None
        self.restored_state = {}

    def status(self):
        return {
            "session_id": self.id,
            "model_version": self.version,
            "step": self.step,
            "sequence": self.sequence,
            "failed": self.failed,
            "closed": self.closed,
            "max_steps": self.max_steps,
            "restored_state": self.restored_state,
            **self.backend.info(),
        }

    def log(self, operation, result, duration):
        self.event_cursor += 1
        event = {
            "cursor": self.event_cursor,
            "operation": operation,
            "step": self.step,
            "model_version": self.version,
            "seconds": duration,
            "metrics": result.get("metrics", {}),
        }
        if result.get("error"):
            event["error"] = result["error"]
        self.events.append(event)
        # prompts, token ids and credentials never enter worker stdout.
        print(json.dumps({"custom_training": event}, allow_nan=False), flush=True)

    def execute(self, request):
        if not isinstance(request, dict) or set(request) != {
            "session_id",
            "sequence",
            "model_version",
            "operation",
            "arguments",
        }:
            raise ValueError("invalid operation envelope")
        sequence = integer(request["sequence"], "sequence", 1, 2**53 - 1)
        integer(request["model_version"], "model_version", 0, 2**53 - 1)
        fingerprint = hashlib.sha256(
            json.dumps(request, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        with self.lock:
            if request["session_id"] != self.id:
                raise SessionConflict(
                    "session changed; reconnect explicitly, never replay into new weights"
                )
            if sequence == self.sequence and self.receipt is not None:
                digest, result = self.receipt
                if digest != fingerprint:
                    raise SessionConflict("sequence already used for a different request")
                return result
            if sequence != self.sequence + 1:
                raise SessionConflict("out-of-order or expired request; no operation was executed")
            if self.closed or self.failed:
                raise SessionConflict(
                    "session is closed or failed; no further operations are allowed"
                )
            if request["model_version"] != self.version:
                raise SessionConflict(
                    "stale model version; inspect current weights before continuing"
                )
            operation = request["operation"]
            if operation == "optim_step" and self.step >= self.max_steps:
                raise ValueError("session optimizer-step limit reached")
            # validation must finish before any CUDA work or gradient mutation.
            arguments = self.backend.validate(operation, request["arguments"])
            if operation == "save_checkpoint":
                store = getattr(self.backend, "checkpoints", None)
                if store is not None and self.step in store.published:
                    raise ValueError(
                        "this step already has an immutable durable checkpoint; use its run/step-N reference"
                    )
                arguments["metadata"] = {
                    "session_id": self.id,
                    "model_version": self.version,
                    "step": self.step,
                }
            start = time.monotonic()
            try:
                output = self.backend.execute(operation, arguments)
                # reject invalid numerical results before committing a success receipt.
                json.dumps(output, allow_nan=False)
            except BaseException as error:
                # a partially accumulated gradient or partially applied optimizer must never be replayed.
                self.failed = True
                self.log(
                    operation,
                    {"error": sanitize_diagnostic(error, limit=2000)},
                    time.monotonic() - start,
                )
                raise
            if operation == "load_checkpoint":
                self.restored_state = output.get("algorithm_state", {})
            if operation in {"optim_step", "load_checkpoint"}:
                self.version += 1
            if operation == "optim_step":
                self.step += 1
            if operation == "close":
                self.closed = True
            if self.on_commit is not None:
                try:
                    self.on_commit(self, operation, output)
                except BaseException:
                    self.failed = True
                    raise
            self.sequence = sequence
            self.last_activity = time.monotonic()
            output = {**output, **self.status()}
            self.receipt = (fingerprint, output)
            self.log(operation, output, time.monotonic() - start)
            return output

    def expired(self):
        # a running GPU call holds the lock, so it cannot be killed as an idle session.
        if not self.lock.acquire(blocking=False):
            return False
        try:
            return (
                self.closed
                or self.failed
                or (time.monotonic() - self.last_activity > self.idle_seconds)
            )
        finally:
            self.lock.release()
