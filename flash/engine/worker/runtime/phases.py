"""Bounded, attempt-owned phase diagnostics. Overlapping spans are never added together."""

from __future__ import annotations

import contextlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps

PREFIX = "FLASH_TRAINING_PHASE "
PHASES = frozenset(
    {
        "model_download",
        "capability_checks",
        "trainer_initialization",
        "model_initialization",
        "compilation_and_warmup",
        "first_update",
        "steady_update",
        "checkpoint_capture",
        "publication",
        "cpu_publication",
    }
)


@dataclass
class PhaseTimings:
    started: float = field(default_factory=time.monotonic)
    active: dict[str, tuple[str, float]] = field(default_factory=dict)
    longest: dict[str, float] = field(default_factory=dict)
    first_update_ready_s: float | None = None
    # customer billing meter. it starts when this attempt's first optimizer update starts, so model
    # download, data loading and trainer/model/vLLM initialization never bill, and stops when the
    # training loop exits, so final adapter publication never bills. monotonic and attempt-local.
    billing_started: float | None = None
    billing_stopped: float | None = None
    lock: object = field(default_factory=threading.Lock)

    def observe(self, event):
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("phase"), str)
            or event["phase"] not in PHASES
        ):
            return
        name, span, state = event["phase"], event.get("span"), event.get("state")
        if not isinstance(span, str) or len(span) != 32:
            return
        with self.lock:
            if state == "started":
                now = time.monotonic()
                if name == "first_update" and self.billing_started is None:
                    self.billing_started = now
                if len(self.active) < 128:
                    self.active.setdefault(span, (name, now))
            elif state == "completed" or state == "failed":
                self.active.pop(span, None)
                seconds = event.get("seconds")
                # JSON integers can exceed float range even within the line-size limit.
                try:
                    finite = type(seconds) in (float, int) and math.isfinite(seconds)
                except OverflowError:
                    finite = False
                if state != "completed" or not finite or seconds < 0:
                    return
                self.longest[name] = max(self.longest.get(name, 0.0), seconds)
                if name == "first_update" and self.first_update_ready_s is None:
                    self.first_update_ready_s = time.monotonic() - self.started

    def stop_billing(self):
        """Freeze the billable training meter when the training loop exits. Idempotent."""
        with self.lock:
            if self.billing_started is not None and self.billing_stopped is None:
                self.billing_stopped = time.monotonic()

    def billable_train_seconds(self):
        """GPU seconds this attempt has spent training (0.0 before its first optimizer update)."""
        with self.lock:
            return self._billable_unlocked(time.monotonic())

    def _billable_unlocked(self, now):
        # an explicit zero before the first optimizer update, never "absent": a metered worker
        # cancelled during trainer/model init must bill zero, not look like a legacy worker that
        # falls back to quote pricing.
        if self.billing_started is None:
            return 0.0
        end = self.billing_stopped if self.billing_stopped is not None else now
        return max(0.0, end - self.billing_started)

    def observe_line(self, line):
        # Ray prefixes stdout with actor identity; retain only our bounded JSON record.
        if PREFIX not in line or len(line) > 4096:
            return
        try:
            event = json.JSONDecoder().raw_decode(line.split(PREFIX, 1)[1])[0]
        except (ValueError, TypeError):
            return
        self.observe(event)

    def snapshot(self):
        with self.lock:
            elapsed = {}
            now = time.monotonic()
            for name, started in self.active.values():
                elapsed[name] = round(max(elapsed.get(name, 0), now - started), 3)
            billable = self._billable_unlocked(now)
            return {
                "billable_train_s": round(billable, 3),
                "active_phase_elapsed_s": elapsed,
                "longest_phase_s": {name: round(value, 3) for name, value in self.longest.items()},
                **(
                    {"worker_to_first_update_s": round(self.first_update_ready_s, 3)}
                    if self.first_update_ready_s is not None
                    else {}
                ),
            }


@contextlib.contextmanager
def training_phase(name, *, context=None, step=None):
    """Log start/end using a monotonic clock; failure never becomes a completed sample."""
    if name not in PHASES:
        raise ValueError(f"unknown training phase: {name}")
    event = {"phase": name, "span": uuid.uuid4().hex, "pid": os.getpid()}
    if step is not None:
        event["update"] = int(step)  # not a native `step:` line or billing evidence

    def emit(state, **fields):
        value = dict(event, state=state, **fields)
        if context is not None:
            context.phase_timings.observe(value)
        # Diagnostics must not hide an original training error if stdout has closed.
        with contextlib.suppress(OSError):
            print(PREFIX + json.dumps(value, separators=(",", ":")), flush=True)

    started = time.monotonic()
    emit("started")
    try:
        yield
    except BaseException:
        emit("failed", seconds=time.monotonic() - started)
        raise
    else:
        emit("completed", seconds=time.monotonic() - started)


def timed_phase(name):
    """Instrument synchronous entrypoints, forwarding their existing attempt context."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with training_phase(name, context=kwargs.get("context")):
                return function(*args, **kwargs)

        return wrapped

    return decorate
