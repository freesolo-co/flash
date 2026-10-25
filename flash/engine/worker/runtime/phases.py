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
                if len(self.active) < 128:
                    self.active.setdefault(span, (name, time.monotonic()))
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
            return {
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
