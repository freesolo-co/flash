"""Shared poll-loop scaffolding for the provider job pollers.

``runpod/jobs.py:poll_job`` and ``vast/jobs.py:poll_vast_job`` are independent live
poll loops with provider-specific terminal-state logic, but they share three verbatim
blocks: a timestamped ``say()`` logger, a consecutive-poll-error retry/give-up counter,
and the heartbeat progress-surfacing block (key on (stage, step, ts), log
``worker: stage=… step=… loss=… reward=…``). Only those provider-neutral pieces live here; each
poller keeps its own status/terminal handling inline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def make_say(log) -> Callable[[str], None]:
    """A timestamped line logger that no-ops when ``log`` is None."""

    def say(msg: str) -> None:
        if log is not None:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log, flush=True)

    return say


class PollErrorTracker:
    """Counts consecutive poll errors and decides when to give up.

    Encapsulates the identical retry block both pollers use: on a transient fetch
    error, log it, give up after ``max_errors`` consecutive failures, otherwise sleep
    a linear backoff (capped at 60 s) before the caller retries.
    """

    def __init__(self, say: Callable[[str], None], interval_s: float, max_errors: int = 8) -> None:
        self._say = say
        self._interval_s = interval_s
        self._max_errors = max_errors
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def record(self, exc: Exception) -> bool:
        """Register a poll error. Returns True if the caller should give up (too many),
        else sleeps the backoff and returns False (caller should ``continue``)."""
        self._count += 1
        self._say(f"poll error ({self._count}): {exc}")
        if self._count >= self._max_errors:
            return True
        time.sleep(min(60, self._interval_s * self._count))
        return False


def surface_heartbeat(
    heartbeat_reader: Callable[[], Any] | None,
    last_hb_key: tuple | None,
    say: Callable[[str], None],
) -> tuple[tuple | None, str | None]:
    """Read a heartbeat and, if it advanced, log worker progress.

    Returns ``(hb_key, stage)`` where ``hb_key`` is the new (stage, step, ts) key (or the
    unchanged ``last_hb_key`` when nothing advanced) and ``stage`` is the stage of the new
    heartbeat when it advanced (else None). Callers use the returned ``stage`` for their
    own setup-vs-training stall bookkeeping.
    """
    if heartbeat_reader is None:
        return last_hb_key, None
    try:
        hb = heartbeat_reader()
    except Exception:
        hb = None
    if not hb:
        return last_hb_key, None
    key = (hb.get("stage"), hb.get("step"), hb.get("ts"))
    if key == last_hb_key:
        return last_hb_key, None
    stage = hb.get("stage")
    step = hb.get("step")
    loss = hb.get("loss")
    reward = hb.get("reward")
    say(
        f"worker: stage={stage}"
        + (f" step={step}" if step is not None else "")
        + (f" loss={loss:.4f}" if isinstance(loss, (int, float)) else "")
        + (f" reward={reward:.3f}" if isinstance(reward, (int, float)) else "")
    )
    return key, stage
