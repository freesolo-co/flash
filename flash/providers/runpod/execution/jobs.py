"""Durable RunPod handles, statuses, decoding, and polling support."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from flash._internal.diagnostics import sanitize_diagnostic
from flash.providers._lifecycle.instances.poll import (
    _attempt_int,
    heartbeat_oom_for_attempt,
    surface_heartbeat,
)
from flash.providers.artifacts.hf import worker_flagged_retriable
from flash.providers.core.base import PollResult
from flash.providers.runpod.client import api as runpod_api

# Per-part cap for sanitized terminal-failure text. The parts are already provider-bounded; the
# limit exists so redaction can never silently truncate a tail we chose to surface. It is applied
# after sanitizing the complete text, keeping the newest bytes.
FAILURE_TEXT_LIMIT = 64_000

# Re-export for callers that import PollResult from here.
__all__ = [
    "JobHandle",
    "PollResult",
    "capacity_grace_multiplier",
    "decode_output",
]


# capacity grace on the LAST candidate class: there is nowhere left to walk, so wait longer before
# giving up. purely a timing knob -- it says nothing about whether a retry follows, because
# on_last_gpu (runner/supervise/retry_decision.py) is also true when the infra retry budget is exhausted.
LAST_GPU_CAPACITY_GRACE_S = 900.0

# multi-card shapes are rarer than single cards, so a grace sized for 1x expires on a 4x wait that
# was merely slow rather than starved. scale the wait with the card count so scarcity is waited out
# on one queue position rather than paying another cold start.
CAPACITY_GRACE_PER_GPU_CAP = 4


def capacity_grace_multiplier(gpu_count: int) -> int:
    """Scarcity multiplier on the capacity grace for a ``gpu_count``-card shape.

    Linear in the card count and capped: a 4x shape waits 4x as long as a 1x one, and anything
    wider than the cap waits the cap rather than growing without bound. Single-card runs multiply
    by 1, so their timing is exactly what it was.
    """
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        return 1
    return min(gpu_count, CAPACITY_GRACE_PER_GPU_CAP)


# how long ONE "a worker is coming up" health reading keeps suppressing the capacity timer. probes
# run every 90s, so this absorbs a couple of missed or failed probes and no more: if health stops
# being readable entirely, the capacity timer re-arms rather than waiting out the run on an
# observation nobody can still confirm.
WORKER_COMING_UP_TTL_S = 300.0

TERMINAL_OK = {"COMPLETED"}
# CANCELLED/TIMED_OUT = provider-killed (retriable); FAILED = worker died on its own (fails fast).
PLATFORM_TERMINATIONS = {"CANCELLED", "TIMED_OUT"}
TERMINAL_FAIL = {"FAILED"} | PLATFORM_TERMINATIONS


def stall_kwargs(on_last_gpu: bool = False, gpu_count: int = 1) -> dict:
    """poll_job stall-window kwargs. queue/throttled grace is ~5 min normally, ~15 min on last GPU
    (nowhere left to walk), then scaled by the card count because multi-card shapes are scarcer."""
    grace = (LAST_GPU_CAPACITY_GRACE_S if on_last_gpu else 300.0) * capacity_grace_multiplier(
        gpu_count
    )
    return {
        "stall_after_s": 1500.0,
        "setup_grace_s": 3000.0,
        "queue_grace_s": grace,
        "throttled_grace_s": grace,
    }


@dataclass
class JobHandle:
    endpoint_id: str
    endpoint_name: str
    key_fingerprint: str
    job_id: str | None
    attempt: int
    started_ts: float

    def to_dict(self) -> dict:
        data = {
            "provider": "runpod",
            "endpoint_id": self.endpoint_id,
            "endpoint_name": self.endpoint_name,
            "key_fingerprint": self.key_fingerprint,
            "attempt": self.attempt,
            "started_ts": self.started_ts,
        }
        if self.job_id is not None:
            data["job_id"] = self.job_id
        return data

    @classmethod
    def from_dict(cls, d: dict) -> JobHandle:
        if d.get("provider") != "runpod":
            raise ValueError("persisted RunPod provider identity is invalid")
        attempt = _attempt_int(d.get("attempt"))
        if attempt is None:
            raise ValueError("persisted RunPod attempt identity is invalid")
        started_raw = d.get("started_ts")
        if isinstance(started_raw, bool) or not isinstance(started_raw, (int, float)):
            raise ValueError("persisted RunPod launch timestamp is invalid")
        started_ts = float(started_raw)
        if not math.isfinite(started_ts) or started_ts <= 0:
            raise ValueError("persisted RunPod launch timestamp is invalid")
        endpoint_id = d.get("endpoint_id")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            raise ValueError("persisted RunPod endpoint identity is invalid")
        endpoint_name = d.get("endpoint_name")
        if not isinstance(endpoint_name, str) or not endpoint_name:
            raise ValueError("persisted RunPod endpoint name is invalid")
        fingerprint = d.get("key_fingerprint")
        if not runpod_api._is_valid_key_fingerprint(fingerprint):
            raise ValueError("persisted RunPod key fingerprint is invalid")
        job_id = d.get("job_id")
        if job_id is not None and (not isinstance(job_id, str) or not job_id):
            raise ValueError("persisted RunPod job identity is invalid")
        return cls(endpoint_id, endpoint_name, fingerprint, job_id, attempt, started_ts)


def _safe_failure_text(value: object, limit: int = FAILURE_TEXT_LIMIT) -> str:
    """Redact credentials out of one part of a user-visible RunPod failure detail.

    Provider errors and worker stdout tails reach the run log verbatim, so a control-plane secret
    the worker echoed would be printed. The instance providers sanitize every part of their failure
    detail; this keeps RunPod symmetric with them. The complete text is sanitized before the bound
    is applied, and the bound keeps the newest bytes: slicing first could cut a credential at the
    boundary so its surviving part no longer value-matches.
    """
    sanitized = sanitize_diagnostic(value, limit=1 << 30)
    return sanitized[-max(0, int(limit)) :]


def decode_output(output) -> dict:
    """Decode a queue-job output into the worker's metrics dict."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"unexpected job output tail: {_safe_failure_text(output, 200)}"
            ) from exc
    if not isinstance(output, dict):
        raise RuntimeError(f"unexpected job output type: {type(output)}")
    if output.get("error"):
        stdout_tail = _safe_failure_text(output.get("stdout") or "")
        msg = f"Remote execution failed: {_safe_failure_text(output['error'])}"
        if stdout_tail:
            msg += f"\n--- worker stdout tail ---\n{stdout_tail}"
        raise RuntimeError(msg)
    return output


def _append_failure_artifacts(detail: str, failure_detail_reader) -> str:
    """Append worker-uploaded failure artifacts to a RunPod terminal-status detail."""
    if failure_detail_reader is None:
        return detail
    extra = failure_detail_reader(force=True)
    if not extra:
        return detail
    if detail:
        return f"{detail}\n{extra}"
    return extra


@dataclass
class GraceTimer:
    """Grace timer: arms on first active poll, expires after continuous active state.

    `since` excludes unobservable intervals; clearing it fully resets the timer.
    """

    since: float | None = None
    seen: float | None = None

    def expired(self, active: bool, now: float, grace: float) -> bool:
        self.seen = now
        if not active:
            self.since = None
            return False
        if self.since is None:
            self.since = now  # first poll the state held -> arm, but never fail on the same poll
            return False
        return now - self.since > grace

    def unknown(self, now: float) -> None:
        """A reading that proves nothing either way: hold confirmed duration, drop the blind gap.

        Resetting hides persistent failure; charging the gap can expire a newly restarted state.
        """
        if self.since is not None and self.seen is not None:
            self.since += now - self.seen
        self.seen = now


# heartbeat readers and provenance checks are provider-neutral and live in
# flash.providers.artifacts.hf; this module imports only the predicate it executes.


def surfaced_worker_flags(
    heartbeat_reader,
    last_hb_key,
    say,
    current_attempt: int | None = None,
    *,
    launch_ts: float | None = None,
) -> tuple:
    """Read once for heartbeat surfacing plus structured retriable/OOM flags."""
    hb = heartbeat_reader(force=True) if heartbeat_reader is not None else None
    last_hb_key, _ = surface_heartbeat(lambda: hb, last_hb_key, say)
    retriable = worker_flagged_retriable(
        lambda force=False: hb, launch_ts=launch_ts, current_attempt=current_attempt
    )
    return last_hb_key, retriable, heartbeat_oom_for_attempt(hb, current_attempt)
