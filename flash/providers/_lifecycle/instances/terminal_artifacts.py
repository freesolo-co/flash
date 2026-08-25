"""one terminal-artifact protocol, shared by live polling and every recovery path.

an attempt's terminal evidence (the strict attempt marker plus its metrics.json) used to be decoded
in two places that disagreed about the same bytes: the live instance poller in ``poll_instance`` and
the reconstruction helpers in ``flash.runner.supervise.recovery``.

two things diverged.

* an unverifiable marker. the live poller classified it as a terminal ``job_failed``; recovery caught
  the same ``TypeError``/``ValueError`` and returned ``None``, which its callers read as "no evidence
  yet". neither adopts the corrupt artifact, so this is a classification split rather than a
  correctness hole, but the same bytes still produced two different stories depending on which layer
  happened to observe the attempt.
* the observation window. the live poller shares one absolute read deadline across its rereads, while
  recovery computed a fresh ~30s window for the marker and then ANOTHER fresh ~30s window for
  metrics. because the second window started only after the first finished, the effective ceiling was
  their SUM, and it drifted with however long the marker read happened to take.

this module owns that decode exactly once. it is evidence only: it never mutates run state, never
performs the exact-remote CAS, and never decides whether a run retries -- the attempt ledger stays
the sole persisted authority. it answers exactly one question, as a ``TerminalKind``, and callers
adapt that into their own vocabulary (a ``PollResult`` for the poller, a metrics dict for recovery)
because finishing a success is provider-specific (cost stamping, done-timestamp refinement) while
the DECISION is not.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from flash.providers._lifecycle.instances.poll import _attempt_int
from flash.providers._lifecycle.net.deadline import remaining_seconds

T = TypeVar("T")


class TerminalKind(Enum):
    """What one bounded observation of an attempt's terminal artifacts established.

    These five are exhaustive and mutually exclusive, so callers dispatch on the kind alone. In
    particular UNVERIFIABLE is its OWN kind rather than a flag on FAILURE: recovery used to fold it
    into "no evidence", and keeping it distinct is what stops that from reappearing.
    """

    # no marker is visible. the attempt may still be running, or hf has not exposed the upload yet.
    ABSENT = "absent"
    # a marker exists but cannot be tied to this attempt, so it cannot be trusted to speak for it.
    # distinct from ABSENT: absence means nothing settled, this means something settled unreadably.
    UNVERIFIABLE = "unverifiable"
    # a strict success marker exists but its metrics are not readable yet. callers must keep
    # observing rather than tearing down: the attempt already finished its paid work.
    PENDING = "pending"
    # a strict success marker plus readable metrics.
    SUCCESS = "success"
    # the attempt is terminally settled and did not succeed.
    FAILURE = "failure"


@dataclass(frozen=True)
class TerminalResolution:
    """One decided observation. ``marker`` is the validated marker when one was decoded."""

    kind: TerminalKind
    metrics: dict | None = None
    marker: dict | None = None
    # PENDING only, and diagnostic only: metrics were present but unparseable rather than missing.
    # both are the same visibility gap behind an already-authorized completion, so this narrows an
    # operator message and nothing else.
    metrics_unparseable: bool = False


ABSENT = TerminalResolution(TerminalKind.ABSENT)
UNVERIFIABLE = TerminalResolution(TerminalKind.UNVERIFIABLE)

INVALID_MARKER_DETAIL = "terminal marker is invalid or unverifiable"


@dataclass(frozen=True)
class AttemptIdentity:
    """The durable identity every terminal artifact must match to speak for this attempt."""

    run_id: str
    attempt: int
    launch_floor: float


@dataclass(frozen=True)
class ProbeBudget:
    """One observation window, computed ONCE and shared by every read inside a resolution.

    the marker and its metrics are two uploads of the same completion racing the same hf
    read-after-write lag, so they are one budget, not two. ``cutoff_at`` is absolute and is threaded
    through both reads: a marker read that burns most of the window leaves metrics the remainder
    instead of starting a fresh allowance. that makes the ceiling the cutoff rather than the sum of
    however many reads a path happens to make.

    ``tries`` and ``wait_s`` stay per-read because provider tests patch the module constants they
    come from; the shared absolute cutoff is what bounds the total.
    """

    tries: int
    wait_s: float
    cutoff_at: float | None = None


def read_within(
    read: Callable[[], T | None],
    budget: ProbeBudget,
    *,
    say: Callable[[str], None] | None = None,
    message: str = "",
) -> T | None:
    """Re-read one artifact until it is visible or the SHARED budget is spent.

    Every wait is clipped to the budget's absolute cutoff, so the window cannot be restarted by a
    later read in the same resolution.
    """
    value = read()
    tries = budget.tries
    while value is None and tries > 0:
        delay = budget.wait_s
        if budget.cutoff_at is not None:
            remaining = remaining_seconds(budget.cutoff_at)
            if remaining <= 0:
                break
            delay = min(delay, remaining)
        if say is not None and message:
            say(message)
        if delay > 0:
            time.sleep(delay)
        if budget.cutoff_at is not None and remaining_seconds(budget.cutoff_at) <= 0:
            break
        value = read()
        tries -= 1
    return value


def decode_terminal_marker(
    raw: str,
    *,
    run_id: str,
    attempt: int,
    launch_floor: float,
    deadline_at: float | None = None,
) -> dict:
    """Validate one attempt-scoped terminal marker against its durable identity.

    Fails closed on anything it cannot tie to exactly this attempt: an unknown schema, another run,
    another attempt, a timestamp before the launch this marker claims to end, or one past the bound
    the caller allows. A marker that cannot be verified must never settle an attempt.
    """
    marker = json.loads(raw)
    base_fields = {"attempt", "error", "ok", "retriable", "run_id", "ts"}
    if not isinstance(marker, dict) or set(marker) not in {
        frozenset(base_fields),
        frozenset(base_fields | {"source_attestation"}),
    }:
        raise ValueError("invalid terminal marker schema")
    marker_attempt = marker["attempt"]
    ts = marker["ts"]
    error = marker["error"]
    now = time.time()
    if "source_attestation" in marker:
        from flash.snapshot.archive import parse_attestation

        attestation = parse_attestation(marker["source_attestation"])
        if attestation["run_id"] != run_id or attestation["attempt"] != attempt:
            raise ValueError("invalid terminal marker source identity")
    if (
        isinstance(launch_floor, bool)
        or not isinstance(launch_floor, (int, float))
        or not math.isfinite(launch_floor)
        or launch_floor <= 0
        or isinstance(now, bool)
        or not isinstance(now, (int, float))
        or not math.isfinite(now)
        or now <= 0
        or _attempt_int(marker_attempt) is None
        or marker_attempt != _attempt_int(attempt)
        or type(marker["run_id"]) is not str
        or marker["run_id"] != run_id
        or type(marker["ok"]) is not bool
        or type(marker["retriable"]) is not bool
        or type(error) is not str
        or len(error) > 4096
        or isinstance(ts, bool)
        or not isinstance(ts, (int, float))
        or not math.isfinite(ts)
        or ts < launch_floor
        or ts > now + 120.0
        or (deadline_at is not None and ts > deadline_at)
    ):
        raise ValueError("invalid terminal marker identity")
    return marker


def _parse_metrics(raw: str | None) -> dict | None:
    if raw is None:
        return None
    try:
        metrics = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return metrics if isinstance(metrics, dict) else None


def resolve_terminal_artifacts(
    identity: AttemptIdentity,
    *,
    read_marker: Callable[[], str | None],
    read_metrics: Callable[[], str | None],
    budget: ProbeBudget,
    say: Callable[[str], None] | None = None,
    marker_deadline_at: float | None = None,
    marker_wait_message: str | None = None,
    metrics_message: str = "successful marker seen; waiting for metrics.json",
) -> TerminalResolution:
    """Decide what an attempt's marker plus metrics establish, under ONE shared budget.

    The strict marker is the sole terminal authority. A DONE file may later refine a
    marker-authorized success's completion timestamp, but it can never complete an attempt on its
    own, so it is deliberately not consulted here.

    ``marker_wait_message`` both enables and narrates waiting for the marker itself: live polling
    reads it once per iteration (the loop is the retry), while recovery re-reads it inside the shared
    budget. Passing the message is what asks for that wait, so neither can be set without the other.
    """
    raw = (
        read_marker()
        if marker_wait_message is None
        else read_within(read_marker, budget, say=say, message=marker_wait_message)
    )
    if raw is None:
        return ABSENT
    try:
        marker = decode_terminal_marker(
            raw,
            run_id=identity.run_id,
            attempt=identity.attempt,
            launch_floor=identity.launch_floor,
            deadline_at=marker_deadline_at,
        )
    except (TypeError, ValueError):
        return UNVERIFIABLE
    if not marker["ok"]:
        return TerminalResolution(TerminalKind.FAILURE, marker=marker)
    raw_metrics = read_within(read_metrics, budget, say=say, message=metrics_message)
    metrics = _parse_metrics(raw_metrics)
    if metrics is None:
        # the marker already authorized completion, so unreadable metrics are a visibility gap in the
        # SECOND upload, not a worker error. callers keep observing within the budget instead of
        # tearing down an attempt that already finished its paid work.
        return TerminalResolution(
            TerminalKind.PENDING,
            marker=marker,
            metrics_unparseable=raw_metrics is not None,
        )
    if "source_attestation" in marker:
        from flash.snapshot.archive import TERMINAL_ATTESTATION_KEY

        metrics = {**metrics, TERMINAL_ATTESTATION_KEY: marker["source_attestation"]}
    return TerminalResolution(TerminalKind.SUCCESS, metrics=metrics, marker=marker)
