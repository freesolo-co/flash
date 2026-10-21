"""The one terminal-artifact protocol shared by live polling and recovery.

These pin the two things that diverged before the protocol existed: how an unverifiable marker is
classified, and how long one resolution may spend observing artifacts.
"""

from __future__ import annotations

import json

import pytest

from flash.providers._lifecycle.instances.terminal_artifacts import (
    AttemptIdentity,
    ProbeBudget,
    TerminalKind,
    decode_terminal_marker,
    read_within,
    resolve_terminal_artifacts,
)
from tests._helpers.source_snapshot import valid_source_snapshot

IDENTITY = AttemptIdentity(run_id="run-1", attempt=0, launch_floor=100.0)
SOURCE_SNAPSHOT = valid_source_snapshot()


def _marker(**overrides) -> str:
    marker = {
        "attempt": 0,
        "error": "",
        "ok": True,
        "retriable": False,
        "run_id": "run-1",
        "ts": 150.0,
    }
    marker.update(overrides)
    return json.dumps(marker)


def _resolve(marker_raw, metrics_raw, *, budget=None, **kwargs):
    return resolve_terminal_artifacts(
        IDENTITY,
        read_marker=lambda: marker_raw,
        read_metrics=lambda: metrics_raw,
        budget=budget or ProbeBudget(tries=0, wait_s=0.0),
        **kwargs,
    )


def test_absent_marker_is_absence_not_failure():
    """No marker means the attempt has not settled. It must never look terminal."""
    resolution = _resolve(None, None)
    assert resolution.kind is TerminalKind.ABSENT


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        json.dumps({"attempt": 0, "ok": True}),  # missing required keys
        _marker(run_id="another-run"),  # speaks for a different run
        _marker(attempt=1),  # speaks for a different attempt
        _marker(ts=50.0),  # predates the launch it claims to end
    ],
)
def test_unverifiable_marker_fails_closed_and_is_never_absence(raw):
    """An artifact that cannot be tied to this attempt is its OWN kind, not silence.

    Recovery used to swallow exactly these into ``None``, so the same bytes told two different
    stories depending on which layer observed them. Reading it as absence is the regression this
    guards: absence means "nothing settled yet", which invites continued waiting on evidence that
    will never become valid.
    """
    resolution = _resolve(raw, json.dumps({"train_tokens": 1}))
    assert resolution.kind is TerminalKind.UNVERIFIABLE
    assert resolution.metrics is None
    assert resolution.marker is None


def test_failure_marker_is_terminal_without_reading_metrics():
    resolution = _resolve(_marker(ok=False, error="worker exploded"), None)
    assert resolution.kind is TerminalKind.FAILURE
    assert resolution.marker["error"] == "worker exploded"


def test_success_marker_with_metrics_is_success():
    resolution = _resolve(_marker(), json.dumps({"train_tokens": 4096}))
    assert resolution.kind is TerminalKind.SUCCESS
    assert resolution.metrics == {"train_tokens": 4096}


def test_success_marker_carries_trusted_source_attestation_into_metrics():
    from flash.snapshot.archive import TERMINAL_ATTESTATION_KEY, source_attestation

    attestation = source_attestation(SOURCE_SNAPSHOT, run_id="run-1", attempt=0)
    resolution = _resolve(
        _marker(source_attestation=attestation),
        json.dumps({"train_tokens": 4096}),
    )
    assert resolution.kind is TerminalKind.SUCCESS
    assert resolution.metrics[TERMINAL_ATTESTATION_KEY] == attestation


@pytest.mark.parametrize(
    ("metrics_raw", "unparseable"),
    [(None, False), ("{truncated", True), ("[]", True)],
)
def test_success_marker_without_readable_metrics_is_pending_not_failure(metrics_raw, unparseable):
    """The marker already authorized completion, so an unreadable second upload is a visibility gap.

    Classifying this as failure would tear down an attempt that already finished its paid work.
    """
    resolution = _resolve(_marker(), metrics_raw)
    assert resolution.kind is TerminalKind.PENDING
    assert resolution.metrics is None
    assert resolution.metrics_unparseable is unparseable


def test_marker_deadline_bound_rejects_a_timestamp_past_the_caller_bound():
    resolution = _resolve(_marker(ts=900.0), None, marker_deadline_at=200.0)
    assert resolution.kind is TerminalKind.UNVERIFIABLE


def test_decode_rejects_bool_disguised_as_attempt_number():
    """``True == 1`` in Python, so a bool must not satisfy an integer attempt match."""
    with pytest.raises(ValueError, match="invalid terminal marker identity"):
        decode_terminal_marker(
            _marker(ok=True, retriable=False, attempt=True),
            run_id="run-1",
            attempt=1,
            launch_floor=100.0,
        )


def test_one_budget_is_shared_across_both_reads(monkeypatch):
    """The marker and metrics reads share ONE window; the second cannot start a fresh allowance.

    Recovery previously computed a window for the marker and then another for metrics, making the
    real ceiling their sum. With a shared absolute cutoff, a marker read that exhausts the window
    leaves metrics no retries at all.
    """
    import flash.providers._lifecycle.instances.terminal_artifacts as ta

    clock = {"now": 1000.0}
    monkeypatch.setattr(ta.time, "time", lambda: clock["now"])
    monkeypatch.setattr(ta.time, "sleep", lambda s: clock.__setitem__("now", clock["now"] + s))

    metrics_reads = {"n": 0}

    def read_metrics():
        metrics_reads["n"] += 1
        return

    # a 10s window: the marker read burns half of it before metrics is ever consulted.
    budget = ProbeBudget(tries=10, wait_s=5.0, cutoff_at=clock["now"] + 10.0)
    marker_reads = {"n": 0}

    def read_marker():
        marker_reads["n"] += 1
        # invisible on the first read, then it surfaces having spent 5s of the shared window
        return None if marker_reads["n"] <= 1 else _marker()

    resolution = resolve_terminal_artifacts(
        IDENTITY,
        read_marker=read_marker,
        read_metrics=read_metrics,
        budget=budget,
        marker_wait_message="waiting for the terminal attempt marker",
    )

    assert resolution.kind is TerminalKind.PENDING
    # the marker spent 5s of the shared 10s window, so metrics inherits only the REMAINDER and the
    # whole resolution stops at the cutoff. under the nested budgets this replaces, metrics would
    # have started a FRESH 10s window at 1005 and run past 1010 -- the sum, not the cutoff.
    assert clock["now"] == 1010.0
    assert metrics_reads["n"] == 1


def test_read_within_stops_at_the_cutoff_rather_than_the_try_count(monkeypatch):
    import flash.providers._lifecycle.instances.terminal_artifacts as ta

    clock = {"now": 1000.0}
    monkeypatch.setattr(ta.time, "time", lambda: clock["now"])
    monkeypatch.setattr(ta.time, "sleep", lambda s: clock.__setitem__("now", clock["now"] + s))

    reads = {"n": 0}

    def read():
        reads["n"] += 1
        return

    # 100 tries would wait ~500s; the 6s cutoff bounds it to one 5s sleep plus a 1s clipped one
    read_within(read, ProbeBudget(tries=100, wait_s=5.0, cutoff_at=clock["now"] + 6.0))
    assert clock["now"] == 1006.0
    assert reads["n"] < 100
