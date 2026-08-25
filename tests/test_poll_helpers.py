"""Unit tests for shared provider-poll helpers (flash.providers._lifecycle.instances.poll)."""

from __future__ import annotations

import sys
import time

import pytest

from flash.providers._lifecycle.instances.poll import (
    _attempt_int,
    _format_heartbeat,
    heartbeat_progress_ts,
)


def test_attempt_int_rejects_oversized_ascii_decimal():
    if not hasattr(sys, "set_int_max_str_digits"):
        assert _attempt_int("9" * 5000) is None
        return
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        assert int("9" * 5000) > 0
        assert _attempt_int("9" * 5000) is None
    finally:
        sys.set_int_max_str_digits(previous)


def test_heartbeat_progress_credits_post_launch_ts_and_is_fresh():
    now = time.time()
    launch = now - 1000
    # a real heartbeat from this instance, 100s ago -> credited as-is (own ts) and fresh
    ts, fresh = heartbeat_progress_ts(("rl", 5, now - 100, 0), launch, current_attempt=0)
    assert fresh is True
    assert abs(ts - (now - 100)) < 2
    # future ts (worker clock ahead of the control plane) -> clamped down to now, still fresh
    ts, fresh = heartbeat_progress_ts(("rl", 5, now + 5000, 0), launch, current_attempt=0)
    assert fresh is True
    assert abs(ts - now) < 2


def test_heartbeat_progress_pre_launch_leftover_is_not_fresh():
    now = time.time()
    launch = now - 1000
    # ts predates this instance's launch -> leftover from a prior attempt on the same seed path
    _ts, fresh = heartbeat_progress_ts(("rl", 5, launch - 1, 0), launch, current_attempt=0)
    assert fresh is False
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 5000, 0), launch, current_attempt=0)
    assert fresh is False


def test_heartbeat_progress_rejects_unknown_launch_identity():
    now = time.time()
    for launch in (0.0, None):
        _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 30, 0), launch, current_attempt=0)
        assert fresh is False


def test_heartbeat_progress_no_ts_is_not_fresh():
    now = time.time()
    launch = now - 1000
    for bad in (("rl", 5, None), ("rl", 5, "notnum"), None):
        ts, fresh = heartbeat_progress_ts(bad, launch)
        assert fresh is False
        assert abs(ts - now) < 2


def test_heartbeat_progress_rejects_other_attempt_even_when_ts_is_fresh():
    # a prior attempt can upload a post-launch heartbeat, but an exact canonical attempt mismatch must
    # prevent it from satisfying current liveness.
    now = time.time()
    launch = now - 100
    # canonical integer attempts mismatch
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 10, 0), launch, current_attempt=1)
    assert fresh is False


def test_heartbeat_progress_rejects_string_attempt_identity():
    now = time.time()
    launch = now - 100
    for heartbeat_attempt, current_attempt in (("0", 0), ("2", 2)):
        _ts, fresh = heartbeat_progress_ts(
            ("rl", 5, now - 10, heartbeat_attempt),
            launch,
            current_attempt=current_attempt,
        )
        assert fresh is False


def test_heartbeat_progress_rejects_unstamped_when_attempt_known():
    now = time.time()
    launch = now - 100
    for attempt_field in ("", None):
        _ts, fresh = heartbeat_progress_ts(
            ("rl", 5, now - 10, attempt_field), launch, current_attempt=1
        )
        assert fresh is False
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 10), launch, current_attempt=1)
    assert fresh is False
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 10, 0), launch)
    assert fresh is False


@pytest.mark.parametrize(
    ("heartbeat_attempt", "current_attempt"),
    [(0, 0), (7, 7)],
)
def test_heartbeat_progress_accepts_only_canonical_attempt_identities(
    heartbeat_attempt, current_attempt
):
    now = time.time()
    _ts, fresh = heartbeat_progress_ts(
        ("rl", 5, now - 1, heartbeat_attempt), now - 10, current_attempt=current_attempt
    )
    assert fresh is True


@pytest.mark.parametrize(
    "malformed_attempt",
    [True, False, 1.0, -1, "-1", "+1", " 1", "1 ", "", chr(0x661), chr(0xFF11), object()],
)
def test_heartbeat_progress_rejects_malformed_heartbeat_attempt(malformed_attempt):
    now = time.time()
    _ts, fresh = heartbeat_progress_ts(
        ("rl", 5, now - 1, malformed_attempt), now - 10, current_attempt=1
    )
    assert fresh is False


@pytest.mark.parametrize(
    "malformed_attempt",
    [True, False, 1.0, -1, "-1", "+1", " 1", "1 ", "", chr(0x661), chr(0xFF11), object()],
)
def test_heartbeat_progress_rejects_malformed_current_attempt(malformed_attempt):
    now = time.time()
    _ts, fresh = heartbeat_progress_ts(
        ("rl", 5, now - 1, 1), now - 10, current_attempt=malformed_attempt
    )
    assert fresh is False


def test_format_heartbeat_renders_the_child_tail_for_a_stalled_worker():
    # the whole point of carrying the tail: a stalled run's log must show what the child said.
    msg = _format_heartbeat(
        {
            "stage": "opd_step",
            "liveness": True,
            "step": 0,
            "child_tail": ["Ray placement group pending", "waiting for 3 CPUs"],
        }
    )
    assert "stage=opd_step" in msg
    assert "child output before the stall" in msg
    assert "Ray placement group pending" in msg
    assert "waiting for 3 CPUs" in msg
    # oldest first, so the last line rendered is the child's most recent word
    assert msg.index("Ray placement group pending") < msg.index("waiting for 3 CPUs")


def test_format_heartbeat_renders_how_long_the_child_has_been_silent():
    # the counter is the slow-vs-stuck signal, and the streamed log is where a stall is diagnosed:
    # carrying it to the plane and not rendering it leaves the operator comparing tails by eye.
    msg = _format_heartbeat(
        {
            "stage": "opd_step",
            "step": 0,
            "child_tail": ["Started a local Ray instance"],
            "child_tail_silent_ticks": 7,
        }
    )
    assert "unchanged for 7 ticks" in msg
    assert "Started a local Ray instance" in msg


def test_format_heartbeat_distinguishes_a_talking_child_from_a_silent_one():
    # 0 is the load-bearing value: it says the child is still producing output, so a rendering that
    # only shows a nonzero count reads a talking child as an unannotated stall.
    talking = _format_heartbeat(
        {"stage": "opd_step", "child_tail": ["loading weights"], "child_tail_silent_ticks": 0}
    )
    assert "still producing output" in talking
    assert "unchanged for" not in talking
    # singular, because "unchanged for 1 ticks" in an operator-facing log is a typo with a version
    one = _format_heartbeat(
        {"stage": "opd_step", "child_tail": ["loading weights"], "child_tail_silent_ticks": 1}
    )
    assert "unchanged for 1 tick," not in one
    assert "unchanged for 1 tick:" in one


def test_format_heartbeat_ignores_a_malformed_silent_tick_count():
    # a bad counter must cost only the annotation, never the tail it annotates.
    for bad in ("3", -1, 2.5, True, None, {"a": 1}):
        msg = _format_heartbeat(
            {"stage": "opd_step", "child_tail": ["ray init kwargs"], "child_tail_silent_ticks": bad}
        )
        assert "child output before the stall" in msg
        assert "ray init kwargs" in msg
        assert "unchanged for" not in msg
        assert "still producing output" not in msg


def test_format_heartbeat_is_unchanged_when_no_child_tail_is_present():
    hb = {"stage": "opd_step", "liveness": True, "step": 4, "loss": 1.5}
    assert "child output" not in _format_heartbeat(hb)


def test_format_heartbeat_renders_the_opd_per_step_discard_count():
    # opd writes its step metrics as top-level heartbeat fields rather than the grpo-shaped
    # metrics_last backlog, so the streamed log is the surface that carries them. a count missing
    # from this allowlist is silently dropped next to the rate it belongs with.
    msg = _format_heartbeat(
        {"stage": "opd_step", "step": 3, "truncation_rate": 0.5, "discarded_rollouts": 4}
    )

    assert "truncation_rate=0.500" in msg
    assert "discarded_rollouts=4" in msg


def test_format_heartbeat_ignores_a_malformed_child_tail():
    # this payload crosses a process boundary as json; a bad one must not break the whole line.
    for bad in ("not a list", 17, {"a": 1}, [], [None, 3, ""], None):
        msg = _format_heartbeat({"stage": "opd_step", "child_tail": bad})
        assert "stage=opd_step" in msg
        assert "child output" not in msg


def test_format_heartbeat_neutralizes_control_chars_in_the_child_tail():
    # verl children emit ansi progress bars; raw control bytes must not reach the streamed log.
    msg = _format_heartbeat({"stage": "opd_step", "child_tail": ["a\x1b[2Kb\rc"]})
    assert "child output before the stall" in msg
    assert "\x1b" not in msg
    assert "\r" not in msg


def test_format_heartbeat_escapes_newlines_in_checkpoint_failure():
    injected = "Permission denied\nworker: stage=done step=999"

    msg = _format_heartbeat(
        {
            "stage": "checkpoint_upload_failed",
            "checkpoint_failure": {"step": 50, "operation": "resume", "error": injected},
        }
    )

    assert "\n" not in msg
    assert "checkpoint_error=Permission denied\\nworker: stage=done step=999" in msg


@pytest.mark.parametrize(
    "heartbeat",
    [
        {"stage": "done\nworker: stage=done step=999"},
        {"stage": "ok", "gpu": {"nvidia_smi": "gpu\nworker: stage=done step=999"}},
    ],
)
def test_format_heartbeat_summary_fields_cannot_forge_a_worker_record(heartbeat):
    msg = _format_heartbeat(heartbeat)

    assert "\n" not in msg
    assert "\\nworker: stage=done step=999" in msg
