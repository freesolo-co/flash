"""Unit tests for shared provider-poll helpers (flash.providers._poll)."""

from __future__ import annotations

import time

from flash.providers._poll import heartbeat_progress_ts


def test_heartbeat_progress_credits_post_launch_ts_and_is_fresh():
    now = time.time()
    launch = now - 1000
    # a real heartbeat from this instance, 100s ago -> credited as-is (own ts) and fresh
    ts, fresh = heartbeat_progress_ts(("rl", 5, now - 100), launch)
    assert fresh is True
    assert abs(ts - (now - 100)) < 2
    # future ts (worker clock ahead of the control plane) -> clamped down to now, still fresh
    ts, fresh = heartbeat_progress_ts(("rl", 5, now + 5000), launch)
    assert fresh is True
    assert abs(ts - now) < 2


def test_heartbeat_progress_pre_launch_leftover_is_not_fresh():
    now = time.time()
    launch = now - 1000
    # ts predates this instance's launch -> leftover from a prior attempt on the same seed path
    _ts, fresh = heartbeat_progress_ts(("rl", 5, launch - 1), launch)
    assert fresh is False
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 5000), launch)
    assert fresh is False


def test_heartbeat_progress_unknown_launch_counts_every_heartbeat_fresh():
    # Regression: started_ts is coerced to 0.0 when unknown -> we cannot date the heartbeat vs
    # launch, so a normal (slightly-past) heartbeat must still be FRESH. Otherwise a healthy
    # recovered worker with an unknown launch is stalled after SETUP_GRACE despite heartbeats.
    now = time.time()
    ts, fresh = heartbeat_progress_ts(("rl", 5, now - 30), 0.0)
    assert fresh is True
    assert abs(ts - (now - 30)) < 2
    # None launch is treated the same as 0.0 (unknown)
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 30), None)
    assert fresh is True


def test_heartbeat_progress_no_ts_is_not_fresh():
    now = time.time()
    launch = now - 1000
    for bad in (("rl", 5, None), ("rl", 5, "notnum"), None):
        ts, fresh = heartbeat_progress_ts(bad, launch)
        assert fresh is False
        assert abs(ts - now) < 2


def test_heartbeat_progress_rejects_other_attempt_even_when_ts_is_fresh():
    # The seed heartbeat path is shared across attempts: a prior attempt's worker still shutting down
    # can upload a heartbeat with ts > this attempt's launch. By ts alone it looks fresh, but it
    # belongs to a DIFFERENT attempt, so it must NOT satisfy this attempt's first-liveness. The worker
    # stamps attempt as a STRING (env var), the poller passes an int -> compare coerced to int.
    now = time.time()
    launch = now - 100
    # worker-stamped string attempt "0"; we are polling attempt=1 -> not fresh despite a post-launch ts
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 10, "0"), launch, current_attempt=1)
    assert fresh is False
    # int attempts mismatch likewise
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 10, 0), launch, current_attempt=1)
    assert fresh is False


def test_heartbeat_progress_same_attempt_string_vs_int_is_fresh():
    # Regression (Cursor HIGH): the worker stamps ``attempt`` as a string ("0") while the poller
    # passes an int (0). A naive `!=` would reject EVERY live heartbeat ("0" != 0) and spuriously
    # stall healthy workers. Coercing both to int makes the SAME attempt match across types.
    now = time.time()
    launch = now - 100
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 10, "0"), launch, current_attempt=0)
    assert fresh is True
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 10, "2"), launch, current_attempt=2)
    assert fresh is True


def test_heartbeat_progress_rejects_unstamped_when_attempt_known():
    now = time.time()
    launch = now - 100
    for attempt_field in ("", None):
        _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 10, attempt_field), launch, current_attempt=1)
        assert fresh is False
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 10), launch, current_attempt=1)
    assert fresh is False
    _ts, fresh = heartbeat_progress_ts(("rl", 5, now - 10, "0"), launch)
    assert fresh is True
