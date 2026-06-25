"""Unit tests for shared provider-poll helpers (flash.providers._poll)."""

from __future__ import annotations

import time

from flash.providers._poll import SETUP_HEARTBEAT_STAGES, heartbeat_progress_ts, is_training_stage


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


def test_is_training_stage_excludes_setup_init_and_error_stages():
    """is_training_stage() means training-REACHED (training-or-later), deliberately broad: any stage
    that is NOT setup/cold-start AND NOT an error_* crash. So the training steps AND every post-training
    stage (sft_trained/done/checkpoint_*) count -- all mean the worker became productive. The CORRECTNESS
    invariant is the EXCLUSIONS: every cold-start/setup stage the worker emits before the first step
    (incl. model_prefetched and the *_initializing stages) and every error_* crash stage must be
    excluded, or reached_training_now() would suppress the region quarantine for a pre-training infra
    fault. Taxonomy guard: a new PRE-training worker stage missing from SETUP_HEARTBEAT_STAGES would
    regress quarantine, so it must also be added there."""
    # Training steps AND post-training/terminal stages -> training-reached (worker became productive).
    for stage in ("sft_step", "rl_step", "sft_trained", "rl_trained", "checkpoint_uploaded", "done"):
        assert is_training_stage(stage) is True
    # Cold-start / setup stages -> NOT training (every member of the shared set, plus model_prefetched).
    assert "model_prefetched" in SETUP_HEARTBEAT_STAGES  # prefetch_model() emits it pre-training
    for stage in SETUP_HEARTBEAT_STAGES:
        assert is_training_stage(stage) is False
    # error_* crash stages and empty/None -> NOT training.
    for stage in ("error_sft", "error_rl", "", None):
        assert is_training_stage(stage) is False
