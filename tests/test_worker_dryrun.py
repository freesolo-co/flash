"""CPU dry-run of node_entry's train->metrics->finalize path with heavy deps mocked.

This can't validate that the model actually trains on a GPU (needs the GPU), but it
exercises the pure control flow (train_meta handoff, adapter resolution, vLLM eval loop,
grading, RunMetrics build, DONE finalize) to catch NameErrors / attribute / shape bugs
before spending GPU budget on the real run.
"""

from __future__ import annotations

import os

import flash.engine.worker.io.heartbeat as worker_heartbeat

# Disable HF I/O in node_entry helpers (they early-return when HF_REPO is empty).
os.environ["HF_REPO"] = ""
os.environ["PHASE"] = "rl"
os.environ["RUN_MODE"] = "rl"
os.environ["SEED"] = "0"


def test_on_policy_epochs_resolve_to_prompt_pool_passes():
    from flash.engine.plan.steps import on_policy_steps

    assert (
        on_policy_steps(
            epochs=2,
            prompt_count=33,
            prompts_per_step=16,
        )
        == 5
    )


def test_heartbeat_concurrent_calls_stay_safe(monkeypatch):
    """Regression: heartbeat() is called concurrently from the trainer reward callback and the
    checkpoint-upload daemon during GRPO. There is no worker-local heartbeat file (the control plane
    reads the HF copy), so each call must hand its HF upload a COMPLETE, valid JSON snapshot (never a
    truncated/interleaved one) and leave no upload temp files behind."""
    import contextlib
    import glob
    import json
    import threading as _threading

    import flash.engine.worker.io.heartbeat as ne

    monkeypatch.setattr(
        worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0
    )  # force every call through the upload path

    # Remove any stale upload temp files a prior failed run (or another process) left behind, so the
    # end-of-test "no temp files" assertion measures only THIS test's cleanup, not pre-existing cruft.
    for _stale in glob.glob("/tmp/.hb-upload-*"):
        with contextlib.suppress(OSError):
            os.remove(_stale)

    bad: list[Exception] = []

    def fake_upload(local_path, repo_subpath, required=False):
        # The bytes handed to the upload must be the caller's own complete snapshot.
        try:
            with open(local_path) as f:
                obj = json.load(f)
            assert obj["stage"] in ("rl_step", "checkpoint_uploaded")
        except Exception as e:  # truncated/garbled/missing -> a concurrency bug
            bad.append(e)

    monkeypatch.setattr(ne.hf_io, "hf_upload_file", fake_upload)

    errors: list[Exception] = []
    barrier = _threading.Barrier(8)

    def hammer(i: int):
        try:
            barrier.wait()
            for j in range(40):
                # vary stage so both the throttled and unthrottled branches run
                stage = "rl_step" if (i + j) % 2 else "checkpoint_uploaded"
                worker_heartbeat.heartbeat(stage, step=i * 1000 + j, payload="x" * 200)
        except Exception as e:
            errors.append(e)

    threads = [_threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert not bad, bad
    assert not glob.glob("/tmp/.hb-upload-*"), "upload temp files must be cleaned up"


def test_heartbeat_uploads_are_serialized_and_use_claimed_snapshot(monkeypatch):
    """Concurrent heartbeat uploads stay serialized and carry one claimed snapshot each.

    Unforced calls now deliberately skip while an upload is in flight: queueing their already-stale
    snapshots would block the caller and publish them after newer progress. The test therefore does
    not require all 120 calls to upload. It requires every upload that *does* claim a slot to be
    serialized and internally match the stage encoded by that caller's unique step.
    """
    import contextlib
    import glob
    import json
    import threading as _threading
    import time

    import flash.engine.worker.io.heartbeat as ne

    monkeypatch.setenv(
        "HF_REPO", ""
    )  # scoped to this test (auto-restored), not a raw os.environ write

    # Clear stale upload temp files up front so the end-of-test "no temp files" assertion isn't a false
    # failure against cruft from a prior failed run or another process on the same host.
    for _stale in glob.glob("/tmp/.hb-upload-*"):
        with contextlib.suppress(OSError):
            os.remove(_stale)

    inflight = 0
    max_inflight = 0
    seen_steps: set[int] = set()
    mismatches: list[tuple] = []
    guard = _threading.Lock()

    # Force every call through the upload path: no throttling.
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)

    def fake_upload(local_path, repo_subpath, required=False):
        nonlocal inflight, max_inflight
        with guard:
            inflight += 1
            max_inflight = max(max_inflight, inflight)
        # read the bytes this upload was handed (the captured snapshot temp file)
        with open(local_path) as f:
            obj = json.load(f)
        time.sleep(0.001)  # widen the window for an overlap to be observed if it can happen
        with guard:
            step = obj["step"]
            seen_steps.add(step)
            producer, offset = divmod(step, 1000)
            expected_stage = "rl_step" if (producer + offset) % 2 else "ckpt"
            if obj["stage"] != expected_stage:
                mismatches.append((expected_stage, obj))
            inflight -= 1

    monkeypatch.setattr(ne.hf_io, "hf_upload_file", fake_upload)

    barrier = _threading.Barrier(6)
    errors: list[Exception] = []

    def hammer(i: int):
        try:
            barrier.wait()
            for j in range(20):
                step = i * 1000 + j
                worker_heartbeat.heartbeat("rl_step" if (i + j) % 2 else "ckpt", step=step)
        except Exception as e:
            errors.append(e)

    threads = [_threading.Thread(target=hammer, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert not mismatches, mismatches
    # serialized: at most one upload in flight at any moment
    assert max_inflight == 1, f"uploads overlapped (max_inflight={max_inflight})"
    # at least one call claimed the upload slot; concurrent unforced callers may be dropped while
    # that upload is in flight, but no caller can invent a step outside the 120 submitted above.
    assert 1 <= len(seen_steps) <= 6 * 20
    # the captured-snapshot temp files are cleaned up
    assert not glob.glob("/tmp/.hb-upload-*"), "upload temp files must be cleaned up"
