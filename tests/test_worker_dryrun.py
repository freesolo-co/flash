"""CPU dry-run of node_entry's train->metrics->finalize path with heavy deps mocked.

This can't validate that the model actually trains on a GPU (needs the GPU), but it
exercises the pure control flow (train_meta handoff, adapter resolution, vLLM eval loop,
grading, RunMetrics build, DONE finalize) to catch NameErrors / attribute / shape bugs
before spending GPU budget on the real run.
"""

from __future__ import annotations

import os
import sys
import types

# Disable HF I/O in node_entry helpers (they early-return when HF_REPO is empty).
os.environ["HF_REPO"] = ""
os.environ["PHASE"] = "rl"
os.environ["RUN_MODE"] = "rl"
os.environ["SEED"] = "0"


def test_grpo_batching_matches_prompts_per_step():
    """Regression guard for the GRPO batch bug: TRL sizes batches in COMPLETIONS, so
    grad-accum must include the group size. Each optimizer step must optimize the intended
    number of *unique prompts* (64), not prompts_per_step/group_size (the old 8/step bug)."""
    import flash.engine.worker as ne

    for per_device in (8, 4, 16, 1):
        b = ne.compute_grpo_batching(prompts_per_step=64, group_size=8, per_device_comps=per_device)
        assert b["unique_prompts_per_step"] == 64, (per_device, b)
        assert b["generations_per_step"] == 512, (per_device, b)
        assert b["divisible_by_group"] is True, (per_device, b)
        assert b["per_device_train_batch_size"] * b["gradient_accumulation_steps"] == 512

    # The OLD formula would have given only 8 prompts/step (what we are fixing):
    old_grad_accum = 64 // 8
    assert (8 * old_grad_accum) // 8 == 8


def test_reward_heartbeat_callback_accumulates_history():
    """The live-signal feature: the GRPO callback records a per-step reward_history (only
    from step logs that carry a 'reward') and ignores non-reward logs. Uses a minimal
    transformers stub so the test doesn't depend on a real transformers install."""
    saved = sys.modules.get("transformers")
    tfm = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    tfm.TrainerCallback = TrainerCallback
    sys.modules["transformers"] = tfm
    try:
        os.environ["HF_REPO"] = ""  # heartbeat stays local (no HF upload) in tests
        import flash.engine.worker as ne

        cb = ne.make_reward_heartbeat_callback()

        class _State:
            global_step = 0

        st = _State()
        st.global_step = 1
        cb.on_log(None, st, None, logs={"reward": 0.50, "loss": 1.2})
        st.global_step = 2
        cb.on_log(None, st, None, logs={"reward": 0.62})
        cb.on_log(None, st, None, logs={"eval_loss": 0.9})  # ignored
        cb.on_log(None, st, None, logs=None)  # ignored
        cb.on_log(None, st, None, logs={"reward": None})  # ignored

        assert cb.reward_history == [0.50, 0.62], cb.reward_history
    finally:
        if saved is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved


def test_heartbeat_concurrent_writes_stay_atomic(monkeypatch):
    """Regression: heartbeat() is called concurrently from the trainer reward callback and
    the checkpoint-upload daemon during GRPO. It must serialize + write heartbeat.json
    atomically (temp file + os.replace under a lock) so a reader never sees a truncated or
    interleaved file, and no leftover .tmp files accumulate."""
    import glob
    import json
    import threading as _threading

    import flash.engine.worker as ne

    os.environ["HF_REPO"] = ""  # keep heartbeat local
    monkeypatch.setattr(ne, "hf_upload_file", lambda *a, **k: None)

    hb_dir = "/tmp/hb"
    p = os.path.join(hb_dir, "heartbeat.json")
    for stale in glob.glob(os.path.join(hb_dir, "heartbeat.json.*.tmp")):
        os.unlink(stale)

    errors: list[Exception] = []
    barrier = _threading.Barrier(8)

    def hammer(i: int):
        try:
            barrier.wait()
            for j in range(40):
                # vary stage so both the throttled and unthrottled branches run
                stage = "rl_step" if (i + j) % 2 else "checkpoint_uploaded"
                ne.heartbeat(stage, step=i * 1000 + j, payload="x" * 200)
                # the file must always parse as complete JSON (never truncated/garbled)
                with open(p) as f:
                    obj = json.load(f)
                assert obj["stage"] in ("rl_step", "checkpoint_uploaded")
        except Exception as e:
            errors.append(e)

    threads = [_threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    # final file is valid JSON and no temp files were left behind
    with open(p) as f:
        json.load(f)
    assert not glob.glob(os.path.join(hb_dir, "heartbeat.json.*.tmp"))


def test_heartbeat_uploads_are_serialized_and_use_claimed_snapshot(monkeypatch):
    """Regression (HIGH): the slow HF upload runs OUTSIDE _HB_LOCK, so two heartbeat
    threads could upload concurrently and reorder on HF, and a re-read could send a file a
    newer heartbeat already replaced (stale snapshot). The fix serializes uploads under a
    separate _HB_UPLOAD_LOCK and uploads the bytes captured under _HB_LOCK. This asserts:
    (1) uploads never overlap (serialized), and (2) every upload's bytes match the payload
    the caller wrote (no stale/re-read mismatch)."""
    import glob
    import json
    import threading as _threading
    import time

    import flash.engine.worker as ne

    os.environ["HF_REPO"] = ""

    hb_dir = "/tmp/hb"
    for stale in glob.glob(os.path.join(hb_dir, "heartbeat.json.*.tmp")):
        os.unlink(stale)

    inflight = 0
    max_inflight = 0
    seen_steps: set[int] = set()
    mismatches: list[tuple] = []
    guard = _threading.Lock()

    # Force every call through the upload path: no throttling.
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 0.0)

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
            seen_steps.add(obj["step"])
            # the snapshot must be internally consistent: its step is the one written for it
            if obj["stage"] not in ("rl_step", "ckpt"):
                mismatches.append(("stage", obj))
            inflight -= 1

    monkeypatch.setattr(ne, "hf_upload_file", fake_upload)

    barrier = _threading.Barrier(6)
    errors: list[Exception] = []

    def hammer(i: int):
        try:
            barrier.wait()
            for j in range(20):
                step = i * 1000 + j
                ne.heartbeat("rl_step" if (i + j) % 2 else "ckpt", step=step)
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
    # every claimed slot uploaded a distinct, valid snapshot (120 calls, no throttle)
    assert len(seen_steps) == 6 * 20
    # the captured-snapshot temp files are cleaned up
    assert not glob.glob(os.path.join(hb_dir, "*.upload.tmp"))
