"""Worker heartbeat: stream stage/progress to the HF artifact repo + TRL callbacks.

Monkeypatch contract: all patchable knobs live on the worker package (_w); locks/frozensets live
here. Tests that monkeypatch worker.<name> then call worker.heartbeat(...) take effect.
"""

from __future__ import annotations

import contextlib
import faulthandler
import json
import os
import sys
import threading
import time

from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import gpu_diagnostics

# Throttled to avoid blowing the 128/hr HF commit cap; terminal transitions are never throttled.
_HB_THROTTLED_STAGES = frozenset(
    {
        "rl_step",
        "sft_step",
        "model_prefetching",
        "sft_pretokenizing",
        "sft_initializing",
        "rl_initializing",
    }
)
_HB_SETUP_LIVENESS_STAGES = frozenset(
    {
        "model_prefetching",
        "sft_pretokenizing",
        "sft_initializing",
        "rl_initializing",
    }
)
_HB_TERMINAL_STAGES = frozenset({"done", "already_done"})
# 600s -> ~6 commits/hr; keeps stall detector alive without hitting the HF commit cap.
_HB_TERMINAL_ONLY_INTERVAL_S = 600.0

# Guards throttle bookkeeping; slow HF commit runs outside this lock so trainer callbacks don't
# block on the network.
_HB_LOCK = threading.Lock()
# Serializes HF commits to prevent reorder; each thread uploads its own per-call temp file.
_HB_UPLOAD_LOCK = threading.Lock()
# Terminal/error commits wait longer — no later heartbeat can repair them.
_HB_UPLOAD_LOCK_TIMEOUT_S = 30.0
_HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S = 120.0
# Monotonic claim counters; rollback guard uses SEQ not wall-clock (two threads can share same ts).
# One per throttle slot (real vs liveness) so a failed commit on one slot never blocks the other.
_HB_CLAIM_SEQ = 0
_HB_CLAIM_SEQ_LIVENESS = 0
# Claim seq of the last REAL payload that actually landed on HF. A due liveness commit skips its
# upload when a real payload CLAIMED AFTER the liveness claim has already landed (heartbeat.json
# keeps only the latest file, and the plane ignores liveness for progress — landing the liveness
# ping second would hide that window's only progress beat). Seq-based, not ts-based: exact under
# the locks and immune to frozen/mocked clocks. A real claimed BEFORE the liveness claim can't
# race this way — its slot update makes the liveness ping not-due in the first place.
_HB_LAST_REAL_LANDED_SEQ = 0

# Stages whose progress counter IS the training step; only these carry ``step`` in ticker-emitted
# heartbeats (model_prefetching's counter is downloaded BYTES — not a step).
_HB_STEP_PROGRESS_STAGES = frozenset({"rl_step", "sft_step"})

_STEP_GPU_DIAG_INTERVAL_S = 300.0
_SFT_HEARTBEAT_INTERVAL_S = 60.0


def _dump_thread_stacks(reason: str) -> None:
    with contextlib.suppress(Exception):
        print(
            f"[stall] {reason}: dumping all thread stacks, then yielding to the provider",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback(all_threads=True)


def _rollback_throttle_slot(my_claim: int, prev_last_upload: float, *, liveness_slot: bool) -> None:
    """Restore the throttle slot after a failed/abandoned upload, but only if this heartbeat still
    owns the latest claim on ITS slot — a newer heartbeat that bumped the slot after us must not be
    rolled back."""
    with _HB_LOCK:
        if liveness_slot:
            if my_claim == _HB_CLAIM_SEQ_LIVENESS:
                _w._HB_LAST_LIVENESS_UPLOAD = prev_last_upload
        elif my_claim == _HB_CLAIM_SEQ:
            _w._HB_LAST_UPLOAD = prev_last_upload


def heartbeat(stage: str, *, liveness: bool = False, **kw) -> bool:
    """Emit one heartbeat. Returns True when the payload was committed to the HF channel; False
    when it was throttled out, skipped, or the commit failed (callers that must re-send a pending
    progress advance — liveness_heartbeat's ticker — key off this)."""
    global _HB_CLAIM_SEQ, _HB_CLAIM_SEQ_LIVENESS, _HB_LAST_REAL_LANDED_SEQ
    ts = time.time()
    # liveness pings don't count as progress; provider stall detection skips them.
    if not liveness:
        _w._HB_LAST_PROGRESS_TS = ts
    payload = {
        "stage": stage,
        "ts": ts,
        "run_id": _w.RUN_ID,
        "mode": _w.RUN_MODE,
        "seed": _w.SEED,
        "attempt": _w.ATTEMPT,
        **({"liveness": True} if liveness else {}),
        **kw,
    }
    _dc = os.environ.get("RUNPOD_DC_ID") or ""
    if _dc:
        payload.setdefault("dc", _dc)
    snapshot = json.dumps(payload)
    with _HB_LOCK:
        now = time.time()
        # Liveness pings commit through their OWN throttle slot. They must never claim the real
        # slot: the 30s liveness ticker would otherwise always win the expiring slot ahead of the
        # rarer step-carrying heartbeats, and the plane deliberately ignores liveness for progress
        # — so a healthy slow-stepping run looks stalled, gets killed, and retries on whatever GPU
        # class has capacity (the 2026-07-05 H200→B200 mid-run switch that resumed a checkpoint
        # onto broken sm100 gradients started exactly this way).
        liveness_slot = liveness
        if stage in _HB_TERMINAL_STAGES or stage.startswith("error_"):
            upload_due = True  # never miss a terminal transition
            liveness_slot = False
        elif _w._HB_TERMINAL_ONLY:
            upload_due = (
                _w._HB_LAST_UPLOAD == 0.0
                or (now - _w._HB_LAST_UPLOAD) >= _HB_TERMINAL_ONLY_INTERVAL_S
            )
            liveness_slot = False
        else:
            throttled = stage in _HB_THROTTLED_STAGES
            interval_s = _w._HB_MIN_INTERVAL_S
            if stage in _HB_SETUP_LIVENESS_STAGES:
                interval_s = min(interval_s, _w._HB_SETUP_LIVENESS_INTERVAL_S)
            if not throttled:
                upload_due = True
            elif liveness:
                # A liveness commit is only worth an HF commit when NOTHING (real or liveness)
                # landed within the interval — a fresh real commit already proves the worker alive.
                last_any = max(_w._HB_LAST_UPLOAD, _w._HB_LAST_LIVENESS_UPLOAD)
                upload_due = (now - last_any) >= interval_s
            else:
                upload_due = (now - _w._HB_LAST_UPLOAD) >= interval_s
        real_seq_at_claim = _HB_CLAIM_SEQ
        if liveness_slot:
            prev_last_upload = _w._HB_LAST_LIVENESS_UPLOAD
            if upload_due:
                _HB_CLAIM_SEQ_LIVENESS += 1
                my_claim = _HB_CLAIM_SEQ_LIVENESS
                _w._HB_LAST_LIVENESS_UPLOAD = now
        else:
            prev_last_upload = _w._HB_LAST_UPLOAD
            if upload_due:
                _HB_CLAIM_SEQ += 1
                my_claim = _HB_CLAIM_SEQ
                _w._HB_LAST_UPLOAD = now
    landed = False
    if upload_due:
        critical = stage in _HB_TERMINAL_STAGES or stage.startswith("error_")
        lock_timeout = _HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S if critical else _HB_UPLOAD_LOCK_TIMEOUT_S
        if _HB_UPLOAD_LOCK.acquire(timeout=lock_timeout):
            try:
                skip_stale_liveness = False
                if liveness_slot:
                    # Both slots can come due in the same window; the poller reads only the LATEST
                    # heartbeat.json and ignores liveness for progress, so a liveness commit landing
                    # after a real one would erase that window's only progress beat. The real-landed
                    # marker is written under the upload lock we now hold, so this check is exact.
                    with _HB_LOCK:
                        skip_stale_liveness = real_seq_at_claim < _HB_LAST_REAL_LANDED_SEQ
                if not skip_stale_liveness:
                    up = f"/tmp/.hb-upload-{os.getpid()}-{threading.get_ident()}.json"
                    with open(up, "w") as f:
                        f.write(snapshot)
                    try:
                        committed = _w.hf_upload_file(up, "heartbeat.json")
                    finally:
                        with contextlib.suppress(OSError):
                            os.remove(up)
                    if committed is False:
                        # ``is False`` (not falsy) so a mock/None never trips the rollback.
                        _rollback_throttle_slot(
                            my_claim, prev_last_upload, liveness_slot=liveness_slot
                        )
                        print(f"HEARTBEAT upload failed; rolled back throttle slot for {stage}")
                    else:
                        landed = True
                        if not liveness_slot:
                            with _HB_LOCK:
                                _HB_LAST_REAL_LANDED_SEQ = max(_HB_LAST_REAL_LANDED_SEQ, my_claim)
            finally:
                _HB_UPLOAD_LOCK.release()
        else:
            _rollback_throttle_slot(my_claim, prev_last_upload, liveness_slot=liveness_slot)
            print(f"HEARTBEAT upload-lock busy >{lock_timeout}s; skipping commit for {stage}")
    print("HEARTBEAT", snapshot)
    return landed


def make_reward_heartbeat_callback():
    """Return a TRL callback that streams per-step reward to the HF heartbeat channel."""
    from transformers import TrainerCallback

    class _RewardHeartbeat(TrainerCallback):
        def __init__(self):
            self.reward_history = []
            self.last_gpu_diag_at = 0.0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            r = logs.get("reward")
            if r is None:
                return
            try:
                r = float(r)
            except (TypeError, ValueError):
                return
            self.reward_history.append(r)
            step = int(getattr(state, "global_step", len(self.reward_history)))
            payload = {
                "step": step,
                "reward": r,
                "reward_last": self.reward_history[-8:],
            }
            now = time.monotonic()
            if (
                self.last_gpu_diag_at == 0.0
                or now - self.last_gpu_diag_at >= _STEP_GPU_DIAG_INTERVAL_S
            ):
                payload["gpu"] = gpu_diagnostics()
                self.last_gpu_diag_at = now
            _w.heartbeat("rl_step", **payload)

    return _RewardHeartbeat()


def make_sft_heartbeat_callback():
    """Stream SFT trainer logs so a run is not silent between model load and completion."""
    from transformers import TrainerCallback

    class _SFTHeartbeat(TrainerCallback):
        def __init__(self):
            self.last_heartbeat_at = 0.0
            self.last_gpu_diag_at = 0.0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            now = time.monotonic()
            if self.last_heartbeat_at and now - self.last_heartbeat_at < _SFT_HEARTBEAT_INTERVAL_S:
                return
            self.last_heartbeat_at = now
            payload = {
                "step": int(getattr(state, "global_step", 0) or 0),
                "epoch": logs.get("epoch"),
                "loss": logs.get("loss"),
                "grad_norm": logs.get("grad_norm"),
                "learning_rate": logs.get("learning_rate"),
            }
            if (
                self.last_gpu_diag_at == 0.0
                or now - self.last_gpu_diag_at >= _STEP_GPU_DIAG_INTERVAL_S
            ):
                payload["gpu"] = gpu_diagnostics()
                self.last_gpu_diag_at = now
            _w.heartbeat("sft_step", **{k: v for k, v in payload.items() if v is not None})

    return _SFTHeartbeat()


_LIVENESS_TICK_S = 30.0
_STALL_DUMP_S = 1200.0


@contextlib.contextmanager
def liveness_heartbeat(stage, progress=None):
    """Emit liveness pings for ``stage`` while the wrapped block runs on the main thread.

    ``progress``: optional ``() -> float | None`` monotonic counter; advances emit a REAL heartbeat.
    Uses nvidia-smi-only diagnostics (main thread holds CUDA/allocator locks).
    """
    done = threading.Event()

    def _loop() -> None:
        # Track the last value the plane actually SAW (committed), not the last one observed: an
        # advance observed while the commit throttle window is closed must be re-sent every tick
        # until it lands, or a slow-stepping run (step time near the throttle interval) goes
        # progress-silent past the plane's stall window and gets killed while healthy.
        committed_val = None
        dumped = False
        while not done.wait(_LIVENESS_TICK_S):
            cur = None
            if progress is not None:
                with contextlib.suppress(Exception):
                    v = progress()
                    if v is not None:
                        cur = float(v)
            advanced = cur is not None and (committed_val is None or cur > committed_val)
            gpu = gpu_diagnostics(include_torch=False)
            if done.is_set():  # the wrapped call may have finished during nvidia-smi
                return
            # A progress advance is a REAL heartbeat; for per-step stages carry the counter as
            # ``step`` so the plane's stall detector both sees progress and can classify the run
            # as in-training (its step-gated stages ignore step-less pings).
            extra = (
                {"step": int(cur)} if (advanced and stage in _HB_STEP_PROGRESS_STAGES) else {}
            )
            landed = _w.heartbeat(stage, liveness=not advanced, gpu=gpu, **extra)
            if advanced and landed is not False:  # mocks returning None count as landed
                committed_val = cur
            last_progress = float(getattr(_w, "_HB_LAST_PROGRESS_TS", 0.0) or 0.0)
            if not dumped and last_progress and (time.time() - last_progress) > _STALL_DUMP_S:
                _dump_thread_stacks(f"{stage}: no progress for >{_STALL_DUMP_S:.0f}s")
                dumped = True

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        done.set()
        t.join(timeout=_HB_UPLOAD_LOCK_TIMEOUT_S)
