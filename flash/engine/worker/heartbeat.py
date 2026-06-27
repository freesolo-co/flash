"""Worker heartbeat channel: stream stage/progress to the HF artifact repo + TRL callbacks.

Each ``heartbeat()`` commits ``heartbeat.json`` to the run's HF artifact repo (throttled). The
control plane reads that file from HF to track the run and detect stalls.

Monkeypatch contract: ``heartbeat`` reads the run-scoped state (``RUN_ID``/``RUN_MODE``/``SEED``/
``ATTEMPT``) and the THREE patchable throttle knobs (``_HB_MIN_INTERVAL_S``/``_HB_LAST_UPLOAD``/
``_HB_TERMINAL_ONLY``, which live on the worker package) and calls ``hf_upload_file`` THROUGH the
worker package (``_w.<name>``) at CALL time, so tests that do
``monkeypatch.setattr(worker, "<name>", ...)`` then call ``worker.heartbeat(...)`` take effect.
The locks/frozensets/intervals that tests never patch live here and are re-exported for access.
"""

from __future__ import annotations

import contextlib
import faulthandler
import json
import os
import threading
import time

from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import gpu_diagnostics

# High-frequency stages whose HF UPLOAD is throttled to _HB_MIN_INTERVAL_S. Two kinds are
# high-frequency:
#   - "rl_step": the per-step training heartbeat (reward callback fires every optimizer step).
#   - the periodic setup pings ("model_prefetching", "sft_initializing", "rl_initializing"): each
#     runs on a side thread every 30s through a long phase (a cold snapshot_download can pull tens
#     of GB for ~40 min => ~80 commits, and disaggregated workers share one HF_REPO), so committing
#     every 30s risks the repo commit cap _HB_MIN_INTERVAL_S exists to avoid, so we throttle the
#     UPLOAD to hold commits well under the 128/hr cap. Terminal transitions are never throttled.
_HB_THROTTLED_STAGES = frozenset(
    {"rl_step", "model_prefetching", "sft_initializing", "rl_initializing"}
)
# Terminal transitions the control plane must never miss — always committed.
_HB_TERMINAL_STAGES = frozenset({"done", "already_done"})
# Even in terminal-only mode, emit a SLOW heartbeat at this cadence so the control plane's stall
# detector keeps seeing progress through a long training phase and doesn't false-stall the run.
# 600s -> ~6 commits/hr, far under the 128/hr cap.
_HB_TERMINAL_ONLY_INTERVAL_S = 600.0

# Serializes heartbeat.json writes and _HB_LAST_UPLOAD reads/updates. During GRPO,
# heartbeat() is called concurrently from the trainer thread (reward callback) and the
# checkpoint-upload daemon thread; without this lock two writers can interleave and
# truncate/garble heartbeat.json (and race _HB_LAST_UPLOAD).
_HB_LOCK = threading.Lock()
# Serializes the actual HF upload (a slow network commit) SEPARATELY from _HB_LOCK so the
# trainer's frequent local writes never block on the network. Without it, two heartbeat
# threads can upload heartbeat.json concurrently: a slower upload could land AFTER a newer
# one on HF (reorder), so this lock makes uploads strictly ordered.
_HB_UPLOAD_LOCK = threading.Lock()
# Acquire the upload lock with a BOUND: hf upload has no hard timeout, so a wedged upload could hold
# the lock and block the next heartbeat. Past the bound we skip the best-effort commit and roll the
# slot claim back so the throttle doesn't defer the NEXT (non-terminal) heartbeat, which re-commits
# fresher state — HF is the only durable channel, there is no local copy. Terminal/error_* commits
# are CRITICAL (no later heartbeat repairs them) so they wait much longer before giving up.
_HB_UPLOAD_LOCK_TIMEOUT_S = 30.0
_HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S = 120.0
# Monotonic counter: each upload slot-claim takes the next value. The rollback guard compares its
# captured claim to this counter to tell "my claim is still the latest" from "a newer claim landed",
# WITHOUT relying on the wall-clock ts — two heartbeats can read the same _HB_LAST_UPLOAD on a coarse
# clock, and equality on that ts would let one thread's rollback undo another thread's live claim.
_HB_CLAIM_SEQ = 0

_STEP_GPU_DIAG_INTERVAL_S = 300.0
_SFT_HEARTBEAT_INTERVAL_S = 60.0

# When a liveness daemon concludes its phase is wedged (progress stalled past its bound), dump every
# thread's stack for a root-cause trace, then stop pinging and let the PROVIDER's stall detection (it
# reads the HF heartbeats; their absence trips it) do the kill + retry. The dump goes to stderr, which
# the parent tees into console_<phase>.txt and the platform captures in the pod log. No separate
# re-armed watchdog: the dump fires on EVIDENCE of a stall, not a fixed timer.
def _dump_thread_stacks(reason: str) -> None:
    with contextlib.suppress(Exception):
        print(f"[stall] {reason}: dumping all thread stacks, then yielding to the provider", flush=True)
        faulthandler.dump_traceback(all_threads=True)


def heartbeat(stage: str, *, liveness: bool = False, **kw):
    global _HB_CLAIM_SEQ
    ts = time.time()
    # A REAL (non-liveness) heartbeat marks worker PROGRESS; a liveness ping only proves the worker is
    # alive. The provider stalls on the absence of real heartbeats (surface_heartbeat skips liveness
    # pings), and the liveness daemon dumps stacks once _HB_LAST_PROGRESS_TS goes stale.
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
    # The datacenter the worker actually landed in (RunPod serverless sets RUNPOD_DC_ID) — a
    # diagnostic so the control plane / logs show which region a run hit (the eager weight-cache fleet
    # already has a volume in every storage DC). Empty/absent on non-RunPod (instance) workers and
    # harmless; only emitted when present.
    _dc = os.environ.get("RUNPOD_DC_ID") or ""
    if _dc:
        payload.setdefault("dc", _dc)
    snapshot = json.dumps(payload)
    # The control plane reads heartbeat.json from HF, so the (throttled) upload below is the only
    # durable record — there is no worker-local copy to keep. _HB_LOCK guards just the throttle
    # bookkeeping (_HB_LAST_UPLOAD); the slow HF commit runs OUTSIDE it so the trainer's per-step
    # reward callback never blocks on the network behind the checkpoint daemon's commit.
    with _HB_LOCK:
        now = time.time()
        if stage in _HB_TERMINAL_STAGES or stage.startswith("error_"):
            upload_due = True  # never miss a terminal transition
        elif _w._HB_TERMINAL_ONLY:
            # Benchmark fan-out: keep commits far under the 128/hour cap, but still emit a SLOW
            # heartbeat (~every _HB_TERMINAL_ONLY_INTERVAL_S) so the control-plane stall detector
            # sees progress during a long training phase and doesn't false-stall the run.
            upload_due = (
                _w._HB_LAST_UPLOAD == 0.0
                or (now - _w._HB_LAST_UPLOAD) >= _HB_TERMINAL_ONLY_INTERVAL_S
            )
        else:
            throttled = stage in _HB_THROTTLED_STAGES
            upload_due = not throttled or (now - _w._HB_LAST_UPLOAD) >= _w._HB_MIN_INTERVAL_S
        prev_last_upload = _w._HB_LAST_UPLOAD
        if upload_due:
            _HB_CLAIM_SEQ += 1
            my_claim = _HB_CLAIM_SEQ  # collision-free identity for THIS slot claim
            _w._HB_LAST_UPLOAD = now  # claim the slot under the lock (throttle stays atomic)
    if upload_due:
        # Serialize the network commit under a SEPARATE lock so uploads can't reorder, and upload the
        # captured snapshot (via a private temp file, since hf_upload_file takes a path) rather than
        # re-reading p — which a newer heartbeat may already have overwritten between our slot-claim
        # and this upload. Acquire the lock with a BOUND (see _HB_UPLOAD_LOCK_TIMEOUT_S) so a wedged
        # upload holding the lock can't block this heartbeat indefinitely; on timeout skip the
        # best-effort commit and roll the slot claim back below so the throttle doesn't defer the next
        # heartbeat, which re-commits fresher state (HF is the only durable channel — no local copy).
        # Terminal/error commits are CRITICAL (no later heartbeat repairs them; error_* carries the
        # retriable flag) so they wait far longer before skipping (_HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S).
        critical = stage in _HB_TERMINAL_STAGES or stage.startswith("error_")
        lock_timeout = _HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S if critical else _HB_UPLOAD_LOCK_TIMEOUT_S
        if _HB_UPLOAD_LOCK.acquire(timeout=lock_timeout):
            try:
                up = f"/tmp/.hb-upload-{os.getpid()}-{threading.get_ident()}.json"
                with open(up, "w") as f:
                    f.write(snapshot)
                try:
                    committed = _w.hf_upload_file(up, "heartbeat.json")
                finally:
                    with contextlib.suppress(OSError):
                        os.remove(up)
                if committed is False:
                    # The best-effort HF commit failed (hf_upload_file swallows the error and reports
                    # False); HF is still stale. Roll the slot claim back exactly as the lock-timeout
                    # branch does, so the throttle doesn't defer the next retry and quiet_gate doesn't
                    # read the channel as fresh. ``is False`` (not falsy) so a mock/None never trips it.
                    with _HB_LOCK:
                        if my_claim == _HB_CLAIM_SEQ:  # ours is still the latest claim
                            _w._HB_LAST_UPLOAD = prev_last_upload
                    print(f"HEARTBEAT upload failed; rolled back throttle slot for {stage}")
            finally:
                _HB_UPLOAD_LOCK.release()
        else:
            # We claimed the upload slot above but never committed. Roll it back so the throttle
            # doesn't defer the NEXT commit by up to _HB_MIN_INTERVAL_S on the strength of an upload
            # that never happened, and so liveness_heartbeat's quiet_gate (which reads
            # _HB_LAST_UPLOAD) doesn't treat the channel as fresh while HF is still stale. Guard on the
            # claim SEQ (not the wall-clock ts, which two heartbeats can share on a coarse clock) so we
            # only undo OUR claim, never a newer one another thread landed meanwhile.
            with _HB_LOCK:
                if my_claim == _HB_CLAIM_SEQ:
                    _w._HB_LAST_UPLOAD = prev_last_upload
            print(f"HEARTBEAT upload-lock busy >{lock_timeout}s; skipping commit for {stage}")
    print("HEARTBEAT", snapshot)


def make_reward_heartbeat_callback():
    """A TRL/transformers callback that streams the per-step mean reward to the HF heartbeat
    channel, giving the worker a live RL signal (no pod log API) and recording a
    ``reward_history``. Built lazily so the module imports without transformers installed."""
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


# Liveness heartbeat for a long blocking call on the MAIN thread. A slow-but-LIVE phase — cold vLLM
# build / *Trainer.__init__, a multi-GB model prefetch, the first GRPO step (vLLM rollout warmup +
# backward, ~17 min observed) — emits no REAL heartbeat while it runs and would look like a hang.
# ``liveness_heartbeat`` runs a daemon that pings ``stage`` as a LIVENESS heartbeat: it advances the
# alive ts (so the console / a human see the worker is alive) but the provider SKIPS liveness pings
# for stall detection, so a wedge can't be masked by pinging and the worker needs no self-limit — the
# provider catches the stall (no real heartbeat) and does the kill + retry. For the operator, the
# daemon also dumps every thread's stack ONCE if no real progress lands for _STALL_DUMP_S (kept under
# the provider's training stall so the dump beats the kill). ``progress`` is an optional monotonic
# counter (e.g. downloaded bytes) whose advance the daemon reports as a REAL heartbeat — for a phase
# like prefetch whose only progress signal is a side effect; phases with their own per-step heartbeat
# (rl_step / sft_step) need nothing.
_LIVENESS_TICK_S = 30.0
_STALL_DUMP_S = 1200.0


@contextlib.contextmanager
def liveness_heartbeat(stage, progress=None):
    """Keep ``stage`` alive (liveness pings) while the wrapped block blocks the main thread.

    ``progress``: optional ``() -> float | None`` monotonic counter; each advance is reported as a
    REAL heartbeat (keeps the provider's stall timer fresh), everything else is a liveness ping the
    provider ignores. nvidia-smi-only diagnostics (the main thread owns the CUDA/allocator locks).
    Reaped with a bounded join so a wedged HF upload can't hang the worker at block exit.
    """
    done = threading.Event()

    def _loop() -> None:
        last_val = None
        dumped = False
        while not done.wait(_LIVENESS_TICK_S):
            made_progress = False
            if progress is not None:
                with contextlib.suppress(Exception):
                    v = progress()
                    if v is not None and (last_val is None or float(v) > last_val):
                        last_val, made_progress = float(v), True
            gpu = gpu_diagnostics(include_torch=False)
            if done.is_set():  # the wrapped call may have finished during nvidia-smi
                return
            _w.heartbeat(stage, liveness=not made_progress, gpu=gpu)
            # Operator root-cause trace: dump every thread's stack ONCE if no REAL progress lands for
            # the window (the provider does the kill + retry off the same stale-progress signal).
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
