"""Worker heartbeat channel: stream stage/progress to the HF artifact repo + TRL callbacks.

Each ``heartbeat()`` writes ``/tmp/hb/heartbeat.json`` locally (always) and commits it to the run's
HF artifact repo (throttled). The control plane reads that file to track the run and detect stalls.

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

# High-frequency stages whose HF UPLOAD is throttled to _HB_MIN_INTERVAL_S (the local write +
# watchdog re-arm still happen every call). Two kinds are high-frequency:
#   - "rl_step": the per-step training heartbeat (reward callback fires every optimizer step).
#   - the periodic setup pings ("model_prefetching", "sft_initializing", "rl_initializing"): each
#     runs on a side thread every 30s through a long phase (a cold snapshot_download can pull tens
#     of GB for ~40 min => ~80 commits, and disaggregated workers share one HF_REPO), so committing
#     every 30s risks the repo commit cap _HB_MIN_INTERVAL_S exists to avoid. Throttling the UPLOAD
#     (not the emit) keeps the local watchdog re-armed every 30s while holding commits well under the
#     128/hr cap. Terminal transitions are never throttled (always committed).
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
# the lock and block the next heartbeat. Past the bound we skip the best-effort commit (the local
# write already landed; the next heartbeat re-commits). Terminal/error_* commits are CRITICAL — no
# later heartbeat repairs them — so they wait much longer before giving up, but still bounded.
_HB_UPLOAD_LOCK_TIMEOUT_S = 30.0
_HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S = 120.0

_STEP_GPU_DIAG_INTERVAL_S = 300.0
_SFT_HEARTBEAT_INTERVAL_S = 60.0

# Stall watchdog: a faulthandler that dumps every thread's stack and exits when NO heartbeat lands
# for the window (re-armed on every heartbeat, so it only fires on a true hang) — turning a silent
# wedge into an uploaded stack trace. DEFAULT-ON; this is the steady-state (training) window, with
# the wider _STALL_STARTUP_GRACE_S used through cold start. Safe because every long phase keeps
# heartbeating (init/prefetch/step liveness). Set FLASH_STALL_FAULTHANDLER_S=0 in [worker_env] to
# disable, or lower it to localize a known hang.
_STALL_FAULTHANDLER_S = 2400
with contextlib.suppress(Exception):
    _STALL_FAULTHANDLER_S = int(os.environ.get("FLASH_STALL_FAULTHANDLER_S", "2400") or 2400)

# Cold start (prefetch, weight load, vLLM/trainer build, the silent full-dataset tokenize) can run
# many minutes with only coarse pings, so until the first training step the watchdog uses this wider
# window for every arm. Kept STRICTLY LONGER than the providers' SETUP_GRACE_S (3000s) so a stuck
# setup trips the provider's RETRIABLE path before this (exit=True) hard-fails it (asserted in tests).
_STALL_STARTUP_GRACE_S = 3600
# Stages that mean real TRAINING has begun (vs. a cold-start/setup ping). The first one tightens the
# watchdog from the wide setup grace to the steady-state window. Mirrors the providers'
# SETUP_HEARTBEAT_STAGES boundary: the cold-start stages there are setup, sft_step/rl_step are not.
_STEP_STAGES = frozenset({"rl_step", "sft_step"})
# False until the first per-step training heartbeat; until then every arm gets the startup grace.
_SAW_STEP_HEARTBEAT = False


def _rearm_stall_faulthandler(stage: str = "") -> None:
    global _SAW_STEP_HEARTBEAT
    if _STALL_FAULTHANDLER_S <= 0:
        return
    if stage in _STEP_STAGES:
        _SAW_STEP_HEARTBEAT = True
    # Until the first per-step training heartbeat the run is still in setup (prefetch, weight load,
    # vLLM/trainer build, full-dataset render/tokenize) — phases that can run many minutes with only
    # coarse (or no) progress pings — so widen to the startup grace so a healthy-but-slow setup
    # can't trip the watchdog. Once training steps are flowing, tighten to the configured interval.
    window = _STALL_FAULTHANDLER_S if _SAW_STEP_HEARTBEAT else max(_STALL_FAULTHANDLER_S, _STALL_STARTUP_GRACE_S)
    with contextlib.suppress(Exception):
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(window, exit=True)


def heartbeat(stage: str, **kw):
    payload = {
        "stage": stage,
        "ts": time.time(),
        "run_id": _w.RUN_ID,
        "mode": _w.RUN_MODE,
        "seed": _w.SEED,
        "attempt": _w.ATTEMPT,
        **kw,
    }
    # The datacenter the worker actually landed in (RunPod serverless sets RUNPOD_DC_ID) — a
    # diagnostic so the control plane / logs show which region a run hit (the eager weight-cache fleet
    # already has a volume in every storage DC). Empty/absent on non-RunPod (instance) workers and
    # harmless; only emitted when present.
    _dc = os.environ.get("RUNPOD_DC_ID") or ""
    if _dc:
        payload.setdefault("dc", _dc)
    os.makedirs("/tmp/hb", exist_ok=True)
    p = "/tmp/hb/heartbeat.json"
    # _HB_LOCK guards ONLY the fast local work (atomic write + _HB_LAST_UPLOAD + snapshot capture);
    # the slow HF commit runs OUTSIDE it so the trainer's per-step reward callback never blocks on
    # the network behind the checkpoint daemon's commit (a GRPO perf regression).
    with _HB_LOCK:
        # Atomic write: temp file + os.replace() so a concurrent reader never sees a partial file.
        tmp = p + f".{os.getpid()}.{threading.get_ident()}.tmp"
        snapshot = json.dumps(payload)
        with open(tmp, "w") as f:
            f.write(snapshot)
        os.replace(tmp, p)
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
            _w._HB_LAST_UPLOAD = now  # claim the slot under the lock (throttle stays atomic)
    # Re-arm the stall watchdog NOW, off the LOCAL write that just landed — the live-progress signal
    # is the local heartbeat, NOT the optional HF commit below. Re-arming after the upload (as before)
    # let a stalled best-effort upload run out the timer and faulthandler-kill a worker that had
    # already produced a live heartbeat. Pass the stage so the watchdog keeps the wide setup grace
    # until the first rl_step/sft_step, then tightens for training.
    _rearm_stall_faulthandler(stage)
    if upload_due:
        # Serialize the network commit under a SEPARATE lock so uploads can't reorder, and upload the
        # captured snapshot (via a private temp file, since hf_upload_file takes a path) rather than
        # re-reading p — which a newer heartbeat may already have overwritten between our slot-claim
        # and this upload. Acquire the lock with a BOUND (see _HB_UPLOAD_LOCK_TIMEOUT_S) so a wedged
        # upload holding the lock can't block this heartbeat indefinitely; on timeout skip the
        # best-effort commit (local write already landed; next heartbeat re-commits fresher state).
        # Terminal/error commits are CRITICAL (no later heartbeat repairs them; error_* carries the
        # retriable flag) so they wait far longer before skipping (_HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S).
        critical = stage in _HB_TERMINAL_STAGES or stage.startswith("error_")
        lock_timeout = _HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S if critical else _HB_UPLOAD_LOCK_TIMEOUT_S
        if _HB_UPLOAD_LOCK.acquire(timeout=lock_timeout):
            try:
                up = p + f".{os.getpid()}.{threading.get_ident()}.upload.tmp"
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
                        if now == _w._HB_LAST_UPLOAD:
                            _w._HB_LAST_UPLOAD = prev_last_upload
                    print(f"HEARTBEAT upload failed; rolled back throttle slot for {stage}")
            finally:
                _HB_UPLOAD_LOCK.release()
        else:
            # We claimed the upload slot above but never committed. Roll it back so the throttle
            # doesn't defer the NEXT commit by up to _HB_MIN_INTERVAL_S on the strength of an upload
            # that never happened, and so liveness_heartbeat's quiet_gate (which reads
            # _HB_LAST_UPLOAD) doesn't treat the channel as fresh while HF is still stale. Guard on
            # equality so we only undo OUR claim, never a newer one another thread landed meanwhile.
            with _HB_LOCK:
                if now == _w._HB_LAST_UPLOAD:
                    _w._HB_LAST_UPLOAD = prev_last_upload
            print(f"HEARTBEAT upload-lock busy >{lock_timeout}s; skipping commit for {stage}")
    print("HEARTBEAT", json.dumps(payload))


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
# backward, ~17 min observed) — emits no heartbeat while it runs and would look like a hang to the
# default-on stall watchdog and the provider setup grace. ``liveness_heartbeat`` runs a daemon next
# to the call that pings ``stage`` to prove the worker is alive.
#
# A liveness ping re-arms the watchdog and refreshes the provider timestamp, so emitting
# unconditionally would also mask a GENUINELY stuck call forever. When the phase exposes a monotonic
# progress counter (downloaded bytes, optimizer ``global_step``), pass ``progress`` + ``max_silence_s``
# and the daemon STOPS pinging once it hasn't advanced for that long, handing a real wedge back to the
# stall path. ``quiet_gate_s`` makes it a pure gap-filler (ping only when nothing else committed for
# that long) for a phase another heartbeat already covers in the normal case (rl_step/sft_step).
_LIVENESS_TICK_S = 30.0
_TRAIN_LIVENESS_QUIET_S = 90.0
_TRAIN_LIVENESS_MAX_STEP_S = 1800.0


@contextlib.contextmanager
def liveness_heartbeat(
    stage, *, progress=None, max_silence_s=None, quiet_gate_s=None, fields=None, join_timeout=10.0
):
    """Keep a ``stage`` heartbeat alive while the wrapped block runs on the main thread.

    - ``progress``: optional ``() -> float | None`` monotonic counter. With ``max_silence_s`` the
      daemon STOPS pinging once it hasn't advanced for that long, so liveness can't mask a real wedge
      (a ``None`` return — unmeasurable — never counts as advancement).
    - ``quiet_gate_s``: if set, ping only when no heartbeat has been uploaded for that long (gap-fill
      a phase another heartbeat — e.g. ``rl_step`` — already covers in the normal case).
    - ``fields``: optional ``dict`` or ``() -> dict`` merged into each heartbeat (e.g. step / model).

    Diagnostics are nvidia-smi-only (``include_torch=False``): the main thread owns the CUDA/allocator
    locks during these phases, so a side-thread ``torch.cuda`` query could block. The daemon is reaped
    with a BOUNDED join so it can never wedge the worker if stuck inside an HF upload.
    """
    done = threading.Event()

    def _loop() -> None:
        last_val = None
        # Measure the no-progress silence with the MONOTONIC clock: a wall-clock jump (NTP step, VM
        # suspend/resume) must not make max_silence_s trip early/late — it decides when liveness STOPS
        # and hands off to the stall watchdog, so a spurious early stop would false-fail a healthy run.
        advanced_at = time.monotonic()
        while not done.wait(_LIVENESS_TICK_S):
            if progress is not None:
                val = None
                with contextlib.suppress(Exception):
                    v = progress()
                    val = None if v is None else float(v)
                if val is not None and (last_val is None or val > last_val):
                    last_val, advanced_at = val, time.monotonic()
                if max_silence_s and (time.monotonic() - advanced_at) > max_silence_s:
                    print(f"liveness[{stage}]: no progress for >{max_silence_s:.0f}s; stopping")
                    return
            if quiet_gate_s is not None:
                # Wall-clock here on purpose: this compares against _HB_LAST_UPLOAD, a time.time()
                # timestamp set in heartbeat(), so both ends must be the same (wall) clock.
                quiet_for = 1e9
                with contextlib.suppress(Exception):
                    quiet_for = time.time() - float(getattr(_w, "_HB_LAST_UPLOAD", 0.0) or 0.0)
                if quiet_for < quiet_gate_s:
                    continue
            gpu = gpu_diagnostics(include_torch=False)
            if done.is_set():  # the wrapped call may have finished during nvidia-smi
                return
            # Compute the merged fields defensively: ``fields`` may be a callback that re-reads live
            # state (train_liveness_heartbeat's lambda calls get_step()), and an exception there must
            # NOT kill this daemon for the rest of the wrapped block — same reason progress() above is
            # suppressed. Fall back to no extra fields; the bare heartbeat still re-arms the watchdog.
            extra = fields if isinstance(fields, dict) else {}
            if callable(fields):
                with contextlib.suppress(Exception):
                    extra = fields() or {}
            _w.heartbeat(stage, gpu=gpu, **extra)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        done.set()
        t.join(timeout=join_timeout)


def train_liveness_heartbeat(stage, get_step):
    """``liveness_heartbeat`` configured for the training phase: gap-fill ``stage`` (rl_step/sft_step)
    when the channel is quiet, gated on ``get_step`` advancement so a genuinely stuck ``train()`` still
    trips the stall path. Shared by run_rl and run_sft."""
    return liveness_heartbeat(
        stage,
        progress=get_step,
        max_silence_s=_TRAIN_LIVENESS_MAX_STEP_S,
        quiet_gate_s=_TRAIN_LIVENESS_QUIET_S,
        fields=lambda: {"step": int(get_step() or 0)},
    )
