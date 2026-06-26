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
# But acquire it with a BOUND: huggingface_hub's upload has no hard per-call timeout, so a wedged
# upload could hold the lock indefinitely and block the NEXT heartbeat — including an unthrottled
# milestone like model_prefetched that sits on the worker's critical path right before trainer
# construction. Past this bound we skip the (best-effort) commit rather than wedge; the local write
# already landed and the next heartbeat re-commits a fresher snapshot. Generous so a healthy-but-slow
# upload is never skipped — only a genuinely stuck one trips it.
_HB_UPLOAD_LOCK_TIMEOUT_S = 30.0

_STEP_GPU_DIAG_INTERVAL_S = 300.0
_SFT_HEARTBEAT_INTERVAL_S = 60.0

# Stall diagnostics: when FLASH_STALL_FAULTHANDLER_S > 0, arm a faulthandler watchdog that dumps
# every thread's Python stack (then exits, so the run FAILS instead of hanging until the
# control-plane stall watchdog kills it ~25 min later, and the dump is uploaded with
# console_<phase>.txt). The timer is re-armed on every heartbeat, so it only fires when NO progress
# heartbeat lands for the whole window -- i.e. a real hang. Used to localize the GRPO sleep-mode
# rollout hang and the consumer-GPU warm-start init hang.
#
# DEFAULT-ON (2400s / 40 min). This is the STEADY-STATE (training) window; the whole cold start runs
# under the wider _STALL_STARTUP_GRACE_S below until the first rl_step/sft_step. It is SAFE — and
# strictly better than the old silent wedge — because every heartbeat re-arms the timer: a slow-but-
# LIVE cold init pings ``rl_initializing``/``sft_initializing`` every 30s (and those now use the
# GIL-friendly nvidia-smi-only diagnostics, so they keep ticking through a CUDA-busy init), so the
# watchdog only fires on a TRUE hang where NO heartbeat lands for the whole window. The training
# window is WELL above the longest observed HEALTHY per-step gap, so it adds a stack dump for a TRUE
# training hang without false-killing a healthy step. When it fires it dumps every thread's stack
# (C-level faulthandler -> fires even if the main thread holds the GIL) and fails the run, turning
# the previously undiagnosable "process wedged, no console upload" hang into an uploaded stack trace.
# Set FLASH_STALL_FAULTHANDLER_S=0 in [worker_env] to disable; lower it to localize a known hang.
_STALL_FAULTHANDLER_S = 2400
with contextlib.suppress(Exception):
    _STALL_FAULTHANDLER_S = int(os.environ.get("FLASH_STALL_FAULTHANDLER_S", "2400") or 2400)

# Startup/setup grace: the ENTIRE cold start — model prefetch, weight load, vLLM build,
# *Trainer.__init__, and the (often silent) full-dataset render/tokenize — can legitimately run for
# many minutes with only coarse 30s setup pings, or, inside dataset tokenization, none at all. A
# tight FLASH_STALL_FAULTHANDLER_S window would false-kill such a HEALTHY-but-slow setup, so until
# the first per-step TRAINING heartbeat (rl_step/sft_step) lands, the watchdog uses this WIDER
# window for EVERY arm (not just the first one). Default 3000s == the providers' SETUP_GRACE_S so
# the provider's own (retriable) setup grace governs a genuinely-stuck setup; only once real
# training steps flow do we tighten to FLASH_STALL_FAULTHANDLER_S, where a fast in-process stack
# dump is the point. Reads its OWN env var (independent of FLASH_STALL_FAULTHANDLER_S); never
# consulted when the watchdog is disabled (_STALL_FAULTHANDLER_S <= 0 -> early return below).
_STALL_STARTUP_GRACE_S = 3000
with contextlib.suppress(Exception):
    _STALL_STARTUP_GRACE_S = int(os.environ.get("FLASH_STALL_FAULTHANDLER_STARTUP_S", "3000") or 3000)
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
        if _HB_UPLOAD_LOCK.acquire(timeout=_HB_UPLOAD_LOCK_TIMEOUT_S):
            try:
                up = p + f".{os.getpid()}.{threading.get_ident()}.upload.tmp"
                with open(up, "w") as f:
                    f.write(snapshot)
                try:
                    _w.hf_upload_file(up, "heartbeat.json")
                finally:
                    with contextlib.suppress(OSError):
                        os.remove(up)
            finally:
                _HB_UPLOAD_LOCK.release()
        else:
            print(f"HEARTBEAT upload-lock busy >{_HB_UPLOAD_LOCK_TIMEOUT_S}s; skipping commit for {stage}")
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
