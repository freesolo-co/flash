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

# The rl_step heartbeat-upload throttle stages + terminal stages. Only the per-step "rl_step"
# stage is high-frequency, so throttle JUST that one; terminal transitions always commit.
_HB_THROTTLED_STAGES = frozenset({"rl_step"})
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

_STEP_GPU_DIAG_INTERVAL_S = 300.0
_SFT_HEARTBEAT_INTERVAL_S = 60.0

# Stall diagnostics: when FLASH_STALL_FAULTHANDLER_S > 0, arm a faulthandler watchdog that dumps
# every thread's Python stack (then exits, so the run FAILS instead of hanging until the
# control-plane stall watchdog kills it ~25 min later, and the dump is uploaded with
# console_<phase>.txt). The timer is re-armed on every heartbeat, so it only fires when NO progress
# heartbeat lands for the whole window -- i.e. a real hang. OFF by default (0); opt-in per run via
# [worker_env]. Used to localize the GRPO sleep-mode rollout hang.
_STALL_FAULTHANDLER_S = 0
with contextlib.suppress(Exception):
    _STALL_FAULTHANDLER_S = int(os.environ.get("FLASH_STALL_FAULTHANDLER_S", "0") or 0)


def _rearm_stall_faulthandler() -> None:
    if _STALL_FAULTHANDLER_S <= 0:
        return
    with contextlib.suppress(Exception):
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(_STALL_FAULTHANDLER_S, exit=True)


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
    if upload_due:
        # Serialize the network commit under a SEPARATE lock so uploads can't reorder, and
        # upload the captured snapshot (via a private temp file, since hf_upload_file takes
        # a path) rather than re-reading p — which a newer heartbeat may already have
        # overwritten between our slot-claim and this upload.
        with _HB_UPLOAD_LOCK:
            up = p + f".{os.getpid()}.{threading.get_ident()}.upload.tmp"
            with open(up, "w") as f:
                f.write(snapshot)
            try:
                _w.hf_upload_file(up, "heartbeat.json")
            finally:
                with contextlib.suppress(OSError):
                    os.remove(up)
    # Re-arm the stall watchdog: progress landed, so reset the no-heartbeat timer.
    _rearm_stall_faulthandler()
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
