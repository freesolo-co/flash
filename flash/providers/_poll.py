"""Shared poll-loop scaffolding for provider job pollers."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from typing import Any

# Generous grace past embedded deadline before orphan sweep reaps a driver-lost warm box.
PRELOAD_REAP_GRACE_S = 1800.0


def preload_instance_run_id(
    provider: str, region: str, reap_deadline_epoch: int, suffix: str
) -> str:
    """Build a ``flash-preload-*`` run id embedding its wall-clock reap deadline.

    Epoch placed right after ``flash-preload-`` so ``run_label_prefix`` tail-truncation never drops it.
    Keep in sync with ``preload_box_reap_due``'s parser."""
    return f"flash-preload-d{int(reap_deadline_epoch)}-{provider}-{region.lower()}-{suffix}"


def preload_box_reap_due(name: str, now: float, grace_s: float = PRELOAD_REAP_GRACE_S) -> bool:
    """True when a preload instance name's embedded deadline has elapsed past grace.

    Names without a parseable deadline return False. 10+ digit guard avoids matching region segments."""
    m = re.search(r"-d(\d{10,})-", name)
    if not m:
        return False
    return float(m.group(1)) + grace_s < now


# Fast-fail threshold: active box with no boot.log and no heartbeat this long means cloud-init never ran.
FIRST_LIVENESS_S = 900.0

# Minimum observed-active time before fast-failover fires — protects a freshly-reattached box on
# reattach whose launch-anchored active_since already exceeds FIRST_LIVENESS_S.
FIRST_LIVENESS_OBSERVED_FLOOR_S = 120.0

# Require absence to persist this many polls before declaring sick — one None could be a Hub blip.
BOOT_LOG_ABSENT_POLLS = 3

# Standardized instance-poll timing defaults, shared by every rent-a-box provider (lambda + vast). The
# per-provider _DEAD_STATES vocabulary and status-field name stay local to each jobs module.
# LOAD_TIMEOUT_S: how long an instance may sit non-running (image pull / provisioning) before we give up.
# SETUP_GRACE_S / STALL_AFTER_S: the staged no-progress window once running — the larger setup grace
# covers the heartbeat-less cold start (pip install + base-model download), tightening to STALL_AFTER_S
# only once a real training heartbeat arrives. PROVISION_GRACE_S: provision + cold-start slack added to
# the run's wall cap for the client-side poll deadline (neither substrate has a server-side timeout).
LOAD_TIMEOUT_S = 900.0
SETUP_GRACE_S = 3000.0
STALL_AFTER_S = 1500.0
PROVISION_GRACE_S = 3000.0


def make_say(log) -> Callable[[str], None]:
    """A timestamped line logger that no-ops when ``log`` is None."""

    def say(msg: str) -> None:
        if log is not None:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log, flush=True)

    return say


class PollErrorTracker:
    """Counts consecutive poll errors; sleeps backoff or signals give-up."""

    def __init__(self, say: Callable[[str], None], interval_s: float, max_errors: int = 8) -> None:
        self._say = say
        self._interval_s = interval_s
        self._max_errors = max_errors
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def record(self, exc: Exception) -> bool:
        """Register a poll error; returns True to give up, False to continue after backoff sleep."""
        self._count += 1
        self._say(f"poll error ({self._count}): {exc}")
        if self._count >= self._max_errors:
            return True
        time.sleep(min(60, self._interval_s * self._count))
        return False


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_gb(value: Any) -> str | None:
    num = _num(value)
    if num is None:
        return None
    return f"{num:.1f}GB"


def _fmt_pct(value: Any) -> str | None:
    num = _num(value)
    if num is None:
        return None
    return f"{num:.0f}%"


def _fmt_watts(value: Any) -> str | None:
    num = _num(value)
    if num is None:
        return None
    return f"{num:.0f}W"


def _short_process_name(name: str) -> str:
    base = os.path.basename(str(name or "").strip())
    return base or "process"


def format_gpu_status(gpu: Any) -> str:
    """Human-readable one-line GPU telemetry summary for heartbeat log lines."""
    if not isinstance(gpu, dict) or not gpu:
        return ""
    parts: list[str] = []
    name = gpu.get("device_name") or gpu.get("name")
    if name:
        parts.append(str(name))
    driver = gpu.get("driver_version")
    cuda = gpu.get("torch_cuda")
    if driver:
        parts.append(f"driver={driver}")
    if cuda:
        parts.append(f"cuda={cuda}")
    util = _fmt_pct(gpu.get("gpu_util_pct"))
    mem_util = _fmt_pct(gpu.get("mem_util_pct"))
    if util:
        parts.append(f"util={util}")
    if mem_util:
        parts.append(f"mem_util={mem_util}")
    used = _fmt_gb(gpu.get("memory_used_gb"))
    total = _fmt_gb(gpu.get("memory_total_gb"))
    free = _fmt_gb(gpu.get("memory_free_gb"))
    if used and total:
        parts.append(f"mem={used}/{total}")
    elif free and total:
        parts.append(f"free={free}/{total}")
    torch_alloc = _fmt_gb(gpu.get("torch_memory_allocated_gb"))
    torch_reserved = _fmt_gb(gpu.get("torch_memory_reserved_gb"))
    if torch_alloc:
        if torch_reserved:
            parts.append(f"torch={torch_alloc}/{torch_reserved}")
        else:
            parts.append(f"torch={torch_alloc}")
    temp = _num(gpu.get("temperature_c"))
    if temp is not None:
        parts.append(f"temp={temp:.0f}C")
    power = _fmt_watts(gpu.get("power_w"))
    power_limit = _fmt_watts(gpu.get("power_limit_w"))
    if power and power_limit:
        parts.append(f"power={power}/{power_limit}")
    elif power:
        parts.append(f"power={power}")
    pstate = gpu.get("pstate")
    if pstate:
        parts.append(f"pstate={pstate}")
    processes = gpu.get("processes")
    if isinstance(processes, list) and processes:
        proc_parts = []
        for proc in processes[:3]:
            if not isinstance(proc, dict):
                continue
            pname = _short_process_name(str(proc.get("process_name") or ""))
            pid = proc.get("pid")
            mem = _fmt_gb(proc.get("used_memory_gb"))
            label = f"{pname}:{pid}" if pid is not None else pname
            if mem:
                label = f"{label}:{mem}"
            proc_parts.append(label)
        if proc_parts:
            parts.append("procs=" + ",".join(proc_parts))
    if not parts:
        if gpu.get("nvidia_smi"):
            parts.append(str(gpu["nvidia_smi"])[:160])
        elif gpu.get("nvidia_smi_err"):
            parts.append(str(gpu["nvidia_smi_err"])[:160])
    return " gpu[" + " ".join(parts) + "]" if parts else ""


def _format_heartbeat(hb: dict) -> str:
    msg = f"worker: stage={hb.get('stage')}"
    if hb.get("liveness"):
        msg += " liveness=true"
    for key, digits in (
        ("step", 0),
        ("epoch", 3),
        ("reward", 3),
        ("loss", 4),
        ("grad_norm", 3),
        ("learning_rate", 8),
        ("setup_seconds", 1),
        ("train_wall", 1),
    ):
        value = hb.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            if digits == 0:
                msg += f" {key}={int(value)}"
            else:
                msg += f" {key}={value:.{digits}f}"
        else:
            msg += f" {key}={value}"
    msg += format_gpu_status(hb.get("gpu") or hb.get("diag"))
    return msg


def _record_heartbeat(hb: dict) -> None:
    run_id = str(hb.get("run_id") or "").strip()
    if not run_id:
        return
    try:
        from flash.runner import record_heartbeat

        record_heartbeat(run_id, hb)
    except Exception:
        # Status persistence is diagnostic only; polling/liveness must not depend on it.
        pass


# Stages emitted during cold-start BEFORE training begins — must not flip stall detection to the tight
# training window. Canonical here so all instance providers share one definition.
SETUP_HEARTBEAT_STAGES = frozenset(
    {
        "boot",
        "sft_start",
        "rl_start",
        "model_prefetching",
        "model_prefetched",
        "sft_model_load",
        "rl_train_start",
        "sft_initializing",
        "rl_initializing",
        # OPD cold-start stages (emitted before the first opd_step): prompt-budget filtering,
        # wait-for-GPU, model load, and LoRA/warm-start init. Without these a slow OPD cold start is
        # judged by the tight training stall window and retried as "stalled" instead of getting the
        # setup grace. opd_filtering_prompts emits REAL progress heartbeats while it renders+tokenizes
        # the split, so it must be listed here or is_training_heartbeat would (stickily) flip the run
        # into the training window mid-setup.
        "opd_start",
        "opd_filtering_prompts",
        "opd_model_load",
        "opd_initializing",
    }
)

# step=0 is emitted throughout the silent first step — only step>=1 tightens the stall window. opd_step
# is gated too: its progress pings report opt_steps (0 until the FIRST optimizer update lands), so a
# long teacher-bound first step keeps the wide setup grace instead of the tight training window — the
# mid-step ping must not prematurely flip a still-running first step out of setup grace.
STEP_GATED_STAGES = frozenset({"rl_step", "sft_step", "opd_step"})


def is_training_heartbeat(stage: str | None, step: Any) -> bool:
    """True when a heartbeat means cold-start setup is over and the tight stall window should apply."""
    if not stage or stage in SETUP_HEARTBEAT_STAGES:
        return False
    if stage in STEP_GATED_STAGES:
        return (_attempt_int(step) or 0) >= 1
    return True


def surface_heartbeat(
    heartbeat_reader: Callable[[], Any] | None,
    last_hb_key: tuple | None,
    say: Callable[[str], None],
) -> tuple[tuple | None, str | None]:
    """Read a heartbeat and log it if it advanced.

    Returns ``(hb_key, stage)``. ``hb_key`` is for deduping any visible heartbeat; ``stage`` is
    ``None`` for liveness pings so provider pollers do not count them as progress.
    """
    if heartbeat_reader is None:
        return last_hb_key, None
    try:
        hb = heartbeat_reader()
    except Exception:
        hb = None
    if not hb:
        return last_hb_key, None
    key = (hb.get("stage"), hb.get("step"), hb.get("ts"), hb.get("attempt"))
    if key == last_hb_key:
        return last_hb_key, None
    if hb.get("liveness"):
        # Liveness pings update visible run status/logs, but must not advance the stall key — a
        # wedged worker pinging "alive" would mask a stall.
        _record_heartbeat(hb)
        say(_format_heartbeat(hb))
        return key, None
    _record_heartbeat(hb)
    stage = hb.get("stage")
    say(_format_heartbeat(hb))
    return key, stage


def _attempt_int(value: Any) -> int | None:
    """Coerce attempt number to int, or None when empty/unparseable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def heartbeat_oom_for_attempt(hb: Any, current_attempt: int | None) -> bool:
    if not isinstance(hb, dict) or not hb.get("oom"):
        return False
    return current_attempt is not None and _attempt_int(hb.get("attempt")) == current_attempt


def heartbeat_progress_ts(
    hb_key: tuple | None, launch_ts: float | None, current_attempt: int | None = None
) -> tuple[float, bool]:
    """Return (ts, fresh): ts is the heartbeat's own timestamp clamped to [launch, now]; fresh is
    False for a prior-attempt leftover (retries reuse the same seed heartbeat path).

    Use the heartbeat's own ts, not poll time — a stale-before-reattach heartbeat must not buy a fresh
    stall window. launch_ts=0.0 means unknown; attempt mismatches are still rejected."""
    now = time.time()
    ts = hb_key[2] if (isinstance(hb_key, tuple) and len(hb_key) >= 3) else None
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return now, False
    lo = (
        float(launch_ts) if launch_ts else 0.0
    )  # unknown launch -> floor 0.0 (all heartbeats fresh)
    fresh = ts >= lo
    # Worker stamps attempt as a str env var, poller passes int; coerce both before comparing.
    hb_attempt = (
        _attempt_int(hb_key[3]) if (isinstance(hb_key, tuple) and len(hb_key) >= 4) else None
    )
    cur_attempt = _attempt_int(current_attempt)
    if fresh and cur_attempt is not None and hb_attempt != cur_attempt:
        fresh = False
    return min(now, max(lo, ts)), fresh
