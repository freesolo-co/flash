"""Shared poll-loop scaffolding for the provider job pollers.

``runpod/jobs.py:poll_job`` and ``vast/jobs.py:poll_vast_job`` are independent live
poll loops with provider-specific terminal-state logic, but they share three verbatim
blocks: a timestamped ``say()`` logger, a consecutive-poll-error retry/give-up counter,
and the heartbeat progress-surfacing block (key on (stage, step, ts), log
``worker: stage=… step=… reward=…``). Only those provider-neutral pieces live here; each
poller keeps its own status/terminal handling inline.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any


def make_say(log) -> Callable[[str], None]:
    """A timestamped line logger that no-ops when ``log`` is None."""

    def say(msg: str) -> None:
        if log is not None:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log, flush=True)

    return say


class PollErrorTracker:
    """Counts consecutive poll errors and decides when to give up.

    Encapsulates the identical retry block both pollers use: on a transient fetch
    error, log it, give up after ``max_errors`` consecutive failures, otherwise sleep
    a linear backoff (capped at 60 s) before the caller retries.
    """

    def __init__(self, say: Callable[[str], None], interval_s: float, max_errors: int = 8) -> None:
        self._say = say
        self._interval_s = interval_s
        self._max_errors = max_errors
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def record(self, exc: Exception) -> bool:
        """Register a poll error. Returns True if the caller should give up (too many),
        else sleeps the backoff and returns False (caller should ``continue``)."""
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


def _fmt_float(value: Any, digits: int = 3) -> str | None:
    num = _num(value)
    if num is None:
        return None
    return f"{num:.{digits}f}"


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
    if not parts and gpu.get("nvidia_smi"):
        parts.append(str(gpu["nvidia_smi"])[:160])
    return " gpu[" + " ".join(parts) + "]" if parts else ""


def _format_heartbeat(hb: dict) -> str:
    msg = f"worker: stage={hb.get('stage')}"
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


def surface_heartbeat(
    heartbeat_reader: Callable[[], Any] | None,
    last_hb_key: tuple | None,
    say: Callable[[str], None],
) -> tuple[tuple | None, str | None]:
    """Read a heartbeat and, if it advanced, log worker progress.

    Returns ``(hb_key, stage)`` where ``hb_key`` is the new (stage, step, ts) key (or the
    unchanged ``last_hb_key`` when nothing advanced) and ``stage`` is the stage of the new
    heartbeat when it advanced (else None). Callers use the returned ``stage`` for their
    own setup-vs-training stall bookkeeping.
    """
    if heartbeat_reader is None:
        return last_hb_key, None
    try:
        hb = heartbeat_reader()
    except Exception:
        hb = None
    if not hb:
        return last_hb_key, None
    key = (hb.get("stage"), hb.get("step"), hb.get("ts"))
    if key == last_hb_key:
        return last_hb_key, None
    _record_heartbeat(hb)
    stage = hb.get("stage")
    say(_format_heartbeat(hb))
    return key, stage


def surface_forced_heartbeat(
    heartbeat_reader: Callable[..., Any] | None,
    last_hb_key: tuple | None,
    say: Callable[[str], None],
) -> tuple[tuple | None, str | None]:
    """Force-read and surface the latest heartbeat, bypassing reader rate limits.

    Used on terminal provider statuses so a fast worker failure still leaves the last worker/GPU
    snapshot in both the run log and status JSON.
    """
    if heartbeat_reader is None:
        return last_hb_key, None
    return surface_heartbeat(lambda: heartbeat_reader(force=True), last_hb_key, say)
