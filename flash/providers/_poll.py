"""Shared poll-loop scaffolding for provider job pollers.

Poll loops share a timestamped ``say()`` logger, a consecutive-poll-error retry/give-up
counter, and the heartbeat progress-surfacing block (key on (stage, step, ts), log
``worker: stage=… step=… reward=…``). Only those neutral pieces live here; each poller
keeps its own status/terminal handling inline.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from typing import Any

# Grace past a preload box's embedded wall deadline before an orphan sweep reaps it. A healthy warm
# self-bounds at its wall cap (the in-box timer ``os._exit``s) and the driver's ``finally`` terminates
# the instance; a box still alive THIS long past its deadline has lost its driver (the only thing that
# tears instance providers down), so it is provably orphaned and safe to reap. Generous so clock skew /
# a slow teardown / a near-deadline box mid-download is never reaped early.
PRELOAD_REAP_GRACE_S = 1800.0


def preload_instance_run_id(provider: str, region: str, reap_deadline_epoch: int, suffix: str) -> str:
    """Build a ``flash-preload-*`` run id that embeds its wall-clock reap deadline (``-d<epoch>-``).

    The epoch lets an orphan sweep reap a driver-lost warm box by NAME alone (no provider creation-time
    field needed). ``reap_deadline_epoch`` is the box's wall-cap deadline in epoch seconds. Kept in sync
    with ``preload_box_reap_due``'s parser — change both together.

    The epoch is placed RIGHT AFTER ``flash-preload-`` (before provider/region) on purpose: the launched
    instance NAME is bounded to the provider name budget by ``run_label_prefix``, which truncates the
    TAIL and appends a hash. A long provider+region (e.g. hyperstack + a long region) would otherwise
    push the deadline token past the cut and the reap parser would never see it — front-loading keeps
    ``-d<epoch>-`` inside the surviving prefix."""
    return f"flash-preload-d{int(reap_deadline_epoch)}-{provider}-{region.lower()}-{suffix}"


def preload_box_reap_due(name: str, now: float, grace_s: float = PRELOAD_REAP_GRACE_S) -> bool:
    """True when a ``flash-preload-*`` instance name carries an embedded reap deadline (``-d<epoch>-``,
    written by ``preload_instance_run_id``) that elapsed more than ``grace_s`` ago.

    Used by the Lambda/Hyperstack orphan sweeps: warm boxes are normally driver-owned and exempt, but a
    driver that died before its ``terminate_run_instances`` finally would leave one billing forever.
    Reaping past deadline+grace bounds that leak. Names WITHOUT a parseable deadline (legacy launches)
    return False — the unconditional driver-owned exemption still applies to them. The 10+ digit guard
    keeps a region segment like ``us-east-1`` from being mistaken for the ``-d<epoch>-`` token."""
    m = re.search(r"-d(\d{10,})-", name)
    if not m:
        return False
    return float(m.group(1)) + grace_s < now


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


def heartbeat_progress_ts(hb_key: tuple | None, launch_ts: float | None) -> tuple[float, bool]:
    """Wall-clock to credit as 'last worker progress' for a just-surfaced heartbeat, plus whether
    that heartbeat actually belongs to THIS attempt.

    Use the heartbeat's OWN ``ts`` (key[2] = when the worker actually made progress), not the
    poll time. On a delayed reattach after a control-plane restart, a heartbeat that was already
    stale BEFORE the restart must not buy a fresh full stall window — crediting the poll time
    would hand a hung worker another grace period while the instance keeps billing. Clamp to
    ``[launch, now]`` so worker/control-plane clock skew can neither make a healthy worker look
    ancient (premature stall) nor land its progress in the future.

    Returns ``(ts, fresh)``. ``fresh`` is False when the heartbeat's ts predates this attempt's
    launch: that is a LEFTOVER heartbeat from a prior attempt (retries reuse the same seed
    heartbeat path), so the caller must NOT treat it as current progress — otherwise a stale
    training-stage heartbeat would arm the tighter training stall window and fail a healthy new
    attempt mid-setup before it has overwritten the old file. ``launch_ts`` uses truthiness (not
    ``is not None``): the instance handles store started_ts as a non-Optional float coerced to 0.0
    when missing, so 0.0 means "unknown launch" (a real launch is a large epoch ts). When launch is
    UNKNOWN we cannot date heartbeats relative to it, so the clamp floor drops to 0.0 and every
    heartbeat counts as fresh (the safe default: don't discard progress we can't date — clamping the
    floor to ``now`` instead would mark every normal heartbeat, timestamped before it is read, stale
    and stall a healthy recovered worker)."""
    now = time.time()
    ts = hb_key[2] if (isinstance(hb_key, tuple) and len(hb_key) >= 3) else None
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return now, False
    lo = float(launch_ts) if launch_ts else 0.0  # unknown launch -> floor 0.0 (all heartbeats fresh)
    fresh = ts >= lo
    return min(now, max(lo, ts)), fresh


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
