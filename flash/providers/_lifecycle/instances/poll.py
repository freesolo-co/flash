"""Shared poll-loop scaffolding for provider job pollers."""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable
from typing import Any

from flash._internal.diagnostics import neutralize_control_chars
from flash.adapters.artifacts import MAX_ATTEMPT_ID
from flash.engine.result.rollout_samples import sampled_completion_scalar
from flash.providers._lifecycle.net.deadline import remaining_seconds

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
# only once a real training heartbeat arrives.
LOAD_TIMEOUT_S = 900.0
SETUP_GRACE_S = 3000.0
STALL_AFTER_S = 1500.0


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

    def record(self, exc: Exception, *, deadline_at: float | None = None) -> bool:
        """Register a poll error and back off no later than the absolute deadline."""
        self._count += 1
        self._say(f"poll error ({self._count}): {type(exc).__name__}")
        if self._count >= self._max_errors:
            return True
        delay = min(60, self._interval_s * self._count)
        if deadline_at is not None:
            remaining = remaining_seconds(deadline_at)
            if remaining <= 0:
                return True
            delay = min(delay, remaining)
        if delay > 0:
            time.sleep(delay)
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


def format_gpu_status(gpu: Any) -> str:
    """Human-readable one-line GPU telemetry summary for heartbeat log lines."""
    if not isinstance(gpu, dict) or not gpu:
        return ""
    parts: list[str] = []
    name = gpu.get("device_name")
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
            pname = os.path.basename(str(proc.get("process_name") or "").strip()) or "process"
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


def _format_child_tail(tail: object, silent_ticks: object = None) -> str:
    """Render a stalled worker's retained child output, or "" when absent.

    ``child_tail`` and ``silent_ticks`` must reach the streamed log to distinguish slow output from a
    wedged child. Skip malformed non-string entries rather than breaking the heartbeat line.
    """
    if not isinstance(tail, list):
        return ""
    lines = [
        f"    {neutralize_control_chars(line)}" for line in tail if isinstance(line, str) and line
    ]
    if not lines:
        return ""
    header = "\n  child output before the stall (most recent last)"
    # bool is an int subclass, and a JSON payload can carry anything; a malformed counter drops the
    # annotation rather than the tail it annotates.
    if isinstance(silent_ticks, int) and not isinstance(silent_ticks, bool) and silent_ticks >= 0:
        if silent_ticks == 0:
            header += ", still producing output"
        else:
            plural = "" if silent_ticks == 1 else "s"
            header += f", unchanged for {silent_ticks} tick{plural}"
    return header + ":\n" + "\n".join(lines)


def _format_heartbeat(hb: dict) -> str:
    msg = f"worker: stage={hb.get('stage')}"
    attempt = _attempt_int(hb.get("attempt"))
    if attempt is not None:
        msg += f" attempt={attempt}"
    if hb.get("liveness"):
        msg += " liveness=true"
    for key, digits in (
        ("step", 0),
        ("epoch", 3),
        ("reward", 3),
        ("reward_std", 3),
        ("loss", 4),
        ("grad_norm", 3),
        ("kl", 3),
        ("entropy", 3),
        ("frac_reward_zero_std", 3),
        ("mean_completion_tokens", 3),
        ("truncation_rate", 3),
        ("discarded_rollouts", 0),
        ("max_completion_tokens", 0),
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
    reward_metrics = hb.get("reward_metrics")
    if isinstance(reward_metrics, dict):
        for name, value in reward_metrics.items():
            if isinstance(value, (int, float)):
                msg += f" {name}={value:.3f}"
    error = hb.get("error")
    if isinstance(error, str) and error.strip():
        msg += f" error={error.strip()}"
    checkpoint_failure = hb.get("checkpoint_failure")
    if isinstance(checkpoint_failure, dict):
        failure_step = checkpoint_failure.get("step")
        if isinstance(failure_step, (int, float)) and not isinstance(failure_step, bool):
            msg += f" checkpoint_failure_step={int(failure_step)}"
        operation = checkpoint_failure.get("operation")
        if isinstance(operation, str) and operation.strip():
            msg += f" checkpoint_failure_stage={operation.strip()}"
        checkpoint_error = checkpoint_failure.get("error")
        if isinstance(checkpoint_error, str) and checkpoint_error.strip():
            msg += f" checkpoint_error={checkpoint_error.strip()}"
    msg += format_gpu_status(hb.get("gpu"))
    # everything above is one summary record, including untrusted stage, error, and gpu fallback text.
    # sanitize it once at that boundary so no field can forge a second worker record. the deliberately
    # multiline sample and child-output sections are appended below and retain their line breaks.
    msg = neutralize_control_chars(msg).replace("\n", "\\n")
    sample_lines: list[str] = []
    rendered_samples = 0
    samples = hb.get("sampled_completions")
    if isinstance(samples, list):
        for sample in samples:
            if rendered_samples >= 3:
                break
            if not isinstance(sample, dict):
                continue
            prompt_tail = sample.get("prompt_tail")
            completion = sample.get("completion")
            if not isinstance(prompt_tail, str) or not isinstance(completion, str):
                continue
            # A sample carries exactly one scalar: GRPO reward (3dp) or OPD distillation loss (4dp).
            scalar = sampled_completion_scalar(sample)
            if scalar is None:
                continue
            scalar_label, scalar_value = scalar
            generated_at_step = sample.get("generated_at_step")
            if generated_at_step is None:
                step_suffix = ""
            else:
                try:
                    step_suffix = f" step={int(generated_at_step)}"
                except (TypeError, ValueError):
                    continue
            rendered_samples += 1
            # Neutralize control chars (defense against a malformed heartbeat.json) but show the full
            # prompt and completion — samples are surfaced untruncated by design.
            prompt_tail = neutralize_control_chars(prompt_tail)
            completion = neutralize_control_chars(completion)
            digits = 4 if scalar_label == "loss" else 3
            sample_lines.extend(
                [
                    f"  sample {rendered_samples} {scalar_label}={scalar_value:.{digits}f}{step_suffix}",
                    f"    prompt: {prompt_tail.replace(chr(10), chr(10) + '      ')}",
                    f"    completion: {completion.replace(chr(10), chr(10) + '      ')}",
                ]
            )
    if sample_lines:
        msg += "\n" + "\n".join(sample_lines)
    msg += _format_child_tail(hb.get("child_tail"), hb.get("child_tail_silent_ticks"))
    return msg


def _record_heartbeat(hb: dict) -> None:
    run_id = str(hb.get("run_id") or "").strip()
    if not run_id:
        return
    try:
        from flash.runner.lifecycle.status import record_heartbeat

        record_heartbeat(run_id, hb)
    except Exception:
        # Status persistence is diagnostic only; polling/liveness must not depend on it.
        pass


# Stages emitted during cold-start BEFORE training begins — must not flip stall detection to the tight
# training window. Canonical here so all instance providers share one definition. Must cover every
# pre-training liveness stage the worker can upgrade to a real heartbeat (progress-carry), or a
# carried setup heartbeat would prematurely tighten the stall window.
SETUP_HEARTBEAT_STAGES = frozenset(
    {
        "boot",
        "sft_start",
        "rl_start",
        "model_prefetching",
        "model_prefetched",
        "checkpoint_prefetching",
        "sft_data_loading",
        "rl_data_loading",
        "rl_adapter_loading",
        "sft_model_load",
        "sft_pretokenizing",
        "sft_configuring",
        "rl_configuring",
        "rl_train_start",
        "sft_initializing",
        "rl_initializing",
        # opd cold-start stages must keep setup grace until the first opd_step. filtering emits real
        # progress, so omitting it would stickily switch preprocessing to the training stall window.
        "opd_start",
        "opd_configuring",
        "opd_filtering_prompts",
        # the prompt-budget scan and image preparation opd_train actually emits today
        # (opd_train.py `liveness_heartbeat("opd_prompt_scan")` / `("opd_image_prep")`). both carry a
        # progress callback, so without them here is_training_heartbeat flips the run into the tight
        # training stall window while it is still preprocessing, and a large split is torn down as
        # "stalled" before the first step. the worker-side registry in heartbeat.py already lists
        # both; this is the provider half of the same pair.
        "opd_prompt_scan",
        "opd_image_prep",
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
    """Validate a bounded nonnegative integer attempt identity."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= MAX_ATTEMPT_ID else None


def heartbeat_oom_for_attempt(hb: Any, current_attempt: int | None) -> bool:
    if not isinstance(hb, dict) or not hb.get("oom"):
        return False
    hb_attempt = _attempt_int(hb.get("attempt"))
    expected_attempt = _attempt_int(current_attempt)
    return expected_attempt is not None and hb_attempt == expected_attempt


def heartbeat_progress_ts(
    hb_key: tuple | None, launch_ts: float | None, current_attempt: int | None = None
) -> tuple[float, bool]:
    """Return (ts, fresh): ts is the heartbeat's own timestamp clamped to [launch, now]; fresh is
    False for a prior-attempt leftover (retries reuse the same seed heartbeat path).

    Use the heartbeat's own ts, not poll time, so stale pre-reattach state buys no fresh window."""
    now = time.time()
    ts = hb_key[2] if (isinstance(hb_key, tuple) and len(hb_key) >= 3) else None
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return now, False
    try:
        lo = float(launch_ts)
    except (TypeError, ValueError):
        return now, False
    if not math.isfinite(lo) or lo <= 0 or not math.isfinite(ts):
        return now, False
    hb_attempt = (
        _attempt_int(hb_key[3]) if (isinstance(hb_key, tuple) and len(hb_key) >= 4) else None
    )
    cur_attempt = _attempt_int(current_attempt)
    fresh = ts >= lo and cur_attempt is not None and hb_attempt == cur_attempt
    return min(now, max(lo, ts)), fresh
