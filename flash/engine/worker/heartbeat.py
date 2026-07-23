"""Worker heartbeat: stream stage/progress to the HF artifact repo + TRL callbacks.

Monkeypatch contract: all patchable knobs live on the worker package (_w); locks/frozensets live
here. Tests that monkeypatch worker.<name> then call worker.heartbeat(...) take effect.
"""

from __future__ import annotations

import contextlib
import faulthandler
import json
import math
import os
import re
import sys
import threading
import time

from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import gpu_diagnostics
from flash.engine.worker.rollout_samples import sanitize_rollout_text

# Setup-phase liveness stages: emitted from a 30s liveness thread WITH a progress callback during the
# cold download / model-load / split-scan phase, kept on the tighter setup-liveness upload cadence
# (parity with sft_pretokenizing) so the stall detector stays fed while nothing is training yet.
# _HB_THROTTLED_STAGES is DERIVED from this set below (⊇ dev #442's explicit list) so the union of
# per-arch stages stays single-source.
_HB_SETUP_LIVENESS_STAGES = frozenset(
    {
        "model_prefetching",
        "checkpoint_prefetching",
        "sft_data_loading",
        "rl_data_loading",
        "rl_adapter_loading",
        "sft_pretokenizing",
        "opd_filtering_prompts",
        "sft_initializing",
        "rl_initializing",
        "opd_initializing",
        "sft_finalizing",
        "rl_finalizing",
    }
)
# Mid-training the per-step checkpoint upload runs SYNCHRONOUSLY on the trainer thread (dev #445),
# which freezes the trainer's global_step — so the ``rl_step``/``sft_step`` liveness daemon can only
# emit bare liveness pings for the whole upload, and those DON'T advance the provider's stall clock.
# A ``checkpoint_uploading`` keepalive daemon (liveness_heartbeat(keepalive=True)) wraps the upload to
# keep that clock fed; it rides the SAME tight, throttled cadence as a setup stage. Kept OUT of
# _HB_SETUP_LIVENESS_STAGES so that set's "setup-phase" meaning (and its tests) stay honest.
_HB_UPLOAD_LIVENESS_STAGES = frozenset(
    {"checkpoint_uploading", "opd_openrlhf_finalizing"}
)
# Liveness stages that ride the tighter setup-liveness upload interval (setup + mid-train upload).
_HB_TIGHT_LIVENESS_STAGES = _HB_SETUP_LIVENESS_STAGES | _HB_UPLOAD_LIVENESS_STAGES

# latest per-step GRPO backlog, exposed so a top-level error heartbeat can preserve it
# for `flash log -f` when a short run raises before the throttled rl_step ping committed
LATEST_GRPO_METRICS_LAST: list = []
# Throttled to avoid blowing the 128/hr HF commit cap; terminal transitions are never throttled. Every
# tight-liveness stage is throttled (⊂) PLUS the per-step training stages: opd_filtering_prompts alone
# emits a REAL (non-liveness) heartbeat every scan tick — ~120/hr on a large split before model load —
# so unthrottled the setup stages blow the cap; throttle them exactly like their sft_pretokenizing
# analogue (codex[bot]). checkpoint_uploading keepalive re-emits every 30s too, so it MUST be throttled.
_HB_THROTTLED_STAGES = _HB_TIGHT_LIVENESS_STAGES | frozenset(
    {
        "rl_step",
        "sft_step",
        "opd_step",
        "opd_openrlhf_training",
    }
)
_HB_TERMINAL_STAGES = frozenset({"done", "already_done"})
# 600s -> ~6 commits/hr; keeps stall detector alive without hitting the HF commit cap.
_HB_TERMINAL_ONLY_INTERVAL_S = 600.0


def _is_critical_stage(stage: str) -> bool:
    """A terminal transition or an error is CRITICAL: never throttled (the commit must land) and
    given the longer upload-lock timeout, because no later heartbeat can repair a missed one."""
    return stage in _HB_TERMINAL_STAGES or stage.startswith("error_")

# Guards throttle bookkeeping; slow HF commit runs outside this lock so trainer callbacks don't
# block on the network.
_HB_LOCK = threading.Lock()
# Serializes HF commits to prevent reorder; each thread uploads its own per-call temp file.
_HB_UPLOAD_LOCK = threading.Lock()
# Terminal/error commits wait longer — no later heartbeat can repair them.
_HB_UPLOAD_LOCK_TIMEOUT_S = 30.0
_HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S = 120.0
# Monotonic claim counter; rollback guard uses SEQ not wall-clock (two threads can share same ts).
_HB_CLAIM_SEQ = 0

_STEP_GPU_DIAG_INTERVAL_S = 300.0
_SFT_HEARTBEAT_INTERVAL_S = 60.0
# retain at least one metric row per second across the 900s training heartbeat throttle window.
_GRPO_METRIC_HISTORY_LIMIT = 1024


def _dump_thread_stacks(reason: str) -> None:
    with contextlib.suppress(Exception):
        print(
            f"[stall] {reason}: dumping all thread stacks, then yielding to the provider",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback(all_threads=True)


def _rollback_throttle_slot(
    my_claim: int, prev_last_upload: float, prev_last_step: int, prev_last_forced: float
) -> None:
    """Restore the throttle slot after a failed/abandoned upload, but only if this heartbeat still
    owns the latest claim — a newer heartbeat that bumped the slot after us must not be rolled back.
    Restores the committed-step marker too, so a failed forced commit doesn't permanently record its
    step as committed (which would wrongly stop the retry from forcing through), and the forced-commit
    clock, so a failed forced commit doesn't start the floor window (which would delay the retry)."""
    with _HB_LOCK:
        if my_claim == _HB_CLAIM_SEQ:
            _w._HB_LAST_UPLOAD = prev_last_upload
            _w._HB_LAST_COMMITTED_STEP = prev_last_step
            _w._HB_LAST_FORCED_UPLOAD = prev_last_forced


def _console_heartbeat_snapshot(payload: dict, payload_committed: bool = True) -> str:
    console_payload = dict(payload)
    metrics_last = console_payload.pop("metrics_last", None)
    if isinstance(metrics_last, list):
        console_payload["metrics_last_count"] = len(metrics_last)
    if not payload_committed and console_payload.get("sampled_completions"):
        console_payload.pop("sampled_completions", None)
    return json.dumps(console_payload)


def heartbeat(stage: str, *, liveness: bool = False, force: bool = False, **kw):
    global _HB_CLAIM_SEQ
    ts = time.time()
    # liveness pings don't count as progress; provider stall detection skips them.
    if not liveness:
        _w._HB_LAST_PROGRESS_TS = ts
    with _HB_LOCK:
        if not liveness:
            _w._HB_PROGRESS_SEQ += 1
        elif _w._HB_PROGRESS_SEQ > _w._HB_PROGRESS_UPLOADED_SEQ:
            # progress-carry: a real heartbeat since the last committed snapshot never reached HF
            # (throttled away or its upload failed). upgrade this ping to a real heartbeat so the
            # control plane's stall clock sees that progress instead of killing a healthy run.
            # deliberately after the _HB_LAST_PROGRESS_TS bump above: carried progress is not NEW
            # progress, so the worker's own stall-dump timer keeps its original reference point.
            liveness = False
        my_progress_seq = _w._HB_PROGRESS_SEQ
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
        if _is_critical_stage(stage):
            upload_due = True  # never miss a terminal transition
        elif _w._HB_TERMINAL_ONLY:
            upload_due = (
                _w._HB_LAST_UPLOAD == 0.0
                or (now - _w._HB_LAST_UPLOAD) >= _HB_TERMINAL_ONLY_INTERVAL_S
            )
        else:
            throttled = stage in _HB_THROTTLED_STAGES
            interval_s = _w._HB_MIN_INTERVAL_S
            if stage in _HB_TIGHT_LIVENESS_STAGES:
                interval_s = min(interval_s, _w._HB_SETUP_LIVENESS_INTERVAL_S)
            upload_due = not throttled or (now - _w._HB_LAST_UPLOAD) >= interval_s
            # ``force`` bypasses the per-stage throttle when this payload must be on record. it normally
            # requires a step advance, but a sample-bearing payload may match the committed step because
            # the liveness daemon can commit that step first without the samples. the per-force floor still
            # coalesces fast bursts to protect the hf commit cap, while an unrelated liveness commit does
            # not arm the floor and therefore cannot suppress the first sample-bearing payload.
            if force and not upload_due:
                fstep = kw.get("step")
                has_samples = bool(kw.get("sampled_completions"))
                force_step_due = isinstance(fstep, (int, float)) and (
                    fstep > _w._HB_LAST_COMMITTED_STEP
                    or (has_samples and fstep == _w._HB_LAST_COMMITTED_STEP)
                )
                if (
                    force_step_due
                    and (now - _w._HB_LAST_FORCED_UPLOAD) >= _w._HB_FORCE_MIN_INTERVAL_S
                ):
                    upload_due = True
        prev_last_upload = _w._HB_LAST_UPLOAD
        prev_last_step = _w._HB_LAST_COMMITTED_STEP
        prev_last_forced = _w._HB_LAST_FORCED_UPLOAD
        if upload_due:
            _HB_CLAIM_SEQ += 1
            my_claim = _HB_CLAIM_SEQ
            _w._HB_LAST_UPLOAD = now
            # Arm the forced-commit floor on ANY committing force=True heartbeat -- not only ones the
            # force branch let through. A force=True ping that commits because the regular throttle was
            # already due (900s elapsed) still refreshed the persisted step, so the NEXT sub-floor
            # forced ping must be coalesced; keying this off the force branch alone left the clock stale
            # and defeated the burst throttle that protects the HF commit cap (cursor[bot]). A non-forced
            # liveness/mid-step commit deliberately does NOT arm it, so a post-update force still punches
            # through immediately after one steals the slot.
            if force:
                _w._HB_LAST_FORCED_UPLOAD = now
            _committed_step = kw.get("step")
            if isinstance(_committed_step, (int, float)) and _committed_step > _w._HB_LAST_COMMITTED_STEP:
                _w._HB_LAST_COMMITTED_STEP = int(_committed_step)
    payload_committed = False
    if upload_due:
        critical = _is_critical_stage(stage)
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
                    # ``is False`` (not falsy) so a mock/None never trips the rollback.
                    _rollback_throttle_slot(my_claim, prev_last_upload, prev_last_step, prev_last_forced)
                    print(f"HEARTBEAT upload failed; rolled back throttle slot for {stage}")
                else:
                    payload_committed = True
                    if not liveness:
                        # this committed snapshot carried real progress; settle the progress-carry
                        # latch up to the seq captured when the snapshot was built (max: a concurrent
                        # newer real heartbeat that lost the upload race must stay pending).
                        with _HB_LOCK:
                            if my_progress_seq > _w._HB_PROGRESS_UPLOADED_SEQ:
                                _w._HB_PROGRESS_UPLOADED_SEQ = my_progress_seq
            finally:
                _HB_UPLOAD_LOCK.release()
        else:
            _rollback_throttle_slot(my_claim, prev_last_upload, prev_last_step, prev_last_forced)
            print(f"HEARTBEAT upload-lock busy >{lock_timeout}s; skipping commit for {stage}")
    print("HEARTBEAT", _console_heartbeat_snapshot(payload, payload_committed))
    return payload_committed


def _maybe_attach_gpu_diag(payload: dict, last_gpu_diag_at: float, now: float) -> float:
    """Attach GPU diagnostics to ``payload`` at most once per ``_STEP_GPU_DIAG_INTERVAL_S``.

    Returns the value to store back as ``last_gpu_diag_at``: ``now`` when diagnostics were
    attached this call, otherwise the unchanged prior timestamp.
    """
    if last_gpu_diag_at == 0.0 or now - last_gpu_diag_at >= _STEP_GPU_DIAG_INTERVAL_S:
        payload["gpu"] = gpu_diagnostics()
        return now
    return last_gpu_diag_at


_REWARD_METRIC_NAME_DISALLOWED = re.compile(r"[^A-Za-z0-9_.-]")
_REWARD_METRIC_RESERVED_NAMES = frozenset(
    {
        "reward",
        "reward_last",
        "step",
        "epoch",
        "loss",
        "grad_norm",
        "learning_rate",
        "stage",
        "gpu",
        "diag",
    }
)
_REWARD_METRIC_LIMIT = 12
# names TRAINING.md tells users to judge on: never dropped by the alphabetical cap.
_REWARD_METRIC_PRIORITY_NAMES = ("success",)


def _bounded_reward_metrics(metrics) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    surviving: dict[str, float] = {}
    for name, value in metrics.items():
        sanitized_name = _REWARD_METRIC_NAME_DISALLOWED.sub("", str(name))[:64]
        if not sanitized_name or sanitized_name in _REWARD_METRIC_RESERVED_NAMES:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        # distinct source names that sanitize to the same key must not silently overwrite each
        # other; disambiguate with a numeric suffix (kept within the 64-char bound, allowed chars).
        unique_name = sanitized_name
        suffix = 2
        while unique_name in surviving:
            tail = f"_{suffix}"
            unique_name = sanitized_name[: 64 - len(tail)] + tail
            suffix += 1
        surviving[unique_name] = score
    if len(surviving) <= _REWARD_METRIC_LIMIT:
        return dict(sorted(surviving.items()))
    # the cap must not drop metrics users are told to judge on (e.g. success); keep those first,
    # then fill the remaining slots alphabetically.
    priority = [n for n in _REWARD_METRIC_PRIORITY_NAMES if n in surviving]
    remaining = max(0, _REWARD_METRIC_LIMIT - len(priority))
    rest = sorted(n for n in surviving if n not in priority)[:remaining]
    return {n: surviving[n] for n in sorted(priority + rest)}


# Exactly three samples per heartbeat, always. Mirrors rollout_samples._SAMPLE_LIMIT.
_SAMPLE_LIMIT = 3


def _sampled_completion_scalar(sample: dict) -> tuple[str, float] | None:
    """Return the (key, finite value) of a sample's scalar: GRPO ``reward`` or OPD ``loss``."""
    for key in ("reward", "loss"):
        if key not in sample:
            continue
        try:
            value = float(sample.get(key))
        except (TypeError, ValueError):
            return None
        return (key, value) if math.isfinite(value) else None
    return None


def _bounded_sampled_completions(samples) -> list[dict]:
    if not isinstance(samples, (list, tuple)):
        return []
    bounded: list[dict] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        prompt_tail = sample.get("prompt_tail")
        completion = sample.get("completion")
        if not isinstance(prompt_tail, str) or not isinstance(completion, str):
            continue
        scalar = _sampled_completion_scalar(sample)
        if scalar is None:
            continue
        generated_at_step = sample.get("generated_at_step")
        if generated_at_step is not None:
            try:
                generated_at_step = int(generated_at_step)
            except (TypeError, ValueError):
                continue
        scalar_key, scalar_value = scalar
        bounded.append(
            {
                "prompt_tail": sanitize_rollout_text(prompt_tail),
                "completion": sanitize_rollout_text(completion),
                scalar_key: scalar_value,
                "generated_at_step": generated_at_step,
            }
        )
        if len(bounded) >= _SAMPLE_LIMIT:
            break
    return bounded


def make_reward_heartbeat_callback(reward_metrics=None, samples=None):
    """Return a TRL callback that streams per-step reward to the HF heartbeat channel."""
    from transformers import TrainerCallback

    class _RewardHeartbeat(TrainerCallback):
        def __init__(self):
            self.reward_history = []
            self.metrics_last = []
            self.last_gpu_diag_at = 0.0
            self.sent_first_sample_heartbeat = False
            self.latest_reward_metrics: dict[str, float] = {}

        def latest_fields(self) -> dict:
            if not self.latest_reward_metrics:
                return {}
            return {"reward_metrics": dict(self.latest_reward_metrics)}

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
            if not math.isfinite(r):
                return
            self.reward_history.append(r)
            step = int(getattr(state, "global_step", len(self.reward_history)))
            metrics = {"step": step, "reward": r}
            for payload_key, log_key in (
                ("reward_std", "reward_std"),
                ("grad_norm", "grad_norm"),
                ("kl", "kl"),
                ("entropy", "entropy"),
                ("frac_reward_zero_std", "frac_reward_zero_std"),
                ("mean_completion_tokens", "completions/mean_length"),
                ("truncation_rate", "completions/clipped_ratio"),
            ):
                value = logs.get(log_key)
                if value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                metrics[payload_key] = value
            max_completion_tokens = getattr(args, "max_completion_length", None)
            if max_completion_tokens is not None:
                with contextlib.suppress(TypeError, ValueError):
                    metrics["max_completion_tokens"] = int(max_completion_tokens)
            self.metrics_last = [item for item in self.metrics_last if item["step"] != step]
            self.metrics_last.append(metrics)
            self.metrics_last = self.metrics_last[-_GRPO_METRIC_HISTORY_LIMIT:]
            LATEST_GRPO_METRICS_LAST[:] = self.metrics_last
            payload = {
                **metrics,
                "reward_last": self.reward_history[-8:],
                "metrics_last": self.metrics_last,
            }
            latest_metrics = reward_metrics() if callable(reward_metrics) else reward_metrics
            self.latest_reward_metrics = _bounded_reward_metrics(latest_metrics)
            if self.latest_reward_metrics:
                payload["reward_metrics"] = dict(self.latest_reward_metrics)
            latest_samples = samples() if callable(samples) else samples
            bounded_samples = _bounded_sampled_completions(latest_samples)
            if bounded_samples:
                payload["sampled_completions"] = bounded_samples
            now = time.monotonic()
            self.last_gpu_diag_at = _maybe_attach_gpu_diag(payload, self.last_gpu_diag_at, now)
            force_first_samples = bool(bounded_samples) and not self.sent_first_sample_heartbeat
            if force_first_samples:
                committed = _w.heartbeat("rl_step", force=True, **payload)
                if committed:
                    self.sent_first_sample_heartbeat = True
            else:
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
            self.last_gpu_diag_at = _maybe_attach_gpu_diag(payload, self.last_gpu_diag_at, now)
            _w.heartbeat("sft_step", **{k: v for k, v in payload.items() if v is not None})

    return _SFTHeartbeat()


_LIVENESS_TICK_S = 30.0
_STALL_DUMP_S = 1200.0


@contextlib.contextmanager
def liveness_heartbeat(stage, progress=None, fields=None, progress_step=False, keepalive=False):
    """Emit liveness pings for ``stage`` while the wrapped block runs on the main thread.

    ``keepalive``: the wrapped block is legitimate BLOCKING I/O with no per-step progress signal the
    daemon can observe — a synchronous checkpoint/adapter upload (dev #445) or the finalize upload —
    so EVERY tick emits a REAL (non-liveness) heartbeat instead of a bare ping. Bare liveness pings do
    NOT advance the provider's stall clock (`_poll.surface_heartbeat` returns stage=None for them), so
    without this a healthy multi-minute upload that outlasts STALL_AFTER_S (1500s) is wrongly killed
    mid-save. Safe because a genuinely wedged upload surfaces as an EXCEPTION through its own retry
    budget (ending this context), not an infinite silent hang — so it does not mask a real stall. Pair
    with a throttled stage (see _HB_UPLOAD_LIVENESS_STAGES) so the 30s re-emit can't blow the HF cap.

    ``progress``: optional ``() -> float | None`` monotonic counter; advances emit a REAL heartbeat.
    ``fields``: optional ``() -> dict`` of EXTRA payload fields merged into every emission (liveness
    and progress alike). Use it to carry the billing/stall ``step`` on a stage the poller step-gates:
    without it this thread emits ``stage=<stage>`` with NO ``step``, and because it shares the
    ``opd_step`` upload-throttle slot it can win the slot and overwrite the main thread's stepped
    heartbeat -- ``actual_steps_run`` then sees a training-stage heartbeat with no step and floors a
    cancelled run to 1 step, mis-billing it (codex[bot]).
    ``progress_step``: the counter IS the trainer global step; stamp it as ``step`` on every emit so
    the poller's step gate and cancel billing see the true step even when this daemon wins the
    upload slot ahead of the trainer's own per-step callback (dev #442). ``fields`` (OPD's custom
    loop) and ``progress_step`` (sft/rl trainers) are complementary ways to carry the step; both are
    applied below, with ``progress_step`` winning if a caller somehow set both.
    Uses nvidia-smi-only diagnostics (main thread holds CUDA/allocator locks).
    """
    done = threading.Event()
    spawner = threading.current_thread()

    def _loop() -> None:
        if threading.current_thread() is spawner:
            # some tests stub threading.Thread to run targets INLINE on .start() (to make the
            # checkpoint-upload daemon synchronous). inlined, this loop would spin forever on
            # the caller's thread — a liveness daemon is only meaningful on its own thread.
            return
        last_val = None
        dumped = False
        while not done.wait(_LIVENESS_TICK_S):
            made_progress = False
            if progress is not None:
                with contextlib.suppress(Exception):
                    v = progress()
                    if v is not None:
                        if last_val is None:
                            # first sample is a BASELINE, not progress: a resumed run's restored
                            # global step must not tighten the provider's stall window seconds
                            # into train(), and a constant counter must never emit phantom
                            # progress. only a later ADVANCE past this baseline is real.
                            last_val = float(v)
                        elif float(v) > last_val:
                            last_val, made_progress = float(v), True
            gpu = gpu_diagnostics(include_torch=False)
            if done.is_set():  # the wrapped call may have finished during nvidia-smi
                return
            extra = {}
            if fields is not None:
                with contextlib.suppress(Exception):
                    extra = fields() or {}
            if progress_step and last_val is not None:
                extra["step"] = int(last_val)
            # keepalive: a legitimate blocking upload IS progress the daemon can't sample per-step, so
            # force a REAL heartbeat every tick to keep the provider's stall clock fed (see docstring).
            _w.heartbeat(stage, liveness=(not made_progress) and not keepalive, gpu=gpu, **extra)
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
