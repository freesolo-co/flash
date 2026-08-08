"""Worker heartbeat: stream stage/progress to the HF artifact repo.

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
from typing import Any

from flash.engine.rollout_samples import (
    sampled_completion_scalar,
    sanitize_rollout_text,
    select_rollout_samples,
)
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import gpu_diagnostics

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
        "sft_configuring",
        "rl_configuring",
        "opd_configuring",
        "opd_filtering_prompts",
        "opd_prompt_scan",
        "opd_image_prep",
        "sft_initializing",
        "rl_initializing",
        "opd_initializing",
        "sft_finalizing",
        "rl_finalizing",
        "opd_finalizing",
    }
)
# synchronous checkpoint uploads freeze global_step (dev #445), so checkpoint_uploading must
# advance the provider stall clock. keep it outside the setup set but on the same tight cadence.
_HB_UPLOAD_LIVENESS_STAGES = frozenset({"checkpoint_uploading"})
# Liveness stages that ride the tighter setup-liveness upload interval (setup + mid-train upload).
_HB_TIGHT_LIVENESS_STAGES = _HB_SETUP_LIVENESS_STAGES | _HB_UPLOAD_LIVENESS_STAGES

# latest per-step GRPO backlog, exposed so a top-level error heartbeat can preserve it
# for `flash runs log -f` when a short run raises before the throttled rl_step ping committed
LATEST_GRPO_METRICS_LAST: list = []
# throttle these stages to protect the hf repository commit budget; terminal transitions are never throttled.
# opd_filtering_prompts emits a real heartbeat on each scan tick, while opd_prompt_scan and
# opd_image_prep emit one when progress advances. opd_finalizing emits one on every keepalive tick.
# without throttling, these opd stages can make roughly 120 commit attempts per hour. tight-liveness,
# per-step training, and checkpoint_uploading keepalive stages share the same throttle.
_HB_THROTTLED_STAGES = _HB_TIGHT_LIVENESS_STAGES | frozenset({"rl_step", "sft_step", "opd_step"})
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
GRPO_METRIC_HISTORY_LIMIT = 1024


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


def heartbeat(
    stage: str, *, liveness: bool = False, force: bool = False, initial: bool = False, **kw
):
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
        # the initial training snapshot must land before the shared throttle can hide it.
        if initial:
            upload_due = True
        prev_last_upload = _w._HB_LAST_UPLOAD
        prev_last_step = _w._HB_LAST_COMMITTED_STEP
        prev_last_forced = _w._HB_LAST_FORCED_UPLOAD
        if upload_due:
            _HB_CLAIM_SEQ += 1
            my_claim = _HB_CLAIM_SEQ
            _w._HB_LAST_UPLOAD = now
            # any committing force=True heartbeat arms the burst floor, even when the regular
            # throttle was due. non-forced commits stay exempt so the next forced update lands.
            if force:
                _w._HB_LAST_FORCED_UPLOAD = now
            _committed_step = kw.get("step")
            if (
                isinstance(_committed_step, (int, float))
                and _committed_step > _w._HB_LAST_COMMITTED_STEP
            ):
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
                    if initial:
                        committed = _w.hf_upload_file(up, "heartbeat.json", required=True)
                    else:
                        committed = _w.hf_upload_file(up, "heartbeat.json")
                finally:
                    with contextlib.suppress(OSError):
                        os.remove(up)
                if committed is False:
                    # ``is False`` (not falsy) so a mock/None never trips the rollback.
                    _rollback_throttle_slot(
                        my_claim, prev_last_upload, prev_last_step, prev_last_forced
                    )
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
            if initial:
                raise _w.RetriableInfraError(
                    f"initial heartbeat upload lock remained busy >{lock_timeout}s for {stage}"
                )
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


class RewardObservabilityBuffer:
    """Buffer generation-scoped rollout samples and rewards from a child trainer.
    Scoring and heartbeat threads share one lock. Close counted generations on the scoring thread;
    asynchronous ``step:N`` stdout can mix N+1 into N, while heartbeat cadence exposes partial data.
    """

    _SAMPLE_BUFFER_LIMIT = 64
    # counted generations held waiting for their step lines. the queue exists because stdout can run
    # a generation or two behind; a child that stops printing step lines entirely would otherwise
    # grow it without bound, at up to _SAMPLE_BUFFER_LIMIT retained completions each.
    _SEALED_QUEUE_LIMIT = 8

    def __init__(self, *, generation_size: int = 0) -> None:
        self._lock = threading.Lock()
        # completions per generation, when the caller knows it: see the class docstring. 0 leaves
        # the boundary entirely caller-driven.
        self._generation_size = max(0, int(generation_size))
        self._scored_this_generation = 0
        # generations the count has already sealed, oldest first, each waiting for the step line
        # that names it. a QUEUE rather than a flag: stdout can fall a whole generation behind, and
        # a flag would let the second seal overwrite the first, dropping a generation and
        # relabelling the next one under its step.
        self._sealed_by_count: list[
            tuple[list[tuple[Any, Any, float]], dict[str, float] | None]
        ] = []
        # generations dropped from that queue before any step line named them. the step lines that
        # WOULD have named them still arrive, and each one has to be spent on the generation it
        # names, so the count is what keeps every later step matched to its own output.
        self._dropped_unnamed = 0
        # gradings since the last boundary; they belong to the generation still being scored.
        self._samples: list[tuple[Any, Any, float]] = []
        # the last COMPLETE generation, with the step it was logged under. only these publish.
        self._published: list[tuple[Any, Any, float]] = []
        self._published_step: int | None = None
        # running per-name sums and the completion count they are over, NOT the completions
        # themselves: see `record`.
        self._pending_totals: dict[str, float] = {}
        self._pending_count = 0
        self._latest_metrics: dict[str, float] = {}

    def record(self, prompt: Any, completion: Any, reward: float, breakdowns=()) -> None:
        """Buffer one rollout after grading; grading must not run under this lock.

        Fold breakdowns into per-name sums. Retaining bounded rows would silently evict valid
        large-batch completions and bias the mean; ``_bounded_reward_metrics`` already caps names.
        """
        with self._lock:
            self._samples.append((prompt, completion, float(reward)))
            del self._samples[: -self._SAMPLE_BUFFER_LIMIT]
            self._scored_this_generation += 1
            for breakdown in breakdowns:
                # a failed grading appends None and still counts: it is a real completion that
                # scored nothing, so it belongs in the denominator of every name (mirrors trl).
                self._pending_count += 1
                if not isinstance(breakdown, dict):
                    continue
                for name, value in breakdown.items():
                    if name == "total":
                        continue
                    self._pending_totals.setdefault(name, 0.0)
                    try:
                        score = float(value)
                    except (TypeError, ValueError, OverflowError):
                        # OverflowError too: an int larger than a float can hold raises it rather
                        # than ValueError. `_score` calls this OUTSIDE score_single_turn's guard, so
                        # anything escaping here 400s the reward request and aborts the run over a
                        # component that is only ever a diagnostic.
                        continue
                    if math.isfinite(score):
                        self._pending_totals[name] += score
            if self._generation_size and self._scored_this_generation >= self._generation_size:
                # the generation is complete BY COUNT, on the thread that completed it: nothing from
                # the next one can be folded in, whenever the child's stdout happens to arrive. it is
                # captured here and named later -- the step it belongs to is the child's to say.
                self._sealed_by_count.append(self._close())
                # a child that has stopped printing step lines is not going to claim these. drop the
                # OLDEST, the same direction `_samples` evicts: what survives is what the model is
                # doing now, rather than a window frozen at the moment stdout went quiet.
                dropped = len(self._sealed_by_count) - self._SEALED_QUEUE_LIMIT
                if dropped > 0:
                    del self._sealed_by_count[:dropped]
                    # their step lines are still coming. counting the drops lets `close_generation`
                    # spend one line per dropped generation instead of handing it the next survivor,
                    # which would offset every remaining step for the rest of the run.
                    self._dropped_unnamed += dropped

    def _close(self) -> tuple[list[tuple[Any, Any, float]], dict[str, float] | None]:
        """Take the open generation's samples and mean metrics while holding the lock.

        ``None`` means no breakdowns existed, unlike a generation whose breakdowns all failed.
        """
        metrics: dict[str, float] | None = None
        if self._pending_count:
            metrics = {
                name: total / self._pending_count for name, total in self._pending_totals.items()
            }
            self._pending_totals = {}
            self._pending_count = 0
        samples = self._samples
        self._samples = []
        self._scored_this_generation = 0
        return samples, metrics

    def _publish(
        self,
        samples: list[tuple[Any, Any, float]],
        metrics: dict[str, float] | None,
        *,
        step: int,
    ) -> None:
        """Make one closed generation the published reading, under ``step``. Caller holds the lock."""
        if metrics:
            self._latest_metrics = metrics
        elif metrics is not None and self._latest_metrics:
            # every completion failed scoring this generation: surface the known metrics as
            # zeros instead of dropping them, so a full scoring outage shows a flat 0 rather
            # than hiding behind missing heartbeat fields.
            self._latest_metrics = dict.fromkeys(self._latest_metrics, 0.0)
        if samples:
            self._published = samples
            self._published_step = int(step)

    def _seal(self, step: int) -> None:
        """Close the open generation and publish it as ``step``. Caller holds the lock."""
        self._publish(*self._close(), step=step)

    def close_generation(self, step: int) -> None:
        """Name the generation verl logged as ``step``, sealing it when needed.

        With ``generation_size`` this labels an already sealed generation; otherwise it closes one.
        A step with no gradings must not relabel older samples.
        """
        with self._lock:
            if self._dropped_unnamed:
                # this line names a generation the queue already dropped. it is spent here rather
                # than on the oldest survivor: that generation's own line is still to come, and
                # publishing it now would shift it and every one after it for the rest of the run.
                # the reading stays on the last generation that was named, which is stale by a known
                # number of steps rather than confidently wrong.
                self._dropped_unnamed -= 1
            elif self._sealed_by_count:
                # the count already sealed this step's generation, and what is open now belongs to a
                # LATER one -- sealing again here is exactly the leak this avoids. this line names
                # the oldest generation still waiting for one, so a stdout delivery that falls a
                # whole generation behind names them in the order they were produced instead of
                # overwriting the earlier one.
                self._publish(*self._sealed_by_count.pop(0), step=step)
            elif self._samples or self._pending_count:
                self._seal(step)
            # otherwise this step generated nothing, and the published rows stay under the step that
            # did produce them: relabelling would republish old text as freshly generated.

    def latest(self) -> tuple[Any, Any, float] | None:
        """Return one published sample for an unlabelled preview.
        Prefer published rows so late step lines cannot mislabel the next generation. Before the
        first publish, use the open generation; use ``latest_for_step`` when printing a step number.
        """
        with self._lock:
            if self._published:
                return self._published[-1]
            return self._samples[-1] if self._samples else None

    def latest_for_step(self, step: int) -> tuple[Any, Any, float] | None:
        """Return ``latest()`` only when its published rows belong to ``step``.

        Dropped generations leave the prior publish in place, so an unverified preview mislabels old
        rows. Open-generation rows have no valid step and are never returned here.
        """
        with self._lock:
            if self._published and self._published_step == int(step):
                return self._published[-1]
            return None

    def heartbeat_fields(self) -> dict:
        """Return bounded reward metrics and sampled completions for one heartbeat.
        Reads are non-destructive; ``close_generation`` owns draining. Metrics are bounded here and
        samples upstream. Omit empty signals so renderers can distinguish absence from emptiness.
        """
        with self._lock:
            # one acquisition covering both reads, or the payload tears: the two fields would
            # describe different generations. only the snapshot is taken here -- building the
            # samples sanitizes full untruncated text, which must not run while the scoring
            # threads want the lock.
            metrics = dict(self._latest_metrics)
            rows = list(self._published)
            step = self._published_step
        fields: dict = {}
        bounded_metrics = _bounded_reward_metrics(metrics)
        if bounded_metrics:
            fields["reward_metrics"] = bounded_metrics
        samples = select_rollout_samples(rows, generated_at_step=step)
        if samples:
            fields["sampled_completions"] = samples
        return fields


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
        except (TypeError, ValueError, OverflowError):
            # an int too large for a float raises OverflowError, not ValueError. this runs on the
            # heartbeat thread over a caller-supplied dict, so letting it out kills liveness
            # reporting for the rest of the run.
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
        scalar = sampled_completion_scalar(sample)
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
    """Emit liveness heartbeats while a main-thread block runs.
    ``keepalive`` marks blocking uploads as progress (dev #445); use a throttled stage. ``fields``
    carries payload data, while ``progress_step`` wins for trainer steps (dev #442). Missing a step
    can misbill cancellation. Use nvidia-smi because the main thread may hold CUDA allocator locks.
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


# checkpoint drain time scales with model size and network throughput; a fixed timeout killed a
# healthy upload in VERL-131. upload retries already obey FLASH_RUN_DEADLINE_AT, so the drain must
# add no tighter deadline.
_DRAIN_POLL_S = 5.0
# only used when the worker has no wall deadline configured (local runs, tests). generous: at this
# point the upload's own retry budget has long since been the real bound.
_DRAIN_NO_DEADLINE_MAX_S = 14400.0


def join_while_draining(thread: threading.Thread, what: str) -> None:
    """Wait for publishing until the run's wall deadline.

    Killing an in-budget drain discards completed work (VERL-131); only the run deadline stops it.
    """
    started = time.monotonic()
    while True:
        thread.join(timeout=_DRAIN_POLL_S)
        if not thread.is_alive():
            return
        remaining = _w._remaining_worker_wall_seconds()
        if remaining is None:
            # no deadline configured: fall back to an absolute ceiling so a hang in a local run
            # cannot wedge the process forever.
            if time.monotonic() - started > _DRAIN_NO_DEADLINE_MAX_S:
                raise RuntimeError(f"{what} did not finish within {_DRAIN_NO_DEADLINE_MAX_S:.0f}s")
        elif remaining <= 0:
            raise RuntimeError(f"{what} was still draining when the run wall deadline expired")
