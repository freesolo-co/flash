"""stream worker stage and progress heartbeats to the artifact repository."""

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

import flash.engine.worker.perf as worker_perf
from flash.engine.result.rollout_samples import (
    select_rollout_samples,
)
from flash.engine.worker.io import hf as hf_io
from flash.engine.worker.runtime import state as worker_state
from flash.engine.worker.verl.parent_work import ParentWorkGauge

_HB_LAST_UPLOAD = 0.0
_HB_LAST_PROGRESS_TS = 0.0
_HB_PROGRESS_SEQ = 0
_HB_PROGRESS_UPLOADED_SEQ = 0
_HB_PENDING_CHECKPOINT_FAILURE: dict[str, int | str] | None = None
_HB_MIN_INTERVAL_S = 900.0
_HB_LAST_COMMITTED_STEP = 0
_HB_LAST_FORCED_UPLOAD = 0.0
_HB_FORCE_MIN_INTERVAL_S = 60.0
_HB_SETUP_LIVENESS_INTERVAL_S = 240.0
_HB_TERMINAL_ONLY = False

# Setup-phase liveness stages: emitted from a 30s liveness thread WITH a progress callback during the
# cold download / model-load / split-scan phase, kept on the tighter setup-liveness upload cadence
# (parity with sft_pretokenizing) so the stall detector stays fed while nothing is training yet.
# _HB_THROTTLED_STAGES is DERIVED from this set below (⊇ dev #442's explicit list) so the union of
# per-arch stages stays single-source.
_HB_SETUP_LIVENESS_STAGES = frozenset(
    {
        "model_prefetching",
        "checkpoint_prefetching",
        # the post-download model setup span (adapter/tokenizer/architecture config reads). it is
        # emitted as a one-shot transition FIRST and then held open by a liveness wrap, so it has to
        # be throttled here like the other setup stages or a slow cold mount spends commits on it.
        # the one-shot transition itself is exempted in ``heartbeat`` (see _HB_MODEL_LOAD_STAGES):
        # only the repeated liveness ticks are what this throttle is for.
        "sft_model_load",
        "opd_model_load",
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
# stages whose one-shot transition must always commit even though their liveness ticks are
# throttled: the transition is the only record that the run reached the stage, and it is emitted
# right after a download that has been pinging throughout, so the throttle would almost always
# swallow it. the wrap that follows re-emits with liveness=True and stays throttled.
_HB_MODEL_LOAD_STAGES = frozenset({"sft_model_load", "opd_model_load"})
_HB_SUCCESS_TERMINAL_STAGES = frozenset({"done", "already_done"})
# 600s -> ~6 commits/hr; keeps stall detector alive without hitting the HF commit cap.
_HB_TERMINAL_ONLY_INTERVAL_S = 600.0


def _is_terminal_stage(stage: str) -> bool:
    """return whether this stage ends the worker attempt."""
    return stage in _HB_SUCCESS_TERMINAL_STAGES or stage.startswith("error_")


def _is_critical_stage(stage: str) -> bool:
    """A terminal transition or an error is CRITICAL: never throttled (the commit must land) and
    given the longer upload-lock timeout, because no later heartbeat can repair a missed one.

    `_failed` earns this as much as the `error_` prefix does: a failure heartbeat is emitted once, at
    the moment the failure is known, and nothing later restates it. `checkpoint_upload_failed` is
    raised from inside a long HF upload, which is precisely when `_HB_UPLOAD_IN_FLIGHT` is set and an
    unforced ping is dropped -- so the one report of the failure would be discarded exactly in the
    case that produces it.
    """
    return _is_terminal_stage(stage) or stage.endswith("_failed")


# Guards throttle bookkeeping; slow HF commit runs outside this lock so heartbeat and liveness
# threads don't block on the network.
_HB_LOCK = threading.Lock()
# Serializes HF commits to prevent reorder; each thread uploads its own per-call temp file.
_HB_UPLOAD_LOCK = threading.Lock()
# Terminal/error commits wait longer — no later heartbeat can repair them.
_HB_UPLOAD_LOCK_TIMEOUT_S = 30.0
_HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S = 120.0
# True while a thread holds _HB_UPLOAD_LOCK and is inside the HF commit. Read under _HB_LOCK.
#
# The throttle clock (_HB_LAST_UPLOAD) only advances AFTER a commit lands, so during a slow HF
# commit every other throttled caller still computes upload_due=True, walks into
# _HB_UPLOAD_LOCK.acquire() and blocks there for up to the acquire timeout -- 30s, or 120s on a
# critical stage. That stalls the liveness daemon, and `liveness_heartbeat`'s join delays leaving
# the wrapped stage by the same window.
#
# A marker rather than claiming the clock before the upload: an optimistic claim has to be rolled
# back when the commit fails, and the rollback has to decide whether a NEWER heartbeat already
# moved the clock (the reason the old code carried a claim sequence). This says only "a commit is
# in flight right now", is cleared in a finally, and so cannot strand the throttle on a failure.
_HB_UPLOAD_IN_FLIGHT = False
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


def _console_heartbeat_snapshot(
    payload: dict, payload_committed: bool = True, upload_due: bool = False
) -> str:
    """render one bounded console record with explicit hf commit state."""
    console_payload = dict(payload)
    metrics_last = console_payload.pop("metrics_last", None)
    if isinstance(metrics_last, list):
        console_payload["metrics_last_count"] = len(metrics_last)
    samples = console_payload.pop("sampled_completions", None)
    if isinstance(samples, list):
        console_payload["samples_count"] = len(samples)
    if not payload_committed:
        console_payload["pending" if upload_due else "throttled"] = True
    return json.dumps(console_payload)


def _set_upload_in_flight(value: bool) -> None:
    """Mark whether an HF heartbeat commit is running. Call with _HB_LOCK held."""
    global _HB_UPLOAD_IN_FLIGHT
    _HB_UPLOAD_IN_FLIGHT = value


def _heartbeat_upload_due(
    stage: str,
    *,
    liveness: bool,
    force: bool,
    initial: bool,
    first_timing: bool,
    fields: dict,
    now: float,
) -> bool:
    """return eligibility from committed throttle state while _HB_LOCK is held."""
    if initial or _is_critical_stage(stage):
        return True
    # a commit is already in flight and will advance the throttle clock when it lands. an unforced
    # ping would only queue behind it on _HB_UPLOAD_LOCK, publish a snapshot older than the one
    # being written, and block its caller for the acquire timeout -- 30s, or 120s on a critical
    # stage. that is what stalls the liveness daemon, whose `join` then delays leaving the wrapped
    # stage by the same window. skipping loses nothing: the in-flight commit carries this stage.
    #
    # three exemptions, all above or inside this check:
    #   - initial and critical stages return True earlier: a terminal or error snapshot has no later
    #     heartbeat to repair it, so it must queue and land.
    #   - `force` is excluded here because it carries billing state. a forced commit marks a
    #     DISTINCT completed step, and a cancel bills the last step a heartbeat recorded, so
    #     dropping one to avoid a wait could under-report a step the run really finished.
    # the daemon's own pings are never forced (liveness and keepalive both leave it False), so this
    # exemption does not reopen the stall it fixes.
    if _HB_UPLOAD_IN_FLIGHT and not force:
        return False
    if _HB_TERMINAL_ONLY:
        return _HB_LAST_UPLOAD == 0.0 or (now - _HB_LAST_UPLOAD) >= _HB_TERMINAL_ONLY_INTERVAL_S
    throttled = stage in _HB_THROTTLED_STAGES
    if stage in _HB_MODEL_LOAD_STAGES and not liveness:
        throttled = False
    interval_s = _HB_MIN_INTERVAL_S
    if stage in _HB_TIGHT_LIVENESS_STAGES:
        interval_s = min(interval_s, _HB_SETUP_LIVENESS_INTERVAL_S)
    upload_due = not throttled or (now - _HB_LAST_UPLOAD) >= interval_s
    if force and not upload_due:
        step = fields.get("step")
        has_samples = bool(fields.get("sampled_completions"))
        force_step_due = isinstance(step, (int, float)) and (
            step > _HB_LAST_COMMITTED_STEP or (has_samples and step == _HB_LAST_COMMITTED_STEP)
        )
        first_timing_due = first_timing and "step_duration_s" in fields
        force_floor_due = (now - _HB_LAST_FORCED_UPLOAD) >= _HB_FORCE_MIN_INTERVAL_S
        upload_due = force_step_due and (first_timing_due or force_floor_due)
    return upload_due


def heartbeat(
    stage: str,
    *,
    liveness: bool = False,
    force: bool = False,
    initial: bool = False,
    first_timing: bool = False,
    **kw,
):
    global _HB_LAST_COMMITTED_STEP, _HB_LAST_FORCED_UPLOAD, _HB_LAST_PROGRESS_TS
    global _HB_LAST_UPLOAD, _HB_PENDING_CHECKPOINT_FAILURE, _HB_PROGRESS_SEQ
    global _HB_PROGRESS_UPLOADED_SEQ
    genuine_progress = not liveness
    with _HB_LOCK:
        if stage == "checkpoint_upload_failed":
            failure = kw.get("checkpoint_failure")
            if isinstance(failure, dict):
                _HB_PENDING_CHECKPOINT_FAILURE = dict(failure)
        elif stage == "checkpoint_uploaded":
            # a later full resume checkpoint landed, so the earlier failure no longer describes the
            # run's outcome. deployable or final adapter publication cannot clear this because neither
            # restores the missing full resume state.
            _HB_PENDING_CHECKPOINT_FAILURE = None
        elif _is_terminal_stage(stage) and _HB_PENDING_CHECKPOINT_FAILURE:
            kw.setdefault("checkpoint_failure", dict(_HB_PENDING_CHECKPOINT_FAILURE))
        ts = time.time()
        if genuine_progress:
            _HB_LAST_PROGRESS_TS = ts
            _HB_PROGRESS_SEQ += 1
        elif _HB_PROGRESS_SEQ > _HB_PROGRESS_UPLOADED_SEQ:
            # carry real progress that has not reached hf yet.
            liveness = False
        latest_progress_ts = float(_HB_LAST_PROGRESS_TS or 0.0)
        my_progress_seq = _HB_PROGRESS_SEQ
    payload = {
        "stage": stage,
        "ts": ts,
        "run_id": worker_state.RUN_ID,
        "mode": worker_state.RUN_MODE,
        "seed": worker_state.SEED,
        "attempt": worker_state.ATTEMPT,
        **({"liveness": True} if liveness else {}),
        **kw,
    }
    if genuine_progress:
        payload["progress_age_s"] = 0.0
    elif latest_progress_ts:
        payload["progress_age_s"] = round(max(0.0, ts - latest_progress_ts), 1)
    else:
        payload.pop("progress_age_s", None)
    dc = os.environ.get("RUNPOD_DC_ID") or ""
    if dc:
        payload.setdefault("dc", dc)
    snapshot = json.dumps(payload)
    with _HB_LOCK:
        upload_due = _heartbeat_upload_due(
            stage,
            liveness=liveness,
            force=force,
            initial=initial,
            first_timing=first_timing,
            fields=kw,
            now=time.time(),
        )
    payload_committed = False
    if upload_due:
        critical = _is_critical_stage(stage)
        lock_timeout = _HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S if critical else _HB_UPLOAD_LOCK_TIMEOUT_S
        if _HB_UPLOAD_LOCK.acquire(timeout=lock_timeout):
            try:
                with _HB_LOCK:
                    upload_due = _heartbeat_upload_due(
                        stage,
                        liveness=liveness,
                        force=force,
                        initial=initial,
                        first_timing=first_timing,
                        fields=kw,
                        now=time.time(),
                    )
                if upload_due:
                    up = f"/tmp/.hb-upload-{os.getpid()}-{threading.get_ident()}.json"
                    with open(up, "w") as f:
                        f.write(snapshot)
                    with _HB_LOCK:
                        _set_upload_in_flight(True)
                    try:
                        if initial:
                            committed = hf_io.hf_upload_file(up, "heartbeat.json", required=True)
                        else:
                            committed = hf_io.hf_upload_file(up, "heartbeat.json")
                    finally:
                        # cleared before the throttle clock is advanced below, and in a finally so a
                        # raising upload cannot leave every later heartbeat permanently skipped.
                        with _HB_LOCK:
                            _set_upload_in_flight(False)
                        with contextlib.suppress(OSError):
                            os.remove(up)
                    if committed is False:
                        print(f"HEARTBEAT upload failed for {stage}")
                    else:
                        payload_committed = True
                        with _HB_LOCK:
                            committed_at = time.time()
                            _HB_LAST_UPLOAD = committed_at
                            if force:
                                _HB_LAST_FORCED_UPLOAD = committed_at
                            committed_step = kw.get("step")
                            if (
                                isinstance(committed_step, (int, float))
                                and committed_step > _HB_LAST_COMMITTED_STEP
                            ):
                                _HB_LAST_COMMITTED_STEP = int(committed_step)
                            if not liveness and my_progress_seq > _HB_PROGRESS_UPLOADED_SEQ:
                                _HB_PROGRESS_UPLOADED_SEQ = my_progress_seq
            finally:
                _HB_UPLOAD_LOCK.release()
        else:
            if initial:
                raise worker_perf.RetriableInfraError(
                    f"initial heartbeat upload lock remained busy >{lock_timeout}s for {stage}"
                )
            print(f"HEARTBEAT upload-lock busy >{lock_timeout}s; skipping commit for {stage}")
    print("HEARTBEAT", _console_heartbeat_snapshot(payload, payload_committed, upload_due))
    return payload_committed


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
        self.parent_work = ParentWorkGauge()
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
        self.parent_work.complete()
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
        work = self.parent_work.snapshot()
        fields["reward_completions"] = work.completed
        fields["reward_grading_depth"] = work.depth
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


_LIVENESS_TICK_S = 30.0
_STALL_DUMP_S = 1200.0


class _OffThreadFieldSampler:
    def __init__(self, fields, done: threading.Event, spawner: threading.Thread) -> None:
        self._fields = fields
        self._done = done
        self._spawner = spawner
        self._lock = threading.Lock()
        self._latest: dict = {}
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        if threading.current_thread() is self._spawner:
            return
        while not self._done.is_set():
            try:
                sampled = self._fields() or {}
            except Exception:
                sampled = None
            if sampled is not None:
                with self._lock:
                    self._latest = dict(sampled)
            if self._done.wait(_LIVENESS_TICK_S):
                return

    def latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def join(self) -> None:
        self._thread.join(timeout=_HB_UPLOAD_LOCK_TIMEOUT_S)


@contextlib.contextmanager
def liveness_heartbeat(
    stage,
    progress=None,
    fields=None,
    progress_step=False,
    keepalive=False,
    sample_off_thread=False,
):
    """Emit liveness heartbeats while a main-thread block runs.
    ``keepalive`` marks blocking uploads as progress (dev #445); use a throttled stage. ``fields``
    carries payload data, while ``progress_step`` wins for trainer steps (dev #442). Missing a step
    can misbill cancellation. Use nvidia-smi because the main thread may hold CUDA allocator locks.
    """
    done = threading.Event()
    spawner = threading.current_thread()
    field_sampler = (
        _OffThreadFieldSampler(fields, done, spawner)
        if sample_off_thread and fields is not None
        else None
    )
    if field_sampler is not None:
        field_sampler.start()

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
            gpu = worker_perf.gpu_diagnostics(include_torch=False)
            if done.is_set():  # the wrapped call may have finished during nvidia-smi
                return
            extra = field_sampler.latest() if field_sampler is not None else {}
            if fields is not None and field_sampler is None:
                with contextlib.suppress(Exception):
                    extra = fields() or {}
            if progress_step and last_val is not None:
                extra["step"] = int(last_val)
            # keepalive: a legitimate blocking upload IS progress the daemon can't sample per-step, so
            # force a REAL heartbeat every tick to keep the provider's stall clock fed (see docstring).
            heartbeat(stage, liveness=(not made_progress) and not keepalive, gpu=gpu, **extra)
            last_progress = float(_HB_LAST_PROGRESS_TS or 0.0)
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
        if field_sampler is not None:
            field_sampler.join()


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
        remaining = worker_state._remaining_worker_wall_seconds()
        if remaining is None:
            # no deadline configured: fall back to an absolute ceiling so a hang in a local run
            # cannot wedge the process forever.
            if time.monotonic() - started > _DRAIN_NO_DEADLINE_MAX_S:
                raise RuntimeError(f"{what} did not finish within {_DRAIN_NO_DEADLINE_MAX_S:.0f}s")
        elif remaining <= 0:
            raise RuntimeError(f"{what} was still draining when the run wall deadline expired")
