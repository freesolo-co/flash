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
from typing import Any

from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import gpu_diagnostics
from flash.engine.worker.rollout_samples import sanitize_rollout_text, select_rollout_samples

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
_HB_UPLOAD_LIVENESS_STAGES = frozenset({"checkpoint_uploading"})
# Liveness stages that ride the tighter setup-liveness upload interval (setup + mid-train upload).
_HB_TIGHT_LIVENESS_STAGES = _HB_SETUP_LIVENESS_STAGES | _HB_UPLOAD_LIVENESS_STAGES

# latest per-step GRPO backlog, exposed so a top-level error heartbeat can preserve it
# for `flash runs log -f` when a short run raises before the throttled rl_step ping committed
LATEST_GRPO_METRICS_LAST: list = []
# Throttled to avoid blowing the 128/hr HF commit cap; terminal transitions are never throttled. Every
# tight-liveness stage is throttled (⊂) PLUS the per-step training stages: opd_filtering_prompts alone
# emits a REAL (non-liveness) heartbeat every scan tick — ~120/hr on a large split before model load —
# so unthrottled the setup stages blow the cap; throttle them exactly like their sft_pretokenizing
# analogue (codex[bot]). checkpoint_uploading keepalive re-emits every 30s too, so it MUST be throttled.
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
                    if initial:
                        committed = _w.hf_upload_file(up, "heartbeat.json", required=True)
                    else:
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


def _mean_named_reward_metrics(breakdowns: list[dict[str, float] | None]) -> dict[str, float]:
    totals: dict[str, float] = {}
    denominator = len(breakdowns)
    for breakdown in breakdowns:
        if not isinstance(breakdown, dict):
            continue
        for name, value in breakdown.items():
            if name == "total":
                continue
            totals.setdefault(name, 0.0)
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                totals[name] += score
    if denominator == 0:
        return {}
    return {name: total / denominator for name, total in totals.items()}


def _latest_named_reward_metrics(
    breakdowns: list[dict[str, float] | None], latest: dict[str, float]
) -> dict[str, float]:
    if breakdowns:
        metrics = _mean_named_reward_metrics(breakdowns)
        breakdowns.clear()
        if metrics:
            latest.clear()
            latest.update(metrics)
        elif latest:
            # every completion failed scoring this generation: surface the known metrics as
            # zeros instead of dropping them, so a full scoring outage shows a flat 0 rather
            # than hiding behind missing heartbeat fields.
            latest.update(dict.fromkeys(latest, 0.0))
    return dict(latest)


class RewardObservabilityBuffer:
    """Rolling rollout samples and per-name reward components for an out-of-process trainer.

    trl publishes both signals from a TrainerCallback running on the trainer's own thread. verl's
    trainer is a child process, so its reward bridge fills this from the scoring server's threads
    while the heartbeat drains it from the liveness thread -- the lock below is the whole reason
    this is an object rather than three locals.

    Both signals describe a GENERATION, so both are published per generation rather than per
    heartbeat. trl gets that boundary for free (its callback fires at ``on_log``, after the step is
    scored); here the caller supplies it via ``close_generation``. Draining on the heartbeat cadence
    instead would publish whichever completions happened to be graded when a 30s tick landed --
    a latency-biased subset -- and stamp samples left over from earlier generations with the
    current step (codex[bot]).

    ``generation_size`` makes that boundary COUNTED rather than observed. The verl caller knows a
    generation is exactly ``prompts_per_step * group_size`` completions, so the last one closes it
    on the scoring thread itself. Closing on the arriving ``step:N`` stdout line instead would be a
    race: the child's pipe is delivered asynchronously, so generation N+1 can already be scoring
    into this buffer when the parent finally reads N's line, sealing both generations under step N
    and leaving N+1 to republish it (codex[bot]).
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
        # a flag would let the second seal overwrite the first, dropping a generation and relabelling
        # the next one under its step (cursor, codex[bot]).
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
        """Buffer one scored rollout. Call AFTER grading: this takes the lock, grading must not.

        ``breakdowns`` is the 0-or-1 element accumulator ``score_single_turn`` filled for this
        completion, empty for a multi-turn episode (the env scores a whole episode to a scalar).

        The breakdown is folded into a running sum instead of being retained. A generation is
        ``[train].batch_size * group_size`` completions, both arbitrary positive integers, so any
        retention bound is one a valid large-batch run can exceed -- and evicting to honour it drops
        completions out of the mean silently, biasing the published metric toward whichever ones
        happened to be graded last (codex[bot]). Sums cost one float per NAME, which
        ``_bounded_reward_metrics`` already caps at 12, so the generation size stops mattering.
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
                        # component that is only ever a diagnostic (codex[bot]).
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
                    # which would offset every remaining step for the rest of the run (cursor,
                    # codex[bot]).
                    self._dropped_unnamed += dropped

    def _close(self) -> tuple[list[tuple[Any, Any, float]], dict[str, float] | None]:
        """Take the open generation's samples and mean metrics. Caller holds the lock.

        The metrics are ``None`` when this generation counted no breakdowns at all -- a multi-turn
        episode grades to a scalar and never reports named components -- which is NOT the same as
        counting completions that all failed to report one. See ``_publish``.
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
        """Name the generation verl logged as ``step``, sealing it if the count has not already.

        Call once per new trainer step. When ``generation_size`` is known this is a RELABEL: the
        scoring thread already sealed on the last completion, and this only corrects the ordinal to
        the step verl printed (they agree unless verl skipped or resumed). When it is not, this is
        the boundary itself.

        A boundary with no gradings leaves the previous generation published under its OWN step: a
        step that generated nothing must not relabel older samples as newly generated.
        """
        with self._lock:
            if self._dropped_unnamed:
                # this line names a generation the queue already dropped. it is spent here rather
                # than on the oldest survivor: that generation's own line is still to come, and
                # publishing it now would shift it and every one after it for the rest of the run.
                # the reading stays on the last generation that was named, which is stale by a known
                # number of steps rather than confidently wrong (cursor, codex[bot]).
                self._dropped_unnamed -= 1
            elif self._sealed_by_count:
                # the count already sealed this step's generation, and what is open now belongs to
                # a LATER one -- sealing again here is exactly the leak this avoids. this line names
                # the oldest generation still waiting for one, so a stdout delivery that falls a
                # whole generation behind names them in the order they were produced instead of
                # overwriting the earlier one (cursor, codex[bot]).
                self._publish(*self._sealed_by_count.pop(0), step=step)
            elif self._samples or self._pending_count:
                self._seal(step)
            # otherwise this step generated nothing, and the published rows stay under the step that
            # did produce them: relabelling would republish old text as freshly generated.

    def latest(self) -> tuple[Any, Any, float] | None:
        """One ``(prompt, completion, reward)`` from the PUBLISHED generation, for a step preview.

        The caller prints this under the step it just closed, so it reads what that step published
        rather than what is being scored now. Those differ exactly when the step line is late: the
        next generation is already recording, and preferring it would label its completion with the
        previous step's number -- the same mislabelling the queue exists to prevent, reintroduced one
        line later, and disagreeing with the heartbeat over the very same step (codex[bot]).

        Falls back to the open generation only before anything has been published, so a caller that
        previews before the first boundary still sees a rollout instead of nothing.
        """
        with self._lock:
            if self._published:
                return self._published[-1]
            return self._samples[-1] if self._samples else None

    def heartbeat_fields(self) -> dict:
        """The bounded ``reward_metrics`` / ``sampled_completions`` fragment of one heartbeat.

        Non-destructive: every heartbeat between two generations republishes that generation's
        reading. ``close_generation`` owns the drain.

        Metrics are bounded here (name sanitization, 12-metric cap). Samples are not: unlike the trl
        callback -- which bounds an opaque caller-supplied list -- these come straight from
        ``select_rollout_samples``, which already sanitizes, drops non-finite scalars, and caps at
        three, so ``_bounded_sampled_completions`` would be a no-op over its own output.

        Empty signals are omitted rather than sent as ``{}``/``[]``: a renderer tells "no metrics
        this step" apart from "this backend does not report them" by the key's absence.
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
            self.metrics_last = self.metrics_last[-GRPO_METRIC_HISTORY_LIMIT:]
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
