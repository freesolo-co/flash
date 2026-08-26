"""publish immutable worker progress records to the artifact repository."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from flash.engine.result.rollout_samples import (
    select_rollout_samples,
)
from flash.engine.worker.io import hf as hf_io
from flash.engine.worker.runtime import state as worker_state
from flash.engine.worker.verl.parent_work import ParentWorkGauge
from flash.runner.lifecycle.protocol import (
    ProgressRecord,
    bounded_json,
    canonical_bytes,
    digest_record,
    progress_path,
)

_PROGRESS_LOCK = threading.Lock()
_PROGRESS_DRAIN_LOCK = threading.Lock()
_PROGRESS_SEQUENCE = 0
_PROGRESS_PREVIOUS_DIGEST: str | None = None
_PROGRESS_TRAINING_ENTERED = False
_PROGRESS_COMPLETED_STEPS = 0
_PROGRESS_PENDING_CHECKPOINT_FAILURE: dict[str, int | str] | None = None
_PROGRESS_FATAL_ERROR: Exception | None = None
_PROGRESS_QUEUE = deque()
LATEST_GRPO_METRICS: list = []
GRPO_METRIC_HISTORY_LIMIT = 1024


class _ProgressUploadAmbiguous(RuntimeError):
    """an upload may have committed, but its exact immutable path could not be read back."""


@dataclass
class _PendingProgress:
    stage: str
    initial: bool
    fields: dict
    committed: bool = False
    record: ProgressRecord | None = None
    error: Exception | None = None


def _progress_kind(stage: str) -> str:
    if stage == "checkpoint_upload_failed":
        return "checkpoint_failed"
    if stage in {"checkpoint_uploaded", "checkpoint_deployable"}:
        return "checkpoint_saved"
    if stage.endswith("_start") or stage in {"boot", "model_prefetching"}:
        return "attempt_started" if stage == "boot" else "phase_changed"
    return "progressed"


def _progress_sections(fields: dict) -> tuple[dict, list, dict, dict, dict, dict]:
    metrics_keys = {
        "epoch",
        "discarded_rollouts",
        "entropy",
        "frac_reward_zero_std",
        "grad_norm",
        "kl",
        "learning_rate",
        "loss",
        "max_completion_tokens",
        "mean_completion_tokens",
        "reward",
        "reward_last",
        "reward_std",
        "truncation_rate",
    }
    timing_keys = {
        "projected_remaining_s",
        "setup_seconds",
        "step_duration_s",
        "train_wall",
        "wall_deadline_at_risk",
    }
    metrics = {key: fields[key] for key in metrics_keys if key in fields}
    reward_metrics = fields.get("reward_metrics")
    if isinstance(reward_metrics, dict):
        metrics["reward_metrics"] = reward_metrics
    timing = {key: fields[key] for key in timing_keys if key in fields}
    checkpoint = fields.get("checkpoint_failure")
    checkpoint = dict(checkpoint) if isinstance(checkpoint, dict) else {}
    gpu = fields.get("gpu")
    gpu = dict(gpu) if isinstance(gpu, dict) else {}
    samples = fields.get("sampled_completions")
    samples = list(samples) if isinstance(samples, list) else []
    known = (
        metrics_keys
        | timing_keys
        | {
            "checkpoint_failure",
            "gpu",
            "reward_metrics",
            "sampled_completions",
            "step",
        }
    )
    diagnostics = {key: value for key, value in fields.items() if key not in known}
    return metrics, samples, timing, checkpoint, gpu, diagnostics


def _write_local_immutable(payload: bytes) -> str:
    import hashlib
    import tempfile

    digest = hashlib.sha256(payload).hexdigest()
    directory = "/tmp/flash-progress"
    os.makedirs(directory, exist_ok=True)
    final = os.path.join(directory, digest + ".json")
    if os.path.exists(final):
        with open(final, "rb") as handle:
            if handle.read() != payload:
                raise RuntimeError("immutable progress digest conflict")
        return final
    fd, temporary = tempfile.mkstemp(dir=directory, prefix="progress-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final)
        return final
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _remote_record_payload(path: str) -> bytes | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    try:
        local = hf_hub_download(
            repo_id=worker_state.HF_REPO,
            repo_type="dataset",
            filename=path,
            token=os.environ.get("HF_TOKEN"),
            force_download=True,
        )
    except RemoteEntryNotFoundError:
        return None
    with open(local, "rb") as handle:
        return handle.read()


def _upload_record(record: ProgressRecord, *, required: bool) -> bool:
    payload = canonical_bytes(record.to_dict())
    local = _write_local_immutable(payload)
    path = progress_path(record)
    try:
        committed = hf_io.hf_upload_absolute(local, path, required=required)
    except Exception as upload_error:
        try:
            remote = _remote_record_payload(path)
        except Exception as verify_error:
            raise _ProgressUploadAmbiguous(
                "progress upload outcome is ambiguous because its immutable path could not be verified"
            ) from verify_error
        if remote == payload:
            return True
        if remote is not None:
            raise RuntimeError("immutable progress path contains different bytes") from upload_error
        raise upload_error
    if committed or not worker_state.HF_REPO:
        return committed
    remote = _remote_record_payload(path)
    if remote == payload:
        return True
    if remote is not None:
        raise RuntimeError("immutable progress path contains different bytes")
    return False


def pending_checkpoint_failure() -> dict[str, int | str] | None:
    """Return the checkpoint failure currently latched into cumulative progress."""
    with _PROGRESS_LOCK:
        return (
            dict(_PROGRESS_PENDING_CHECKPOINT_FAILURE)
            if _PROGRESS_PENDING_CHECKPOINT_FAILURE
            else None
        )


def _build_pending_record(pending: _PendingProgress) -> ProgressRecord:
    global _PROGRESS_COMPLETED_STEPS, _PROGRESS_PENDING_CHECKPOINT_FAILURE
    global _PROGRESS_TRAINING_ENTERED

    fields = dict(pending.fields)
    step = fields.get("step")
    if isinstance(step, (int, float)) and not isinstance(step, bool) and step >= 0:
        _PROGRESS_COMPLETED_STEPS = max(_PROGRESS_COMPLETED_STEPS, int(step))
    if pending.stage in {"rl_step", "sft_step", "opd_step"}:
        _PROGRESS_TRAINING_ENTERED = True
    if pending.stage == "checkpoint_upload_failed":
        failure = fields.get("checkpoint_failure")
        if isinstance(failure, dict):
            _PROGRESS_PENDING_CHECKPOINT_FAILURE = dict(failure)
    elif pending.stage == "checkpoint_uploaded":
        _PROGRESS_PENDING_CHECKPOINT_FAILURE = None
    if _PROGRESS_PENDING_CHECKPOINT_FAILURE and "checkpoint_failure" not in fields:
        fields["checkpoint_failure"] = dict(_PROGRESS_PENDING_CHECKPOINT_FAILURE)
    metrics, samples, timing, checkpoint, gpu, diagnostics = _progress_sections(fields)
    return ProgressRecord(
        run_id=worker_state.RUN_ID,
        phase_namespace=worker_state.PHASE,
        attempt_id=worker_state.ATTEMPT,
        fence=worker_state.FENCE,
        sequence=_PROGRESS_SEQUENCE + 1,
        previous_digest=_PROGRESS_PREVIOUS_DIGEST,
        occurred_at=time.time(),
        kind=_progress_kind(pending.stage),
        phase=pending.stage,
        training_entered=_PROGRESS_TRAINING_ENTERED,
        completed_steps=_PROGRESS_COMPLETED_STEPS,
        metrics=bounded_json(metrics),
        samples=bounded_json(samples),
        timing=bounded_json(timing),
        checkpoint=bounded_json(checkpoint),
        gpu_observation=bounded_json(gpu),
        diagnostics=bounded_json(diagnostics),
    )


def _drain_progress_until(target: _PendingProgress) -> None:
    global _PROGRESS_FATAL_ERROR, _PROGRESS_PREVIOUS_DIGEST, _PROGRESS_SEQUENCE

    with _PROGRESS_DRAIN_LOCK:
        while True:
            with _PROGRESS_LOCK:
                if not any(item is target for item in _PROGRESS_QUEUE):
                    return
                pending = _PROGRESS_QUEUE[0]
                record = _build_pending_record(pending)
                pending.record = record
            try:
                committed = _upload_record(record, required=pending.initial)
            except Exception as exc:
                with _PROGRESS_LOCK:
                    pending.error = exc
                    _PROGRESS_FATAL_ERROR = exc
                    while _PROGRESS_QUEUE:
                        blocked = _PROGRESS_QUEUE.popleft()
                        blocked.error = exc
                return
            with _PROGRESS_LOCK:
                _PROGRESS_QUEUE.popleft()
                pending.committed = committed
                if committed:
                    _PROGRESS_SEQUENCE = record.sequence
                    _PROGRESS_PREVIOUS_DIGEST = digest_record(record.to_dict())
            print("PROGRESS", json.dumps(record.to_dict(), allow_nan=False, sort_keys=True))
            if pending is target:
                return


def publish_progress(stage: str, *, initial: bool = False, **fields):
    """Publish one immutable cumulative progress record for observed work only."""
    pending = _PendingProgress(stage, initial, dict(fields))
    with _PROGRESS_LOCK:
        if _PROGRESS_FATAL_ERROR is not None:
            raise _PROGRESS_FATAL_ERROR
        _PROGRESS_QUEUE.append(pending)
    _drain_progress_until(pending)
    if pending.error is not None:
        raise pending.error
    return pending.committed


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
    Scoring and progress threads share one lock. Close counted generations on the scoring thread;
    asynchronous ``step:N`` stdout can mix N+1 into N, while progress cadence exposes partial data.
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
            # than hiding behind missing progress fields.
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

    def progress_fields(self) -> dict:
        """Return bounded reward metrics and sampled completions for one progress.
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
            # progress publication over a caller-supplied dict must not fail because one metric is
            # not representable as a finite float.
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


@contextlib.contextmanager
def observe_phase(stage, progress=None, fields=None, progress_step=False):
    """Record a phase transition and any cumulative work observed when it completes."""
    initial_fields = fields() if callable(fields) else {}
    if not isinstance(initial_fields, dict):
        initial_fields = {}
    publish_progress(stage, **initial_fields)
    baseline = None
    if progress is not None:
        with contextlib.suppress(Exception):
            baseline = progress()
    try:
        yield
    finally:
        current = None
        if progress is not None:
            with contextlib.suppress(Exception):
                current = progress()
        if current is not None and current != baseline:
            final_fields = fields() if callable(fields) else {}
            if not isinstance(final_fields, dict):
                final_fields = {}
            if progress_step:
                final_fields["step"] = int(current)
            publish_progress(stage, **final_fields)


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
