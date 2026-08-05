"""Flash OPD orchestration through verl 0.8.0 in an isolated child interpreter."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import http.client
import json
import math
import os
import random
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from flash.engine.recipe import RECIPE
from flash.engine.steps import (
    final_save_due,
    on_policy_steps,
    resolve_update_horizon,
    validate_save_steps,
)
from flash.engine.structured_outputs import reasoning_parser_for
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.backend_common import (
    BoundedThreadingHTTPServer,
    ChildOutputTail,
    ChildTailStaleness,
    agent_loop_workers,
    clamp_engine_len,
    latest_global_step_dir,
    model_max_position_embeddings,
    parse_verl_metric,
    parse_wandb_link,
    ray_num_cpus,
    render_wandb_link_shim,
    resolve_blackwell_attention_backends,
    resolve_rollout_enforce_eager,
    resolve_verl_device_capability,
    resolve_verl_loggers,
    resolve_verl_python,
    rollout_resident_overrides,
    rollout_sleep_unsupported,
    run_verl_training,
    stall_tail_fields,
    trainer_dtype_overrides,
    verl_step_number,
)
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.multiturn_glue import (
    EnvGlueTokenizer,
    dedup_seam_terminator,
    validate_glue_template,
    validate_transcript_messages,
)
from flash.engine.worker.opd import (
    _drop_fully_forced_groups,
    _resolve_opd_knobs,
    _thinking_prefill_text,
)
from flash.engine.worker.opd_gkd import (
    _rollout_terminated,
    _teacher_prompt_text,
    _trim_trailing_stop,
    generation_eos_from_cached_config,
    student_tokens_with_offsets,
)
from flash.engine.worker.rng import seed_training_rngs
from flash.engine.worker.sft_train import (
    _build_verl_child_env,
    _cached_model_path,
    _durable_required_save_steps,
    _export_checkpoint_adapter,
    _hydra_val,
    _materialize_verl_images,
    _multimodal_messages_with_images,
    _NvidiaSmiPeakSampler,
    _probe_gpu_in_subprocess,
    _verl_image_message_content,
    _VerlCheckpointWatcher,
    _warmstart_adapter_path,
)
from flash.engine.worker.teacher import (
    _MAX_LOGPROB_ROUNDING_ERROR,
    TeacherError,
    TeacherScore,
)
from flash.engine.worker.tokenizer_align import (
    TeacherToken,
    groupwise_alignment,
    groupwise_coverage,
)
from flash.opd_limits import OPD_TEACHER_SCORING_CONCURRENCY
from flash.opd_retry_contract import OPD_RESUME_STATE_VERSION, validate_opd_resume_state_metadata

_PERMANENT_TEACHER_EXIT = 86
_TRANSIENT_TEACHER_EXIT = 87
_TEXT_TEACHER_FLUSH_WAIT_S = 0.1
_TEXT_TEACHER_SHUTDOWN_WAIT_S = 5.0
_TEXT_TEACHER_REQUEST_BACKLOG = 64

# opd supervises the teacher's distribution, not a task reward, so every rollout scores zero. the
# score is unreachable either way: use_task_rewards=false makes verl zero the whole policy loss
# (distillation/losses.py:211), so nothing a scorer returns can enter the gradient. this exists
# only to keep the reward loop out of its builtin data_source registry -- see the call site.
_OPD_ZERO_REWARD_SOURCE = '''"""flash opd reward shim (generated). opd carries no task reward."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return 0.0
'''


def _align_granularity(groups, student_tokens) -> float:
    """mean student-tokens-per-group: the alignment-health signal coverage cannot provide.

    a degenerate alignment that collapses every student token onto a single group still scores
    coverage ~1.0, so coverage alone never flags it. this is the trl opd formula verbatim
    (``opd.py``): non-empty student spans over surviving groups, counted after fully-forced groups
    are dropped so the ratio describes the signal that actually reaches the loss.
    """
    if not groups:
        return 0.0
    n_align = sum(1 for st in student_tokens if st.end > st.start)
    return n_align / len(groups)


class _TeacherBridgeHTTPServer(BoundedThreadingHTTPServer):
    request_queue_size = _TEXT_TEACHER_REQUEST_BACKLOG


@dataclass
class _TextTeacherWaiter:
    item: tuple[str, str]
    enqueued_at: float
    done: threading.Event = field(default_factory=threading.Event)
    result: TeacherScore | None = None
    error: Exception | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def complete(
        self,
        *,
        result: TeacherScore | None = None,
        error: Exception | None = None,
    ) -> None:
        with self._lock:
            if self.done.is_set():
                return
            self.result = result
            self.error = error
            self.done.set()

    def wait(self) -> TeacherScore:
        self.done.wait()
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError("text teacher batch waiter completed without a result")
        return self.result


def _teacher_batch_error(error: Exception) -> Exception:
    if isinstance(error, TeacherError):
        return TeacherError(str(error), permanent=error.permanent)
    return RuntimeError(str(error))


def _validate_text_teacher_batch(
    scored,
    items: list[tuple[str, str]],
) -> list[TeacherScore]:
    expected = len(items)
    if not isinstance(scored, list) or len(scored) != expected:
        actual = len(scored) if isinstance(scored, list) else type(scored).__name__
        raise TeacherError(
            f"teacher text batch returned {actual} result(s) for {expected} unique input(s)",
            permanent=True,
        )
    for result_index, (score, (_prompt_text, completion_text)) in enumerate(
        zip(scored, items, strict=True)
    ):
        if not isinstance(score, TeacherScore):
            raise TeacherError(
                f"teacher text batch result {result_index} is not a TeacherScore",
                permanent=True,
            )
        previous_start = -1
        previous_end = -1
        for token_index, token in enumerate(score.tokens):
            if not isinstance(token, TeacherToken):
                raise TeacherError(
                    f"teacher text batch result {result_index} contains an invalid token",
                    permanent=True,
                )
            if not isinstance(token.text, str):
                raise TeacherError(
                    f"teacher text batch result {result_index} token {token_index} has invalid text",
                    permanent=True,
                )
            if (
                isinstance(token.logprob, bool)
                or not isinstance(token.logprob, int | float)
                or not math.isfinite(token.logprob)
                or token.logprob > _MAX_LOGPROB_ROUNDING_ERROR
            ):
                raise TeacherError(
                    f"teacher text batch result {result_index} token {token_index} has invalid logprob",
                    permanent=True,
                )
            if (
                isinstance(token.start, bool)
                or isinstance(token.end, bool)
                or not isinstance(token.start, int)
                or not isinstance(token.end, int)
                or token.start < 0
                or token.end < token.start
                or token.end > len(completion_text)
                or token.start < previous_start
                or token.end < previous_end
            ):
                raise TeacherError(
                    f"teacher text batch result {result_index} token {token_index} has invalid offsets",
                    permanent=True,
                )
            previous_start = token.start
            previous_end = token.end
        if score.input_tokens <= 0 or score.output_tokens != 1:
            raise TeacherError(
                f"teacher text batch result {result_index} is missing authoritative token usage",
                permanent=True,
            )
    return scored


class _TextTeacherBatcher:
    def __init__(
        self,
        teacher,
        *,
        max_batch_size: int = OPD_TEACHER_SCORING_CONCURRENCY,
        flush_wait_s: float = _TEXT_TEACHER_FLUSH_WAIT_S,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("text teacher batch size must be positive")
        if flush_wait_s <= 0:
            raise ValueError("text teacher flush wait must be positive")
        self.teacher = teacher
        self.max_batch_size = int(max_batch_size)
        self.flush_wait_s = float(flush_wait_s)
        self._condition = threading.Condition()
        self._pending: list[_TextTeacherWaiter] = []
        self._in_flight: list[_TextTeacherWaiter] = []
        self._closed = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("text teacher batcher is closed")
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="flash-opd-text-teacher-batcher",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def score(self, prompt_text: str, completion_text: str) -> TeacherScore:
        with self._condition:
            if self._closed:
                raise TeacherError("text teacher batcher shut down", permanent=True)
            waiter = _TextTeacherWaiter(
                (prompt_text, completion_text),
                enqueued_at=time.monotonic(),
            )
            self._pending.append(waiter)
            self._condition.notify_all()
        return waiter.wait()

    def _take_batch(self) -> list[_TextTeacherWaiter] | None:
        with self._condition:
            while not self._pending:
                if self._closed:
                    return None
                self._condition.wait()
            deadline = self._pending[0].enqueued_at + self.flush_wait_s
            while len(self._pending) < self.max_batch_size and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._closed:
                return None
            batch = self._pending[: self.max_batch_size]
            del self._pending[: len(batch)]
            self._in_flight = batch
            return batch

    def _score_batch(self, batch: list[_TextTeacherWaiter]) -> None:
        unique_items: list[tuple[str, str]] = []
        item_indexes: dict[tuple[str, str], int] = {}
        scatter_indexes: list[int] = []
        for waiter in batch:
            index = item_indexes.get(waiter.item)
            if index is None:
                index = len(unique_items)
                item_indexes[waiter.item] = index
                unique_items.append(waiter.item)
            scatter_indexes.append(index)
        scored = _validate_text_teacher_batch(
            self.teacher.score_many(unique_items),
            unique_items,
        )
        billed_indexes: set[int] = set()
        for waiter, index in zip(batch, scatter_indexes, strict=True):
            result = scored[index]
            if index in billed_indexes:
                result = result.without_billing()
            else:
                billed_indexes.add(index)
            waiter.complete(result=result)

    def _run(self) -> None:
        try:
            while True:
                batch = self._take_batch()
                if batch is None:
                    return
                try:
                    self._score_batch(batch)
                except Exception as error:
                    for waiter in batch:
                        waiter.complete(error=_teacher_batch_error(error))
                finally:
                    with self._condition:
                        self._in_flight = []
                        self._condition.notify_all()
        finally:
            error = TeacherError("text teacher batcher stopped", permanent=True)
            with self._condition:
                stranded = [*self._pending, *self._in_flight]
                self._pending.clear()
                self._in_flight = []
                self._closed = True
                self._condition.notify_all()
            for waiter in stranded:
                waiter.complete(error=_teacher_batch_error(error))

    def close(self, timeout_s: float = _TEXT_TEACHER_SHUTDOWN_WAIT_S) -> None:
        error = TeacherError("text teacher batcher shut down", permanent=True)
        with self._condition:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()
            thread = self._thread
        for waiter in pending:
            waiter.complete(error=_teacher_batch_error(error))
        if thread is not None:
            thread.join(timeout=max(0.0, timeout_s))
        with self._condition:
            in_flight = list(self._in_flight)
        for waiter in in_flight:
            waiter.complete(error=_teacher_batch_error(error))


class _RecordedMutationCallbackFailure(RuntimeError):
    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


@dataclass(frozen=True)
class _BridgePrompt:
    student_messages: list[dict]
    teacher_messages: list[dict]
    prompt_ids: tuple[int, ...]
    image_descriptors: tuple[str, ...]
    package_root: str | None
    example: dict | None = None


def _prompt_pool_fingerprint(prompts: list[_BridgePrompt]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        fingerprint_fields = [prompt.student_messages, list(prompt.prompt_ids)]
        if prompt.example is not None:
            fingerprint_fields.append(prompt.example)
        if prompt.image_descriptors:
            fingerprint_fields.extend([prompt.teacher_messages, list(prompt.image_descriptors)])
        payload = json.dumps(
            fingerprint_fields,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _normalize_prompt_ids(value) -> tuple[int, ...]:
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if value and isinstance(value[0], list | tuple):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError("processor prompt input_ids must be list-like")
    return tuple(
        int(token_id.item() if hasattr(token_id, "item") else token_id) for token_id in value
    )


def _processor_expanded_prompt_ids(
    processor,
    messages: list[dict],
    image_descriptors: tuple[str, ...],
    package_root: str | None,
    *,
    enable_thinking: bool,
) -> tuple[int, ...]:
    from flash.multimodal import decode_image_descriptors

    images = decode_image_descriptors(list(image_descriptors), package_root)
    prepared = _multimodal_messages_with_images(messages, images)
    raw_prompt = processor.apply_chat_template(
        prepared,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = processor(
        text=[raw_prompt],
        images=images,
        videos=None,
        return_tensors="pt",
    )
    return _normalize_prompt_ids(model_inputs)


def encode_shifted_group_metadata(
    prompt_length: int,
    response_length: int,
    groups,
) -> tuple[list[int], list[float]]:
    """Encode response group metadata on verl's full-sequence one-token-shift layout."""
    if prompt_length <= 0:
        raise ValueError("flash OPD prompts must contain at least one token")
    response_group_ids = [-1] * response_length
    response_teacher_logsums = [0.0] * response_length
    for local_group_id, (student_indices, teacher_logsum) in enumerate(groups):
        for student_index in student_indices:
            if student_index < 0 or student_index >= response_length:
                raise ValueError("flash OPD alignment group index is outside the response")
            if response_group_ids[student_index] != -1:
                raise ValueError("flash OPD response token belongs to multiple alignment groups")
            response_group_ids[student_index] = local_group_id
            response_teacher_logsums[student_index] = float(teacher_logsum)
    teacher_ids = [-1] * (prompt_length - 1) + response_group_ids + [-1]
    teacher_logprobs = [0.0] * (prompt_length - 1) + response_teacher_logsums + [0.0]
    if len(teacher_ids) != prompt_length + response_length:
        raise AssertionError("flash OPD shifted teacher metadata has the wrong length")
    return teacher_ids, teacher_logprobs


def _validate_forced_mask(
    forced,
    response_length: int,
    *,
    required: bool,
) -> list[bool]:
    if not required:
        return []
    if forced is None:
        raise ValueError("structured OPD bridge payload is missing the forced mask")
    if not isinstance(forced, list) or any(type(value) is not bool for value in forced):
        raise ValueError("structured OPD bridge forced mask must contain only booleans")
    if len(forced) != response_length:
        raise ValueError(
            "structured OPD bridge forced mask length does not match the untrimmed response"
        )
    return forced


def _trim_response_and_forced(
    tokenizer,
    response_ids: list[int],
    stop_text: str,
    stop_sequences: tuple[str, ...],
    forced: list[bool],
) -> tuple[list[int], str, list[bool]]:
    kept_ids, completion_text = _trim_trailing_stop(
        tokenizer, response_ids, stop_text, stop_sequences
    )
    return kept_ids, completion_text, forced[: len(kept_ids)]


class _TeacherAlignmentBridge:
    def __init__(
        self,
        *,
        prompts: list[_BridgePrompt],
        tokenizer,
        teacher,
        thinking_prefill: str,
        eos_token_ids: frozenset[int],
        stop_sequences: tuple[str, ...],
        mutation_callback,
        structured: bool = False,
        active_env=None,
        multi_turn: bool = False,
        max_turns: int = 0,
        thinking: bool = False,
        session_lease_s: float = 1800.0,
        session_reap_interval_s: float = 30.0,
        initial_state: dict | None = None,
    ) -> None:
        state = initial_state or {}
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.teacher = teacher
        self.thinking_prefill = thinking_prefill
        self.eos_token_ids = eos_token_ids
        self.stop_sequences = stop_sequences
        self.structured = bool(structured)
        self.active_env = active_env
        self.multi_turn = bool(multi_turn)
        self.max_turns = int(max_turns)
        self._env_glue = (
            EnvGlueTokenizer(tokenizer, thinking=bool(thinking)) if self.multi_turn else None
        )
        self.session_lease_s = float(session_lease_s)
        self.session_reap_interval_s = float(session_reap_interval_s)
        if self.multi_turn and self.session_lease_s <= 0:
            raise ValueError("multi-turn OPD session lease must be positive")
        if self.multi_turn and self.session_reap_interval_s <= 0:
            raise ValueError("multi-turn OPD session reaper interval must be positive")
        self.mutation_callback = mutation_callback
        self.token = hashlib.sha256(os.urandom(32)).hexdigest()
        self._server = None
        self._thread = None
        self._text_teacher_batcher: _TextTeacherBatcher | None = None
        self._env_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._sessions: dict[str, dict] = {}
        self._session_tombstones: dict[str, float] = {}
        self._session_reaper_stop = threading.Event()
        self._session_reaper_thread = None
        self._mutation_lock = threading.Lock()
        self._mutation_notified = False
        self._stats_lock = threading.Lock()
        self.generated_tokens = int(state.get("generated_tokens", 0))
        self.teacher_input_tokens = int(state.get("teacher_input_tokens", 0))
        self.teacher_output_tokens = int(state.get("teacher_output_tokens", 0))
        self.aligned_sequences = int(state.get("aligned_sequences", state.get("granularity_n", 0)))
        self.empty_alignments = int(
            state.get(
                "empty_alignments", dict(state.get("skip_counts", {})).get("empty_alignment", 0)
            )
        )
        self.truncated_rollouts = int(state.get("truncated_rollouts", 0))
        self.forced_tokens = int(state.get("forced_tokens", 0))
        self.dropped_forced_groups = int(state.get("dropped_forced_groups", 0))
        self.coverage_sum = float(state.get("coverage_sum", state.get("granularity_sum", 0.0)))
        # alignment GRANULARITY (mean aligned-groups-per-sequence), distinct from coverage: a
        # collapsed alignment that maps every student token onto one group still scores coverage
        # ~1.0, so coverage alone cannot flag that failure mode. no legacy alias here -- the old
        # granularity_* state keys held coverage, so reading them would restore the wrong quantity.
        self.align_group_sum = float(state.get("align_group_sum", 0.0))
        self.align_group_n = int(state.get("align_group_n", 0))
        # resume: baseline the per-step delta counters at the restored cumulative mass, so the
        # first resumed step reports its own coverage instead of the whole prior run's.
        self._prev_aligned = self.aligned_sequences
        self._prev_cov_sum = self.coverage_sum
        self.teacher_ok = int(state.get("teacher_ok", 0))
        self.teacher_transient = int(state.get("teacher_transient", 0))
        self.teacher_error = int(state.get("teacher_error", 0))
        self.score_requests = int(state.get("samples_seen", 0))
        self.episodes_seen = int(state.get("episodes_seen", self.score_requests))
        self.mt_turn_records = int(state.get("mt_turn_records", 0))
        self.no_signal_resamples = int(state.get("no_signal_resamples", 0))
        self.no_signal_skipped_steps = int(state.get("no_signal_skipped_steps", 0))
        self.skip_counts = dict(state.get("skip_counts", {}))
        self.opd_phase_seconds = dict(state.get("opd_phase_seconds", {}))
        self.opd_phase_counts = dict(state.get("opd_phase_counts", {}))
        self._teacher_failure: tuple[str, str] | None = None
        self._mutation_failure: tuple[str, str] | None = None
        self._mutation_callback_failure: tuple[str, str] | None = None
        self._mutation_callback_succeeded = False
        self._pending_teacher_transient: tuple[str, str] | None = None
        self._pending_teacher_success = False

    def _record_teacher_failure(
        self,
        classification: str,
        message: str,
        *,
        terminal: bool = False,
    ) -> None:
        with self._stats_lock:
            if classification == "transient":
                self.teacher_transient += 1
                if terminal and self._teacher_failure is None:
                    self._teacher_failure = (classification, message)
                elif self._pending_teacher_transient is None:
                    self._pending_teacher_transient = (classification, message)
            else:
                self.teacher_error += 1
                self._teacher_failure = (classification, message)

    @property
    def teacher_failure(self) -> tuple[str, str] | None:
        with self._stats_lock:
            return self._teacher_failure

    def _promote_recovered_teacher_failure(self, failure: tuple[str, str]) -> None:
        with self._stats_lock:
            if self._teacher_failure is None:
                self._teacher_failure = failure

    def _record_teacher_delivery_failure(self, error: Exception) -> None:
        with self._stats_lock:
            if self._teacher_failure is None:
                self._teacher_failure = (
                    "transient",
                    f"teacher bridge response delivery failed: {type(error).__name__}",
                )

    def _record_mutation_failure(self, classification: str, message: str) -> None:
        with self._stats_lock:
            if self._mutation_callback_failure is not None:
                return
            if self._mutation_callback_succeeded:
                return
            if classification == "permanent" or self._mutation_failure is None:
                self._mutation_failure = (classification, message)

    def _record_mutation_callback_failure(
        self,
        classification: str,
        message: str,
    ) -> tuple[str, str]:
        with self._stats_lock:
            if self._mutation_callback_failure is None:
                self._mutation_callback_failure = (classification, message)
            return self._mutation_callback_failure

    @staticmethod
    def _raise_recorded_mutation_failure(failure: tuple[str, str]) -> None:
        classification, message = failure
        raise _RecordedMutationCallbackFailure(classification, message)

    @property
    def mutation_failure(self) -> tuple[str, str] | None:
        with self._stats_lock:
            if self._mutation_callback_failure is not None:
                return self._mutation_callback_failure
            if self._mutation_callback_succeeded:
                return None
            return self._mutation_failure

    def _promote_pending_teacher_failure(self) -> bool:
        with self._stats_lock:
            if (
                self._teacher_failure is None
                and self._pending_teacher_transient is not None
                and not self._pending_teacher_success
            ):
                self._teacher_failure = self._pending_teacher_transient
                self._pending_teacher_transient = None
                self._pending_teacher_success = False
                return True
            return False

    def accounting_snapshot(self) -> dict:
        with self._stats_lock:
            skip_counts = dict(self.skip_counts)
            skip_counts["empty_alignment"] = self.empty_alignments
            return {
                "generated_tokens": self.generated_tokens,
                "teacher_input_tokens": self.teacher_input_tokens,
                "teacher_output_tokens": self.teacher_output_tokens,
                "truncated_rollouts": self.truncated_rollouts,
                "forced_tokens": self.forced_tokens,
                "dropped_forced_groups": self.dropped_forced_groups,
                "granularity_n": self.aligned_sequences,
                "samples_seen": self.score_requests,
                "teacher_ok": self.teacher_ok,
                "teacher_transient": self.teacher_transient,
                "teacher_error": self.teacher_error,
                "no_signal_resamples": self.no_signal_resamples,
                "no_signal_skipped_steps": self.no_signal_skipped_steps,
                "episodes_seen": self.episodes_seen,
                "mt_turn_records": self.mt_turn_records,
                "granularity_sum": self.coverage_sum,
                "skip_counts": skip_counts,
                "opd_phase_seconds": dict(self.opd_phase_seconds),
                "opd_phase_counts": dict(self.opd_phase_counts),
                "aligned_sequences": self.aligned_sequences,
                "empty_alignments": self.empty_alignments,
                "coverage_sum": self.coverage_sum,
                "align_group_sum": self.align_group_sum,
                "align_group_n": self.align_group_n,
            }

    def _empty(self, prompt_length: int, response_length: int) -> dict:
        teacher_ids, teacher_logprobs = encode_shifted_group_metadata(
            prompt_length, response_length, []
        )
        return {"teacher_ids": teacher_ids, "teacher_logprobs": teacher_logprobs}

    def score(
        self,
        index: int,
        prompt_length: int,
        sequence_ids: list[int],
        image_count: int = 0,
        forced=None,
        *,
        recovered_failure: list[tuple[str, str]] | None = None,
    ) -> dict:
        with self._stats_lock:
            self.score_requests += 1
            self.episodes_seen += 1
        if index < 0 or index >= len(self.prompts):
            raise ValueError("flash OPD bridge received an unknown dataset index")
        prompt = self.prompts[index]
        expected_image_count = len(prompt.image_descriptors)
        if int(image_count) != expected_image_count:
            raise ValueError(
                f"verl rollout reported {int(image_count)} image(s) for dataset index {index}; "
                f"the frozen prompt has {expected_image_count}"
            )
        if expected_image_count:
            raise ValueError("image-bearing opd is not supported by managed Parasail teachers")
        prompt_ids = list(prompt.prompt_ids)
        prompt_length = int(prompt_length)
        sequence_ids = [int(token_id) for token_id in sequence_ids]
        if prompt_length != len(prompt_ids) or sequence_ids[:prompt_length] != prompt_ids:
            raise ValueError(
                "verl rollout prompt ids do not exactly match the frozen flash prompt pool"
            )
        response_ids = sequence_ids[prompt_length:]
        forced = _validate_forced_mask(
            forced,
            len(response_ids),
            required=self.structured,
        )
        with self._stats_lock:
            self.generated_tokens += len(response_ids)
            self.forced_tokens += sum(forced)
        if not response_ids:
            return self._empty(prompt_length, 0)
        stop_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        if not _rollout_terminated(
            response_ids, stop_text, self.eos_token_ids, self.stop_sequences
        ):
            with self._stats_lock:
                self.truncated_rollouts += 1
            return self._empty(prompt_length, len(response_ids))
        kept_ids, completion_text, kept_forced = _trim_response_and_forced(
            self.tokenizer,
            response_ids,
            stop_text,
            self.stop_sequences,
            forced,
        )
        if not completion_text.strip() or "�" in completion_text:
            return self._empty(prompt_length, len(response_ids))
        teacher_prompt = _teacher_prompt_text(prompt.teacher_messages, self.thinking_prefill)
        try:
            if self._text_teacher_batcher is None:
                teacher_score = self.teacher.score(teacher_prompt, completion_text)
            else:
                teacher_score = self._text_teacher_batcher.score(teacher_prompt, completion_text)
        except TeacherError as error:
            if error.permanent:
                raise
            failure = ("transient", str(error))
            self._record_teacher_failure(*failure)
            if recovered_failure is not None:
                recovered_failure.append(failure)
            return self._empty(prompt_length, len(response_ids))
        teacher_input_tokens = teacher_score.input_tokens
        teacher_output_tokens = teacher_score.output_tokens
        if not (
            (teacher_input_tokens > 0 and teacher_output_tokens == 1)
            or (teacher_input_tokens == 0 and teacher_output_tokens == 0)
        ):
            raise RuntimeError("teacher score has invalid authoritative Parasail token usage")
        with self._stats_lock:
            self.teacher_ok += 1
            self._pending_teacher_success = True
        _student_ids, student_tokens = student_tokens_with_offsets(
            self.tokenizer, kept_ids, completion_text
        )
        groups = groupwise_alignment(student_tokens, teacher_score.tokens)
        groups = [(indices, logsum) for indices, logsum in groups if indices]
        aligned_group_count = len(groups)
        groups = _drop_fully_forced_groups(groups, kept_forced)
        coverage = groupwise_coverage(groups, student_tokens)
        granularity = _align_granularity(groups, student_tokens)
        with self._stats_lock:
            self.teacher_input_tokens += teacher_input_tokens
            self.teacher_output_tokens += teacher_output_tokens
            self.dropped_forced_groups += aligned_group_count - len(groups)
            self.coverage_sum += coverage
            if groups:
                self.aligned_sequences += 1
                self.align_group_sum += granularity
                self.align_group_n += 1
            else:
                self.empty_alignments += 1
        teacher_ids, teacher_logprobs = encode_shifted_group_metadata(
            prompt_length, len(response_ids), groups
        )
        return {"teacher_ids": teacher_ids, "teacher_logprobs": teacher_logprobs}

    def _require_multiturn(self) -> None:
        if not self.multi_turn or self.active_env is None:
            raise ValueError("flash OPD bridge multi-turn mode is not enabled")
        if self.max_turns <= 0:
            raise ValueError("flash OPD bridge multi-turn limit is invalid")

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
            raise ValueError("flash OPD bridge received an invalid rollout session id")
        return session_id

    def _reap_stale_sessions_locked(self, now: float) -> None:
        stale = [
            session_id
            for session_id, session in self._sessions.items()
            if session["lease_deadline"] <= now
        ]
        for session_id in stale:
            self._sessions.pop(session_id, None)
            self._session_tombstones[session_id] = now + self.session_lease_s
        expired_tombstones = [
            session_id
            for session_id, deadline in self._session_tombstones.items()
            if deadline <= now
        ]
        for session_id in expired_tombstones:
            self._session_tombstones.pop(session_id, None)

    def _reap_stale_sessions(self, now: float | None = None) -> None:
        with self._sessions_lock:
            self._reap_stale_sessions_locked(time.monotonic() if now is None else float(now))

    def _session_reaper_loop(self) -> None:
        while not self._session_reaper_stop.wait(self.session_reap_interval_s):
            self._reap_stale_sessions()

    @property
    def active_session_count(self) -> int:
        with self._sessions_lock:
            self._reap_stale_sessions_locked(time.monotonic())
            return len(self._sessions)

    def _session(self, session_id: str) -> dict:
        session_id = self._validate_session_id(session_id)
        with self._sessions_lock:
            now = time.monotonic()
            self._reap_stale_sessions_locked(now)
            session = self._sessions.get(session_id)
            if session is not None:
                session["lease_deadline"] = now + self.session_lease_s
        if session is None:
            raise ValueError("flash OPD bridge received an unknown rollout session id")
        return session

    def start_multiturn(
        self,
        *,
        index: int,
        session_id: str,
        prompt_ids: list[int],
        raw_prompt: list[dict],
    ) -> dict:
        self._require_multiturn()
        if index < 0 or index >= len(self.prompts):
            raise ValueError("flash OPD bridge received an unknown dataset index")
        prompt = self.prompts[index]
        if prompt.example is None:
            raise ValueError("multi-turn OPD prompt is missing its environment example")
        prompt_ids = [int(token_id) for token_id in prompt_ids]
        if prompt_ids != list(prompt.prompt_ids):
            raise ValueError("multi-turn rollout prompt ids do not match the frozen flash prompt")
        raw_prompt = validate_transcript_messages(raw_prompt, source="child initial prompt")
        if raw_prompt != prompt.student_messages:
            raise ValueError("multi-turn child prompt does not match the frozen environment prompt")
        session_id = self._validate_session_id(session_id)
        start_identity = (
            int(index),
            tuple(prompt_ids),
            json.dumps(raw_prompt, sort_keys=True, separators=(",", ":")),
        )
        with self._sessions_lock:
            now = time.monotonic()
            self._reap_stale_sessions_locked(now)
            if session_id in self._session_tombstones:
                raise ValueError("flash OPD bridge rollout session was already closed")
            existing = self._sessions.get(session_id)
            if existing is not None:
                if existing["start_identity"] != start_identity:
                    raise ValueError("flash OPD bridge rollout session id was reused")
                existing["lease_deadline"] = now + self.session_lease_s
                return {"max_turns": existing["turn_limit"]}
            with self._env_lock:
                state = self.active_env.new_rollout_state(prompt.example)
                initial_messages = state.get("prompt") or state.get("messages")
                initial_messages = validate_transcript_messages(
                    initial_messages, source="environment initial prompt"
                )
            if initial_messages != prompt.student_messages:
                raise ValueError(
                    "multi-turn environment initial prompt changed after prompt freezing"
                )
            per_example_limit = state.get("max_episode_turns")
            if per_example_limit is not None:
                try:
                    per_example_limit = int(per_example_limit)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "multi-turn environment returned an invalid per-example turn limit"
                    ) from error
                if per_example_limit <= 0:
                    raise ValueError(
                        "multi-turn environment requires a positive per-example turn limit"
                    )
            turn_limit = min(self.max_turns, per_example_limit or self.max_turns)
            self._sessions[session_id] = {
                "index": int(index),
                "state": state,
                "messages": [dict(message) for message in initial_messages],
                "turns": [],
                "required_prefix": list(prompt.prompt_ids),
                "terminal": False,
                "turn_limit": turn_limit,
                "start_identity": start_identity,
                "score_cache": None,
                "score_lock": threading.Lock(),
                "lease_deadline": now + self.session_lease_s,
            }
        with self._stats_lock:
            self.episodes_seen += 1
        return {"max_turns": turn_limit}

    def _validated_multiturn_response(self, payload: dict) -> tuple[list[int], list[int], str, str]:
        raw_response_ids = [int(token_id) for token_id in payload.get("raw_response_ids", [])]
        response_ids = [int(token_id) for token_id in payload.get("response_ids", [])]
        completion_text = payload.get("completion_text")
        if not isinstance(completion_text, str):
            raise ValueError("multi-turn assistant completion text must be a string")
        termination = str(payload.get("termination") or "")
        truncated = bool(payload.get("truncated"))
        skip_reason = str(payload.get("skip_reason") or "")
        if skip_reason not in {
            "",
            "empty_completion",
            "replacement_char",
            "truncated_rollout",
        }:
            raise ValueError("multi-turn assistant turn has an unknown skip reason")
        if truncated:
            if termination != "truncated" or skip_reason != "truncated_rollout":
                raise ValueError("multi-turn truncated assistant turn has inconsistent metadata")
            if response_ids != raw_response_ids:
                raise ValueError(
                    "multi-turn truncated assistant ids must preserve the sampled span"
                )
        elif termination == "eos":
            if self.eos_token_ids.isdisjoint(raw_response_ids):
                raise ValueError("multi-turn eos termination is not present in the sampled ids")
            if response_ids != raw_response_ids:
                raise ValueError("multi-turn eos response ids must preserve the sampled span")
        elif termination == "stop":
            stop_text = self.tokenizer.decode(raw_response_ids, skip_special_tokens=False)
            expected_ids, expected_text = _trim_trailing_stop(
                self.tokenizer, raw_response_ids, stop_text, self.stop_sequences
            )
            if expected_ids != response_ids or expected_text != completion_text:
                raise ValueError("multi-turn stop trimming does not match the legacy OPD contract")
        elif termination == "accepted_stop":
            max_tokens = int(payload.get("max_tokens", 0))
            if (
                payload.get("stop_reason") != "completed"
                or max_tokens <= 0
                or len(raw_response_ids) >= max_tokens
                or response_ids != raw_response_ids
            ):
                raise ValueError("multi-turn accepted-stop metadata is not verifiable")
        else:
            raise ValueError("multi-turn assistant turn did not end at a verified boundary")
        decoded = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        if decoded != completion_text:
            raise ValueError("multi-turn assistant text does not match its accepted token span")
        return raw_response_ids, response_ids, completion_text, skip_reason

    def step_multiturn(self, payload: dict) -> dict:
        self._require_multiturn()
        session = self._session(payload.get("session_id"))
        turn_ordinal = int(payload.get("turn_ordinal", -1))
        accepted_prefix = [int(token_id) for token_id in payload.get("accepted_prefix", [])]
        raw_response_ids, response_ids, completion_text, skip_reason = (
            self._validated_multiturn_response(payload)
        )
        request_identity = (
            turn_ordinal,
            tuple(accepted_prefix),
            tuple(raw_response_ids),
            tuple(response_ids),
            completion_text,
            str(payload.get("termination") or ""),
            str(payload.get("stop_reason") or ""),
            int(payload.get("max_tokens", 0)),
            bool(payload.get("truncated")),
            skip_reason,
        )
        with self._env_lock:
            if turn_ordinal < len(session["turns"]):
                prior = session["turns"][turn_ordinal]
                if prior["request_identity"] != request_identity:
                    raise ValueError("multi-turn rollout assistant turn ordinal was reused")
                return copy.deepcopy(prior["step_response"])
            if session["terminal"]:
                raise ValueError("multi-turn rollout attempted to step a terminal session")
            if turn_ordinal != len(session["turns"]):
                raise ValueError("multi-turn rollout assistant turn ordinal is out of order")
            required_prefix = session["required_prefix"]
            if accepted_prefix != required_prefix:
                raise ValueError(
                    "multi-turn rollout prompt does not exactly match the authenticated environment context"
                )
            context_messages = [dict(message) for message in session["messages"]]
            state = session["state"]
            self.active_env.record_model_turn(state, completion_text)
            session["messages"].append({"role": "assistant", "content": completion_text})
            terminal = bool(payload.get("truncated")) or bool(skip_reason)
            messages: list[dict] = []
            next_prefix = [*accepted_prefix, *response_ids]
            if not terminal:
                # check episode termination BEFORE requesting an environment reply: at the turn
                # limit or when the env already reports done, the extra env_reply both wastes an
                # env call and appends a user turn no model turn will ever answer.
                assistant_turns = turn_ordinal + 1
                turn_limit = session["turn_limit"]
                terminal = assistant_turns >= turn_limit or self.active_env.rollout_done(
                    state, turn_limit
                )
            if not terminal:
                messages = self.active_env.env_reply(session["messages"], state)
                messages = validate_transcript_messages(messages, source="environment reply")
                session["messages"].extend(messages)
                # the env's reply may itself end the episode (rollout_done consults the updated
                # state); recheck before gluing a next-turn prompt no model turn will answer.
                terminal = not messages or self.active_env.rollout_done(
                    state, session["turn_limit"]
                )
                if not terminal:
                    assert self._env_glue is not None
                    next_prefix.extend(
                        dedup_seam_terminator(response_ids, self._env_glue(messages))
                    )
            step_response = {"messages": messages, "terminal": bool(terminal)}
            session["terminal"] = bool(terminal)
            session["required_prefix"] = next_prefix
            session["score_cache"] = None
            session["turns"].append(
                {
                    "prompt_ids": accepted_prefix,
                    "response_ids": response_ids,
                    "raw_response_ids": raw_response_ids,
                    "completion_text": completion_text,
                    "context_messages": context_messages,
                    "truncated": bool(payload.get("truncated")),
                    "skip_reason": skip_reason,
                    "request_identity": request_identity,
                    "step_response": copy.deepcopy(step_response),
                }
            )
        with self._stats_lock:
            self.score_requests += 1
            self.mt_turn_records += 1
            self.generated_tokens += len(response_ids)
            if payload.get("truncated"):
                self.truncated_rollouts += 1
            if skip_reason:
                self.skip_counts[skip_reason] = self.skip_counts.get(skip_reason, 0) + 1
        return step_response

    def score_multiturn(self, session_id: str) -> dict:
        self._require_multiturn()
        session = self._session(session_id)
        with session["score_lock"]:
            turn_count = len(session["turns"])
            cached = session["score_cache"]
            if cached is not None and cached["turn_count"] == turn_count:
                return copy.deepcopy(cached["result"])
            turns = list(session["turns"])
            results = [
                self._empty(len(turn["prompt_ids"]), len(turn["response_ids"])) for turn in turns
            ]
            scorable = [
                position
                for position, turn in enumerate(turns)
                if not turn["truncated"] and not turn["skip_reason"] and turn["response_ids"]
            ]
            if scorable:
                items = [
                    (
                        _teacher_prompt_text(
                            turns[position]["context_messages"], self.thinking_prefill
                        ),
                        turns[position]["completion_text"],
                    )
                    for position in scorable
                ]
                teacher_batches = []
                for start in range(0, len(items), OPD_TEACHER_SCORING_CONCURRENCY):
                    teacher_batches.extend(
                        self.teacher.score_many(
                            items[start : start + OPD_TEACHER_SCORING_CONCURRENCY]
                        )
                    )
                if len(teacher_batches) != len(scorable):
                    raise RuntimeError("teacher returned the wrong number of multi-turn OPD scores")
                with self._stats_lock:
                    self.teacher_ok += len(teacher_batches)
                for position, teacher_score in zip(scorable, teacher_batches, strict=True):
                    teacher_input_tokens = teacher_score.input_tokens
                    teacher_output_tokens = teacher_score.output_tokens
                    if teacher_input_tokens <= 0 or teacher_output_tokens != 1:
                        raise RuntimeError(
                            "teacher score is missing authoritative Parasail token usage"
                        )
                    turn = turns[position]
                    response_ids = turn["response_ids"]
                    _student_ids, student_tokens = student_tokens_with_offsets(
                        self.tokenizer, response_ids, turn["completion_text"]
                    )
                    groups = groupwise_alignment(student_tokens, teacher_score.tokens)
                    groups = [(indices, logsum) for indices, logsum in groups if indices]
                    coverage = groupwise_coverage(groups, student_tokens)
                    granularity = _align_granularity(groups, student_tokens)
                    with self._stats_lock:
                        self.teacher_input_tokens += teacher_input_tokens
                        self.teacher_output_tokens += teacher_output_tokens
                        self.coverage_sum += coverage
                        if groups:
                            self.aligned_sequences += 1
                            self.align_group_sum += granularity
                            self.align_group_n += 1
                        else:
                            self.empty_alignments += 1
                    teacher_ids, teacher_logprobs = encode_shifted_group_metadata(
                        len(turn["prompt_ids"]), len(response_ids), groups
                    )
                    results[position] = {
                        "teacher_ids": teacher_ids,
                        "teacher_logprobs": teacher_logprobs,
                    }
            result = {"turns": results}
            session["score_cache"] = {
                "turn_count": turn_count,
                "result": copy.deepcopy(result),
            }
            return result

    def close_multiturn(self, session_id: str) -> dict:
        session_id = self._validate_session_id(session_id)
        with self._sessions_lock:
            now = time.monotonic()
            self._reap_stale_sessions_locked(now)
            self._sessions.pop(session_id, None)
            self._session_tombstones[session_id] = now + self.session_lease_s
        return {"ok": True}

    def record_no_signal_resample(self) -> dict:
        with self._stats_lock:
            self.no_signal_resamples += 1
        return {"ok": True}

    def record_no_signal_abandoned(self) -> dict:
        with self._stats_lock:
            self.no_signal_skipped_steps += 1
            if (
                self._teacher_failure is None
                and self._pending_teacher_transient is not None
                and not self._pending_teacher_success
            ):
                self._teacher_failure = self._pending_teacher_transient
            self._pending_teacher_transient = None
            self._pending_teacher_success = False
        return {"ok": True}

    def commit_teacher_cycle(self) -> dict:
        with self._stats_lock:
            self._pending_teacher_transient = None
            self._pending_teacher_success = False
        return {"ok": True}

    def notify_mutation(self) -> None:
        with self._mutation_lock:
            if self._mutation_notified:
                return
            with self._stats_lock:
                callback_failure = self._mutation_callback_failure
            if callback_failure is not None:
                self._raise_recorded_mutation_failure(callback_failure)
            try:
                self.mutation_callback()
            except Exception as error:
                classification = (
                    "transient" if isinstance(error, _w.RetriableInfraError) else "permanent"
                )
                callback_failure = self._record_mutation_callback_failure(
                    classification,
                    str(error),
                )
                self._raise_recorded_mutation_failure(callback_failure)
            with self._stats_lock:
                self._mutation_callback_succeeded = True
            self._mutation_notified = True

    def start(self) -> None:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def _send_json(self, status: int, payload: dict) -> None:
                encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_POST(self):
                recovered_teacher_failure = None
                request_succeeded = False
                try:
                    if self.headers.get("Authorization") != f"Bearer {bridge.token}":
                        raise PermissionError("flash OPD bridge authorization failed")
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                    if self.path == "/score":
                        recovered_failures: list[tuple[str, str]] = []
                        result = bridge.score(
                            payload["index"],
                            payload["prompt_length"],
                            payload["sequence_ids"],
                            payload.get("image_count", 0),
                            payload.get("forced"),
                            recovered_failure=recovered_failures,
                        )
                        if recovered_failures:
                            recovered_teacher_failure = recovered_failures[0]
                    elif self.path == "/multiturn/start":
                        result = bridge.start_multiturn(
                            index=payload["index"],
                            session_id=payload["session_id"],
                            prompt_ids=payload["prompt_ids"],
                            raw_prompt=payload["raw_prompt"],
                        )
                    elif self.path == "/multiturn/step":
                        result = bridge.step_multiturn(payload)
                    elif self.path == "/multiturn/score":
                        result = bridge.score_multiturn(payload["session_id"])
                    elif self.path == "/multiturn/close":
                        result = bridge.close_multiturn(payload["session_id"])
                    elif self.path == "/no-signal/resample":
                        result = bridge.record_no_signal_resample()
                    elif self.path == "/no-signal/abandoned":
                        result = bridge.record_no_signal_abandoned()
                    elif self.path == "/teacher-cycle/committed":
                        result = bridge.commit_teacher_cycle()
                    elif self.path == "/mutation":
                        bridge.notify_mutation()
                        result = {"ok": True}
                    else:
                        raise ValueError("flash OPD bridge path is unknown")
                    request_succeeded = True
                    self._send_json(200, result)
                except Exception as error:
                    teacher_delivery_failure = (
                        request_succeeded
                        and self.path in {"/score", "/multiturn/score"}
                        and isinstance(error, (OSError, http.client.HTTPException))
                    )
                    if teacher_delivery_failure:
                        classification = "transient"
                    elif isinstance(error, _RecordedMutationCallbackFailure):
                        classification = error.classification
                    else:
                        classification = (
                            "transient"
                            if isinstance(error, _w.RetriableInfraError)
                            or (isinstance(error, TeacherError) and not error.permanent)
                            else "permanent"
                        )
                    if teacher_delivery_failure:
                        if recovered_teacher_failure is not None:
                            bridge._promote_recovered_teacher_failure(recovered_teacher_failure)
                        else:
                            bridge._record_teacher_delivery_failure(error)
                    elif self.path == "/score":
                        if recovered_teacher_failure is not None:
                            bridge._promote_recovered_teacher_failure(recovered_teacher_failure)
                        else:
                            bridge._record_teacher_failure(classification, str(error))
                    elif self.path == "/multiturn/score":
                        bridge._record_teacher_failure(
                            classification,
                            str(error),
                            terminal=True,
                        )
                    self._send_json(
                        503 if classification == "transient" else 422,
                        {
                            "error": {
                                "classification": classification,
                                "message": str(error),
                            }
                        },
                    )

        self._text_teacher_batcher = _TextTeacherBatcher(
            self.teacher,
            max_batch_size=OPD_TEACHER_SCORING_CONCURRENCY,
            flush_wait_s=_TEXT_TEACHER_FLUSH_WAIT_S,
        )
        self._text_teacher_batcher.start()
        try:
            self._server = _TeacherBridgeHTTPServer(("127.0.0.1", 0), Handler)
        except Exception:
            self._text_teacher_batcher.close()
            raise
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        if self.multi_turn:
            self._session_reaper_stop.clear()
            self._session_reaper_thread = threading.Thread(
                target=self._session_reaper_loop,
                daemon=True,
            )
            self._session_reaper_thread.start()

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("flash OPD bridge has not started")
        return f"http://127.0.0.1:{self._server.server_port}"

    def close(self) -> None:
        self._session_reaper_stop.set()
        if self._session_reaper_thread is not None:
            self._session_reaper_thread.join(timeout=self.session_reap_interval_s + 1.0)
        with self._sessions_lock:
            self._sessions.clear()
            self._session_tombstones.clear()
        if self._server is not None:
            self._server.shutdown()
        if self._text_teacher_batcher is not None:
            self._text_teacher_batcher.close()
        if self._server is not None:
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


class _OpdProgressState:
    def __init__(self, resume_state: dict | None = None) -> None:
        state = resume_state or {}
        self._condition = threading.Condition()
        self.loss_curve = [float(value) for value in state.get("loss_curve", [])]
        self.coverage_curve = [float(value) for value in state.get("coverage_curve", [])]
        self.base_train_wall_seconds = float(state.get("train_wall_seconds", 0.0))
        self._prev_aligned = int(state.get("aligned_sequences", state.get("granularity_n", 0)))
        self._prev_cov_sum = float(state.get("coverage_sum", state.get("granularity_sum", 0.0)))
        self._train_started_at: float | None = None
        self._step_states: dict[int, dict] = {}
        if resume_state is not None:
            self._step_states[int(state["opt_steps"])] = dict(state)

    def start_training(self) -> None:
        self._train_started_at = time.time()

    def _train_wall_seconds(self) -> float:
        elapsed = 0.0
        if self._train_started_at is not None:
            elapsed = max(0.0, time.time() - self._train_started_at)
        return self.base_train_wall_seconds + elapsed

    def record_step(self, step: int, loss: float, bridge: _TeacherAlignmentBridge) -> None:
        with self._condition:
            expected_step = len(self.loss_curve) + 1
            if step != expected_step:
                raise RuntimeError(
                    f"verl OPD metric step {step} does not follow accumulated step {expected_step - 1}"
                )
            snapshot = bridge.accounting_snapshot()
            self.loss_curve.append(float(loss))
            aligned = int(snapshot["aligned_sequences"])
            cov_sum = float(snapshot["coverage_sum"])
            # per-step coverage: delta over the previous snapshot, so the curve shows each step's
            # own alignment quality instead of a cumulative average that flattens regressions.
            d_aligned = aligned - self._prev_aligned
            d_cov = cov_sum - self._prev_cov_sum
            self._prev_aligned, self._prev_cov_sum = aligned, cov_sum
            coverage = (
                (d_cov / d_aligned) if d_aligned > 0 else (cov_sum / aligned if aligned else 0.0)
            )
            self.coverage_curve.append(coverage)
            snapshot.update(
                {
                    "train_wall_seconds": self._train_wall_seconds(),
                    "loss_curve": list(self.loss_curve),
                    "coverage_curve": list(self.coverage_curve),
                }
            )
            self._step_states[step] = snapshot
            self._condition.notify_all()

    def checkpoint_state(self, step: int, *, timeout_s: float = 300.0) -> dict:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while step not in self._step_states:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"timed out waiting for honest OPD accounting through checkpoint step {step}"
                    )
                self._condition.wait(remaining)
            return dict(self._step_states[step])

    def final_state(self, bridge: _TeacherAlignmentBridge) -> dict:
        snapshot = bridge.accounting_snapshot()
        snapshot.update(
            {
                "train_wall_seconds": self._train_wall_seconds(),
                "loss_curve": list(self.loss_curve),
                "coverage_curve": list(self.coverage_curve),
            }
        )
        return snapshot


_REQUIRED_OVERRIDE_KEYS = (
    "train_files",
    "val_files",
    "train_batch_size",
    "max_prompt_length",
    "max_response_length",
    "max_sequence_length",
    "model_path",
    "lora_rank",
    "lora_alpha",
    "target_modules",
    "learning_rate",
    "local_dir",
    "save_freq",
    "n_gpus_per_node",
    "ulysses_sequence_parallel_size",
    "seed",
    "project_name",
    "experiment_name",
    "total_training_steps",
    "group_size",
    "bridge_url",
    "bridge_token",
    "kl_penalty_coef",
    "reward_path",
)


def build_opd_overrides(config: dict) -> list[str]:
    """Render the exact verl 0.8.0 synchronous PPO and distillation config surface."""
    missing = [key for key in _REQUIRED_OVERRIDE_KEYS if key not in config]
    if missing:
        raise KeyError(f"build_opd_overrides missing required config keys: {missing}")
    # the full sequence length the engine is sized for. the caller derives max_prompt_length by
    # carving max_response_length out of this same value, so the token budget, the prompt filter,
    # and the engine always agree.
    max_tokens = int(config["max_sequence_length"])
    overrides = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.norm_adv_by_std_in_grpo=false",
        f"data.train_files={_hydra_val(config['train_files'])}",
        f"data.val_files={_hydra_val(config['val_files'])}",
        f"data.train_batch_size={_hydra_val(config['train_batch_size'])}",
        f"data.max_prompt_length={_hydra_val(config['max_prompt_length'])}",
        f"data.max_response_length={_hydra_val(config['max_response_length'])}",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.shuffle=false",
        f"data.seed={_hydra_val(config.get('seed', 42))}",
        # rollout engine seed. NOT `rollout.seed`: verl 0.8.0's RolloutConfig declares no such field,
        # so a bare key fails hydra composition and a `+`/`++` prefix composes but then dies in
        # omega_conf_to_dataclass with an unexpected-kwarg TypeError. engine_kwargs is a declared
        # dict on both 0.8.0 and 0.9.x and is spread into the vllm engine args *after* verl's own
        # "seed" entry, so it wins. `++` because the sub-key is absent from the composed node.
        # per-request sampling is seeded separately by the plugin's deterministic_rollout_seed.
        f"++actor_rollout_ref.rollout.engine_kwargs.vllm.seed={_hydra_val(config.get('seed', 42))}",
        # fp8 kv cache, exactly as the deleted trl colocate engine reserved it (it set
        # kv_cache_dtype="fp8" on cc >= 8.9) and as rl_train.py does for grpo. this is not an
        # optimization: flash/engine/vram.py sizes an opd run against an fp8 kv pool once the
        # requirement clears the largest non-fp8 card, so a bf16 cache here would allocate twice the
        # kv the allocator reserved and OOM at rollout init on a card sizing called sufficient.
        # the caller resolves the flag (cc probe + gdn exclusion); absent/false means bf16.
        *(
            ["+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8"]
            if config.get("fp8_kv")
            else []
        ),
        "data.dataloader_num_workers=0",
        "data.image_key=images",
        "data.return_raw_chat=true",
        "data.return_multi_modal_inputs=false",
        # `++`, not a bare key: data.apply_chat_template_kwargs exists but holds an empty struct
        # dict, so assigning a whole dict to it is rejected as an unknown sub-key.
        "++data.apply_chat_template_kwargs={enable_thinking:"
        + _hydra_val(config.get("thinking", False))
        + "}",
        f"actor_rollout_ref.model.path={_hydra_val(config['model_path'])}",
        "actor_rollout_ref.model.trust_remote_code=true",
        "actor_rollout_ref.model.use_remove_padding=true",
        # 32k contexts: the fused linear-CE forward computes per-token log_probs from hidden states
        # + lm_head in chunks (FusedLinearForPPO), never materializing the [tokens, vocab] logits
        # tensor (~130 GB at 32k on a 248k vocab). the distillation loss consumes exactly
        # model_output["log_probs"], so the fused path is loss-equivalent. torch backend = exact.
        "actor_rollout_ref.model.use_fused_kernels=true",
        "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        f"actor_rollout_ref.model.lora_rank={_hydra_val(config['lora_rank'])}",
        f"actor_rollout_ref.model.lora_alpha={_hydra_val(config['lora_alpha'])}",
        f"actor_rollout_ref.model.target_modules={_hydra_val(config['target_modules'])}",
        *(
            [
                "++actor_rollout_ref.model.target_parameters="
                + _hydra_val(config["target_parameters"])
            ]
            if config.get("target_parameters")
            else []
        ),
        f"actor_rollout_ref.model.lora_adapter_path={_hydra_val(config.get('lora_adapter_path'))}",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={_hydra_val(config['train_batch_size'])}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={_hydra_val(max_tokens)}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.shuffle=false",
        "actor_rollout_ref.actor.use_torch_compile=true",
        f"actor_rollout_ref.actor.optim.lr={_hydra_val(config['learning_rate'])}",
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        f"actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size={_hydra_val(config['ulysses_sequence_parallel_size'])}",
        # store the frozen base in bf16, not verl's fp32 yaml default. shared with the rl driver.
        *trainer_dtype_overrides(),
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        # safetensors load format is required for lora rollout on vllm, exactly as on the grpo path.
        # the default is `dummy`, which makes verl set base_sync_done=False and push base weights to
        # vllm itself. that path routes every name through replace_lora_wrapper, which appends
        # `.base_layer` to anything matching the all-linear target set, including the vision tower.
        # plain vllm has no `visual.blocks.N.attn.qkv.base_layer.weight` slot, so the first weight
        # sync dies with a KeyError before a rollout is ever produced. loading the base from
        # safetensors keeps vllm authoritative for base weights and transfers only lora deltas.
        "actor_rollout_ref.rollout.load_format=safetensors",
        # rollout.enforce_eager is a real verl field, so this is a plain override, not a '+' append.
        # the caller resolves it from the device capability; absent/false keeps verl's default.
        *(["actor_rollout_ref.rollout.enforce_eager=True"] if config.get("enforce_eager") else []),
        # blackwell attention pins, resolved by the caller and absent off blackwell. both are real
        # AsyncEngineArgs fields in the pinned vllm 0.19.1 and verl spreads engine_kwargs.vllm
        # straight into them, so '+' appends under the existing struct exactly as on the grpo path.
        *(
            [
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend="
                f"{config['attention_backend']}"
            ]
            if config.get("attention_backend")
            else []
        ),
        *(
            [
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend="
                f"{config['mm_encoder_attn_backend']}"
            ]
            if config.get("mm_encoder_attn_backend")
            else []
        ),
        # keep the rollout engine RESIDENT for models whose vLLM wake/reload HANGS (catalog
        # sleep_unsupported), exactly as the grpo path does. opd is NOT exempt: main_ppo_sync calls
        # checkpoint_manager.sleep_replicas() during init_workers and again around validation, which
        # lands in the same vllm_async_server sleep(). the flagged model declares algos including
        # opd and the parse-time gate admits it, so without this an opd run on it wedges.
        *rollout_resident_overrides(bool(config.get("sleep_unsupported"))),
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={_hydra_val(config['n_gpus_per_node'])}",
        f"actor_rollout_ref.rollout.n={_hydra_val(config['group_size'])}",
        # `++`, not a bare key: limit_images is a real RolloutConfig field but is absent from the
        # composed rollout node, so hydra rejects a bare assignment.
        "++actor_rollout_ref.rollout.limit_images=8",
        f"actor_rollout_ref.rollout.max_model_len={_hydra_val(max_tokens)}",
        f"actor_rollout_ref.rollout.temperature={_hydra_val(config.get('temperature', 1.0))}",
        f"actor_rollout_ref.rollout.top_p={_hydra_val(config.get('top_p', 1.0))}",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.calculate_log_probs=false",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={_hydra_val(max_tokens)}",
        "actor_rollout_ref.rollout.agent.default_agent_loop="
        + ("flash_multi_turn" if config.get("multi_turn") else "flash_single_turn"),
        # opd runs async rollout through verl's AgentLoopManager, which chunks the rollout batch
        # across agent.num_workers and asserts an exact split. the batch is
        # train_batch_size * group_size, so verl's default of 8 aborts the run before the first step
        # whenever that product is not a multiple of 8. size the pool to the batch instead.
        "actor_rollout_ref.rollout.agent.num_workers="
        f"{agent_loop_workers(int(config['train_batch_size']) * int(config['group_size']))}",
        # verl force-enables TransferQueue on the opd entry point (main_ppo_sync.main sets
        # transfer_queue.enable = True with no opt-out), and its SimpleStorage default asks for 8
        # storage units sized for a multi-node cluster. tq.init reserves them through a SPREAD
        # placement group and blocks in ray.get(pg.ready()) until every 1-cpu bundle is placed,
        # with no timeout: any ray cluster with fewer free cpus than units hangs the run forever
        # before a single gpu is touched. one unit is correct for flash's single-node trainer and
        # keeps the reservation satisfiable regardless of how ray sized the cluster.
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=1",
        # ray autodetects the HOST's cpu count inside a rented pod and eagerly forks one idle worker
        # per core. on a 1x4090 pod that is 48 forks nothing asked for, which oom-killed the actor
        # that mattered (VERL-123). size the pool to the container instead. this also keeps the
        # storage-unit reservation above satisfiable, since it stays well under the cpu floor.
        f"ray_kwargs.ray_init.num_cpus={ray_num_cpus(config['n_gpus_per_node'])}",
        # num_gpus is absent from verl's generated ray_init node, so hydra requires add-key syntax.
        f"+ray_kwargs.ray_init.num_gpus={config['n_gpus_per_node']}",
        "critic.enable=false",
        "reward.reward_model.enable=false",
        # disabling the reward MODEL does not disable reward SCORING. with no custom function the
        # loop still calls the default rule-based scorer, which dispatches on data_source and raises
        # NotImplementedError for "flash_opd" (VERL-153). point it at the generated zero function.
        # both key forms are required: the legacy top-level key is migrated in the main process but
        # is not visible to the RewardLoopWorker actor, which reads reward.custom_reward_function.
        f"custom_reward_function.path={_hydra_val(config['reward_path'])}",
        "custom_reward_function.name=compute_score",
        f"reward.custom_reward_function.path={_hydra_val(config['reward_path'])}",
        "reward.custom_reward_function.name=compute_score",
        "distillation._target_=flash_opd_plugin.FlashRemoteDistillationConfig",
        "distillation.enabled=true",
        "distillation.n_gpus_per_node=0",
        "distillation.nnodes=0",
        "distillation.teacher_key=index",
        f"+distillation.bridge_url={_hydra_val(config['bridge_url'])}",
        f"+distillation.bridge_token={_hydra_val(config['bridge_token'])}",
        f"+distillation.kl_penalty_coef={_hydra_val(config['kl_penalty_coef'])}",
        "distillation.distillation_loss.loss_mode=flash_groupwise_reverse_kl",
        "distillation.distillation_loss.topk=null",
        "distillation.distillation_loss.use_task_rewards=false",
        "distillation.distillation_loss.use_policy_gradient=false",
        "distillation.distillation_loss.loss_max_clamp=null",
        "distillation.distillation_loss.log_prob_min_clamp=null",
        f"trainer.default_local_dir={_hydra_val(config['local_dir'])}",
        f"trainer.save_freq={_hydra_val(config['save_freq'])}",
        f"trainer.n_gpus_per_node={_hydra_val(config['n_gpus_per_node'])}",
        "trainer.nnodes=1",
        f"trainer.project_name={_hydra_val(config['project_name'])}",
        f"trainer.experiment_name={_hydra_val(config['experiment_name'])}",
        f"trainer.logger={_hydra_val(config.get('loggers', ['console']))}",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={_hydra_val(config['total_training_steps'])}",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",
        "trainer.resume_mode=auto",
        "trainer.max_actor_ckpt_to_keep=null",
    ]
    if config.get("multi_turn"):
        overrides.append(f"actor_rollout_ref.rollout.prompt_length={_hydra_val(max_tokens)}")
    structured_outputs = config.get("structured_outputs")
    if structured_outputs:
        structured_outputs_config = {
            "backend": "xgrammar",
            "disable_any_whitespace": bool(structured_outputs.get("disable_any_whitespace", False)),
        }
        reasoning_parser = reasoning_parser_for(
            thinking=bool(config.get("thinking", False)),
            structured_outputs=structured_outputs,
        )
        if reasoning_parser is not None:
            structured_outputs_config["reasoning_parser"] = reasoning_parser
        overrides.append(
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.structured_outputs_config="
            + _hydra_val(structured_outputs_config)
        )
    return overrides


def _render_opd_sitecustomize(*, save_at_steps: tuple[int, ...], total_steps: int) -> str:
    required_steps = tuple(int(step) for step in save_at_steps)
    return f"""# generated flash opd runtime patches for verl 0.8
from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler as _FlashCheckpointHandler

_flash_required_save_steps = frozenset({required_steps!r})
_flash_total_steps = {int(total_steps)}
_flash_original_save_checkpoint = _FlashCheckpointHandler.save_checkpoint


def _flash_save_exact_checkpoint(self, step):
    if _flash_required_save_steps and step not in _flash_required_save_steps and step != _flash_total_steps:
        return None
    return _flash_original_save_checkpoint(self, step)


_FlashCheckpointHandler.save_checkpoint = _flash_save_exact_checkpoint
"""


def _build_opd_child_env(
    *,
    shim_dir: str,
    wandb_enabled: bool,
    bridge_url: str,
    bridge_token: str,
    seed: int,
    stop_sequences: tuple,
    eos_token_ids: frozenset[int],
    structured_outputs: dict | None,
    model_vocab_size: int,
    thinking: bool,
    multi_turn: bool = False,
    max_turns: int = 0,
    max_model_len: int = 32768,
    mutation_failure_path: str = "",
    score_delivery_failure_path: str = "",
    abandonment_failure_path: str = "",
    resample_failure_path: str = "",
    cycle_commit_failure_path: str = "",
) -> dict[str, str]:
    child = _build_verl_child_env(shim_dir=shim_dir, wandb_enabled=wandb_enabled)
    child.update(
        {
            "VERL_USE_EXTERNAL_MODULES": "flash_opd_plugin",
            "FLASH_OPD_BRIDGE_URL": bridge_url,
            "FLASH_OPD_BRIDGE_TOKEN": bridge_token,
            "FLASH_OPD_SEED": str(int(seed)),
            "FLASH_OPD_STOP_SEQUENCES": json.dumps(list(stop_sequences)),
            "FLASH_OPD_EOS_TOKEN_IDS": json.dumps(sorted(eos_token_ids)),
        }
    )
    if mutation_failure_path:
        child["FLASH_OPD_MUTATION_FAILURE_PATH"] = mutation_failure_path
    if score_delivery_failure_path:
        child["FLASH_OPD_SCORE_DELIVERY_FAILURE_PATH"] = score_delivery_failure_path
    if abandonment_failure_path:
        child["FLASH_OPD_ABANDONMENT_FAILURE_PATH"] = abandonment_failure_path
    if resample_failure_path:
        child["FLASH_OPD_RESAMPLE_FAILURE_PATH"] = resample_failure_path
    if cycle_commit_failure_path:
        child["FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH"] = cycle_commit_failure_path
    if multi_turn:
        child.update(
            {
                "FLASH_OPD_THINKING": "1" if thinking else "0",
                "FLASH_OPD_MAX_TURNS": str(int(max_turns)),
                "FLASH_OPD_MAX_MODEL_LEN": str(int(max_model_len)),
                "FLASH_OPD_ENV_CAPABILITIES": json.dumps(
                    [
                        "new_rollout_state",
                        "record_model_turn",
                        "env_reply",
                        "rollout_done",
                    ]
                ),
            }
        )
    if structured_outputs:
        child.update(
            {
                "FLASH_OPD_STRUCTURED_OUTPUTS": json.dumps(
                    structured_outputs, sort_keys=True, separators=(",", ":")
                ),
                "FLASH_OPD_MODEL_VOCAB_SIZE": str(int(model_vocab_size)),
                "FLASH_OPD_THINKING": "1" if thinking else "0",
            }
        )
    return child


def _opd_multimodal_parquet_features():
    from datasets import Features, Value

    return Features(
        {
            "prompt": [{"role": Value("string"), "content": Value("string")}],
            "images": [{"image": Value("string")}],
            "data_source": Value("string"),
            "reward_model": {
                "style": Value("string"),
                "ground_truth": Value("string"),
            },
            "extra_info": {"index": Value("int64")},
        }
    )


# arrow expands every shared python reference into its own copy, and the opd row list is one
# reference per prompt repeated across the whole horizon, so converting the rows in a single table
# costs peak host ram proportional to horizon * prompts_per_step * prompt_bytes. that is the parent
# worker's ram, not the gpu's, and the allocator sizes vram only -- a host-ram kill surfaces as a
# generic child exit with oom:false. writing in fixed batches holds the peak flat instead
# (measured: 24k rows of ~8k-token prompts, 6143 MB in one table vs 189 MB batched).
_OPD_PARQUET_WRITE_BATCH_ROWS = 2000


def _write_opd_parquet(rows: list[dict], path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        raise ValueError("refusing to write an empty OPD parquet")
    features = _opd_multimodal_parquet_features() if any("images" in row for row in rows) else None
    # pin one schema for every batch. multimodal takes it from the declared features exactly as the
    # single-table write did; text infers it from the first batch so a later batch cannot silently
    # infer a different type and be rejected mid-file.
    schema = (
        features.arrow_schema
        if features is not None
        else pa.Table.from_pylist(rows[:_OPD_PARQUET_WRITE_BATCH_ROWS]).schema
    )
    # write to a sibling temp file and rename only once every batch landed. closing a partially
    # written ParquetWriter still emits a valid footer, so failing in place would leave a READABLE
    # short file at `path` -- a truncated horizon that trains silently instead of raising.
    partial = f"{path}.partial"
    try:
        writer = pq.ParquetWriter(partial, schema)
        try:
            for start in range(0, len(rows), _OPD_PARQUET_WRITE_BATCH_ROWS):
                writer.write_table(
                    pa.Table.from_pylist(
                        rows[start : start + _OPD_PARQUET_WRITE_BATCH_ROWS], schema=schema
                    )
                )
        finally:
            writer.close()
        os.replace(partial, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(partial)
        raise


# the verl bridge stores the empty-alignment skip under its own internal key,
# which the resume state also reads. trl publishes the same condition as
# alignment_empty, so translate it at the metadata boundary only.
_CANONICAL_SKIP_REASONS = {"empty_alignment": "alignment_empty"}


def _canonical_skip_reasons(skip_counts: dict) -> dict:
    canonical: dict[str, int] = {}
    for reason, count in skip_counts.items():
        count = int(count)
        # trl records skip reasons in a counter, so only reasons that actually
        # occurred appear. the verl snapshot always injects empty_alignment.
        if count <= 0:
            continue
        name = _CANONICAL_SKIP_REASONS.get(reason, reason)
        canonical[name] = canonical.get(name, 0) + count
    return dict(sorted(canonical.items()))


def _failure_accounting_metadata(accounting: dict) -> dict:
    return {
        "teacher_transient_failures": int(accounting["teacher_transient"]),
        "teacher_errors": int(accounting["teacher_error"]),
        "no_signal_resamples": int(accounting["no_signal_resamples"]),
        "no_signal_skipped_steps": int(accounting["no_signal_skipped_steps"]),
        "skip_reasons": _canonical_skip_reasons(accounting["skip_counts"]),
    }


def _read_failure_fallback_records(base_path: str) -> list[tuple[str, str]]:
    if not base_path:
        return []
    base = Path(base_path)
    failures: list[tuple[str, str]] = []
    for path in sorted(base.parent.glob(f"{base.name}.*.json")):
        try:
            with path.open(encoding="utf-8") as file:
                encoded = file.read(8193)
            if len(encoded) > 8192:
                continue
            record = json.loads(encoded)
        except (OSError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        classification = record.get("classification")
        message = record.get("message")
        if classification not in {"permanent", "transient"}:
            continue
        if not isinstance(message, str) or not message.strip():
            continue
        failures.append((classification, message.strip()))
    return failures


def _read_classified_failure_fallback(base_path: str) -> tuple[str, str] | None:
    failures = _read_failure_fallback_records(base_path)
    for classification in ("permanent", "transient"):
        for failure_classification, message in failures:
            if failure_classification == classification:
                return classification, message
    return None


def _reconcile_score_delivery_failure(
    bridge: _TeacherAlignmentBridge,
    failure: tuple[str, str] | None,
) -> tuple[str, str] | None:
    if failure is None or bridge.teacher_failure is not None:
        return None
    if bridge._promote_pending_teacher_failure():
        return None
    return failure


def _reconcile_no_signal_notification_failure(
    bridge: _TeacherAlignmentBridge,
    failures: tuple[tuple[str, str] | None, ...],
) -> tuple[str, str] | None:
    if bridge.teacher_failure is not None:
        return None
    selected = None
    for failure in failures:
        if failure is None:
            continue
        if failure[0] == "permanent" or selected is None:
            selected = failure
    if (
        selected is not None
        and selected[0] == "transient"
        and bridge._promote_pending_teacher_failure()
    ):
        return None
    return selected


def _raise_verl_failure(
    return_code: int,
    teacher_failure: tuple[str, str] | None,
    mutation_failure: tuple[str, str] | None = None,
    cycle_commit_failure: tuple[str, str] | None = None,
    no_signal_failure: tuple[str, str] | None = None,
    score_delivery_failure: tuple[str, str] | None = None,
) -> None:
    if return_code == 0:
        return
    if mutation_failure is not None:
        classification, message = mutation_failure
        if classification == "transient":
            raise _w.RetriableInfraError(f"optimizer marker failure: {message}")
        raise RuntimeError(f"permanent optimizer marker failure: {message}")
    if cycle_commit_failure is not None:
        classification, message = cycle_commit_failure
        if classification == "transient":
            raise _w.RetriableInfraError(f"pre-update cycle commitment failure: {message}")
        raise RuntimeError(f"permanent pre-update cycle commitment failure: {message}")
    if no_signal_failure is not None:
        classification, message = no_signal_failure
        if classification == "transient":
            raise _w.RetriableInfraError(f"transient no-signal notification failure: {message}")
        raise RuntimeError(f"permanent no-signal notification failure: {message}")
    if score_delivery_failure is not None:
        classification, message = score_delivery_failure
        if classification == "transient":
            raise _w.RetriableInfraError(f"transient teacher score delivery failure: {message}")
        raise RuntimeError(f"permanent teacher score delivery failure: {message}")
    if teacher_failure is not None:
        classification, message = teacher_failure
        if classification == "transient":
            raise _w.RetriableInfraError(
                f"transient teacher failure after bounded retries: {message}"
            )
        raise RuntimeError(f"permanent teacher failure: {message}")
    if return_code == _TRANSIENT_TEACHER_EXIT:
        raise _w.RetriableInfraError("transient teacher bridge failure")
    if return_code == _PERMANENT_TEACHER_EXIT:
        raise RuntimeError("permanent teacher bridge failure")
    raise RuntimeError(f"verl OPD subprocess exited with status {return_code}")


def _find_checkpoint_file(checkpoint_dir: str, needles: tuple[str, ...]) -> str | None:
    for root, _dirs, files in os.walk(checkpoint_dir):
        for name in sorted(files):
            lowered = name.lower()
            if any(needle in lowered for needle in needles):
                return os.path.join(root, name)
    return None


def _stage_retry_contract(
    checkpoint_dir: str,
    *,
    step: int,
    seed: int,
    prompt_pool_fingerprint: str,
    prompts_per_step: int,
    group_size: int,
    adapter_dir: str,
    accounting_state: dict,
) -> None:
    for name in os.listdir(adapter_dir):
        if name == "adapter_config.json" or name.startswith("adapter_model"):
            shutil.copy2(os.path.join(adapter_dir, name), os.path.join(checkpoint_dir, name))
    optimizer_source = _find_checkpoint_file(checkpoint_dir, ("optim", "optimizer"))
    if optimizer_source is None:
        raise RuntimeError("verl OPD checkpoint has no optimizer state")
    shutil.copy2(optimizer_source, os.path.join(checkpoint_dir, "optimizer.pt"))
    rng_source = os.path.join(checkpoint_dir, "data.pt")
    if not os.path.isfile(rng_source):
        rng_source = _find_checkpoint_file(checkpoint_dir, ("extra", "rng"))
    if rng_source is None:
        raise RuntimeError("verl OPD checkpoint has no resumable dataloader or rng state")
    shutil.copy2(rng_source, os.path.join(checkpoint_dir, "rng_state.pth"))
    state = {
        **accounting_state,
        "contract_version": OPD_RESUME_STATE_VERSION,
        "seed": seed,
        "opt_steps": step,
        "step": step,
        "rollout_seed_ordinal": step * prompts_per_step * group_size,
        "prompt_pool_fingerprint": prompt_pool_fingerprint,
        "verl_checkpoint": True,
    }
    validate_opd_resume_state_metadata(state, expected_seed=seed, checkpoint_step=step)
    with open(os.path.join(checkpoint_dir, "opd_state.json"), "w", encoding="utf-8") as file:
        json.dump(state, file, sort_keys=True)


class _OpdVerlCheckpointWatcher(_VerlCheckpointWatcher):
    def __init__(
        self,
        *,
        seed: int,
        prompt_pool_fingerprint: str,
        prompts_per_step: int,
        group_size: int,
        accounting_state,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seed = seed
        self.prompt_pool_fingerprint = prompt_pool_fingerprint
        self.prompts_per_step = prompts_per_step
        self.group_size = group_size
        self.accounting_state = accounting_state

    def _should_publish(self, step: int) -> bool:
        return True

    def _publish(self, step: int, checkpoint_dir: str) -> None:
        actor_dir = os.path.join(checkpoint_dir, "actor")
        adapter_dir = os.path.join(self.export_root, f"step-{step}")
        _export_checkpoint_adapter(
            actor_dir,
            adapter_dir,
            model_id=self.model_id,
            model_revision=self.model_revision,
            python_bin=self.python_bin,
        )
        _stage_retry_contract(
            checkpoint_dir,
            step=step,
            seed=self.seed,
            prompt_pool_fingerprint=self.prompt_pool_fingerprint,
            prompts_per_step=self.prompts_per_step,
            group_size=self.group_size,
            adapter_dir=adapter_dir,
            accounting_state=self.accounting_state(step),
        )

        def publish_required_adapter() -> None:
            if step in self.required_steps:
                _w.publish_deployable_checkpoint(
                    adapter_dir,
                    step,
                    required=True,
                    _provenance_ready=True,
                )

        uploaded = _w.upload_resume_checkpoint(
            step, checkpoint_dir, before_upload=publish_required_adapter
        )
        if step in self.required_steps and not uploaded:
            raise RuntimeError(f"required save step {step} full-state checkpoint was not published")
        self.processed_steps.add(step)


def _restore_verl_resume(
    local_dir: str,
    *,
    prompt_pool_fingerprint: str,
    update_horizon: int,
) -> tuple[int, dict | None]:
    revision = _w.OPD_RESUME_REVISION or None
    resume = _w.hf_resume_checkpoint(fail_closed=bool(revision), revision=revision)
    if not resume:
        return 0, None
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(resume))
    if match is None:
        raise RuntimeError(f"invalid OPD resume checkpoint path {resume!r}")
    step = int(match.group(1))
    with open(os.path.join(resume, "opd_state.json"), encoding="utf-8") as file:
        state = validate_opd_resume_state_metadata(
            json.load(file), expected_seed=int(_w.SEED), checkpoint_step=step
        )
    if state["prompt_pool_fingerprint"] != prompt_pool_fingerprint:
        raise RuntimeError("OPD resume prompt pool does not match the current run")
    if step > update_horizon:
        raise RuntimeError("OPD resume checkpoint is beyond the requested update horizon")
    target = os.path.join(local_dir, f"global_step_{step}")
    shutil.copytree(resume, target, dirs_exist_ok=True)
    with open(os.path.join(local_dir, "latest_checkpointed_iteration.txt"), "w") as file:
        file.write(str(step))
    return step, state


def _processed_resume_steps(required_steps: tuple[int, ...], resume_step: int) -> set[int]:
    processed = _durable_required_save_steps(required_steps, resume_step)
    if resume_step and resume_step not in required_steps:
        processed.add(resume_step)
    return processed


def run_opd_train(spec=None) -> None:
    """Run flash OPD through verl's native rollout and weight-sync path."""
    from flash.engine.worker.teacher import TeacherClient
    from flash.multimodal import (
        image_teacher_prompt_messages,
        normalize_prompt_images,
        record_has_images,
        validate_multimodal_training,
    )

    spec = spec or _w.JOB_SPEC
    env = _w.require_active_env()
    if getattr(env, "is_tool_env", False):
        raise RuntimeError("native tool-calling OPD environments are not supported")
    multi_turn = bool(getattr(env, "multi_turn", False))
    if multi_turn:
        required_methods = (
            "new_rollout_state",
            "record_model_turn",
            "env_reply",
            "rollout_done",
        )
        missing = [name for name in required_methods if not callable(getattr(env, name, None))]
        if missing:
            raise RuntimeError(
                f"multi-turn OPD environment is missing required rollout methods: {missing}"
            )
        max_turns = int(getattr(env, "max_turns", 0) or 0)
        if max_turns <= 0:
            raise RuntimeError("multi-turn OPD environment requires a positive bounded turn limit")
    else:
        max_turns = 0
    knobs = _resolve_opd_knobs()
    if multi_turn and knobs.structured_outputs:
        raise RuntimeError(
            "multi-turn structured-output OPD is not supported until a per-turn constraint contract exists"
        )
    model_id = spec.model if spec else RECIPE.hf_model_id
    model_revision = getattr(spec, "model_revision", "") if spec else ""
    from flash.opd_validation import validate_opd_structured_outputs
    from flash.spec import gpu_count_of

    structured_validation = validate_opd_structured_outputs(
        knobs.structured_outputs,
        model_id=model_id,
        model_revision=model_revision,
        model_policy=getattr(spec, "model_policy", "catalog") if spec else "catalog",
        gpu=spec.gpu.type if spec else None,
        gpu_count=gpu_count_of(spec) if spec else 1,
    )
    structured_outputs = structured_validation.constraint
    model_vocab_size = structured_validation.model_vocab_size
    # the child trainer is seeded through its own config, but the environment's dataset /
    # prompt_messages calls run HERE in the parent. an unseeded parent can build a different prompt
    # pool across attempts, whose fingerprint then rejects a valid resume checkpoint. seed just
    # before the first env call that can consume randomness, so the cheap fail-closed guards above
    # still raise without paying for the torch import.
    seed_training_rngs(_w.SEED)
    train = list(env.dataset())
    if not train:
        raise RuntimeError("opd environment dataset is empty")
    max_examples = int(getattr(spec.train, "max_examples", 0) or 0) if spec else 0
    if max_examples > 0:
        train = train[:max_examples]
    _scanned = [0]
    with liveness_heartbeat("opd_prompt_scan", progress=lambda: _scanned[0]):
        # rendering prompts for a large dataset can outlast the heartbeat window; keep the worker
        # alive while scanning (the scan is O(dataset) tokenizer/template work).
        prompt_rows = []
        for example in train:
            prompt_rows.append((example, env.prompt_messages(example)))
            _scanned[0] += 1
    multimodal = any(record_has_images(example, messages) for example, messages in prompt_rows)
    if multimodal:
        validate_multimodal_training(model_id, "opd")
    random.Random(_w.SEED).shuffle(train)

    started_at = time.time()
    # validate the control-panel broker transport before the gpu probe and model prefetch so a malformed
    # attempt fails
    # before any additional paid setup. raw managed-teacher provider credentials never enter the worker.
    from flash.spec import CONTROL_PANEL_URL_ENV, TEACHER_CAPABILITY_ENV

    control_panel_url = os.environ.get(CONTROL_PANEL_URL_ENV, "").strip()
    capability = os.environ.get(TEACHER_CAPABILITY_ENV, "").strip()
    if not control_panel_url or not capability:
        raise RuntimeError(
            "managed teacher control-panel transport is missing from the OPD parent worker"
        )
    _w.heartbeat("opd_start", gpu=_w.gpu_diagnostics(include_torch=False))
    _probe_gpu_in_subprocess(
        spec.gpu.type if spec else None,
        exact_type=spec.gpu.type if spec else "",
    )
    teacher = TeacherClient(capability, control_panel_url, knobs.teacher_model)
    processor = None
    if multimodal:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            **_w.model_revision_kwargs(model_revision),
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = _w.load_tokenizer(model_id, revision=model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    thinking_prefill = _thinking_prefill_text(tokenizer)
    requested_len = knobs.max_length or (RECIPE.opd.max_prompt_len + knobs.max_completion)
    # clamp to the architecture BEFORE deriving the prompt budget, so every downstream length agrees.
    # clamping only the engine would admit prompts sized against the unclamped budget and then fail
    # them at rollout instead of training on the shorter context.
    max_model_len = clamp_engine_len(
        requested_len, model_max_position_embeddings(model_id, model_revision)
    )
    if max_model_len < requested_len:
        print(
            f"[opd-verl] max_context_tokens {requested_len} exceeds the {model_id} context limit; "
            f"training at {max_model_len}",
            flush=True,
        )
    prompt_budget = max_model_len - knobs.max_completion
    if prompt_budget < 1:
        raise RuntimeError("opd max_context_tokens leaves no room for a prompt")
    if multi_turn:
        validate_glue_template(tokenizer, thinking=bool(_w.THINKING))

    prompts: list[_BridgePrompt] = []
    dropped_long = 0
    package_root_value = getattr(env, "package_root", None)
    package_root = str(Path(package_root_value).resolve()) if package_root_value else None
    _prepped = [0]
    with liveness_heartbeat("opd_image_prep", progress=lambda: _prepped[0]):
        for example in train:
            _prepped[0] += 1
            messages = env.prompt_messages(example)
            if multi_turn:
                messages = validate_transcript_messages(
                    messages, source="environment initial prompt"
                )
            if record_has_images(example, messages):
                assert processor is not None
                normalized = normalize_prompt_images(example, messages, package_root)
                student_messages = normalized.messages
                image_descriptors = tuple(normalized.descriptors)
                teacher_messages = image_teacher_prompt_messages(
                    student_messages, len(image_descriptors)
                )
                prompt_ids = _processor_expanded_prompt_ids(
                    processor,
                    student_messages,
                    image_descriptors,
                    package_root,
                    enable_thinking=bool(_w.THINKING),
                )
            else:
                student_messages = messages
                teacher_messages = messages
                image_descriptors = ()
                if processor is not None:
                    # mixed job: the verl child tokenizes EVERY row through the multimodal dataset
                    # path (the processor), so text-only rows must freeze via the same path or the
                    # bridge's exact prompt-id check trips on tokenizer-vs-processor differences.
                    prompt_ids = _processor_expanded_prompt_ids(
                        processor,
                        student_messages,
                        (),
                        package_root,
                        enable_thinking=bool(_w.THINKING),
                    )
                else:
                    prompt_ids = _normalize_prompt_ids(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=_w.THINKING,
                        )
                    )
            if len(prompt_ids) > prompt_budget:
                dropped_long += 1
                continue
            prompts.append(
                _BridgePrompt(
                    student_messages=student_messages,
                    teacher_messages=teacher_messages,
                    prompt_ids=prompt_ids,
                    image_descriptors=image_descriptors,
                    package_root=package_root,
                    example=example if multi_turn else None,
                )
            )
    if not prompts:
        raise RuntimeError("every OPD prompt exceeds the configured prompt budget")
    # weights come AFTER the budget filter: a dataset whose every prompt is over budget is a
    # deterministic input error, and downloading tens of GB before raising it burns paid worker
    # minutes for a verdict the tokenizer already had. the tokenizer/processor/config loads above
    # fetch kilobytes, not weights, so they are cheap to run first.
    download_seconds = _w.prefetch_model(model_id, revision=model_revision)
    # reads the snapshot with local_files_only, so it has to follow the prefetch.
    eos_token_ids = generation_eos_from_cached_config(model_id, model_revision, tokenizer)
    prompts_per_step = min(knobs.prompts_per_step, len(prompts))
    derived_steps = on_policy_steps(
        epochs=knobs.epochs,
        prompt_count=len(prompts),
        prompts_per_step=prompts_per_step,
    )
    update_horizon = resolve_update_horizon(derived_steps, knobs.max_steps)
    validate_save_steps(knobs.save_at_steps, update_horizon)
    prompt_pool_fingerprint = _prompt_pool_fingerprint(prompts)

    workdir = os.path.join("/tmp", "flash-opd-verl", _w.RUN_ID, f"seed-{_w.SEED}")
    shutil.rmtree(workdir, ignore_errors=True)
    data_dir = os.path.join(workdir, "data")
    image_dir = os.path.join(workdir, "images")
    shim_dir = os.path.join(workdir, "shim")
    local_dir = os.path.join(workdir, "checkpoints")
    export_root = os.path.join(workdir, "checkpoint-adapters")
    mutation_failure_path = os.path.join(workdir, "mutation-failure")
    score_delivery_failure_path = os.path.join(workdir, "score-delivery-failure")
    abandonment_failure_path = os.path.join(workdir, "abandonment-failure")
    resample_failure_path = os.path.join(workdir, "resample-failure")
    cycle_commit_failure_path = os.path.join(workdir, "cycle-commit-failure")
    for path in (data_dir, shim_dir, local_dir, export_root):
        os.makedirs(path, exist_ok=True)

    materialized_images: dict[int, list[dict[str, str]]] = {}
    if multimodal:
        for index, prompt in enumerate(prompts):
            uris = _materialize_verl_images(
                list(prompt.image_descriptors),
                prompt.package_root,
                image_dir,
                index,
            )
            materialized_images[index] = [{"image": uri} for uri in uris]

    rows = []
    for ordinal in range(update_horizon * prompts_per_step):
        index = ordinal % len(prompts)
        prompt = prompts[index]
        row = {
            "prompt": (
                [
                    {
                        "role": str(message.get("role") or ""),
                        "content": _verl_image_message_content(message.get("content")),
                    }
                    for message in prompt.student_messages
                ]
                if multimodal
                else prompt.student_messages
            ),
            "data_source": "flash_opd",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"index": index},
        }
        if multimodal:
            row["images"] = materialized_images[index]
        rows.append(row)
    train_file = os.path.join(data_dir, "train.parquet")
    val_file = os.path.join(data_dir, "val.parquet")
    _write_opd_parquet(rows, train_file)
    _write_opd_parquet([rows[0]], val_file)

    lora_config = _w.make_lora(model_id)
    lora_rank = int(lora_config.r)
    lora_alpha = int(lora_config.lora_alpha)
    target_modules = lora_config.target_modules
    if isinstance(target_modules, set | frozenset):
        target_modules = sorted(target_modules)
    warmstart_adapter = _warmstart_adapter_path(model_id, model_revision, lora_rank)
    # same silent boundary the sft path guards: with no prebuilt worker image this builds a venv and
    # installs the training stack, minutes long with nothing to report and no liveness thread running.
    with liveness_heartbeat("opd_configuring"):
        python_bin = resolve_verl_python(
            workdir, install_wandb=bool(os.environ.get("WANDB_API_KEY"))
        )
    model_path = _cached_model_path(model_id, model_revision)
    gpu_count = int(getattr(spec.gpu, "count", 1) or 1)
    save_freq = math.gcd(*knobs.save_at_steps) if knobs.save_at_steps else knobs.save_every
    # verl logs from python_bin, so gate wandb on THAT interpreter (see resolve_verl_loggers).
    loggers = resolve_verl_loggers(python_bin)
    project_name = (spec.wandb.project if spec and spec.wandb else None) or "flash"
    experiment_name = _w.wandb_run_name()
    # fp8 kv cache on ada/hopper+ (cc >= 8.9), matching rl_train's grpo gate -- but NOT for hybrid
    # linear-attention (gdn) models: vllm's fp8-kv wake path (init_fp8_kv_scales) assumes a plain kv
    # tensor and crashes on the hybrid cache under verl's sleep/wake, which opd leaves enabled.
    # the vram estimator applies an fp8 discount to opd above the non-fp8 card ceiling, so this must
    # stay in lockstep with it: bf16 here against an fp8-sized reservation OOMs at rollout init.
    try:
        import torch as _torch_cc

        from flash.engine.worker.packing import model_is_gdn_hybrid

        _cc_ok = bool(
            _torch_cc.cuda.is_available() and _torch_cc.cuda.get_device_capability() >= (8, 9)
        )
        fp8_kv = _cc_ok and not model_is_gdn_hybrid(model_id, revision=model_revision)
    except Exception:  # no cuda / probe failure -> conservative bf16 kv
        fp8_kv = False

    # vllm 0.19.1 graph capture is only validated on a100/h100/blackwell; elsewhere it dies in
    # aot_compile or triton slot-mapping, so the rollout runs eagerly. grpo has resolved this since
    # the trl driver; opd never did, and opd is the MORE exposed of the two because it always runs
    # `rollout.mode=async`, whose server hardcodes cudagraph_mode=FULL_AND_PIECEWISE
    # (vllm_async_server.py:240). on an rtx 4090 (sm89, the catalog's recommended card for the small
    # models) that captured 102 graphs and pushed the box to 41.51GB/42.84GB of HOST ram, and the
    # weight sync at the end of the first opd step was killed. the graphs live in vllm's EngineCore
    # CHILD process, so ray's own accounting saw only 12.45GB of it and the run read as a mystery
    # oom. see resolve_rollout_enforce_eager for why this one knob is enough and cannot fight
    # verl's: vllm resolves enforce_eager LAST (config/vllm.py:1024), after the async server has set
    # cudagraph_mode, and forces both compilation mode and cudagraph_mode to NONE.
    # one capability probe feeds both rollout decisions, as it does on the grpo path.
    verl_cc = resolve_verl_device_capability(python_bin)
    enforce_eager = resolve_rollout_enforce_eager(verl_cc)
    # the same grpo/opd divergence as enforce_eager above, one knob over: blackwell needs both
    # rollout attention backends pinned because vllm 0.19.1's defaults are wrong there, and opd
    # never pinned them. the ViT default routes into a CUTE flash-attn that is unimportable against
    # every published nvidia-cutlass-dsl, which aborts the engine with
    # `RuntimeError: Worker failed with error 'module 'cutlass.cute.core' has no attribute
    # 'ThrMma''` -- and a VL model builds its vision tower even for a text-only rollout, so this
    # reaches text-only opd too. no-op off blackwell. see resolve_blackwell_attention_backends.
    attention_backend, mm_encoder_attn_backend = resolve_blackwell_attention_backends(
        python_bin, verl_cc
    )

    plugin_path = os.path.join(shim_dir, "flash_opd_plugin.py")
    shutil.copy2(os.path.join(os.path.dirname(__file__), "opd_plugin.py"), plugin_path)
    structured_helper_path = os.path.join(shim_dir, "flash_opd_structured.py")
    shutil.copy2(
        os.path.join(os.path.dirname(__file__), "opd_structured.py"),
        structured_helper_path,
    )
    multiturn_helper_path = os.path.join(shim_dir, "flash_opd_multiturn.py")
    shutil.copy2(
        os.path.join(os.path.dirname(__file__), "opd_multiturn.py"),
        multiturn_helper_path,
    )
    glue_helper_path = os.path.join(shim_dir, "flash_multiturn_glue.py")
    shutil.copy2(
        os.path.join(os.path.dirname(__file__), "multiturn_glue.py"),
        glue_helper_path,
    )
    entry_path = os.path.join(shim_dir, "flash_opd_entry.py")
    with open(entry_path, "w", encoding="utf-8") as file:
        file.write("import verl\nfrom flash_opd_plugin import main\nmain()\n")
    # opd carries no task reward: use_task_rewards=false makes verl discard the policy loss the
    # score would feed. the reward loop still runs regardless, and with no custom function it falls
    # through to the default rule-based scorer, which dispatches on data_source against a builtin
    # registry that has never heard of "flash_opd" and raises NotImplementedError on every rollout
    # (reward_loop.py:146-155). supply the zero function so the loop takes the custom branch and
    # never consults that registry.
    reward_path = os.path.join(shim_dir, "flash_opd_reward.py")
    with open(reward_path, "w", encoding="utf-8") as file:
        file.write(_OPD_ZERO_REWARD_SOURCE)
    opd_shim_source = _render_opd_sitecustomize(
        save_at_steps=knobs.save_at_steps,
        total_steps=update_horizon,
    )
    if "wandb" in loggers:
        opd_shim_source += render_wandb_link_shim()
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as file:
        file.write(opd_shim_source)

    resume_step, resume_state = _restore_verl_resume(
        local_dir,
        prompt_pool_fingerprint=prompt_pool_fingerprint,
        update_horizon=update_horizon,
    )
    bridge = _TeacherAlignmentBridge(
        prompts=prompts,
        tokenizer=tokenizer,
        teacher=teacher,
        thinking_prefill=thinking_prefill,
        eos_token_ids=eos_token_ids,
        stop_sequences=tuple(str(value) for value in knobs.stop_sequences),
        structured=structured_outputs is not None,
        active_env=env if multi_turn else None,
        multi_turn=multi_turn,
        max_turns=max_turns,
        thinking=bool(_w.THINKING),
        mutation_callback=_w.publish_opd_optimizer_start_marker,
        initial_state=resume_state,
    )
    bridge.start()
    try:
        config = {
            "train_files": [train_file],
            "val_files": [val_file],
            "train_batch_size": prompts_per_step,
            "max_prompt_length": prompt_budget,
            "max_response_length": knobs.max_completion,
            "model_path": model_path,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
            "target_parameters": _w.lora_target_parameters(model_id),
            "lora_adapter_path": warmstart_adapter,
            "learning_rate": knobs.learning_rate,
            "local_dir": local_dir,
            "save_freq": save_freq,
            "n_gpus_per_node": gpu_count,
            "ulysses_sequence_parallel_size": gpu_count,
            "seed": _w.backend_seed(_w.SEED),
            "project_name": project_name,
            "experiment_name": experiment_name,
            "total_training_steps": update_horizon,
            "group_size": knobs.group_size,
            "bridge_url": bridge.url,
            "bridge_token": bridge.token,
            "reward_path": reward_path,
            "kl_penalty_coef": knobs.kl_coef,
            "temperature": knobs.temperature,
            "top_p": knobs.top_p,
            # the job's own engine length (prompt + completion), already clamped to the model's
            # limit. prompt_budget above is carved out of this same value, so the engine, the prompt
            # filter, and the token budget cannot disagree. a hardcoded engine would size vllm's kv
            # cache for a context the job never uses, and -- above it -- admit prompts the engine
            # cannot hold.
            "max_sequence_length": max_model_len,
            "multi_turn": multi_turn,
            "thinking": bool(_w.THINKING),
            "structured_outputs": structured_outputs,
            "fp8_kv": fp8_kv,
            "enforce_eager": enforce_eager,
            "attention_backend": attention_backend,
            "mm_encoder_attn_backend": mm_encoder_attn_backend,
            "sleep_unsupported": rollout_sleep_unsupported(model_id),
            "loggers": loggers,
        }
        overrides = build_opd_overrides(config)
        progress_state = _OpdProgressState(resume_state)
        watcher = _OpdVerlCheckpointWatcher(
            local_dir=local_dir,
            export_root=export_root,
            python_bin=python_bin,
            model_id=model_id,
            model_revision=model_revision,
            required_steps=knobs.save_at_steps,
            seed=int(_w.SEED),
            prompt_pool_fingerprint=prompt_pool_fingerprint,
            prompts_per_step=prompts_per_step,
            group_size=knobs.group_size,
            accounting_state=progress_state.checkpoint_state,
        )
        watcher.processed_steps.update(_processed_resume_steps(knobs.save_at_steps, resume_step))
        child_env = _build_opd_child_env(
            shim_dir=shim_dir,
            wandb_enabled="wandb" in loggers,
            bridge_url=bridge.url,
            bridge_token=bridge.token,
            seed=int(_w.SEED),
            stop_sequences=knobs.stop_sequences,
            eos_token_ids=eos_token_ids,
            structured_outputs=structured_outputs,
            model_vocab_size=model_vocab_size,
            thinking=bool(_w.THINKING),
            multi_turn=multi_turn,
            max_turns=max_turns,
            max_model_len=max_model_len,
            mutation_failure_path=mutation_failure_path,
            score_delivery_failure_path=score_delivery_failure_path,
            abandonment_failure_path=abandonment_failure_path,
            resample_failure_path=resample_failure_path,
            cycle_commit_failure_path=cycle_commit_failure_path,
        )
        command = [python_bin, entry_path, *overrides]
        progress = {"step": resume_step, "loss": None}
        wandb_link: dict[str, str | None] = {}

        def on_line(line: str) -> None:
            watcher.raise_if_failed()
            link = parse_wandb_link(line)
            if link is not None:
                wandb_link.update(link)
            step_number = verl_step_number(line)
            if step_number is None:
                return
            # parse_verl_metric, not a local float(): verl aggregates this metric with
            # Metric(SUM) -> np.sum (verl/utils/metric/utils.py), and LocalLogger renders it
            # through pprint, so under the image's numpy 2.2.6 it prints as
            # "np.float64(0.64)". a bare float() raises on that spelling, dropping every step
            # and leaving loss_curve empty -- which the publish guard below turns into a hard
            # failure on a run that actually trained.
            loss = parse_verl_metric(line, "actor/distillation/loss")
            if loss is None:
                loss = parse_verl_metric(line, "distillation/loss")
            if loss is None:
                # verl emits step-tagged lines that are not metric summaries (timers, val lines);
                # skip those rather than killing the run. the end-of-run guard still fails loud
                # when NO step ever produced a distillation loss.
                return
            progress["loss"] = loss
            progress_state.record_step(step_number, loss, bridge)

        def on_step(step: int) -> None:
            progress["step"] = step
            payload = {"step": step}
            if progress["loss"] is not None:
                payload["loss"] = progress["loss"]
            _w.heartbeat("opd_step", **payload)

        def child_heartbeat() -> None:
            _w.heartbeat("opd_step", liveness=True, step=int(progress["step"] or 0))

        child_tail = ChildOutputTail()
        # one instance for the whole run: it measures silence ACROSS ticks, so it cannot live inside
        # the per-tick callback.
        tail_staleness = ChildTailStaleness()

        def liveness_fields() -> dict[str, object]:
            return stall_tail_fields(
                int(progress["step"] or 0), child_tail, staleness=tail_staleness
            )

        gpu_sampler = _NvidiaSmiPeakSampler().start()
        train_started_at = time.time()
        return_code = 0
        training_completed = resume_step >= update_horizon
        watcher.start()
        try:
            if resume_step < update_horizon:
                progress_state.start_training()
                with liveness_heartbeat(
                    "opd_step",
                    progress=lambda: int(progress["step"] or 0),
                    progress_step=True,
                    fields=liveness_fields,
                ):
                    return_code = run_verl_training(
                        command,
                        env=child_env,
                        on_step=on_step,
                        on_line=on_line,
                        heartbeat=child_heartbeat,
                        tail=child_tail,
                    )
                    training_completed = return_code == 0
        finally:
            watcher.stop(require_complete=training_completed)
        peak_gpu_gb = gpu_sampler.stop_gb()
        score_delivery_failure = _reconcile_score_delivery_failure(
            bridge,
            _read_classified_failure_fallback(score_delivery_failure_path),
        )
        no_signal_failure = _reconcile_no_signal_notification_failure(
            bridge,
            (
                _read_classified_failure_fallback(resample_failure_path),
                _read_classified_failure_fallback(abandonment_failure_path),
            ),
        )
        fallback_mutation_failure = _read_classified_failure_fallback(mutation_failure_path)
        if fallback_mutation_failure is not None:
            bridge._record_mutation_failure(*fallback_mutation_failure)
        cycle_commit_failure = _read_classified_failure_fallback(cycle_commit_failure_path)
        _raise_verl_failure(
            return_code,
            bridge.teacher_failure,
            bridge.mutation_failure,
            cycle_commit_failure,
            no_signal_failure,
            score_delivery_failure,
        )
        final_accounting = progress_state.final_state(bridge)
        train_wall = float(final_accounting["train_wall_seconds"])

        actor_dir, final_step = latest_global_step_dir(local_dir)
        if final_step < update_horizon:
            raise RuntimeError(
                f"opd completed {final_step}/{update_horizon} requested optimizer updates"
            )
        if not final_accounting["loss_curve"]:
            raise RuntimeError(
                "verl OPD produced no distillation-loss metrics for the whole run — the "
                "distillation path never engaged; refusing to publish"
            )
        if len(final_accounting["loss_curve"]) != final_step:
            # record_step only checks that each metric line FOLLOWS the last one, so a missing
            # trailing metric (on_line skips any step-tagged line whose loss it cannot parse) leaves
            # a curve shorter than the checkpoint verl actually wrote, and nothing later arrives to
            # catch it. opt_steps is published from this curve, so a short curve would understate the
            # updates applied. fail loud instead of reporting a number the curve cannot support.
            raise RuntimeError(
                f"verl OPD recorded {len(final_accounting['loss_curve'])} distillation-loss metrics "
                f"for {final_step} optimizer updates; refusing to publish an accounting that does "
                "not cover every update"
            )
        if int(final_accounting.get("aligned_sequences", 0) or 0) <= 0:
            # zeroed-mask pass-through batches still emit a (zero) loss metric, so the loss-curve
            # check alone cannot distinguish real distillation from a run where the teacher never
            # aligned once. require at least one aligned sequence before publishing.
            raise RuntimeError(
                "verl OPD saw zero aligned teacher sequences for the whole run — every batch was "
                "no-signal; refusing to publish an unchanged adapter"
            )
        adapter_dir = os.path.join(workdir, "adapter")
        with liveness_heartbeat(
            "opd_finalizing", progress=lambda: final_step, progress_step=True, keepalive=True
        ):
            _export_checkpoint_adapter(
                actor_dir,
                adapter_dir,
                model_id=model_id,
                model_revision=model_revision,
                python_bin=python_bin,
            )
            _w.hf_upload_folder(adapter_dir, "adapter", required=True)
            # preserve the final checkpoint only when exact save steps are not configured, exactly as
            # the grpo path does: with save_at_steps set the customer asked for those steps and
            # nothing else, and the watcher has already published each of them.
            #
            # NOT also gated on watcher.processed_steps. the watcher marks every step it processes
            # but publishes a deployable only for a step in required_steps (== save_at_steps), and
            # final_save_due is true only when save_at_steps is EMPTY -- so the two publish paths are
            # disjoint and that guard could never prevent a double-publish, only suppress the last
            # step's deployable on every default run.
            if final_save_due(final_step, knobs.save_at_steps):
                _w.publish_deployable_checkpoint(adapter_dir, final_step, _provenance_ready=True)

        setup_seconds = train_started_at - started_at
        _w.heartbeat(
            "opd_trained",
            step=final_step,
            train_wall=train_wall,
            gpu=_w.gpu_diagnostics(include_torch=False),
        )
        _w.write_train_meta(
            phase="opd",
            step=final_step,
            adapter_dir=adapter_dir,
            model_id=model_id,
            train_wall=train_wall,
            setup_seconds=setup_seconds,
            train_tokens=0,
            generated_tokens=int(final_accounting["generated_tokens"]),
            notes={
                "steps": update_horizon,
                # optimizer updates that actually produced a distillation loss. record_step enforces
                # loss_curve length == the metric step, and the guard above rejects a curve shorter
                # than final_step, so this is measured, not assumed.
                "opt_steps": len(final_accounting["loss_curve"]),
                "epochs": knobs.epochs,
                "retained_prompts": len(prompts),
                "dropped_long_prompts": dropped_long,
                "method": "gkd",
                "init_from_adapter": spec.train.init_from_adapter or None,
                "teacher_model": knobs.teacher_model,
                "download_seconds": download_seconds,
                "thinking": _w.THINKING,
                "loss_curve": final_accounting["loss_curve"],
                "mean_coverage": (
                    float(final_accounting["coverage_sum"])
                    / int(final_accounting["aligned_sequences"])
                    if final_accounting["aligned_sequences"]
                    else 0.0
                ),
                # the real alignment-health signal. mean_coverage reads ~1.0 even when the alignment
                # has collapsed every student token onto one group, so it cannot flag that failure
                # mode on its own; this ratio can.
                "mean_align_granularity": (
                    float(final_accounting["align_group_sum"])
                    / int(final_accounting["align_group_n"])
                    if final_accounting["align_group_n"]
                    else 0.0
                ),
                "truncated_rollouts": int(final_accounting["truncated_rollouts"]),
                "forced_tokens": int(final_accounting["forced_tokens"]),
                "dropped_forced_groups": int(final_accounting["dropped_forced_groups"]),
                "teacher_input_tokens": int(final_accounting["teacher_input_tokens"]),
                "teacher_output_tokens": int(final_accounting["teacher_output_tokens"]),
                "aligned_sequences": int(final_accounting["aligned_sequences"]),
                "empty_alignments": int(final_accounting["empty_alignments"]),
                "teacher_ok": int(final_accounting["teacher_ok"]),
                **_failure_accounting_metadata(final_accounting),
                "temperature": knobs.temperature,
                "group_size": knobs.group_size,
                "prompts_per_step": prompts_per_step,
                "max_completion_len": knobs.max_completion,
                "multi_turn": multi_turn,
                "max_turns": max_turns if multi_turn else None,
                "episodes": int(final_accounting["episodes_seen"]) if multi_turn else None,
                "mean_turns_per_episode": (
                    int(final_accounting["mt_turn_records"])
                    / int(final_accounting["episodes_seen"])
                    if multi_turn and final_accounting["episodes_seen"]
                    else None
                ),
                # the engine length actually handed to vllm (prompt + completion), already clamped to
                # the model's own limit. the prompt filter is carved out of this same number.
                "vllm_max_model_len": max_model_len,
                # teacher call shape. only the single-turn TEXT path goes through the batcher, which
                # holds a fixed cap and one serial scoring thread. the multimodal path scores one
                # item per call and the multi-turn path batches a whole episode, and both run on the
                # bridge's own request threads -- neither has a constant, so None records "not
                # batched by flash" instead of asserting a number nothing enforces.
                # the cap is bounded by the samples one step can actually produce (trl does the same
                # in _opd_teacher_batch_size): a step of 1 rollout can never fill a batch of 8, and
                # reporting the global cap there would describe a shape the run cannot reach.
                "opd_teacher_batch_size": (
                    min(
                        OPD_TEACHER_SCORING_CONCURRENCY, max(1, prompts_per_step * knobs.group_size)
                    )
                    if not multimodal and not multi_turn
                    else None
                ),
                "opd_teacher_workers": 1 if not multimodal and not multi_turn else None,
                "rollout_backend": "verl_vllm",
                "verl_version": "0.8.0",
                "verl_backend": "fsdp",
                "ulysses_sequence_parallel_size": gpu_count,
                "peak_gpu_gb": peak_gpu_gb,
                "warm_started": bool(warmstart_adapter),
                "resumed": bool(resume_step),
                "wandb_project": project_name if "wandb" in loggers else None,
                "wandb_run_name": experiment_name if "wandb" in loggers else None,
                # the sdk's link_wandb reads notes["wandb_url"]; trl gets it from the parent's live
                # wandb.run, verl from the child marker (see backend_common.render_wandb_link_shim).
                "wandb_url": wandb_link.get("wandb_url"),
                "wandb_id": wandb_link.get("wandb_id"),
            },
        )
    finally:
        bridge.close()
