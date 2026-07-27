"""Flash OPD orchestration through verl 0.8.0 in an isolated child interpreter."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.opd import (
    _drop_fully_forced_groups,
    _resolve_opd_knobs,
    _thinking_prefill_text,
)
from flash.engine.worker.opd_gkd import (
    _generation_eos_ids,
    _rollout_terminated,
    _teacher_prompt_text,
    _trim_trailing_stop,
    student_tokens_with_offsets,
)
from flash.engine.worker.opd_verl_multiturn import (
    EnvGlueTokenizer,
    _dedup_seam_terminator,
    validate_glue_template,
    validate_teacher_messages,
)
from flash.engine.worker.sft_verl import (
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
from flash.engine.worker.teacher import TeacherError
from flash.engine.worker.tokenizer_align import groupwise_alignment, groupwise_coverage
from flash.engine.worker.verl_common import (
    latest_global_step_dir,
    resolve_verl_python,
    run_verl_training,
)
from flash.opd_retry_contract import OPD_RESUME_STATE_VERSION, validate_opd_resume_state_metadata

_VERL_STEP_RE = re.compile(r"(?:^|\s)step:(\d+)(?:\s|$)")
_VERL_METRIC_RE = re.compile(r"(?:^| - )(?P<name>[^:]+):(?P<value>[^ ]+)")
_PERMANENT_TEACHER_EXIT = 86
_TRANSIENT_TEACHER_EXIT = 87


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
            fingerprint_fields.extend(
                [prompt.teacher_messages, list(prompt.image_descriptors)]
            )
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
    return tuple(int(token_id.item() if hasattr(token_id, "item") else token_id) for token_id in value)


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
        self.aligned_sequences = int(state.get("aligned_sequences", state.get("granularity_n", 0)))
        self.empty_alignments = int(
            state.get("empty_alignments", dict(state.get("skip_counts", {})).get("empty_alignment", 0))
        )
        self.truncated_rollouts = int(state.get("truncated_rollouts", 0))
        self.forced_tokens = int(state.get("forced_tokens", 0))
        self.dropped_forced_groups = int(state.get("dropped_forced_groups", 0))
        self.coverage_sum = float(state.get("coverage_sum", state.get("granularity_sum", 0.0)))
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

    def _record_mutation_failure(self, classification: str, message: str) -> None:
        with self._stats_lock:
            if classification == "permanent" or self._mutation_failure is None:
                self._mutation_failure = (classification, message)

    @property
    def mutation_failure(self) -> tuple[str, str] | None:
        with self._stats_lock:
            return self._mutation_failure

    def accounting_snapshot(self) -> dict:
        with self._stats_lock:
            skip_counts = dict(self.skip_counts)
            skip_counts["empty_alignment"] = self.empty_alignments
            return {
                "generated_tokens": self.generated_tokens,
                "teacher_input_tokens": self.teacher_input_tokens,
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
        prompt_ids = list(prompt.prompt_ids)
        prompt_length = int(prompt_length)
        sequence_ids = [int(token_id) for token_id in sequence_ids]
        if prompt_length != len(prompt_ids) or sequence_ids[:prompt_length] != prompt_ids:
            raise ValueError("verl rollout prompt ids do not exactly match the frozen flash prompt pool")
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
        teacher_images = None
        if prompt.image_descriptors:
            from flash.multimodal import image_descriptors_to_data_uris

            teacher_images = image_descriptors_to_data_uris(
                prompt.image_descriptors, prompt.package_root
            )
        try:
            if teacher_images is not None:
                teacher_batches = self.teacher.score_many_multimodal(
                    [(teacher_prompt, completion_text, teacher_images)]
                )
            else:
                teacher_tokens = self.teacher.score(teacher_prompt, completion_text)
        except TeacherError as error:
            if error.permanent:
                raise
            failure = ("transient", str(error))
            self._record_teacher_failure(*failure)
            if recovered_failure is not None:
                recovered_failure.append(failure)
            return self._empty(prompt_length, len(response_ids))
        if teacher_images is not None:
            teacher_tokens = teacher_batches[0]
            teacher_input_tokens = int(getattr(teacher_tokens, "input_tokens", 0) or 0)
        else:
            teacher_input_tokens = 0
        with self._stats_lock:
            self.teacher_ok += 1
            self._pending_teacher_success = True
        student_ids, student_tokens = student_tokens_with_offsets(
            self.tokenizer, kept_ids, completion_text
        )
        if not prompt.image_descriptors:
            teacher_input_tokens = prompt_length + len(student_ids)
        groups = groupwise_alignment(student_tokens, teacher_tokens)
        groups = [(indices, logsum) for indices, logsum in groups if indices]
        aligned_group_count = len(groups)
        groups = _drop_fully_forced_groups(groups, kept_forced)
        coverage = groupwise_coverage(groups, student_tokens)
        with self._stats_lock:
            self.teacher_input_tokens += teacher_input_tokens
            self.dropped_forced_groups += aligned_group_count - len(groups)
            self.coverage_sum += coverage
            if groups:
                self.aligned_sequences += 1
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
        raw_prompt = validate_teacher_messages(raw_prompt, source="child initial prompt")
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
                initial_messages = validate_teacher_messages(
                    initial_messages, source="environment initial prompt"
                )
            if initial_messages != prompt.student_messages:
                raise ValueError("multi-turn environment initial prompt changed after prompt freezing")
            per_example_limit = state.get("max_episode_turns")
            if per_example_limit is not None:
                try:
                    per_example_limit = int(per_example_limit)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "multi-turn environment returned an invalid per-example turn limit"
                    ) from error
                if per_example_limit <= 0:
                    raise ValueError("multi-turn environment requires a positive per-example turn limit")
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
                raise ValueError("multi-turn truncated assistant ids must preserve the sampled span")
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
            session["messages"].append(
                {"role": "assistant", "content": completion_text}
            )
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
                messages = validate_teacher_messages(messages, source="environment reply")
                session["messages"].extend(messages)
                # the env's reply may itself end the episode (rollout_done consults the updated
                # state); recheck before gluing a next-turn prompt no model turn will answer.
                terminal = not messages or self.active_env.rollout_done(
                    state, session["turn_limit"]
                )
                if not terminal:
                    assert self._env_glue is not None
                    next_prefix.extend(
                        _dedup_seam_terminator(response_ids, self._env_glue(messages))
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
                self._empty(len(turn["prompt_ids"]), len(turn["response_ids"]))
                for turn in turns
            ]
            scorable = [
                position
                for position, turn in enumerate(turns)
                if not turn["truncated"]
                and not turn["skip_reason"]
                and turn["response_ids"]
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
                teacher_batches = self.teacher.score_many(items)
                if len(teacher_batches) != len(scorable):
                    raise RuntimeError("teacher returned the wrong number of multi-turn OPD scores")
                with self._stats_lock:
                    self.teacher_ok += len(teacher_batches)
                for position, teacher_tokens in zip(scorable, teacher_batches, strict=True):
                    turn = turns[position]
                    response_ids = turn["response_ids"]
                    student_ids, student_tokens = student_tokens_with_offsets(
                        self.tokenizer, response_ids, turn["completion_text"]
                    )
                    groups = groupwise_alignment(student_tokens, teacher_tokens)
                    groups = [(indices, logsum) for indices, logsum in groups if indices]
                    coverage = groupwise_coverage(groups, student_tokens)
                    with self._stats_lock:
                        self.teacher_input_tokens += len(turn["prompt_ids"]) + len(student_ids)
                        self.coverage_sum += coverage
                        if groups:
                            self.aligned_sequences += 1
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

    def notify_mutation(self) -> None:
        with self._stats_lock:
            self._pending_teacher_transient = None
            self._pending_teacher_success = False
        with self._mutation_lock:
            if self._mutation_notified:
                return
            try:
                self.mutation_callback()
            except Exception as error:
                classification = (
                    "transient"
                    if isinstance(error, _w.RetriableInfraError)
                    else "permanent"
                )
                self._record_mutation_failure(classification, str(error))
                raise
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
                    elif self.path == "/mutation":
                        bridge.notify_mutation()
                        result = {"ok": True}
                    else:
                        raise ValueError("flash OPD bridge path is unknown")
                    self._send_json(200, result)
                except Exception as error:
                    classification = (
                        "transient"
                        if isinstance(error, _w.RetriableInfraError)
                        or (isinstance(error, TeacherError) and not error.permanent)
                        else "permanent"
                    )
                    if self.path == "/score":
                        if recovered_teacher_failure is not None:
                            bridge._promote_recovered_teacher_failure(
                                recovered_teacher_failure
                            )
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

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
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
            coverage = (d_cov / d_aligned) if d_aligned > 0 else (cov_sum / aligned if aligned else 0.0)
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
)


def build_opd_verl_overrides(config: dict) -> list[str]:
    """Render the exact verl 0.8.0 synchronous PPO and distillation config surface."""
    missing = [key for key in _REQUIRED_OVERRIDE_KEYS if key not in config]
    if missing:
        raise KeyError(f"build_opd_verl_overrides missing required config keys: {missing}")
    sequence_length = config.get("max_sequence_length")
    max_tokens = int(
        sequence_length
        if sequence_length is not None
        else int(config["max_prompt_length"]) + int(config["max_response_length"])
    )
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
        f"actor_rollout_ref.rollout.seed={_hydra_val(config.get('seed', 42))}",
        "data.dataloader_num_workers=0",
        "data.image_key=images",
        "data.return_raw_chat=true",
        "data.return_multi_modal_inputs=false",
        "data.apply_chat_template_kwargs={enable_thinking:" + _hydra_val(config.get("thinking", False)) + "}",
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
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={_hydra_val(config['n_gpus_per_node'])}",
        f"actor_rollout_ref.rollout.n={_hydra_val(config['group_size'])}",
        "actor_rollout_ref.rollout.limit_images=8",
        f"actor_rollout_ref.rollout.max_model_len={_hydra_val(config.get('max_model_len', 32768))}",
        f"actor_rollout_ref.rollout.temperature={_hydra_val(config.get('temperature', 1.0))}",
        f"actor_rollout_ref.rollout.top_p={_hydra_val(config.get('top_p', 1.0))}",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.calculate_log_probs=false",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={_hydra_val(max_tokens)}",
        "actor_rollout_ref.rollout.agent.default_agent_loop="
        + ("flash_multi_turn" if config.get("multi_turn") else "flash_single_turn"),
        "critic.enable=false",
        "reward.reward_model.enable=false",
        "distillation._target_=flash_opd_verl_plugin.FlashRemoteDistillationConfig",
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
        overrides.append(
            f"actor_rollout_ref.rollout.prompt_length={_hydra_val(max_tokens)}"
        )
    structured_outputs = config.get("structured_outputs")
    if structured_outputs:
        structured_outputs_config = {
            "backend": "xgrammar",
            "disable_any_whitespace": bool(
                structured_outputs.get("disable_any_whitespace", False)
            ),
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


def _render_opd_sitecustomize(
    *, save_at_steps: tuple[int, ...], total_steps: int
) -> str:
    required_steps = tuple(int(step) for step in save_at_steps)
    return f'''# generated flash opd runtime patches for verl 0.8
from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler as _FlashCheckpointHandler

_flash_required_save_steps = frozenset({required_steps!r})
_flash_total_steps = {int(total_steps)}
_flash_original_save_checkpoint = _FlashCheckpointHandler.save_checkpoint


def _flash_save_exact_checkpoint(self, step):
    if _flash_required_save_steps and step not in _flash_required_save_steps and step != _flash_total_steps:
        return None
    return _flash_original_save_checkpoint(self, step)


_FlashCheckpointHandler.save_checkpoint = _flash_save_exact_checkpoint
'''


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
) -> dict[str, str]:
    child = _build_verl_child_env(shim_dir=shim_dir, wandb_enabled=wandb_enabled)
    child.update(
        {
            "VERL_USE_EXTERNAL_MODULES": "flash_opd_verl_plugin",
            "FLASH_OPD_BRIDGE_URL": bridge_url,
            "FLASH_OPD_BRIDGE_TOKEN": bridge_token,
            "FLASH_OPD_SEED": str(int(seed)),
            "FLASH_OPD_STOP_SEQUENCES": json.dumps(list(stop_sequences)),
            "FLASH_OPD_EOS_TOKEN_IDS": json.dumps(sorted(eos_token_ids)),
        }
    )
    if mutation_failure_path:
        child["FLASH_OPD_MUTATION_FAILURE_PATH"] = mutation_failure_path
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


def _write_opd_parquet(rows: list[dict], path: str) -> None:
    from datasets import Dataset

    features = _opd_multimodal_parquet_features() if any("images" in row for row in rows) else None
    Dataset.from_list(rows, features=features).to_parquet(path)


def _metric_value(line: str, name: str) -> float | None:
    for match in _VERL_METRIC_RE.finditer(line):
        if match.group("name").strip() != name:
            continue
        try:
            return float(match.group("value"))
        except ValueError:
            return None
    return None


def _read_mutation_failure_fallback(base_path: str) -> tuple[str, str] | None:
    if not base_path:
        return None
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
    for classification in ("permanent", "transient"):
        for failure_classification, message in failures:
            if failure_classification == classification:
                return classification, message
    return None


def _raise_verl_failure(
    return_code: int,
    teacher_failure: tuple[str, str] | None,
    mutation_failure: tuple[str, str] | None = None,
) -> None:
    if return_code == 0:
        return
    if mutation_failure is not None:
        classification, message = mutation_failure
        if classification == "transient":
            raise _w.RetriableInfraError(f"optimizer marker failure: {message}")
        raise RuntimeError(f"permanent optimizer marker failure: {message}")
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


def _processed_resume_steps(
    required_steps: tuple[int, ...], resume_step: int
) -> set[int]:
    processed = _durable_required_save_steps(required_steps, resume_step)
    if resume_step and resume_step not in required_steps:
        processed.add(resume_step)
    return processed


def _generation_eos_from_cached_config(model_id: str, model_revision: str, tokenizer) -> frozenset[int]:
    from transformers import AutoConfig, GenerationConfig

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, revision=model_revision or None, local_files_only=True
    )
    generation_config = None
    with contextlib.suppress(OSError):
        generation_config = GenerationConfig.from_pretrained(
            model_id, revision=model_revision or None, local_files_only=True
        )
    model_like = type(
        "ModelGenerationMetadata",
        (),
        {"config": config, "generation_config": generation_config},
    )()
    return _generation_eos_ids(model_like, tokenizer)


def run_opd_verl(spec=None) -> None:
    """Run flash OPD through verl's native rollout and weight-sync path."""
    from flash.engine.worker.teacher import TeacherClient
    from flash.multimodal import (
        image_teacher_prompt_messages,
        normalize_prompt_images,
        record_has_images,
        validate_image_opd_teacher,
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
    from flash.opd_verl_validation import validate_opd_verl_structured_outputs

    structured_validation = validate_opd_verl_structured_outputs(
        knobs.structured_outputs,
        model_id=model_id,
        model_revision=model_revision,
        model_policy=getattr(spec, "model_policy", "catalog") if spec else "catalog",
        gpu=spec.gpu.type if spec else None,
    )
    structured_outputs = structured_validation.constraint
    model_vocab_size = structured_validation.model_vocab_size
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
        validate_multimodal_training(model_id, "opd", multi_turn=multi_turn)
        validate_image_opd_teacher(knobs.teacher_model)
    random.Random(_w.SEED).shuffle(train)

    started_at = time.time()
    # validate the teacher credential BEFORE the gpu probe + model prefetch: a missing key fails
    # in milliseconds instead of after minutes of paid setup.
    api_key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("the managed teacher api key is missing from the OPD parent worker")
    _w.heartbeat("opd_start", gpu=_w.gpu_diagnostics(include_torch=False))
    _probe_gpu_in_subprocess(
        spec.gpu.type if spec else None,
        exact_type=spec.gpu.type if spec else "",
    )
    download_seconds = _w.prefetch_model(model_id, revision=model_revision)
    teacher = TeacherClient(api_key, knobs.teacher_base_url, knobs.teacher_model)
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
    eos_token_ids = _generation_eos_from_cached_config(model_id, model_revision, tokenizer)
    max_model_len = knobs.max_length or (RECIPE.opd.max_prompt_len + knobs.max_completion)
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
                messages = validate_teacher_messages(messages, source="environment initial prompt")
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
    python_bin = resolve_verl_python(workdir)
    model_path = _cached_model_path(model_id, model_revision)
    gpu_count = int(getattr(spec.gpu, "count", 1) or 1)
    save_freq = math.gcd(*knobs.save_at_steps) if knobs.save_at_steps else knobs.save_every
    loggers = ["console"]
    if os.environ.get("WANDB_API_KEY"):
        loggers.append("wandb")
    project_name = (spec.wandb.project if spec and spec.wandb else None) or "flash"
    experiment_name = _w.wandb_run_name()

    plugin_path = os.path.join(shim_dir, "flash_opd_verl_plugin.py")
    shutil.copy2(os.path.join(os.path.dirname(__file__), "opd_verl_plugin.py"), plugin_path)
    structured_helper_path = os.path.join(shim_dir, "flash_opd_verl_structured.py")
    shutil.copy2(
        os.path.join(os.path.dirname(__file__), "opd_verl_structured.py"),
        structured_helper_path,
    )
    multiturn_helper_path = os.path.join(shim_dir, "flash_opd_verl_multiturn.py")
    shutil.copy2(
        os.path.join(os.path.dirname(__file__), "opd_verl_multiturn.py"),
        multiturn_helper_path,
    )
    entry_path = os.path.join(shim_dir, "flash_opd_verl_entry.py")
    with open(entry_path, "w", encoding="utf-8") as file:
        file.write("import verl\nfrom flash_opd_verl_plugin import main\nmain()\n")
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as file:
        file.write(
            _render_opd_sitecustomize(
                save_at_steps=knobs.save_at_steps,
                total_steps=update_horizon,
            )
        )

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
            "kl_penalty_coef": knobs.kl_coef,
            "temperature": knobs.temperature,
            "top_p": knobs.top_p,
            "max_model_len": max_model_len if multi_turn else 32768,
            "max_sequence_length": max_model_len if multi_turn else None,
            "multi_turn": multi_turn,
            "thinking": bool(_w.THINKING),
            "structured_outputs": structured_outputs,
            "loggers": loggers,
        }
        overrides = build_opd_verl_overrides(config)
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
        watcher.processed_steps.update(
            _processed_resume_steps(knobs.save_at_steps, resume_step)
        )
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
        )
        command = [python_bin, entry_path, *overrides]
        progress = {"step": resume_step, "loss": None}

        def on_line(line: str) -> None:
            watcher.raise_if_failed()
            step_match = _VERL_STEP_RE.search(line)
            if step_match is None:
                return
            loss = _metric_value(line, "actor/distillation/loss")
            if loss is None:
                loss = _metric_value(line, "distillation/loss")
            if loss is None:
                # verl emits step-tagged lines that are not metric summaries (timers, val lines);
                # skip those rather than killing the run. the end-of-run guard still fails loud
                # when NO step ever produced a distillation loss.
                return
            step = int(step_match.group(1))
            progress["loss"] = loss
            progress_state.record_step(step, loss, bridge)

        def on_step(step: int) -> None:
            progress["step"] = step
            payload = {"step": step}
            if progress["loss"] is not None:
                payload["loss"] = progress["loss"]
            _w.heartbeat("opd_step", **payload)

        def child_heartbeat() -> None:
            _w.heartbeat("opd_step", liveness=True, step=int(progress["step"] or 0))

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
                ):
                    return_code = run_verl_training(
                        command,
                        env=child_env,
                        on_step=on_step,
                        on_line=on_line,
                        heartbeat=child_heartbeat,
                    )
                    training_completed = return_code == 0
        finally:
            watcher.stop(require_complete=training_completed)
        peak_gpu_gb = gpu_sampler.stop_gb()
        fallback_mutation_failure = _read_mutation_failure_fallback(
            mutation_failure_path
        )
        if fallback_mutation_failure is not None:
            bridge._record_mutation_failure(*fallback_mutation_failure)
        _raise_verl_failure(
            return_code,
            bridge.teacher_failure,
            bridge.mutation_failure,
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
            if final_save_due(final_step, knobs.save_at_steps) and final_step not in watcher.processed_steps:
                _w.publish_deployable_checkpoint(
                    adapter_dir, final_step, _provenance_ready=True
                )

        setup_seconds = train_started_at - started_at
        _w.heartbeat("opd_trained", step=final_step, train_wall=train_wall, gpu=_w.gpu_diagnostics(include_torch=False))
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
                "truncated_rollouts": int(final_accounting["truncated_rollouts"]),
                "forced_tokens": int(final_accounting["forced_tokens"]),
                "dropped_forced_groups": int(
                    final_accounting["dropped_forced_groups"]
                ),
                "teacher_input_tokens": int(final_accounting["teacher_input_tokens"]),
                "aligned_sequences": int(final_accounting["aligned_sequences"]),
                "empty_alignments": int(final_accounting["empty_alignments"]),
                "teacher_ok": int(final_accounting["teacher_ok"]),
                "teacher_transient": int(final_accounting["teacher_transient"]),
                "teacher_error": int(final_accounting["teacher_error"]),
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
                "rollout_backend": "verl_vllm",
                "verl_version": "0.8.0",
                "verl_backend": "fsdp",
                "ulysses_sequence_parallel_size": gpu_count,
                "peak_gpu_gb": peak_gpu_gb,
                "warm_started": bool(warmstart_adapter),
                "resumed": bool(resume_step),
                "wandb_project": project_name if "wandb" in loggers else None,
                "wandb_run_name": experiment_name if "wandb" in loggers else None,
            },
        )
    finally:
        bridge.close()
