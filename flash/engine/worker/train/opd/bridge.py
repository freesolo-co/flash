"""The teacher-alignment bridge: the http server the verl child scores rollouts against.

OPD supervises the student against a teacher's token distribution, which means every rollout the
child produces has to be tokenized under the teacher's vocabulary, aligned back to the student's
tokens, and returned as per-token logprobs. That exchange runs over a local http server so the
verl child (a separate interpreter) can reach it, and the alignment bookkeeping it needs -- group
coverage, forced-mask validation, seam dedup, no-signal accounting -- makes it by far the largest
piece of the OPD orchestrator.

Split out of `flash.engine.worker.opd_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING

from flash.engine.worker.entry.opd import _drop_fully_forced_groups
from flash.engine.worker.runtime.pkg_proxy import W as _w
from flash.engine.worker.teacher.client import TeacherError
from flash.engine.worker.teacher.tokenizer_align import groupwise_alignment, groupwise_coverage
from flash.engine.worker.train.core.child.glue import (
    EnvGlueTokenizer,
    dedup_seam_terminator,
    validate_transcript_messages,
)
from flash.engine.worker.train.opd.batching import (
    _align_granularity,
    _TeacherBridgeHTTPServer,
    _TextTeacherBatcher,
)
from flash.engine.worker.train.opd.gkd import (
    _rollout_terminated,
    _teacher_prompt_text,
    _trim_trailing_stop,
    student_tokens_with_offsets,
)
from flash.engine.worker.train.opd.prompts import (
    _trim_response_and_forced,
    _validate_forced_mask,
    encode_shifted_group_metadata,
)
from flash.engine.worker.train.opd.scoring import score_rollout
from flash.teacher.limits import OPD_TEACHER_SCORING_CONCURRENCY

if TYPE_CHECKING:  # annotation-only: `opd_train` imports this module, so a runtime import
    # here would be circular. the raise below goes through `_opd_train()`.
    from flash.engine.worker.opd_train import _BridgePrompt


def _opd_train():
    """The orchestrator module, imported lazily because it imports this one.

    Everything reached through this function is a name the OPD tests patch on `opd_train` itself
    (`_TEXT_TEACHER_FLUSH_WAIT_S` is shortened to keep the batcher's flush fast under test). Binding
    them here with a `from ... import` would capture the originals at import time, so the patches
    would rebind an object this module never reads and the bridge would run the real values.
    """
    from flash.engine.worker import opd_train

    return opd_train


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
        raise _opd_train()._RecordedMutationCallbackFailure(classification, message)

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
        try:
            teacher_score = score_rollout(
                prompt,
                completion_text,
                teacher=self.teacher,
                thinking_prefill=self.thinking_prefill,
                text_teacher_batcher=self._text_teacher_batcher,
            )
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
            terminal = bool(payload.get("truncated")) or bool(skip_reason)
            # an unusable turn is excluded from teacher scoring below (see the `scorable` filter),
            # so showing it to the environment buys nothing and can cost the whole run: an env that
            # parses or validates each action may raise on a truncated or empty action, turning a
            # routine no-signal rollout into a permanent paid failure. the grpo bridge already
            # returns before its own `record_model_turn` on exactly this predicate.
            if not terminal:
                self.active_env.record_model_turn(state, completion_text)
                session["messages"].append({"role": "assistant", "content": completion_text})
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
                # ONE call, not a chunk-and-drain loop. score_many already bounds itself to
                # OPD_TEACHER_SCORING_CONCURRENCY workers, so slicing the items first did not lower
                # the provider-facing rate -- it only added a BARRIER every 32 items, where the
                # slowest request in a wave held back the whole next wave and the gpu idled behind
                # it. handing the full list over keeps the same concurrency ceiling while letting a
                # finished worker start the next item immediately. `executor.map` preserves input
                # order, so the zip with `scorable` below is unchanged.
                teacher_batches = self.teacher.score_many(items)
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

    def _routes(self, recovered: list[tuple[str, str]]) -> dict:
        """The bridge's routing table: path -> handler taking the decoded payload.

        Built per request so ``/score`` can hand its recovered-failure list to the scorer, and built
        from ``self`` attributes at call time so a test that swaps a bound method on the instance is
        the method this dispatches to.
        """

        def score(payload: dict) -> dict:
            return self.score(
                payload["index"],
                payload["prompt_length"],
                payload["sequence_ids"],
                payload.get("image_count", 0),
                payload.get("forced"),
                recovered_failure=recovered,
            )

        def mutation(_payload: dict) -> dict:
            self.notify_mutation()
            return {"ok": True}

        return {
            "/score": score,
            "/multiturn/start": lambda payload: self.start_multiturn(
                index=payload["index"],
                session_id=payload["session_id"],
                prompt_ids=payload["prompt_ids"],
                raw_prompt=payload["raw_prompt"],
            ),
            "/multiturn/step": self.step_multiturn,
            "/multiturn/score": lambda payload: self.score_multiturn(payload["session_id"]),
            "/multiturn/close": lambda payload: self.close_multiturn(payload["session_id"]),
            "/no-signal/resample": lambda _payload: self.record_no_signal_resample(),
            "/no-signal/abandoned": lambda _payload: self.record_no_signal_abandoned(),
            "/teacher-cycle/committed": lambda _payload: self.commit_teacher_cycle(),
            "/mutation": mutation,
        }

    def _classify_failure(self, error: BaseException, *, delivery: bool) -> str:
        """Whether ``error`` should make the child retry (``transient``) or give up (``permanent``)."""
        if delivery:
            return "transient"
        if isinstance(error, _opd_train()._RecordedMutationCallbackFailure):
            return error.classification
        retriable = isinstance(error, _w.RetriableInfraError) or (
            isinstance(error, TeacherError) and not error.permanent
        )
        return "transient" if retriable else "permanent"

    def _record_route_failure(
        self,
        path: str,
        error: BaseException,
        classification: str,
        *,
        delivery: bool,
        recovered: tuple[str, str] | None,
    ) -> None:
        """Attribute a failed request to the teacher when the route speaks for the teacher.

        A recovered failure always wins: the scorer already knows which upstream call went wrong, so
        promoting it keeps the run's diagnosis pointed at the original fault rather than the delivery
        error that surfaced it.
        """
        if delivery or path == "/score":
            if recovered is not None:
                self._promote_recovered_teacher_failure(recovered)
            elif delivery:
                self._record_teacher_delivery_failure(error)
            else:
                self._record_teacher_failure(classification, str(error))
        elif path == "/multiturn/score":
            self._record_teacher_failure(classification, str(error), terminal=True)

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
                recovered_failures: list[tuple[str, str]] = []
                request_succeeded = False
                try:
                    if self.headers.get("Authorization") != f"Bearer {bridge.token}":
                        raise PermissionError("flash OPD bridge authorization failed")
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                    route = bridge._routes(recovered_failures).get(self.path)
                    if route is None:
                        raise ValueError("flash OPD bridge path is unknown")
                    result = route(payload)
                    request_succeeded = True
                    self._send_json(200, result)
                except Exception as error:
                    # only a route that already answered can be failing on DELIVERY; before that, an
                    # OSError is the teacher call itself.
                    delivery = (
                        request_succeeded
                        and self.path in {"/score", "/multiturn/score"}
                        and isinstance(error, (OSError, http.client.HTTPException))
                    )
                    classification = bridge._classify_failure(error, delivery=delivery)
                    bridge._record_route_failure(
                        self.path,
                        error,
                        classification,
                        delivery=delivery,
                        recovered=recovered_failures[0] if recovered_failures else None,
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
            flush_wait_s=_opd_train()._TEXT_TEACHER_FLUSH_WAIT_S,
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
