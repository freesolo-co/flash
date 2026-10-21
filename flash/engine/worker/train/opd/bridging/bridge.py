"""The teacher-alignment bridge: the http server the verl child scores rollouts against.

OPD supervises the student against a teacher's token distribution, which means every rollout the
child produces has to be tokenized under the teacher's vocabulary, aligned back to the student's
tokens, and returned as per-token logprobs. That exchange runs over a local http server so the
verl child (a separate interpreter) can reach it, and the alignment bookkeeping it needs -- group
coverage, forced-mask validation, seam dedup, no-signal accounting -- makes it by far the largest
piece of the OPD orchestrator.

Split out of `flash.engine.worker.train.entry.opd_train` to keep that module under the file-size limit.
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

import flash.engine.worker.perf as _worker_perf
from flash.content.multimodal import normalize_environment_reply
from flash.engine.worker.entry.opd import _drop_fully_forced_groups
from flash.engine.worker.teacher.client import TeacherClient, TeacherError
from flash.engine.worker.teacher.tokenizer_align import groupwise_alignment, groupwise_coverage
from flash.engine.worker.train.core.child.glue import validate_structured_messages
from flash.engine.worker.train.opd.bridging.batching import (
    _align_granularity,
    _TeacherBridgeHTTPServer,
    _TextTeacherBatcher,
)
from flash.engine.worker.train.opd.bridging.failures import (
    TeacherFailureRecording,
    _RecordedMutationCallbackFailure,
)
from flash.engine.worker.train.opd.bridging.prompts import (
    _trim_response_and_forced,
    _validate_forced_mask,
    encode_shifted_group_metadata,
)
from flash.engine.worker.train.opd.bridging.scoring import (
    build_multimodal_score_items,
    score_multimodal_items,
    score_rollout,
)
from flash.engine.worker.train.opd.multiturn.media import (
    normalize_initial_prompt,
    prepare_environment_reply,
    step_media_identity,
    validate_start_media,
)
from flash.engine.worker.train.opd.multiturn.validation import validated_multiturn_response
from flash.engine.worker.train.opd.orchestration import protocol
from flash.engine.worker.train.opd.orchestration.gkd import (
    _rollout_terminated,
    _teacher_prompt_text,
    student_tokens_with_offsets,
)
from flash.engine.worker.train.opd.orchestration.state import _BridgePrompt
from flash.engine.worker.verl.parent_work import ParentWorkGauge
from flash.teacher.limits import OPD_TEACHER_SCORING_CONCURRENCY


class _TeacherAlignmentBridge(TeacherFailureRecording):
    def __init__(
        self,
        *,
        prompts: list[_BridgePrompt],
        processor=None,
        tokenizer=None,
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
        self.processor = processor
        self.tokenizer = tokenizer
        self.teacher = teacher
        self.thinking_prefill = thinking_prefill
        self.eos_token_ids = eos_token_ids
        self.stop_sequences = stop_sequences
        self.structured = bool(structured)
        self.active_env = active_env
        self.multi_turn = bool(multi_turn)
        self.max_turns = int(max_turns)
        self.thinking = bool(thinking)
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
        self.parent_work = ParentWorkGauge()
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
        self.aligned_sequences = int(state.get("aligned_sequences", 0))
        self.empty_alignments = int(
            state.get(
                "empty_alignments", dict(state.get("skip_counts", {})).get("empty_alignment", 0)
            )
        )
        self.truncated_rollouts = int(state.get("truncated_rollouts", 0))
        self.forced_tokens = int(state.get("forced_tokens", 0))
        self.dropped_forced_groups = int(state.get("dropped_forced_groups", 0))
        self.coverage_sum = float(state.get("coverage_sum", 0.0))
        # alignment GRANULARITY (mean aligned-groups-per-sequence), distinct from coverage: a
        # collapsed alignment that maps every student token onto one group still scores coverage
        # ~1.0, so coverage alone cannot flag that failure mode. the zero default is reachable only
        # on a FRESH start (no `initial_state`), where nothing has been measured yet and 0 is the
        # true count; a resume state is required by `validate_opd_resume_state_metadata` to carry
        # both, so an absent field there is rejected rather than silently defaulted -- otherwise the
        # published `mean_align_granularity` would read 0.0 for a run that measured every group.
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
        # what `skip_counts` already held at the start of the current step. the fatal no-signal
        # message subtracts this so it reports only the gates that fired for THAT step. seeded from
        # the restored counts, never from empty: on resume the rehydrated totals belong to steps
        # that already happened, and a zero baseline would re-blame every one of them.
        self._skip_baseline = dict(self.skip_counts)
        self.opd_phase_seconds = dict(state.get("opd_phase_seconds", {}))
        self.opd_phase_counts = dict(state.get("opd_phase_counts", {}))
        self._init_failure_state()

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
                "samples_seen": self.score_requests,
                "teacher_ok": self.teacher_ok,
                "teacher_transient": self.teacher_transient,
                "teacher_error": self.teacher_error,
                "no_signal_resamples": self.no_signal_resamples,
                "no_signal_skipped_steps": self.no_signal_skipped_steps,
                "episodes_seen": self.episodes_seen,
                "mt_turn_records": self.mt_turn_records,
                "skip_counts": skip_counts,
                "opd_phase_seconds": dict(self.opd_phase_seconds),
                "opd_phase_counts": dict(self.opd_phase_counts),
                "aligned_sequences": self.aligned_sequences,
                "empty_alignments": self.empty_alignments,
                "coverage_sum": self.coverage_sum,
                "align_group_sum": self.align_group_sum,
                "align_group_n": self.align_group_n,
            }

    def _record_skip(self, reason: str) -> None:
        """Count one single-turn rollout dropped BEFORE the teacher was called.

        Call with ``_stats_lock`` already held. Multi-turn already records its own reason; the
        single-turn gates did not, and that gap cost a paid GPU run: a run that lost every rollout
        reported only "no aligned teacher signal", which names the symptom and not one of the three
        very different causes (cap reached without EOS, empty response, undecodable text). The
        counters live in the same ``skip_counts`` dict the stats snapshot already publishes.
        """
        self.skip_counts[reason] = self.skip_counts.get(reason, 0) + 1

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
            with self._stats_lock:
                self._record_skip("empty_response")
            return self._empty(prompt_length, 0)
        stop_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        if not _rollout_terminated(
            response_ids, stop_text, self.eos_token_ids, self.stop_sequences
        ):
            with self._stats_lock:
                self.truncated_rollouts += 1
                self._record_skip("not_terminated")
            return self._empty(prompt_length, len(response_ids))
        kept_ids, completion_text, kept_forced = _trim_response_and_forced(
            self.tokenizer,
            response_ids,
            stop_text,
            self.stop_sequences,
            forced,
        )
        if not completion_text.strip() or "�" in completion_text:
            with self._stats_lock:
                self._record_skip("undecodable_or_blank")
            return self._empty(prompt_length, len(response_ids))
        try:
            teacher_score = score_rollout(
                prompt,
                completion_text,
                teacher=self.teacher,
                thinking_prefill=self.thinking_prefill,
                text_teacher_batcher=self._text_teacher_batcher,
                on_scored=self.parent_work.complete,
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

    def _env_call(self, method: str, *args, **kwargs):
        with self.parent_work.busy():
            return getattr(self.active_env, method)(*args, **kwargs)

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
        image_count: int,
        image_digests: list[str] | None = None,
    ) -> dict:
        self._require_multiturn()
        if index < 0 or index >= len(self.prompts):
            raise ValueError("flash OPD bridge received an unknown dataset index")
        prompt = self.prompts[index]
        if prompt.example is None:
            raise ValueError("multi-turn OPD prompt is missing its environment example")
        expected_digests = validate_start_media(
            prompt,
            self.processor,
            index,
            image_count,
            image_digests,
        )
        prompt_ids = [int(token_id) for token_id in prompt_ids]
        if prompt_ids != list(prompt.prompt_ids):
            raise ValueError("multi-turn rollout prompt ids do not match the frozen flash prompt")
        raw_prompt = validate_structured_messages(raw_prompt, source="child initial prompt")
        frozen_prompt = validate_structured_messages(
            prompt.student_messages, source="frozen environment prompt"
        )
        if raw_prompt != frozen_prompt:
            raise ValueError("multi-turn child prompt does not match the frozen environment prompt")
        session_id = self._validate_session_id(session_id)
        start_identity = (
            int(index),
            tuple(prompt_ids),
            expected_digests,
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
                state = self._env_call(
                    "new_rollout_state",
                    prompt.example,
                    prepared_prompt=prompt.student_messages,
                )
                # the prepared prompt and its descriptors were frozen together before the child
                # authenticated them. normalized image blocks intentionally carry no source, so an
                # exact prepared state must reuse those frozen descriptors instead of rehydrating the
                # prompt and rerunning environment preparation.
                if state.get("prompt") == prompt.student_messages:
                    initial_messages = validate_structured_messages(
                        state["prompt"], source="environment initial prompt"
                    )
                    fresh_descriptors = tuple(prompt.image_descriptors)
                else:
                    initial_messages, fresh_descriptors = normalize_initial_prompt(
                        prompt,
                        state,
                        self.processor,
                    )
            if initial_messages != frozen_prompt or fresh_descriptors != prompt.image_descriptors:
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
                "messages": copy.deepcopy(initial_messages),
                "descriptors": list(prompt.image_descriptors),
                "image_digests": list(expected_digests),
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
        return validated_multiturn_response(
            payload,
            tokenizer=self.tokenizer,
            eos_token_ids=self.eos_token_ids,
            stop_sequences=self.stop_sequences,
        )

    def step_multiturn(self, payload: dict) -> dict:
        self._require_multiturn()
        session = self._session(payload.get("session_id"))
        turn_ordinal = int(payload.get("turn_ordinal", -1))
        accepted_prefix = [int(token_id) for token_id in payload.get("accepted_prefix", [])]
        image_count, image_digests = step_media_identity(payload)
        raw_response_ids, response_ids, completion_text, skip_reason = (
            self._validated_multiturn_response(payload)
        )
        request_identity = (
            turn_ordinal,
            tuple(accepted_prefix),
            image_count,
            tuple(image_digests),
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
            if image_count != len(session["image_digests"]) or tuple(image_digests) != tuple(
                session["image_digests"]
            ):
                raise ValueError(
                    "multi-turn rollout media does not match the authenticated environment context"
                )
            context_messages = copy.deepcopy(session["messages"])
            context_descriptors = tuple(session["descriptors"])
            context_image_digests = tuple(session["image_digests"])
            prompt = self.prompts[session["index"]]
            state = session["state"]
            terminal = bool(payload.get("truncated")) or bool(skip_reason)
            # an unusable turn is excluded from teacher scoring below (see the `scorable` filter),
            # so showing it to the environment buys nothing and can cost the whole run: an env that
            # parses or validates each action may raise on a truncated or empty action, turning a
            # routine no-signal rollout into a permanent paid failure. the grpo bridge already
            # returns before its own `record_model_turn` on exactly this predicate.
            if not terminal:
                self._env_call("record_model_turn", state, completion_text)
                session["messages"].append({"role": "assistant", "content": completion_text})
            messages: list[dict] = []
            next_prefix = [*accepted_prefix, *response_ids]
            if not terminal:
                # check episode termination BEFORE requesting an environment reply: at the turn
                # limit or when the env already reports done, the extra env_reply both wastes an
                # env call and appends a user turn no model turn will ever answer.
                assistant_turns = turn_ordinal + 1
                turn_limit = session["turn_limit"]
                terminal = assistant_turns >= turn_limit or self._env_call(
                    "rollout_done", state, turn_limit
                )
            image_data_uris: tuple[str, ...] = ()
            if not terminal:
                raw_messages = self._env_call("env_reply", session["messages"], state)
                # the env's reply may itself end the episode. terminal replies remain in env state
                # for task semantics but never enter actor or teacher context.
                terminal = not raw_messages or self._env_call(
                    "rollout_done", state, session["turn_limit"]
                )
                if not terminal:
                    prepared_reply = prepare_environment_reply(
                        raw_messages,
                        normalize_reply=normalize_environment_reply,
                        prompt=prompt,
                        cumulative_descriptors=session["descriptors"],
                        processor=self.processor,
                        tokenizer=self.tokenizer,
                        thinking=self.thinking,
                        response_ids=response_ids,
                    )
                    next_prefix.extend(prepared_reply.glue_ids)
                    messages = prepared_reply.messages
                    image_data_uris = prepared_reply.data_uris
                    session["messages"].extend(copy.deepcopy(messages))
                    session["descriptors"].extend(prepared_reply.descriptors)
                    session["image_digests"].extend(prepared_reply.image_digests)
            step_response = {
                "messages": messages,
                "terminal": bool(terminal),
                "image_data_uris": list(image_data_uris),
                "image_count": len(session["image_digests"]),
                "image_digests": list(session["image_digests"]),
            }
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
                    "image_descriptors": context_descriptors,
                    "image_digests": context_image_digests,
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
                prompt = self.prompts[session["index"]]
                teacher_batches = []
                group_start = 0
                while group_start < len(scorable):
                    uses_images = bool(turns[scorable[group_start]]["image_descriptors"])
                    group_end = group_start + 1
                    while (
                        group_end < len(scorable)
                        and bool(turns[scorable[group_end]]["image_descriptors"]) == uses_images
                    ):
                        group_end += 1
                    positions = scorable[group_start:group_end]
                    if uses_images:
                        items = build_multimodal_score_items(
                            prompt,
                            [
                                (
                                    turns[position]["context_messages"],
                                    turns[position]["completion_text"],
                                    turns[position]["image_descriptors"],
                                )
                                for position in positions
                            ],
                            thinking_prefill=self.thinking_prefill,
                        )
                        scored = score_multimodal_items(
                            self.teacher,
                            items,
                            on_scored=self.parent_work.complete,
                        )
                    else:
                        items = [
                            (
                                _teacher_prompt_text(
                                    turns[position]["context_messages"], self.thinking_prefill
                                ),
                                turns[position]["completion_text"],
                            )
                            for position in positions
                        ]
                        if isinstance(self.teacher, TeacherClient):
                            scored = self.teacher.score_many(
                                items,
                                on_scored=self.parent_work.complete,
                            )
                        else:
                            scored = self.teacher.score_many(items)
                            for _score in scored:
                                self.parent_work.complete()
                    if len(scored) != len(positions):
                        raise RuntimeError(
                            "teacher returned the wrong number of multi-turn OPD scores"
                        )
                    teacher_batches.extend(scored)
                    group_start = group_end
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
            # hand the per-reason skip tally back to the child so the fatal error can NAME the
            # cause. the stats snapshot that also carries these is only built after the child has
            # already raised, so without this a run that loses every rollout dies reporting "no
            # aligned teacher signal" and the artifacts cannot say which gate dropped them.
            #
            # report the DELTA since the last committed step, not `skip_counts` itself. that dict is
            # a lifetime accumulator -- it is never zeroed and line 156 even rehydrates it from
            # resume state -- so returning it raw would let a gate that fired in an earlier step be
            # named as the cause of this one, which is worse than saying nothing.
            skips = {
                reason: count - self._skip_baseline.get(reason, 0)
                for reason, count in self.skip_counts.items()
                if count - self._skip_baseline.get(reason, 0) > 0
            }
            self._skip_baseline = dict(self.skip_counts)
        return {"ok": True, "skip_counts": skips}

    def commit_teacher_cycle(self) -> dict:
        with self._stats_lock:
            self._pending_teacher_transient = None
            self._pending_teacher_success = False
            # a committed step closes the window the next failure reports on. without this the
            # baseline would only move on failures, so skips absorbed by a SUCCESSFUL step would be
            # attributed to whatever step failed next.
            self._skip_baseline = dict(self.skip_counts)
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
                    "transient"
                    if isinstance(error, _worker_perf.RetriableInfraError)
                    else "permanent"
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
                # required like every sibling field: defaulting it to 0 would let a child that
                # never sent the count pass the check for an image-bearing prompt.
                image_count=payload["image_count"],
                image_digests=payload["image_digests"],
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
        if isinstance(error, _RecordedMutationCallbackFailure):
            return error.classification
        retriable = isinstance(error, _worker_perf.RetriableInfraError) or (
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
            flush_wait_s=protocol.TEXT_TEACHER_FLUSH_WAIT_S,
            on_scored=self.parent_work.complete,
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
