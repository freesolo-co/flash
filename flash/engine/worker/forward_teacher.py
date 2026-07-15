"""Managed ForwardTeacher chat-generation client for hybrid OPD targets."""

from __future__ import annotations

import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from threading import Lock

from flash.engine.recipe import FORWARD_TEACHER_MODEL_ID, FORWARD_TEACHER_URL


class ForwardTeacherError(RuntimeError):
    """A sanitized permanent ForwardTeacher request or response failure."""

    retriable = False

    def __init__(
        self,
        reason: str,
        *,
        attempts: int = 0,
        latency_seconds: float = 0.0,
        ambiguous_paid_requests: int = 0,
        provider_generations: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        super().__init__(reason)
        self.attempts = max(0, int(attempts))
        latency = float(latency_seconds)
        self.latency_seconds = latency if math.isfinite(latency) and latency >= 0 else 0.0
        self.ambiguous_paid_requests = max(0, int(ambiguous_paid_requests))
        self.provider_generations = max(0, int(provider_generations))
        self.prompt_tokens = max(0, int(prompt_tokens))
        self.completion_tokens = max(0, int(completion_tokens))


class ForwardTeacherTransientError(ForwardTeacherError):
    """A sanitized ForwardTeacher failure exhausted after bounded retries."""

    retriable = True


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _remaining_run_wall_seconds() -> float | None:
    raw_deadline = os.environ.get("FLASH_RUN_DEADLINE_AT")
    if raw_deadline is None:
        return None
    try:
        deadline = float(raw_deadline)
    except (TypeError, ValueError):
        raise ForwardTeacherError("forward_teacher run wall deadline is invalid") from None
    now = time.time()
    if not math.isfinite(deadline) or deadline <= 0 or not math.isfinite(now) or now <= 0:
        raise ForwardTeacherError("forward_teacher run wall deadline is invalid")
    return max(0.0, deadline - now)


@dataclass(frozen=True)
class ForwardTeacherTopLogprob:
    token: str = field(repr=False)
    logprob: float


@dataclass(frozen=True)
class ForwardTeacherContentLogprob:
    token: str = field(repr=False)
    logprob: float
    top_logprobs: tuple[ForwardTeacherTopLogprob, ...] = field(repr=False)


FORWARD_TEACHER_THINK_BOUNDARY = "</think>"
FORWARD_TEACHER_TERMINAL = (
    "<" + chr(0xFF5C) + "end" + chr(0x2581) + "of" + chr(0x2581) + "sentence" + chr(0xFF5C) + ">"
)
FORWARD_TEACHER_TOP_LOGPROBS = 20
FORWARD_TEACHER_REALIZED_LOGPROB_ABS_TOLERANCE = 1e-6


class ForwardTeacherRecordKind(StrEnum):
    HIDDEN_REASONING = "hidden_reasoning"
    THINK_BOUNDARY = "think_boundary"
    VISIBLE_CONTENT = "visible_content"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class ForwardTeacherSemanticRecord:
    kind: ForwardTeacherRecordKind
    index: int
    token: str = field(repr=False)
    logprob: float
    top_logprobs: tuple[ForwardTeacherTopLogprob, ...] = field(repr=False)


@dataclass(frozen=True)
class ForwardTeacherParsedCompletion:
    records: tuple[ForwardTeacherSemanticRecord, ...]
    hidden_reasoning_records: tuple[ForwardTeacherSemanticRecord, ...]
    boundary_record: ForwardTeacherSemanticRecord
    visible_content_records: tuple[ForwardTeacherSemanticRecord, ...]
    terminal_record: ForwardTeacherSemanticRecord


@dataclass(frozen=True)
class ForwardTeacherResult:
    content: str = field(repr=False)
    reasoning: str = field(repr=False)
    finish_reason: str
    content_logprobs: tuple[ForwardTeacherContentLogprob, ...] = field(repr=False)
    parsed_completion: ForwardTeacherParsedCompletion = field(repr=False)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    attempts: int
    ambiguous_paid_requests: int = 0
    provider_generations: int = 1


def _parse_semantic_completion(
    records: tuple[ForwardTeacherContentLogprob, ...],
    *,
    reasoning: str,
    content: str,
) -> ForwardTeacherParsedCompletion:
    boundary_indices = [
        index for index, record in enumerate(records) if record.token == FORWARD_TEACHER_THINK_BOUNDARY
    ]
    terminal_indices = [
        index for index, record in enumerate(records) if record.token == FORWARD_TEACHER_TERMINAL
    ]
    if not terminal_indices:
        raise ForwardTeacherError(
            "forward_teacher semantic completion terminal count is invalid: count=0"
        )
    terminal_index = len(records) - 1
    if records[terminal_index].token != FORWARD_TEACHER_TERMINAL:
        raise ForwardTeacherError("forward_teacher semantic completion terminal position is invalid")

    matching_boundaries = [
        index
        for index in boundary_indices
        if index < terminal_index - 1
        and "".join(record.token for record in records[:index]) == reasoning
        and "".join(record.token for record in records[index + 1 : terminal_index]) == content
    ]
    if len(matching_boundaries) != 1:
        if len(boundary_indices) == 1:
            candidate = boundary_indices[0]
            if candidate >= terminal_index - 1:
                raise ForwardTeacherError(
                    "forward_teacher semantic completion visible interval is empty"
                )
            if "".join(record.token for record in records[:candidate]) != reasoning:
                raise ForwardTeacherError(
                    "forward_teacher semantic completion reasoning reconstruction mismatch"
                )
            raise ForwardTeacherError(
                "forward_teacher semantic completion content reconstruction mismatch"
            )
        raise ForwardTeacherError(
            f"forward_teacher semantic completion boundary count is invalid: count={len(matching_boundaries)}"
        )
    boundary_index = matching_boundaries[0]

    semantic_records = []
    for index, record in enumerate(records):
        if index < boundary_index:
            kind = ForwardTeacherRecordKind.HIDDEN_REASONING
        elif index == boundary_index:
            kind = ForwardTeacherRecordKind.THINK_BOUNDARY
        elif index < terminal_index:
            kind = ForwardTeacherRecordKind.VISIBLE_CONTENT
        else:
            kind = ForwardTeacherRecordKind.TERMINAL
        semantic_records.append(
            ForwardTeacherSemanticRecord(
                kind=kind,
                index=index,
                token=record.token,
                logprob=record.logprob,
                top_logprobs=record.top_logprobs,
            )
        )
    frozen = tuple(semantic_records)
    return ForwardTeacherParsedCompletion(
        records=frozen,
        hidden_reasoning_records=frozen[:boundary_index],
        boundary_record=frozen[boundary_index],
        visible_content_records=frozen[boundary_index + 1 : terminal_index],
        terminal_record=frozen[terminal_index],
    )


class ForwardTeacherClient:
    def __init__(
        self,
        api_key: str,
        *,
        seed: int,
        timeout: float = 90.0,
        opener: Callable | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ForwardTeacherError("forward_teacher credential is unavailable")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
            raise ForwardTeacherError("forward_teacher seed is invalid")
        try:
            timeout = float(timeout)
        except (TypeError, ValueError, OverflowError):
            raise ForwardTeacherError("forward_teacher timeout is invalid") from None
        if not math.isfinite(timeout) or timeout <= 0:
            raise ForwardTeacherError("forward_teacher timeout is invalid")
        self._api_key = api_key
        self._seed = seed
        self._timeout = timeout
        self._opener = opener or urllib.request.build_opener(_RejectRedirects()).open
        self._sleep = sleep
        self._clock = clock
        self._start_lock = Lock()
        self._last_request_started: float | None = None

    def _pace_attempt(self, *, max_delay: float | None = None) -> float:
        with self._start_lock:
            started = self._clock()
            now = started
            if self._last_request_started is not None:
                delay = self._last_request_started + 2.0 - now
                if delay > 0:
                    if max_delay is not None:
                        delay = min(delay, max_delay)
                    self._sleep(delay)
                    now = self._clock()
            self._last_request_started = now
            return started

    @staticmethod
    def _retry_after(headers) -> float:
        value = headers.get("Retry-After") if headers is not None else None
        try:
            delay = float(value)
        except (TypeError, ValueError):
            delay = 10.0
        if not math.isfinite(delay):
            delay = 10.0
        return min(60.0, max(1.0, delay))

    @staticmethod
    def _validated_usage(payload: object) -> tuple[int, int, int]:
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            raise ForwardTeacherError("forward_teacher response usage is malformed")
        values = []
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ForwardTeacherError("forward_teacher response usage is malformed")
            values.append(value)
        prompt_tokens, completion_tokens, total_tokens = values
        if completion_tokens <= 0 or prompt_tokens + completion_tokens != total_tokens:
            raise ForwardTeacherError("forward_teacher response usage arithmetic is invalid")
        return prompt_tokens, completion_tokens, total_tokens

    @staticmethod
    def _validate(
        payload: object,
        attempts: int,
        latency: float,
        ambiguous_paid_requests: int = 0,
    ) -> ForwardTeacherResult:
        if not isinstance(payload, dict):
            raise ForwardTeacherError("forward_teacher response is malformed")
        if payload.get("model") != FORWARD_TEACHER_MODEL_ID:
            raise ForwardTeacherError("forward_teacher response model does not match the pinned model")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ForwardTeacherError("forward_teacher response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index") != 0:
            raise ForwardTeacherError("forward_teacher response choice index is invalid")
        finish_reason = choice.get("finish_reason")
        if finish_reason == "insufficient_system_resource":
            raise ForwardTeacherTransientError(
                "forward_teacher provider reported insufficient system resources"
            )
        if finish_reason != "stop":
            raise ForwardTeacherError("forward_teacher response did not finish naturally")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        reasoning = None
        if isinstance(message, dict):
            reasoning = message.get("reasoning_content")
            if reasoning is None:
                reasoning = message.get("reasoning")
        if not isinstance(content, str) or not content.strip():
            raise ForwardTeacherError("forward_teacher response visible content is empty")
        if not isinstance(reasoning, str):
            raise ForwardTeacherError("forward_teacher response reasoning is malformed")
        logprobs = choice.get("logprobs")
        content_records = logprobs.get("content") if isinstance(logprobs, dict) else None
        if not isinstance(content_records, list) or not content_records:
            raise ForwardTeacherError("forward_teacher response content logprobs are malformed")
        validated_records = []
        for record in content_records:
            if not isinstance(record, dict):
                raise ForwardTeacherError("forward_teacher response content logprobs are malformed")
            token = record.get("token")
            logprob = record.get("logprob")
            top_logprobs = record.get("top_logprobs")
            if (
                not isinstance(token, str)
                or not token
                or isinstance(logprob, bool)
                or not isinstance(logprob, (int, float))
                or not math.isfinite(float(logprob))
                or float(logprob) > 0
                or not isinstance(top_logprobs, list)
                or not top_logprobs
                or len(top_logprobs) > FORWARD_TEACHER_TOP_LOGPROBS
            ):
                raise ForwardTeacherError("forward_teacher response content logprobs are malformed")
            validated_top = []
            for candidate in top_logprobs:
                if not isinstance(candidate, dict):
                    raise ForwardTeacherError("forward_teacher response top logprobs are malformed")
                candidate_token = candidate.get("token")
                candidate_logprob = candidate.get("logprob")
                if (
                    not isinstance(candidate_token, str)
                    or isinstance(candidate_logprob, bool)
                    or not isinstance(candidate_logprob, (int, float))
                    or not math.isfinite(float(candidate_logprob))
                    or float(candidate_logprob) > 0
                ):
                    raise ForwardTeacherError("forward_teacher response top logprobs are malformed")
                validated_top.append(
                    ForwardTeacherTopLogprob(
                        token=candidate_token,
                        logprob=float(candidate_logprob),
                    )
                )
            realized_candidates = [
                candidate for candidate in validated_top if candidate.token == token
            ]
            if not realized_candidates:
                raise ForwardTeacherError("forward_teacher response realized token top-logprob count is invalid")
            if not any(
                math.isclose(
                    candidate.logprob,
                    float(logprob),
                    rel_tol=0.0,
                    abs_tol=FORWARD_TEACHER_REALIZED_LOGPROB_ABS_TOLERANCE,
                )
                for candidate in realized_candidates
            ):
                raise ForwardTeacherError("forward_teacher response realized token logprob does not match")
            validated_records.append(
                ForwardTeacherContentLogprob(
                    token=token,
                    logprob=float(logprob),
                    top_logprobs=tuple(validated_top),
                )
            )
        prompt_tokens, completion_tokens, total_tokens = ForwardTeacherClient._validated_usage(
            payload
        )
        # the pinned route counts the full completion, including separately surfaced
        # reasoning, with one logprob record per completion token in 211/211 probes.
        if len(validated_records) != completion_tokens:
            raise ForwardTeacherError("forward_teacher response completion token count is inconsistent")
        frozen_records = tuple(validated_records)
        parsed_completion = _parse_semantic_completion(
            frozen_records,
            reasoning=reasoning,
            content=content,
        )
        return ForwardTeacherResult(
            content=content,
            reasoning=reasoning,
            finish_reason="stop",
            content_logprobs=frozen_records,
            parsed_completion=parsed_completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=float(latency),
            attempts=int(attempts),
            ambiguous_paid_requests=max(0, int(ambiguous_paid_requests)),
        )

    def generate(self, messages: object) -> ForwardTeacherResult:
        sanitized: tuple[type[ForwardTeacherError], str, int, float, int, int, int, int] | None = None
        try:
            return self._generate(messages)
        except ForwardTeacherError as exc:
            error_type = ForwardTeacherTransientError if exc.retriable else ForwardTeacherError
            sanitized = (
                error_type,
                str(exc),
                exc.attempts,
                exc.latency_seconds,
                exc.ambiguous_paid_requests,
                exc.provider_generations,
                exc.prompt_tokens,
                exc.completion_tokens,
            )
        (
            error_type,
            reason,
            attempts,
            latency_seconds,
            ambiguous_paid_requests,
            provider_generations,
            prompt_tokens,
            completion_tokens,
        ) = sanitized
        raise error_type(
            reason,
            attempts=attempts,
            latency_seconds=latency_seconds,
            ambiguous_paid_requests=ambiguous_paid_requests,
            provider_generations=provider_generations,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ) from None

    def _generate(self, messages: object) -> ForwardTeacherResult:
        if not isinstance(messages, list) or not messages:
            raise ForwardTeacherError("forward_teacher messages are invalid")
        for message in messages:
            if not isinstance(message, dict):
                raise ForwardTeacherError("forward_teacher messages are invalid")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role.strip() or not isinstance(content, str):
                raise ForwardTeacherError("forward_teacher messages are invalid")
            if any(not isinstance(key, str) for key in message):
                raise ForwardTeacherError("forward_teacher messages are invalid")
        try:
            body = json.dumps(
                {
                    "model": FORWARD_TEACHER_MODEL_ID,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.2,
                    "seed": self._seed,
                    "logprobs": True,
                    "top_logprobs": FORWARD_TEACHER_TOP_LOGPROBS,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            raise ForwardTeacherError("forward_teacher messages are invalid") from None
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        started = None
        ambiguous_paid_requests = 0
        provider_generations = 0
        prompt_tokens = 0
        completion_tokens = 0

        def deadline_failure(attempts: int) -> ForwardTeacherTransientError:
            latency = self._clock() - started if started is not None else 0.0
            return ForwardTeacherTransientError(
                "forward_teacher request exceeded the run wall deadline",
                attempts=attempts,
                latency_seconds=latency,
                ambiguous_paid_requests=ambiguous_paid_requests,
                provider_generations=provider_generations,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        def remaining_or_fail(attempts: int) -> float | None:
            remaining = _remaining_run_wall_seconds()
            if remaining is not None and remaining <= 0:
                raise deadline_failure(attempts)
            return remaining

        def retry_sleep(delay: float, attempts: int) -> None:
            remaining = remaining_or_fail(attempts)
            if remaining is not None:
                delay = min(delay, remaining)
            if delay > 0:
                self._sleep(delay)

        def failure(
            error_type,
            reason: str,
            *,
            attempts: int,
            ambiguous_increment: int,
            latency_seconds: float | None = None,
        ):
            return error_type(
                reason,
                attempts=attempts,
                latency_seconds=(
                    self._clock() - started if latency_seconds is None else latency_seconds
                ),
                ambiguous_paid_requests=(
                    ambiguous_paid_requests + max(0, int(ambiguous_increment))
                ),
                provider_generations=provider_generations,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        for attempt in (1, 2):
            remaining = remaining_or_fail(attempt - 1)
            attempt_started = self._pace_attempt(max_delay=remaining)
            if started is None:
                started = attempt_started
            remaining = remaining_or_fail(attempt - 1)
            request_timeout = self._timeout if remaining is None else min(self._timeout, remaining)
            request = urllib.request.Request(
                FORWARD_TEACHER_URL, data=body, headers=headers, method="POST"
            )

            try:
                with self._opener(request, timeout=request_timeout) as response:
                    raw = response.read()
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    ambiguous_paid_requests += 1
                    if attempt == 2:
                        raise failure(
                            ForwardTeacherTransientError,
                            "forward_teacher response decoding failure",
                            attempts=attempt,
                            ambiguous_increment=0,
                        ) from None
                    retry_sleep(10.0, attempt)
                    continue
                latency = self._clock() - started
                try:
                    result = self._validate(
                        payload,
                        attempt,
                        latency,
                        ambiguous_paid_requests,
                    )
                except ForwardTeacherError as exc:
                    try:
                        usage = self._validated_usage(payload)
                    except ForwardTeacherError:
                        usage = None
                    if usage is not None:
                        provider_generations += 1
                        prompt_tokens += usage[0]
                        completion_tokens += usage[1]
                    if exc.retriable:
                        if attempt == 2:
                            raise failure(
                                ForwardTeacherTransientError,
                                str(exc),
                                attempts=attempt,
                                ambiguous_increment=0,
                                latency_seconds=latency,
                            ) from None
                        retry_sleep(10.0, attempt)
                        continue
                    raise failure(
                        ForwardTeacherError,
                        str(exc),
                        attempts=attempt,
                        ambiguous_increment=0,
                        latency_seconds=latency,
                    ) from None
                return replace(
                    result,
                    prompt_tokens=prompt_tokens + result.prompt_tokens,
                    completion_tokens=completion_tokens + result.completion_tokens,
                    total_tokens=(
                        prompt_tokens
                        + completion_tokens
                        + result.prompt_tokens
                        + result.completion_tokens
                    ),
                    provider_generations=provider_generations + 1,
                )
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (408, 409, 425, 429) or 500 <= exc.code <= 599
                if not retryable:
                    raise failure(
                        ForwardTeacherError,
                        f"forward_teacher HTTP {exc.code}",
                        attempts=attempt,
                        ambiguous_increment=0,
                    ) from None
                if attempt == 2:
                    raise failure(
                        ForwardTeacherTransientError,
                        f"forward_teacher HTTP {exc.code}",
                        attempts=attempt,
                        ambiguous_increment=0,
                    ) from None
                retry_sleep(self._retry_after(exc.headers), attempt)
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
                if attempt == 2:
                    raise failure(
                        ForwardTeacherTransientError,
                        "forward_teacher transport failure",
                        attempts=attempt,
                        ambiguous_increment=1,
                    ) from None
                ambiguous_paid_requests += 1
                retry_sleep(10.0, attempt)
            except (TypeError, ValueError):
                raise failure(
                    ForwardTeacherError,
                    "forward_teacher response is malformed",
                    attempts=attempt,
                    ambiguous_increment=0,
                ) from None
        raise ForwardTeacherTransientError(
            "forward_teacher request attempts exhausted",
            attempts=2,
            latency_seconds=(self._clock() - started) if started is not None else 0.0,
            ambiguous_paid_requests=ambiguous_paid_requests,
            provider_generations=provider_generations,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
