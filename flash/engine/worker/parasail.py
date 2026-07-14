"""Managed Parasail chat-generation client for hybrid OPD targets."""

from __future__ import annotations

import http.client
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock

from flash.engine.recipe import PARASAIL_MODEL_ID, PARASAIL_URL


class ParasailError(RuntimeError):
    """A sanitized permanent Parasail request or response failure."""

    retriable = False

    def __init__(
        self,
        reason: str,
        *,
        attempts: int = 0,
        latency_seconds: float = 0.0,
        ambiguous_paid_requests: int = 0,
    ) -> None:
        super().__init__(reason)
        self.attempts = max(0, int(attempts))
        latency = float(latency_seconds)
        self.latency_seconds = latency if math.isfinite(latency) and latency >= 0 else 0.0
        self.ambiguous_paid_requests = max(0, int(ambiguous_paid_requests))


class ParasailTransientError(ParasailError):
    """A sanitized Parasail failure exhausted after bounded retries."""

    retriable = True


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class ParasailTopLogprob:
    token: str = field(repr=False)
    logprob: float


@dataclass(frozen=True)
class ParasailContentLogprob:
    token: str = field(repr=False)
    logprob: float
    top_logprobs: tuple[ParasailTopLogprob, ...] = field(repr=False)


PARASAIL_THINK_BOUNDARY = "</think>"
PARASAIL_TERMINAL = (
    "<" + chr(0xFF5C) + "end" + chr(0x2581) + "of" + chr(0x2581) + "sentence" + chr(0xFF5C) + ">"
)
PARASAIL_TOP_LOGPROBS = 20
PARASAIL_REALIZED_LOGPROB_ABS_TOLERANCE = 1e-6


class ParasailRecordKind(StrEnum):
    HIDDEN_REASONING = "hidden_reasoning"
    THINK_BOUNDARY = "think_boundary"
    VISIBLE_CONTENT = "visible_content"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class ParasailSemanticRecord:
    kind: ParasailRecordKind
    index: int
    token: str = field(repr=False)
    logprob: float
    top_logprobs: tuple[ParasailTopLogprob, ...] = field(repr=False)


@dataclass(frozen=True)
class ParasailParsedCompletion:
    records: tuple[ParasailSemanticRecord, ...]
    hidden_reasoning_records: tuple[ParasailSemanticRecord, ...]
    boundary_record: ParasailSemanticRecord
    visible_content_records: tuple[ParasailSemanticRecord, ...]
    terminal_record: ParasailSemanticRecord


@dataclass(frozen=True)
class ParasailResult:
    content: str = field(repr=False)
    reasoning: str = field(repr=False)
    finish_reason: str
    content_logprobs: tuple[ParasailContentLogprob, ...] = field(repr=False)
    parsed_completion: ParasailParsedCompletion = field(repr=False)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    attempts: int
    ambiguous_paid_requests: int = 0


def _parse_semantic_completion(
    records: tuple[ParasailContentLogprob, ...],
    *,
    reasoning: str,
    content: str,
) -> ParasailParsedCompletion:
    boundary_indices = [
        index for index, record in enumerate(records) if record.token == PARASAIL_THINK_BOUNDARY
    ]
    terminal_indices = [
        index for index, record in enumerate(records) if record.token == PARASAIL_TERMINAL
    ]
    if len(boundary_indices) != 1:
        raise ParasailError(
            f"parasail semantic completion boundary count is invalid: count={len(boundary_indices)}"
        )
    if len(terminal_indices) != 1:
        raise ParasailError(
            f"parasail semantic completion terminal count is invalid: count={len(terminal_indices)}"
        )
    boundary_index = boundary_indices[0]
    terminal_index = terminal_indices[0]
    if terminal_index != len(records) - 1:
        raise ParasailError("parasail semantic completion terminal position is invalid")
    if boundary_index >= terminal_index - 1:
        raise ParasailError("parasail semantic completion visible interval is empty")

    hidden = records[:boundary_index]
    visible = records[boundary_index + 1 : terminal_index]
    if any(
        record.token in (PARASAIL_THINK_BOUNDARY, PARASAIL_TERMINAL)
        for record in (*hidden, *visible)
    ):
        raise ParasailError("parasail semantic completion record ordering is invalid")
    if "".join(record.token for record in hidden) != reasoning:
        raise ParasailError("parasail semantic completion reasoning reconstruction mismatch")
    if "".join(record.token for record in visible) != content:
        raise ParasailError("parasail semantic completion content reconstruction mismatch")

    semantic_records = []
    for index, record in enumerate(records):
        if index < boundary_index:
            kind = ParasailRecordKind.HIDDEN_REASONING
        elif index == boundary_index:
            kind = ParasailRecordKind.THINK_BOUNDARY
        elif index < terminal_index:
            kind = ParasailRecordKind.VISIBLE_CONTENT
        else:
            kind = ParasailRecordKind.TERMINAL
        semantic_records.append(
            ParasailSemanticRecord(
                kind=kind,
                index=index,
                token=record.token,
                logprob=record.logprob,
                top_logprobs=record.top_logprobs,
            )
        )
    frozen = tuple(semantic_records)
    return ParasailParsedCompletion(
        records=frozen,
        hidden_reasoning_records=frozen[:boundary_index],
        boundary_record=frozen[boundary_index],
        visible_content_records=frozen[boundary_index + 1 : terminal_index],
        terminal_record=frozen[terminal_index],
    )


class ParasailClient:
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
            raise ParasailError("parasail credential is unavailable")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
            raise ParasailError("parasail seed is invalid")
        try:
            timeout = float(timeout)
        except (TypeError, ValueError, OverflowError):
            raise ParasailError("parasail timeout is invalid") from None
        if not math.isfinite(timeout) or timeout <= 0:
            raise ParasailError("parasail timeout is invalid")
        self._api_key = api_key
        self._seed = seed
        self._timeout = timeout
        self._opener = opener or urllib.request.build_opener(_RejectRedirects()).open
        self._sleep = sleep
        self._clock = clock
        self._start_lock = Lock()
        self._last_request_started: float | None = None

    def _pace_attempt(self) -> float:
        with self._start_lock:
            now = self._clock()
            if self._last_request_started is not None:
                delay = self._last_request_started + 2.0 - now
                if delay > 0:
                    self._sleep(delay)
                    now = self._clock()
            self._last_request_started = now
            return now

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
    def _validate(
        payload: object,
        attempts: int,
        latency: float,
        ambiguous_paid_requests: int = 0,
    ) -> ParasailResult:
        if not isinstance(payload, dict):
            raise ParasailError("parasail response is malformed")
        if payload.get("model") != PARASAIL_MODEL_ID:
            raise ParasailError("parasail response model does not match the pinned model")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ParasailError("parasail response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index") != 0:
            raise ParasailError("parasail response choice index is invalid")
        if choice.get("finish_reason") != "stop":
            raise ParasailError("parasail response did not finish naturally")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        reasoning = message.get("reasoning") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ParasailError("parasail response visible content is empty")
        if not isinstance(reasoning, str):
            raise ParasailError("parasail response reasoning is malformed")
        logprobs = choice.get("logprobs")
        content_records = logprobs.get("content") if isinstance(logprobs, dict) else None
        if not isinstance(content_records, list) or not content_records:
            raise ParasailError("parasail response content logprobs are malformed")
        validated_records = []
        for record in content_records:
            if not isinstance(record, dict):
                raise ParasailError("parasail response content logprobs are malformed")
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
                or len(top_logprobs) > PARASAIL_TOP_LOGPROBS
            ):
                raise ParasailError("parasail response content logprobs are malformed")
            validated_top = []
            seen_candidate_tokens: set[str] = set()
            for candidate in top_logprobs:
                if not isinstance(candidate, dict):
                    raise ParasailError("parasail response top logprobs are malformed")
                candidate_token = candidate.get("token")
                candidate_logprob = candidate.get("logprob")
                if (
                    not isinstance(candidate_token, str)
                    or isinstance(candidate_logprob, bool)
                    or not isinstance(candidate_logprob, (int, float))
                    or not math.isfinite(float(candidate_logprob))
                    or float(candidate_logprob) > 0
                ):
                    raise ParasailError("parasail response top logprobs are malformed")
                if candidate_token in seen_candidate_tokens:
                    raise ParasailError(
                        "parasail response top logprobs contain duplicate token strings"
                    )
                seen_candidate_tokens.add(candidate_token)
                validated_top.append(
                    ParasailTopLogprob(
                        token=candidate_token,
                        logprob=float(candidate_logprob),
                    )
                )
            realized_candidates = [
                candidate for candidate in validated_top if candidate.token == token
            ]
            if len(realized_candidates) != 1:
                raise ParasailError("parasail response realized token top-logprob count is invalid")
            if not math.isclose(
                realized_candidates[0].logprob,
                float(logprob),
                rel_tol=0.0,
                abs_tol=PARASAIL_REALIZED_LOGPROB_ABS_TOLERANCE,
            ):
                raise ParasailError("parasail response realized token logprob does not match")
            validated_records.append(
                ParasailContentLogprob(
                    token=token,
                    logprob=float(logprob),
                    top_logprobs=tuple(validated_top),
                )
            )
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise ParasailError("parasail response usage is malformed")
        values = []
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ParasailError("parasail response usage is malformed")
            values.append(value)
        prompt_tokens, completion_tokens, total_tokens = values
        if completion_tokens <= 0 or prompt_tokens + completion_tokens != total_tokens:
            raise ParasailError("parasail response usage arithmetic is invalid")
        # the pinned route counts the full completion, including separately surfaced
        # reasoning, with one logprob record per completion token in 211/211 probes.
        if len(validated_records) != completion_tokens:
            raise ParasailError("parasail response completion token count is inconsistent")
        frozen_records = tuple(validated_records)
        parsed_completion = _parse_semantic_completion(
            frozen_records,
            reasoning=reasoning,
            content=content,
        )
        return ParasailResult(
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

    def generate(self, messages: object) -> ParasailResult:
        sanitized: tuple[type[ParasailError], str, int, float, int] | None = None
        try:
            return self._generate(messages)
        except ParasailError as exc:
            error_type = ParasailTransientError if exc.retriable else ParasailError
            sanitized = (
                error_type,
                str(exc),
                exc.attempts,
                exc.latency_seconds,
                exc.ambiguous_paid_requests,
            )
        error_type, reason, attempts, latency_seconds, ambiguous_paid_requests = sanitized
        raise error_type(
            reason,
            attempts=attempts,
            latency_seconds=latency_seconds,
            ambiguous_paid_requests=ambiguous_paid_requests,
        ) from None

    def _generate(self, messages: object) -> ParasailResult:
        if not isinstance(messages, list) or not messages:
            raise ParasailError("parasail messages are invalid")
        for message in messages:
            if not isinstance(message, dict):
                raise ParasailError("parasail messages are invalid")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role.strip() or not isinstance(content, str):
                raise ParasailError("parasail messages are invalid")
            if any(not isinstance(key, str) for key in message):
                raise ParasailError("parasail messages are invalid")
        try:
            body = json.dumps(
                {
                    "model": PARASAIL_MODEL_ID,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.2,
                    "seed": self._seed,
                    "logprobs": True,
                    "top_logprobs": PARASAIL_TOP_LOGPROBS,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            raise ParasailError("parasail messages are invalid") from None
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        started = None
        ambiguous_paid_requests = 0
        for attempt in (1, 2):
            attempt_started = self._pace_attempt()
            if started is None:
                started = attempt_started
            request = urllib.request.Request(
                PARASAIL_URL, data=body, headers=headers, method="POST"
            )

            def failure(
                error_type,
                reason: str,
                *,
                ambiguous_increment: int,
                _attempt=attempt,
                _started=started,
                _ambiguous_paid_requests=ambiguous_paid_requests,
            ):
                return error_type(
                    reason,
                    attempts=_attempt,
                    latency_seconds=self._clock() - _started,
                    ambiguous_paid_requests=(
                        _ambiguous_paid_requests + max(0, int(ambiguous_increment))
                    ),
                )

            try:
                with self._opener(request, timeout=self._timeout) as response:
                    raw = response.read()
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if attempt == 2:
                        raise failure(
                            ParasailTransientError,
                            "parasail response decoding failure",
                            ambiguous_increment=1,
                        ) from None
                    ambiguous_paid_requests += 1
                    self._sleep(10.0)
                    continue
                latency = self._clock() - started
                try:
                    return self._validate(
                        payload,
                        attempt,
                        latency,
                        ambiguous_paid_requests,
                    )
                except ParasailError as exc:
                    raise ParasailError(
                        str(exc),
                        attempts=attempt,
                        latency_seconds=latency,
                        ambiguous_paid_requests=ambiguous_paid_requests + 1,
                    ) from None
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (408, 409, 425, 429) or 500 <= exc.code <= 599
                if not retryable:
                    raise failure(
                        ParasailError,
                        f"parasail HTTP {exc.code}",
                        ambiguous_increment=0,
                    ) from None
                if attempt == 2:
                    raise failure(
                        ParasailTransientError,
                        f"parasail HTTP {exc.code}",
                        ambiguous_increment=1,
                    ) from None
                ambiguous_paid_requests += 1
                self._sleep(self._retry_after(exc.headers))
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
                if attempt == 2:
                    raise failure(
                        ParasailTransientError,
                        "parasail transport failure",
                        ambiguous_increment=1,
                    ) from None
                ambiguous_paid_requests += 1
                self._sleep(10.0)
            except (TypeError, ValueError):
                raise failure(
                    ParasailError,
                    "parasail response is malformed",
                    ambiguous_increment=1,
                ) from None
        raise ParasailTransientError(
            "parasail request attempts exhausted",
            attempts=2,
            latency_seconds=(self._clock() - started) if started is not None else 0.0,
            ambiguous_paid_requests=max(1, ambiguous_paid_requests),
        )
