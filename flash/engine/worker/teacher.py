"""Managed teacher-broker client for on-policy distillation."""

from __future__ import annotations

import http.client
import io
import json
import math
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, Literal

from flash.engine.worker.teacher_encoding import (
    EncodedTeacherToken,
    TeacherTokenizer,
    load_teacher_tokenizer,
)
from flash.engine.worker.tokenizer_align import TeacherToken
from flash.envs.base import map_bounded
from flash.opd_limits import OPD_TEACHER_SCORING_CONCURRENCY

_MAX_LOGPROB_ROUNDING_ERROR = 1e-6
_BROKER_PROVIDER_TIMEOUT_CEILING_S = 90.0
_DEFAULT_TEACHER_TIMEOUT_S = 105.0


class TeacherError(RuntimeError):
    """A classified managed-teacher failure."""

    def __init__(self, *args, permanent: bool = False) -> None:
        super().__init__(*args)
        self.permanent = permanent


def _remaining_run_wall_seconds() -> float | None:
    raw_deadline = os.environ.get("FLASH_RUN_DEADLINE_AT")
    if raw_deadline is None:
        return None
    try:
        deadline = float(raw_deadline)
    except (TypeError, ValueError):
        raise TeacherError("worker run wall deadline is invalid", permanent=True) from None
    now = time.time()
    if not math.isfinite(deadline) or deadline <= 0 or not math.isfinite(now) or now <= 0:
        raise TeacherError("worker run wall deadline is invalid", permanent=True)
    return max(0.0, deadline - now)


class _ReusableHttpsResponse:
    def __init__(self, transport, connection, response) -> None:
        self._transport = transport
        self._connection = connection
        self._response = response
        self._read_complete = False

    def read(self) -> bytes:
        try:
            body = self._response.read()
        except http.client.IncompleteRead:
            self._transport.discard(self._connection)
            raise
        self._read_complete = True
        if self._response.will_close or self._connection.sock is None:
            self._transport.discard(self._connection)
        return body

    def close(self) -> None:
        self._response.close()
        if not self._read_complete or self._response.will_close or self._connection.sock is None:
            self._transport.discard(self._connection)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> Literal[False]:
        self.close()
        return False


class _ThreadLocalHttpsTransport:
    _stale_errors = (
        http.client.RemoteDisconnected,
        http.client.CannotSendRequest,
        http.client.ResponseNotReady,
        BrokenPipeError,
        ConnectionResetError,
    )

    def __init__(self) -> None:
        self._local = threading.local()

    def discard(self, connection) -> None:
        if getattr(self._local, "connection", None) is connection:
            self._local.connection = None
            self._local.origin = None
        connection.close()

    def _connection(self, parsed, timeout: float):
        origin = (parsed.hostname, parsed.port or 443)
        connection = getattr(self._local, "connection", None)
        if connection is not None and getattr(self._local, "origin", None) != origin:
            self.discard(connection)
            connection = None
        if connection is None:
            connection = http.client.HTTPSConnection(origin[0], origin[1], timeout=timeout)
            self._local.connection = connection
            self._local.origin = origin
        else:
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
        return connection

    def urlopen(self, req: urllib.request.Request, *, timeout: float):
        parsed = urllib.parse.urlsplit(req.full_url)
        if parsed.scheme != "https":
            return urllib.request.urlopen(req, timeout=timeout)
        connection = self._connection(parsed, timeout)
        selector = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        try:
            connection.request(
                req.get_method(),
                selector,
                body=req.data,
                headers=dict(req.header_items()),
            )
            response = connection.getresponse()
        except self._stale_errors as exc:
            self.discard(connection)
            raise urllib.error.URLError(exc) from exc
        except OSError:
            self.discard(connection)
            raise
        except http.client.HTTPException as exc:
            self.discard(connection)
            raise urllib.error.URLError(exc) from exc
        if response.status < 200 or response.status >= 300:
            try:
                body = response.read()
            except http.client.IncompleteRead:
                self.discard(connection)
                raise
            if response.will_close or connection.sock is None:
                self.discard(connection)
            raise urllib.error.HTTPError(
                req.full_url,
                response.status,
                response.reason,
                response.headers,
                io.BytesIO(body),
            )
        return _ReusableHttpsResponse(self, connection, response)


@dataclass(frozen=True)
class TeacherScore:
    """Immutable scored tokens and authoritative provider usage for one request."""

    tokens: tuple[TeacherToken, ...]
    input_tokens: int
    output_tokens: int

    def without_billing(self) -> TeacherScore:
        return replace(self, input_tokens=0, output_tokens=0)


def _permanent(message: str) -> TeacherError:
    return TeacherError(message, permanent=True)


def _integer_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise _permanent(f"teacher response {field} must be a list of nonnegative integers")
    return list(value)


def _realized_logprob(value: Any, *, index: int) -> float:
    if isinstance(value, dict):
        value = value.get("logprob")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _permanent(
            f"teacher response has a nonnumeric realized score at prompt token {index}"
        )
    score = float(value)
    if not math.isfinite(score):
        raise _permanent(f"teacher response has a nonfinite realized score at prompt token {index}")
    if score > _MAX_LOGPROB_ROUNDING_ERROR:
        raise _permanent(f"teacher response has a positive realized score at prompt token {index}")
    return min(score, 0.0)


def _token_keyed_scores(value: Any, token_ids: list[int]) -> list[float | None] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != len(token_ids):
        raise _permanent("teacher response prompt_logprobs length does not match prompt_token_ids")
    scores: list[float | None] = []
    for index, (entry, token_id) in enumerate(zip(value, token_ids, strict=True)):
        if entry is None:
            scores.append(None)
            continue
        if not isinstance(entry, dict):
            raise _permanent(f"teacher response prompt_logprobs entry {index} is not an object")
        realized = entry.get(str(token_id), entry.get(token_id))
        if realized is None:
            raise _permanent(
                f"teacher response prompt_logprobs entry {index} omits realized token id {token_id}"
            )
        scores.append(_realized_logprob(realized, index=index))
    return scores


def _positional_scores(choice: dict[str, Any], token_ids: list[int]) -> list[float | None]:
    logprobs = choice.get("logprobs")
    values = logprobs.get("token_logprobs") if isinstance(logprobs, dict) else None
    if not isinstance(values, list) or len(values) != len(token_ids) + 1:
        raise _permanent(
            "teacher response positional token_logprobs must contain every prompt token and one output token"
        )
    scores: list[float | None] = []
    for index, value in enumerate(values[: len(token_ids)]):
        scores.append(None if value is None else _realized_logprob(value, index=index))
    _realized_logprob(values[-1], index=len(token_ids))
    return scores


def _validated_usage(response: dict[str, Any], input_tokens: int) -> tuple[int, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise _permanent("teacher response is missing usage")
    counts: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _permanent(f"teacher response usage.{field} is invalid")
        counts[field] = value
    if (
        counts["prompt_tokens"] != input_tokens
        or counts["completion_tokens"] != 1
        or counts["total_tokens"] != input_tokens + 1
    ):
        raise _permanent(
            "teacher response usage does not match returned input and output token ids"
        )
    return counts["prompt_tokens"], counts["completion_tokens"]


def _completion_tokens(
    encoded: list[EncodedTeacherToken],
    scores: list[float | None],
    *,
    full: str,
    prompt_length: int,
) -> list[TeacherToken]:
    completion = full[prompt_length:]
    if not completion:
        raise _permanent("teacher scoring requires a nonempty completion")
    output: list[TeacherToken] = []
    coverage: list[tuple[int, int]] = []
    for index, token in enumerate(encoded):
        if token.end <= prompt_length:
            continue
        score = scores[index]
        if score is None:
            raise _permanent(
                f"teacher response has a null realized score for completion token {index}"
            )
        start = max(0, token.start - prompt_length)
        end = min(len(completion), token.end - prompt_length)
        if end <= start:
            continue
        output.append(
            TeacherToken(
                text=full[token.start : token.end],
                logprob=score,
                start=start,
                end=end,
            )
        )
        coverage.append((start, end))
    if not output:
        raise _permanent("teacher response returned no scored completion token")
    cursor = 0
    for start, end in sorted(coverage):
        if start > cursor:
            raise _permanent("teacher response does not exactly cover the completion text")
        cursor = max(cursor, end)
    if cursor != len(completion):
        raise _permanent("teacher response does not exactly cover the completion text")
    return output


def _normalize_response(
    response: dict[str, Any],
    *,
    encoded: list[EncodedTeacherToken],
    full: str,
    prompt_length: int,
) -> TeacherScore:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise _permanent("teacher response must contain exactly one choice")
    choice = choices[0]
    local_ids = [token.token_id for token in encoded]
    remote_ids = _integer_list(choice.get("prompt_token_ids"), field="prompt_token_ids")
    if remote_ids != local_ids:
        raise _permanent(
            "teacher response prompt_token_ids do not match the pinned local tokenizer"
        )
    generated_ids = _integer_list(choice.get("token_ids"), field="token_ids")
    if len(generated_ids) != 1:
        raise _permanent("teacher response must contain exactly one generated token id")
    input_tokens, output_tokens = _validated_usage(response, len(remote_ids))
    scores = _token_keyed_scores(choice.get("prompt_logprobs"), remote_ids)
    if scores is None:
        scores = _positional_scores(choice, remote_ids)
    tokens = _completion_tokens(
        encoded,
        scores,
        full=full,
        prompt_length=prompt_length,
    )
    return TeacherScore(
        tokens=tuple(tokens),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class TeacherClient:
    def __init__(
        self,
        capability: str,
        control_panel_url: str,
        model: str,
        *,
        timeout: float = _DEFAULT_TEACHER_TIMEOUT_S,
        max_retries: int = 4,
        tokenizer: TeacherTokenizer | None = None,
    ) -> None:
        if not capability:
            raise _permanent("no managed teacher capability available on the worker")
        if not control_panel_url:
            raise _permanent("no Flash control-panel URL available on the worker")
        self.capability = capability
        self.base_url = control_panel_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.tokenizer = tokenizer or load_teacher_tokenizer(model)
        self._transport = _ThreadLocalHttpsTransport()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
        request_id = secrets.token_urlsafe(24)
        headers = {
            "Authorization": f"Bearer {self.capability}",
            "Content-Type": "application/json",
            "X-Flash-Teacher-Request-Id": request_id,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            remaining = _remaining_run_wall_seconds()
            if remaining is not None and remaining <= 0:
                raise TeacherError("teacher call exceeded the run wall deadline") from None
            timeout = self.timeout if remaining is None else min(self.timeout, remaining)
            request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
            try:
                with self._transport.urlopen(request, timeout=timeout) as response:
                    value = json.loads(response.read().decode())
                    if not isinstance(value, dict):
                        raise json.JSONDecodeError("response is not an object", "", 0)
                    return value
            except urllib.error.HTTPError as error:
                code = "broker_http_error"
                classification = "permanent"
                try:
                    payload = json.loads(error.read(64 * 1024 + 1).decode("utf-8"))
                    broker_error = payload.get("error") if isinstance(payload, dict) else None
                    if isinstance(broker_error, dict):
                        raw_code = broker_error.get("code")
                        raw_classification = broker_error.get("classification")
                        if isinstance(raw_code, str) and raw_code:
                            code = raw_code
                        if isinstance(raw_classification, str) and raw_classification in {
                            "permanent",
                            "transient",
                        }:
                            classification = raw_classification
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    OSError,
                    http.client.IncompleteRead,
                ):
                    pass
                retryable = classification == "transient"
                broker_failure = TeacherError(
                    f"teacher broker HTTP {error.code} for {request_id} on {path}: {code} "
                    f"({'transient' if retryable else 'permanent'})",
                    permanent=not retryable,
                )
                if not retryable:
                    raise broker_failure from None
                last_error = broker_failure
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = TeacherError(
                    f"teacher broker transport error for {request_id} on {path}: "
                    f"{type(error).__name__}"
                )
            except http.client.IncompleteRead:
                last_error = TeacherError(
                    f"teacher broker response was truncated for {request_id} on {path}"
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                last_error = TeacherError(
                    f"teacher broker returned an unparseable response for {request_id} on {path}"
                )
            if attempt + 1 >= self.max_retries:
                break
            delay = min(2.0 * (2**attempt), 20.0)
            remaining = _remaining_run_wall_seconds()
            if remaining is not None:
                if remaining <= 0:
                    raise TeacherError("teacher call exceeded the run wall deadline") from None
                delay = min(delay, remaining)
            if delay > 0:
                time.sleep(delay)
        raise last_error or TeacherError(f"teacher broker call to {path} failed")

    def _score_one(self, prompt_text: str, completion_text: str) -> TeacherScore:
        full = prompt_text + completion_text
        encoded = self.tokenizer.encode(full)
        response = self._post(
            "/v1/teacher/completions",
            {
                "model": self.model,
                "prompt": full,
                "max_tokens": 1,
                "echo": True,
                "logprobs": 1,
                "prompt_logprobs": 1,
                "return_token_ids": True,
                "temperature": 0,
                "top_p": 1,
                "seed": 0,
            },
        )
        return _normalize_response(
            response,
            encoded=encoded,
            full=full,
            prompt_length=len(prompt_text),
        )

    def score_many(self, items: list[tuple[str, str]]) -> list[TeacherScore]:
        """Score each unique prompt and completion through one idempotent broker request.

        Bounded by `map_bounded`, which consumes completions as they arrive rather than in input
        order. That matters here because these are PAID requests: on a failure the caller retries
        the whole OPD attempt, so anything submitted after the first raise is billed for a result
        nobody reads. `executor.map` bounds that to about one pool width only when requests finish
        roughly in order -- with one slow leading request it bills the ENTIRE list (measured 64,
        128, 256 of 256 at width 32, against 37 here). Teacher latency varies with completion
        length, so that is the normal case.
        """
        if not items:
            return []
        return map_bounded(
            items,
            lambda item: self._score_one(*item),
            cap=OPD_TEACHER_SCORING_CONCURRENCY,
        )

    def score(self, prompt_text: str, completion_text: str) -> TeacherScore:
        return self._score_one(prompt_text, completion_text)
