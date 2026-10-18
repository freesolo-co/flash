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
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from flash.engine.worker.teacher.encoding import (
    EncodedTeacherToken,
    TeacherTokenizer,
    load_teacher_tokenizer,
)
from flash.engine.worker.teacher.tokenizer_align import TeacherToken
from flash.envs.loading.base import map_bounded
from flash.teacher.limits import OPD_TEACHER_SCORING_CONCURRENCY
from flash.teacher.provider_status import (
    BODY_INDEPENDENT_TRANSIENT_STATUSES,
    validated_provider_status,
)

_MAX_LOGPROB_ROUNDING_ERROR = 1e-6
_BROKER_PROVIDER_TIMEOUT_CEILING_S = 90.0
_DEFAULT_TEACHER_TIMEOUT_S = 105.0
# the token every managed vision teacher's chat template uses to close a turn. it bounds the text
# we supplied: the provider's own generation header comes after it.
_TURN_END_TOKEN = "<|im_end|>"
# the text an assistant header ends with ("<|im_start|>assistant\n"). a completion opening a new
# turn is tokenized directly after it, so it is the prefix that completion merges against.
_ASSISTANT_HEADER_TRAILING_TEXT = "\n"
# imported rather than redeclared: the renderer rejects both markers from source text, and the
# drop guard below counts pad runs. if the two ever named different strings, text-origin runs
# would silently re-enter the count the guard depends on.
from flash.content.multimodal import (  # noqa: E402
    IMAGE_PAD_TOKEN as _IMAGE_PAD_TOKEN,
)
from flash.content.multimodal import (  # noqa: E402
    IMAGE_TEACHER_PLACEHOLDER as _IMAGE_TEACHER_PLACEHOLDER,
)


class TeacherError(RuntimeError):
    """A classified managed-teacher failure."""

    def __init__(self, *args, permanent: bool = False, provider_status: int | None = None) -> None:
        super().__init__(*args)
        self.permanent = permanent
        self.provider_status = validated_provider_status(provider_status)


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


def _chat_messages(
    prompt_messages: list[dict[str, Any]],
    completion_text: str,
    image_data_uris: list[str] | tuple[str, ...],
    *,
    continue_final_assistant: bool = False,
) -> list[dict[str, Any]]:
    if not prompt_messages:
        raise _permanent("teacher multimodal scoring requires prompt messages")
    if not completion_text:
        raise _permanent("teacher scoring requires a nonempty completion")
    images = iter(image_data_uris)
    image_count = 0
    output: list[dict[str, Any]] = []
    for message_index, message in enumerate(prompt_messages):
        if not isinstance(message, dict):
            raise _permanent(f"teacher prompt message {message_index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str):
            raise _permanent(
                f"teacher prompt message {message_index} requires string role and content"
            )
        if _IMAGE_TEACHER_PLACEHOLDER not in content:
            output.append({"role": role, "content": content})
            continue
        blocks: list[dict[str, Any]] = []
        parts = content.split(_IMAGE_TEACHER_PLACEHOLDER)
        for part_index, part in enumerate(parts):
            if part:
                blocks.append({"type": "text", "text": part})
            if part_index == len(parts) - 1:
                continue
            try:
                image_uri = next(images)
            except StopIteration:
                raise _permanent(
                    "teacher prompt has more image placeholders than image data URIs"
                ) from None
            if not isinstance(image_uri, str) or not image_uri.startswith("data:image/"):
                raise _permanent("teacher multimodal scoring requires image data URIs")
            blocks.append({"type": "image_url", "image_url": {"url": image_uri}})
            image_count += 1
        output.append({"role": role, "content": blocks})
    try:
        next(images)
    except StopIteration:
        pass
    else:
        raise _permanent("teacher prompt has fewer image placeholders than image data URIs")
    if image_count == 0:
        raise _permanent("teacher multimodal scoring requires at least one image")
    # the student froze its prompt with add_generation_prompt=True, so the sampled tokens live
    # after a NEW assistant boundary. concatenating onto a historical assistant turn would score
    # them as a continuation of that turn, conditioning the opd target on a different prefix than
    # the one the student actually sampled under. the text path has the same contract: it always
    # appends a fresh "Assistant: " (see _teacher_prompt_text). the ONE case that genuinely
    # continues the trailing turn is the synthetic thinking prefill, which the caller flags,
    # because the student sampled after that prefill within the same assistant turn.
    if (
        continue_final_assistant
        and output[-1]["role"] == "assistant"
        and isinstance(output[-1]["content"], str)
    ):
        output[-1]["content"] += completion_text
    else:
        output.append({"role": "assistant", "content": completion_text})
    return output


def _supplied_turn_end(values: list[int], turn_end_token_id: int) -> int | None:
    """Locate the terminator closing the assistant turn that carries the completion.

    the completion is the last thing inside the LAST supplied message, so its turn terminator is
    the last one in the rendered prompt. anything after it is template the provider appended for
    its own generation turn, never content we supplied.
    """
    for index in range(len(values) - 1, -1, -1):
        if values[index] == turn_end_token_id:
            return index
    return None


def _image_pad_runs(values: list[int], image_pad_token_id: int) -> int:
    """Count maximal runs of the image-pad token, i.e. how many images actually expanded.

    one image expands to MANY identical pad tokens (64 for a 64px image), so a raw count cannot
    distinguish "two images" from "one image, one silently dropped": the surviving image alone
    supplies far more pads than the image count. runs are per-image, so they can.
    """
    runs = 0
    previous_was_pad = False
    for value in values:
        is_pad = value == image_pad_token_id
        if is_pad and not previous_was_pad:
            runs += 1
        previous_was_pad = is_pad
    return runs


def _normalize_multimodal_response(
    response: dict[str, Any],
    *,
    encoded_completion: list[EncodedTeacherToken],
    completion_text: str,
    image_count: int,
    image_pad_token_id: int,
    turn_end_token_id: int,
) -> TeacherScore:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise _permanent("teacher response must contain exactly one choice")
    generated_ids = _integer_list(choices[0].get("token_ids"), field="token_ids")
    if len(generated_ids) != 1:
        raise _permanent("teacher response must contain exactly one generated token id")
    remote_ids = _integer_list(response.get("prompt_token_ids"), field="prompt_token_ids")
    local_completion_ids = [token.token_id for token in encoded_completion]
    # anchor to the terminator of the supplied assistant turn, NOT to the last matching run. the
    # chat route renders its own "<|im_start|>assistant\n" generation header after our final
    # message, and those template ids collide with ordinary completions: a completion of
    # "assistant" is a single id that occurs both where the student actually answered and inside
    # that trailing header, so picking the last run scores fixed template tokens as the model's
    # answer -- silently, since every structural check still passes. confirmed live: the provider
    # returns [..., 77091, 151645, 198, 151644, 77091, 198] for that completion.
    turn_end = _supplied_turn_end(remote_ids, turn_end_token_id)
    if turn_end is None:
        raise _permanent(
            "teacher response does not terminate the supplied assistant turn; "
            "cannot locate the completion token ids"
        )
    start = turn_end - len(local_completion_ids)
    if start < 0 or remote_ids[start:turn_end] != local_completion_ids:
        # fail closed rather than score a guess. the completion ends its turn by construction, so
        # a mismatch means the rendered ids are not what we encoded -- a boundary merge with the
        # template's own "assistant\n" (a completion starting with "\n" merges into it), or a
        # provider-side template change. scoring the wrong span is worse than retrying.
        raise _permanent(
            "teacher response does not end the supplied assistant turn with the completion "
            "token ids; cannot attribute prompt logprobs to the sampled tokens"
        )
    # count pads over the PROMPT only, which is everything before the completion. the completion is
    # sampled by the student, and a vision student can emit the literal image-pad token: that token
    # lands in remote_ids as its own run and would satisfy the count for an image the provider
    # actually dropped, handing back image-unconditioned targets. rejecting the marker in prompt
    # text does not cover this -- the completion is appended after that check, straight from the
    # sampled ids.
    if _image_pad_runs(remote_ids[:start], image_pad_token_id) < image_count:
        raise _permanent(
            "teacher response expanded fewer images than were supplied; the provider may have "
            "silently dropped an image"
        )
    input_tokens, output_tokens = _validated_usage(response, len(remote_ids))
    scores = _token_keyed_scores(response.get("prompt_logprobs"), remote_ids)
    if scores is None:
        raise _permanent("teacher response is missing top-level prompt_logprobs")
    completion_scores = scores[start : start + len(encoded_completion)]
    tokens = _completion_tokens(
        encoded_completion,
        completion_scores,
        full=completion_text,
        prompt_length=0,
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
        public_url: str,
        model: str,
        *,
        timeout: float = _DEFAULT_TEACHER_TIMEOUT_S,
        max_retries: int = 4,
        tokenizer: TeacherTokenizer | None = None,
    ) -> None:
        if not capability:
            raise _permanent("no managed teacher capability available on the worker")
        if not public_url:
            raise _permanent("no Flash control-panel URL available on the worker")
        self.capability = capability
        self.base_url = public_url.rstrip("/")
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
                provider_status = None
                structured_error = False
                try:
                    payload = json.loads(error.read(64 * 1024 + 1).decode("utf-8"))
                    broker_error = payload.get("error") if isinstance(payload, dict) else None
                    if isinstance(broker_error, dict):
                        raw_code = broker_error.get("code")
                        raw_classification = broker_error.get("classification")
                        provider_status = validated_provider_status(
                            broker_error.get("provider_status")
                        )
                        if isinstance(raw_code, str) and raw_code:
                            code = raw_code
                        if isinstance(raw_classification, str) and raw_classification in {
                            "permanent",
                            "transient",
                        }:
                            # authoritative only once the body actually CLASSIFIES. an
                            # intermediary can answer a safe-to-retry status with its own generic
                            # JSON -- `{"error": {"message": "rate limited"}}` is a dict with no
                            # code and no classification -- and treating that shape as the
                            # broker's verdict suppresses the status-only fallback below, so the
                            # default `permanent` aborts a paid run the broker meant us to retry.
                            structured_error = True
                            classification = raw_classification
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    OSError,
                    http.client.IncompleteRead,
                ):
                    pass
                if not structured_error and error.code in BODY_INDEPENDENT_TRANSIENT_STATUSES:
                    # the body was lost or replaced in transit. the broker raises a retryable
                    # failure only on these statuses precisely so the signal survives that. 408 and
                    # 5xx stay ambiguous after dispatch and must not spend twice. see
                    # BODY_INDEPENDENT_TRANSIENT_STATUSES.
                    #
                    # gated on the CLASSIFICATION being absent, not on the code still being the
                    # default: an intermediary can name its own code (`gateway_rate_limited`)
                    # without ever classifying, and requiring the default code there would leave
                    # the same paid run aborted on a status the broker meant us to retry.
                    classification = "transient"
                retryable = classification == "transient"
                provider_status_detail = (
                    f" provider_status={provider_status}" if provider_status is not None else ""
                )
                broker_failure = TeacherError(
                    f"teacher broker HTTP {error.code} for {request_id} on {path}: {code}"
                    f"{provider_status_detail} "
                    f"({'transient' if retryable else 'permanent'})",
                    permanent=not retryable,
                    provider_status=provider_status,
                )
                if not retryable:
                    raise broker_failure from None
                last_error = broker_failure
            except OSError as error:
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

    def _encode_completion_in_context(
        self,
        prompt_messages: list[dict[str, Any]],
        completion_text: str,
        *,
        continue_final_assistant: bool,
    ) -> list[EncodedTeacherToken]:
        """Encode the completion as it tokenizes AFTER the text it follows, not in isolation.

        BPE merges across the boundary. a completion starting with "\\n" placed after a thinking
        prefill ending in "\\n" renders as the single token 271, so the standalone encoding
        ``[198, ...]`` never appears in the provider's prompt ids: the contiguous-run lookup then
        rejects a perfectly valid rollout, or -- worse -- matches the standalone ids somewhere
        EARLIER in the conversation and silently returns logprobs for the wrong text.

        so encode ``prefix + completion`` and keep only the tail the prefix does not already
        account for, which is what the text path gets for free by encoding the whole prompt and
        slicing.

        a completion that opens its own turn still has a boundary: the chat template renders
        "<|im_start|>assistant\\n", so the header's trailing newline merges with a completion that
        starts with one exactly as a thinking prefill would. measured on the pinned Qwen3-VL
        tokenizer, encoding in isolation loses 5 of 8 whitespace-leading completions -- "\\nred"
        encodes to [198, 1151] while the render contains [..., 77091, 271, 1151]. so the new-turn
        prefix is that newline, not the empty string.

        the boundary token can belong to BOTH sides -- "<think>\\n" + "\\nred" merges the prefix's
        trailing "\\n" and the completion's leading "\\n" into one "\\n\\n". that token is kept: it is
        the token the provider actually emitted, it carries the completion's first character, and
        dropping it would leave a run that does not occur in the rendered prompt at all. keeping it
        scores one boundary token that is partly prefill, which is honest about what the model saw;
        refusing instead would permanently fail every thinking-mode rollout that starts with a
        newline, which is most of them.
        """
        prefix = ""
        if continue_final_assistant and prompt_messages:
            last = prompt_messages[-1]
            if isinstance(last, dict) and last.get("role") == "assistant":
                content = last.get("content")
                if isinstance(content, str):
                    prefix = content
        if not prefix:
            # the assistant header's own trailing newline, which is what a new turn merges against.
            prefix = _ASSISTANT_HEADER_TRAILING_TEXT
        encoded_prefix = self.tokenizer.encode(prefix)
        encoded_joined = self.tokenizer.encode(prefix + completion_text)
        # keep every token from the first position where the two encodings diverge. an unmerged
        # boundary diverges exactly at len(prefix), reducing this to a plain slice; a merged one
        # diverges at the shared token and keeps it.
        shared = 0
        # strict=False on purpose: the joined encoding is longer than the prefix by construction,
        # and comparing only the overlap is exactly what locates the divergence point.
        for joined, pre in zip(encoded_joined, encoded_prefix, strict=False):
            if joined.token_id != pre.token_id:
                break
            shared += 1
        tail = encoded_joined[shared:]
        if not tail:
            raise _permanent(
                "teacher completion contributed no tokens after its assistant prefix; "
                "cannot attribute prompt logprobs to the sampled tokens"
            )
        # the tail must start no earlier than the boundary token. real BPE merges only the token
        # straddling the seam, so the tail begins at most one token inside the prefix; a tokenizer
        # that rewrote MORE than that would hand back tokens whose text is largely prefill, and
        # scoring them as the completion would attribute the prefill's logprobs to sampled tokens.
        if tail[0].start < len(prefix) and shared + 1 < len(encoded_prefix):
            raise _permanent(
                "teacher completion boundary rewrote more than one prefix token; "
                "cannot attribute prompt logprobs to the sampled tokens"
            )
        # rebase the character spans onto completion_text. they currently index into
        # prefix + completion, and the caller slices the COMPLETION with them. a merged boundary
        # token starts inside the prefix, so its rebased start clamps to 0 -- it covers the
        # completion's first character, which is the part that belongs to the completion.
        shift = len(prefix)
        return [
            EncodedTeacherToken(
                token_id=token.token_id,
                start=max(0, token.start - shift),
                end=max(0, token.end - shift),
            )
            for token in tail
        ]

    def _score_one_multimodal(
        self,
        prompt_messages: list[dict[str, Any]],
        completion_text: str,
        image_data_uris: list[str] | tuple[str, ...],
        continue_final_assistant: bool = False,
    ) -> TeacherScore:
        encoded_completion = self._encode_completion_in_context(
            prompt_messages,
            completion_text,
            continue_final_assistant=continue_final_assistant,
        )
        image_pad_tokens = self.tokenizer.encode(_IMAGE_PAD_TOKEN)
        if len(image_pad_tokens) != 1:
            raise _permanent(
                "managed vision teacher tokenizer must encode the image-pad marker as one token"
            )
        turn_end_tokens = self.tokenizer.encode(_TURN_END_TOKEN)
        if len(turn_end_tokens) != 1:
            raise _permanent(
                "managed vision teacher tokenizer must encode the turn terminator as one token"
            )
        messages = _chat_messages(
            prompt_messages,
            completion_text,
            image_data_uris,
            continue_final_assistant=continue_final_assistant,
        )
        response = self._post(
            "/v1/teacher/chat_completions",
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1,
                "temperature": 0,
                "seed": 0,
                "prompt_logprobs": 1,
                "return_token_ids": True,
            },
        )
        return _normalize_multimodal_response(
            response,
            encoded_completion=encoded_completion,
            completion_text=completion_text,
            image_count=len(image_data_uris),
            image_pad_token_id=image_pad_tokens[0].token_id,
            turn_end_token_id=turn_end_tokens[0].token_id,
        )

    def score_many(
        self,
        items: list[tuple[str, str]],
        *,
        on_scored: Callable[[], None] | None = None,
    ) -> list[TeacherScore]:
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

        def score_one(item):
            scored = self._score_one(*item)
            if on_scored is not None:
                on_scored()
            return scored

        return map_bounded(
            items,
            score_one,
            cap=OPD_TEACHER_SCORING_CONCURRENCY,
        )

    def score_many_multimodal(
        self,
        items: list[tuple[list[dict[str, Any]], str, list[str] | tuple[str, ...], bool]],
        *,
        on_scored: Callable[[], None] | None = None,
    ) -> list[TeacherScore]:
        """Score image-conditioned completions through the managed chat broker route.

        Reports each completion through `on_scored` for the same reason the text route does: the
        child is silent while the parent waits on the teacher, so without this signal a long image
        scoring phase is indistinguishable from a wedged child.
        """
        if not items:
            return []

        def score_one(item):
            scored = self._score_one_multimodal(*item)
            if on_scored is not None:
                on_scored()
            return scored

        return map_bounded(
            items,
            score_one,
            cap=OPD_TEACHER_SCORING_CONCURRENCY,
        )

    def score(self, prompt_text: str, completion_text: str) -> TeacherScore:
        return self._score_one(prompt_text, completion_text)
