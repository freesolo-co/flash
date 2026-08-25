"""dependency-neutral typed transport for serving chat requests."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from flash.serve.errors import RetryableServingUnavailable

_RETRYABLE_SMOKE_503_CODES = frozenset({"adapter_loading", "engine_unavailable"})
_SMOKE_RETRY_FALLBACK_DELAY_SECONDS = 2.0


class RawChatStream(Protocol):
    status_code: int
    headers: dict[str, str]

    def iter_bytes(self) -> Iterator[bytes]: ...

    def close(self) -> None: ...


def retryable_smoke_unavailable(
    response: httpx.Response,
    *,
    requested_model: str,
    expected_adapter_revision: str,
    fallback_delay_seconds: float = _SMOKE_RETRY_FALLBACK_DELAY_SECONDS,
) -> RetryableServingUnavailable | None:
    if response.status_code != 503:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return None
    error = payload["error"]
    code = error.get("code")
    if (
        error.get("type") != "adapter_unavailable"
        or error.get("retryable") is not True
        or code not in _RETRYABLE_SMOKE_503_CODES
        or error.get("requested_model") != requested_model
        or error.get("adapter_revision") != expected_adapter_revision
    ):
        return None
    raw_delay = response.headers.get("Retry-After") or error.get("retry_after_seconds")
    try:
        retry_after_seconds = float(raw_delay)
    except (TypeError, ValueError):
        retry_after_seconds = fallback_delay_seconds
    if not math.isfinite(retry_after_seconds) or retry_after_seconds <= 0:
        retry_after_seconds = fallback_delay_seconds
    return RetryableServingUnavailable(str(code), retry_after_seconds)


def chat_request_body(
    run_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    thinking: bool,
    *,
    stream: bool,
    top_p: float,
    stop: list[str] | None,
    chat_template_kwargs: dict[str, Any] | None,
    structured_outputs: dict[str, Any] | None,
    stream_options: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """serialize the supported serving request fields exactly once."""

    body: dict[str, Any] = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "chat_template_kwargs": chat_template_kwargs
        if chat_template_kwargs is not None
        else {"enable_thinking": bool(thinking)},
    }
    if stream:
        body["stream"] = True
    if stop:
        body["stop"] = [str(value) for value in stop]
    if structured_outputs is not None:
        body["structured_outputs"] = structured_outputs
    if stream_options is not None:
        body["stream_options"] = stream_options
    return body


@dataclass(frozen=True, slots=True)
class BufferedChatResponse:
    payload: Any
    headers: dict[str, str]
    status_code: int


def post_chat_json(
    client_context: AbstractContextManager[httpx.Client],
    *,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    before_raise: Callable[[httpx.Response], None] | None = None,
) -> BufferedChatResponse:
    """post one buffered request and decode its json response."""

    with client_context as client:
        response = client.post(url, json=body, headers=headers, timeout=timeout)
        if before_raise is not None:
            before_raise(response)
        response.raise_for_status()
        return BufferedChatResponse(
            payload=response.json(),
            headers=dict(getattr(response, "headers", {})),
            status_code=int(getattr(response, "status_code", 200)),
        )


class _OwnedByteIterator:
    def __init__(self, chunks: Iterator[bytes], close: Callable[[], None]) -> None:
        self._chunks = chunks
        self._close: Callable[[], None] | None = close

    def __iter__(self) -> _OwnedByteIterator:
        return self

    def __next__(self) -> bytes:
        try:
            return next(self._chunks)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        close = self._close
        if close is None:
            return
        self._close = None
        try:
            close_chunks = getattr(self._chunks, "close", None)
            if close_chunks is not None:
                close_chunks()
        finally:
            close()


@dataclass(slots=True)
class OpenAIStreamResponse:
    """an entered upstream response whose bytes have one owner."""

    status_code: int
    headers: dict[str, str]
    context: AbstractContextManager[httpx.Response]
    response: httpx.Response
    frame_bytes: Callable[[Iterator[bytes]], Iterator[bytes]]
    _claimed: bool = False
    _closed: bool = False

    def iter_bytes(self) -> Iterator[bytes]:
        if self._claimed:
            raise RuntimeError("chat stream bytes have already been claimed")
        self._claimed = True
        chunks = iter(self.response.iter_bytes())
        if "text/event-stream" in self.headers.get("content-type", "").lower():
            chunks = self.frame_bytes(chunks)
        return _OwnedByteIterator(chunks, self.close)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.context.__exit__(None, None, None)


def request_chat_sse(
    client: httpx.Client,
    *,
    url: str,
    headers: dict[str, str],
    run_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    thinking: bool,
    top_p: float,
    stop: list[str] | None,
    chat_template_kwargs: dict[str, Any] | None,
    structured_outputs: dict[str, Any] | None,
    stream_options: dict[str, bool] | None,
    frame_bytes: Callable[[Iterator[bytes]], Iterator[bytes]],
) -> OpenAIStreamResponse:
    body = chat_request_body(
        run_id,
        messages,
        temperature,
        max_tokens,
        thinking,
        stream=True,
        top_p=top_p,
        stop=stop,
        chat_template_kwargs=chat_template_kwargs,
        structured_outputs=structured_outputs,
        stream_options=stream_options,
    )
    return open_chat_stream(
        client,
        url=url,
        body=body,
        headers=headers,
        timeout=30 * 60.0,
        frame_bytes=frame_bytes,
    )


def request_chat_stream(
    client: httpx.Client,
    *,
    url: str,
    headers: dict[str, str],
    run_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    thinking: bool,
    stop: list[str] | None,
    frame_bytes: Callable[[Iterator[bytes]], Iterator[bytes]],
    decode_body: Callable[[OpenAIStreamResponse, bool], Iterator[str]],
) -> Iterator[str]:
    body = chat_request_body(
        run_id,
        messages,
        temperature,
        max_tokens,
        thinking,
        stream=True,
        top_p=0.95,
        stop=stop,
        chat_template_kwargs=None,
        structured_outputs=None,
    )
    upstream = open_chat_stream(
        client,
        url=url,
        body=body,
        headers=headers,
        timeout=30 * 60.0,
        frame_bytes=frame_bytes,
    )
    try:
        upstream.response.raise_for_status()
    except BaseException:
        upstream.close()
        raise
    stream = decode_body(upstream, thinking)
    next(stream)
    return stream


def request_chat_json(
    client_context: AbstractContextManager[httpx.Client],
    *,
    url: str,
    headers: dict[str, str],
    run_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    thinking: bool,
    top_p: float,
    stop: list[str] | None,
    chat_template_kwargs: dict[str, Any] | None,
    structured_outputs: dict[str, Any] | None,
    timeout: float,
    before_raise: Callable[[httpx.Response], None] | None,
    balance_payload: Callable[[Any, bool], None],
    expected_checkpoint: str | None,
) -> Any:
    body = chat_request_body(
        run_id,
        messages,
        temperature,
        max_tokens,
        thinking,
        stream=False,
        top_p=top_p,
        stop=stop,
        chat_template_kwargs=chat_template_kwargs,
        structured_outputs=structured_outputs,
    )
    response = post_chat_json(
        client_context,
        url=url,
        body=body,
        headers=headers,
        timeout=timeout,
        before_raise=before_raise,
    )
    payload = response.payload
    balance_payload(payload, thinking)
    if expected_checkpoint and isinstance(payload, dict):
        response_headers = {key.lower(): value for key, value in response.headers.items()}
        payload["_freesolo_headers"] = {
            "adapter_revision": response_headers.get("x-freesolo-adapter-revision"),
            "checkpoint": response_headers.get("x-freesolo-checkpoint"),
            "hf_revision": response_headers.get("x-freesolo-hf-revision"),
        }
        payload["_freesolo_lora_request_adapter"] = response_headers.get(
            "x-freesolo-lora-request-adapter"
        )
    return payload


def open_chat_stream(
    client: httpx.Client,
    *,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    frame_bytes: Callable[[Iterator[bytes]], Iterator[bytes]],
) -> OpenAIStreamResponse:
    """enter one raw streaming response and transfer its lifetime to the caller."""

    context = client.stream(
        "POST",
        url,
        json=body,
        headers=headers,
        timeout=timeout,
    )
    response = context.__enter__()
    return OpenAIStreamResponse(
        status_code=int(getattr(response, "status_code", 200)),
        headers=dict(getattr(response, "headers", {})),
        context=context,
        response=response,
        frame_bytes=frame_bytes,
    )
