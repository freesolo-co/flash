"""HTTP transport and credential scoping for the serving backend."""

from __future__ import annotations

import atexit
import math
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import httpx

import flash.serve.contract.errors as serving_errors
from flash.serve.contract.errors import ServingError
from flash.serve.contract.responses import serving_status_error
from flash.serve.contract.urls import (
    default_serving_url,
    openai_base_url,
    serving_base_url,
    serving_control_url,
)

_INTERNAL_KEY_HEADER_NAME = "X-Freesolo-Internal-Key"
_MAX_REDIRECTS = 100
_RETRYABLE_SMOKE_503_CODES = frozenset({"adapter_loading", "engine_unavailable"})
_SMOKE_RETRY_FALLBACK_DELAY_SECONDS = 2.0
_HTTP_CLIENT: httpx.Client | None = None
_CHAT_HTTP_CLIENT: httpx.Client | None = None
_HTTP_CLIENT_LOCK = threading.Lock()


def _url_origin(url: httpx.URL) -> tuple[str, str, int | None]:
    return (url.scheme.lower(), (url.host or "").rstrip(".").lower(), url.port)


def _configured_serving_origin() -> tuple[str, str, int | None] | None:
    """Return the configured serving origin, or None when it cannot be parsed."""
    import httpx

    configured = (os.environ.get("FREESOLO_SERVING_URL") or "").strip()
    base = serving_control_url(configured or default_serving_url())
    try:
        url = httpx.URL(base)
    except Exception:
        return None
    if not url.host:
        return None
    return _url_origin(url)


def _internal_key_header() -> dict[str, str]:
    """Return the serving credential header when a key is configured."""
    key = (os.environ.get("FREESOLO_INTERNAL_KEY") or "").strip()
    return {_INTERNAL_KEY_HEADER_NAME: key} if key else {}


def _strip_internal_key_off_origin(request: httpx.Request) -> None:
    """Drop the plane credential from requests that leave the serving origin."""
    if _INTERNAL_KEY_HEADER_NAME not in request.headers:
        return
    origin = _configured_serving_origin()
    if origin is None or _url_origin(request.url) != origin:
        del request.headers[_INTERNAL_KEY_HEADER_NAME]


def _new_serving_client(**kwargs) -> httpx.Client:
    import httpx

    return httpx.Client(
        follow_redirects=True,
        max_redirects=_MAX_REDIRECTS,
        event_hooks={"request": [_strip_internal_key_off_origin]},
        **kwargs,
    )


def _http_client() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = _new_serving_client()
    return _HTTP_CLIENT


def _chat_http_client() -> httpx.Client:
    import httpx

    global _CHAT_HTTP_CLIENT
    if _CHAT_HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _CHAT_HTTP_CLIENT is None:
                _CHAT_HTTP_CLIENT = _new_serving_client(
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
                )
    return _CHAT_HTTP_CLIENT


def _close_http_client() -> None:
    global _CHAT_HTTP_CLIENT, _HTTP_CLIENT
    with _HTTP_CLIENT_LOCK:
        clients = (_HTTP_CLIENT, _CHAT_HTTP_CLIENT)
        _HTTP_CLIENT = None
        _CHAT_HTTP_CLIENT = None
    for client in clients:
        if client is not None:
            client.close()


def serving_request(
    method: str,
    url: str,
    *,
    json: dict | None = None,
    ok_statuses: tuple[int, ...] = (),
    timeout_s: float | None = None,
) -> httpx.Response:
    """Issue a serving request and translate transport failures."""
    import httpx

    timeout = 60.0 if timeout_s is None else min(60.0, max(0.0, float(timeout_s)))
    kwargs: dict = {
        "headers": _internal_key_header(),
        "timeout": timeout,
        "follow_redirects": True,
    }
    if json is not None:
        kwargs["json"] = json
    try:
        response = _http_client().request(method, url, **kwargs)
        if response.status_code in ok_statuses:
            return response
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as exc:
        raise serving_status_error(url, exc) from exc
    except httpx.RequestError as exc:
        raise ServingError(f"could not reach the serving backend at {url}: {exc}") from exc


def serving_openai_base_url() -> str:
    """Return the OpenAI-compatible serving base URL."""
    return openai_base_url(serving_base_url())


atexit.register(_close_http_client)


def is_event_stream_content_type(content_type: str) -> bool:
    """Return whether a content type is exactly the server-sent events media type."""
    media_type, _, _ = content_type.partition(";")
    return media_type.strip().lower() == "text/event-stream"


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
) -> serving_errors.RetryableServingUnavailable | None:
    if response.status_code not in {429, 503}:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return None
    error = payload["error"]
    code = error.get("code")
    exact_transient_server_error = (
        response.status_code == 503 and code == "serving_capacity_unavailable"
    ) or (response.status_code == 429 and code == "serving_overloaded")
    if error.get("type") == "server_error" and exact_transient_server_error:
        raw_delay = response.headers.get("Retry-After")
        try:
            retry_after_seconds = float(raw_delay)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(retry_after_seconds) or retry_after_seconds <= 0:
            return None
        return serving_errors.RetryableServingUnavailable(str(code), retry_after_seconds)
    if response.status_code != 503:
        return None
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
    return serving_errors.RetryableServingUnavailable(str(code), retry_after_seconds)


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
        if is_event_stream_content_type(self.headers.get("content-type", "")):
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
