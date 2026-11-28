"""HTTP transport and credential scoping for the serving backend."""

from __future__ import annotations

import atexit
import contextvars
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import httpx

import flash.serve.contract.errors as serving_errors
from flash.serve.contract.errors import ServingError
from flash.serve.contract.protocol import MAX_CHAT_REQUEST_BYTES, TRAFFIC_SOURCE_HEADER
from flash.serve.contract.responses import serving_status_error
from flash.serve.contract.urls import (
    default_serving_url,
    openai_base_url,
    serving_base_url,
    serving_control_url,
)
from flash.serve.deployment.readiness import REVISION_READY_MAX_BUDGET_SECONDS
from flash.serve.request.tool_calls import (
    FunctionTool,
    tools_wire,
    validate_tool_control_presence,
)

_INTERNAL_KEY_HEADER_NAME = "X-Freesolo-Internal-Key"
_ORG_ID_HEADER_NAME = "X-Freesolo-Org-Id"
_MAX_REDIRECTS = 100
_RETRYABLE_SMOKE_503_CODES = ("adapter_loading", "engine_unavailable")
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


def _internal_key_header(
    *, org_id: str | None = None, traffic_source: str | None = None
) -> dict[str, str]:
    """return serving credentials with an explicit tenant scope when required.

    ``traffic_source`` names whose traffic this is for durable accounting (serving honors it only
    from a trusted internal caller): the training agent's own chats settle at $0.
    """

    key = (os.environ.get("FREESOLO_INTERNAL_KEY") or "").strip()
    headers = {_INTERNAL_KEY_HEADER_NAME: key} if key else {}
    normalized_org = (org_id or "").strip()
    if normalized_org:
        headers[_ORG_ID_HEADER_NAME] = normalized_org
    if traffic_source:
        headers[TRAFFIC_SOURCE_HEADER] = traffic_source
    return headers


def _strip_internal_key_off_origin(request: httpx.Request) -> None:
    """drop internal serving authority from requests that leave the serving origin."""

    origin = _configured_serving_origin()
    if origin is not None and _url_origin(request.url) == origin:
        return
    for name in (_INTERNAL_KEY_HEADER_NAME, _ORG_ID_HEADER_NAME):
        request.headers.pop(name, None)


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
    org_id: str | None = None,
) -> httpx.Response:
    """Issue a serving request and translate transport failures."""
    import httpx

    timeout = 60.0 if timeout_s is None else min(60.0, max(0.0, float(timeout_s)))
    kwargs: dict = {
        "headers": _internal_key_header(org_id=org_id),
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
    expected_checkpoint_id: str,
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
    )
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
        or error.get("checkpoint_id") != expected_checkpoint_id
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
    n: int,
    seed: int | None,
    frequency_penalty: float,
    presence_penalty: float,
    logprobs: bool,
    top_logprobs: int,
    stop: list[str] | None,
    chat_template_kwargs: dict[str, Any] | None,
    structured_outputs: dict[str, Any] | None,
    tools: tuple[FunctionTool, ...] | list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    parallel_tool_calls: bool | None = None,
    stream_options: dict[str, bool] | None = None,
    logprob_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """serialize the supported serving request fields exactly once."""

    validate_tool_control_presence(tools, tool_choice, parallel_tool_calls)
    body: dict[str, Any] = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": n,
        "seed": seed,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "chat_template_kwargs": chat_template_kwargs
        if chat_template_kwargs is not None
        else {"enable_thinking": bool(thinking)},
    }
    # omitted when unset so existing request bodies stay byte-identical.
    if logprob_token_ids is not None:
        body["logprob_token_ids"] = [int(token_id) for token_id in logprob_token_ids]
    if stream:
        body["stream"] = True
    if stop:
        body["stop"] = [str(value) for value in stop]
    if structured_outputs is not None:
        body["structured_outputs"] = structured_outputs
    if tools is not None:
        body["tools"] = (
            tools_wire(tools) if all(type(tool) is FunctionTool for tool in tools) else tools
        )
        body["tool_choice"] = "auto" if tool_choice is None else tool_choice
        body["parallel_tool_calls"] = True if parallel_tool_calls is None else parallel_tool_calls
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
    """post one buffered request and decode its json response.

    Inside ``ChatPostBatcher.run`` the exchange is shared with the batcher's other callers;
    everything around it (the body, ``before_raise``, status and decoding) is unchanged.
    """

    batching = _CHAT_POST_BATCH.get()
    response = None
    if batching is not None and not batching[1].posted:
        batching[1].posted = True
        response = batching[0].exchange(client_context, url, body, headers, timeout)
    if response is None:
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


# Serving's batch route: the buffered chat requests in one body, answered in order with what
# ``/chat/completions`` would have answered each (status, headers, body).
CHAT_BATCH_PATH_SUFFIX = "/batch"
# How long a serving origin without the batch route (an older release) is not asked again.
CHAT_BATCH_UNSUPPORTED_SECONDS = 10 * 60
_CHAT_BATCH_UNSUPPORTED: dict[str, float] = {}


class _BatchMember:
    __slots__ = ("posted",)

    def __init__(self) -> None:
        self.posted = False


class _BatchPost:
    __slots__ = ("body", "client_context", "done", "error", "headers", "response", "timeout", "url")

    def __init__(self, client_context, url, body, headers, timeout) -> None:
        self.client_context, self.url, self.body = client_context, url, body
        self.headers, self.timeout = dict(headers), timeout
        self.done = threading.Event()
        self.response: httpx.Response | None = None
        self.error: BaseException | None = None


_CHAT_POST_BATCH: contextvars.ContextVar[tuple[ChatPostBatcher, _BatchMember] | None] = (
    contextvars.ContextVar("flash_chat_post_batch", default=None)
)


class ChatPostBatcher:
    """Coalesce the buffered chat posts of ``size`` concurrent calls into one serving request.

    Each call runs in its own thread through ``run`` and builds its request exactly as it would
    alone. Its post waits until every call has either posted or finished without posting; the
    last one sends all posts as one batch request, and each call then handles its own item as
    the response to its own post. When the posts cannot share one request (different
    destinations or headers, too large) or serving has no batch route, every call posts on
    its own, as it would have without the batcher.
    """

    def __init__(self, size: int) -> None:
        self._size = size
        self._lock = threading.Lock()
        self._arrived = 0
        self._posts: list[_BatchPost] = []

    def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        member = _BatchMember()
        token = _CHAT_POST_BATCH.set((self, member))
        try:
            return fn(*args)
        finally:
            _CHAT_POST_BATCH.reset(token)
            if not member.posted:
                self._arrive(None)

    def exchange(self, client_context, url, body, headers, timeout) -> httpx.Response | None:
        """This call's response from the shared request, or None to post on its own."""
        post = _BatchPost(client_context, url, body, headers, timeout)
        self._arrive(post)
        post.done.wait()
        if post.error is not None:
            raise post.error
        return post.response

    def _arrive(self, post: _BatchPost | None) -> None:
        with self._lock:
            self._arrived += 1
            if post is not None:
                self._posts.append(post)
            last = self._arrived == self._size
        if last:
            self._flush(self._posts)

    def _flush(self, posts: list[_BatchPost]) -> None:
        try:
            responses = self._send(posts) if posts else None
            for post, response in zip(posts, responses or [None] * len(posts), strict=True):
                post.response = response
        except BaseException as exc:
            for post in posts:
                post.error = exc
        finally:
            for post in posts:
                post.done.set()

    @staticmethod
    def _send(posts: list[_BatchPost]) -> list[httpx.Response] | None:
        import httpx

        first = posts[0]
        url = first.url + CHAT_BATCH_PATH_SUFFIX
        if len(posts) < 2 or _CHAT_BATCH_UNSUPPORTED.get(url, 0.0) > time.monotonic():
            return None
        if any(
            (p.url, p.headers, p.timeout) != (first.url, first.headers, first.timeout)
            for p in posts
        ):
            return None
        body = {"requests": [post.body for post in posts]}
        if len(json.dumps(body, separators=(",", ":"))) > MAX_CHAT_REQUEST_BYTES:
            return None
        with first.client_context as client:
            response = client.post(url, json=body, headers=first.headers, timeout=first.timeout)
        if response.status_code in (404, 405):
            _CHAT_BATCH_UNSUPPORTED[url] = time.monotonic() + CHAT_BATCH_UNSUPPORTED_SECONDS
            return None
        response.raise_for_status()
        results = response.json().get("results")
        if not isinstance(results, list) or len(results) != len(posts):
            raise httpx.RemoteProtocolError("serving returned a malformed chat batch")
        return [
            _batch_item_response(item, post.url) for item, post in zip(results, posts, strict=True)
        ]


def _batch_item_response(item: Any, url: str) -> httpx.Response:
    import httpx

    if not isinstance(item, dict) or type(item.get("status_code")) is not int:
        raise httpx.RemoteProtocolError("serving returned a malformed chat batch item")
    headers = item.get("headers") if isinstance(item.get("headers"), dict) else {}
    request = httpx.Request("POST", url)
    if "body" in item:
        return httpx.Response(
            item["status_code"], headers=headers, json=item["body"], request=request
        )
    return httpx.Response(
        item["status_code"], headers=headers, text=str(item.get("text", "")), request=request
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
    n: int,
    seed: int | None,
    frequency_penalty: float,
    presence_penalty: float,
    logprobs: bool,
    top_logprobs: int,
    stop: list[str] | None,
    chat_template_kwargs: dict[str, Any] | None,
    structured_outputs: dict[str, Any] | None,
    tools: tuple[FunctionTool, ...] | list[dict[str, Any]] | None,
    tool_choice: str | None,
    parallel_tool_calls: bool | None,
    stream_options: dict[str, bool] | None,
    frame_bytes: Callable[[Iterator[bytes]], Iterator[bytes]],
    logprob_token_ids: Sequence[int] | None = None,
) -> OpenAIStreamResponse:
    body = chat_request_body(
        run_id,
        messages,
        temperature,
        max_tokens,
        thinking,
        stream=True,
        top_p=top_p,
        n=n,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        stop=stop,
        chat_template_kwargs=chat_template_kwargs,
        structured_outputs=structured_outputs,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        stream_options=stream_options,
        logprob_token_ids=logprob_token_ids,
    )
    return open_chat_stream(
        client,
        url=url,
        body=body,
        headers=headers,
        timeout=float(REVISION_READY_MAX_BUDGET_SECONDS + 60),
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
    n: int,
    seed: int | None,
    frequency_penalty: float,
    presence_penalty: float,
    logprobs: bool,
    top_logprobs: int,
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
        n=n,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        stop=stop,
        chat_template_kwargs=None,
        structured_outputs=None,
    )
    upstream = open_chat_stream(
        client,
        url=url,
        body=body,
        headers=headers,
        timeout=float(REVISION_READY_MAX_BUDGET_SECONDS + 60),
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
    n: int,
    seed: int | None,
    frequency_penalty: float,
    presence_penalty: float,
    logprobs: bool,
    top_logprobs: int,
    stop: list[str] | None,
    chat_template_kwargs: dict[str, Any] | None,
    structured_outputs: dict[str, Any] | None,
    tools: tuple[FunctionTool, ...] | list[dict[str, Any]] | None,
    tool_choice: str | None,
    parallel_tool_calls: bool | None,
    timeout: float,
    before_raise: Callable[[httpx.Response], None] | None,
    balance_payload: Callable[[Any, bool], None],
    expected_checkpoint: str | None,
    logprob_token_ids: Sequence[int] | None = None,
) -> Any:
    body = chat_request_body(
        run_id,
        messages,
        temperature,
        max_tokens,
        thinking,
        stream=False,
        top_p=top_p,
        n=n,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        stop=stop,
        chat_template_kwargs=chat_template_kwargs,
        structured_outputs=structured_outputs,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        logprob_token_ids=logprob_token_ids,
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
            "checkpoint_id": response_headers.get("x-freesolo-checkpoint"),
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
