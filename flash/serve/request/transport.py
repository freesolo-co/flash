"""HTTP transport and credential scoping for the serving backend."""

from __future__ import annotations

import atexit
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

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
_HTTP_CLIENT: httpx.Client | None = None
_CHAT_HTTP_CLIENT: httpx.Client | None = None
_STREAM_HTTP_CLIENT: httpx.Client | None = None
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


def _stream_http_client() -> httpx.Client:
    import httpx

    global _STREAM_HTTP_CLIENT
    if _STREAM_HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _STREAM_HTTP_CLIENT is None:
                _STREAM_HTTP_CLIENT = _new_serving_client(
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
                )
    return _STREAM_HTTP_CLIENT


def _close_http_client() -> None:
    global _CHAT_HTTP_CLIENT, _HTTP_CLIENT, _STREAM_HTTP_CLIENT
    with _HTTP_CLIENT_LOCK:
        clients = (_HTTP_CLIENT, _CHAT_HTTP_CLIENT, _STREAM_HTTP_CLIENT)
        _HTTP_CLIENT = None
        _CHAT_HTTP_CLIENT = None
        _STREAM_HTTP_CLIENT = None
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
