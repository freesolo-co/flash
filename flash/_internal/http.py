"""private stdlib http transport policies shared across flash clients."""

from __future__ import annotations

import copy
import inspect
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_UrlOpen = Callable[..., Any]
_ORIGINAL_URLOPEN = urllib.request.urlopen
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
_HANDLER_COPY_ERROR = "installed urllib handler cannot be copied safely"
_OPENER_COPY_ERROR = "installed urllib opener cannot be copied safely"


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    handler_order = 0

    def http_error_redirect(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)

    def http_response(self, request, response):
        if response.code in _REDIRECT_STATUSES:
            raise urllib.error.HTTPError(
                request.full_url,
                response.code,
                response.msg,
                response.info(),
                response,
            )
        return response

    http_error_301 = http_error_redirect
    http_error_302 = http_error_redirect
    http_error_303 = http_error_redirect
    http_error_307 = http_error_redirect
    http_error_308 = http_error_redirect
    https_response = http_response


@dataclass(frozen=True)
class _InstalledOpenerCache:
    installed: urllib.request.OpenerDirector
    handler_signature: tuple[int, tuple[int, ...]]
    addheader_signature: tuple[int, tuple[tuple[str, str], ...]]
    private: urllib.request.OpenerDirector


def _build_no_redirect_opener(*handlers: object) -> urllib.request.OpenerDirector:
    """build a private opener that returns redirects as httperrors."""

    blocker = _NoRedirectHandler()
    opener = urllib.request.build_opener(blocker, *handlers)
    for protocol in ("http", "https"):
        processors = opener.process_response.get(protocol)
        if processors is not None:
            processors.remove(blocker)
            processors.insert(0, blocker)
    return opener


_OPENER_LOCK = threading.Lock()
_DEFAULT_NO_REDIRECT_OPENER: urllib.request.OpenerDirector | None = None
_INSTALLED_OPENER_CACHE: _InstalledOpenerCache | None = None


def _handles_redirect_error(handler: urllib.request.BaseHandler) -> bool:
    try:
        return any(
            callable(inspect.getattr_static(handler, f"http_error_{status}", None))
            for status in _REDIRECT_STATUSES
        )
    except Exception:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR) from None


def _copy_installed_handler(
    handler: urllib.request.BaseHandler,
    opener: urllib.request.OpenerDirector,
) -> urllib.request.BaseHandler:
    try:
        copied = copy.copy(handler)
        if isinstance(handler, urllib.request.ProxyHandler):
            urllib.request.ProxyHandler.__init__(copied, handler.proxies)
        invalid = (
            copied is handler
            or type(copied) is not type(handler)
            or copied.__dict__ is handler.__dict__
            or getattr(copied, "parent", None) is not opener
        )
    except Exception:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR) from None
    if invalid:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR)
    return copied


def _installed_config(
    opener: urllib.request.OpenerDirector,
) -> tuple[
    tuple[urllib.request.BaseHandler, ...],
    list[tuple[str, str]],
    tuple[int, tuple[int, ...]],
    tuple[int, tuple[tuple[str, str], ...]],
]:
    try:
        handlers = tuple(opener.handlers)
        addheaders = []
        for item in opener.addheaders:
            name, value = item
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError
            addheaders.append((name, value))
    except Exception:
        raise urllib.error.URLError(_OPENER_COPY_ERROR) from None
    handler_signature = (id(opener.handlers), tuple(id(handler) for handler in handlers))
    addheader_signature = (id(opener.addheaders), tuple(addheaders))
    return handlers, addheaders, handler_signature, addheader_signature


def _clone_installed_opener(
    installed: urllib.request.OpenerDirector,
    handlers: tuple[urllib.request.BaseHandler, ...],
    addheaders: list[tuple[str, str]],
) -> urllib.request.OpenerDirector:
    copied = [
        _copy_installed_handler(handler, installed)
        for handler in handlers
        if not _handles_redirect_error(handler)
    ]
    private = _build_no_redirect_opener(*copied)
    private.addheaders = list(addheaders)
    return private


def _default_no_redirect_opener() -> urllib.request.OpenerDirector:
    global _DEFAULT_NO_REDIRECT_OPENER

    if _DEFAULT_NO_REDIRECT_OPENER is None:
        _DEFAULT_NO_REDIRECT_OPENER = _build_no_redirect_opener()
    return _DEFAULT_NO_REDIRECT_OPENER


def _active_no_redirect_opener() -> urllib.request.OpenerDirector:
    global _INSTALLED_OPENER_CACHE

    with _OPENER_LOCK:
        for _attempt in range(3):
            installed = urllib.request._opener
            if installed is None:
                return _default_no_redirect_opener()
            handlers, addheaders, handler_signature, addheader_signature = _installed_config(
                installed
            )
            cached = _INSTALLED_OPENER_CACHE
            if (
                cached is not None
                and cached.installed is installed
                and cached.handler_signature == handler_signature
            ):
                if cached.addheader_signature != addheader_signature:
                    cached.private.addheaders = list(addheaders)
                    cached = _InstalledOpenerCache(
                        installed,
                        handler_signature,
                        addheader_signature,
                        cached.private,
                    )
                    _INSTALLED_OPENER_CACHE = cached
                return cached.private
            private = _clone_installed_opener(installed, handlers, addheaders)
            current = _installed_config(installed)
            if urllib.request._opener is installed and current[2:] == (
                handler_signature,
                addheader_signature,
            ):
                _INSTALLED_OPENER_CACHE = _InstalledOpenerCache(
                    installed,
                    handler_signature,
                    addheader_signature,
                    private,
                )
                return private
        raise urllib.error.URLError(_OPENER_COPY_ERROR)


def _urlopen_no_redirect(
    request: urllib.request.Request,
    *,
    timeout: float,
    urlopen: _UrlOpen | None = None,
):
    """open one authenticated request without allowing a credential-bearing redirect."""

    transport = urllib.request.urlopen if urlopen is None else urlopen
    if transport is not _ORIGINAL_URLOPEN:
        return transport(request, timeout=timeout)
    opener = _active_no_redirect_opener()
    return opener.open(request, timeout=timeout)
