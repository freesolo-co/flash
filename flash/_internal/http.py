"""private stdlib http transport policies shared across flash clients."""

from __future__ import annotations

import copy
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

_UrlOpen = Callable[..., Any]
_ORIGINAL_URLOPEN = urllib.request.urlopen


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _build_no_redirect_opener(*handlers: object) -> urllib.request.OpenerDirector:
    """build a private opener that returns redirects as httperrors."""

    return urllib.request.build_opener(_NoRedirectHandler(), *handlers)


_NO_REDIRECT_OPENER = _build_no_redirect_opener()
_HANDLER_COPY_ERROR = "installed urllib handler cannot be copied safely"


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


def _copy_installed_handlers(
    opener: urllib.request.OpenerDirector,
) -> list[urllib.request.BaseHandler]:
    return [
        _copy_installed_handler(handler, opener)
        for handler in tuple(opener.handlers)
        if not isinstance(handler, urllib.request.HTTPRedirectHandler)
    ]


def _active_no_redirect_opener() -> urllib.request.OpenerDirector:
    installed = urllib.request._opener
    if installed is None:
        return _NO_REDIRECT_OPENER
    return _build_no_redirect_opener(*_copy_installed_handlers(installed))


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
    return _active_no_redirect_opener().open(request, timeout=timeout)
