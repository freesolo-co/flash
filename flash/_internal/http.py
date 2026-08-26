"""private stdlib http transport policies shared across flash clients."""

from __future__ import annotations

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
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)
