"""authenticated stdlib HTTP that refuses to follow a redirect.

`urllib.request.Request` keeps two header bags: `headers`, which survives a redirect, and
`unredirected_hdrs`, which does not. `HTTPRedirectHandler.redirect_request` builds the follow-up
request from `headers` alone, stripping only content headers. application code that writes
`Request(url, headers={"Authorization": ...})` lands the credential in the surviving bag, so the
stdlib will replay the token against whatever host the `Location` names.

no flash or freesolo endpoint answers 3xx, so this raises the redirect as an `HTTPError` rather
than trying to decide which hops are safe. each caller already classifies `HTTPError`, so the
rejection surfaces through its existing error path.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from typing import Any

_UrlOpen = Callable[..., Any]

# the stdlib transport as it was before any test replaced it. comparing against the live
# `urllib.request.urlopen` instead would defeat the check: a replaced attribute is trivially
# identical to itself, so every fake would be routed through the opener stack and never called.
_STDLIB_URLOPEN: _UrlOpen = urllib.request.urlopen


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """raise 3xx instead of replaying the credential against the redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _urlopen_no_redirect(
    request: urllib.request.Request,
    *,
    timeout: float,
    urlopen: _UrlOpen | None = None,
):
    """open one authenticated request without allowing a credential-bearing redirect.

    the transport is read at call time rather than bound at import, because many tests replace
    `urllib.request.urlopen` with a fake. a replaced transport is called directly: it is not the
    stdlib opener stack, so routing it through a no-redirect opener would not reach it.
    """

    transport = urlopen if urlopen is not None else urllib.request.urlopen
    if transport is not _STDLIB_URLOPEN:
        return transport(request, timeout=timeout)
    return urllib.request.build_opener(_NoRedirectHandler()).open(request, timeout=timeout)
