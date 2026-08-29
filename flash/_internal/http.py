"""authenticated stdlib HTTP that cannot leak a credential across a redirect.

`urllib.request.Request` keeps two header bags. `headers` survives a redirect:
`HTTPRedirectHandler.redirect_request` builds the follow-up request by copying it, stripping only
content headers. `unredirected_hdrs` does not survive, and `AbstractHTTPHandler.do_open` sends both.
application code that writes `Request(url, headers={"Authorization": ...})` lands the credential in
the surviving bag, so the stdlib replays the token against whatever host the `Location` names.

the fix is to hold the credential in the bag a redirect cannot copy, for as long as the request
lives. that holds whatever transport runs, which is what makes it the guarantee rather than the
redirect refusal below: an identity check on the transport cannot tell a test fake from a tracing or
compatibility wrapper that delegates to the real stdlib, and the wrapper's inner call follows
redirects normally. the bag stays closed for the request's lifetime rather than being swept once,
because handlers introduce credentials mid-open: a credentialed `ProxyHandler` adds
`Proxy-Authorization` while the request is being processed, well after any one-shot sweep.

on top of that the stdlib path refuses the hop outright. no flash or freesolo endpoint answers 3xx,
so this rejects rather than deciding which hops are safe, and surfaces as an `HTTPError` that every
caller already classifies.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from typing import Any

_UrlOpen = Callable[..., Any]

# headers that authenticate the request and must never reach a redirect target.
_CREDENTIAL_HEADERS = ("authorization", "proxy-authorization", "cookie", "x-api-key")

# the stdlib transport as it was before any test replaced it. comparing against the live
# `urllib.request.urlopen` instead would defeat the check: a replaced attribute is trivially
# identical to itself, so every fake would be routed through the opener stack and never called.
_STDLIB_URLOPEN: _UrlOpen = urllib.request.urlopen


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """raise 3xx instead of following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _keep_credentials_out_of_the_redirect_bag(request: urllib.request.Request) -> None:
    """keep credential headers in the bag `redirect_request` does not copy, for this request.

    `do_open` sends `unredirected_hdrs` and `headers` together, so the original request is
    unchanged on the wire. only the follow-up request a redirect would build loses them.

    relocating once is not enough: handlers add credentials during processing. a credentialed
    `ProxyHandler` calls `add_header("Proxy-Authorization", ...)` while opening, landing a fresh
    secret in the surviving bag after any one-shot pass would have finished. so the redirect bag
    is held closed to credentials for the request's whole lifetime rather than swept at one instant.
    """

    # `remove_header` pops from both bags, so it has to run before the value is re-added.
    for name in list(request.headers):
        if name.lower() in _CREDENTIAL_HEADERS:
            value = request.headers[name]
            request.remove_header(name)
            request.add_unredirected_header(name, value)

    add_header = request.add_header

    def add_header_without_exposing_credentials(key: str, val: str) -> None:
        if key.lower() in _CREDENTIAL_HEADERS:
            request.add_unredirected_header(key, val)
        else:
            add_header(key, val)

    request.add_header = add_header_without_exposing_credentials  # type: ignore[method-assign]


def _urlopen_no_redirect(
    request: urllib.request.Request,
    *,
    timeout: float,
    urlopen: _UrlOpen | None = None,
):
    """open one authenticated request so that no redirect can carry its credential.

    the transport is read at call time rather than bound at import, because many tests replace
    `urllib.request.urlopen` with a fake. a replaced transport is called directly: it is not the
    stdlib opener stack, so routing it through a no-redirect opener would not reach it. that
    dispatch is a compatibility concern rather than the security boundary: the relocation above
    already ran, so a transport this check misjudges still cannot leak the credential.
    """

    _keep_credentials_out_of_the_redirect_bag(request)
    transport = urlopen if urlopen is not None else urllib.request.urlopen
    if transport is not _STDLIB_URLOPEN:
        return transport(request, timeout=timeout)
    return urllib.request.build_opener(_NoRedirectHandler()).open(request, timeout=timeout)
