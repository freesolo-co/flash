"""authenticated stdlib HTTP that cannot leak a credential across a redirect.

`urllib.request.Request` keeps two header bags. `headers` survives a redirect:
`HTTPRedirectHandler.redirect_request` builds the follow-up request by copying it, stripping only
content headers. `unredirected_hdrs` does not survive, and `AbstractHTTPHandler.do_open` sends both.
application code that writes `Request(url, headers={"Authorization": ...})` lands the credential in
the surviving bag, so the stdlib replays the token against whatever host the `Location` names.

the fix is to move the caller's credential into the bag a redirect cannot copy. that holds whatever
transport runs, which is what makes it the guarantee rather than the redirect refusal below: an
identity check on the transport cannot tell a test fake from a tracing or compatibility wrapper that
delegates to the real stdlib, and the wrapper's inner call follows redirects normally.

it covers the credential the caller supplied, which is the one flash owns. a credential a handler
adds mid-open cannot be covered here: `redirect_request` builds each hop as a fresh `Request`, so
anything scoped to the object we were handed is gone by the second hop. a credentialed
`ProxyHandler` re-adds `Proxy-Authorization` to that fresh request, and no amount of work on ours
reaches it. containing that is the redirect refusal's job, not this function's.

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


def _move_credentials_to_the_unredirected_bag(request: urllib.request.Request) -> None:
    """move the caller's credential headers into the bag `redirect_request` does not copy.

    `do_open` sends `unredirected_hdrs` and `headers` together, so the request is unchanged on the
    wire. only the follow-up request a redirect would build loses them.
    """

    for name in list(request.headers):
        if name.lower() not in _CREDENTIAL_HEADERS:
            continue
        # `do_open` gives `unredirected_hdrs` precedence, so an entry already there is the value
        # being sent. dropping the redirectable duplicate keeps that principal on the wire;
        # promoting it instead would silently authenticate as someone else.
        canonical = request.unredirected_hdrs.get(name)
        value = request.headers[name]
        # `remove_header` pops from both bags, so it has to run before the value is re-added.
        request.remove_header(name)
        request.add_unredirected_header(name, value if canonical is None else canonical)


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

    _move_credentials_to_the_unredirected_bag(request)
    transport = urlopen if urlopen is not None else urllib.request.urlopen
    if transport is not _STDLIB_URLOPEN:
        return transport(request, timeout=timeout)
    return urllib.request.build_opener(_NoRedirectHandler()).open(request, timeout=timeout)
