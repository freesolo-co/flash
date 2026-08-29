"""read the headers a `Request` actually puts on the wire.

`urllib.request.Request` keeps two header bags and `AbstractHTTPHandler.do_open` sends both.
`flash._internal.http` deliberately keeps credentials in `unredirected_hdrs`, the bag a redirect
cannot copy, so a fake transport that reads `req.headers` alone sees the request without its
credential - which is not what the real transport is handed.
"""

from __future__ import annotations

import urllib.request


def sent_headers(request: urllib.request.Request) -> dict[str, str]:
    """the merged header view `do_open` builds, in the same precedence order."""

    headers = dict(request.unredirected_hdrs)
    headers.update(request.headers)
    return headers
