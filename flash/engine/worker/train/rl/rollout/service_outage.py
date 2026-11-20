"""Tell an external environment service's outage apart from a bug in the environment's own code.

A multi-turn or reward environment often fronts a remote service (a browser farm, a sandbox, a
grader). When that service is down, the environment's client retries and then raises -- usually a
plain ``RuntimeError`` that names the service, raised inside the handler of the transport error.
Training cannot make progress until the service returns, but nothing about the run is wrong: a
replacement worker resuming from the last checkpoint is the right answer, not a terminal failure.

The evidence is the transport error itself, so this walks the exception chain -- ``__context__``
included, because ``raise RuntimeError(...) from None`` only hides the context from tracebacks; it
still records it. An exception whose chain holds no transport fault (a ``KeyError`` in scoring, a
``ValueError`` on a malformed reply, an HTTP 4xx) stays the environment's own failure.
"""

from __future__ import annotations

import urllib.error

# the bridge prefixes an outage's detail with this, after the status code, so the trainer's
# classifier can tell it from an environment raising the same words in its own message.
ENV_SERVICE_OUTAGE_STATUS = 502
ENV_SERVICE_OUTAGE_MARKER = "environment service outage"
ENV_SERVICE_OUTAGE_EVIDENCE = f"(HTTP {ENV_SERVICE_OUTAGE_STATUS}): {ENV_SERVICE_OUTAGE_MARKER}:"

# statuses that say "the service could not answer right now", never "your request is wrong".
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
# third-party clients whose transport errors do not inherit the builtin ConnectionError/TimeoutError.
_TRANSPORT_LIBRARIES = frozenset({"requests", "httpx", "urllib3", "aiohttp", "httpcore"})
_TRANSPORT_CLASS_NAMES = frozenset(
    {
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "TimeoutException",
        "TransportError",
        "NetworkError",
        "ProtocolError",
        "RemoteProtocolError",
        "NewConnectionError",
        "MaxRetryError",
        "ClientConnectionError",
        "ServerDisconnectedError",
        "ServerTimeoutError",
    }
)
_CHAIN_LIMIT = 16


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    if code is None:
        code = getattr(exc, "status", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    return code


def _is_transport_fault(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _TRANSIENT_HTTP_STATUSES
    if isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError)):
        return True
    for cls in type(exc).__mro__:
        library = cls.__module__.split(".", 1)[0]
        if library not in _TRANSPORT_LIBRARIES:
            continue
        if cls.__name__ in _TRANSPORT_CLASS_NAMES:
            return True
        if cls.__name__ in {"HTTPError", "HTTPStatusError", "ClientResponseError"}:
            return _status_code(exc) in _TRANSIENT_HTTP_STATUSES
    return False


def is_environment_service_outage(exc: BaseException) -> bool:
    """True when a transport fault reaching an external service is anywhere in ``exc``'s chain."""
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending and len(seen) < _CHAIN_LIMIT:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if _is_transport_fault(current):
            return True
        pending.extend(
            link for link in (current.__cause__, current.__context__) if link is not None
        )
    return False
