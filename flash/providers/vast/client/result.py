"""Origin-contained HTTPS retrieval for Vast's asynchronously materialized log results.

Vast hands back a signed URL it chose. Fetching it blindly lets whatever Vast names decide where
the control plane connects, so the URL is admitted only when its origin is one the operator listed
in ``FLASH_VAST_RESULT_ORIGINS``. Two more bounds follow from that: the transport refuses redirects,
because a followed hop would leave the vetted origin, and the body must declare a length within the
cap, so a large or endless response cannot exhaust the control plane and a peer that hangs up
mid-log cannot pass a partial log off as a complete one.
"""

from __future__ import annotations

import http.client
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from flash._internal.http import _urlopen_no_redirect

RESULT_ORIGINS_ENV = "FLASH_VAST_RESULT_ORIGINS"
_DEFAULT_RESULT_ORIGINS = ("https://s3.amazonaws.com",)
_MAX_RESULT_BODY_BYTES = 1_048_576
_READ_CHUNK_BYTES = 65_536

_CONFIG_RULE = (
    "must be a comma-separated list of exact canonical HTTPS origins without credentials, "
    "ports, paths, queries, fragments, wildcards, spaces, or control characters"
)
_RESULT_URL_RULE = "Vast result URL violates the configured HTTPS origin policy"
_HOST_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-.")


class VastResultError(RuntimeError):
    """A result URL was refused, or could not be fetched within the origin policy."""


def configured_result_origins() -> tuple[str, ...]:
    """Return the operator's exact origin allowlist, or the S3 default when blank or unset."""
    raw = os.environ.get(RESULT_ORIGINS_ENV)
    if not raw:
        return _DEFAULT_RESULT_ORIGINS
    parts = raw.split(",")
    if _has_space_or_control(raw) or any(not part for part in parts):
        raise ValueError(f"{RESULT_ORIGINS_ENV} {_CONFIG_RULE}")
    try:
        origins = tuple(_canonical_origin(part) for part in parts)
    except ValueError:
        raise ValueError(f"{RESULT_ORIGINS_ENV} {_CONFIG_RULE}") from None
    if len(set(origins)) != len(origins):
        raise ValueError(f"{RESULT_ORIGINS_ENV} {_CONFIG_RULE}")
    return origins


def fetch_result(url: object, *, timeout: float) -> bytes | None:
    """Fetch one signed result URL, returning ``None`` while the result is not yet materialized.

    Raises ``VastResultError`` for a refused origin, a redirect, any status other than 200, a
    transport failure, a body that declares no length or declares one over the cap, a body the peer
    stops sending early, or a peer that keeps the body open past ``timeout``. Vast answers 404 until
    the log is written, which is the one status the caller polls through rather than gives up on.
    """
    request = urllib.request.Request(_admitted_url(url), method="GET")
    deadline = time.monotonic() + timeout
    try:
        with _urlopen_no_redirect(request, timeout=timeout) as response:
            status = response.getcode()
            if status != 200:
                # every other 2xx describes something that is not a complete materialized log:
                # 206 is a fragment, 202 is an acknowledgement, 204 has no body at all. urllib
                # raises only for non-2xx, so these arrive here looking like success.
                raise VastResultError(f"Vast result retrieval returned HTTP {status}")
            body = _read_bounded(response, _declared_length(response), deadline)
    except urllib.error.HTTPError as exc:
        exc.close()
        if exc.code == 404:
            return None
        raise VastResultError(f"Vast result retrieval returned HTTP {exc.code}") from None
    except (OSError, http.client.HTTPException):
        # a truncated or malformed body raises HTTPException, which is not an OSError.
        raise VastResultError("Vast result retrieval failed") from None
    return body


def _declared_length(response: object) -> int:
    """Return the byte count the peer committed to, refusing a body that declares none.

    ``length`` is the stdlib's count of bytes still owed, and before any read it is the whole
    declared body. It is ``None`` for the two framings that never state a size: chunked, and a
    close-delimited body. Requiring a declared length is what makes the rest of this module
    honest, and it removes three separate problems at once rather than handling each.

    Completeness becomes checkable. A close-delimited peer that hangs up mid-log is
    indistinguishable from one that finished, because the framing carries nothing to compare
    against; a declared length is the comparison.

    The deadline becomes real. On a chunked body one ``read1`` first parses the chunk-size line
    through ``fp.readline()``, which loops over receives until it finds a newline, each receive
    getting a fresh inactivity timeout. A peer trickling that line blocks a single call for up to
    ``_MAXLINE`` timeouts, so the deadline checked between calls bounds nothing. On a declared
    body ``read1`` is one receive, so one transport timeout is a true per-call ceiling.

    And the cap becomes a rejection instead of a download. An oversized body is refused from its
    own declaration, before a single payload byte is read.

    Vast serves these logs from S3, which always declares a length for a complete object.
    """
    length = getattr(response, "length", None)
    if not isinstance(length, int):
        # chunked or close-delimited. either can end mid-log with nothing to detect it by.
        raise VastResultError("Vast result response declared no body length")
    if length > _MAX_RESULT_BODY_BYTES:
        raise VastResultError(f"Vast result body exceeds the {_MAX_RESULT_BODY_BYTES}-byte limit")
    return length


def _read_bounded(response: object, declared: int, deadline: float) -> bytes:
    """Read exactly the declared body, bounding the whole transfer rather than the gaps in it.

    The transport timeout bounds inactivity between packets, not the transfer as a whole, so a peer
    trickling bytes under it drags the transfer out indefinitely. ``read1`` returns after a single
    receive rather than looping until the request is filled, so rechecking the deadline between
    calls bounds the transfer. That holds only because the body is declared: see
    ``_declared_length``.

    The bound is one transport timeout wide, since a call already in progress cannot be cut short.
    Tightening it further would mean re-arming the socket timeout through ``response.fp.raw._sock``
    before every read, which is undocumented internals of an object we do not own.

    ``read1`` reports a short body as a clean EOF rather than raising, so a peer that stops early
    is caught by the byte count instead.
    """
    body = bytearray()
    while len(body) < declared:
        if time.monotonic() >= deadline:
            raise VastResultError("Vast result retrieval exceeded its deadline")
        chunk = response.read1(min(_READ_CHUNK_BYTES, declared - len(body)))
        if not chunk:
            raise VastResultError("Vast result retrieval returned a truncated body")
        body += chunk
    return bytes(body)


def _admitted_url(url: object) -> str:
    if not isinstance(url, str) or _has_space_or_control(url) or "#" in url:
        raise VastResultError(_RESULT_URL_RULE)
    try:
        allowed = configured_result_origins()
    except ValueError:
        raise VastResultError("Vast result origin configuration is invalid") from None
    try:
        origin = _origin(url)
    except ValueError:
        raise VastResultError(_RESULT_URL_RULE) from None
    if origin not in allowed:
        raise VastResultError(_RESULT_URL_RULE)
    return url


def _origin(url: str) -> str:
    """Return the canonical ``https://host`` origin of a URL, rejecting anything ambiguous.

    Comparing ``netloc`` against ``hostname`` is what makes this exact: ``hostname`` strips
    userinfo and the port and lowercases the host, so any URL carrying credentials, an explicit
    port, an IPv6 literal, or mixed case fails the comparison instead of being silently normalized
    onto an allowlisted origin.
    """
    split = urllib.parse.urlsplit(url)
    host = split.hostname
    if split.scheme != "https" or not host or split.netloc != host:
        raise ValueError("scheme or authority is not canonical")
    if not _HOST_CHARACTERS.issuperset(host) or ".." in host:
        raise ValueError("host is not a canonical DNS name")
    if host.startswith((".", "-")) or host.endswith(("-",)) or ".-" in host or "-." in host:
        raise ValueError("host is not a canonical DNS name")
    return f"https://{host}"


def _canonical_origin(value: str) -> str:
    split = urllib.parse.urlsplit(value)
    if split.path or split.query or split.fragment:
        raise ValueError("origin has non-authority components")
    origin = _origin(value)
    if origin != value:
        raise ValueError("origin is not canonical")
    return origin


def _has_space_or_control(value: str) -> bool:
    return any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in value
    )
