"""Origin-contained HTTPS retrieval for Vast's asynchronously materialized log results.

Vast hands back a signed URL it chose. Fetching it blindly lets whatever Vast names decide where
the control plane connects, so the URL is admitted only when its origin is one the operator listed
in ``FLASH_VAST_RESULT_ORIGINS``. Two more bounds follow from that: the transport refuses redirects,
because a followed hop would leave the vetted origin, and the body is capped so a large or endless
response cannot exhaust the control plane.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request

from flash._internal.http import _urlopen_no_redirect

RESULT_ORIGINS_ENV = "FLASH_VAST_RESULT_ORIGINS"
_DEFAULT_RESULT_ORIGINS = ("https://s3.amazonaws.com",)
_MAX_RESULT_BODY_BYTES = 1_048_576

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

    Raises ``VastResultError`` for a refused origin, a redirect, any other non-200 status, a
    transport failure, or a body over the cap. Vast answers 404 until the log is written, which is
    the one status the caller polls through rather than gives up on.
    """
    request = urllib.request.Request(_admitted_url(url), method="GET")
    try:
        with _urlopen_no_redirect(request, timeout=timeout) as response:
            # one byte over the cap is enough to detect the overflow without buffering the rest.
            body = response.read(_MAX_RESULT_BODY_BYTES + 1)
    except urllib.error.HTTPError as exc:
        exc.close()
        if exc.code == 404:
            return None
        raise VastResultError(f"Vast result retrieval returned HTTP {exc.code}") from None
    except OSError:
        raise VastResultError("Vast result retrieval failed") from None
    if len(body) > _MAX_RESULT_BODY_BYTES:
        raise VastResultError(f"Vast result body exceeds the {_MAX_RESULT_BODY_BYTES}-byte limit")
    return body


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
