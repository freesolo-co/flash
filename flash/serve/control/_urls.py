"""canonical provider-managed public url validation."""

from __future__ import annotations

import re
from urllib.parse import SplitResult, urlsplit

_MODAL_HOST_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.modal\.run")


def _split_canonical_origin(value: object, name: str) -> tuple[str, SplitResult]:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty unpadded string")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical provider-managed https origin") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(f"{name} must be a canonical provider-managed https origin")
    hostname = parsed.hostname
    if parsed.netloc != hostname:
        raise ValueError(f"{name} must use a canonical lowercase hostname")
    return hostname, parsed


def validate_modal_public_url(value: object) -> str:
    hostname, parsed = _split_canonical_origin(value, "public_url")
    if _MODAL_HOST_RE.fullmatch(hostname) is None:
        raise ValueError("modal public_url must use an exact modal.run hostname")
    canonical = f"https://{hostname}"
    if parsed.path == "/":
        canonical += "/"
    if value != canonical:
        raise ValueError("modal public_url must be a canonical provider-managed https origin")
    return value
