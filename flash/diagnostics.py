"""Bounded credential-safe diagnostic rendering."""

from __future__ import annotations

import os
import re
import urllib.parse
from typing import Any

_SECRET_KEY_RE = re.compile(
    r"(?i)(authorization|api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|token|secret|password)"
    r"(\s*[:=]\s*)(?:bearer\s+)?([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def _configured_secrets() -> tuple[str, ...]:
    values: set[str] = set()
    for key, value in os.environ.items():
        upper = key.upper()
        if not value or not (
            upper in {"AUTHORIZATION", "HF_TOKEN"}
            or upper.endswith(_SECRET_ENV_SUFFIXES)
        ):
            continue
        values.add(value)
        encoded = urllib.parse.quote(value, safe="")
        if encoded != value:
            values.add(encoded)
    return tuple(sorted(values, key=len, reverse=True))


def sanitize_diagnostic(value: Any, *, limit: int = 2000) -> str:
    """Keep useful failure context while removing credentials and bounding output."""
    text = f"{type(value).__name__}: {value}" if isinstance(value, BaseException) else str(value)
    for secret in _configured_secrets():
        text = text.replace(secret, "<redacted>")
    text = _BEARER_RE.sub("Bearer <redacted>", text)
    text = _SECRET_KEY_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text)
    return text[: max(0, int(limit))]
