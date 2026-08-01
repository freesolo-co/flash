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

# a secret value may be multiline -- a PEM private key or a JSON service-account blob is a legal
# `environment.secrets` value. redaction below is a whole-value match, so text holding only PART of
# such a value (a log tail that began mid-secret, a truncated capture) would match nothing and be
# emitted verbatim. registering each line as its own needle closes that, but only for lines long
# enough to be secret-bearing: a PEM `-----BEGIN...-----` or base64 body line is worth redacting,
# while a JSON `}` is shared with every innocent line in the log and would gut the diagnostic.
_MIN_SECRET_COMPONENT = 8


def _configured_secrets() -> tuple[str, ...]:
    values: set[str] = set()
    for key, value in os.environ.items():
        upper = key.upper()
        if not value or not (
            upper in {"AUTHORIZATION", "HF_TOKEN"} or upper.endswith(_SECRET_ENV_SUFFIXES)
        ):
            continue
        parts = [value]
        if "\n" in value:
            parts.extend(
                line
                for raw in value.splitlines()
                if len(line := raw.strip()) >= _MIN_SECRET_COMPONENT
            )
        for part in parts:
            values.add(part)
            encoded = urllib.parse.quote(part, safe="")
            if encoded != part:
                values.add(encoded)
    # longest first: a component is a substring of the whole value, so replacing the whole value
    # before its parts keeps the redaction count honest instead of leaving `<redacted>` fragments.
    return tuple(sorted(values, key=len, reverse=True))


def sanitize_diagnostic(value: Any, *, limit: int = 2000) -> str:
    """Keep useful failure context while removing credentials and bounding output."""
    text = f"{type(value).__name__}: {value}" if isinstance(value, BaseException) else str(value)
    for secret in _configured_secrets():
        text = text.replace(secret, "<redacted>")
    text = _BEARER_RE.sub("Bearer <redacted>", text)
    text = _SECRET_KEY_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text)
    return text[: max(0, int(limit))]


def neutralize_control_chars(value: Any) -> str:
    """Escape terminal control characters while preserving newlines as separators."""
    text = str(value)
    return "".join(
        char
        if char == "\n" or 0x20 <= ord(char) < 0x7F or ord(char) >= 0xA0
        else f"\\x{ord(char):02x}"
        for char in text
    )
