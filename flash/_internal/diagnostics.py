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

# comma-separated env-variable names whose values are secrets regardless of naming shape: declared
# runtime secrets can carry any name (AWS_SECRET_ACCESS_KEY, FLASH_TEACHER_CAPABILITY, ...), so the
# control plane lists them explicitly in the worker env instead of relying on the suffix heuristic.
SECRET_ENV_KEYS_ENV = "FLASH_SECRET_ENV_KEYS"

# multiline secrets may appear only partially in truncated logs. register long component lines as
# needles, but ignore short common fragments such as ``}`` that would erase innocent diagnostics.
_MIN_SECRET_COMPONENT = 8
_PERCENT_ESCAPE_RE = re.compile(r"%([0-9A-Fa-f]{2})")


def _percent_pattern(needle: str) -> str:
    """Regex for ``needle`` with only percent-escape hex digits matched case-insensitively."""
    parts: list[str] = []
    offset = 0
    for match in _PERCENT_ESCAPE_RE.finditer(needle):
        parts.append(re.escape(needle[offset : match.start()]))
        parts.append("%")
        parts.extend(
            f"[{char.lower()}{char.upper()}]" if char.isalpha() else char for char in match.group(1)
        )
        offset = match.end()
    parts.append(re.escape(needle[offset:]))
    return "".join(parts)


def _bounded_pattern(needle: str) -> str:
    """``needle`` anchored so it cannot match as part of a longer word.

    The guard is applied per EDGE, and only where the needle's own edge is a word character. A
    value with a punctuation edge already separates itself from neighbouring text, and demanding a
    non-word character beyond it asks the wrong question: ``/a`` inside ``https://host/a/repo`` is
    preceded by the ``t`` of ``host``, so an unconditional left guard fails and the secret prints
    verbatim. ``ati`` keeps both guards and so still cannot rewrite ``authentication``.
    Mirrors flash.providers._lifecycle.bootstrapping.secrets._bounded_pattern.
    """
    escaped = re.escape(needle)
    left = r"(?<!\w)" if needle[:1].isalnum() or needle[:1] == "_" else ""
    right = r"(?!\w)" if needle[-1:].isalnum() or needle[-1:] == "_" else ""
    return f"{left}{escaped}{right}"


_ValueMatcher = tuple[str, bool, bool]


def _configured_secrets() -> tuple[tuple[_ValueMatcher, ...], tuple[str, ...], frozenset[str]]:
    """Credential matchers plus shape-only and exact raw values.

    Each value matcher carries ``(needle, bounded, encoded)`` metadata. Plain values are replaced as
    substrings. Bounded values are shorter than ``_MIN_SECRET_COMPONENT`` and may only match where
    they are not adjacent to a word character: a 3-char global needle would mangle every diagnostic
    containing those characters (the value ``ati`` rewrites ``authentication``).

    A short raw candidate with no alphanumeric or underscore character is shape-only because it is
    indistinguishable from ordinary punctuation. Explicit percent-octet forms remain bounded and
    safely distinguishable. Keyed and bearer matching uses the known shape-only value directly.

    Component lines of a multiline value keep the floor as a hard skip: a short component is
    punctuation such as ``}``, not a credential.
    """
    declared = {
        name.strip().upper()
        for name in os.environ.get(SECRET_ENV_KEYS_ENV, "").split(",")
        if name.strip()
    }
    matchers: set[_ValueMatcher] = set()
    shaped: set[str] = set()
    raw_values: set[str] = set()
    for key, value in os.environ.items():
        upper = key.upper()
        if not value or not (
            upper in {"AUTHORIZATION", "HF_TOKEN"}
            or upper in declared
            or upper.endswith(_SECRET_ENV_SUFFIXES)
        ):
            continue
        raw_values.add(value)
        candidates = {(value, False)}
        encoded = urllib.parse.quote(value, safe="")
        if encoded != value:
            candidates.add((encoded, True))
        if len(value) < _MIN_SECRET_COMPONENT and not any(
            char.isalnum() or char == "_" for char in value
        ):
            candidates.add(("".join(f"%{byte:02X}" for byte in value.encode()), True))
        if len(value) >= _MIN_SECRET_COMPONENT:
            matchers.update((candidate, False, is_encoded) for candidate, is_encoded in candidates)
        else:
            for candidate, is_encoded in candidates:
                if any(char.isalnum() or char == "_" for char in candidate):
                    matchers.add((candidate, True, is_encoded))
                else:
                    shaped.add(candidate)
        if "\n" in value:
            for raw in value.splitlines():
                if len(line := raw.strip()) >= _MIN_SECRET_COMPONENT:
                    matchers.add((line, False, False))
                    encoded_line = urllib.parse.quote(line, safe="")
                    if encoded_line != line:
                        matchers.add((encoded_line, False, True))
    return tuple(matchers), tuple(shaped), frozenset(raw_values)


def sanitize_diagnostic(value: Any, *, limit: int = 2000) -> str:
    """Keep useful failure context while removing credentials and bounding output."""
    text = f"{type(value).__name__}: {value}" if isinstance(value, BaseException) else str(value)
    matchers, shaped, raw_values = _configured_secrets()
    # protect exact punctuation credentials before a separate value can erase their syntax.
    for secret in sorted(shaped, key=len, reverse=True):
        escaped = re.escape(secret)
        text = re.sub(
            rf"(?i)(authorization|api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|token|secret|password)(\s*[:=]\s*)(?:bearer\s+)?{escaped}(?=[\s,;]|$)",
            lambda match: (
                "<redacted>"
                if match.group(1) in raw_values
                else f"{match.group(1)}{match.group(2)}<redacted>"
            ),
            text,
        )
        text = re.sub(
            rf"(?i)\b(bearer)\s+{escaped}(?=[\s,;]|$)",
            lambda match: "<redacted>" if match.group(1) in raw_values else "Bearer <redacted>",
            text,
        )
    # apply both matcher types in one longest-first order so a shorter plain value cannot consume
    # the prefix of a longer bounded encoded value.
    for secret, is_bounded, is_encoded in sorted(
        matchers, key=lambda item: len(item[0]), reverse=True
    ):
        if is_encoded:
            pattern = _percent_pattern(secret)
            if is_bounded:
                left = r"(?<!\w)" if secret[:1].isalnum() or secret[:1] == "_" else ""
                right = r"(?!\w)" if secret[-1:].isalnum() or secret[-1:] == "_" else ""
                pattern = f"{left}{pattern}{right}"
            text = re.sub(pattern, "<redacted>", text)
        elif is_bounded:
            text = re.sub(_bounded_pattern(secret), "<redacted>", text)
        else:
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
