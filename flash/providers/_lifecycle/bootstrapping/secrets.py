"""Credential redaction for the shared instance bootstrap. Runs inside the worker container.

Stdlib only — never import flash here. Every instance provider's launch script ships this file
next to ``bootstrap.py``, which imports it as a bare sibling module on the box and as a package
module locally (tests, tooling).
"""

from __future__ import annotations

import os
import re
import urllib.parse
from typing import Literal

_SECRET_RE = re.compile(
    r"(?i)(authorization|api[-_ ]?key|access[-_ ]?token|token|secret|password)"
    r"(\s*[:=]\s*)(?:bearer\s+)?([^\s,;]+)"
)

# a multiline secret (a PEM key) reaches diagnostics one component line at a time -- console tails
# are truncated and child stdout is sanitized per line -- so the whole value never matches. long
# component lines are registered as needles too; the floor keeps a common fragment such as ``}``
# from erasing innocent output. Mirrors flash._internal.diagnostics.
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
    """
    escaped = re.escape(needle)
    left = r"(?<!\w)" if needle[:1].isalnum() or needle[:1] == "_" else ""
    right = r"(?!\w)" if needle[-1:].isalnum() or needle[-1:] == "_" else ""
    return f"{left}{escaped}{right}"


_ValueMatcher = tuple[str, bool, bool]


def _needles(values: set[str]) -> tuple[set[_ValueMatcher], set[str]]:
    """The typed value matchers and shape-only needles for ``values``.

    each value matcher carries ``(needle, bounded, encoded)`` metadata. plain needles are replaced as
    substrings. bounded needles are values under the floor, replaced only where not adjacent to a word
    character. a short raw candidate with no alphanumeric or underscore character is shape-only
    because it is indistinguishable from ordinary punctuation; explicit percent-octet forms remain
    bounded. component lines of a multiline value keep the floor as a hard skip: a short one is
    punctuation such as ``}``, not a credential.
    """
    matchers: set[_ValueMatcher] = set()
    shaped: set[str] = set()
    for secret in values:
        candidates = {(secret, False)}
        encoded = urllib.parse.quote(secret, safe="")
        if encoded != secret:
            candidates.add((encoded, True))
        if len(secret) < _MIN_SECRET_COMPONENT and not any(
            char.isalnum() or char == "_" for char in secret
        ):
            candidates.add(("".join(f"%{byte:02X}" for byte in secret.encode()), True))
        if len(secret) >= _MIN_SECRET_COMPONENT:
            matchers.update((candidate, False, is_encoded) for candidate, is_encoded in candidates)
        else:
            for candidate, is_encoded in candidates:
                if any(char.isalnum() or char == "_" for char in candidate):
                    matchers.add((candidate, True, is_encoded))
                else:
                    shaped.add(candidate)
        if "\n" in secret:
            for raw in secret.splitlines():
                if len(line := raw.strip()) >= _MIN_SECRET_COMPONENT:
                    matchers.add((line, False, False))
                    encoded_line = urllib.parse.quote(line, safe="")
                    if encoded_line != line:
                        matchers.add((encoded_line, False, True))
    return matchers, shaped


def _redact_values(text: str, values: set[str]) -> str:
    """Replace every credential value in ``text``; see ``_needles`` for how each form is matched.

    Longest-first across both matcher types, so one secret containing another cannot leave a suffix
    of the longer one behind.
    """
    matchers, shaped = _needles(values)
    # protect exact punctuation credentials before a separate value can erase their syntax.
    for needle in sorted(shaped, key=len, reverse=True):
        escaped = re.escape(needle)
        text = re.sub(
            rf"(?i)(authorization|api[-_ ]?key|access[-_ ]?token|token|secret|password)(\s*[:=]\s*)(?:bearer\s+)?{escaped}(?=[\s,;]|$)",
            lambda match: (
                "<redacted>"
                if match.group(1) in values
                else f"{match.group(1)}{match.group(2)}<redacted>"
            ),
            text,
        )
        text = re.sub(
            rf"(?i)\b(bearer)\s+{escaped}(?=[\s,;]|$)",
            lambda match: "<redacted>" if match.group(1) in values else "Bearer <redacted>",
            text,
        )
    for needle, is_bounded, is_encoded in sorted(
        matchers, key=lambda item: len(item[0]), reverse=True
    ):
        if is_encoded:
            pattern = _percent_pattern(needle)
            if is_bounded:
                left = r"(?<!\w)" if needle[:1].isalnum() or needle[:1] == "_" else ""
                right = r"(?!\w)" if needle[-1:].isalnum() or needle[-1:] == "_" else ""
                pattern = f"{left}{pattern}{right}"
            text = re.sub(pattern, "<redacted>", text)
        elif is_bounded:
            text = re.sub(_bounded_pattern(needle), "<redacted>", text)
        else:
            text = text.replace(needle, "<redacted>")
    return text


def _secret_env_name(name: str) -> bool:
    upper = str(name).upper()
    return upper in {"AUTHORIZATION", "HF_TOKEN"} or upper.endswith(
        ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
    )


def _payload_secrets(payload: dict) -> dict:
    """The payload env entries whose values are credentials.

    the payload env is the only place run secrets exist in this process (the container starts with
    an empty environment). membership comes from the control plane's explicit FLASH_SECRET_ENV_KEYS
    list, because declared runtime secrets can carry any name (AWS_SECRET_ACCESS_KEY, ...); the
    name-shape rule stays as the fail-closed fallback for names the list does not carry.
    """
    env = payload.get("env") or {}
    declared = {
        name.strip().upper()
        for name in str(env.get("FLASH_SECRET_ENV_KEYS") or "").split(",")
        if name.strip()
    }
    return {
        key: value
        for key, value in env.items()
        if value and (str(key).upper() in declared or _secret_env_name(key))
    }


def _safe_detail(
    value: object,
    limit: int = 1000,
    secrets: dict | None = None,
    keep: Literal["start", "end"] = "start",
) -> str:
    """Redact secrets by value, then by shape, then bound.

    ``secrets`` carries the run's credential values (``_payload_secrets``): the container starts
    with an empty environment and the run's HF_TOKEN / GITHUB_TOKEN / user runtime secrets only
    ever reach the worker subprocess, so ``os.environ`` alone would value-redact nothing. every
    entry of the mapping is treated as a secret regardless of its name. ``_redact_values`` handles
    the encoded forms, the ordering and the short-value case. Mirrors flash._internal.diagnostics.

    ``keep`` selects which side of an over-limit string survives. The default keeps the front, which
    suits a message whose subject comes first. ``"end"`` is for a streamed console line, where the
    root cause is the last thing written -- the end of a native stack, a json blob, or a progress
    stream -- and cutting the front is what preserves it. An unknown value raises rather than
    silently keeping the front: a typo would quietly cut the side the caller asked to keep.
    """
    if keep not in ("start", "end"):
        raise ValueError(f"keep must be 'start' or 'end', got {keep!r}")
    text = f"{type(value).__name__}: {value}" if isinstance(value, BaseException) else str(value)
    values: set[str] = set()
    for key, secret in os.environ.items():
        if secret and _secret_env_name(key):
            values.add(str(secret))
    for secret in (secrets or {}).values():
        if secret:
            values.add(str(secret))
    text = _redact_values(text, values)
    text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", text)
    text = _SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text)
    # redaction happens on the WHOLE text first, so a credential cannot be split by this bound.
    # the zero case is spelled out because ``text[-0:]`` is ``text[0:]`` -- the whole string, the
    # exact opposite of a zero bound.
    bound = max(0, int(limit))
    if not bound:
        return ""
    return text[-bound:] if keep == "end" else text[:bound]


def _read_console_tail(path: str, limit: int, secrets: dict | None = None) -> str:
    """Read the last ``limit`` bytes of a file, dropping a leading PARTIAL line.

    the byte boundary can land inside a one-line credential, and a partial value no longer matches
    full-value redaction, so a truncated first line is dropped before sanitizing. a boundary
    landing exactly after a newline already starts a complete line, so nothing is dropped there:
    discarding it would throw away a whole line of diagnostics -- possibly the root-cause
    exception -- to solve a split that did not happen. one byte before the boundary is over-read to
    tell the two cases apart.

    when the retained region holds NO newline the whole tail is one unterminated line, and it is
    dropped. that loses the only diagnostic on a crash whose evidence is one huge line (a json
    blob, a native stack), which is a real cost -- but every bound that would let the line through
    is measured against the credentials this process KNOWS, and the value at risk here is the one
    it does not. a credential minted at runtime (a presigned url, a broker capability, a token a
    provider echoed back) is in neither the payload nor the environment, so it contributes no
    needle, and a margin computed from an unrelated configured secret removes a prefix sized for
    that secret while a long fragment of the runtime one survives the cut. an empty tail never
    leaked, so it stays the floor.
    """
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        start = max(0, handle.tell() - limit)
        handle.seek(max(0, start - 1))
        raw = handle.read()
    if start == 0:
        return raw.decode("utf-8", "replace")
    tail = raw[1:].decode("utf-8", "replace")
    if raw[:1] == b"\n":
        return tail
    cut = tail.find("\n")
    return tail[cut + 1 :] if cut >= 0 else ""
