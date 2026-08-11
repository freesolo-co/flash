"""Credential redaction for the shared instance bootstrap. Runs inside the worker container.

Stdlib only — never import flash here. Every instance provider's launch script ships this file
next to ``bootstrap.py``, which imports it as a bare sibling module on the box and as a package
module locally (tests, tooling).
"""

from __future__ import annotations

import os
import re
import urllib.parse

_SECRET_RE = re.compile(
    r"(?i)(authorization|api[-_ ]?key|access[-_ ]?token|token|secret|password)"
    r"(\s*[:=]\s*)(?:bearer\s+)?([^\s,;]+)"
)

# a multiline secret (a PEM key) reaches diagnostics one component line at a time -- console tails
# are truncated and child stdout is sanitized per line -- so the whole value never matches. long
# component lines are registered as needles too; the floor keeps a common fragment such as ``}``
# from erasing innocent output. Mirrors flash._internal.diagnostics.
_MIN_SECRET_COMPONENT = 8

# leading characters dropped from a single oversized unterminated line: the byte boundary can land
# mid-credential, and the surviving fragment no longer matches full-value redaction. sized past any
# realistic token rather than tuned to one -- the cost is a few hundred characters at the start of a
# line already over the tail limit.
_SPLIT_VALUE_MARGIN = 512
_TRUNCATED_LINE_MARKER = (
    f"[flash: console tail is one unterminated line; its first {_SPLIT_VALUE_MARGIN} chars were "
    "dropped so a credential split by the read boundary cannot survive]\n"
)


def _redact_values(text: str, values: set[str]) -> str:
    """Replace every credential value in ``text``.

    A value at or above ``_MIN_SECRET_COMPONENT`` is replaced as a plain substring, longest-first,
    so one secret containing another cannot leave a suffix of the longer one behind.

    A SHORTER value is matched only where it is not adjacent to a word character. Dropping such
    values from the needle set leaked them verbatim, since ``[environment] secrets`` accepts any
    name and value; plain replacement is not the alternative either, as the 3-char value ``ati``
    would rewrite ``authentication``.

    Percent-encoded forms are registered too: encoded urls are what http and git errors print.
    Component lines of a MULTILINE value keep the floor as a hard skip -- a short component is
    punctuation such as ``}``, not a credential.
    """
    plain: set[str] = set()
    bounded: set[str] = set()
    for secret in values:
        target = plain if len(secret) >= _MIN_SECRET_COMPONENT else bounded
        target.update({secret, urllib.parse.quote(secret, safe="")})
        if "\n" in secret:
            for raw in secret.splitlines():
                if len(line := raw.strip()) >= _MIN_SECRET_COMPONENT:
                    plain.update({line, urllib.parse.quote(line, safe="")})
    for needle in sorted(plain, key=len, reverse=True):
        text = text.replace(needle, "<redacted>")
    for needle in sorted(bounded, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(needle)}(?!\w)", "<redacted>", text)
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


def _safe_detail(value: object, limit: int = 1000, secrets: dict | None = None) -> str:
    """Redact secrets by value, then by shape, then bound.

    ``secrets`` carries the run's credential values (``_payload_secrets``): the container starts
    with an empty environment and the run's HF_TOKEN / GITHUB_TOKEN / user runtime secrets only
    ever reach the worker subprocess, so ``os.environ`` alone would value-redact nothing. every
    entry of the mapping is treated as a secret regardless of its name. ``_redact_values`` handles
    the encoded forms, the ordering and the short-value case. Mirrors flash._internal.diagnostics.
    """
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
    return text[:limit]


def _read_console_tail(path: str, limit: int) -> str:
    """Read the last ``limit`` bytes of a file, dropping a leading PARTIAL line.

    the byte boundary can land inside a one-line credential, and a partial value no longer matches
    full-value redaction, so a truncated first line is dropped before sanitizing. a boundary
    landing exactly after a newline already starts a complete line, so nothing is dropped there:
    discarding it would throw away a whole line of diagnostics -- possibly the root-cause
    exception -- to solve a split that did not happen. one byte before the boundary is over-read to
    tell the two cases apart.

    when the retained region holds NO newline at all, the whole tail is one unterminated line, and
    dropping it returned an empty string: a crash whose only evidence is a single huge line (a json
    blob, a native stack, unterminated progress output) uploaded an empty console and lost the root
    cause exactly where it mattered. the line is kept minus ``_SPLIT_VALUE_MARGIN`` leading
    characters, which is what carries off a value straddling the cut; the caller's sanitizer still
    redacts any WHOLE credential that remains.
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
    if cut >= 0:
        return tail[cut + 1 :]
    return _TRUNCATED_LINE_MARKER + tail[_SPLIT_VALUE_MARGIN:]
