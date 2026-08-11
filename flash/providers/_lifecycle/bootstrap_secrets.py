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

_TRUNCATED_LINE_MARKER = (
    "[flash: console tail is one unterminated line; a leading fragment was dropped so a credential "
    "split by the read boundary cannot survive]\n"
)


def _needles(values: set[str]) -> tuple[set[str], set[str]]:
    """The ``(plain, bounded)`` needle sets for ``values``.

    plain = replaced as a substring. bounded = values under the floor, replaced only where not
    adjacent to a word character: dropping them leaked them verbatim, and replacing them plainly
    would rewrite innocent text (``ati`` inside ``authentication``). encoded forms are registered
    too. component lines of a MULTILINE value keep the floor as a hard skip -- a short one is
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
    return plain, bounded


def _split_margin(values: set[str]) -> int:
    """Leading characters of a split line that could still hold part of a credential.

    a value cut by the read boundary leaves up to ``len(value) - 1`` characters of itself at the
    front, and that fragment no longer matches full-value redaction. the bound comes from the
    LONGEST configured needle: a fixed margin covers only values up to its own size, and a
    903-char token straddling the cut left a 411-char suffix in the uploaded tail.
    """
    plain, bounded = _needles(values)
    return max((len(needle) for needle in plain | bounded), default=0)


def _redact_values(text: str, values: set[str]) -> str:
    """Replace every credential value in ``text``; see ``_needles`` for how each form is matched.

    Longest-first, so one secret containing another cannot leave a suffix of the longer one behind.
    """
    plain, bounded = _needles(values)
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


def _safe_detail(
    value: object, limit: int = 1000, secrets: dict | None = None, keep: str = "start"
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
    stream -- and cutting the front is what preserves it.
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
    # redaction happens on the WHOLE text first, so a credential cannot be split by this bound.
    return text[-limit:] if keep == "end" else text[:limit]


def _read_console_tail(path: str, limit: int, secrets: dict | None = None) -> str:
    """Read the last ``limit`` bytes of a file, dropping a leading PARTIAL line.

    the byte boundary can land inside a one-line credential, and a partial value no longer matches
    full-value redaction, so a truncated first line is dropped before sanitizing. a boundary
    landing exactly after a newline already starts a complete line, so nothing is dropped there:
    discarding it would throw away a whole line of diagnostics -- possibly the root-cause
    exception -- to solve a split that did not happen. one byte before the boundary is over-read to
    tell the two cases apart.

    when the retained region holds NO newline, the whole tail is one unterminated line, and
    dropping it returned "" -- a crash whose only evidence is one huge line (a json blob, a native
    stack) uploaded an empty console. the line is kept minus ``_split_margin``, the smallest prefix
    guaranteed to carry off a value straddling the cut; whole credentials are still redacted by the
    caller.

    that margin is only trustworthy for values this process KNOWS. a credential minted at runtime --
    a presigned url, a broker capability, a token echoed back by a provider -- is in neither the
    payload nor the environment, so it contributes no needle and a zero margin would retain it
    verbatim. a positive bound is therefore required to keep anything: with no configured secret to
    measure against, or a margin that would consume the line, the line is DROPPED. the empty tail
    never leaked, so it stays the floor rather than being traded for a partial credential.
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
    values: set[str] = set()
    for key, secret in os.environ.items():
        if secret and _secret_env_name(key):
            values.add(str(secret))
    for secret in (secrets or {}).values():
        if secret:
            values.add(str(secret))
    margin = _split_margin(values)
    if not margin or margin >= len(tail):
        return ""
    # the marker is prepended to a region already sized to ``limit``, so it has to be paid for out
    # of the body -- and out of its FRONT, since the root cause is at the end. returning more than
    # ``limit`` instead pushes the caller's own bound into play, whose front-biased cut would drop
    # exactly the end this branch exists to preserve.
    return (
        _TRUNCATED_LINE_MARKER
        + tail[max(margin, len(tail) - limit + len(_TRUNCATED_LINE_MARKER)) :]
    )
