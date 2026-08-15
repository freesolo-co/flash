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
_CONSOLE_SCAN_BYTES = 1_048_576


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


def _redact_values(text: str, values: set[str]) -> str:
    """Replace every credential value in ``text``; see ``_needles`` for how each form is matched.

    Longest-first, so one secret containing another cannot leave a suffix of the longer one behind.
    """
    plain, bounded = _needles(values)
    for needle in sorted(plain, key=len, reverse=True):
        text = text.replace(needle, "<redacted>")
    for needle in sorted(bounded, key=len, reverse=True):
        text = re.sub(_bounded_pattern(needle), "<redacted>", text)
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

    Redaction happens on the WHOLE text first, so a credential cannot be split by this bound. The
    zero case is spelled out because ``text[-0:]`` is ``text[0:]`` -- the whole string, the exact
    opposite of a zero bound.
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


def _console_progress(console: str, offset: int) -> tuple[int, int, int]:
    """Return ``(cursor, committed, any)`` for complete heartbeat lines after ``offset``.

    Reads are capped per poll. A line larger than the cap is skipped in bounded chunks; checking the
    preceding byte keeps a cursor inside that line from treating its tail as a new record. Ordinary
    partial lines remain behind the cursor until their newline arrives, so they are judged once.
    """
    try:
        with open(console, "rb") as f:
            line_start = True
            if offset:
                f.seek(offset - 1)
                line_start = f.read(1) == b"\n"
            f.seek(offset)
            buf = f.read(_CONSOLE_SCAN_BYTES)
    except OSError:
        return -1, 0, 0

    cursor = offset
    if not line_start:
        first_newline = buf.find(b"\n")
        if first_newline < 0:
            return cursor + len(buf), 0, 0
        cursor += first_newline + 1
        buf = buf[first_newline + 1 :]

    cut = buf.rfind(b"\n") + 1
    if not cut and len(buf) == _CONSOLE_SCAN_BYTES:
        return cursor + len(buf), 0, 0
    hb = re.findall(rb'(?m)^HEARTBEAT (?!.*"liveness":).*$', buf[:cut])
    committed = sum(b'"pending":' not in line and b'"throttled":' not in line for line in hb)
    return cursor + cut, committed, len(hb)
