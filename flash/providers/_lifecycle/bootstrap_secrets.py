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
    entry of the mapping is treated as a secret regardless of its name. replacement handles the
    percent-encoded form of each value (encoded request urls are exactly what http and git errors
    print) and runs longest-first, so one secret that contains another cannot leave a suffix of
    the longer one behind. Mirrors flash._internal.diagnostics.
    """
    text = f"{type(value).__name__}: {value}" if isinstance(value, BaseException) else str(value)
    values: set[str] = set()
    for key, secret in os.environ.items():
        if secret and _secret_env_name(key):
            values.add(str(secret))
    for secret in (secrets or {}).values():
        if secret:
            values.add(str(secret))
    needles: set[str] = set()
    for secret in values:
        needles.add(secret)
        encoded = urllib.parse.quote(secret, safe="")
        if encoded != secret:
            needles.add(encoded)
    for needle in sorted(needles, key=len, reverse=True):
        text = text.replace(needle, "<redacted>")
    text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", text)
    text = _SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text)
    return text[:limit]
