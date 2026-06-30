"""RunPod multi-account key pool with quota failover ("waterfall").

``RUNPOD_API_KEY`` may be a comma-separated list of keys. ``_idx`` tracks the active
account for new provisioning; ``ordered_keys`` puts the active account first for
per-call REST tries so endpoints on any account still resolve after a failover.
"""

from __future__ import annotations

import os
import threading
import urllib.error

_ENV_VAR = "RUNPOD_API_KEY"
_lock = threading.Lock()
_pool: list[str] | None = None
_idx = 0

# Account-specific failures worth retrying on another key; hard 4xx/5xx are NOT failover triggers.
_FAILOVER_CODES = frozenset({401, 402, 403, 404, 429})


def _ensure_pool() -> list[str]:
    global _pool
    with _lock:
        if _pool is None:
            raw = os.environ.get(_ENV_VAR, "") or ""
            _pool = [k.strip() for k in raw.split(",") if k.strip()]
        return _pool


def keys() -> list[str]:
    """The configured key pool, in order (empty if ``RUNPOD_API_KEY`` is unset)."""
    return list(_ensure_pool())


def key_count() -> int:
    return len(_ensure_pool())


def active_key() -> str | None:
    """The preferred account's key, or None if no key is configured."""
    pool = _ensure_pool()
    if not pool:
        return None
    with _lock:
        return pool[min(_idx, len(pool) - 1)]


def ordered_keys() -> list[str]:
    """All keys with the active account first (preferred-first per-call try order)."""
    pool = _ensure_pool()
    if not pool:
        return []
    with _lock:
        i = min(_idx, len(pool) - 1)
    return pool[i:] + pool[:i]


def select_active() -> str | None:
    """Collapse ``RUNPOD_API_KEY`` to the single active key (for the SDK) and return it.

    The SDK reads the raw env var; a comma-list would become one invalid bearer token.
    """
    k = active_key()
    if k is not None:
        os.environ[_ENV_VAR] = k
    return k


def advance_key() -> bool:
    """Cycle to the next key for new provisioning; returns True iff the pool has >1 key.

    DON'T loop on ``while advance_key()`` — the pointer wraps, so True is not an
    exhaustion signal. Bound failover attempts by ``key_count() - 1`` instead.
    """
    global _idx
    pool = _ensure_pool()
    with _lock:
        if len(pool) <= 1:
            return False
        _idx = (_idx + 1) % len(pool)
        os.environ[_ENV_VAR] = pool[_idx]
        return True


def reset() -> None:
    """Re-read the pool from the environment and reset to the first account (tests)."""
    global _pool, _idx
    with _lock:
        _pool, _idx = None, 0


def is_failover_error(exc: Exception) -> bool:
    """True only for account-specific HTTP statuses in ``_FAILOVER_CODES``."""
    cause = exc.__cause__
    return isinstance(cause, urllib.error.HTTPError) and cause.code in _FAILOVER_CODES
