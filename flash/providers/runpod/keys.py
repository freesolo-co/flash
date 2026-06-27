"""RunPod multi-account key pool with quota failover ("waterfall").

``RUNPOD_API_KEY`` may hold a single key or a comma-separated list of keys, each for a
distinct RunPod account. The pool cycles in order: when the preferred account is
exhausted — out of worker quota or credits, or its key is rejected — provisioning
fails over to the next account so runs keep landing, and after the last account the
pointer wraps back to the first so quota recovered on earlier accounts is reused.
A single key (no comma) behaves exactly as before: the pool is a list of one and no
failover ever triggers.

Two cooperating notions of "which key":

* the **active** key (``_idx``) — the preferred account for *new provisioning*. Only
  ``advance_key`` moves it (on a deploy-time quota failover), and it also collapses
  ``RUNPOD_API_KEY`` to that single key so the ``runpod_flash`` SDK — which reads the
  raw env var and would otherwise send ``"key1,key2"`` as one bearer token (a 401) —
  authenticates against exactly one account.
* the **ordered** keys (``ordered_keys``) — the active account first, then the rest.
  The REST client tries them in this order *per call* without moving ``_idx``, so an
  operation on an endpoint that lives on a non-preferred account still resolves (RunPod
  endpoints are account-scoped) even after a provisioning failover moved the pointer.

The pool is captured from the environment ONCE and cached, so collapsing
``RUNPOD_API_KEY`` to a single active key never loses the rest of the pool.
"""

from __future__ import annotations

import os
import threading
import urllib.error

_ENV_VAR = "RUNPOD_API_KEY"
_lock = threading.Lock()
_pool: list[str] | None = None
_idx = 0

# HTTP statuses that mean "this account/key can't serve the request — try the next key":
# 401 key rejected, 402 payment required (out of credits), 403 forbidden / spend limit,
# 404 endpoint/job not on THIS account, 429 quota/rate. A genuine hard 4xx (400/409/422)
# and a 5xx server error are the same on every account, so they are NOT failover triggers.
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

    The runpod_flash SDK reads the raw env var, so a comma-list would be sent as one
    bearer token. Collapsing to the active key keeps the SDK authenticated against one
    account; the cached pool still holds the rest for failover.
    """
    k = active_key()
    if k is not None:
        os.environ[_ENV_VAR] = k
    return k


def advance_key() -> bool:
    """Cycle the active account to the next key for new provisioning. True iff it advanced.

    Wraps around after the last key so quota recovered on earlier accounts is reused
    (e.g. key1 → key2 → key1 → ...). The pointer ALWAYS advances for a multi-key pool, so this
    returns:

    * ``True``  — a multi-key pool: the pointer moved to the next account (possibly wrapping).
    * ``False`` — a single-key pool: there is nowhere to advance (no failover possible).

    IMPORTANT — the return value is NOT an exhaustion signal. Because the pointer wraps, a True
    does NOT mean "a fresh, untried account is now active", and there is deliberately no
    "every account tried" boolean: that depends on where a given failover loop STARTED (a prior
    run may have already advanced ``_idx``), which this function can't know. A wrap-to-index-0
    heuristic is wrong for a loop that starts mid-pool — it would stop before the wrapped-over
    accounts are tried. So callers must NOT loop on ``while advance_key(): ...`` to drain the
    pool; bound the number of failovers by ``key_count()`` instead — exactly ``key_count() - 1``
    advances visit every OTHER account once from any start (see ``deploy_train_endpoint``).

    Also collapses ``RUNPOD_API_KEY`` to the newly-active key so the SDK and the
    preferred-first REST ordering both follow the failover.
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
    """True only for an account-specific HTTP status — the cases where another account can
    actually serve the request (auth/credit/quota/not-found, ``_FAILOVER_CODES``).

    The REST client chains the underlying ``HTTPError`` as ``__cause__`` (``raise ... from e``
    on a fast-failed 4xx, ``raise ... from last`` after the retry loop), so the status code on
    the cause is authoritative. A hard 4xx (400/409/422), a 5xx server error, and network /
    timeout failures are the same on every account — the per-key retry loop already absorbs
    transient blips — so none of them fail over.
    """
    cause = exc.__cause__
    return isinstance(cause, urllib.error.HTTPError) and cause.code in _FAILOVER_CODES
