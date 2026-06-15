"""Key claiming + bearer auth for the managed control plane.

Keys are claimed instantly (no billing yet): ``POST /v1/keys`` mints one and returns it
once. The only abuse guard is a small in-memory per-IP throttle on claiming — the
operator docs are explicit that an open control plane means open GPU spend.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request

from . import db

# Operators set this to the shared freesolo internal key; a bearer token equal to it
# authenticates as the service identity (see db.ensure_internal_key). Unset -> only
# minted sk-autoslm- keys are accepted.
INTERNAL_KEY_ENV = "FREESOLO_INTERNAL_KEY"

# Freesolo USER-key acceptance: a user who `slm login`s with a freesolo API key sends it as
# the bearer to this control plane. When freesolo auth is enabled (AUTOSLM_FREESOLO_AUTH
# truthy OR FREESOLO_BASE_URL set), an unknown token is verified against the freesolo backend
# and resolved to a per-token identity. AUTOSLM_SKIP_NET disables the network call entirely.
FREESOLO_AUTH_ENV = "AUTOSLM_FREESOLO_AUTH"
FREESOLO_BASE_URL_ENV = "FREESOLO_BASE_URL"
DEFAULT_FREESOLO_BASE_URL = "https://api.freesolo.co"
_VERIFY_TIMEOUT_S = 5.0
_VERIFY_CACHE_TTL_S = 300.0  # short TTL so it isn't a backend round-trip per request

CLAIMS_PER_HOUR = 5
_claims: dict[str, list[float]] = {}
_claims_lock = threading.Lock()

# In-process verify cache: token -> (verified_bool, expires_at). Caches positives AND
# negatives so a burst of requests for the same token hits the backend at most once per TTL.
# Bounded: pruned of expired entries on every write and capped at _VERIFY_CACHE_MAX so a
# stream of unique bearer tokens can't grow it without bound (each token is a distinct key).
_verify_cache: dict[str, tuple[bool, float]] = {}
_verify_cache_lock = threading.Lock()
_VERIFY_CACHE_MAX = 1024


def _prune_verify_cache_locked(now: float) -> None:
    """Drop expired entries, then cap the cache size (oldest-expiry first).

    Caller must hold ``_verify_cache_lock``. Keeps the cache from growing unbounded as
    many distinct bearer tokens are verified over time.
    """
    for tok in [t for t, (_v, exp) in _verify_cache.items() if exp <= now]:
        del _verify_cache[tok]
    if len(_verify_cache) >= _VERIFY_CACHE_MAX:
        # Still over the cap after dropping expired entries: evict the soonest-to-expire
        # (oldest) entries until we're back under the cap.
        for tok, _exp in sorted(_verify_cache.items(), key=lambda kv: kv[1][1])[
            : len(_verify_cache) - _VERIFY_CACHE_MAX + 1
        ]:
            del _verify_cache[tok]


def _truthy(value: str | None) -> bool:
    return bool(value) and value not in ("0", "false", "False", "no", "")


def _freesolo_auth_enabled() -> bool:
    """Freesolo user-key verification is on only when explicitly configured."""
    return _truthy(os.environ.get(FREESOLO_AUTH_ENV)) or bool(os.environ.get(FREESOLO_BASE_URL_ENV))


def _freesolo_verify(token: str) -> bool:
    """Verify a token against the freesolo backend (cached, short TTL, network errors = False).

    Never raises and never makes a network call when AUTOSLM_SKIP_NET is set — a swallowed
    error or a skipped call is treated as "not authenticated" (returns False), never a 500."""
    now = time.time()
    with _verify_cache_lock:
        cached = _verify_cache.get(token)
        if cached is not None and cached[1] > now:
            return cached[0]
    # Offline guard: with AUTOSLM_SKIP_NET set we never touch the network; treat as unverified.
    if os.environ.get("AUTOSLM_SKIP_NET"):
        return False
    base = os.environ.get(FREESOLO_BASE_URL_ENV) or DEFAULT_FREESOLO_BASE_URL
    url = f"{base.rstrip('/')}/api/auth/verify"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=_VERIFY_TIMEOUT_S) as resp:
            verified = resp.status == 200
    except urllib.error.HTTPError:
        # A definitive rejection from the backend (401/403/…): cache it like a bad key.
        verified = False
    except (urllib.error.URLError, OSError, ValueError):
        # A TRANSIENT network/connection error is NOT a verdict: don't cache it, so a valid
        # key isn't locked out for the whole TTL after the backend recovers. (Never a 500.)
        return False
    with _verify_cache_lock:
        # Prune expired entries and cap the size before inserting so unbounded distinct
        # tokens can't grow the cache.
        _prune_verify_cache_locked(now)
        _verify_cache[token] = (verified, now + _VERIFY_CACHE_TTL_S)
    return verified


class ThrottledError(RuntimeError):
    pass


def claim_key(email: str | None = None, client_ip: str = "") -> dict:
    """Mint a fresh API key (throttled per client IP)."""
    now = time.time()
    with _claims_lock:
        recent = [t for t in _claims.get(client_ip, []) if now - t < 3600]
        if len(recent) >= CLAIMS_PER_HOUR:
            raise ThrottledError("too many key claims from this address; try again later")
        recent.append(now)
        _claims[client_ip] = recent
        # Bound memory: drop IPs whose claims have all aged out so the throttle table
        # doesn't grow unbounded as new client IPs hit /v1/keys over time.
        for ip in [ip for ip, ts in _claims.items() if not any(now - t < 3600 for t in ts)]:
            del _claims[ip]
    return db.create_key(email=email)


def authenticate(authorization: str | None) -> dict | None:
    """Resolve an ``Authorization: Bearer ...`` header to a key row.

    Accepts a minted ``sk-autoslm-`` key, or — when the operator has configured
    ``FREESOLO_INTERNAL_KEY`` — that shared internal key, resolved to a single service
    identity. Finally, when freesolo auth is enabled, a token that is neither is verified
    against the freesolo backend and (on success) resolved to a per-token user identity so a
    user who ``slm login``s with their freesolo key can drive the control plane."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    internal = os.environ.get(INTERNAL_KEY_ENV)
    if internal and token == internal:
        return db.lookup_key(token) or db.ensure_internal_key(token)
    if token.startswith(db.KEY_PREFIX):
        return db.lookup_key(token)
    # Neither the internal key nor a minted key: try freesolo USER-key verification, but only
    # when it's configured. (AUTOSLM_SKIP_NET / disabled flag => no network call, returns None.)
    if _freesolo_auth_enabled() and _freesolo_verify(token):
        # A verified freesolo key gets its own per-token run-ownership identity.
        return db.lookup_key(token) or db.ensure_external_key(token)
    return None
