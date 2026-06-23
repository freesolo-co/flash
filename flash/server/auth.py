"""Bearer auth for the managed control plane.

User authentication is freesolo API keys only — there is no native key system. A bearer
token equal to the operator's shared ``FREESOLO_INTERNAL_KEY`` resolves to the service
identity; any other token is verified against the freesolo backend and (on success)
resolved to a per-token user identity. A failed/unreachable verify returns False (the key
is treated as unverified), so a backend outage never admits an unverified key.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from . import db

# Operators set this to the shared freesolo internal key; a bearer token equal to it
# authenticates as the service identity (see db.ensure_internal_key).
INTERNAL_KEY_ENV = "FREESOLO_INTERNAL_KEY"

# Freesolo USER-key acceptance: a user who `flash login`s with a freesolo API key sends it as
# the bearer to this control plane. Any non-internal token is verified against the freesolo
# backend and (on success) resolved to a per-token identity.
FREESOLO_BASE_URL_ENV = "FREESOLO_BASE_URL"
DEFAULT_FREESOLO_BASE_URL = "https://api.freesolo.co"
_VERIFY_TIMEOUT_S = 5.0
_VERIFY_CACHE_TTL_S = 300.0  # short TTL so it isn't a backend round-trip per request
# Negative verdicts get a much SHORTER TTL than positives. The freesolo verify endpoint
# returns 401 not only for a genuinely-bad key but also when the backend converts an
# auth-LOOKUP infra exception (authenticate_api_key failure) into a 401 — a transient outage.
# Caching such a negative for the full 300s would lock out an otherwise-valid key for 5
# minutes after the backend recovers. A short negative TTL keeps persistent bad tokens
# rate-limited (~30s, so they don't hammer the backend) while letting a transient 401 clear
# quickly. Positives keep the long TTL.
_VERIFY_CACHE_NEG_TTL_S = 30.0
# Upper bound on a bearer token we'll cache/verify. Real freesolo API keys are short; an
# arbitrarily long bearer is rejected up front so it can't bloat _verify_cache (keyed by the
# raw token) or produce an oversized outbound Authorization header.
_MAX_TOKEN_LEN = 256

# In-process verify cache: token -> (verified_bool, expires_at). Caches positives AND
# negatives so a burst of requests for the same token hits the backend at most once per TTL.
# Bounded: pruned of expired entries on every write and capped at _VERIFY_CACHE_MAX so a
# stream of unique bearer tokens can't grow it without bound (each token is a distinct key).
_verify_cache: dict[str, tuple[bool, float]] = {}
_identity_cache: dict[str, tuple[dict[str, Any], float]] = {}
_verify_cache_lock = threading.Lock()
_VERIFY_CACHE_MAX = 1024


def _prune_verify_cache_locked(now: float) -> None:
    """Drop expired entries, then cap the cache size (oldest-expiry first).

    Caller must hold ``_verify_cache_lock``. Keeps the cache from growing unbounded as
    many distinct bearer tokens are verified over time.
    """
    for tok in [t for t, (_v, exp) in _verify_cache.items() if exp <= now]:
        del _verify_cache[tok]
        _identity_cache.pop(tok, None)
    for tok in [t for t, (_v, exp) in _identity_cache.items() if exp <= now]:
        del _identity_cache[tok]
    if len(_verify_cache) >= _VERIFY_CACHE_MAX:
        # Still over the cap after dropping expired entries: evict the soonest-to-expire
        # (oldest) entries until we're back under the cap.
        for tok, _exp in sorted(_verify_cache.items(), key=lambda kv: kv[1][1])[
            : len(_verify_cache) - _VERIFY_CACHE_MAX + 1
        ]:
            del _verify_cache[tok]
            _identity_cache.pop(tok, None)


def _freesolo_key_prefix(token: str) -> str:
    """Non-secret preview of a Freesolo API key, matching the dashboard-style public prefix."""
    parts = token.split("_", 2)
    if len(parts) >= 2 and parts[0] == "fslo" and parts[1]:
        return f"fslo_{parts[1]}"
    return f"fslo_{db.hash_key(token)[:12]}"


def _external_key_prefix(token: str, identity: dict[str, Any]) -> str:
    prefix = _str_field(identity.get("key_prefix"))
    if prefix and prefix.startswith("fslo_"):
        return prefix
    if not identity and token.startswith("fslo-user-"):
        return "freesolo"
    return _freesolo_key_prefix(token)


def _str_field(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _identity_from_verify_body(raw: bytes) -> dict[str, Any]:
    """Extract optional identity fields from the freesolo verify response.

    The backend historically returned only ``{"ok": true}``. This parser is deliberately
    tolerant so Flash surfaces real fields when the backend includes them.
    """
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(body, dict):
        return {}

    user = body.get("user") if isinstance(body.get("user"), dict) else {}
    org = body.get("org") if isinstance(body.get("org"), dict) else {}
    api_key = body.get("api_key") if isinstance(body.get("api_key"), dict) else {}

    fields = {
        "email": _str_field(body.get("email")) or _str_field(user.get("email")),
        "user_id": (
            _str_field(body.get("user_id"))
            or _str_field(body.get("created_by"))
            or _str_field(user.get("id"))
        ),
        "org_id": _str_field(body.get("org_id")) or _str_field(org.get("id")),
        "api_key_id": (
            _str_field(body.get("api_key_id"))
            or _str_field(body.get("key_id"))
            or _str_field(api_key.get("id"))
        ),
        "key_prefix": _str_field(body.get("key_prefix")) or _str_field(api_key.get("key_prefix")),
        "training_agent_job_id": _str_field(body.get("training_agent_job_id")),
        "project_id": _str_field(body.get("project_id")),
    }
    return {k: v for k, v in fields.items() if v}


def _response_body(resp: Any) -> bytes:
    read = getattr(resp, "read", None)
    if not callable(read):
        return b""
    data = read()
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode()
    return b""


def _cached_identity(token: str) -> dict[str, Any]:
    now = time.time()
    with _verify_cache_lock:
        cached = _identity_cache.get(token)
        if cached is not None and cached[1] > now:
            return dict(cached[0])
    return {}


def _external_row(row: dict, token: str, identity: dict[str, Any]) -> dict:
    out = dict(row)
    out["auth_kind"] = "freesolo_api_key"
    out["key_prefix"] = _external_key_prefix(token, identity)
    if identity.get("email"):
        out["email"] = identity["email"]
    elif not out.get("email"):
        out["email"] = "freesolo-user"
    for key in ("user_id", "org_id", "api_key_id", "training_agent_job_id", "project_id"):
        if identity.get(key):
            out[key] = identity[key]
    return out


def freesolo_base_url() -> str:
    """The freesolo backend base URL (``FREESOLO_BASE_URL`` env, else the default), trailing
    slash trimmed. Shared by auth verify and the billing client."""
    return (os.environ.get(FREESOLO_BASE_URL_ENV) or DEFAULT_FREESOLO_BASE_URL).rstrip("/")


def _freesolo_verify(token: str) -> bool:
    """Verify a token against the freesolo backend (cached, short TTL, network errors = False).

    Never raises — a swallowed network/HTTP error is treated as "not authenticated" (returns
    False), never a 500."""
    # Reject obviously-invalid oversized tokens before they touch the cache or the network.
    if not token or len(token) > _MAX_TOKEN_LEN:
        return False
    now = time.time()
    with _verify_cache_lock:
        cached = _verify_cache.get(token)
        if cached is not None and cached[1] > now:
            return cached[0]
    url = f"{freesolo_base_url()}/api/auth/verify"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    identity: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(req, timeout=_VERIFY_TIMEOUT_S) as resp:
            verified = resp.status == 200
            if verified:
                identity = _identity_from_verify_body(_response_body(resp))
    except urllib.error.HTTPError as exc:
        # Only a DEFINITIVE rejection (4xx other than 429) is a verdict worth caching as a bad
        # key. A 5xx or 429 is a transient backend hiccup — treat it like a network error
        # (return False WITHOUT caching) so a valid key isn't locked out for the whole TTL
        # while the backend is briefly unhealthy.
        if exc.code >= 500 or exc.code == 429:
            return False
        verified = False
    except (urllib.error.URLError, OSError, ValueError):
        # A TRANSIENT network/connection error is NOT a verdict: don't cache it, so a valid
        # key isn't locked out for the whole TTL after the backend recovers.
        return False
    with _verify_cache_lock:
        # Prune expired entries and cap the size before inserting so unbounded distinct
        # tokens can't grow the cache.
        _prune_verify_cache_locked(now)
        # Pick the TTL by verdict: positives last the full TTL; a negative (which may be a
        # transient backend 401 rather than a real rejection) expires quickly so a valid key
        # isn't locked out for 5 minutes after the backend recovers.
        ttl = _VERIFY_CACHE_TTL_S if verified else _VERIFY_CACHE_NEG_TTL_S
        _verify_cache[token] = (verified, now + ttl)
        if verified:
            _identity_cache[token] = (identity, now + ttl)
        else:
            _identity_cache.pop(token, None)
    return verified


def authenticate(authorization: str | None) -> dict | None:
    """Resolve an ``Authorization: Bearer ...`` header to a key row.

    Freesolo keys are the only user auth. When the operator has configured
    ``FREESOLO_INTERNAL_KEY``, that shared internal key resolves to a single service
    identity. Any other token is verified against the freesolo backend and (on success)
    resolved to a per-token user identity so a user who ``flash login``s with their freesolo
    key can drive the control plane. A token that can't be verified (bad key, or the backend
    is unreachable) is treated as unverified -> authenticate returns None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    internal = os.environ.get(INTERNAL_KEY_ENV)
    if internal and token == internal:
        row = db.lookup_key(token) or db.ensure_internal_key(token)
        out = dict(row)
        out["auth_kind"] = "internal"
        return out
    # Any non-internal token is a freesolo USER key: verify it against the freesolo backend.
    if _freesolo_verify(token):
        # A verified freesolo key gets its own per-token run-ownership identity.
        identity = _cached_identity(token)
        row = db.lookup_key(token) or db.ensure_external_key(
            token,
            key_prefix=_external_key_prefix(token, identity),
            email=identity.get("email"),
        )
        return _external_row(row, token, identity) if row is not None else None
    return None
