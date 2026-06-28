"""Bearer auth for the managed control plane."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from . import db

INTERNAL_KEY_ENV = "FREESOLO_INTERNAL_KEY"

FREESOLO_BASE_URL_ENV = "FREESOLO_BASE_URL"
DEFAULT_FREESOLO_BASE_URL = "https://api.freesolo.co"
_VERIFY_TIMEOUT_S = 5.0
_VERIFY_CACHE_TTL_S = 300.0
# Short negative TTL: a transient backend 401 shouldn't lock out a valid key for 5 min.
_VERIFY_CACHE_NEG_TTL_S = 30.0
_MAX_TOKEN_LEN = 256

# token -> (verified_bool, identity_dict, expires_at); positives and negatives both cached.
_verify_cache: dict[str, tuple[bool, dict[str, Any], float]] = {}
_verify_cache_lock = threading.Lock()
_VERIFY_CACHE_MAX = 1024


def _prune_verify_cache_locked(now: float) -> None:
    """Drop expired entries then cap size. Caller must hold ``_verify_cache_lock``."""
    for tok in [t for t, entry in _verify_cache.items() if entry[2] <= now]:
        del _verify_cache[tok]
    if len(_verify_cache) >= _VERIFY_CACHE_MAX:
        for tok, _entry in sorted(_verify_cache.items(), key=lambda kv: kv[1][2])[
            : len(_verify_cache) - _VERIFY_CACHE_MAX + 1
        ]:
            del _verify_cache[tok]


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
    """Extract identity fields from the freesolo verify response body."""
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
        "org_slug": _str_field(body.get("org_slug")) or _str_field(org.get("slug")),
        "org_name": _str_field(body.get("org_name")) or _str_field(org.get("name")),
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
        cached = _verify_cache.get(token)
        if cached is not None and cached[2] > now:
            return dict(cached[1])
    return {}


def _external_row(row: dict, token: str, identity: dict[str, Any]) -> dict:
    out = dict(row)
    out["auth_kind"] = "freesolo_api_key"
    out["key_prefix"] = _external_key_prefix(token, identity)
    if identity.get("email"):
        out["email"] = identity["email"]
    for key in (
        "user_id",
        "org_id",
        "org_slug",
        "org_name",
        "api_key_id",
        "training_agent_job_id",
        "project_id",
    ):
        if identity.get(key):
            out[key] = identity[key]
    return out


def _identity_email(identity: dict[str, Any]) -> str:
    email = str(identity.get("email") or "").strip()
    return email if "@" in email else ""


def freesolo_base_url() -> str:
    """Freesolo backend base URL from env, trailing slash trimmed."""
    return (os.environ.get(FREESOLO_BASE_URL_ENV) or DEFAULT_FREESOLO_BASE_URL).rstrip("/")


def _freesolo_verify(token: str) -> bool:
    """Verify a token against the freesolo backend; network errors return False, never raise."""
    if not token or len(token) > _MAX_TOKEN_LEN:
        return False
    now = time.time()
    with _verify_cache_lock:
        cached = _verify_cache.get(token)
        if cached is not None and cached[2] > now:
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
        # 5xx/429 are transient — don't cache so a valid key isn't locked out.
        if exc.code >= 500 or exc.code == 429:
            return False
        verified = False
    except (urllib.error.URLError, OSError, ValueError):
        return False
    with _verify_cache_lock:
        _prune_verify_cache_locked(now)
        ttl = _VERIFY_CACHE_TTL_S if verified else _VERIFY_CACHE_NEG_TTL_S
        _verify_cache[token] = (verified, identity if verified else {}, now + ttl)
    return verified


def authenticate(authorization: str | None) -> dict | None:
    """Resolve an ``Authorization: Bearer ...`` header to a key row, or None if unverified."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    internal = os.environ.get(INTERNAL_KEY_ENV)
    if internal and token == internal:
        row = db.lookup_key(token) or db.ensure_internal_key(token)
        out = dict(row)
        out["auth_kind"] = "internal"
        return out
    if _freesolo_verify(token):
        identity = _cached_identity(token)
        if not identity.get("org_slug"):
            return None
        email = _identity_email(identity)
        row = db.lookup_key(token) or db.ensure_external_key(
            token,
            key_prefix=_external_key_prefix(token, identity),
            email=email or None,
        )
        return _external_row(row, token, identity) if row is not None else None
    return None
