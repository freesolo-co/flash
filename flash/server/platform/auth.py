"""Bearer auth for the managed control plane."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from typing import Any

from flash.server.platform import db

INTERNAL_KEY_ENV = "FREESOLO_INTERNAL_KEY"

FREESOLO_BASE_URL_ENV = "FREESOLO_BASE_URL"
DEFAULT_FREESOLO_BASE_URL = "https://api.freesolo.co"

STANDALONE_ENV = "FLASH_STANDALONE"
STANDALONE_SERVING_ORG_ID = "flash-standalone"
_VERIFY_TIMEOUT_S = 5.0
# short negative ttl: a definitive rejection should not lock out a rotated key for long.
_VERIFY_CACHE_NEG_TTL_S = 30.0
_MAX_TOKEN_LEN = 256

# token digest -> monotonic expiry. only definitive negative results are cached.
_verify_cache: dict[str, float] = {}
_verify_cache_lock = threading.Lock()
_verify_inflight: dict[str, Future[dict[str, Any] | None]] = {}
_VERIFY_CACHE_MAX = 1024


def _verify_key_digest(token: str) -> str:
    """Return the collision-resistant state key without retaining bearer material."""
    return db.hash_key(token)


def _prune_verify_cache_locked(now: float) -> None:
    """Drop expired entries then cap size. Caller must hold ``_verify_cache_lock``."""
    for digest in [digest for digest, expires_at in _verify_cache.items() if expires_at <= now]:
        del _verify_cache[digest]
    if len(_verify_cache) >= _VERIFY_CACHE_MAX:
        for digest, _expires_at in sorted(_verify_cache.items(), key=lambda item: item[1])[
            : len(_verify_cache) - _VERIFY_CACHE_MAX + 1
        ]:
            del _verify_cache[digest]


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

    raw_user = body.get("user")
    raw_org = body.get("org")
    raw_api_key = body.get("api_key")
    user = raw_user if isinstance(raw_user, dict) else {}
    org = raw_org if isinstance(raw_org, dict) else {}
    api_key = raw_api_key if isinstance(raw_api_key, dict) else {}

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


# Identity fields copied verbatim from a verified caller identity into the auth row, and
# forwarded again by the /v1/me endpoint. Kept as one list so the two passthroughs cannot drift.
_IDENTITY_PASSTHROUGH_FIELDS = (
    "user_id",
    "org_id",
    "org_slug",
    "org_name",
    "api_key_id",
    "training_agent_job_id",
    "project_id",
)


def _external_row(row: dict, token: str, identity: dict[str, Any]) -> dict:
    out = dict(row)
    out["auth_kind"] = "freesolo_api_key"
    out["key_prefix"] = _external_key_prefix(token, identity)
    if identity.get("email"):
        out["email"] = identity["email"]
    for key in _IDENTITY_PASSTHROUGH_FIELDS:
        if identity.get(key):
            out[key] = identity[key]
    return out


def freesolo_base_url() -> str:
    """Freesolo backend base URL from env, trailing slash trimmed."""
    return (os.environ.get(FREESOLO_BASE_URL_ENV) or DEFAULT_FREESOLO_BASE_URL).rstrip("/")


def standalone() -> bool:
    """True when this self-hosted plane has no Freesolo backend.

    Standalone mode trusts ``FREESOLO_INTERNAL_KEY`` as the single-tenant authorization boundary and
    cannot distinguish organizations. Use it only under the trust model in ``SELF_HOSTING.md``.
    """
    return (os.environ.get(STANDALONE_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}


def serving_org_id(org_id: str | None) -> str:
    """return the explicit tenant or the stable standalone serving scope."""

    normalized = (org_id or "").strip()
    if normalized:
        return normalized
    return STANDALONE_SERVING_ORG_ID if standalone() else ""


def _freesolo_verify(token: str) -> dict[str, Any] | None:
    """Verify a token and return its identity; transport failures reject without caching."""
    if not token or len(token) > _MAX_TOKEN_LEN:
        return None
    digest = _verify_key_digest(token)
    now = time.monotonic()
    with _verify_cache_lock:
        negative_expiry = _verify_cache.get(digest)
        if negative_expiry is not None:
            if negative_expiry > now:
                return None
            del _verify_cache[digest]
        pending = _verify_inflight.get(digest)
        if pending is None:
            pending = Future[dict[str, Any] | None]()
            _verify_inflight[digest] = pending
            owns_verify = True
        else:
            owns_verify = False
    if not owns_verify:
        # drop the bearer material from this frame before waiting: pending.result() re-raises the
        # owner's exception here, and a traceback through this frame would otherwise carry the token.
        del token
        return pending.result()

    try:
        url = f"{freesolo_base_url()}/api/auth/verify"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        cache_negative = False
        try:
            response = urllib.request.urlopen(req, timeout=_VERIFY_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            identity = None
            cache_negative = exc.code in {401, 403}
        except (OSError, ValueError):
            identity = None
        else:
            try:
                with response as resp:
                    identity = (
                        _identity_from_verify_body(_response_body(resp))
                        if resp.status == 200
                        else None
                    )
                    response_status = resp.status
            except (urllib.error.HTTPError, OSError, ValueError):
                identity = None
            else:
                cache_negative = response_status in {401, 403}
        if cache_negative:
            with _verify_cache_lock:
                now = time.monotonic()
                _prune_verify_cache_locked(now)
                _verify_cache[digest] = now + _VERIFY_CACHE_NEG_TTL_S
    except BaseException as exc:
        with _verify_cache_lock:
            if _verify_inflight.get(digest) is pending:
                del _verify_inflight[digest]
            pending.set_exception(exc)
        raise
    else:
        with _verify_cache_lock:
            if _verify_inflight.get(digest) is pending:
                del _verify_inflight[digest]
            pending.set_result(identity)
        return identity


def authenticate(authorization: str | None) -> dict | None:
    """Resolve an ``Authorization: Bearer ...`` header to a key row, or None if unverified."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    # strip both sides to match startup preflight; otherwise a newline can pass startup but 401 every
    # standalone request. `or ""` keeps whitespace-only credentials invalid.
    internal = (os.environ.get(INTERNAL_KEY_ENV) or "").strip()
    if internal and token == internal:
        # Standalone owns its runs through a key-independent row: rotating the operator secret
        # must not orphan the runs it started (see db.ensure_standalone_owner).
        row = (
            db.ensure_standalone_owner()
            if standalone()
            else (db.lookup_key(token) or db.ensure_internal_key(token))
        )
        out = dict(row)
        out["auth_kind"] = "internal"
        return out
    if standalone():
        # No backend to verify an external key against, and an unverifiable token must never be
        # accepted. The operator key above is the ONLY credential a standalone plane honours.
        return None
    external_identity = _freesolo_verify(token)
    if external_identity is None or not external_identity.get("org_slug"):
        return None
    email = str(external_identity.get("email") or "").strip()
    if "@" not in email:
        email = ""
    external_row_data = db.lookup_key(token) or db.ensure_external_key(
        token,
        key_prefix=_external_key_prefix(token, external_identity),
        email=email or None,
    )
    return (
        _external_row(external_row_data, token, external_identity)
        if external_row_data is not None
        else None
    )
