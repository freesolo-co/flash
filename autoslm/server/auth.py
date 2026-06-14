"""Key claiming + bearer auth for the managed control plane.

Keys are claimed instantly (no billing yet): ``POST /v1/keys`` mints one and returns it
once. The only abuse guard is a small in-memory per-IP throttle on claiming — the
operator docs are explicit that an open control plane means open GPU spend.
"""

from __future__ import annotations

import os
import threading
import time

from . import db

# Operators set this to the shared freesolo internal key; a bearer token equal to it
# authenticates as the service identity (see db.ensure_internal_key). Unset -> only
# minted sk-autoslm- keys are accepted.
INTERNAL_KEY_ENV = "FREESOLO_INTERNAL_KEY"

CLAIMS_PER_HOUR = 5
_claims: dict[str, list[float]] = {}
_claims_lock = threading.Lock()


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
    ``FREESOLO_INTERNAL_KEY`` — that shared internal key, resolved to a single
    service identity (provisioned on first use so run ownership works normally)."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    internal = os.environ.get(INTERNAL_KEY_ENV)
    if internal and token == internal:
        return db.lookup_key(token) or db.ensure_internal_key(token)
    if not token.startswith(db.KEY_PREFIX):
        return None
    return db.lookup_key(token)
