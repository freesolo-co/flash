"""Client for the shared RunPod endpoint-slot quota (freesolo backend).

Advisory-locked cross-process cap via ``POST /api/runpod/internal/slots/*``; queue/fallback policy lives in ``endpoints``.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request

from flash.server.auth import INTERNAL_KEY_ENV, freesolo_base_url

_TIMEOUT_S = 10.0
_CLAIM_PATH = "/api/runpod/internal/slots/claim"
_RELEASE_PATH = "/api/runpod/internal/slots/release"
_RECONCILE_PATH = "/api/runpod/internal/slots/reconcile"


class SlotStoreError(RuntimeError):
    """The shared slot store could not be reached or returned an error."""


def internal_key() -> str | None:
    """The operator internal key, or None when unset (local/dev: use the in-process semaphore)."""
    return os.environ.get(INTERNAL_KEY_ENV, "").strip() or None


def claimed_by_ident() -> str:
    """A best-effort host:pid tag so a slot row records which replica holds it (debugging only)."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _post(path: str, body: dict) -> dict:
    key = internal_key()
    if not key:
        raise SlotStoreError(f"{INTERNAL_KEY_ENV} is not configured")
    req = urllib.request.Request(
        f"{freesolo_base_url()}{path}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            raw = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise SlotStoreError(str(exc)) from exc
    try:
        return json.loads(raw or b"{}")
    except ValueError as exc:
        raise SlotStoreError(f"bad slot-store response: {exc}") from exc


def claim(name: str, *, cap: int, claimed_by: str | None = None) -> tuple[bool, int]:
    """Atomically claim a slot. Returns ``(claimed, in_use)``; False means cap full, not an error."""
    out = _post(_CLAIM_PATH, {"name": name, "cap": cap, "claimedBy": claimed_by})
    return bool(out.get("claimed")), int(out.get("inUse") or 0)


def release(name: str) -> bool:
    """Release a slot (idempotent server-side). Returns whether a row was actually removed."""
    return bool(_post(_RELEASE_PATH, {"name": name}).get("released"))


def reconcile(live_names: list[str]) -> dict:
    """Reclaim slots whose endpoint is no longer live. Returns ``{"inUse", "reclaimed"}``."""
    return _post(_RECONCILE_PATH, {"liveNames": list(live_names)})
