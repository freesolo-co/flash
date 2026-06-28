"""Shared transport for internal-backend POSTs (key gate, org-id extraction, request builder)."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from .auth import INTERNAL_KEY_ENV, freesolo_base_url

DEFAULT_TIMEOUT_S = 10.0


def internal_key() -> str | None:
    """The operator INTERNAL key (whitespace-stripped), or ``None`` when unset OR blank (which
    disables internal reporting). Normalizing here means a stray trailing newline or a
    whitespace-only value can't masquerade as 'enabled' and emit an invalid
    ``Authorization: Bearer <whitespace>`` header — every internal reporter shares this gate."""
    key = (os.environ.get(INTERNAL_KEY_ENV) or "").strip()
    return key or None


def enabled() -> bool:
    """Internal-backend reporting is on only when the operator INTERNAL key is set and non-blank.
    Derives from ``internal_key()`` so every call site shares ONE definition of 'enabled'."""
    return internal_key() is not None


def org_id_of(context: dict[str, Any] | None) -> str:
    """Return stripped org id from context, or ``""`` when absent."""
    return str((context or {}).get("org_id") or "").strip()


def run_org_id(status: Any) -> str:
    """The org that owns a run: its ``billing_context`` then ``platform_context`` (the submit-path
    order), each isinstance-guarded against a non-dict legacy value; ``""`` if none. NOTE: this is the
    OPPOSITE order to ``run_registry`` (which prefers ``platform_context`` — see its comment); the two
    are intentionally different, do not conflate them."""
    for ctx in (getattr(status, "billing_context", None), getattr(status, "platform_context", None)):
        if isinstance(ctx, dict):
            org = str(ctx.get("org_id") or "").strip()
            if org:
                return org
    return ""


def build_internal_request(path: str, body: dict[str, Any], *, token: str) -> urllib.request.Request:
    """Build a POST ``Request`` to ``<backend>{path}`` with Bearer token and JSON body."""
    return urllib.request.Request(
        f"{freesolo_base_url()}{path}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
