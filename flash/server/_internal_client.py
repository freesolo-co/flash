"""Shared transport for internal-backend POSTs (key gate, org-id extraction, request builder)."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from .auth import INTERNAL_KEY_ENV, freesolo_base_url

DEFAULT_TIMEOUT_S = 10.0


def internal_key() -> str | None:
    """Return the operator INTERNAL key, or ``None`` when unset."""
    return os.environ.get(INTERNAL_KEY_ENV)


def enabled() -> bool:
    """Return True when the operator INTERNAL key is set."""
    return bool(os.environ.get(INTERNAL_KEY_ENV))


def org_id_of(context: dict[str, Any] | None) -> str:
    """Return stripped org id from context, or ``""`` when absent."""
    return str((context or {}).get("org_id") or "").strip()


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
