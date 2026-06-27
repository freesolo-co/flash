"""Shared transport for the control plane's internal-backend POSTs.

Several server features report to the Freesolo backend with the operator INTERNAL key:
the run/checkpoint registry, the environment registry, realized-cost reconciliation, and
completion billing. They share one request shape -- a ``Bearer <internal-key>`` header over a
JSON body to ``<backend>/<path>`` -- but deliberately differ in error policy (best-effort bool
vs. raise-through vs. ``BillingError`` translation) and in which logger/subject they report
under. This module owns only the parts that are genuinely identical: the key gate, the org-id
extraction, and the single ``Request`` builder. Each feature keeps its own ``urlopen`` call and
error handling, so the network seam stays patchable per-module and the distinct failure
semantics stay explicit at the call site.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from .auth import INTERNAL_KEY_ENV, freesolo_base_url

# Every internal POST is off the request hot path and best-effort, so a single bounded timeout
# fits them all (a charge does a touch more than a verify but stays well under this).
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
    """The non-empty, stripped org id from a run/key context, or ``""`` when absent."""
    return str((context or {}).get("org_id") or "").strip()


def build_internal_request(path: str, body: dict[str, Any], *, token: str) -> urllib.request.Request:
    """A POST ``Request`` to ``<backend>{path}`` with the internal Bearer token and JSON body.

    The single place the Bearer + ``Content-Type`` + JSON-encode shape is defined; callers add
    their own ``urlopen`` invocation and error policy on top.
    """
    return urllib.request.Request(
        f"{freesolo_base_url()}{path}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
