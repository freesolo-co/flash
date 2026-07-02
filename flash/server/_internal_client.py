"""Shared transport for internal-backend POSTs (key gate, org-id extraction, request builder)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from logging import Logger
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


def build_internal_request(
    path: str,
    body: dict[str, Any],
    *,
    token: str,
    method: str = "POST",
) -> urllib.request.Request:
    """Build a JSON ``Request`` to ``<backend>{path}`` with Bearer token auth."""
    return urllib.request.Request(
        f"{freesolo_base_url()}{path}",
        data=json.dumps(body).encode("utf-8"),
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )


UrlOpen = Callable[..., Any]


def request_internal_json(
    path: str,
    body: dict[str, Any],
    *,
    method: str,
    subject: str,
    logger: Logger,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> bool:
    """Best-effort internal JSON request; returns True on 2xx, False when disabled or failed."""
    key = internal_key()
    if not key:
        return False
    req = build_internal_request(path, body, token=key, method=method)
    try:
        with urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        detail = ""
        with suppress(Exception):
            detail = exc.read().decode("utf-8", "replace")[:500]
        logger.warning("failed to %s: HTTP %s %s", subject, exc.code, detail)
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("failed to %s: %s", subject, exc)
    return False


def post_internal_json(
    path: str,
    body: dict[str, Any],
    *,
    subject: str,
    logger: Logger,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> bool:
    return request_internal_json(
        path,
        body,
        method="POST",
        subject=subject,
        logger=logger,
        urlopen=urlopen,
    )


def delete_internal_json(
    path: str,
    body: dict[str, Any],
    *,
    subject: str,
    logger: Logger,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> bool:
    return request_internal_json(
        path,
        body,
        method="DELETE",
        subject=subject,
        logger=logger,
        urlopen=urlopen,
    )
