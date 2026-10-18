"""shared transport for internal-backend JSON requests (key gate, org-id extraction, request builder)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from contextlib import suppress
from logging import Logger
from typing import Any

from flash._internal.http import _urlopen_no_redirect
from flash.server.platform.auth import (
    INTERNAL_KEY_ENV,
    freesolo_base_url,
    serving_org_id,
    standalone,
)

DEFAULT_TIMEOUT_S = 10.0


def internal_key() -> str | None:
    """The operator INTERNAL key (whitespace-stripped), or ``None`` when unset OR blank (which
    disables internal reporting). Normalizing here means a stray trailing newline or a
    whitespace-only value can't masquerade as 'enabled' and emit an invalid
    ``Authorization: Bearer <whitespace>`` header - every internal reporter shares this gate.

    ``None`` in standalone mode too: a self-hosted plane SETS this key (it is how its own clients
    authenticate) but has no Freesolo backend to report to, so every reporter would otherwise send
    the operator's key to ``api.freesolo.co``. Disabling at the shared gate turns billing,
    checkpoint mirroring, and environment recording off together, rather than one caller at a
    time."""
    if standalone():
        return None
    key = (os.environ.get(INTERNAL_KEY_ENV) or "").strip()
    return key or None


def org_id_of(context: dict[str, Any] | None) -> str:
    """Return stripped org id from context, or ``""`` when absent."""
    return str((context or {}).get("org_id") or "").strip()


def run_org_id(status: Any) -> str:
    """The org that owns a run: its ``billing_context`` then ``platform_context`` (the submit-path
    order), each isinstance-guarded against a non-dict legacy value; ``""`` if none. NOTE: this is the
    OPPOSITE order to ``runs`` (which prefers ``platform_context`` — see its comment); the two
    are intentionally different, do not conflate them."""
    for ctx in (
        getattr(status, "billing_context", None),
        getattr(status, "platform_context", None),
    ):
        if isinstance(ctx, dict):
            org = str(ctx.get("org_id") or "").strip()
            if org:
                return org
    return ""


def run_serving_org_id(status: Any) -> str:
    """return the managed run org or the stable standalone serving scope."""

    return serving_org_id(run_org_id(status))


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


class InternalRequestError(Exception):
    """An internal-backend response the caller asked to see instead of a flat ``False``.

    The bool contract below is deliberately lossy: every failure collapses to ``False`` so
    best-effort reporters (billing, run mirroring) can never break the operation they annotate.
    But some statuses are a permanent verdict on the request itself, not a transient hiccup, and
    a caller that renders an error to a user has to tell those apart — otherwise it advises a
    retry for a request that can never succeed. ``raise_for`` opts a single call site into
    receiving those statuses as this exception; every other caller keeps the bool.
    """

    def __init__(self, *, status: int, detail: str) -> None:
        super().__init__(detail or f"internal request failed with HTTP {status}")
        self.status = status
        self.detail = detail


def error_detail(exc: urllib.error.HTTPError) -> str:
    """The backend's own message for ``exc``, or ``""``.

    Reads the body once (it is not re-readable) and unwraps the ``detail``/``error`` envelope
    FastAPI and the Next.js routes both use, falling back to the raw text. Shared because every
    internal-backend caller needs exactly this and each hand-rolled copy drifts differently.
    """
    raw = ""
    with suppress(Exception):
        raw = exc.read().decode("utf-8", "replace")[:500]
    with suppress(Exception):
        payload = json.loads(raw)
        if isinstance(payload, dict):
            message = payload.get("detail") or payload.get("error")
            if message:
                return str(message).strip()
    return raw.strip()


def request_internal_json(
    path: str,
    body: dict[str, Any],
    *,
    method: str,
    subject: str,
    logger: Logger,
    urlopen: UrlOpen | None = None,
    raise_for: AbstractSet[int] | None = None,
    expected: AbstractSet[int] | None = None,
) -> bool:
    """Best-effort internal JSON request; returns True on 2xx, False when disabled or failed.

    ``raise_for`` names status codes to surface as :class:`InternalRequestError` rather than
    fold into ``False``. Defaults to none, so the best-effort contract is unchanged.

    ``expected`` names status codes that are a normal answer for this particular caller rather
    than a fault, and so are logged at debug instead of warning. A probe whose whole purpose is
    to ask "does this exist yet?" gets a 404 on the ordinary path; logging that as
    "failed to ..." trains readers to ignore the warning that matters.
    """
    key = internal_key()
    if not key:
        return False
    req = build_internal_request(path, body, token=key, method=method)
    try:
        with _urlopen_no_redirect(req, timeout=DEFAULT_TIMEOUT_S, urlopen=urlopen) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        detail = error_detail(exc)
        if expected and exc.code in expected:
            logger.debug("%s: HTTP %s %s", subject, exc.code, detail)
        else:
            logger.warning("failed to %s: HTTP %s %s", subject, exc.code, detail)
        if raise_for and exc.code in raise_for:
            raise InternalRequestError(status=exc.code, detail=detail) from exc
    except OSError as exc:
        logger.warning("failed to %s: %s", subject, exc)
    return False


def post_internal_json(
    path: str,
    body: dict[str, Any],
    *,
    subject: str,
    logger: Logger,
    urlopen: UrlOpen | None = None,
    raise_for: AbstractSet[int] | None = None,
    expected: AbstractSet[int] | None = None,
) -> bool:
    return request_internal_json(
        path,
        body,
        method="POST",
        subject=subject,
        logger=logger,
        urlopen=urlopen,
        raise_for=raise_for,
        expected=expected,
    )


def delete_internal_json(
    path: str,
    body: dict[str, Any],
    *,
    subject: str,
    logger: Logger,
    urlopen: UrlOpen | None = None,
) -> bool:
    return request_internal_json(
        path,
        body,
        method="DELETE",
        subject=subject,
        logger=logger,
        urlopen=urlopen,
    )
