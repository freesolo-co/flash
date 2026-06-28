"""Best-effort reporting of published Flash environments to the Freesolo backend."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import urllib.error
import urllib.request

from .auth import INTERNAL_KEY_ENV, freesolo_base_url

_LOG = logging.getLogger("flash.server.environments")
_TIMEOUT_S = 10.0
_PATH = "/api/flash/environments/internal"
_USE_PATH = "/api/flash/environments/use/internal"
_DEFAULT_HUB_REPO = "freesolo-co/environment-hub"
_DEFAULT_HUB_REF = "main"


def record_published_environment(*, slug: str, name: str, key: dict) -> bool:
    """Persist Hub metadata in the platform backend.

    The GitHub publish is the source of truth for the environment package. This
    metadata write exists so the web UI can list Flash environments, so it is
    deliberately best-effort and never blocks `flash env push`.
    """
    internal_key = os.environ.get(INTERNAL_KEY_ENV)
    org_id = str(key.get("org_id") or "").strip()
    if not internal_key or not org_id:
        return False

    body = {
        "orgId": org_id,
        "slug": slug,
        "name": name,
        "hubRepo": _DEFAULT_HUB_REPO,
        "hubRef": _DEFAULT_HUB_REF,
        "hubPath": f"{slug}/environment.py",
        "publishedByUserId": key.get("user_id"),
        "apiKeyId": key.get("api_key_id"),
        "metadata": {"source": "flash.env.push"},
    }
    req = urllib.request.Request(
        f"{freesolo_base_url()}{_PATH}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {internal_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = exc.read().decode("utf-8", "replace")[:500]
        _LOG.warning(
            "failed to record published environment %s: HTTP %s %s",
            slug,
            exc.code,
            detail,
        )
    except (urllib.error.URLError, OSError) as exc:
        _LOG.warning("failed to record published environment %s: %s", slug, exc)
    return False


def record_deleted_environment(*, slug: str, key: dict) -> bool:
    """Remove the platform-backend metadata mirror for a deleted environment.

    Symmetric to :func:`record_published_environment`: the package store (GitHub) is the source
    of truth and is already updated by the time this runs, so dropping the row the web UI lists
    is deliberately best-effort and never blocks ``flash env delete``.
    """
    internal_key = os.environ.get(INTERNAL_KEY_ENV)
    org_id = str(key.get("org_id") or "").strip()
    if not internal_key or not org_id:
        return False

    req = urllib.request.Request(
        f"{freesolo_base_url()}{_PATH}",
        data=json.dumps({"orgId": org_id, "slug": slug}).encode("utf-8"),
        method="DELETE",
        headers={
            "Authorization": f"Bearer {internal_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = exc.read().decode("utf-8", "replace")[:500]
        _LOG.warning(
            "failed to record deleted environment %s: HTTP %s %s",
            slug,
            exc.code,
            detail,
        )
    except (urllib.error.URLError, OSError) as exc:
        _LOG.warning("failed to record deleted environment %s: %s", slug, exc)
    return False


def record_environment_use(*, slug: str, run_id: str, key: dict) -> bool:
    internal_key = os.environ.get(INTERNAL_KEY_ENV)
    org_id = str(key.get("org_id") or "").strip()
    if not internal_key or not org_id:
        return False
    req = urllib.request.Request(
        f"{freesolo_base_url()}{_USE_PATH}",
        data=json.dumps({"orgId": org_id, "slug": slug, "runId": run_id}).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {internal_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = exc.read().decode("utf-8", "replace")[:500]
        _LOG.warning(
            "failed to record environment use %s for run %s: HTTP %s %s",
            slug,
            run_id,
            exc.code,
            detail,
        )
    except (urllib.error.URLError, OSError) as exc:
        _LOG.warning("failed to record environment use %s for run %s: %s", slug, run_id, exc)
    return False
