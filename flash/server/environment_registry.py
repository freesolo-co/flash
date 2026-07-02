"""Best-effort reporting of published Flash environments to the Freesolo backend."""

from __future__ import annotations

import logging
import urllib.request

from ._internal_client import (
    delete_internal_json,
    org_id_of,
    post_internal_json,
)

_LOG = logging.getLogger("flash.server.environments")
_PATH = "/api/flash/environments/internal"
_USE_PATH = "/api/flash/environments/use/internal"
_DEFAULT_HUB_REPO = "freesolo-co/environment-hub"
_DEFAULT_HUB_REF = "main"


def _post(path: str, body: dict, *, subject: str) -> bool:
    return post_internal_json(
        path,
        body,
        subject=subject,
        logger=_LOG,
        urlopen=urllib.request.urlopen,
    )


def record_published_environment(*, slug: str, name: str, key: dict) -> bool:
    """Persist Hub metadata in the platform backend; best-effort, never blocks env push."""
    org_id = org_id_of(key)
    if not org_id:
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
    return _post(_PATH, body, subject=f"record published environment {slug}")


def record_deleted_environment(*, slug: str, key: dict, org_id: str | None = None) -> bool:
    """Remove the platform-backend metadata mirror for a deleted environment.

    Symmetric to :func:`record_published_environment`: the package store (GitHub) is the source
    of truth and is already updated by the time this runs, so dropping the row the web UI lists
    is deliberately best-effort and never blocks ``flash env delete``.

    ``key`` supplies the org for a user-key delete (``flash env delete``). The internal key is
    org-agnostic, so for the web UI delete the caller passes the authenticated user's ``org_id``
    explicitly. We prefer the key's own org and only fall back to the supplied one — so a user
    key never honors a caller-supplied override and can't drop another org's row.
    """
    resolved_org_id = org_id_of(key) or str(org_id or "").strip()
    if not resolved_org_id:
        return False

    return delete_internal_json(
        _PATH,
        {"orgId": resolved_org_id, "slug": slug},
        subject=f"record deleted environment {slug}",
        logger=_LOG,
        urlopen=urllib.request.urlopen,
    )


def record_environment_use(*, slug: str, run_id: str, key: dict) -> bool:
    org_id = org_id_of(key)
    if not org_id:
        return False
    body = {"orgId": org_id, "slug": slug, "runId": run_id}
    return _post(_USE_PATH, body, subject=f"record environment use {slug} for run {run_id}")
