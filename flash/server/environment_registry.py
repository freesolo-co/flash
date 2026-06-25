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


def record_published_environment(*, slug: str, name: str, key: dict) -> bool:
    """Persist the environment's Azure-blob pointer in the platform backend.

    Azure Blob (indexed in Azure Postgres) is the source of truth for the package; this metadata
    write mirrors the pointer into the platform DB so the web UI can list Flash environments. It is
    deliberately best-effort and never blocks `flash env push`. The blob pointer is read back from
    the Azure Postgres index just written by `publish_package`.
    """
    internal_key = os.environ.get(INTERNAL_KEY_ENV)
    org_id = str(key.get("org_id") or "").strip()
    if not internal_key or not org_id:
        return False

    # Read the pointer the publish step just indexed. Best-effort: if the index is unreachable we
    # skip the mirror rather than fail the push.
    from .azure_blob import container_name

    blob_container = container_name()
    blob_key = ""
    package_sha256: str | None = None
    try:
        from . import environment_store

        record = environment_store.lookup(slug)
        if record is not None:
            blob_container = record.blob_container
            blob_key = record.blob_key
            package_sha256 = record.package_sha256
    except Exception as exc:
        _LOG.warning("could not read Azure env pointer for %s: %s", slug, exc)
        return False
    if not blob_key:
        _LOG.warning("no Azure env pointer found for %s; skipping platform mirror", slug)
        return False

    body = {
        "orgId": org_id,
        "slug": slug,
        "name": name,
        "blobContainer": blob_container,
        "blobKey": blob_key,
        "packageSha256": package_sha256,
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
