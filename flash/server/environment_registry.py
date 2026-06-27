"""Best-effort reporting of published Flash environments to the Freesolo backend."""

from __future__ import annotations

import contextlib
import logging
import urllib.error
import urllib.request

from ._internal_client import DEFAULT_TIMEOUT_S, build_internal_request, internal_key, org_id_of

_LOG = logging.getLogger("flash.server.environments")
_PATH = "/api/flash/environments/internal"
_USE_PATH = "/api/flash/environments/use/internal"
_DEFAULT_HUB_REPO = "freesolo-co/environment-hub"
_DEFAULT_HUB_REF = "main"


def _post(path: str, body: dict, *, subject: str) -> bool:
    """Best-effort internal POST; returns True on 2xx, False on any failure."""
    key = internal_key()
    if not key:
        return False
    req = build_internal_request(path, body, token=key)
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = exc.read().decode("utf-8", "replace")[:500]
        _LOG.warning("failed to %s: HTTP %s %s", subject, exc.code, detail)
    except (urllib.error.URLError, OSError) as exc:
        _LOG.warning("failed to %s: %s", subject, exc)
    return False


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


def record_environment_use(*, slug: str, run_id: str, key: dict) -> bool:
    org_id = org_id_of(key)
    if not org_id:
        return False
    body = {"orgId": org_id, "slug": slug, "runId": run_id}
    return _post(_USE_PATH, body, subject=f"record environment use {slug} for run {run_id}")
