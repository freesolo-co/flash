"""Refuse a serving rollout when the configured backend lacks its required routes.

This read-only deployment preflight uses the backend's published OpenAPI contract.
It does not prove authentication or settlement; the real post-deploy canary still must pass.
Only the standard library is needed before Modal replaces the production router.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

BACKEND_UNAVAILABLE = "backend_contract_unavailable"
BACKEND_INCOMPATIBLE = "backend_serving_contract_missing"
REQUIRED_POST_ROUTES = ("/api/serving/authorize", "/api/billing/serving-usage/durable")
MAX_SCHEMA_BYTES = 4 * 1024 * 1024


def schema_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("backend_url_invalid")
    return base_url.rstrip("/") + "/openapi.json"


def compatible_schema(payload: Any) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("paths"), dict):
        return False
    return all(
        isinstance(payload["paths"].get(path), dict)
        and isinstance(payload["paths"][path].get("post"), dict)
        for path in REQUIRED_POST_ROUTES
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def check_backend(base_url: str) -> str:
    """Return a safe reason code; never log URLs, response bodies or credentials."""
    try:
        request = urllib.request.Request(
            schema_url(base_url), headers={"Accept": "application/json"}
        )
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(request, timeout=15) as response:
            body = response.read(MAX_SCHEMA_BYTES + 1)
        if len(body) > MAX_SCHEMA_BYTES:
            return BACKEND_INCOMPATIBLE
        payload = json.loads(body)
    except (OSError, ValueError, urllib.error.URLError):
        return BACKEND_UNAVAILABLE
    return "" if compatible_schema(payload) else BACKEND_INCOMPATIBLE


def main() -> int:
    reason = check_backend(os.environ.get("PLATFORM_BACKEND_URL", ""))
    if reason:
        print(f"backend preflight failed: {reason}", file=sys.stderr)
        return 1
    print("backend serving contract is present; durable settlement remains unverified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
