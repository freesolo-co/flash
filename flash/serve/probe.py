"""Dependency-free HTTP probes for a serving backend."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass

from flash.serve.urls import displayable_url, url_origin

HEALTHZ_STARTUP_BUDGET_SECONDS = 90.0
HEALTHZ_RETRY_DELAY_SECONDS = 5.0
# matches the control plane's own per-request ceiling (flash/serve/deploy.py). an authenticated
# probe wakes a scaled-to-zero backend, so a tighter bound here reports "unreachable" for a
# backend the control plane would have waited for and reached.
PROBE_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ProbeResult:
    """One HTTP probe response, including expected non-2xx statuses."""

    status_code: int
    payload: object | None = None
    error: str = ""


def request_json(url: str, headers: dict[str, str], path: str = "/healthz") -> ProbeResult:
    """Get one JSON document without sending serving credentials off-origin."""
    origin = url_origin(url)

    class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            new = super().redirect_request(req, fp, code, msg, hdrs, newurl)
            if new is not None and url_origin(newurl) != origin:
                raise urllib.error.HTTPError(
                    newurl,
                    code,
                    f"{displayable_url(url)} redirected to another origin "
                    f"({displayable_url(newurl)}); refusing to send the serving key there",
                    hdrs,
                    fp,
                )
            return new

    opener = urllib.request.build_opener(_SameOriginRedirect)
    request = urllib.request.Request(f"{url}{path}", headers=headers, method="GET")
    try:
        with opener.open(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
            return ProbeResult(status_code=response.status, payload=json.load(response))
    except urllib.error.HTTPError as exc:
        try:
            return ProbeResult(status_code=exc.code, error=str(exc))
        finally:
            exc.close()


def probe_serving_key(url: str, headers: dict[str, str]) -> ProbeResult:
    """Probe authenticated unknown-adapter semantics without mutating backend state."""
    adapter_id = f"flash-serve-status-probe-{uuid.uuid4().hex}"
    return request_json(url, headers, f"/adapters/{adapter_id}")


def healthz_with_retry(
    url: str,
    budget_s: float = HEALTHZ_STARTUP_BUDGET_SECONDS,
    retry_delay_s: float = HEALTHZ_RETRY_DELAY_SECONDS,
) -> dict | None:
    """Read an advisory health response within a bounded startup window."""
    deadline = time.monotonic() + max(0.0, float(budget_s))
    while True:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=15) as response:
                payload = json.load(response)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            return payload
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(retry_delay_s, remaining))


def redacted_error(error: object, base: str) -> str:
    """Return an error message with credentials from ``base`` removed."""
    text = str(error)
    authority = base.split("://", 1)[-1].split("/", 1)[0]
    userinfo = authority.rsplit("@", 1)[0] if "@" in authority else ""
    secrets = [base, base.rstrip("/"), authority, userinfo, *userinfo.split(":")]
    for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
        replacement = displayable_url(base) if "://" in secret else "(redacted)"
        text = text.replace(secret, replacement)
    return text
