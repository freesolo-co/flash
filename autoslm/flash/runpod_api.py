"""Thin RunPod REST client (no SDK state): endpoints, queue jobs, health.

Used by the durable-run supervisor and endpoint GC so that a *fresh process* can
reattach to / clean up after any run using only the persisted ids + RUNPOD_API_KEY —
independent of the Flash SDK's local resource registry (which is per-directory,
whole-dict, last-writer-wins and therefore unreliable across processes).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

REST_BASE = "https://rest.runpod.io/v1"
QUEUE_BASE = "https://api.runpod.ai/v2"


class RunpodApiError(RuntimeError):
    pass


def _api_key() -> str:
    # Env-only by design: ~/.autoslm/config.json holds the *AutoSLM* key (client-side),
    # never the RunPod key — the operator sets RUNPOD_API_KEY on the control-plane host.
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RunpodApiError(
            "RUNPOD_API_KEY not configured on the control-plane host (see docs/self-hosting.md)"
        )
    return key


def _request(url: str, method: str = "GET", body: dict | None = None, timeout: float = 30.0):
    req = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def request_with_retries(
    url: str,
    method: str = "GET",
    body: dict | None = None,
    retries: int = 4,
    base_delay: float = 2.0,
) -> Any:
    """REST call hardened against transient network/5xx blips (jittered backoff)."""
    import random

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _request(url, method=method, body=body)
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise RunpodApiError(f"{method} {url} -> HTTP {e.code}: {e.reason}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
        if attempt < retries:
            delay = min(base_delay * (2 ** min(attempt, 6)), 30.0)
            time.sleep(delay * random.uniform(0.7, 1.3))
    raise RunpodApiError(f"{method} {url} failed after {retries + 1} attempts: {last}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def list_endpoints() -> list[dict]:
    out = request_with_retries(f"{REST_BASE}/endpoints")
    return out if isinstance(out, list) else []


def find_endpoints_by_name(substr: str) -> list[dict]:
    return [e for e in list_endpoints() if substr in (e.get("name") or "")]


def delete_endpoint(endpoint_id: str) -> bool:
    try:
        request_with_retries(f"{REST_BASE}/endpoints/{endpoint_id}", method="DELETE", retries=2)
        return True
    except RunpodApiError:
        return False


def endpoint_health(endpoint_id: str) -> dict:
    return request_with_retries(f"{QUEUE_BASE}/{endpoint_id}/health")


# ---------------------------------------------------------------------------
# Queue jobs
# ---------------------------------------------------------------------------
def submit_job(endpoint_id: str, input_payload: dict) -> str:
    """POST /run -> job id (async queue submission)."""
    out = request_with_retries(
        f"{QUEUE_BASE}/{endpoint_id}/run", method="POST", body={"input": input_payload}
    )
    job_id = out.get("id")
    if not job_id:
        raise RunpodApiError(f"submit_job: no job id in response: {out}")
    return job_id


def job_status(endpoint_id: str, job_id: str) -> dict:
    """GET /status/<job_id> -> {status, output?, error?, ...}."""
    return request_with_retries(f"{QUEUE_BASE}/{endpoint_id}/status/{job_id}")


def cancel_job(endpoint_id: str, job_id: str) -> dict:
    return request_with_retries(
        f"{QUEUE_BASE}/{endpoint_id}/cancel/{job_id}", method="POST", retries=2
    )
