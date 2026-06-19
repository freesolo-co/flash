"""Thin RunPod REST client (no SDK state): endpoints, queue jobs, health.

Used by the run supervisor and endpoint GC so that a *fresh process* can
reattach to / clean up after any run using only the persisted ids + RUNPOD_API_KEY —
independent of the Flash SDK's local resource registry (which is per-directory,
whole-dict, last-writer-wins and therefore unreliable across processes).
"""

from __future__ import annotations

import urllib.error
from typing import Any

from flash.providers._http import RestClient

REST_BASE = "https://rest.runpod.io/v1"
QUEUE_BASE = "https://api.runpod.ai/v2"


class RunpodApiError(RuntimeError):
    pass


# Shared urllib client (full-URL form: callers pass absolute REST/QUEUE urls).
# Env-only by design: ~/.flash/config.json holds the *Flash* key (client-side),
# never the RunPod key — the operator sets RUNPOD_API_KEY on the control-plane host.
_CLIENT = RestClient(env_var="RUNPOD_API_KEY", error_cls=RunpodApiError)


def _api_key() -> str:
    return _CLIENT.api_key()


def _request(url: str, method: str = "GET", body: dict | None = None, timeout: float = 30.0):
    return _CLIENT.request(url, method=method, body=body, timeout=timeout)


def request_with_retries(
    url: str,
    method: str = "GET",
    body: dict | None = None,
    retries: int = 4,
    base_delay: float = 2.0,
) -> Any:
    """REST call hardened against transient network/5xx blips (jittered backoff)."""
    return _CLIENT.request_with_retries(
        url, method=method, body=body, retries=retries, base_delay=base_delay
    )


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
    except RunpodApiError as e:
        # An already-gone endpoint is a clean teardown, not a failure: a 404 (or a body
        # saying the endpoint "does not exist") means the desired end state — no such
        # endpoint — already holds. Reporting False here makes undeploy_adapter surface a
        # misleading "may still be running" 502 for something that's provably gone.
        return _is_not_found(e)


def _is_not_found(err: RunpodApiError) -> bool:
    """True only when a RunpodApiError represents a genuine 404 (endpoint already gone).

    request_with_retries chains the original urllib HTTPError as ``__cause__`` for every
    fast-failed 4xx (``raise ... from e``), so the status code is authoritative when a
    cause is present: a 404 is "already gone", anything else (403/401/5xx) is a real
    failure and must NOT be swallowed — a body that merely *mentions* "does not exist" on a
    403 is still a 403. We only fall back to a text match when there is no HTTPError cause
    (e.g. the "failed after N attempts" path), and even then only on an unambiguous 404.
    """
    cause = err.__cause__
    if isinstance(cause, urllib.error.HTTPError):
        return cause.code == 404
    return "http 404" in str(err).lower()


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
