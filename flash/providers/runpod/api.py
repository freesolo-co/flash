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
from flash.providers.runpod import keys as _keys

REST_BASE = "https://rest.runpod.io/v1"
QUEUE_BASE = "https://api.runpod.ai/v2"


class RunpodApiError(RuntimeError):
    pass


# Shared urllib client (full-URL form: callers pass absolute REST/QUEUE urls).
# Env-only by design: ~/.flash/config.json holds the *Flash* key (client-side),
# never the RunPod key — the operator sets RUNPOD_API_KEY on the control-plane host.
#
# ``RUNPOD_API_KEY`` may be a comma-separated pool of per-account keys: the client tries
# them active-account-first per call (``keys.ordered_keys``) and fails over to the next
# account on an auth/quota/not-found error (``keys.is_failover_error``). RunPod endpoints
# are account-scoped, so a single-account op (status/cancel/delete) resolves no matter
# which account a failed-over run was provisioned on. A single key => a pool of one.
_CLIENT = RestClient(
    env_var="RUNPOD_API_KEY",
    error_cls=RunpodApiError,
    keys_provider=_keys.ordered_keys,
    failover_predicate=_keys.is_failover_error,
)


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


# ---------------------------------------------------------------------------
# Realized billing (COGS) -- what RunPod actually charged, for estimator accuracy.
# ---------------------------------------------------------------------------
def billing_endpoints(
    *,
    start_time: str,
    end_time: str,
    endpoint_id: str | None = None,
    bucket_size: str = "day",
) -> list[dict]:
    """Realized serverless spend per endpoint over [start_time, end_time] (ISO-8601).

    GET /v1/billing/endpoints -> records of {endpointId, time, amount (USD), timeBilledMs, ...}.
    RunPod has no per-job cost; the finest realized granularity is per-endpoint per time bucket.
    Flash provisions one endpoint per run, so filtering by ``endpoint_id`` yields that run's
    realized cost even after the endpoint is torn down (billing history survives deletion).
    """
    from urllib.parse import urlencode

    params: dict[str, str] = {
        "startTime": start_time,
        "endTime": end_time,
        "bucketSize": bucket_size,
    }
    if endpoint_id:
        params["endpointId"] = endpoint_id
    out = request_with_retries(f"{REST_BASE}/billing/endpoints?{urlencode(params)}")
    if isinstance(out, list):
        return out
    # Defensive: some RunPod list responses wrap rows under a key.
    if isinstance(out, dict):
        rows = out.get("data") or out.get("endpoints") or out.get("billing")
        return rows if isinstance(rows, list) else []
    return []
