"""Thin RunPod REST client (no SDK state): endpoints, queue jobs, health."""

from __future__ import annotations

import hashlib
from typing import Any

from flash.providers._http import RestClient, is_not_found
from flash.providers.runpod import keys as _keys

REST_BASE = "https://rest.runpod.io/v1"
QUEUE_BASE = "https://api.runpod.ai/v2"


def key_fingerprint(key: str) -> str:
    """Stable non-secret identifier for a pool key — safe to log; never the raw credential."""
    return "rpk-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _key_for_fingerprint(fingerprint: str) -> str:
    """Resolve a key_fingerprint back to its raw pool key."""
    pool = _keys.keys()
    for key in pool:
        if key_fingerprint(key) == fingerprint:
            return key
    raise RunpodApiError(f"no RunPod pool key matches fingerprint {fingerprint}")


class RunpodApiError(RuntimeError):
    pass


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


def list_endpoints() -> list[dict]:
    # Queries all pool accounts and aggregates; raises on any per-key failure so callers
    # never act on a partial fleet view.
    pool = _keys.keys()
    if not pool:
        raise RunpodApiError(
            "RUNPOD_API_KEY is not set; refusing to report an empty endpoint fleet"
        )
    all_endpoints: list[dict] = []
    for key in pool:
        out = _CLIENT.request_with_retries_for_key(key, f"{REST_BASE}/endpoints", retries=2)
        if not isinstance(out, list):
            raise RunpodApiError(
                f"unexpected /endpoints response for a pool key (got {type(out).__name__}, want list)"
            )
        all_endpoints.extend(out)
    return all_endpoints


def list_endpoints_by_key() -> tuple[dict[str, list[dict]], list[str]]:
    """Best-effort per-account endpoint listing for the idle reaper.

    Returns ({key_fingerprint: [endpoints]}, [failed_fingerprints]). One flaky account
    can't abort cleanup of healthy accounts (unlike list_endpoints which is all-or-nothing).
    Fingerprints are used instead of raw keys so the return value is safe to log/persist.
    """
    pool = _keys.keys()
    if not pool:
        raise RunpodApiError(
            "RUNPOD_API_KEY is not set; refusing to report an empty endpoint fleet"
        )
    by_fingerprint: dict[str, list[dict]] = {}
    failed: list[str] = []
    for key in pool:
        fp = key_fingerprint(key)
        try:
            out = _CLIENT.request_with_retries_for_key(key, f"{REST_BASE}/endpoints", retries=2)
        except RunpodApiError:
            failed.append(fp)
            continue
        if not isinstance(out, list):
            failed.append(fp)
            continue
        by_fingerprint[fp] = out
    return by_fingerprint, failed


def find_endpoints_by_name(substr: str) -> list[dict]:
    return [e for e in list_endpoints() if substr in (e.get("name") or "")]


def delete_endpoint(endpoint_id: str) -> bool:
    try:
        request_with_retries(f"{REST_BASE}/endpoints/{endpoint_id}", method="DELETE", retries=2)
        return True
    except RunpodApiError as e:
        # 404 = already gone = clean teardown; anything else is a real failure.
        return _is_not_found(e)


def delete_endpoint_for_key(endpoint_id: str, key: str) -> bool:
    """Delete using a specific pool key (no failover waterfall — avoids masking failures)."""
    try:
        _CLIENT.request_with_retries_for_key(
            key, f"{REST_BASE}/endpoints/{endpoint_id}", method="DELETE", retries=2
        )
        return True
    except RunpodApiError as e:
        return _is_not_found(e)


def _is_not_found(err: RunpodApiError) -> bool:
    """True when the error represents a genuine 404 (endpoint already gone)."""
    return is_not_found(err)


def endpoint_health(endpoint_id: str) -> dict:
    return request_with_retries(f"{QUEUE_BASE}/{endpoint_id}/health")


def endpoint_health_for_key(endpoint_id: str, key: str) -> dict:
    """Endpoint health via a specific pool key (no failover waterfall)."""
    return _CLIENT.request_with_retries_for_key(key, f"{QUEUE_BASE}/{endpoint_id}/health")


def delete_endpoint_for_fingerprint(endpoint_id: str, fingerprint: str) -> bool:
    """delete_endpoint_for_key addressed by fingerprint; raw key resolved internally."""
    return delete_endpoint_for_key(endpoint_id, _key_for_fingerprint(fingerprint))


def endpoint_health_for_fingerprint(endpoint_id: str, fingerprint: str) -> dict:
    """endpoint_health_for_key addressed by fingerprint; raw key resolved internally."""
    return endpoint_health_for_key(endpoint_id, _key_for_fingerprint(fingerprint))


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


def billing_endpoints(
    *,
    start_time: str,
    end_time: str,
    endpoint_id: str | None = None,
    bucket_size: str = "day",
) -> list[dict]:
    """Realized serverless spend per endpoint over [start_time, end_time] (ISO-8601).

    Queries all pool accounts and aggregates; billing is account-scoped so a waterfall
    would silently report $0 for endpoints provisioned on a failover account.
    """
    from urllib.parse import urlencode

    params: dict[str, str] = {
        "startTime": start_time,
        "endTime": end_time,
        "bucketSize": bucket_size,
    }
    if endpoint_id:
        params["endpointId"] = endpoint_id
    url = f"{REST_BASE}/billing/endpoints?{urlencode(params)}"
    pool = _keys.keys()
    if not pool:
        return _billing_rows(request_with_retries(url))
    rows: list[dict] = []
    for key in pool:
        try:
            rows.extend(_billing_rows(_CLIENT.request_with_retries_for_key(key, url, retries=2)))
        except RunpodApiError:
            continue  # foreign/failing account: skip so it can't zero out the owning account's rows
    return rows


def _billing_rows(out) -> list[dict]:
    """Extract billing rows from a RunPod billing response (bare list or nested under data/endpoints/billing)."""
    if isinstance(out, list):
        return out
    if isinstance(out, dict):
        rows = out.get("data") or out.get("endpoints") or out.get("billing")
        return rows if isinstance(rows, list) else []
    return []
