"""Thin RunPod REST client (no SDK state): endpoints, queue jobs, health.

Used by the run supervisor and endpoint GC so that a *fresh process* can
reattach to / clean up after any run using only the persisted ids + RUNPOD_API_KEY —
independent of the Flash SDK's local resource registry (which is per-directory,
whole-dict, last-writer-wins and therefore unreliable across processes).
"""

from __future__ import annotations

import hashlib
from typing import Any

from flash.providers._http import RestClient, is_not_found
from flash.providers.runpod import keys as _keys

REST_BASE = "https://rest.runpod.io/v1"
QUEUE_BASE = "https://api.runpod.ai/v2"


def key_fingerprint(key: str) -> str:
    """A stable, NON-secret identifier for a pool account key — safe to log, store, and return.

    ``RUNPOD_API_KEY`` is a secret, so the raw key must never escape this module (a stray log of a
    ``{key: ...}`` dict, an exception's ``args``, or a metrics tag would leak a live credential).
    Callers that need to act account-scoped (the idle reaper) hold this opaque fingerprint instead
    and pass it back to ``*_for_fingerprint`` helpers, which resolve it to the real key internally.
    A truncated SHA-256 is collision-safe for a handful of pool keys and reveals nothing about the
    secret."""
    return "rpk-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _key_for_fingerprint(fingerprint: str) -> str:
    """Resolve a ``key_fingerprint`` back to its raw pool key (kept inside this module)."""
    pool = _keys.keys()
    for key in pool:
        if key_fingerprint(key) == fingerprint:
            return key
    raise RunpodApiError(f"no RunPod pool key matches fingerprint {fingerprint}")


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


def list_endpoints() -> list[dict]:
    # ``RUNPOD_API_KEY`` may be a comma-separated pool of per-account keys. RunPod
    # endpoints are account-scoped: a plain request_with_retries() call stops at the
    # first key that succeeds and returns only *that* account's endpoints. Idle-sweep
    # and slot-reconcile need the full fleet across every account in the pool, so we
    # query each key independently (with per-key retries) and aggregate.
    #
    # Raises on any per-key failure so callers that treat an empty result as "confirmed
    # absent" (teardown, slot-reconcile) don't act on an incomplete view. Both
    # sweep_idle_endpoints() and the slot reconcile already catch and skip on exception.
    pool = _keys.keys()
    if not pool:
        # No RUNPOD_API_KEY at all: an empty `pool` would make this return [] WITHOUT a single
        # authenticated call, and callers read [] as "the fleet is empty / confirmed absent" and may
        # act on that (teardown, slot-reconcile). Fail loud instead — matching the old single-call
        # request_with_retries() behavior, which raised on a missing key.
        raise RunpodApiError(
            "RUNPOD_API_KEY is not set; refusing to report an empty endpoint fleet"
        )
    all_endpoints: list[dict] = []
    for key in pool:
        out = _CLIENT.request_with_retries_for_key(key, f"{REST_BASE}/endpoints", retries=2)
        if not isinstance(out, list):
            # A 200 whose body isn't the expected list is NOT an empty account — silently skipping it
            # (the old behavior) yields a partial fleet view that callers trust as complete. Raise so
            # the per-key failure surfaces, consistent with this function's "fail, don't under-report"
            # contract above.
            raise RunpodApiError(
                f"unexpected /endpoints response for a pool key (got {type(out).__name__}, want list)"
            )
        all_endpoints.extend(out)
    return all_endpoints


def list_endpoints_by_key() -> tuple[dict[str, list[dict]], list[str]]:
    """Best-effort per-account endpoint listing for the idle reaper.

    Unlike ``list_endpoints()`` — which is all-or-nothing: one pool key's failure aborts
    the whole call so teardown / slot-reconcile never act on a partial fleet — this returns
    ``({key_fingerprint: [endpoints]}, [failed_fingerprints])``: the endpoints for every pool
    account that DID respond, plus the fingerprints of the keys whose ``/endpoints`` call failed
    this cycle.

    Accounts are identified by their NON-secret ``key_fingerprint`` — never the raw
    ``RUNPOD_API_KEY`` — so the return value is safe to log and to persist in the reaper's idle
    grace timers without leaking a live credential. The reaper passes a fingerprint back to
    ``endpoint_health_for_fingerprint`` / ``delete_endpoint_for_fingerprint`` to act account-scoped
    (no 404-failover waterfall); the raw key is resolved internally and never leaves this module.

    The reaper reaps the accounts it can see and leaves the rest for the next sweep, so one
    flaky/expired key can't starve cleanup of every OTHER account. (``list_endpoints``'s
    abort-on-any-failure was exactly that footgun: a single rejected or rate-limited pool
    key silently no-op'd the whole sweep, and idle orphans on healthy accounts piled up.)
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
            # This account is unreachable this cycle (rejected key, credit/quota, or a
            # transient blip that outlasted the per-key retries). Record its fingerprint so the
            # caller can WARN and preserve its in-flight idle grace, and move on to the others.
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
        # An already-gone endpoint is a clean teardown, not a failure: a 404 (or a body
        # saying the endpoint "does not exist") means the desired end state — no such
        # endpoint — already holds. Reporting False here makes undeploy_adapter surface a
        # misleading "may still be running" 502 for something that's provably gone.
        return _is_not_found(e)


def delete_endpoint_for_key(endpoint_id: str, key: str) -> bool:
    """Delete an endpoint using a SPECIFIC pool account's key (no failover waterfall).

    The idle reaper already knows which account owns each endpoint (it listed per key), so
    it deletes with that exact key — cheaper than the pool failover, and it avoids the
    waterfall masking a real failure as a 404 'already gone' on the wrong account.
    """
    try:
        _CLIENT.request_with_retries_for_key(
            key, f"{REST_BASE}/endpoints/{endpoint_id}", method="DELETE", retries=2
        )
        return True
    except RunpodApiError as e:
        return _is_not_found(e)


def _is_not_found(err: RunpodApiError) -> bool:
    """True only when a RunpodApiError represents a genuine 404 (endpoint already gone).

    Delegates to the shared ``_http.is_not_found``: the original urllib HTTPError is chained as
    ``__cause__`` for every fast-failed 4xx (``raise ... from e``), so the status code is
    authoritative when a cause is present — a 404 is "already gone", anything else (403/401/5xx) is a
    real failure and must NOT be swallowed (a body that merely *mentions* "does not exist" on a 403 is
    still a 403). The no-cause fallback (e.g. the "failed after N attempts" path) matches only an
    unambiguous ``HTTP 404`` TOKEN, never a bare substring (so ``HTTP 4040`` can't be misread).
    """
    return is_not_found(err)


def endpoint_health(endpoint_id: str) -> dict:
    return request_with_retries(f"{QUEUE_BASE}/{endpoint_id}/health")


def endpoint_health_for_key(endpoint_id: str, key: str) -> dict:
    """Endpoint health via a SPECIFIC pool account's key (no failover waterfall) — the
    reaper's account-scoped counterpart of ``endpoint_health``."""
    return _CLIENT.request_with_retries_for_key(key, f"{QUEUE_BASE}/{endpoint_id}/health")


def delete_endpoint_for_fingerprint(endpoint_id: str, fingerprint: str) -> bool:
    """``delete_endpoint_for_key`` addressed by a non-secret ``key_fingerprint`` (resolved to the
    raw key inside this module) — the reaper holds only fingerprints, never the credential."""
    return delete_endpoint_for_key(endpoint_id, _key_for_fingerprint(fingerprint))


def endpoint_health_for_fingerprint(endpoint_id: str, fingerprint: str) -> dict:
    """``endpoint_health_for_key`` addressed by a non-secret ``key_fingerprint`` (resolved to the
    raw key inside this module) — the reaper holds only fingerprints, never the credential."""
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


# Realized billing (COGS) -- what RunPod actually charged, for estimator accuracy.
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
