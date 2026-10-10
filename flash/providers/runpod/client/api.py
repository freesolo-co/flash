"""Thin RunPod REST client (no SDK state): endpoints, queue jobs, health."""

from __future__ import annotations

import hashlib
import time
import urllib.error
from typing import Any

from flash._internal.logging import get_logger
from flash.providers._lifecycle.net.deadline import remaining_seconds
from flash.providers._lifecycle.net.http import RestClient, is_not_found
from flash.providers.runpod.client import auth as _keys

logger = get_logger(__name__)

REST_BASE = "https://rest.runpod.io/v1"
QUEUE_BASE = "https://api.runpod.ai/v2"


def key_fingerprint(key: str) -> str:
    """Stable non-secret owner identity for a pool key; safe to log, never the raw credential."""
    return "rpk-" + hashlib.sha256(key.encode()).hexdigest()


def _is_valid_key_fingerprint(fingerprint: object) -> bool:
    return (
        isinstance(fingerprint, str)
        and len(fingerprint) == 68
        and fingerprint.startswith("rpk-")
        and all(char in "0123456789abcdef" for char in fingerprint[4:])
    )


def _is_prefix_key_fingerprint(fingerprint: object) -> bool:
    """The 12-hex-digit prefix form, which is what DEPLOYED `dev` persists right now.

    Deliberately not called "legacy": `dev`'s `key_fingerprint` is
    `sha256(...).hexdigest()[:12]` today, so every endpoint any currently deployed release
    creates carries this shape. Only this branch widened it to the full digest. Naming it
    legacy already caused one wrong deletion of the resolver below (it was removed as dead
    compatibility, then restored a day later) -- and a RunPod endpoint whose owner cannot be
    resolved bills forever with nothing able to tear it down.
    """
    return (
        isinstance(fingerprint, str)
        and len(fingerprint) == 16
        and fingerprint.startswith("rpk-")
        and all(char in "0123456789abcdef" for char in fingerprint[4:])
    )


def _unique_matching_key(matches: list[str], message: str) -> str:
    """The sole distinct credential among fingerprint matches.

    Uniqueness is counted over distinct key VALUES, not pool entries: an operator who repeats
    the same credential in the comma-separated ``RUNPOD_API_KEY`` still has unambiguous
    ownership, and refusing to resolve it would strand submission, polling, cancellation and
    endpoint deletion on a benign configuration typo.
    """
    distinct = set(matches)
    if len(distinct) != 1:
        raise RunpodApiError(message)
    return next(iter(distinct))


def _key_for_fingerprint(fingerprint: str) -> str:
    """Resolve a full key fingerprint back to its unique raw pool key."""
    if not _is_valid_key_fingerprint(fingerprint):
        raise RunpodApiError("persisted RunPod key fingerprint is invalid")
    configured_keys = _keys.keys()
    matches = [key for key in configured_keys if key_fingerprint(key) == fingerprint]
    return _unique_matching_key(
        matches, "expected exactly one RunPod pool key for the persisted fingerprint"
    )


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
    deadline_at: float | None = None,
) -> Any:
    """REST call hardened against transient network/5xx blips (jittered backoff)."""
    return _CLIENT.request_with_retries(
        url,
        method=method,
        body=body,
        retries=retries,
        base_delay=base_delay,
        deadline_at=deadline_at,
    )


def list_endpoints(*, deadline_at: float | None = None) -> list[dict]:
    # Queries all pool accounts and aggregates; raises on any per-key failure so callers
    # never act on a partial fleet view.
    pool = _keys.keys()
    if not pool:
        raise RunpodApiError(
            "RUNPOD_API_KEY is not set; refusing to report an empty endpoint fleet"
        )
    all_endpoints: list[dict] = []
    for key in pool:
        out = _CLIENT.request_with_retries_for_key(
            key,
            f"{REST_BASE}/endpoints",
            retries=2,
            deadline_at=deadline_at,
        )
        if not isinstance(out, list):
            raise RunpodApiError(
                f"unexpected /endpoints response for a pool key (got {type(out).__name__}, want list)"
            )
        all_endpoints.extend(out)
    return all_endpoints


def _list_endpoints_for_key(
    key: str,
    *,
    deadline_at: float | None = None,
) -> list[dict]:
    out = _CLIENT.request_with_retries_for_key(
        key,
        f"{REST_BASE}/endpoints",
        retries=2,
        deadline_at=deadline_at,
    )
    if not isinstance(out, list):
        raise RunpodApiError(
            f"unexpected /endpoints response for a pool key (got {type(out).__name__}, want list)"
        )
    return out


def resolve_prefix_key_fingerprint(endpoint_id: str, fingerprint: str) -> str:
    """Widen a persisted prefix only after its sole matching key proves endpoint ownership.

    Required by CURRENTLY DEPLOYED releases, not by history: see `_is_prefix_key_fingerprint`.
    """
    if not _is_prefix_key_fingerprint(fingerprint):
        raise RunpodApiError("persisted RunPod key fingerprint prefix is invalid")
    configured_keys = _keys.keys()
    matches = [
        (key, full_fingerprint)
        for key in configured_keys
        if (full_fingerprint := key_fingerprint(key)).startswith(fingerprint)
    ]
    key = _unique_matching_key(
        [match_key for match_key, _ in matches],
        "expected exactly one RunPod pool key matching the persisted fingerprint prefix",
    )
    full_fingerprint = key_fingerprint(key)
    try:
        endpoints = _list_endpoints_for_key(key)
    except Exception:
        raise RunpodApiError(
            f"runpod endpoint ownership lookup failed for {endpoint_id}; owner unconfirmed"
        ) from None
    if not any(
        isinstance(endpoint, dict) and endpoint.get("id") == endpoint_id for endpoint in endpoints
    ):
        # absence from this key's listing is not proof of foreign ownership: a process that died
        # between deleting the endpoint and clearing its cleanup record leaves exactly this state,
        # and refusing the upgrade would strand that record forever. but it is not proof of
        # deletion either, and neither available signal settles it alone -- a 404 under the matching
        # key means "invisible to this credential", which RunPod also answers for an endpoint alive
        # under another account, while the pool listing cannot see an owner outside the pool at all.
        # so require BOTH to agree it is gone: binding a record to the wrong credential is worse
        # than stranding it, because teardown would then read 404, report success, and leave the
        # real endpoint billing.
        _confirm_deleted(endpoint_id, full_fingerprint)
    return full_fingerprint


def _confirm_deleted(endpoint_id: str, fingerprint: str) -> None:
    """Raise unless both the pool-wide listing and the owner's own lookup agree it is gone."""
    try:
        fleet = list_endpoints()
    except Exception:
        raise RunpodApiError(
            f"runpod endpoint ownership lookup failed for {endpoint_id}; owner unconfirmed"
        ) from None
    foreign = f"runpod endpoint {endpoint_id} is not owned by the fingerprint prefix match"
    if any(isinstance(endpoint, dict) and endpoint.get("id") == endpoint_id for endpoint in fleet):
        raise RunpodApiError(foreign)
    try:
        # raises unless the lookup 404s, so a still-live endpoint can never read as deleted here.
        absent = endpoint_absent_for_fingerprint(endpoint_id, fingerprint)
    except RunpodApiError:
        raise RunpodApiError(foreign) from None
    if not absent:
        raise RunpodApiError(foreign)


def list_endpoints_by_key(
    *,
    deadline_at: float | None = None,
) -> tuple[dict[str, list[dict]], list[str]]:
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
            by_fingerprint[fp] = _list_endpoints_for_key(key, deadline_at=deadline_at)
        except RunpodApiError:
            failed.append(fp)
    return by_fingerprint, failed


def delete_endpoint_for_key(endpoint_id: str, key: str) -> bool:
    """Delete using a specific pool key (no failover waterfall — avoids masking failures)."""
    try:
        _CLIENT.request_with_retries_for_key(
            key, f"{REST_BASE}/endpoints/{endpoint_id}", method="DELETE", retries=2
        )
        return True
    except RunpodApiError as e:
        return is_not_found(e)


def endpoint_health_for_key(
    endpoint_id: str,
    key: str,
    *,
    deadline_at: float | None = None,
) -> dict:
    """Endpoint health via a specific pool key (no failover waterfall)."""
    return _CLIENT.request_with_retries_for_key(
        key,
        f"{QUEUE_BASE}/{endpoint_id}/health",
        deadline_at=deadline_at,
    )


def delete_endpoint_for_fingerprint(endpoint_id: str, fingerprint: str) -> bool:
    """delete_endpoint_for_key addressed by fingerprint; raw key resolved internally."""
    return delete_endpoint_for_key(endpoint_id, _key_for_fingerprint(fingerprint))


def endpoint_absent_for_fingerprint(endpoint_id: str, fingerprint: str) -> bool:
    """Confirm absence only from an exact owner-authenticated endpoint lookup returning 404."""
    key = _key_for_fingerprint(fingerprint)
    try:
        _CLIENT.request_with_retries_for_key(
            key,
            f"{REST_BASE}/endpoints/{endpoint_id}",
            retries=2,
        )
    except Exception as exc:
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, urllib.error.HTTPError) and cause.code == 404:
            return True
        raise RunpodApiError(
            f"runpod endpoint lookup failed for {endpoint_id}; cleanup unconfirmed"
        ) from None
    raise RunpodApiError(f"runpod endpoint {endpoint_id} still exists; cleanup unconfirmed")


def endpoint_health_for_fingerprint(
    endpoint_id: str,
    fingerprint: str,
    *,
    deadline_at: float | None = None,
) -> dict:
    """endpoint_health_for_key addressed by fingerprint; raw key resolved internally."""
    return endpoint_health_for_key(
        endpoint_id,
        _key_for_fingerprint(fingerprint),
        deadline_at=deadline_at,
    )


def grow_network_volumes_for_key(
    key: str,
    wanted: dict[str, int],
    *,
    deadline_at: float | None = None,
) -> dict[str, int]:
    """Raise every already-provisioned volume in ``wanted`` ({name: gb}) to its target size.

    Creating a NetworkVolume only sizes it on the CREATE. The SDK matches an existing volume by
    name+datacenter and returns it untouched, so a volume provisioned at an older, smaller size stays
    at that size forever and a later "size" bump is silently a no-op: the attach succeeds and the
    download then fails with "Disk quota exceeded". Growing is the only way to reconcile the fleet.

    Returns {name: new_size} for the volumes actually grown. Volumes already at or above target are
    left alone; RunPod rejects a shrink, so only under-sized ones are touched.

    Volumes are independent, so one that cannot be grown -- concurrently deleted, momentarily
    unmodifiable -- only skips itself. Aborting the loop would leave every later datacenter's volume
    unreconciled, and a run placed in one of those would still hit "Disk quota exceeded".

    Skipping itself is not enough on its own, though: a PATCH that times out or 5xxes spends real
    time inside its own retries, and handing every PATCH the same shared deadline lets the first
    under-sized volume consume all of it. Every later one then fails its deadline check immediately
    and the loop reconciles nothing -- the same fleet-wide gap as aborting, reached by a slower
    route. So each PATCH gets an equal share of what is LEFT, recomputed per volume: a stalling
    volume can spend only its own share, and time a fast one does not use flows to the volumes
    behind it.
    """
    out = _CLIENT.request_with_retries_for_key(
        key, f"{REST_BASE}/networkvolumes", retries=2, deadline_at=deadline_at
    )
    vols = out if isinstance(out, list) else (out or {}).get("networkVolumes", []) or []
    grown: dict[str, int] = {}
    pending = []
    for vol in vols:
        name, vol_id = vol.get("name"), vol.get("id")
        target = wanted.get(name)
        if not name or not vol_id or target is None:
            continue
        try:
            current = int(vol.get("size") or 0)
        except (TypeError, ValueError):
            continue
        if current >= int(target):
            continue  # already at or above target; RunPod rejects a shrink
        pending.append((name, vol_id, int(target)))
    for position, (name, vol_id, target) in enumerate(pending):
        patch_deadline = deadline_at
        if deadline_at is not None:
            left = remaining_seconds(deadline_at)
            if left <= 0:
                logger.warning("weight cache: out of time before %s; leaving it as-is", name)
                continue
            patch_deadline = time.time() + left / max(1, len(pending) - position)
        try:
            _CLIENT.request_with_retries_for_key(
                key,
                f"{REST_BASE}/networkvolumes/{vol_id}",
                method="PATCH",
                body={"size": target},
                retries=2,
                deadline_at=patch_deadline,
            )
        except Exception as exc:
            logger.warning("weight cache: could not grow %s (%s); leaving it as-is", name, exc)
            continue
        grown[name] = target
    return grown


def submit_job(
    endpoint_id: str,
    input_payload: dict,
    *,
    key_fingerprint: str,
    deadline_at: float,
) -> str:
    """POST /run through the endpoint's owning account and return the job id."""
    out = _CLIENT.request_with_retries_for_key(
        _key_for_fingerprint(key_fingerprint),
        f"{QUEUE_BASE}/{endpoint_id}/run",
        method="POST",
        body={"input": input_payload},
        retries=0,
        deadline_at=deadline_at,
    )
    job_id = out.get("id") if isinstance(out, dict) else None
    if not isinstance(job_id, str) or not job_id:
        raise RunpodApiError("submit_job: response did not contain a valid job id")
    return job_id


def job_status(
    endpoint_id: str,
    job_id: str,
    *,
    key_fingerprint: str,
    deadline_at: float | None = None,
) -> dict:
    """GET /status/<job_id> through the endpoint's owning account."""
    return _CLIENT.request_with_retries_for_key(
        _key_for_fingerprint(key_fingerprint),
        f"{QUEUE_BASE}/{endpoint_id}/status/{job_id}",
        deadline_at=deadline_at,
    )


def cancel_job(endpoint_id: str, job_id: str, *, key_fingerprint: str) -> dict:
    return _CLIENT.request_with_retries_for_key(
        _key_for_fingerprint(key_fingerprint),
        f"{QUEUE_BASE}/{endpoint_id}/cancel/{job_id}",
        method="POST",
        retries=2,
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
