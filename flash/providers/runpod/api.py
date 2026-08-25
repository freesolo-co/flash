"""Thin account-scoped RunPod REST transport for managed Secure Cloud Pods."""

from __future__ import annotations

import hashlib
import time
import urllib.request
from typing import Any

from flash._internal.logging import get_logger
from flash.providers._lifecycle.deadline import remaining_seconds
from flash.providers._lifecycle.http import RestClient
from flash.providers.runpod import auth as _keys

logger = get_logger(__name__)

REST_BASE = "https://rest.runpod.io/v1"
CATALOG_BASE = "https://api.runpod.io/v2/catalog"


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def key_fingerprint(key: str) -> str:
    """Return a stable non-secret account identity for one pool key."""
    return "rpk-" + hashlib.sha256(key.encode()).hexdigest()


def _is_valid_key_fingerprint(fingerprint: object) -> bool:
    return (
        isinstance(fingerprint, str)
        and len(fingerprint) == 68
        and fingerprint.startswith("rpk-")
        and all(char in "0123456789abcdef" for char in fingerprint[4:])
    )


def _unique_matching_key(matches: list[str], message: str) -> str:
    distinct = set(matches)
    if len(distinct) != 1:
        raise RunpodApiError(message)
    return next(iter(distinct))


def _key_for_fingerprint(fingerprint: str) -> str:
    """Resolve a complete persisted account fingerprint to one configured key."""
    if not _is_valid_key_fingerprint(fingerprint):
        raise RunpodApiError("persisted RunPod key fingerprint is invalid")
    configured_keys = _keys.keys
    matches = [key for key in configured_keys() if key_fingerprint(key) == fingerprint]
    return _unique_matching_key(
        matches,
        "expected exactly one RunPod pool key for the persisted fingerprint",
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
    """Issue a shared transport request with bounded retry and account failover."""
    return _CLIENT.request_with_retries(
        url,
        method=method,
        body=body,
        retries=retries,
        base_delay=base_delay,
        deadline_at=deadline_at,
    )


def grow_network_volumes_for_key(
    key: str,
    wanted: dict[str, int],
    *,
    deadline_at: float | None = None,
) -> dict[str, int]:
    """Grow existing account volumes independently, sharing the remaining deadline."""
    out = _CLIENT.request_with_retries_for_key(
        key,
        f"{REST_BASE}/networkvolumes",
        retries=2,
        deadline_at=deadline_at,
    )
    volumes = out if isinstance(out, list) else (out or {}).get("networkVolumes", []) or []
    pending = []
    for volume in volumes:
        name, volume_id = volume.get("name"), volume.get("id")
        target = wanted.get(name)
        if not name or not volume_id or target is None:
            continue
        try:
            current = int(volume.get("size") or 0)
        except (TypeError, ValueError):
            continue
        if current < int(target):
            pending.append((name, volume_id, int(target)))
    grown: dict[str, int] = {}
    for position, (name, volume_id, target) in enumerate(pending):
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
                f"{REST_BASE}/networkvolumes/{volume_id}",
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


from flash.providers.runpod.pod_api import (  # noqa: E402,F401
    RunpodCapacityError,
    RunpodDataCenter,
    RunpodMutationAmbiguous,
    RunpodNetworkVolume,
    RunpodPod,
    RunpodRequestError,
    RunpodSecret,
    _graphql_error_status,
    _mutation_once,
    _parse_data_centers,
    _parse_pod,
    _parse_secret_account,
    _pod_rows,
    create_network_volume_for_fingerprint,
    create_pod_for_fingerprint,
    create_secret_for_fingerprint,
    delete_network_volume_for_fingerprint,
    delete_pod_for_fingerprint,
    delete_secret_for_fingerprint,
    get_pod_for_fingerprint,
    grow_network_volumes_for_fingerprint,
    list_network_volumes_for_fingerprint,
    list_pods_by_key,
    list_pods_for_key,
    list_secrets_for_fingerprint,
    list_storage_datacenters_for_fingerprint,
)
