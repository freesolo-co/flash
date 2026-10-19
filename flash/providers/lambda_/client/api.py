"""Thin Lambda Cloud REST client (no SDK state): instance-types + instance lifecycle.

Two gotchas: Cloudflare 403s the stdlib UA (use real UA via ``extra_headers``); launch is
NON-IDEMPOTENT so it is never retried (blind retry = double-provision + double-bill).
"""

from __future__ import annotations

import time
from typing import Any

from flash._internal.logging import get_logger
from flash.providers._lifecycle.net.deadline import require_create_allowance, require_deadline_at
from flash.providers._lifecycle.net.http import RestClient, is_not_found

logger = get_logger(__name__)

LAMBDA_BASE = "https://cloud.lambdalabs.com/api/v1"
# Cloudflare rejects Python-urllib UA with 403 — must use a real UA.
_USER_AGENT = "flash-lambda/1.0 (+https://freesolo.co)"


class LambdaApiError(RuntimeError):
    pass


_CLIENT = RestClient(
    env_var="LAMBDA_API_KEY",
    error_cls=LambdaApiError,
    base_url=LAMBDA_BASE,
    extra_headers={"User-Agent": _USER_AGENT},
)


def request_with_retries(
    path: str,
    method: str = "GET",
    body: dict | None = None,
    retries: int = 4,
    base_delay: float = 2.0,
    deadline_at: float | None = None,
) -> Any:
    """REST call hardened against transient network/5xx blips (jittered backoff)."""
    return _CLIENT.request_with_retries(
        path,
        method=method,
        body=body,
        retries=retries,
        base_delay=base_delay,
        deadline_at=deadline_at,
    )


def _data(out: Any) -> Any:
    """Unwrap Lambda's ``{"data": ...}`` envelope (every 2xx response uses it)."""
    if isinstance(out, dict) and "data" in out:
        return out["data"]
    return out


_TYPES_TTL_S = 45.0
_types_cache: dict[str, Any] = {"ts": 0.0, "data": None}


def list_instance_types(
    force: bool = False,
    *,
    deadline_at: float | None = None,
) -> dict[str, dict]:
    """Map of ``instance_type_name -> {instance_type, regions_with_capacity_available}`` (cached)."""
    now = time.time()
    if not force and _types_cache["data"] is not None and now - _types_cache["ts"] < _TYPES_TTL_S:
        return _types_cache["data"]
    out = _data(request_with_retries("/instance-types", deadline_at=deadline_at))
    if not isinstance(out, dict):
        raise LambdaApiError(f"unexpected /instance-types response: {repr(out)[:1000]}")
    _types_cache.update(ts=now, data=out)
    return out


def regions_with_capacity(
    instance_type: str,
    force: bool = False,
    *,
    deadline_at: float | None = None,
) -> list[str]:
    """Region names that currently have capacity for ``instance_type``."""
    info = list_instance_types(force=force, deadline_at=deadline_at).get(instance_type) or {}
    return [r.get("name") for r in info.get("regions_with_capacity_available", []) if r.get("name")]


def all_regions(
    force: bool = False,
    *,
    deadline_at: float | None = None,
) -> list[str]:
    """Union of all regions with capacity across instance types (Lambda has no standalone region API)."""
    regions: set[str] = set()
    for info in list_instance_types(force=force, deadline_at=deadline_at).values():
        for r in (info or {}).get("regions_with_capacity_available", []):
            if r.get("name"):
                regions.add(r["name"])
    return sorted(regions)


def instance_type_price_usd_hr(
    instance_type: str,
    *,
    deadline_at: float | None = None,
) -> float | None:
    """Live $/hr for a Lambda instance type (``price_cents_per_hour`` / 100), or None."""
    info = (list_instance_types(deadline_at=deadline_at).get(instance_type) or {}).get(
        "instance_type"
    ) or {}
    cents = info.get("price_cents_per_hour")
    return float(cents) / 100.0 if cents else None


def list_ssh_keys(*, deadline_at: float | None = None) -> list[dict]:
    out = _data(request_with_retries("/ssh-keys", deadline_at=deadline_at))
    return out if isinstance(out, list) else []


def launch_instance(
    *,
    region_name: str,
    instance_type_name: str,
    ssh_key_names: list[str],
    name: str,
    user_data: str,
    file_system_names: list[str] | None = None,
    deadline_at: float | None = None,
) -> str:
    """Launch one instance -> its id. NON-IDEMPOTENT: never retried (blind retry = double-provision)."""
    deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    if deadline is not None:
        require_create_allowance(deadline)
    body = {
        "region_name": region_name,
        "instance_type_name": instance_type_name,
        "ssh_key_names": list(ssh_key_names),
        "name": name,
        "quantity": 1,
        "user_data": user_data,
    }
    if file_system_names:
        body["file_system_names"] = list(file_system_names)
    out = _data(
        request_with_retries(
            "/instance-operations/launch",
            method="POST",
            body=body,
            retries=0,
            **({} if deadline is None else {"deadline_at": deadline}),
        )
    )
    ids = out.get("instance_ids") if isinstance(out, dict) else None
    instance_id = ids[0] if isinstance(ids, list) and len(ids) == 1 else None
    if not isinstance(instance_id, str) or not instance_id:
        raise LambdaApiError(
            f"launch({instance_type_name}@{region_name}) returned an invalid instance identity"
        )
    return instance_id


# IMPORTANT: Lambda filesystem paths are ASYMMETRIC by design — LIST uses /file-systems (hyphenated),
# CREATE/DELETE use /filesystems (no hyphen). DO NOT unify them; /file-systems 404s for write ops.
def list_filesystems(*, deadline_at: float | None = None) -> list[dict]:
    """All filesystems on the account: ``[{id, name, mount_point, region:{name}, is_in_use}, ...]``."""
    out = _data(request_with_retries("/file-systems", deadline_at=deadline_at))
    return out if isinstance(out, list) else []


def create_filesystem(name: str, region_name: str, *, deadline_at: float) -> dict:
    """Issue one deadline-gated filesystem create request and return its object."""
    deadline = require_deadline_at(deadline_at)
    require_create_allowance(deadline)
    out = _data(
        request_with_retries(
            "/filesystems",
            method="POST",
            body={"name": name, "region": region_name},
            retries=0,
            deadline_at=deadline,
        )
    )
    if not isinstance(out, dict) or not out:
        raise LambdaApiError("filesystem create returned no verifiable object")
    return out


def delete_filesystem(
    filesystem_id: str,
    *,
    deadline_at: float | None = None,
) -> bool:
    """Delete a filesystem by id (best-effort). Returns True if the request didn't raise."""
    try:
        request_with_retries(
            f"/filesystems/{filesystem_id}",
            method="DELETE",
            retries=2,
            **({} if deadline_at is None else {"deadline_at": deadline_at}),
        )
        return True
    except Exception as exc:
        logger.warning(
            "lambda delete_filesystem(%s) failed (%s)",
            filesystem_id,
            type(exc).__name__,
        )
        return False


def _matching_filesystems(filesystems: list[dict], name: str, region_name: str) -> list[dict]:
    return [
        fs
        for fs in filesystems
        if fs.get("name") == name and (fs.get("region") or {}).get("name") == region_name
    ]


def ensure_filesystem(name: str, region_name: str, *, deadline_at: float) -> str:
    """Reuse an exact filesystem or reconcile one ambiguous non-idempotent create."""
    deadline = require_deadline_at(deadline_at)
    matches = _matching_filesystems(
        list_filesystems(deadline_at=deadline),
        name,
        region_name,
    )
    if len(matches) > 1:
        raise LambdaApiError("filesystem identity is ambiguous")
    if matches:
        return matches[0].get("mount_point") or f"/lambda/nfs/{name}"
    try:
        created = create_filesystem(name, region_name, deadline_at=deadline)
    except Exception:
        matches = _matching_filesystems(
            list_filesystems(deadline_at=deadline),
            name,
            region_name,
        )
        if len(matches) != 1:
            raise LambdaApiError("ambiguous filesystem create could not be reconciled") from None
        return matches[0].get("mount_point") or f"/lambda/nfs/{name}"
    return created.get("mount_point") or f"/lambda/nfs/{name}"


def get_instance(
    instance_id: str,
    *,
    deadline_at: float | None = None,
    strict: bool = False,
) -> dict | None:
    """Instance detail dict, or None once it no longer exists (terminated)."""
    try:
        out = request_with_retries(
            f"/instances/{instance_id}",
            deadline_at=deadline_at,
        )
    except LambdaApiError as e:
        if is_not_found(e):
            return None
        raise
    data = _data(out)
    if isinstance(data, dict):
        return data
    if strict:
        raise LambdaApiError("lambda instance lookup returned a malformed response")
    return None


def list_instances(
    *,
    deadline_at: float | None = None,
    strict: bool = False,
) -> list[dict]:
    out = _data(request_with_retries("/instances", deadline_at=deadline_at))
    if isinstance(out, list):
        return out
    if strict:
        raise LambdaApiError("lambda instance listing returned a malformed response")
    return []


def terminate_instances(
    instance_ids: list[str],
    *,
    deadline_at: float | None = None,
) -> list[str]:
    """Terminate instances; return ids that succeeded. Per-id isolation: Lambda's batch endpoint
    rejects the whole request if any id is invalid, so one stale id would leak billing for the rest."""
    deleted: list[str] = []
    for iid in [str(i) for i in instance_ids if i]:
        try:
            request_with_retries(
                "/instance-operations/terminate",
                method="POST",
                body={"instance_ids": [iid]},
                retries=2,
                **({} if deadline_at is None else {"deadline_at": deadline_at}),
            )
            deleted.append(iid)
        except Exception as exc:
            logger.warning("lambda terminate(%s) failed (%s)", iid, type(exc).__name__)
    return deleted


def terminate_instance_confirmed(instance_id: str) -> None:
    """Terminate one exact instance and prove both acceptance and disappearance.

    This cleanup intentionally has no run-deadline dependency: a box that outlived its run must still
    be removable. Any unconfirmed step raises so replacement provisioning keeps the persisted handle.
    """
    iid = str(instance_id).strip()
    if not iid:
        raise LambdaApiError("lambda instance teardown identity is invalid")
    deleted = terminate_instances([iid])
    if deleted != [iid]:
        raise LambdaApiError(f"lambda terminate({iid}) was not confirmed")
    if get_instance(iid, strict=True) is not None:
        raise LambdaApiError(f"lambda instance {iid} remains after confirmed termination")
