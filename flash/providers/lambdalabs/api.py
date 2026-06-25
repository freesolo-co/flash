"""Thin Lambda Cloud REST client (no SDK state): instance-types + instance lifecycle.

Mirrors ``providers/runpod/api.py`` / the historical Vast client: stdlib urllib only (via the
shared ``RestClient``), hardened retries, and nothing persisted locally — a fresh process can
list/terminate any instance using only the persisted ids + ``LAMBDA_API_KEY``.

Two Lambda-specific quirks the rest of the provider relies on:

* **Cloudflare WAF.** Lambda's API sits behind Cloudflare, which 403s the stdlib default
  ``Python-urllib/<v>`` User-Agent. The client therefore sends a real UA (``extra_headers``);
  without it EVERY call fails 403 (verified live).
* **Non-idempotent launch.** ``POST /instance-operations/launch`` provisions a NEW (billed)
  instance every time it succeeds, so it is NEVER retried — a blind retry on a timeout where
  Lambda actually accepted the first request would double-provision. Idempotent calls
  (instance-types, list, detail, terminate) keep their retries.
"""

from __future__ import annotations

import time
from typing import Any

from flash._logging import get_logger
from flash.providers._http import RestClient

logger = get_logger(__name__)

LAMBDA_BASE = "https://cloud.lambdalabs.com/api/v1"
# A real User-Agent: Lambda's Cloudflare edge rejects the stdlib default with 403 (verified live).
_USER_AGENT = "flash-lambda/1.0 (+https://freesolo.co)"


class LambdaApiError(RuntimeError):
    pass


_CLIENT = RestClient(
    env_var="LAMBDA_API_KEY",
    error_cls=LambdaApiError,
    base_url=LAMBDA_BASE,
    missing_key_message="LAMBDA_API_KEY not configured on the control-plane host",
    extra_headers={"User-Agent": _USER_AGENT},
)


def _api_key() -> str:
    return _CLIENT.api_key()


def request_with_retries(
    path: str,
    method: str = "GET",
    body: dict | None = None,
    retries: int = 4,
    base_delay: float = 2.0,
) -> Any:
    """REST call hardened against transient network/5xx blips (jittered backoff)."""
    return _CLIENT.request_with_retries(
        path, method=method, body=body, retries=retries, base_delay=base_delay
    )


def _data(out: Any) -> Any:
    """Unwrap Lambda's ``{"data": ...}`` envelope (every 2xx response uses it)."""
    if isinstance(out, dict) and "data" in out:
        return out["data"]
    return out


# ---------------------------------------------------------------------------
# Instance types + capacity (cached: pricing, the allocator, and the launcher all read this)
# ---------------------------------------------------------------------------
_TYPES_TTL_S = 45.0
_types_cache: dict[str, Any] = {"ts": 0.0, "data": None}


def list_instance_types(force: bool = False) -> dict[str, dict]:
    """Map of ``instance_type_name -> {instance_type, regions_with_capacity_available}``.

    Cached for ``_TYPES_TTL_S`` so pricing + allocation + the launch path share one fetch within an
    allocation pass. ``force`` bypasses the cache. Raises ``LambdaApiError`` on a hard failure;
    callers that must degrade gracefully (pricing) catch it.
    """
    now = time.time()
    if not force and _types_cache["data"] is not None and now - _types_cache["ts"] < _TYPES_TTL_S:
        return _types_cache["data"]
    out = _data(request_with_retries("/instance-types"))
    if not isinstance(out, dict):
        raise LambdaApiError(f"unexpected /instance-types response: {out!r}")
    _types_cache.update(ts=now, data=out)
    return out


def regions_with_capacity(instance_type: str, force: bool = False) -> list[str]:
    """Region names that currently have capacity for ``instance_type`` (cheapest source of truth
    for whether a launch can succeed at all)."""
    info = list_instance_types(force=force).get(instance_type) or {}
    return [
        r.get("name")
        for r in info.get("regions_with_capacity_available", [])
        if r.get("name")
    ]


def instance_type_price_usd_hr(instance_type: str) -> float | None:
    """Live $/hr for a Lambda instance type (``price_cents_per_hour`` / 100), or None."""
    info = (list_instance_types().get(instance_type) or {}).get("instance_type") or {}
    cents = info.get("price_cents_per_hour")
    return float(cents) / 100.0 if cents else None


# ---------------------------------------------------------------------------
# SSH keys (launch requires exactly one; the box is bootstrapped via user_data, not SSH)
# ---------------------------------------------------------------------------
def list_ssh_keys() -> list[dict]:
    out = _data(request_with_retries("/ssh-keys"))
    return out if isinstance(out, list) else []


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------
def launch_instance(
    *,
    region_name: str,
    instance_type_name: str,
    ssh_key_names: list[str],
    name: str,
    user_data: str,
) -> str:
    """Launch one instance -> its id. Raises ``LambdaApiError`` on rejection (no capacity, etc.).

    NON-IDEMPOTENT (see module docstring): never retried. A transient failure surfaces to the
    launcher, which walks to the next region/class.
    """
    body = {
        "region_name": region_name,
        "instance_type_name": instance_type_name,
        "ssh_key_names": list(ssh_key_names),
        "name": name[:60],  # Lambda caps ``name`` at 64 chars
        "quantity": 1,
        "user_data": user_data,
    }
    out = _data(request_with_retries("/instance-operations/launch", method="POST", body=body, retries=0))
    ids = out.get("instance_ids") if isinstance(out, dict) else None
    if not ids:
        raise LambdaApiError(f"launch({instance_type_name}@{region_name}) returned no instance id: {out}")
    return str(ids[0])


def get_instance(instance_id: str) -> dict | None:
    """Instance detail dict, or None once it no longer exists (terminated)."""
    try:
        out = request_with_retries(f"/instances/{instance_id}")
    except LambdaApiError as e:
        if "404" in str(e):
            return None
        raise
    data = _data(out)
    return data if isinstance(data, dict) else None


def list_instances() -> list[dict]:
    out = _data(request_with_retries("/instances"))
    return out if isinstance(out, list) else []


def terminate_instances(instance_ids: list[str]) -> bool:
    """Terminate (and stop billing for) instances. Best-effort: never raises."""
    ids = [str(i) for i in instance_ids if i]
    if not ids:
        return False
    try:
        request_with_retries(
            "/instance-operations/terminate",
            method="POST",
            body={"instance_ids": ids},
            retries=2,
        )
        return True
    except Exception as exc:
        logger.warning("lambda terminate(%s) failed: %s", ids, exc)
        return False
