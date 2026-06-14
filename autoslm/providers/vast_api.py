"""Thin Vast.ai REST client (no SDK state): offer search + instance lifecycle.

Mirrors ``flash/runpod_api.py``: stdlib urllib only, hardened retries, and nothing
persisted locally — a fresh process can list/destroy any instance using only the
persisted ids + VAST_API_KEY. Only verified DATACENTER offers are ever searched
(``verified``/``datacenter`` server-side filters; callers re-check client-side).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

VAST_BASE = "https://console.vast.ai/api"


class VastApiError(RuntimeError):
    pass


def _api_key() -> str:
    # Env-only by design, like RUNPOD_API_KEY: the operator sets VAST_API_KEY on the
    # control-plane host; it is never written to config files or shipped to workers.
    key = os.environ.get("VAST_API_KEY")
    if not key:
        raise VastApiError(
            "VAST_API_KEY not configured on the control-plane host (see docs/self-hosting.md)"
        )
    return key


def _request(path: str, method: str = "GET", body: dict | None = None, timeout: float = 30.0):
    req = urllib.request.Request(
        f"{VAST_BASE}{path}",
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
    path: str,
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
            return _request(path, method=method, body=body)
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                detail = ""
                with contextlib.suppress(Exception):
                    detail = e.read().decode()[:500]
                raise VastApiError(f"{method} {path} -> HTTP {e.code}: {e.reason} {detail}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
        if attempt < retries:
            delay = min(base_delay * (2 ** min(attempt, 6)), 30.0)
            time.sleep(delay * random.uniform(0.7, 1.3))
    raise VastApiError(f"{method} {path} failed after {retries + 1} attempts: {last}")


# ---------------------------------------------------------------------------
# Offer search
# ---------------------------------------------------------------------------
def search_offers(
    min_vram_mb: int,
    *,
    min_disk_gb: float = 0,
    min_reliability: float = 0.95,
    limit: int = 64,
    extra_q: dict | None = None,
) -> list[dict]:
    """Rentable single-GPU offers from VERIFIED DATACENTER hosts, cheapest first.

    ``datacenter`` here is Vast's hosting-type filter (professional datacenters vs
    consumer/hobbyist machines); results additionally carry ``hosting_type`` which
    callers must re-check (``usable_offers``) — never trust one filter layer alone.
    """
    q: dict[str, Any] = {
        "verified": {"eq": True},
        "datacenter": {"eq": True},
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": int(min_vram_mb)},
        "reliability2": {"gte": float(min_reliability)},
        "type": "ask",
        "order": [["dph_total", "asc"]],
        "limit": int(limit),
    }
    if min_disk_gb:
        q["disk_space"] = {"gte": float(min_disk_gb)}
    if extra_q:
        q.update(extra_q)
    out = request_with_retries("/v0/search/asks/", method="PUT", body={"q": q})
    offers = out.get("offers") if isinstance(out, dict) else None
    return offers if isinstance(offers, list) else []


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------
def create_instance(
    offer_id: int,
    *,
    image: str,
    disk_gb: float,
    env: dict[str, str],
    onstart: str,
    label: str,
    runtype: str = "args",
) -> int:
    """Rent an offer -> instance id. Raises VastApiError on rejection (offer taken).

    Default ``args`` runtype (verified live): the script IS the container command
    (``bash -c``), so the job needs no SSH key on the account, the container's
    lifecycle is the job's lifecycle, and the Vast-injected CONTAINER_API_KEY /
    CONTAINER_ID env vars are available for the self-destroy backstop. ``ssh``
    runtype requires an SSH key attached to the Vast account.
    """
    body = {
        "client_id": "me",
        "image": image,
        "disk": float(disk_gb),
        "env": dict(env),
        "label": label,
        "runtype": runtype,
    }
    if runtype == "args":
        body["args"] = ["bash", "-c", onstart]
    else:
        body["onstart"] = onstart
    out = request_with_retries(f"/v0/asks/{int(offer_id)}/", method="PUT", body=body, retries=1)
    if not isinstance(out, dict) or not out.get("success"):
        raise VastApiError(f"create_instance({offer_id}) rejected: {out}")
    instance_id = out.get("new_contract")
    if not instance_id:
        raise VastApiError(f"create_instance({offer_id}): no instance id in response: {out}")
    return int(instance_id)


def get_instance(instance_id: int) -> dict | None:
    """Instance detail dict, or None once it no longer exists (destroyed).

    The v0 detail route answers 200 with ``{"instances": null}`` for unknown ids
    (verified live) — that is the "gone" signal, not a 404.
    """
    try:
        out = request_with_retries(f"/v0/instances/{int(instance_id)}/")
    except VastApiError as e:
        if "404" in str(e):
            return None
        raise
    if isinstance(out, dict):
        if "instances" in out:
            inst = out["instances"]
            return inst if isinstance(inst, dict) else None
        return out
    return None


def list_instances() -> list[dict]:
    # The v0 list route is deprecated (410 "use /api/v1/instances/", verified live);
    # detail/destroy remain on v0.
    out = request_with_retries("/v1/instances/")
    inst = out.get("instances") if isinstance(out, dict) else None
    return inst if isinstance(inst, list) else []


def instance_logs(instance_id: int, tail: int = 400, wait_s: float = 20.0) -> str | None:
    """Container log tail via the logs API (request -> poll the result URL).

    The only place early-bootstrap failures (pip/env errors before the worker can
    reach HF) are visible. Best-effort: returns None when logs are unavailable
    (e.g. the instance is already destroyed); never raises.
    """
    import urllib.request

    try:
        out = request_with_retries(
            f"/v0/instances/request_logs/{int(instance_id)}/",
            method="PUT",
            body={"tail": str(int(tail))},
            retries=1,
        )
        url = out.get("result_url") if isinstance(out, dict) else None
        if not url:
            return None
        deadline = time.time() + wait_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    body = resp.read().decode(errors="replace")
                if body.strip():
                    return body
            except urllib.error.HTTPError as e:
                if e.code != 404:  # 404 = not materialized yet
                    return None
            time.sleep(2.0)
    except Exception:
        return None
    return None


def destroy_instance(instance_id: int) -> bool:
    """Destroy (and stop billing for) an instance. Best-effort: never raises."""
    try:
        request_with_retries(f"/v0/instances/{int(instance_id)}/", method="DELETE", retries=2)
        return True
    except Exception:
        return False
