"""Thin Hyperstack (NexGen Cloud) REST client (no SDK state): flavors/capacity + VM lifecycle.

Mirrors ``providers/lambdalabs/api.py``: stdlib urllib via the shared ``RestClient``, hardened
retries, nothing persisted locally. Hyperstack specifics:

* **Auth header.** Hyperstack presents the key as a bare ``api_key: <key>`` header (NOT
  ``Authorization: Bearer``); the ``RestClient`` is configured with ``auth_header_name="api_key"``.
* **Capacity = ``stock_available``.** ``/core/flavors`` carries per-flavor ``stock_available`` per
  region (the Hyperstack analog of Lambda's ``regions_with_capacity_available``); a flavor with no
  stock in any region can't launch.
* **Launch needs an environment + a keypair.** Every region has a ``default-<region>`` environment;
  a launch requires exactly one keypair name (env-scoped). The box is bootstrapped via cloud-init
  ``user_data`` and we never SSH, so the key is a formality — ``resolve_key_name`` reuses an
  existing key or imports a throwaway one (private half discarded; no inbound-SSH rule is opened).
* **Non-idempotent launch.** ``POST /core/virtual-machines`` provisions a NEW (billed) VM every
  time it succeeds, so it is NEVER retried.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from typing import Any

from flash._logging import get_logger
from flash.providers._http import RestClient, is_not_found

logger = get_logger(__name__)

HYPERSTACK_BASE = "https://infrahub-api.nexgencloud.com/v1"
_USER_AGENT = "flash-hyperstack/1.0 (+https://freesolo.co)"
# The managed keypair name (env-scoped). Operators can pin an existing key via HYPERSTACK_KEYPAIR_NAME.
_MANAGED_KEYPAIR = "flash-managed"


class HyperstackApiError(RuntimeError):
    pass


_CLIENT = RestClient(
    env_var="HYPERSTACK_API_KEY",
    error_cls=HyperstackApiError,
    base_url=HYPERSTACK_BASE,
    missing_key_message="HYPERSTACK_API_KEY not configured on the control-plane host",
    extra_headers={"User-Agent": _USER_AGENT},
    auth_header_name="api_key",
    auth_value_format="{key}",
)


def _api_key() -> str:
    return _CLIENT.api_key()


def request_with_retries(
    path: str, method: str = "GET", body: dict | None = None, retries: int = 4, base_delay: float = 2.0
) -> Any:
    return _CLIENT.request_with_retries(
        path, method=method, body=body, retries=retries, base_delay=base_delay
    )


# ---------------------------------------------------------------------------
# Flavors + capacity (cached: pricing, the allocator, and the launcher all read this)
# ---------------------------------------------------------------------------
_FLAVORS_TTL_S = 45.0
_flavors_cache: dict[str, Any] = {"ts": 0.0, "by_region": None}


def _regions() -> list[str]:
    out = request_with_retries("/core/regions")
    regs = out.get("regions", []) if isinstance(out, dict) else []
    names = [r.get("name") for r in regs if r.get("name")]
    return names or ["NORWAY-1", "CANADA-1", "US-1", "CANADA-2"]


def flavors_by_region(force: bool = False) -> dict[str, list[dict]]:
    """``region -> [flavor dict]`` across all regions, cached for ``_FLAVORS_TTL_S``.

    Each flavor dict carries ``name``, ``gpu``, ``gpu_count``, ``stock_available``. Raises
    ``HyperstackApiError`` on a hard failure; callers that must degrade gracefully catch it.
    """
    now = time.time()
    if not force and _flavors_cache["by_region"] is not None and now - _flavors_cache["ts"] < _FLAVORS_TTL_S:
        return _flavors_cache["by_region"]
    by_region: dict[str, list[dict]] = {}
    for region in _regions():
        out = request_with_retries(f"/core/flavors?region={region}")
        data = out.get("data", []) if isinstance(out, dict) else []
        by_region[region] = [f for grp in data for f in grp.get("flavors", [])]
    _flavors_cache.update(ts=now, by_region=by_region)
    return by_region


def regions_with_stock(flavor_name: str, force: bool = False) -> list[str]:
    """Region names where ``flavor_name`` currently has stock (the launchability signal)."""
    out = []
    for region, flavors in flavors_by_region(force=force).items():
        for f in flavors:
            if f.get("name") == flavor_name and f.get("stock_available"):
                out.append(region)
                break
    return out


# ---------------------------------------------------------------------------
# Environments (a launch targets ``default-<region>``)
# ---------------------------------------------------------------------------
_env_cache: dict[str, Any] = {"ts": 0.0, "by_region": None}


def environment_for_region(region: str) -> str:
    """The environment name to launch into for ``region`` (the per-region default env)."""
    now = time.time()
    if _env_cache["by_region"] is None or now - _env_cache["ts"] > 300:
        out = request_with_retries("/core/environments")
        envs = out.get("environments", []) if isinstance(out, dict) else []
        _env_cache.update(ts=now, by_region={e.get("region"): e.get("name") for e in envs if e.get("name")})
    return (_env_cache["by_region"] or {}).get(region) or f"default-{region}"


# ---------------------------------------------------------------------------
# Keypairs (launch requires exactly one; we never SSH)
# ---------------------------------------------------------------------------
def list_keypairs() -> list[dict]:
    out = request_with_retries("/core/keypairs")
    return out.get("keypairs", []) if isinstance(out, dict) else []


def _generate_throwaway_public_key() -> str:
    """An OpenSSH ed25519 public key whose private half is immediately discarded.

    Hyperstack requires a key_name at launch even though the box is bootstrapped via cloud-init and
    we never SSH. The private key is thrown away here and no inbound-SSH security rule is opened, so
    the key is inert."""
    with tempfile.TemporaryDirectory() as d:
        kp = f"{d}/k"
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", kp, "-N", "", "-q"], check=True, timeout=30
        )
        with open(kp + ".pub") as f:
            return f.read().strip()


def resolve_key_name(environment_name: str) -> str:
    """A keypair name usable to launch into ``environment_name``.

    Pins ``HYPERSTACK_KEYPAIR_NAME`` if set; else reuses an existing key in that environment; else
    imports a throwaway ``flash-managed`` key (private half discarded). Idempotent per env.
    """
    import os

    pinned = os.environ.get("HYPERSTACK_KEYPAIR_NAME")
    if pinned:
        return pinned
    existing = list_keypairs()

    def _env_name(k: dict) -> str:
        env = k.get("environment")
        return env.get("name") if isinstance(env, dict) else (env or "")

    # Reuse only a key bound to the EXACT target environment (Hyperstack keypairs are env-scoped, so
    # an env-less / other-env key may be rejected at launch). Otherwise fall through to creating the
    # env-scoped managed key below.
    for k in existing:
        if k.get("name") and _env_name(k) == environment_name:
            return k["name"]
    name = f"{_MANAGED_KEYPAIR}-{environment_name}"
    if any(k.get("name") == name for k in existing):
        return name
    request_with_retries(
        "/core/keypairs",
        method="POST",
        body={
            "name": name,
            "environment_name": environment_name,
            "public_key": _generate_throwaway_public_key(),
        },
        retries=0,
    )
    return name


# ---------------------------------------------------------------------------
# Images (need a Docker-preinstalled, CUDA-12.8 Ubuntu image to run WORKER_IMAGE)
# ---------------------------------------------------------------------------
_image_cache: dict[str, str] = {}


def docker_image_for_region(region: str) -> str:
    """Name of a Docker-preinstalled CUDA-12.8 Ubuntu image in ``region`` (matches the cu128 worker
    image). Prefers the newest (24.04 / R570). Raises if none is found."""
    if region in _image_cache:
        return _image_cache[region]
    out = request_with_retries(f"/core/images?region={region}")
    images = out.get("images", []) if isinstance(out, dict) else []
    flat: list[dict] = []
    for x in images:
        if isinstance(x, dict) and "images" in x:
            flat += x["images"]
        elif isinstance(x, dict):
            flat.append(x)
    names = [im.get("name", "") for im in flat]
    # Prefer "with Docker" + CUDA 12.8 (matches WORKER_IMAGE cu128); then any "with Docker".
    def _score(n: str) -> tuple:
        nl = n.lower()
        return (
            "with docker" in nl,
            "cuda 12.8" in nl,
            "24.04" in nl,
        )
    # Docker-preinstalled ONLY: a plain-Ubuntu fallback would boot, never start Docker (the
    # cloud-init does not install it), and silently stall. Raise so launch_and_submit skips the
    # region (its except handler) instead of provisioning a box that can never run the worker.
    candidates = [n for n in names if "with docker" in n.lower()]
    if not candidates:
        raise HyperstackApiError(f"no Docker-preinstalled image found in {region}")
    best = max(candidates, key=_score)
    _image_cache[region] = best
    return best


# ---------------------------------------------------------------------------
# Virtual machines
# ---------------------------------------------------------------------------
def launch_vm(
    *, name: str, environment_name: str, image_name: str, flavor_name: str, key_name: str, user_data: str
) -> str:
    """Launch one VM -> its id. Raises ``HyperstackApiError`` on rejection (no stock, etc.).

    NON-IDEMPOTENT: never retried (a blind retry on a timeout where Hyperstack accepted the first
    request would double-provision)."""
    body = {
        "name": name[:60],
        "environment_name": environment_name,
        "image_name": image_name,
        "flavor_name": flavor_name,
        "key_name": key_name,
        "count": 1,
        "assign_floating_ip": True,  # public IP for outbound (Docker pull + HF egress)
        "user_data": user_data,
    }
    out = request_with_retries("/core/virtual-machines", method="POST", body=body, retries=0)
    insts = out.get("instances") if isinstance(out, dict) else None
    if not insts:
        # Some responses nest a single instance under "instance".
        one = out.get("instance") if isinstance(out, dict) else None
        insts = [one] if one else None
    if not insts or not insts[0] or not insts[0].get("id"):
        raise HyperstackApiError(f"launch({flavor_name}@{environment_name}) returned no VM id: {out}")
    return str(insts[0]["id"])


def get_vm(vm_id: str) -> dict | None:
    """VM detail dict, or None once it no longer exists (deleted)."""
    try:
        out = request_with_retries(f"/core/virtual-machines/{vm_id}")
    except HyperstackApiError as e:
        # Robust 404 check (NOT a bare "404" substring): Hyperstack VM ids are short integers, so a
        # transient 5xx on a VM whose id contains "404" must not be misread as "deleted".
        if is_not_found(e):
            return None
        raise
    inst = out.get("instance") if isinstance(out, dict) else None
    return inst if isinstance(inst, dict) else None


def list_vms() -> list[dict]:
    out = request_with_retries("/core/virtual-machines")
    insts = out.get("instances") if isinstance(out, dict) else None
    return insts if isinstance(insts, list) else []


def delete_vm(vm_id: str) -> bool:
    """Delete (and stop billing for) a VM. Best-effort: never raises."""
    if not vm_id:
        return False
    try:
        request_with_retries(f"/core/virtual-machines/{vm_id}", method="DELETE", retries=2)
        return True
    except Exception as exc:
        logger.warning("hyperstack delete_vm(%s) failed: %s", vm_id, exc)
        return False


def delete_vms(vm_ids: list[str]) -> list[str]:
    """Delete several VMs (best-effort, per-id isolated). Return the ids that ACTUALLY deleted so
    callers (sweep_orphans / terminate_run_instances) report only what was truly torn down — a
    partial failure must not log/return a still-billing VM as reaped."""
    return [str(v) for v in vm_ids if v and delete_vm(str(v))]
