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

import re
import subprocess
import tempfile
import time
from typing import Any

from flash._logging import get_logger
from flash.providers._http import RestClient, is_conflict, is_not_found

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
    the key is inert.

    Shells out to ``ssh-keygen``; a slim control-plane image may not ship it. A missing/failed
    ``ssh-keygen`` raises a clear, actionable ``HyperstackApiError`` (install ``openssh-client`` or
    pin ``HYPERSTACK_KEYPAIR_NAME``) instead of a bare ``FileNotFoundError`` that looks like an
    API/stock failure."""
    with tempfile.TemporaryDirectory() as d:
        kp = f"{d}/k"
        try:
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", kp, "-N", "", "-q"], check=True, timeout=30
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise HyperstackApiError(
                "could not generate a throwaway Hyperstack keypair: ssh-keygen is unavailable or "
                f"failed ({e}). Install openssh-client on the control plane (to provide ssh-keygen), "
                "or set HYPERSTACK_KEYPAIR_NAME to an existing keypair to skip generation."
            ) from e
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
    try:
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
    except HyperstackApiError as e:
        # Create race: two concurrent launches into the same env both see no managed key and both
        # POST. The loser gets an "already exists" rejection — that is SUCCESS, the env-scoped key now
        # exists (the winner created it), so return the name. Re-list to confirm the key is really
        # present before swallowing the error, so an UNRELATED 4xx (e.g. a bad public key / perms)
        # still surfaces rather than being silently treated as a benign duplicate.
        if _keypair_create_conflict(e) and any(
            k.get("name") == name for k in list_keypairs()
        ):
            return name
        raise
    return name


def _keypair_create_conflict(err: Exception) -> bool:
    """True when a keypair POST failed because the name already exists (a benign create race), not
    for some other reason. Hyperstack returns 409/400 with an 'already exists'/'duplicate' body."""
    s = str(err).lower()
    if "-> http 409" in s:
        return True
    return ("-> http 4" in s) and ("already exist" in s or "duplicate" in s or "in use" in s)


# ---------------------------------------------------------------------------
# Images (need a Docker-preinstalled, CUDA-12.8 Ubuntu image to run WORKER_IMAGE)
# ---------------------------------------------------------------------------
_image_cache: dict[str, str] = {}


def _image_cuda(name: str) -> float:
    """Parse the CUDA version out of a Hyperstack image name (e.g. '... CUDA 12.8 ...' -> 12.8)."""
    m = re.search(r"cuda (\d+(?:\.\d+)?)", name.lower())
    return float(m.group(1)) if m else 0.0


def docker_image_for_region(region: str, min_cuda: str = "12.8") -> str:
    """A Docker-preinstalled Ubuntu image in ``region`` whose host CUDA covers the run.

    The host driver's CUDA must be at least the GPU class's floor (``min_cuda`` — e.g. 13.0 for
    Blackwell) AND the cu128 worker container's 12.8. Among the fitting Docker images we pick the
    LOWEST qualifying CUDA (closest to the worker stack) and prefer the newest Ubuntu. Raises if
    none qualifies, so ``launch_and_submit`` skips the region rather than booting a box on a driver
    that can't JIT the GPU's kernels (a Blackwell class on a CUDA-12.8 image would fail at setup)."""
    required = max(12.8, float(min_cuda))
    key = f"{region}|{required}"
    if key in _image_cache:
        return _image_cache[key]
    out = request_with_retries(f"/core/images?region={region}")
    images = out.get("images", []) if isinstance(out, dict) else []
    flat: list[dict] = []
    for x in images:
        if isinstance(x, dict) and "images" in x:
            flat += x["images"]
        elif isinstance(x, dict):
            flat.append(x)
    names = [im.get("name", "") for im in flat]
    # Docker-preinstalled ONLY (the cloud-init does not install Docker) AND CUDA >= required.
    docker_imgs = [n for n in names if "with docker" in n.lower()]
    fitting = [n for n in docker_imgs if _image_cuda(n) >= required - 1e-9]
    if not fitting:
        have = sorted({_image_cuda(n) for n in docker_imgs})
        raise HyperstackApiError(
            f"no Docker image with CUDA>={required} in {region} (available Docker CUDA: {have})"
        )
    best = min(fitting, key=lambda n: (_image_cuda(n), "24.04" not in n))
    _image_cache[key] = best
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
        # ``name`` is bounded <=60 by ``_instance.run_label_prefix`` (NOT truncated here) so the
        # stored name always equals the prefix ``sweep_orphans`` matches on.
        "name": name,
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


# Hyperstack paginates ``/core/virtual-machines``: a single GET returns only the FIRST page, so an
# account with more VMs than one page silently hides the rest. ``sweep_orphans`` lists every VM to
# reap orphans, so a missed page = a leaked, still-billing box. Walk every page (bounded so a buggy
# server that never shrinks the page can't loop forever) and concatenate.
_VM_PAGE_SIZE = 100
_VM_MAX_PAGES = 1000


def list_vms() -> list[dict]:
    """Every Flash-listable VM across ALL pages (orphan-sweep + run-terminate read this).

    Paginates ``/core/virtual-machines`` via ``page``/``per_page`` until a short/empty page is
    reached. A page fetch that errors propagates AND a malformed page schema raises
    ``HyperstackApiError`` — so a partial/incomplete list never masquerades as the authoritative
    fleet and lets a sweep miss still-billing VMs (or reap survivors)."""
    out: list[dict] = []
    seen_ids: set[str] = set()
    terminated = False
    # Walk one page PAST the cap so a fleet that's an exact multiple of _VM_PAGE_SIZE can be CONFIRMED
    # complete: its last in-cap page is full, so the only way to tell "complete (next page empty)"
    # from "truncated (next page still has VMs)" is to fetch that extra probe page. The probe is
    # gated below to fire ONLY at the cap boundary, so the normal walk still stops at _VM_MAX_PAGES.
    for page in range(1, _VM_MAX_PAGES + 2):
        resp = request_with_retries(
            f"/core/virtual-machines?page={page}&per_page={_VM_PAGE_SIZE}"
        )
        # Distinguish a malformed response from a valid empty page. An unexpected schema (resp not a
        # dict, or "instances" not a list) means we CANNOT trust this as the authoritative fleet —
        # returning the partial list gathered so far would let an orphan sweep miss still-billing VMs
        # past this point, so RAISE instead. A valid empty list is the legitimate end-of-pagination.
        if not isinstance(resp, dict) or not isinstance(resp.get("instances"), list):
            raise HyperstackApiError(
                f"unexpected /core/virtual-machines response on page {page} "
                f"(no 'instances' list): {resp!r}"
            )
        insts = resp["instances"]
        if not insts:
            terminated = True
            break  # valid empty page -> done paginating (incl. the boundary probe confirming complete)
        added = 0
        for v in insts:
            # De-dupe across pages: an older API that ignores the ``page`` param would echo page 1
            # forever, so we count only NEW ids and stop below when a full page adds none.
            # Use ``is not None`` (not truthiness) so a legitimately falsy id like 0 still de-dupes.
            vid = str(v["id"]) if isinstance(v, dict) and v.get("id") is not None else None
            if vid is not None and vid in seen_ids:
                continue
            if vid is not None:
                seen_ids.add(vid)
            out.append(v)
            added += 1
        # Last page (short) OR a full page that added nothing new (server ignoring pagination): done.
        if len(insts) < _VM_PAGE_SIZE or added == 0:
            terminated = True
            break
        # A FULL page that added new VMs and we've now consumed the cap (page == _VM_MAX_PAGES): the
        # fleet might be exactly a multiple of the page size (complete) OR genuinely larger
        # (truncated). DON'T raise yet — let the loop fetch ONE more page (_VM_MAX_PAGES + 1) as a
        # probe: if it comes back empty, the break above marks ``terminated`` and we return the full
        # fleet; if it still has VMs, the fleet is genuinely over-cap and we raise below.
    if not terminated:
        # The boundary probe page (_VM_MAX_PAGES + 1) ALSO came back full — pagination did not
        # terminate within the cap and there are genuinely more VMs past it. Returning ``out`` here
        # would hand back a truncated fleet as if authoritative, and an orphan sweep / run-terminate
        # keying off it would miss (or fail to reap) still-billing VMs. Surface it as a
        # HyperstackApiError — same as the malformed-page guard above — so the incomplete list never
        # masquerades as the complete one. (An exact-multiple fleet whose probe page was empty
        # already returned normally above and does NOT reach here.)
        raise HyperstackApiError(
            f"/core/virtual-machines pagination did not terminate within {_VM_MAX_PAGES} pages "
            f"(the page past the cap was still full at {_VM_PAGE_SIZE}/page, {len(out)} VMs "
            f"collected) — refusing to return a truncated fleet that an orphan sweep could read as "
            f"authoritative"
        )
    return out


def delete_vm(vm_id: str) -> bool:
    """Delete (and stop billing for) a VM. Best-effort: never raises."""
    if not vm_id:
        return False
    try:
        request_with_retries(f"/core/virtual-machines/{vm_id}", method="DELETE", retries=2)
        return True
    except Exception as exc:
        # Hyperstack returns 409 when a prior delete request is still being processed: the VM is
        # already queued for teardown so billing will stop — treat as success. Key off the chained
        # HTTPError status (``is_conflict``), NOT a bare "409"/"conflict" substring — a non-409
        # failure whose text merely contains "409" (e.g. a "4090" GPU name) must still surface, or a
        # real delete failure would be silently dropped and the VM left billing.
        if is_conflict(exc):
            return True
        logger.warning("hyperstack delete_vm(%s) failed: %s", vm_id, exc)
        return False


def delete_vms(vm_ids: list[str]) -> list[str]:
    """Delete several VMs (best-effort, per-id isolated). Return the ids that ACTUALLY deleted so
    callers (sweep_orphans / terminate_run_instances) report only what was truly torn down — a
    partial failure must not log/return a still-billing VM as reaped."""
    return [str(v) for v in vm_ids if v and delete_vm(str(v))]
