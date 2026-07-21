"""RunPod keep-warm orchestration: reuse a compatible, same-owner warm endpoint instead of
provisioning a fresh one, and keep a finished run's endpoint warm for ``gpu.keep_alive_seconds``.

The durable, owner-scoped record lives in the control-plane db (``flash.server.db`` warm_endpoints).
This module owns the RunPod-specific policy on top of it: what makes two runs *compatible* for
reuse (the compat signature), health-checking a claimed endpoint before trusting it, and reaping
endpoints whose keep-alive window has elapsed.

Reuse is fail-closed and best-effort. It is gated on:
  * exact owner (owner_key_id, owner_org_id) -- the security domain, because a warm worker retains
    the prior run's fetched environment code and on-disk secrets;
  * an identical compat signature -- everything that determines the provisioned gpu resource and the
    environment code, so a reused box is byte-for-byte a box the new run would have provisioned.
Any miss, race, or unhealthy endpoint falls back to a fresh deploy, which is always correct.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from flash.server import db

logger = logging.getLogger(__name__)


def keep_warm_owner(run_id: str) -> tuple[int | None, str]:
    """The authenticated (owner_key_id, owner_org_id) security domain for a run, from control-plane
    state only -- never from the worker payload. Defaults to the null/local domain on any lookup
    failure (local runs, tests), which reuses only within that same null domain."""
    owner_key_id: int | None = None
    owner_org_id = ""
    try:
        from flash.server import db

        owner_key_id = db.run_owner(run_id)
    except Exception:
        owner_key_id = None
    try:
        from flash.runner import _status_org_id, get_status

        owner_org_id = _status_org_id(get_status(run_id)) or ""
    except Exception:
        owner_org_id = ""
    return owner_key_id, owner_org_id


def keep_warm_after_run_from_status(spec: Any, status: Any, *, now: float) -> bool:
    """Register a finished RunPod run's endpoint to stay warm, reading its identity from the persisted
    handle + control-plane owner. Returns True iff the caller must NOT tear the endpoint down."""
    remote = getattr(status, "remote", None) or {}
    endpoint_id = remote.get("endpoint_id")
    if not endpoint_id or remote.get("provider") != "runpod":
        return False
    owner_key_id, owner_org_id = keep_warm_owner(spec.run_id)
    return keep_warm_after_run(
        spec,
        endpoint_id=endpoint_id,
        name=remote.get("name") or endpoint_id,
        owning_fingerprint=remote.get("key_fingerprint") or "",
        owner_key_id=owner_key_id,
        owner_org_id=owner_org_id,
        code_digest=remote.get("code_prefix") or "",
        worker_image=_worker_image_for(spec),
        now=now,
    )


def _worker_image_for(spec: Any) -> str:
    try:
        from flash.providers.runpod.jobs import worker_image_for_gpu

        return worker_image_for_gpu(spec.gpu.type, allow_default=True) or ""
    except Exception:
        return ""


def compat_signature(spec: Any, *, code_digest: str, worker_image: str) -> str:
    """A stable hash of everything that must match for a warm endpoint to be safely reused.

    Captures the resolved gpu resource (type/provider/disk/image), the base-model identity
    (id + immutable revision), the environment CODE identity (``code_digest`` = the immutable
    ``code/<digest>/flash`` prefix), and the resource-sizing inputs the user is allowed to hold
    fixed (context length, lora rank, algorithm). Everything the user may vary without changing the
    gpu or the environment code (learning rate, epochs, batch size, seed, temperature, ...) is
    deliberately excluded, so those runs still reuse the warm box.
    """
    gpu = spec.gpu
    payload = {
        "model": spec.model,
        "model_revision": getattr(spec, "model_revision", "") or "",
        "algorithm": spec.algorithm,
        "gpu_type": gpu.type,
        "gpu_exact_type": getattr(gpu, "exact_type", "") or "",
        "gpu_provider": getattr(gpu, "provider", "") or "",
        "disk_gb": int(gpu.disk_gb),
        "max_context_tokens": spec.train.max_context_tokens,
        "lora_rank": int(spec.train.lora_rank),
        "code_digest": code_digest,
        "worker_image": worker_image or "",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _endpoint_is_reusable(endpoint_id: str, owning_fingerprint: str) -> bool:
    """True only if the endpoint still exists and is not mid-job for another caller.

    A warm endpoint reads as reusable when it exists and has no in-flight work; a scaled-to-zero
    (cold) endpoint is still reusable (it just cold-boots the container, still skipping deploy).
    A deleted/unknown endpoint is not reusable.
    """
    from flash.providers.runpod import api as runpod_api

    try:
        health = runpod_api.endpoint_health_for_fingerprint(endpoint_id, owning_fingerprint) or {}
    except Exception:
        logger.debug("keep-warm: health check failed for %s", endpoint_id, exc_info=True)
        return False
    workers = health.get("workers")
    jobs_info = health.get("jobs")
    if not isinstance(workers, dict) or not isinstance(jobs_info, dict):
        # No health payload at all => the endpoint is gone; do not reuse.
        return bool(health) and workers is not None
    in_flight = (jobs_info.get("inQueue") or 0) + (jobs_info.get("inProgress") or 0)
    return in_flight == 0


def try_acquire(
    spec: Any,
    *,
    owner_key_id: int | None,
    owner_org_id: str,
    code_digest: str,
    worker_image: str,
    run_id: str,
    now: float,
) -> dict | None:
    """Claim a compatible, same-owner, healthy warm endpoint, or return None to deploy fresh.

    Returns the claimed db record (endpoint_id/name/owning_fingerprint/...) on success. On a race or
    an endpoint that turns out to be gone, the claim is released and None is returned.
    """
    sig = compat_signature(spec, code_digest=code_digest, worker_image=worker_image)
    record = db.acquire_warm_endpoint(
        owner_key_id=owner_key_id,
        owner_org_id=owner_org_id,
        compat_sig=sig,
        now=now,
    )
    if record is None:
        return None
    if not _endpoint_is_reusable(record["endpoint_id"], record["owning_fingerprint"]):
        # Claim already consumed the row; the endpoint turned out to be gone, so just deploy fresh.
        logger.info("keep-warm: claimed endpoint %s was gone; deploying fresh", record["endpoint_id"])
        return None
    logger.info("keep-warm: reusing warm endpoint %s for run %s", record["endpoint_id"], run_id)
    return record


def keep_warm_after_run(
    spec: Any,
    *,
    endpoint_id: str,
    name: str,
    owning_fingerprint: str,
    owner_key_id: int | None,
    owner_org_id: str,
    code_digest: str,
    worker_image: str,
    now: float,
) -> bool:
    """Register a finished run's endpoint to stay warm. Returns True iff the caller must NOT tear it
    down. False (keep_alive disabled) leaves teardown to the caller -- the historical behavior."""
    keep_alive = int(getattr(spec.gpu, "keep_alive_seconds", 0) or 0)
    if keep_alive <= 0:
        return False
    sig = compat_signature(spec, code_digest=code_digest, worker_image=worker_image)
    db.register_warm_endpoint(
        endpoint_id=endpoint_id,
        name=name,
        owning_fingerprint=owning_fingerprint,
        owner_key_id=owner_key_id,
        owner_org_id=owner_org_id,
        compat_sig=sig,
        gpu_type=spec.gpu.type,
        expiry_ts=now + keep_alive,
    )
    logger.info(
        "keep-warm: holding endpoint %s warm for %ds (run %s)", endpoint_id, keep_alive, spec.run_id
    )
    return True


def reap_expired(now: float) -> int:
    """Tear down endpoints whose keep-alive window elapsed and drop their rows. Returns count."""
    from flash.providers.runpod import api as runpod_api

    reaped = 0
    for record in db.expired_warm_endpoints(now):
        endpoint_id = record["endpoint_id"]
        try:
            runpod_api.delete_endpoint_for_fingerprint(endpoint_id, record["owning_fingerprint"])
        except Exception:
            logger.debug("keep-warm: reap delete failed for %s", endpoint_id, exc_info=True)
        db.release_warm_endpoint(endpoint_id)
        reaped += 1
        logger.info("keep-warm: reaped expired endpoint %s", endpoint_id)
    return reaped
