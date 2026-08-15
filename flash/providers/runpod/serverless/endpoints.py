"""RunPod Flash endpoint lifecycle: provision, cache, teardown.

The worker handler itself lives in ``handler``; this module only hands it to the SDK. They were one
file until the handler's 662 lines left no room for either to grow.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import threading
from typing import Any

from flash._internal.diagnostics import sanitize_diagnostic
from flash.providers._lifecycle.worker import (
    DEFAULT_EXECUTION_TIMEOUT_MS,
    WORKER_SYSTEM_DEPS,
    logger,
    resolve_worker_deps,
    worker_image_for_gpu,
)
from flash.providers.base import canonical_gpu, gpu_short
from flash.providers.runpod.gpus import flash_gpu

# re-exported: callers and tests reach these as `endpoints.<name>` and through the package. The
# console constants stay importable from here because they describe the handler's upload cadence,
# which this module's endpoint lifecycle is what tears down.
from flash.providers.runpod.serverless.handler import (  # noqa: F401
    _CONSOLE_UPLOAD_FIRST_SNAPSHOT_S,
    _CONSOLE_UPLOAD_INTERVAL_S,
    _CONSOLE_UPLOAD_POLL_S,
    _train_body,
)

# runpod_flash asyncio singleton is bound to one event loop; serialize all deploy/undeploy.
FLASH_SDK_LOCK = threading.Lock()

_ENDPOINT_CACHE: dict[str, Any] = {}


def _patch_runpod_backoff() -> None:
    """Cap the backoff exponent before the power to prevent overflow on long runs."""
    try:
        import math
        import random

        from runpod_flash.core.utils import backoff as runpod_backoff

        if getattr(runpod_backoff, "_flash_backoff_patched", False):
            return

        def _safe_get_backoff_delay(
            attempt,
            base=0.1,
            max_seconds=10.0,
            jitter=0.2,
            strategy=runpod_backoff.BackoffStrategy.EXPONENTIAL,
        ):
            capped_attempt = min(int(attempt), 30)
            if strategy == runpod_backoff.BackoffStrategy.EXPONENTIAL:
                delay = base * (2**capped_attempt)
            elif strategy == runpod_backoff.BackoffStrategy.LINEAR:
                delay = base + (attempt * base)
            elif strategy == runpod_backoff.BackoffStrategy.LOGARITHMIC:
                delay = base * math.log2(attempt + 2)
            else:
                raise ValueError(f"Unsupported backoff strategy: {strategy}")
            return min(delay, max_seconds) * random.uniform(1 - jitter, 1 + jitter)

        runpod_backoff.get_backoff_delay = _safe_get_backoff_delay
        runpod_backoff._flash_backoff_patched = True
        try:
            from runpod_flash.core.resources import serverless

            serverless.get_backoff_delay = _safe_get_backoff_delay
        except Exception:
            pass
    except Exception as exc:
        logger.warning("runpod backoff patch skipped: %s", exc)


def _reset_flash_resource_manager(rm_module) -> None:
    """Drop runpod_flash's in-memory ResourceManager state after switching state files."""
    manager = getattr(rm_module, "ResourceManager", None)
    if manager is None:
        return

    instances = getattr(manager, "_instances", None)
    instance = instances.get(manager) if isinstance(instances, dict) else None
    for target in (manager, instance):
        if target is None:
            continue
        for attr in ("_resources", "_resource_configs", "_deployment_locks"):
            state = getattr(target, attr, None)
            if isinstance(state, dict):
                state.clear()
            elif state is not None:
                with contextlib.suppress(Exception):
                    setattr(target, attr, {})
    with contextlib.suppress(Exception):
        manager._resources_initialized = False


def isolate_flash_state(scope: str | None = None) -> None:
    """Point the Flash SDK's resource registry at a per-process dir under <data dir>/flash-state/."""
    try:
        import runpod_flash.core.resources.resource_manager as rm

        from flash._internal.paths import data_dir

        scope = scope or f"pid{os.getpid()}"
        state_dir = data_dir() / "flash-state" / scope
        state_dir.mkdir(parents=True, exist_ok=True)
        previous_state_file = getattr(rm, "RESOURCE_STATE_FILE", None)
        rm.FLASH_STATE_DIR = state_dir
        rm.RESOURCE_STATE_FILE = state_dir / "resources.pkl"
        if hasattr(rm, "RUNPOD_FLASH_DIR"):
            rm.RUNPOD_FLASH_DIR = state_dir
        if previous_state_file != rm.RESOURCE_STATE_FILE:
            _reset_flash_resource_manager(rm)
    except Exception as exc:
        logger.warning("flash state isolation skipped: %s", exc)


def min_cuda_for(friendly_gpu: str) -> str:
    """Minimum host CUDA driver version for this GPU class (Blackwell requires >=13.0)."""
    from flash.providers.base import min_cuda_modern

    return min_cuda_modern(friendly_gpu)


def endpoint_name(friendly_gpu: str, suffix: str | None = None) -> str:
    """Flash endpoint name for a GPU class, with a per-run suffix to avoid template name collisions."""
    base = f"flash-{gpu_short(friendly_gpu)}"
    if not suffix:
        return base
    safe = "".join(c for c in str(suffix) if c.isalnum() or c == "-").strip("-")[:24]
    return f"{base}-{safe}" if safe else base


def get_train_endpoint(
    friendly_gpu: str,
    execution_timeout_ms: int | None = None,
    name_suffix: str | None = None,
    disk_gb: int | None = None,
    spec=None,
):
    """Build (and cache) the live Flash endpoint handler for a GPU class."""
    from runpod_flash import Endpoint

    from flash.core.spec import gpu_count_of
    from flash.providers.runpod.auth import ensure_auth

    ensure_auth()
    _patch_runpod_backoff()

    friendly = canonical_gpu(friendly_gpu)
    name = endpoint_name(friendly, name_suffix)
    cache_handler = name_suffix is None
    with FLASH_SDK_LOCK:
        isolate_flash_state(name_suffix)
        if cache_handler and name in _ENDPOINT_CACHE:
            return _ENDPOINT_CACHE[name]
        kwargs = {
            "name": name,
            "gpu": flash_gpu(friendly),
            # one worker occupies gpu.count cards of this class; count == 1 is the historical path.
            "gpu_count": gpu_count_of(spec),
            "min_cuda_version": min_cuda_for(friendly),
            "execution_timeout_ms": execution_timeout_ms or DEFAULT_EXECUTION_TIMEOUT_MS,
            "workers": (0, 1),
        }
        image = worker_image_for_gpu(friendly, allow_default=False)
        if image:
            kwargs["image"] = image
        else:
            kwargs["dependencies"] = resolve_worker_deps()
            kwargs["system_dependencies"] = WORKER_SYSTEM_DEPS
        # Local import: avoids a jobs<->endpoints import cycle (jobs imports this module).
        from flash.providers.runpod.jobs import (
            grow_weight_cache_volumes,
            weight_cache_endpoint_kwargs,
        )

        # resize before attach because existing volumes keep their provisioned size.
        # reread the key after waiting for the lock so resize and Endpoint use the same account.
        grow_weight_cache_volumes(spec, ensure_auth())
        kwargs.update(weight_cache_endpoint_kwargs(spec))
        ep = Endpoint(**kwargs)
        handler = ep(_train_body)
        from flash.providers.runpod.jobs import apply_disk_gb

        cfg = ep._build_resource_config()
        apply_disk_gb(cfg, disk_gb)
        if cache_handler:
            _ENDPOINT_CACHE[name] = handler
        return handler


def _run_suffix(run_id: str | None) -> str | None:
    """Stable, collision-free per-run endpoint suffix: sha1(run_id)[:8] with a readable prefix.

    Using only the last segment of run_id collides when run_ids end in a GPU name.
    """
    if not run_id:
        return None
    import hashlib
    import re

    h = hashlib.sha1(run_id.encode()).hexdigest()[:8]
    prefix = re.sub(r"[^a-z0-9]", "", run_id.lower())[-12:]
    return f"{prefix}{h}" if prefix else h


def stop_endpoint(friendly_gpu: str, name: str | None = None) -> None:
    """Scale cached endpoint(s) to zero. Only touches in-process cache; use terminate_endpoint for cross-process teardown."""
    friendly = canonical_gpu(friendly_gpu)
    prefix = f"flash-{gpu_short(friendly)}"
    if name:
        match = [k for k in _ENDPOINT_CACHE if k == name]
    else:
        match = [k for k in _ENDPOINT_CACHE if k.startswith(prefix)]
    for key in match:
        handler = _ENDPOINT_CACHE.pop(key, None)
        ep = getattr(handler, "__self__", None) or getattr(handler, "endpoint", None)
        for meth in ("scale_to_zero", "stop", "delete"):
            fn = getattr(ep, meth, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    continue


def _endpoint_name_matches_run(name: str, target: str) -> bool:
    canonical = str(name or "").removeprefix("live-")
    return (
        canonical == target
        or re.fullmatch(re.escape(target) + r"r[1-9][0-9]*", canonical) is not None
    )


def _select_endpoint_resources(resources: dict, target: str) -> list[str]:
    """Return exact base and canonical retry endpoint resource ids for one run."""
    if not target:
        return []
    return [
        uid
        for uid, resource in (resources or {}).items()
        if _endpoint_name_matches_run(getattr(resource, "name", ""), target)
    ]


def terminate_endpoint(friendly_gpu: str, run_id: str | None = None) -> list[dict]:
    """Delete the remote Flash endpoint(s) for a run via the RunPod API. Best-effort, never raises."""
    friendly = canonical_gpu(friendly_gpu)
    target = endpoint_name(friendly, _run_suffix(run_id))
    # Serialize isolation + lookup + undeploy: isolate_flash_state swaps process-wide globals,
    # and a concurrent call could swap the registry scope between our lookup and undeploy.
    with FLASH_SDK_LOCK:
        try:
            from flash.providers.runpod.auth import ensure_auth

            ensure_auth()
            isolate_flash_state(_run_suffix(run_id))
            from runpod_flash.core.resources.resource_manager import ResourceManager
        except Exception as exc:
            detail = sanitize_diagnostic(exc, limit=1000)
            return [{"success": False, "name": target, "message": f"flash unavailable: {detail}"}]

        try:
            rm = ResourceManager()
            resources = rm.list_all_resources()
            uids = _select_endpoint_resources(resources, target)
        except Exception as exc:
            detail = sanitize_diagnostic(exc, limit=1000)
            return [
                {"success": False, "name": target, "message": f"resource lookup failed: {detail}"}
            ]

        async def _undeploy_all() -> list:
            out = []
            for uid in uids:
                res = resources.get(uid)
                name = getattr(res, "name", None)
                try:
                    out.append(
                        await rm.undeploy_resource(uid, resource_name=name, force_remove=True)
                    )
                except Exception as exc:
                    out.append(
                        {
                            "success": False,
                            "name": name,
                            "message": sanitize_diagnostic(exc, limit=1000),
                        }
                    )
            return out

        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                results = asyncio.run(_undeploy_all())
            else:
                # Running event loop (FastAPI lifespan etc) — run in a daemon thread.
                _out: list = []
                _err: list = []

                def _run_undeploy() -> None:
                    try:
                        _out.append(asyncio.run(_undeploy_all()))
                    except Exception as _e:
                        _err.append(_e)

                _t = threading.Thread(target=_run_undeploy, daemon=True)
                _t.start()
                _t.join(timeout=30)
                if _err:
                    raise _err[0]
                if not _out:
                    raise TimeoutError("undeploy timed out after 30s")
                results = _out[0]
        except Exception as exc:
            results = [
                {
                    "success": False,
                    "name": target,
                    "message": sanitize_diagnostic(exc, limit=1000),
                }
            ]

    # registry-less cleanup must inspect every configured account and exact retry suffix.
    try:
        from flash.providers.runpod import api as runpod_api

        by_fingerprint, failed_fingerprints = runpod_api.list_endpoints_by_key()
        for fingerprint, endpoints in by_fingerprint.items():
            for endpoint in endpoints:
                if not _endpoint_name_matches_run(endpoint.get("name", ""), target):
                    continue
                endpoint_id = endpoint.get("id")
                if not endpoint_id or not runpod_api.delete_endpoint_for_fingerprint(
                    endpoint_id, fingerprint
                ):
                    results.append(
                        {
                            "success": False,
                            "name": endpoint.get("name") or target,
                            "message": "REST endpoint deletion was unconfirmed",
                        }
                    )
                    continue
                results.append(
                    {
                        "success": True,
                        "name": endpoint.get("name") or target,
                        "message": "deleted via REST API",
                    }
                )
        if failed_fingerprints:
            results.append(
                {
                    "success": False,
                    "name": target,
                    "message": (
                        f"could not enumerate {len(failed_fingerprints)} configured RunPod account(s)"
                    ),
                }
            )
    except Exception as exc:
        logger.warning("REST endpoint cleanup failed for %s: %s", target, exc)

    with contextlib.suppress(Exception):
        stop_endpoint(friendly, name=target)
    return results
