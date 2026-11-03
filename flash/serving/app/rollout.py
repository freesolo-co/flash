"""Keep model dispatch identities alive and gate new routers on active GPU pools."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from flash.serving.app.timeouts import ROUTER_TIMEOUT_SECONDS

# The identities served by main before model-scoped classes were introduced.
# Old router containers retain these handles until their requests finish. Removing
# a class at deployment deletes the handle immediately, even in a rolling deploy.
# Keep this frozen rather than recomputing it when GPU or concurrency settings change.
LEGACY_ENGINE_KEYS: dict[str, tuple[str, int]] = {
    "Qwen/Qwen3.5-0.8B": ("L4", 64),
    "Qwen/Qwen3.5-4B": ("L40S", 8),
    "Qwen/Qwen3.5-9B": ("L40S", 64),
    "Qwen/Qwen3.5-9B-Base": ("L40S", 64),
    "Qwen/Qwen3.8-27B": ("H100", 8),
    "Qwen/Qwen3.6-35B-A3B": ("H200", 8),
}


def model_engine_name(base_model: str) -> str:
    """Hardware changes must update a function, never replace its dispatch name."""
    readable = "".join(ch if ch.isalnum() else "_" for ch in base_model)
    digest = hashlib.sha256(base_model.encode()).hexdigest()[:12]
    return f"LoraEngine_{readable}_{digest}"


async def _is_active(engine: Any) -> bool:
    # A stats lookup does not start a GPU. Do not turn scale-to-zero models into
    # permanently warm pools merely because a CPU router starts or scales out.
    stats = await engine.health.get_current_stats.aio()
    return any((stats.num_total_runners, stats.num_running_inputs, stats.backlog))


async def _warm_model(model: str, engine: Any, legacy: Any | None) -> None:
    active = await _is_active(engine)
    if legacy is not None:
        active = await _is_active(legacy) or active
    if not active:
        return
    health = await engine.health.remote.aio()
    if health.get("ok") is not True or health.get("base_model") != model:
        raise RuntimeError(f"Replacement engine is not healthy for {model}")


async def warm_active_models(engines: dict[str, Any], legacy: dict[str, Any]) -> None:
    """Finish before ASGI startup succeeds and Modal admits traffic to this router.

    Keep old routers serving during first migration to model-scoped pools. Later
    deploys keep each pool's identity stable, allowing Modal's rolling strategy to
    drain its old replicas. Health may reach a still-serving old replica during
    that transition; this is an availability gate, not a release-identity canary.
    Stats/readiness failures fail startup rather than declaring an unready router
    healthy. TaskGroup cancels sibling probes on failure or cancellation.
    """
    async with asyncio.timeout(ROUTER_TIMEOUT_SECONDS), asyncio.TaskGroup() as tasks:
        for model, engine in engines.items():
            tasks.create_task(_warm_model(model, engine, legacy.get(model)))
