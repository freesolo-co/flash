"""Keep model dispatch identities alive and gate new routers on active GPU pools."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from flash.serving.app.timeouts import ROUTER_TIMEOUT_SECONDS


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


async def _warm_model(model: str, engine: Any) -> None:
    active = await _is_active(engine)
    if not active:
        return
    health = await engine.health.remote.aio()
    if health.get("ok") is not True or health.get("base_model") != model:
        raise RuntimeError(f"Replacement engine is not healthy for {model}")


async def warm_active_models(engines: dict[str, Any]) -> None:
    """Finish before ASGI startup succeeds and Modal admits traffic to this router.

    Deploys keep each model pool's identity stable, allowing Modal's rolling strategy to
    drain its old replicas. Health may reach a still-serving old replica during
    that transition; this is an availability gate, not a release-identity canary.
    Stats/readiness failures fail startup rather than declaring an unready router
    healthy. TaskGroup cancels sibling probes on failure or cancellation.
    """
    async with asyncio.timeout(ROUTER_TIMEOUT_SECONDS), asyncio.TaskGroup() as tasks:
        for model, engine in engines.items():
            tasks.create_task(_warm_model(model, engine))
