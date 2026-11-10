"""Keep model dispatch identities alive and gate new routers on active GPU pools."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from flash.serving.app.inference_provider import capacity_for, policy_for
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


async def _warm_model(model: str, engine: Any, legacy: Any | None, keep_warm: bool) -> None:
    active = keep_warm or await _is_active(engine)
    if legacy is not None:
        active = await _is_active(legacy) or active
    if not active:
        return
    health = await engine.health.remote.aio()
    if health.get("ok") is not True or health.get("base_model") != model:
        raise RuntimeError(f"Replacement engine is not healthy for {model}")


async def warm_active_models(
    engines: dict[str, Any],
    legacy: dict[str, Any] | None = None,
    *,
    keep_warm: set[str] | None = None,
) -> None:
    """Finish before ASGI startup succeeds and Modal admits traffic to this router.

    Explicit warm floors start even on the first deployment with no traffic.
    Production supplies only model-scoped pools. The isolated rollout qualification
    may supply legacy handles to exercise migration readiness. Deploys keep each
    pool's identity stable, allowing Modal's rolling strategy to
    drain its old replicas. Health may reach a still-serving old replica during
    that transition; this is an availability gate, not a release-identity canary.
    Stats/readiness failures fail startup rather than declaring an unready router
    healthy. TaskGroup cancels sibling probes on failure or cancellation.
    """
    async with asyncio.timeout(ROUTER_TIMEOUT_SECONDS), asyncio.TaskGroup() as tasks:
        for model, engine in engines.items():
            tasks.create_task(
                _warm_model(model, engine, (legacy or {}).get(model), model in (keep_warm or set()))
            )


async def warm_configured_models(engines: dict[str, Any], *, fallback: bool = False) -> None:
    """Start configured warm pools and gate already-active pools during rollout."""
    keep_warm = set()
    for model in engines:
        provider = "modal" if fallback or policy_for(model).provider == "modal" else "lambda"
        minimum, maximum = capacity_for(model, provider)
        await engines[model].update_autoscaler.aio(min_containers=minimum, max_containers=maximum)
        if minimum > 0:
            keep_warm.add(model)
    await warm_active_models(engines, keep_warm=keep_warm)
