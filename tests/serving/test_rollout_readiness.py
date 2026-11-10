"""Availability during the migration and later rolling engine replacements."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from flash.serving.app import rollout
from flash.serving.src.engine.model_config import base_models
from flash.serving.src.http.router import _lifespan_for


def _engine(model="base", *, runners=0, running=0, backlog=0):
    return SimpleNamespace(
        update_autoscaler=SimpleNamespace(aio=AsyncMock()),
        health=SimpleNamespace(
            get_current_stats=SimpleNamespace(
                aio=AsyncMock(
                    return_value=SimpleNamespace(
                        num_total_runners=runners, num_running_inputs=running, backlog=backlog
                    )
                )
            ),
            remote=SimpleNamespace(aio=AsyncMock(return_value={"ok": True, "base_model": model})),
        ),
    )


def test_model_identity_is_distinct_and_independent_of_hardware():
    names = [rollout.model_engine_name(model) for model in base_models()]
    assert len(set(names)) == len(names)
    assert all(name.replace("_", "").isalnum() for name in names)
    # Normalization must not merge different public model IDs.
    assert rollout.model_engine_name("a/b-c") != rollout.model_engine_name("a/b_c")


def test_inactive_models_stay_at_zero():
    current, legacy = _engine(), _engine()
    asyncio.run(rollout.warm_active_models({"base": current}, {"base": legacy}))
    current.health.remote.aio.assert_not_awaited()
    legacy.health.remote.aio.assert_not_awaited()


@pytest.mark.parametrize("activity", [{"runners": 1}, {"running": 1}, {"backlog": 1}])
@pytest.mark.parametrize("migrating", [True, False])
def test_active_or_queued_models_require_readiness(activity, migrating):
    current = _engine(**({} if migrating else activity))
    legacy = _engine(**(activity if migrating else {}))
    asyncio.run(rollout.warm_active_models({"base": current}, {"base": legacy}))
    current.health.remote.aio.assert_awaited_once()
    legacy.health.remote.aio.assert_not_awaited()


def test_first_deployment_does_not_need_a_legacy_pool():
    current = _engine()
    asyncio.run(rollout.warm_active_models({"base": current}))
    current.health.remote.aio.assert_not_awaited()


@pytest.mark.parametrize(
    "health", [{"ok": False, "base_model": "base"}, {"ok": True, "base_model": "wrong"}]
)
def test_unhealthy_replacements_cannot_pass_startup(health):
    current = _engine(runners=1)
    current.health.remote.aio.return_value = health
    with pytest.raises(ExceptionGroup, match="TaskGroup") as error:
        asyncio.run(rollout.warm_active_models({"base": current}))
    assert isinstance(error.value.exceptions[0], RuntimeError)


def test_stats_failure_does_not_silently_mark_a_pool_inactive():
    current = _engine()
    current.health.get_current_stats.aio.side_effect = RuntimeError("stats unavailable")
    with pytest.raises(ExceptionGroup) as error:
        asyncio.run(rollout.warm_active_models({"base": current}))
    assert str(error.value.exceptions[0]) == "stats unavailable"
    current.health.remote.aio.assert_not_awaited()


def test_router_does_not_admit_requests_until_replacement_is_ready():
    async def scenario():
        warming, ready = asyncio.Event(), asyncio.Event()
        current, legacy = _engine(), _engine(runners=2, running=128)

        async def health():
            warming.set()
            await ready.wait()
            return {"ok": True, "base_model": "base"}

        current.health.remote.aio.side_effect = health
        usage = SimpleNamespace(start=AsyncMock(), aclose=AsyncMock())
        pool = SimpleNamespace(
            start=lambda: rollout.warm_active_models({"base": current}, {"base": legacy}),
            aclose=AsyncMock(),
        )
        context = SimpleNamespace(usage=usage, pool=pool, rate_limiter=None)
        lifespan = _lifespan_for(context, None)(None)
        startup = asyncio.create_task(lifespan.__aenter__())
        await warming.wait()
        assert not startup.done()  # Modal must keep the old router serving.
        ready.set()
        await startup
        await lifespan.__aexit__(None, None, None)
        usage.aclose.assert_awaited_once()
        pool.aclose.assert_awaited_once()

    asyncio.run(scenario())


def test_startup_timeout_cancels_probes_and_closes_router_resources(monkeypatch):
    monkeypatch.setattr(rollout, "ROUTER_TIMEOUT_SECONDS", 0.01)

    async def scenario():
        cancelled = asyncio.Event()
        current = _engine(runners=1)

        async def blocked():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        current.health.remote.aio.side_effect = blocked
        usage = SimpleNamespace(start=AsyncMock(), aclose=AsyncMock())
        pool = SimpleNamespace(
            start=lambda: rollout.warm_active_models({"base": current}, {}),
            aclose=AsyncMock(),
        )
        context = SimpleNamespace(usage=usage, pool=pool, rate_limiter=None)
        with pytest.raises(TimeoutError):
            async with _lifespan_for(context, None)(None):
                pytest.fail("unready router accepted traffic")
        assert cancelled.is_set()
        usage.aclose.assert_awaited_once()
        pool.aclose.assert_awaited_once()

    asyncio.run(scenario())


def test_warm_floor_starts_on_first_deployment_without_waking_zero_floor_models():
    warm, idle = _engine("warm"), _engine("idle")
    asyncio.run(rollout.warm_active_models({"warm": warm, "idle": idle}, keep_warm={"warm"}))
    warm.health.remote.aio.assert_awaited_once()
    idle.health.remote.aio.assert_not_awaited()


def test_configured_warm_models_start_and_modal_models_remain_idle():
    models = {model: _engine(model) for model in base_models()}
    asyncio.run(rollout.warm_configured_models(models))
    for model, engine in models.items():
        minimum, maximum = rollout.capacity_for(model, rollout.policy_for(model).provider)
        engine.update_autoscaler.aio.assert_awaited_once_with(
            min_containers=minimum, max_containers=maximum
        )
        if model in {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-9B-Base"}:
            engine.health.remote.aio.assert_awaited_once()
        else:
            engine.health.remote.aio.assert_not_awaited()


def test_capacity_update_failure_blocks_startup():
    model = "Qwen/Qwen3.5-0.8B"
    engine = _engine(model)
    engine.update_autoscaler.aio.side_effect = RuntimeError("autoscaler unavailable")
    with pytest.raises(RuntimeError, match="autoscaler unavailable"):
        asyncio.run(rollout.warm_configured_models({model: engine}))
    engine.health.remote.aio.assert_not_awaited()


def test_dual_provider_capacity_is_applied_to_each_bound_pool(monkeypatch):
    from flash.serving.app.inference_provider import MODEL_POLICIES, ModelPolicy

    model = "Qwen/Qwen3.5-9B"
    monkeypatch.setitem(MODEL_POLICIES, model, ModelPolicy("all", 4, 7))
    primary, fallback = _engine(model), _engine(model)
    asyncio.run(rollout.warm_configured_models({model: primary}))
    asyncio.run(rollout.warm_configured_models({model: fallback}, fallback=True))
    primary.update_autoscaler.aio.assert_awaited_once_with(min_containers=4, max_containers=7)
    fallback.update_autoscaler.aio.assert_awaited_once_with(min_containers=0, max_containers=7)
    primary.health.remote.aio.assert_awaited_once()
    fallback.health.remote.aio.assert_not_awaited()
