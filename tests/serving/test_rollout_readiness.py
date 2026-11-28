"""Stable model identities and router startup gating during rolling engine replacements."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from flash.serving.app import rollout
from flash.serving.src.engine.model_config import base_models
from flash.serving.src.http.router import _lifespan_for


def test_model_identity_is_distinct_and_independent_of_hardware():
    names = [rollout.model_engine_name(model) for model in base_models()]
    assert len(set(names)) == len(names)
    assert all(name.replace("_", "").isalnum() for name in names)
    # Normalization must not merge different public model IDs.
    assert rollout.model_engine_name("a/b-c") != rollout.model_engine_name("a/b_c")


def test_router_does_not_admit_requests_until_pool_start_finishes():
    async def scenario():
        warming, ready = asyncio.Event(), asyncio.Event()

        async def start():
            warming.set()
            await ready.wait()

        usage = SimpleNamespace(start=AsyncMock(), aclose=AsyncMock())
        pool = SimpleNamespace(start=start, aclose=AsyncMock())
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


def test_failed_pool_start_closes_router_resources():
    async def scenario():
        async def start():
            async with asyncio.timeout(0.01):
                await asyncio.Event().wait()

        usage = SimpleNamespace(start=AsyncMock(), aclose=AsyncMock())
        pool = SimpleNamespace(start=start, aclose=AsyncMock())
        context = SimpleNamespace(usage=usage, pool=pool, rate_limiter=None)
        with pytest.raises(TimeoutError):
            async with _lifespan_for(context, None)(None):
                pytest.fail("unready router accepted traffic")
        usage.aclose.assert_awaited_once()
        pool.aclose.assert_awaited_once()

    asyncio.run(scenario())
