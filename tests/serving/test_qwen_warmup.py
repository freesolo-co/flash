"""Readiness waits for completed generation, with isolated cache and abort on failure."""

from types import SimpleNamespace

import pytest

from flash.serving.src.engine.warmup import warmup


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_warmup_drains_real_generation_interface_without_customer_cache():
    calls, closed = [], []

    class Engine:
        async def generate(self, prompt, params, request_id):
            calls.append((prompt, params, request_id))
            try:
                yield SimpleNamespace(finished=False)
                yield SimpleNamespace(finished=True)
            finally:
                closed.append(request_id)

    tokenizer = SimpleNamespace(encode=lambda *a, **k: [1] * 8192)
    await warmup(Engine(), tokenizer, 8)
    assert len(calls) == len(closed) == 15
    assert len({prompt["cache_salt"] for prompt, _, _ in calls}) == 15
    assert {len(prompt["prompt_token_ids"]) for prompt, _, _ in calls} == {32, 768, 2048, 4096}


@pytest.mark.anyio
async def test_incomplete_warmup_aborts_and_fails_startup():
    aborted = []

    class Engine:
        async def generate(self, *args):
            yield SimpleNamespace(finished=False)

        async def abort(self, request_id):
            aborted.append(request_id)

    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await warmup(Engine(), SimpleNamespace(encode=lambda *a, **k: [1] * 8192), 8)
    assert len(aborted) == 1
