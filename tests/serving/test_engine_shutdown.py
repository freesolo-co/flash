import asyncio
from types import SimpleNamespace

import pytest

from flash.serving.src.engine.lora_engine import _LoraEngineImpl


def test_shutdown_finishes_background_tasks_before_loop_closes() -> None:
    async def run() -> None:
        owner_loop = asyncio.get_running_loop()
        cleaned = []

        async def background(name):
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.append(name)

        impl = _LoraEngineImpl()
        impl._liveness_task = asyncio.create_task(background("monitor"))
        handler = asyncio.create_task(background("output"))
        shutdown_calls = []

        def shutdown(*, timeout):
            assert asyncio.get_running_loop() is owner_loop
            assert cleaned == ["monitor"]
            shutdown_calls.append(timeout)
            owner_loop.call_soon(handler.cancel)

        impl.engine = SimpleNamespace(shutdown=shutdown, output_handler=handler)
        await asyncio.sleep(0)
        await impl._shutdown()
        assert handler.done()
        assert cleaned == ["monitor", "output"]
        await impl._shutdown()
        assert shutdown_calls == [10.0]

    asyncio.run(run())


def test_shutdown_handles_partial_initialization() -> None:
    impl = _LoraEngineImpl()
    asyncio.run(impl._shutdown())
    calls = []
    impl.engine = SimpleNamespace(shutdown=lambda **kwargs: calls.append(kwargs))
    asyncio.run(impl._shutdown())
    assert calls == [{"timeout": 10.0}]


def test_shutdown_engine_failure_still_joins_output_handler() -> None:
    async def run() -> None:
        impl = _LoraEngineImpl()
        handler = asyncio.create_task(asyncio.Event().wait())

        def shutdown(*, timeout):
            raise RuntimeError("engine shutdown failed")

        impl.engine = SimpleNamespace(shutdown=shutdown, output_handler=handler)
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="engine shutdown failed"):
            await impl._shutdown()
        assert handler.done()

    asyncio.run(run())
