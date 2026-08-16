"""shared modal plumbing: import purity, cache environment, runtime lifecycle, and draining."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import types
from typing import Any

import pytest

from flash.serve.modal import (
    APT_PACKAGES,
    CUDA_IMAGE,
    RuntimeContainer,
    base_image,
    cache_environment,
)
from flash.serve.runtime import EngineConfig

HEAVY_MODULES = ("modal", "vllm", "torch", "transformers", "PIL")


def test_package_imports_without_modal_or_the_gpu_stack() -> None:
    # `modal deploy` discovers the app locally with only the lightweight extra installed, so
    # importing this package must not pull in modal or the gpu runtime stack.
    program = (
        "import sys;"
        f"[sys.modules.__setitem__(name, None) for name in {HEAVY_MODULES!r}];"
        "import flash.serve.modal as m;"
        "print(','.join(sorted(m.__all__)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RuntimeContainer" in result.stdout
    assert "bind_module_class" not in result.stdout


def test_cache_environment_pins_every_cache_to_one_mount() -> None:
    env = cache_environment("/vol/flash-serving")
    assert env["HF_HOME"] == "/vol/flash-serving"
    assert env["HF_HUB_CACHE"] == "/vol/flash-serving/hub"
    # an unset vllm cache root recompiles the model on every cold start.
    assert env["VLLM_CACHE_ROOT"] == "/vol/flash-serving/vllm"
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert cache_environment("/vol/flash-serving/")["VLLM_CACHE_ROOT"] == "/vol/flash-serving/vllm"


@pytest.mark.parametrize("value", ["", "   ", None, 7])
def test_cache_environment_rejects_an_unusable_mount(value) -> None:
    with pytest.raises(ValueError, match="cache_mount"):
        cache_environment(value)


def test_base_image_applies_toolchain_and_cache_environment(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class _Image:
        def apt_install(self, *packages: str):
            calls["apt"] = packages
            return self

        def env(self, mapping: dict[str, str]):
            calls["env"] = mapping
            return self

    class _ImageFactory:
        @staticmethod
        def from_registry(tag: str, add_python: str):
            calls["registry"] = (tag, add_python)
            return _Image()

    modal = types.ModuleType("modal")
    modal.Image = _ImageFactory
    monkeypatch.setitem(sys.modules, "modal", modal)

    base_image("/vol/cache")
    assert calls["registry"] == (CUDA_IMAGE, "3.12")
    assert calls["apt"] == APT_PACKAGES
    assert calls["env"] == cache_environment("/vol/cache")
    # the shared image never pins a serving dependency; the caller installs its own.
    assert "pip" not in calls


class _Container(RuntimeContainer):
    def engine_config(self) -> EngineConfig:
        return EngineConfig(model="model")


def test_engine_death_drains_the_container_once(monkeypatch) -> None:
    stops: list[int] = []
    experimental = types.ModuleType("modal.experimental")
    experimental.stop_fetching_inputs = lambda: stops.append(1)
    monkeypatch.setitem(sys.modules, "modal.experimental", experimental)

    container = _Container()
    asyncio.run(container._drain_on_engine_death(None))
    asyncio.run(container._drain_on_engine_death(None))
    # repeated death notifications must not drain repeatedly.
    assert stops == [1]


def test_engine_death_exits_when_no_drain_hook_exists(monkeypatch) -> None:
    experimental = types.ModuleType("modal.experimental")
    monkeypatch.setitem(sys.modules, "modal.experimental", experimental)
    exits: list[int] = []
    monkeypatch.setattr("os._exit", lambda code: exits.append(code))

    asyncio.run(_Container()._drain_on_engine_death(None))
    assert exits == [1]


def test_runtime_access_before_start_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="has not been started"):
        _ = _Container().runtime


def test_engine_config_is_required() -> None:
    with pytest.raises(NotImplementedError):
        RuntimeContainer().engine_config()


def test_concurrent_start_constructs_and_starts_one_runtime(monkeypatch) -> None:
    constructed = []
    started = []

    class _Runtime:
        def __init__(self, config, on_engine_death=None) -> None:
            constructed.append(self)

        async def start(self) -> None:
            started.append(self)
            await asyncio.sleep(0)

        async def close(self) -> None:
            pass

    monkeypatch.setattr("flash.serve.modal.engine.VllmLoraRuntime", _Runtime)
    container = _Container()

    async def run():
        return await asyncio.gather(container.start_runtime(), container.start_runtime())

    first, second = asyncio.run(run())
    assert constructed == [first]
    assert started == [first]
    assert second is first
    assert container.runtime is first


def test_failed_start_is_not_published_and_can_be_retried(monkeypatch) -> None:
    attempts = []

    class _Runtime:
        def __init__(self, config, on_engine_death=None) -> None:
            self.attempt = len(attempts) + 1
            attempts.append(self)

        async def start(self) -> None:
            if self.attempt == 1:
                raise RuntimeError("engine core died during startup")

        async def close(self) -> None:
            pass

    monkeypatch.setattr("flash.serve.modal.engine.VllmLoraRuntime", _Runtime)
    container = _Container()
    with pytest.raises(RuntimeError, match="died during startup"):
        asyncio.run(container.start_runtime())
    with pytest.raises(RuntimeError, match="has not been started"):
        _ = container.runtime

    recovered = asyncio.run(container.start_runtime())
    assert len(attempts) == 2
    assert attempts[1] is recovered
    assert recovered.attempt == 2
    assert container.runtime is recovered


def test_close_serializes_with_startup(monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    closed = []

    class _Runtime:
        def __init__(self, config, on_engine_death=None) -> None:
            pass

        async def start(self) -> None:
            started.set()
            await release.wait()

        async def close(self) -> None:
            closed.append(self)

    monkeypatch.setattr("flash.serve.modal.engine.VllmLoraRuntime", _Runtime)
    container = _Container()

    async def run():
        start_task = asyncio.create_task(container.start_runtime())
        await started.wait()
        close_task = asyncio.create_task(container.close_runtime())
        await asyncio.sleep(0)
        assert not close_task.done()
        release.set()
        runtime = await start_task
        await close_task
        return runtime

    runtime = asyncio.run(run())
    assert closed == [runtime]
    with pytest.raises(RuntimeError, match="has not been started"):
        _ = container.runtime
