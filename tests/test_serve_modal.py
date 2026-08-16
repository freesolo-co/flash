"""shared modal plumbing: import purity, cache environment, drain handling, and class binding."""

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
    bind_module_class,
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


def test_bind_module_class_names_the_real_class_before_decoration() -> None:
    namespace: dict[str, Any] = {}

    class _Engine:
        pass

    bound = bind_module_class(namespace, _Engine, "LoraEngine_A100_c16")
    assert bound is _Engine
    assert _Engine.__name__ == "LoraEngine_A100_c16"
    # a `<locals>` qualname fails modal's global-scope validation.
    assert "<locals>" not in _Engine.__qualname__
    assert namespace["LoraEngine_A100_c16"] is _Engine


@pytest.mark.parametrize("name", ["", "not an identifier", "9leading"])
def test_bind_module_class_rejects_an_unusable_name(name) -> None:
    with pytest.raises(ValueError, match="class_name"):
        bind_module_class({}, type("X", (), {}), name)


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


def test_a_failed_start_leaves_no_runtime_behind(monkeypatch) -> None:
    """a runtime that never finished starting must not look started.

    publishing it before `start()` returns would let later requests reach a half-initialized
    engine, and the container would answer them instead of failing and being replaced.
    """

    class _DeadRuntime:
        def __init__(self, config, on_engine_death=None) -> None:
            self.config = config

        async def start(self) -> None:
            raise RuntimeError("engine core died during startup")

    monkeypatch.setattr("flash.serve.modal.engine.VllmLoraRuntime", _DeadRuntime)
    container = _Container()
    with pytest.raises(RuntimeError, match="died during startup"):
        asyncio.run(container.start_runtime())
    # the failure must be visible to every later caller, not swallowed into a usable-looking one.
    with pytest.raises(RuntimeError, match="has not been started"):
        _ = container.runtime
