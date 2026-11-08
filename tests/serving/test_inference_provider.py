"""Provider selection is explicit and never retries a dispatched generation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from flash.serving.app import inference_provider as policy
from flash.serving.app.provider_selection import ProviderSelection
from flash.serving.src.engine.model_config import base_models


@pytest.fixture
def anyio_backend():
    return "asyncio"


def passthrough(*args, **kwargs):
    return lambda cls: cls


@pytest.mark.parametrize("mode", ["modal", "lambda", "all"])
def test_registered_pools_and_combined_limit(mode, monkeypatch, load_modal_app_under_stub):
    monkeypatch.setattr(policy, "INFERENCE_PROVIDER", mode)
    modal = MagicMock()
    for name in ("concurrent", "method", "enter", "exit", "asgi_app"):
        getattr(modal, name).side_effect = passthrough
    modal.parameter.return_value = None
    app = modal.App.return_value
    for name in ("cls", "function", "local_entrypoint"):
        getattr(app, name).side_effect = passthrough
    mod = load_modal_app_under_stub(modal)
    gpu_calls = [call for call in app.cls.call_args_list if "gpu" in call.kwargs]
    cpu_calls = [call for call in app.cls.call_args_list if "gpu" not in call.kwargs]
    n = len(base_models())
    assert len(gpu_calls) == (n if mode in {"modal", "all"} else 0)
    assert len(cpu_calls) == (n if mode in {"lambda", "all"} else 0)
    assert sum(call.kwargs["max_containers"] for call in app.cls.call_args_list) == 2 * n
    for model, cls in mod.ENGINE_BY_MODEL.items():
        if mode == "modal":
            assert issubclass(cls, mod.HostedModalEngine)
            assert cls.pinned_gpu == policy.modal_gpu_for(model)
        else:
            assert issubclass(cls, mod.LambdaEngineProxy)
    assert bool(mod.MODAL_FALLBACK_BY_MODEL) == (mode == "all")
    assert not any("DeepSeek" in model for model in mod.ENGINE_BY_MODEL)


@pytest.mark.anyio
async def test_all_selects_by_capacity_and_caches_concurrent_lookups():
    now = 100
    capacity = AsyncMock(return_value=True)
    selection = ProviderSelection(capacity, clock=lambda: now)
    assert await asyncio.gather(*(selection.choose("model") for _ in range(10))) == ["lambda"] * 10
    capacity.assert_awaited_once_with("model")
    capacity.return_value = False
    assert await selection.choose("model") == "lambda"
    now += 31
    assert await selection.choose("model") == "modal"


@pytest.mark.anyio
async def test_provider_api_failure_is_not_treated_as_missing_capacity():
    capacity = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await ProviderSelection(capacity).choose("model")


def test_unknown_mode_and_deepseek_are_not_reinterpreted():
    with pytest.raises(ValueError, match="INFERENCE_PROVIDER"):
        policy.validate_provider("automatic")
    with pytest.raises(ValueError, match="only defined for hosted Qwen"):
        policy.modal_gpu_for("deepseek-ai/DeepSeek-V4.1-Flash-1017")


@pytest.mark.anyio
async def test_unregister_clears_both_active_pools_without_waking_idle_gpu():
    from flash.serving.app.provider_selection import unregister_active

    engines = [
        SimpleNamespace(
            health=SimpleNamespace(
                get_current_stats=SimpleNamespace(
                    aio=AsyncMock(
                        return_value=SimpleNamespace(
                            num_total_runners=runners, num_running_inputs=0, backlog=0
                        )
                    )
                )
            ),
            unregister=SimpleNamespace(remote=SimpleNamespace(aio=AsyncMock())),
        )
        for runners in (1, 1, 0)
    ]
    await unregister_active(engines, "org", "adapter", "generation")
    for engine in engines[:2]:
        engine.unregister.remote.aio.assert_awaited_once_with("org", "adapter", "generation")
    engines[2].unregister.remote.aio.assert_not_awaited()
