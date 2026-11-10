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
    for model in base_models():
        monkeypatch.setitem(policy.MODEL_POLICIES, model, policy.ModelPolicy(mode))
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
    # Modal rejects non-preemptible GPU workloads; Lambda proxies are CPU-only.
    assert all(not call.kwargs.get("nonpreemptible", False) for call in gpu_calls)
    assert all(call.kwargs["nonpreemptible"] for call in cpu_calls)
    assert sum(call.kwargs["max_containers"] for call in app.cls.call_args_list) == 2 * n
    for model, cls in mod.ENGINE_BY_MODEL.items():
        if mode == "modal":
            assert issubclass(cls, mod.HostedModalEngine)
            assert cls.pinned_gpu == policy.modal_gpus_for(model)
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
    with pytest.raises(ValueError, match="provider must"):
        policy.ModelPolicy("automatic")
    with pytest.raises(ValueError, match="only defined for hosted Qwen"):
        policy.modal_gpus_for("deepseek-ai/DeepSeek-V4.1-Flash-1017")


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


@pytest.mark.parametrize(
    ("model", "options"),
    [
        ("Qwen/Qwen3.5-0.8B", ["A10G", "A100-80GB"]),
        ("Qwen/Qwen3.5-4B", ["A10G", "L40S", "A100-80GB"]),
        ("Qwen/Qwen3.5-9B", ["L40S", "A100-80GB"]),
        ("Qwen/Qwen3.5-9B-Base", ["L40S", "A100-80GB"]),
        ("Qwen/Qwen3.8-27B", ["A100-80GB"]),
        ("Qwen/Qwen3.6-35B-A3B", ["H200"]),
    ],
)
def test_modal_fallbacks_and_hardware_profiles(model, options):
    from flash.serving.lambda_cloud.profiles import engine_profile

    assert policy.modal_gpus_for(model) == options
    for gpu in options:
        profile = engine_profile(model, "A10" if gpu == "A10G" else gpu)
        assert profile["max_model_len"] == 262144
    if "9B" in model:
        assert engine_profile(model, "A100-80GB")["quantization"] == "fp8_per_tensor"
        assert engine_profile(model, "L40S")["serve_model_id"] != model


def test_modal_gpu_selection_uses_actual_device_and_rejects_wrong_shapes():
    from flash.serving.app.modal_engine import selected_gpu

    options = ["A10G", "L40S", "A100-80GB"]
    assert selected_gpu(options, "NVIDIA L40S", 48 * 2**30, 1) == "L40S"
    assert selected_gpu(options, "NVIDIA A100-SXM4-80GB", 80 * 2**30, 1) == "A100-80GB"
    assert selected_gpu(options, "NVIDIA A10", 24 * 2**30, 1) == "A10G"
    for name, gib, count in [("NVIDIA L40", 48, 1), ("NVIDIA A100", 40, 1), ("NVIDIA L40S", 48, 2)]:
        with pytest.raises(RuntimeError, match="unapproved GPU"):
            selected_gpu(options, name, gib * 2**30, count)


@pytest.mark.anyio
async def test_modal_load_selects_actual_profile_and_isolates_compiled_cache(monkeypatch):
    import os
    import sys

    from flash.serving.app.modal_engine import HostedModalEngine, _LoraEngineImpl

    monkeypatch.setenv("VLLM_CACHE_ROOT", "/cache/vllm")
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_name=lambda _: "NVIDIA L40S",
            get_device_properties=lambda _: SimpleNamespace(total_memory=48 * 2**30),
            device_count=lambda: 1,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    load = AsyncMock()
    monkeypatch.setattr(_LoraEngineImpl, "_load", load)
    engine = HostedModalEngine()
    engine.base_model = "Qwen/Qwen3.5-9B"
    engine.pinned_gpu = policy.modal_gpus_for(engine.base_model)
    await engine._load()
    assert engine.pinned_gpu == "L40S"
    assert engine.engine_overrides["serve_model_id"] == "Freesolo-Co/Qwen3.5-9B-FP8"
    assert os.environ["VLLM_CACHE_ROOT"] == "/cache/vllm/L40S"
    load.assert_awaited_once()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_capacity": -1},
        {"min_capacity": 3, "max_capacity": 2},
        {"min_capacity": True},
        {"max_capacity": 0},
        {"max_capacity": None},
        {"max_capacity": 1.5},
        {"provider": "all", "max_capacity": 1},
        {"dev_max_capacity": 0},
        {"lambda_gpus": ("T4",)},
    ],
)
def test_invalid_capacity_fails_before_deployment(kwargs):
    with pytest.raises(ValueError, match=r"capacity|lambda_gpus"):
        policy.ModelPolicy(**kwargs)


def test_shipping_policy_covers_catalog_and_requested_placement():
    from flash.serving.deepinfer.contract.config import MODEL_ID
    from flash.serving.lambda_cloud.config import placements

    assert set(policy.MODEL_POLICIES) == set(base_models()) | {MODEL_ID}
    selected = {"Qwen/Qwen3.5-0.8B": "A10", "Qwen/Qwen3.5-9B-Base": "H100"}
    for model, config in policy.MODEL_POLICIES.items():
        if model in selected:
            assert config.provider == "lambda"
            assert policy.capacity_for(model, "lambda") == (1, 2)
            assert placements(model) == (selected[model],)
        else:
            assert config.provider == "modal"
            assert config.min_capacity == 0
    with pytest.raises(ValueError, match="No serving"):
        policy.policy_for("typo")


def test_mixed_model_decorators_use_independent_capacity(monkeypatch, load_modal_app_under_stub):
    monkeypatch.setitem(policy.MODEL_POLICIES, "Qwen/Qwen3.5-4B", policy.ModelPolicy("modal", 2, 5))
    monkeypatch.setitem(policy.MODEL_POLICIES, "Qwen/Qwen3.5-9B", policy.ModelPolicy("all", 4, 7))
    modal = MagicMock()
    for name in ("concurrent", "method", "enter", "exit", "asgi_app"):
        getattr(modal, name).side_effect = passthrough
    modal.parameter.return_value = None
    app = modal.App.return_value
    for name in ("cls", "function", "local_entrypoint"):
        getattr(app, name).side_effect = passthrough
    mod = load_modal_app_under_stub(modal)
    expected = [(0, 2), (0, 5), (0, 4), (0, 2), (0, 2), (0, 2), (0, 3)]
    assert [
        (c.kwargs["min_containers"], c.kwargs["max_containers"]) for c in app.cls.call_args_list
    ] == expected
    assert set(mod.MODAL_FALLBACK_BY_MODEL) == {"Qwen/Qwen3.5-9B"}
    # A fallback on another model must never redirect a Lambda-only or Modal-only model.
    for model in mod.ENGINE_BY_MODEL:
        monkeypatch.setitem(mod.ENGINE_BY_MODEL, model, MagicMock())
    monkeypatch.setitem(mod.MODAL_FALLBACK_BY_MODEL, "Qwen/Qwen3.5-9B", MagicMock())
    pool = mod._ModalEnginePool()
    pool._provider_selection = SimpleNamespace(choose=AsyncMock(return_value="modal"))
    asyncio.run(pool._engine_for("Qwen/Qwen3.5-0.8B"))
    asyncio.run(pool._engine_for("Qwen/Qwen3.5-4B"))
    pool._provider_selection.choose.assert_not_awaited()
    asyncio.run(pool._engine_for("Qwen/Qwen3.5-9B"))
    pool._provider_selection.choose.assert_awaited_once_with("Qwen/Qwen3.5-9B")
