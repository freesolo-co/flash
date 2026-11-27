"""Every hosted Qwen model is served by Lambda GPUs only, with no provider mode setting."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from flash.serving.app import inference_provider as policy
from flash.serving.src.engine.model_config import base_models


@pytest.fixture
def anyio_backend():
    return "asyncio"


def passthrough(*args, **kwargs):
    return lambda cls: cls


def test_registered_pools_and_combined_limit(monkeypatch, load_modal_app_under_stub):
    for model in base_models():
        monkeypatch.setitem(policy.MODEL_POLICIES, model, policy.ModelPolicy())
    modal = MagicMock()
    for name in ("concurrent", "method", "enter", "exit", "asgi_app"):
        getattr(modal, name).side_effect = passthrough
    modal.parameter.return_value = None
    app = modal.App.return_value
    for name in ("cls", "function", "local_entrypoint"):
        getattr(app, name).side_effect = passthrough
    mod = load_modal_app_under_stub(modal)
    n = len(base_models())
    # No Modal GPU class: every engine is a CPU-only Lambda proxy.
    assert not [call for call in app.cls.call_args_list if "gpu" in call.kwargs]
    assert len(app.cls.call_args_list) == n
    assert all(call.kwargs["nonpreemptible"] for call in app.cls.call_args_list)
    assert sum(call.kwargs["max_containers"] for call in app.cls.call_args_list) == 2 * n
    assert all(issubclass(cls, mod.LambdaEngineProxy) for cls in mod.ENGINE_BY_MODEL.values())
    assert not hasattr(mod, "MODAL_FALLBACK_BY_MODEL")
    assert not any("DeepSeek" in model for model in mod.ENGINE_BY_MODEL)
    monkeypatch.setenv("FREESOLO_DEPLOYMENT_SHA", "a" * 40)
    pool = mod._ModalEnginePool()
    activation = AsyncMock()
    pool._fleet = SimpleNamespace(activate=SimpleNamespace(remote=SimpleNamespace(aio=activation)))

    async def startup():
        await pool.start()
        await pool.aclose()

    asyncio.run(startup())
    activation.assert_awaited_once_with("a" * 40)


@pytest.mark.parametrize("removed_mode", ["modal", "lambda", "all"])
def test_provider_setting_is_removed(removed_mode):
    with pytest.raises(TypeError):
        policy.ModelPolicy(provider=removed_mode)
    with pytest.raises(TypeError):
        policy.ModelPolicy(removed_mode)
    assert not hasattr(policy.ModelPolicy(), "provider")


def test_deepseek_is_not_reinterpreted_as_a_qwen_worker():
    from flash.serving.lambda_cloud.config import placements

    with pytest.raises(ValueError, match="only defined for the hosted Qwen"):
        placements("deepseek-ai/DeepSeek-V4.1-Flash-1017")


@pytest.mark.anyio
async def test_unregister_clears_active_pools_without_waking_idle_gpu():
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
    "kwargs",
    [
        {"min_capacity": -1},
        {"min_capacity": 3, "max_capacity": 2},
        {"min_capacity": True},
        {"max_capacity": 0},
        {"max_capacity": None},
        {"max_capacity": 1.5},
        {"dev_max_capacity": 0},
        {"lambda_gpus": ("T4",)},
        {"lambda_gpus": ("RTX6000",)},
        {"lambda_gpus": ("GH200",)},
    ],
)
def test_invalid_capacity_fails_before_deployment(kwargs):
    with pytest.raises(ValueError, match=r"capacity|lambda_gpus"):
        policy.ModelPolicy(**kwargs)


def test_shipping_policy_covers_catalog_and_requested_placement():
    from flash.serving.deepinfer.contract.config import MODEL_ID
    from flash.serving.lambda_cloud.config import placements

    assert set(policy.MODEL_POLICIES) == set(base_models()) | {MODEL_ID}
    assert set(policy.lambda_models()) == set(base_models())
    selected = {
        "Qwen/Qwen3.5-0.8B": ("A10", "A6000", "A100-40GB", "A100-80GB", "H100", "H200", "B200"),
        "Qwen/Qwen3.5-9B-Base": ("A6000", "A100-40GB", "A100-80GB", "H100", "H200", "B200"),
    }
    for model, config in policy.MODEL_POLICIES.items():
        # every model scales to zero when idle: no GPU floor in main or dev
        assert config.min_capacity == 0
        assert policy.minimum_for(model, "main") == 0
        assert policy.capacity_for(model, "modal") == (0, 0)
        if model in selected:
            assert policy.capacity_for(model, "lambda") == (
                (0, 4) if "9B-Base" in model else (0, 2)
            )
            # No policy pins one GPU type: any qualified Lambda shape may serve.
            assert config.lambda_gpus == ()
            assert placements(model) == selected[model]
    with pytest.raises(ValueError, match="No serving"):
        policy.policy_for("typo")


def test_mixed_model_decorators_use_independent_capacity(monkeypatch, load_modal_app_under_stub):
    monkeypatch.setitem(
        policy.MODEL_POLICIES, "Qwen/Qwen3.5-4B", policy.ModelPolicy(min_capacity=2, max_capacity=5)
    )
    monkeypatch.setitem(
        policy.MODEL_POLICIES, "Qwen/Qwen3.5-9B", policy.ModelPolicy(min_capacity=4, max_capacity=7)
    )
    modal = MagicMock()
    for name in ("concurrent", "method", "enter", "exit", "asgi_app"):
        getattr(modal, name).side_effect = passthrough
    modal.parameter.return_value = None
    app = modal.App.return_value
    for name in ("cls", "function", "local_entrypoint"):
        getattr(app, name).side_effect = passthrough
    mod = load_modal_app_under_stub(modal)
    # 0.8B, 0.8B Base, 4B, 9B, 9B Base, 27B, 35B (the stub environment is not main).
    expected = [(0, 2), (0, 2), (0, 5), (0, 7), (0, 4), (0, 2), (0, 2)]
    assert [
        (c.kwargs["min_containers"], c.kwargs["max_containers"]) for c in app.cls.call_args_list
    ] == expected
    for model in base_models():
        primary = MagicMock()
        monkeypatch.setitem(mod.ENGINE_BY_MODEL, model, primary)
        pool = mod._ModalEnginePool()
        # Every request routes to the Lambda proxy pool; there is no provider choice.
        assert asyncio.run(pool._engine_for(model)) is primary.return_value
