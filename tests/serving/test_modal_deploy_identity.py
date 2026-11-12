"""Import-time deployment identity checks for the hosted Modal serving app."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

_DEPLOYMENT_ENV_VARS = (
    "SERVING_DEPLOYMENT_MODE",
    "SERVING_CUSTOM_DOMAIN",
    "MODAL_IS_REMOTE",
    "FREESOLO_INTERNAL_KEY",
    "HF_TOKEN",
    "FLASH_ARTIFACT_BLOB_SAS_URL",
    "FLASH_ARTIFACT_BLOB_ACCOUNT_URL",
    "FLASH_ARTIFACT_BLOB_CONTAINER",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "PLATFORM_BACKEND_URL",
    "SUPABASE_PROJECT_REF",
    "SUPABASE_PROJECT_REF_DEV",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_URL",
)
_DEVELOPMENT_WIRING = {
    "FREESOLO_INTERNAL_KEY": "dev-internal-key",
    "HF_TOKEN": "dev-hf-key",
    "FLASH_ARTIFACT_BLOB_SAS_URL": "https://test.blob.core.windows.net/artifacts-dev?sig=test",
    "PLATFORM_BACKEND_URL": "https://api-dev.freesolo.co",
    "SERVING_CUSTOM_DOMAIN": "serve-dev.freesolo.co",
    "SERVING_DEPLOYMENT_MODE": "development",
    "SUPABASE_PROJECT_REF": "production-project-ref",
    "SUPABASE_PROJECT_REF_DEV": "dev-project-ref",
    "SUPABASE_SERVICE_ROLE_KEY": "dev-service-key",
    "SUPABASE_URL": "https://dev-project-ref.supabase.co",
}


def _passthrough_decorator(*_args: Any, **_kwargs: Any):
    def decorator(obj: Any) -> Any:
        return obj

    return decorator


@pytest.fixture(autouse=True)
def _restore_modal_app_module():
    missing = object()
    previous = sys.modules.get("flash.serving.app.modal_app", missing)
    sys.modules.pop("flash.serving.app.modal_app", None)
    try:
        yield
    finally:
        sys.modules.pop("flash.serving.app.modal_app", None)
        if previous is not missing:
            sys.modules["flash.serving.app.modal_app"] = previous


def _import_modal_app(monkeypatch: pytest.MonkeyPatch, *, is_local: bool, environment: str):
    modal_stub = MagicMock(name="modal")
    modal_stub.is_local.return_value = is_local
    modal_stub.config.config.get.return_value = environment
    modal_stub.concurrent.side_effect = _passthrough_decorator
    modal_stub.method.side_effect = _passthrough_decorator
    modal_stub.enter.side_effect = _passthrough_decorator
    modal_stub.exit.side_effect = _passthrough_decorator
    modal_stub.asgi_app.side_effect = _passthrough_decorator
    modal_stub.parameter.return_value = None
    app_mock = MagicMock(name="app")
    app_mock.cls.side_effect = _passthrough_decorator
    app_mock.function.side_effect = _passthrough_decorator
    app_mock.local_entrypoint.side_effect = _passthrough_decorator
    modal_stub.App.return_value = app_mock
    modal_stub.Period.return_value = MagicMock()

    monkeypatch.setitem(sys.modules, "modal", modal_stub)
    monkeypatch.delitem(sys.modules, "flash.serving.app.modal_app", raising=False)
    return importlib.import_module("flash.serving.app.modal_app")


def _clear_deployment_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _DEPLOYMENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("is_local", "modal_is_remote"),
    [(False, "1"), (True, "1")],
    ids=["remote-main", "remote-child"],
)
def test_remote_process_imports_with_only_hf_token_in_dev(
    monkeypatch, is_local: bool, modal_is_remote: str
) -> None:
    _clear_deployment_environment(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "fake")
    monkeypatch.setenv("MODAL_IS_REMOTE", modal_is_remote)

    modal_app = _import_modal_app(monkeypatch, is_local=is_local, environment="dev")

    assert modal_app.SERVING_DEPLOYMENT_MODE == "production"
    assert modal_app.MODAL_ENVIRONMENT == "dev"
    assert modal_app.SERVING_CUSTOM_DOMAIN == ""


def test_deploy_time_development_requires_wiring(monkeypatch) -> None:
    _clear_deployment_environment(monkeypatch)
    for name, value in _DEVELOPMENT_WIRING.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY")

    with pytest.raises(
        ValueError,
        match="development serving requires explicit environment wiring: SUPABASE_SERVICE_ROLE_KEY",
    ):
        _import_modal_app(monkeypatch, is_local=True, environment="dev")


@pytest.mark.parametrize("missing", [None, "AZURE_CLIENT_SECRET", "FLASH_ARTIFACT_BLOB_CONTAINER"])
def test_development_identity_requires_complete_container_credentials(monkeypatch, missing):
    _clear_deployment_environment(monkeypatch)
    values = {
        key: value
        for key, value in _DEVELOPMENT_WIRING.items()
        if key != "FLASH_ARTIFACT_BLOB_SAS_URL"
    }
    values.update(
        {
            "FLASH_ARTIFACT_BLOB_ACCOUNT_URL": "https://test.blob.core.windows.net",
            "FLASH_ARTIFACT_BLOB_CONTAINER": "artifacts-dev",
            "AZURE_TENANT_ID": "tenant",
            "AZURE_CLIENT_ID": "reader",
            "AZURE_CLIENT_SECRET": "reader-secret",
        }
    )
    for name, value in values.items():
        if name != missing:
            monkeypatch.setenv(name, value)
    if missing:
        with pytest.raises(ValueError, match=missing):
            _import_modal_app(monkeypatch, is_local=True, environment="dev")
    else:
        app = _import_modal_app(monkeypatch, is_local=True, environment="dev")
        app.modal.Secret.from_dict.reset_mock()
        app._engine_secret()
        engine = app.modal.Secret.from_dict.call_args.args[0]
        assert engine["AZURE_CLIENT_SECRET"] == "reader-secret"
        assert engine["FLASH_ARTIFACT_BLOB_CONTAINER"] == "artifacts-dev"
        assert "FREESOLO_INTERNAL_KEY" not in engine
        assert "SUPABASE_SERVICE_ROLE_KEY" not in engine


def test_engine_exit_hook_awaits_shutdown(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    _clear_deployment_environment(monkeypatch)
    app = _import_modal_app(monkeypatch, is_local=False, environment="main")
    engine_cls = app._build_engine("L4", "TestShutdownEngine", 64, 48)
    engine = engine_cls()
    engine._shutdown = AsyncMock()
    asyncio.run(engine.shutdown())
    engine._shutdown.assert_awaited_once_with()
    assert app.modal.exit.called


@pytest.mark.parametrize("environment", ["main", "", None])
def test_main_9b_base_has_four_reserved_cpu_proxies_and_no_gpu_fallback(monkeypatch, environment):
    _clear_deployment_environment(monkeypatch)
    module = _import_modal_app(monkeypatch, is_local=False, environment=environment)
    model = "Qwen/Qwen3.5-9B-Base"
    assert len(module.RESERVED_ENGINES) == 4
    assert {cls.reserved_slot for cls in module.RESERVED_ENGINES} == {1, 2, 3, 4}
    assert all(cls.pinned_gpu is None for cls in module.RESERVED_ENGINES)
    assert model not in module.ENGINE_BY_MODEL
    assert model not in module.MODAL_FALLBACK_BY_MODEL


def test_reserved_requests_rotate_without_calling_capacity_service(monkeypatch):
    import asyncio
    import itertools
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    _clear_deployment_environment(monkeypatch)
    module = _import_modal_app(monkeypatch, is_local=False, environment="main")
    pool = module._ModalEnginePool.__new__(module._ModalEnginePool)
    pool._provider_selection = SimpleNamespace(choose=AsyncMock())
    pool._reserved_cycle = itertools.cycle(
        [lambda *, base_model, slot=i: (slot, base_model) for i in range(4)]
    )

    async def check():
        return [await pool._engine_for("Qwen/Qwen3.5-9B-Base") for _ in range(8)]

    results = asyncio.run(check())
    assert [slot for slot, _ in results] == [0, 1, 2, 3, 0, 1, 2, 3]
    pool._provider_selection.choose.assert_not_awaited()


@pytest.mark.parametrize("environment", ["main", "dev"])
@pytest.mark.parametrize("all_models", [False, True])
def test_start_all_resolves_reserved_workers_only_in_main(monkeypatch, environment, all_models):
    _clear_deployment_environment(monkeypatch)
    module = _import_modal_app(monkeypatch, is_local=False, environment=environment)
    model = module.reserved.MODEL
    ordinary = "Qwen/Qwen3.5-4B"
    monkeypatch.setattr(
        "flash.serving.src.engine.model_config.base_models", lambda: [ordinary, model]
    )
    module.start_all(None if all_models else model)
    names = [c.args[1] for c in module.modal.Cls.from_name.call_args_list]
    expected = (
        [module.model_engine_name(model) + f"_ReservedH100{i}" for i in range(1, 5)]
        if environment == "main"
        else [module.model_engine_name(model)]
    )
    assert names == ([module.model_engine_name(ordinary)] if all_models else []) + expected
    instance = module.modal.Cls.from_name.return_value.return_value
    assert instance.health.spawn.call_count == len(expected) + int(all_models)
    assert instance.health.spawn.return_value.get.call_count == len(expected) + int(all_models)


@pytest.mark.parametrize(
    ("runners", "inputs", "backlog"), [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
)
def test_dev_lambda_only_unregister_does_not_wake_idle_pool(monkeypatch, runners, inputs, backlog):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    _clear_deployment_environment(monkeypatch)
    module = _import_modal_app(monkeypatch, is_local=False, environment="dev")
    model = module.reserved.MODEL
    engine = MagicMock()
    engine.health.get_current_stats.aio = AsyncMock(
        return_value=SimpleNamespace(
            num_total_runners=runners, num_running_inputs=inputs, backlog=backlog
        )
    )
    engine.unregister.remote.aio = AsyncMock()
    module.ENGINE_BY_MODEL[model] = MagicMock(return_value=engine)
    pool = module._ModalEnginePool.__new__(module._ModalEnginePool)
    asyncio.run(pool.unregister(model, "org", "adapter", "generation"))
    if runners or inputs or backlog:
        engine.unregister.remote.aio.assert_awaited_once_with("org", "adapter", "generation")
    else:
        engine.unregister.remote.aio.assert_not_awaited()
