"""Import-time deployment identity checks for the hosted Modal serving app."""

from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

_DEPLOYMENT_ENV_VARS = (
    "SERVING_DEPLOYMENT_MODE",
    "SERVING_CUSTOM_DOMAIN",
    "MODAL_IS_REMOTE",
    "LAMBDA_API_KEY",
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


@pytest.mark.parametrize("environment", ["main", "dev", ""])
def test_27b_and_08b_base_use_the_lease_pool_with_modal_fallback(monkeypatch, environment):
    _clear_deployment_environment(monkeypatch)
    module = _import_modal_app(monkeypatch, is_local=False, environment=environment)
    for model in ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-0.8B-Base"):
        assert issubclass(module.ENGINE_BY_MODEL[model], module.LambdaEngineProxy)
        assert model in module.MODAL_FALLBACK_BY_MODEL
        assert model not in module.reserved.MODELS
    # Only main declares the reserved 9B Base fleet: its secret and its watchdog.
    main = environment != "dev"
    secret_names = {c.args[0] for c in module.modal.Secret.from_name.call_args_list if c.args}
    assert {n for n in secret_names if "h100" in n} == (
        {module.reserved.SECRET_NAME} if main else set()
    )
    scheduled = [
        call
        for call in module.app.function.call_args_list
        if call.kwargs.get("schedule") is not None and call.kwargs.get("cpu") == 0.25
    ]
    assert len(scheduled) == int(main)
    assert all(c.kwargs["max_containers"] == 1 for c in scheduled)


def _reserved_secret_refs(module):
    return {
        c.args[0]
        for c in module.modal.Secret.from_name.call_args_list
        if c.args and "h100" in c.args[0]
    }


def _watchdog_schedules(module):
    return [
        call
        for call in module.app.function.call_args_list
        if call.kwargs.get("schedule") is not None and call.kwargs.get("cpu") == 0.25
    ]


def _assert_on_demand(module):
    assert module.RESERVED_ENGINES == ()
    assert _reserved_secret_refs(module) == set()
    assert _watchdog_schedules(module) == []
    for model in ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-9B-Base"):
        assert issubclass(module.ENGINE_BY_MODEL[model], module.LambdaEngineProxy)
        assert model in module.MODAL_FALLBACK_BY_MODEL
    assert module._reserved_env == {"FLASH_RESERVED_H100_ACTIVE": "0"}


@pytest.mark.parametrize("environment", ["main", ""])
def test_main_without_the_reserved_secret_deploys_without_referencing_it(monkeypatch, environment):
    """Deploy run 45330840461 failed: the new reserved secret did not exist in main yet."""
    _clear_deployment_environment(monkeypatch)
    monkeypatch.setenv("FLASH_RESERVED_H100_ACTIVE", "0")
    _assert_on_demand(_import_modal_app(monkeypatch, is_local=False, environment=environment))


@pytest.mark.parametrize("exists", [True, False])
def test_deploy_decides_from_the_secret_lookup_and_bakes_the_decision(monkeypatch, exists):
    from flash.serving.lambda_cloud import reserved

    _clear_deployment_environment(monkeypatch)
    monkeypatch.delenv("FLASH_RESERVED_H100_ACTIVE")
    looked_up = []
    monkeypatch.setattr(reserved, "secret_exists", lambda env: looked_up.append(env) or exists)
    module = _import_modal_app(monkeypatch, is_local=False, environment="main")
    assert looked_up == ["main"]
    if not exists:
        _assert_on_demand(module)
        return
    assert module._reserved_env == {"FLASH_RESERVED_H100_ACTIVE": "1"}
    assert len(module.RESERVED_ENGINES) == 12
    assert _reserved_secret_refs(module) == {reserved.SECRET_NAME}
    assert len(_watchdog_schedules(module)) == 1


def test_control_app_bakes_the_same_reserved_decision(monkeypatch):
    from flash.serving.lambda_cloud import reserved

    monkeypatch.setenv("FLASH_RESERVED_H100_ACTIVE", "0")
    modal_stub = MagicMock(name="modal")
    modal_stub.config.config.get.return_value = "main"
    monkeypatch.setitem(sys.modules, "modal", modal_stub)
    monkeypatch.delitem(sys.modules, "flash.serving.lambda_cloud.control_app", raising=False)
    monkeypatch.setattr(
        "flash.serving.lambda_cloud.modal_control.register_control", lambda *a: (None, None)
    )
    importlib.import_module("flash.serving.lambda_cloud.control_app")
    secret = modal_stub.Secret.from_dict.call_args.args[0]
    assert secret[reserved.ACTIVE_ENV] == "0"
    # The shared image carries it too, so the secret-less reaper never looks the secret up.
    image_env = modal_stub.Image.debian_slim.return_value.pip_install.return_value.env
    image_env.assert_called_once_with({reserved.ACTIVE_ENV: "0"})


@pytest.mark.parametrize("environment", ["main", "", "dev"])
def test_main_serves_9b_base_only_from_twelve_reserved_slots(monkeypatch, environment):
    _clear_deployment_environment(monkeypatch)
    module = _import_modal_app(monkeypatch, is_local=False, environment=environment)
    base = "Qwen/Qwen3.5-9B-Base"
    if environment == "dev":
        assert module.RESERVED_ENGINES == ()
        assert base in module.ENGINE_BY_MODEL
        assert base in module.MODAL_FALLBACK_BY_MODEL
        return
    engines = module.RESERVED_ENGINES
    assert [cls.reserved_slot for cls in engines] == list(range(1, 13))
    assert [cls.__name__ for cls in engines] == [
        module.model_engine_name(base) + f"_ReservedH100{slot}" for slot in range(1, 13)
    ]
    assert all(issubclass(cls, module.reserved.ReservedLambdaEngineProxy) for cls in engines)
    assert all(cls.pinned_gpu is None for cls in engines)
    assert base not in module.ENGINE_BY_MODEL
    assert base not in module.MODAL_FALLBACK_BY_MODEL
    assert "Qwen/Qwen3.8-27B" in module.ENGINE_BY_MODEL


@pytest.mark.parametrize(("environment", "scheduled"), [("main", True), ("", True), ("dev", False)])
def test_acquisition_is_scheduled_only_in_main(monkeypatch, environment, scheduled):
    _clear_deployment_environment(monkeypatch)
    monkeypatch.setenv("FLASH_RESERVED_H100_ACTIVE", "0")
    module = _import_modal_app(monkeypatch, is_local=False, environment=environment)
    assert hasattr(module, "reserved_h100_acquire") is scheduled
    # Reserved serving stays off: 9B Base keeps its on-demand pool while GPUs are acquired.
    assert "Qwen/Qwen3.5-9B-Base" in module.ENGINE_BY_MODEL
    assert _reserved_secret_refs(module) == set()


def test_reserved_requests_rotate_over_the_9b_base_slots(monkeypatch):
    from types import SimpleNamespace

    _clear_deployment_environment(monkeypatch)
    module = _import_modal_app(monkeypatch, is_local=False, environment="main")
    pool = module._ModalEnginePool.__new__(module._ModalEnginePool)
    pool._provider_selection = SimpleNamespace(choose=AsyncMock())
    monkeypatch.setattr(
        module,
        "RESERVED_ENGINES",
        tuple(lambda *, base_model, slot=i + 1: (slot, base_model) for i in range(12)),
    )
    pool._reserved_health = module.reserved_watchdog.HealthView(
        SimpleNamespace(get=AsyncMock(return_value=None)), module.reserved.SLOT_MODELS
    )

    async def picks(model, n):
        return [(await pool._engine_for(model))[0] for _ in range(n)]

    assert asyncio.run(picks("Qwen/Qwen3.5-9B-Base", 14)) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        1,
        2,
    ]
    pool._provider_selection.choose.assert_not_awaited()
