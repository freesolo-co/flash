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
    "PLATFORM_BACKEND_URL",
    "SUPABASE_PROJECT_REF",
    "SUPABASE_PROJECT_REF_DEV",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_URL",
)
_DEVELOPMENT_WIRING = {
    "FREESOLO_INTERNAL_KEY": "dev-internal-key",
    "HF_TOKEN": "dev-hf-key",
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
