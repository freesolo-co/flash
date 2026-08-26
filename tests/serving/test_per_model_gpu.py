"""Per-model GPU tier routing: each base model's engine runs on the GPU class from the catalog.

Modal fixes a class's GPU and concurrency at decoration time, so the serving app registers one
``LoraEngine`` ``@app.cls`` per distinct (GPU tier, max_inputs) key and dispatches each base model to
its class (9b -> l40s, 35b -> h200). modal_app imports the ``modal`` sdk
at module top, which isn't installed offline, so we stub it just enough to import the module and
reach the built engine classes.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from flash.serving.src.engine.model_config import (
    base_models,
    configured_warm_container_floor,
    gpu_for,
)


def _passthrough_decorator(*_a: Any, **_k: Any):
    def deco(obj: Any) -> Any:
        return obj

    return deco


@pytest.fixture(scope="module")
def modal_app_module(load_modal_app_under_stub):
    modal_stub = MagicMock(name="modal")
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
    return load_modal_app_under_stub(modal_stub)


_DEVELOPMENT_CUSTOM_DOMAIN = "serve-dev.freesolo.co"
_DEVELOPMENT_WIRING = {
    "FREESOLO_INTERNAL_KEY": "dev-internal-key",
    "HF_TOKEN": "dev-hf-key",
    "PLATFORM_BACKEND_URL": "https://api-dev.freesolo.co",
    "SUPABASE_PROJECT_REF": "production-project-ref",
    "SUPABASE_PROJECT_REF_DEV": "dev-project-ref",
    "SUPABASE_SERVICE_ROLE_KEY": "dev-service-key",
    "SUPABASE_URL": "https://dev-project-ref.supabase.co",
    "TEST_MODAL_ENVIRONMENT": "dev",
}
_DEPLOYMENT_ENV_VARS = (
    "SERVING_DEPLOYMENT_MODE",
    "SERVING_CUSTOM_DOMAIN",
    "FREESOLO_INTERNAL_KEY",
    "HF_TOKEN",
    "PLATFORM_BACKEND_URL",
    "SUPABASE_PROJECT_REF",
    "SUPABASE_PROJECT_REF_DEV",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_URL",
    "TEST_MODAL_ENVIRONMENT",
    "TEST_RAISE_ON_DOTENV",
)
_MODAL_APP_IMPORT_PROBE = """
import json
import os
import types
from unittest.mock import MagicMock


def passthrough_decorator(*_args, **_kwargs):
    def decorator(obj):
        return obj

    return decorator


modal_stub = MagicMock(name="modal")
modal_stub.concurrent.side_effect = passthrough_decorator
modal_stub.method.side_effect = passthrough_decorator
modal_stub.enter.side_effect = passthrough_decorator
modal_stub.asgi_app.side_effect = passthrough_decorator
modal_stub.parameter.return_value = None
app_mock = MagicMock(name="app")
app_mock.cls.side_effect = passthrough_decorator
app_mock.function.side_effect = passthrough_decorator
app_mock.local_entrypoint.side_effect = passthrough_decorator
modal_stub.App.return_value = app_mock
modal_stub.Period.return_value = MagicMock()
modal_stub.config.config.get.return_value = os.environ.get("TEST_MODAL_ENVIRONMENT", "")
sys.modules["modal"] = modal_stub
dotenv_stub = types.ModuleType("dotenv")


def load_dotenv(*_args, **_kwargs):
    if os.environ.get("TEST_RAISE_ON_DOTENV") == "1":
        raise AssertionError("development mode must not load the repository root dotenv")
    return False


dotenv_stub.load_dotenv = load_dotenv
sys.modules["dotenv"] = dotenv_stub

import flash.serving.app.modal_app as modal_app

print(json.dumps({
    "mode": modal_app.SERVING_DEPLOYMENT_MODE,
    "custom_domain": modal_app.SERVING_CUSTOM_DOMAIN,
    "asgi_custom_domains": modal_stub.asgi_app.call_args.kwargs["custom_domains"],
    "min_containers": modal_app.MIN_CONTAINERS,
}))
"""


def _probe_modal_app_import(**environment: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in _DEPLOYMENT_ENV_VARS:
        env.pop(name, None)
    for name, value in environment.items():
        if value is not None:
            env[name] = value
    return subprocess.run(
        [sys.executable, "-c", "import sys\n" + _MODAL_APP_IMPORT_PROBE],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _successful_import_payload(
    result: subprocess.CompletedProcess[str],
) -> dict[str, str | int | list[str] | None]:
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_deployment_mode_unset_defaults_to_production_with_warm_floor() -> None:
    payload = _successful_import_payload(_probe_modal_app_import())
    assert payload == {
        "mode": "production",
        "custom_domain": "",
        "asgi_custom_domains": None,
        "min_containers": 1,
    }


def test_explicit_production_mode_accepts_production_custom_domain() -> None:
    payload = _successful_import_payload(
        _probe_modal_app_import(
            SERVING_DEPLOYMENT_MODE="production",
            SERVING_CUSTOM_DOMAIN="serve.freesolo.co",
        )
    )
    assert payload == {
        "mode": "production",
        "custom_domain": "serve.freesolo.co",
        "asgi_custom_domains": ["serve.freesolo.co"],
        "min_containers": 1,
    }


def test_development_mode_accepts_exact_custom_domain_with_warm_floor() -> None:
    payload = _successful_import_payload(
        _probe_modal_app_import(
            SERVING_DEPLOYMENT_MODE="development",
            SERVING_CUSTOM_DOMAIN=_DEVELOPMENT_CUSTOM_DOMAIN,
            **_DEVELOPMENT_WIRING,
        )
    )
    assert payload == {
        "mode": "development",
        "custom_domain": _DEVELOPMENT_CUSTOM_DOMAIN,
        "asgi_custom_domains": [_DEVELOPMENT_CUSTOM_DOMAIN],
        "min_containers": 1,
    }


def test_development_mode_never_loads_the_repository_root_dotenv() -> None:
    payload = _successful_import_payload(
        _probe_modal_app_import(
            SERVING_DEPLOYMENT_MODE="development",
            SERVING_CUSTOM_DOMAIN=_DEVELOPMENT_CUSTOM_DOMAIN,
            TEST_RAISE_ON_DOTENV="1",
            **_DEVELOPMENT_WIRING,
        )
    )
    assert payload["mode"] == "development"


def test_development_mode_requires_the_dev_modal_environment() -> None:
    result = _probe_modal_app_import(
        SERVING_DEPLOYMENT_MODE="development",
        SERVING_CUSTOM_DOMAIN=_DEVELOPMENT_CUSTOM_DOMAIN,
        **{
            **_DEVELOPMENT_WIRING,
            "TEST_MODAL_ENVIRONMENT": None,
        },
    )
    assert result.returncode != 0
    assert "development serving must target Modal environment 'dev'" in result.stderr


def test_production_mode_rejects_the_dev_modal_environment() -> None:
    result = _probe_modal_app_import(
        SERVING_DEPLOYMENT_MODE="production",
        TEST_MODAL_ENVIRONMENT="dev",
    )
    assert result.returncode != 0
    assert "production serving must not target Modal environment 'dev'" in result.stderr


def test_development_mode_rejects_production_backend_wiring() -> None:
    result = _probe_modal_app_import(
        SERVING_DEPLOYMENT_MODE="development",
        SERVING_CUSTOM_DOMAIN=_DEVELOPMENT_CUSTOM_DOMAIN,
        **{
            **_DEVELOPMENT_WIRING,
            "PLATFORM_BACKEND_URL": "https://api.freesolo.co",
        },
    )
    assert result.returncode != 0
    assert "PLATFORM_BACKEND_URL must be https://api-dev.freesolo.co" in result.stderr


def test_development_mode_rejects_mismatched_supabase_project() -> None:
    result = _probe_modal_app_import(
        SERVING_DEPLOYMENT_MODE="development",
        SERVING_CUSTOM_DOMAIN=_DEVELOPMENT_CUSTOM_DOMAIN,
        **{
            **_DEVELOPMENT_WIRING,
            "SUPABASE_URL": "https://production-project-ref.supabase.co",
        },
    )
    assert result.returncode != 0
    assert "SUPABASE_URL must match SUPABASE_PROJECT_REF_DEV" in result.stderr


def test_development_mode_rejects_equal_supabase_project_refs() -> None:
    result = _probe_modal_app_import(
        SERVING_DEPLOYMENT_MODE="development",
        SERVING_CUSTOM_DOMAIN=_DEVELOPMENT_CUSTOM_DOMAIN,
        **{
            **_DEVELOPMENT_WIRING,
            "SUPABASE_PROJECT_REF": "dev-project-ref",
        },
    )
    assert result.returncode != 0
    assert "production and development Supabase project refs must differ" in result.stderr


@pytest.mark.parametrize(
    "custom_domain",
    ["", "serve.freesolo.co", "dev-serving.example.com"],
)
def test_development_mode_rejects_incorrect_custom_domain(custom_domain: str) -> None:
    result = _probe_modal_app_import(
        SERVING_DEPLOYMENT_MODE="development",
        SERVING_CUSTOM_DOMAIN=custom_domain,
        **_DEVELOPMENT_WIRING,
    )
    assert result.returncode != 0
    assert (
        f"development SERVING_CUSTOM_DOMAIN must be {_DEVELOPMENT_CUSTOM_DOMAIN}" in result.stderr
    )


def test_invalid_deployment_mode_fails_import() -> None:
    result = _probe_modal_app_import(SERVING_DEPLOYMENT_MODE="staging")
    assert result.returncode != 0
    assert "SERVING_DEPLOYMENT_MODE must be 'production' or 'development'" in result.stderr


def test_engine_secret_allowlists_only_hf_token(modal_app_module, monkeypatch) -> None:
    values = {
        "HF_TOKEN": "hf-secret",
        "SERVING_DEPLOYMENT_MODE": "development",
        "SERVING_CUSTOM_DOMAIN": _DEVELOPMENT_CUSTOM_DOMAIN,
        "FREESOLO_INTERNAL_KEY": "internal-secret",
        "PLATFORM_BACKEND_URL": "https://api-dev.freesolo.co",
        "FREESOLO_DEPLOYMENT_SHA": "deployment-sha",
        "FREESOLO_DEPLOYMENT_ID": "deployment-id",
        "SUPABASE_PROJECT_REF": "production-project-ref",
        "SUPABASE_PROJECT_REF_DEV": "dev-project-ref",
        "SUPABASE_URL": "https://dev-project-ref.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "supabase-secret",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    modal_app_module.modal.Secret.from_dict.reset_mock()
    modal_app_module._engine_secret()
    engine_values = modal_app_module.modal.Secret.from_dict.call_args.args[0]

    assert "SUPABASE_SERVICE_ROLE_KEY" not in engine_values
    assert "SUPABASE_URL" not in engine_values
    assert engine_values == {"HF_TOKEN": "hf-secret"}


def test_router_secret_keeps_supabase_credentials(modal_app_module, monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "supabase-secret")

    modal_app_module.modal.Secret.from_dict.reset_mock()
    modal_app_module._runtime_secret()
    router_values = modal_app_module.modal.Secret.from_dict.call_args.args[0]

    assert router_values["SUPABASE_URL"] == "https://project.supabase.co"
    assert router_values["SUPABASE_SERVICE_ROLE_KEY"] == "supabase-secret"


def test_cold_engine_resolves_forwarded_adapter_record(modal_app_module, tmp_path) -> None:
    from flash.serving.src.store.registry import AdapterRegistry

    revision = "a" * 40
    adapter_id = f"run-1@step-1.{revision}"
    record_dict = {
        "adapter_id": adapter_id,
        "repo_id": "org/private-adapter",
        "base_model": "Qwen/Qwen3.5-9B",
        "org_id": "org-1",
        "checkpoint": "run-1/step-1",
        "private": True,
        "thinking": False,
        "status": "ready",
        "metadata": {
            "record_type": "revision",
            "run_id": "run-1",
            "checkpoint_step": 1,
            "hf_revision": revision,
        },
    }
    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine.base_model = record_dict["base_model"]
    engine.registry = AdapterRegistry()
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    resolved_path = tmp_path / "adapter"
    resolved_request = object()

    async def ensure_adapter_local(record):
        assert record.adapter_id == adapter_id
        return resolved_path

    def cached_lora_request(record, path):
        assert record.adapter_id == adapter_id
        assert path == resolved_path
        return resolved_request

    engine._ensure_adapter_local_locked = ensure_adapter_local
    engine._cached_lora_request_locked = cached_lora_request
    assert engine.registry.list_ready() == []

    lora_request, record = asyncio.run(engine._lora_request(adapter_id, record_dict))

    assert lora_request is resolved_request
    assert record.adapter_id == adapter_id
    assert engine.registry.get(adapter_id) == record


def test_lora_engine_import_does_not_require_pillow() -> None:
    code = """
import builtins
import sys

real_import = builtins.__import__


def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "PIL" or name.startswith("PIL."):
        raise ModuleNotFoundError("PIL is unavailable on the deploy runner")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = blocked_import
import flash.serving.src.engine.lora_engine

assert "PIL" not in sys.modules
assert "flash.serving.src.io.multimodal" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_one_engine_class_per_supported_model(modal_app_module):
    assert set(modal_app_module.ENGINE_BY_MODEL) == set(base_models())
    assert len(modal_app_module.ENGINE_BY_MODEL) == len(base_models())
    assert len({id(engine) for engine in modal_app_module.ENGINE_BY_MODEL.values()}) == len(
        base_models()
    )


def test_engine_concurrency_comes_from_validated_model_policy(modal_app_module):
    assert modal_app_module._engine_concurrency("Qwen/Qwen3.5-9B") == (8, 6)
    assert modal_app_module._engine_concurrency("Qwen/Qwen3.6-35B-A3B") == (8, 6)
    with pytest.raises(ValueError, match="Unsupported base model"):
        modal_app_module._engine_concurrency("unsupported/model")


def test_class_names_are_deterministic_distinct_and_modal_safe(modal_app_module):
    names = [modal_app_module._engine_class_name(model) for model in base_models()]
    assert len(set(names)) == len(base_models())
    assert names == [modal_app_module._engine_class_name(model) for model in base_models()]
    assert all(name.startswith("LoraEngine_") for name in names)
    assert all(name.replace("_", "").isalnum() for name in names)


@pytest.mark.parametrize("base_model", base_models())
def test_each_model_routes_to_its_exact_identity(modal_app_module, base_model: str):
    engine = modal_app_module.ENGINE_BY_MODEL[base_model]
    assert modal_app_module._engine_cls_for(base_model) is engine
    assert engine.pinned_gpu == gpu_for(base_model)


def test_pending_qwen38_candidate_has_no_active_engine_dispatch(modal_app_module):
    with pytest.raises(ValueError, match="Unsupported base model"):
        gpu_for("Qwen/Qwen3.8-27B")
    with pytest.raises(ValueError, match="Unsupported base model"):
        modal_app_module._engine_cls_for("Qwen/Qwen3.8-27B")


def test_unknown_base_model_is_rejected_before_engine_dispatch(modal_app_module):
    """An unseen base model must not silently dispatch to the L4 default tier."""
    with pytest.raises(ValueError, match="Unsupported base model"):
        modal_app_module._engine_cls_for("Qwen/Qwen3.5-99B")


def test_model_classes_inherit_the_shared_impl(modal_app_module):
    for engine in modal_app_module.ENGINE_BY_MODEL.values():
        assert issubclass(engine, modal_app_module._LoraEngineImpl)
        for impl in (
            "_load",
            "_register",
            "_generate",
            "_stream_generate",
            "_unregister",
            "_health",
        ):
            assert hasattr(engine, impl)
        for entry in (
            "load",
            "register",
            "generate",
            "stream_generate_call",
            "unregister",
            "health",
        ):
            assert entry in engine.__dict__
        assert "stream_generate" not in engine.__dict__


def test_model_class_identity_is_fixed_before_decoration(modal_app_module):
    for model, engine in modal_app_module.ENGINE_BY_MODEL.items():
        class_name = modal_app_module._engine_class_name(model)
        assert engine.__name__ == class_name
        assert engine.__qualname__ == class_name
        assert getattr(modal_app_module, class_name) is engine


def test_installed_modal_registers_unparameterized_exact_warm_classes() -> None:
    code = """
import inspect
import json
import modal

import flash.serving.app.modal_app as modal_app

app_inner = next(
    value for name, value in vars(modal_app.app).items() if name.startswith("_sync_original")
)
registered = app_inner._local_state_attr.functions
observed = {}
for model, engine_class in modal_app.ENGINE_BY_MODEL.items():
    instance = engine_class()
    service = registered[f"{engine_class.__name__}.*"]
    registration = inspect.getclosurevars(service._load).nonlocals
    user_class = registration["info"].user_cls
    observed[model] = {
        "class_name": engine_class.__name__,
        "parameters": instance._get_parameter_values(),
        "instance_model": instance.base_model,
        "class_model": user_class.base_model,
        "min_containers": registration["min_containers"],
        "buffer_containers": registration["buffer_containers"],
        "max_containers": registration["max_containers"],
    }
router_registration = inspect.getclosurevars(registered["router"]._load).nonlocals
router_policy = {
    "min_containers": router_registration["min_containers"],
    "buffer_containers": router_registration["buffer_containers"],
    "max_containers": router_registration["max_containers"],
}
print(json.dumps({
    "modal_version": modal.__version__,
    "engines": observed,
    "router": router_policy,
}, sort_keys=True))
"""
    env = os.environ.copy()
    env["SERVING_DEPLOYMENT_MODE"] = "production"
    env.pop("SERVING_CUSTOM_DOMAIN", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["modal_version"] == "1.5.4"
    assert payload["router"] == {
        "min_containers": 1,
        "buffer_containers": 1,
        "max_containers": None,
    }
    observed = payload["engines"]
    assert set(observed) == set(base_models())
    assert len({entry["class_name"] for entry in observed.values()}) == len(base_models())
    for model, entry in observed.items():
        assert entry["parameters"] == {}
        assert entry["instance_model"] == model
        assert entry["class_model"] == model
        assert entry["min_containers"] == 1
        assert entry["buffer_containers"] == 1
        assert entry["max_containers"] is None


def test_changed_hosted_sources_describe_warm_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    sources = (
        root / "flash/serving/app/modal_app.py",
        root / "flash/serving/src/http/context.py",
        root / "flash/serving/src/engine/lora_engine.py",
        root / "flash/serving/src/http/routing.py",
        root / "flash/serving/app/README.md",
    )
    forbidden = (
        "scale" + "-to-zero",
        "scale" + "-from-zero",
        "scaled" + "-to-zero",
        "zero" + "-floor",
        "leave gpu engines at " + "zero",
        "demand" + "-only",
        "min_containers" + " = 0",
    )

    hits = {
        str(path.relative_to(root)): phrase
        for path in sources
        for phrase in forbidden
        if phrase in path.read_text().lower()
    }
    assert hits == {}

    readme = (root / "flash/serving/app/README.md").read_text(encoding="utf-8")
    assert len(base_models()) == 2
    assert "current two-model catalog" in readme
    assert f"warm floor of {configured_warm_container_floor()} GPU containers" in readme
    assert "no flash-configured maximum" in readme.lower()
    assert "`max_inputs=36`" in readme
    assert "`target_inputs=27`" in readme


def test_health_reports_pinned_gpu_over_derived_tier(modal_app_module):
    """_health reports the class's actual pinned tier rather than the derived model tier."""
    impl = modal_app_module._LoraEngineImpl

    class _Fake:
        pinned_gpu = "A100-80GB"
        base_model = "Qwen/Qwen3.5-9B"
        registry = type("R", (), {"list_ready": lambda self: []})()

    health = impl._health(_Fake())
    assert (
        health["configured_gpu"] == "A100-80GB"
    )  # actual pinned tier, not the derived catalog tier

    class _Bare:
        # No pinned_gpu (the shared impl used directly) -> fall back to the expected tier.
        base_model = "Qwen/Qwen3.5-9B"
        registry = type("R", (), {"list_ready": lambda self: []})()

    assert impl._health(_Bare())["configured_gpu"] == "L40S"


def test_health_reports_effective_max_model_len_override(modal_app_module):
    """_health must advertise the EFFECTIVE context limit. The 35B MoE overrides max_model_len via its
    per-model engine override, so health must report that — not the global default — or monitoring
    misreports the context window vLLM actually serves."""
    from flash.serving.src.store import settings as cfg

    impl = modal_app_module._LoraEngineImpl

    class _Big:
        base_model = "Qwen/Qwen3.6-35B-A3B"  # engine override max_model_len=32768
        registry = type("R", (), {"list_ready": lambda self: []})()

    assert impl._health(_Big())["max_model_len"] == 32768  # the override, not cfg.MAX_MODEL_LEN

    class _Dense:
        base_model = "Qwen/Qwen3.5-9B"  # per-model 32k override
        registry = type("R", (), {"list_ready": lambda self: []})()

    assert impl._health(_Dense())["max_model_len"] == 32768
    assert impl._health(_Dense())["max_model_len"] != cfg.MAX_MODEL_LEN


def test_start_all_raises_after_any_engine_fails(modal_app_module, monkeypatch):
    from flash.serving.src.engine import model_config

    mod = modal_app_module
    monkeypatch.setattr(model_config, "base_models", lambda: ["ok", "boom"])
    monkeypatch.setattr(model_config, "gpu_for", lambda _bm: "L4")
    monkeypatch.setattr(mod, "_engine_class_name", lambda model: f"LoraEngine_{model}")
    spawned: list[str] = []

    class _Handle:
        def __init__(self, base_model: str) -> None:
            self.base_model = base_model

        def get(self, timeout: int = 0) -> str:
            if self.base_model == "boom":
                raise RuntimeError("cold start failed")
            return "ok"

    def _from_name(_app_name: str, cls_name: str):
        base_model = cls_name.removeprefix("LoraEngine_")

        def _factory():
            class _Health:
                @staticmethod
                def spawn():
                    spawned.append(base_model)
                    return _Handle(base_model)

            return type("Instance", (), {"health": _Health()})()

        return _factory

    monkeypatch.setattr(mod.modal.Cls, "from_name", _from_name)

    with pytest.raises(RuntimeError, match="boom"):
        mod.start_all()

    assert sorted(spawned) == ["boom", "ok"]


def test_warm_pool_dispatches_inference_and_registration(modal_app_module, monkeypatch):
    """The pool dispatches every call through the model's immutable warm class."""
    mod = modal_app_module
    bound_models: list[str] = []
    generate_calls: list[tuple[dict, dict, str | None, str, float]] = []
    stream_calls: list[tuple[dict, dict, str | None, str, float]] = []
    register_calls: list[tuple[dict, str | None]] = []

    class _Dump:
        def __init__(
            self,
            value: dict,
            *,
            deployment_generation: str | None = None,
            generation_id: str | None = None,
            pre_header_dispatch_deadline: float = 123.0,
        ) -> None:
            self.value = value
            self.deployment_generation = deployment_generation
            self.generation_id = generation_id
            self._pre_header_dispatch_deadline = pre_header_dispatch_deadline

        def model_dump(self, *, by_alias: bool) -> dict:
            assert by_alias is True
            return self.value

    async def _generate(
        payload: dict,
        record: dict,
        checkpoint: str | None,
        generation_id: str,
        deadline: float,
    ) -> dict:
        generate_calls.append((payload, record, checkpoint, generation_id, deadline))
        return {"ok": True}

    class _Channel:
        def __init__(
            self,
            *,
            spawn_method: Any,
            payload_dict: dict,
            record_dict: dict,
            expected_checkpoint: str | None,
            generation_id: str,
            dispatch_deadline_unix: float,
            invocation_nonce: str,
        ) -> None:
            assert spawn_method is stream_call_method
            assert invocation_nonce
            stream_calls.append(
                (
                    payload_dict,
                    record_dict,
                    expected_checkpoint,
                    generation_id,
                    dispatch_deadline_unix,
                )
            )

        def __aiter__(self):
            return self._events()

        async def _events(self):
            yield {"delta": "hello"}
            yield {"delta": " world"}

    async def _register(record: dict, deployment_generation: str | None) -> None:
        register_calls.append((record, deployment_generation))

    class _Call:
        def __init__(self, result: dict) -> None:
            self.result = result
            self.cancelled = False
            self.get = types.SimpleNamespace(aio=self._get)
            self.cancel = types.SimpleNamespace(aio=self._cancel)

        async def _get(self) -> dict:
            return self.result

        async def _cancel(self) -> None:
            self.cancelled = True

    async def _spawn_generate(*args: Any) -> _Call:
        return _Call(await _generate(*args))

    stream_call_method = types.SimpleNamespace(spawn=types.SimpleNamespace(aio=None))
    monkeypatch.setattr(
        "flash.serving.src.stream_channel.client.CancellableStreamChannel",
        _Channel,
    )

    class _FakeEngine:
        generate = types.SimpleNamespace(spawn=types.SimpleNamespace(aio=_spawn_generate))
        stream_generate_call = stream_call_method
        register = types.SimpleNamespace(remote=types.SimpleNamespace(aio=_register))

    engine = _FakeEngine()

    def _bind():
        bound_models.append("Qwen/Qwen3.5-9B")
        return engine

    monkeypatch.setattr(mod, "_engine_cls_for", lambda _base_model: _bind)
    pool = mod._ModalEnginePool()
    generation_id = "fsgen-00000000000000000000000000000001"
    deadline = time.time() + 60
    payload = _Dump(
        {"messages": [{"role": "user", "content": "hello"}]},
        generation_id=generation_id,
        pre_header_dispatch_deadline=deadline,
    )
    record = _Dump(
        {"adapter_id": "run@step-1.sha"},
        deployment_generation="generation-1",
    )

    result = asyncio.run(
        pool.generate(
            "Qwen/Qwen3.5-9B",
            payload,
            record,
            expected_checkpoint="step-1",
        )
    )

    async def _collect_stream() -> list[dict]:
        return [
            event
            async for event in pool.stream_generate(
                "Qwen/Qwen3.5-9B",
                payload,
                record,
                expected_checkpoint="step-1",
            )
        ]

    stream_events = asyncio.run(_collect_stream())
    asyncio.run(pool.register("Qwen/Qwen3.5-9B", record))

    assert result == {"ok": True}
    assert stream_events == [{"delta": "hello"}, {"delta": " world"}]
    assert bound_models == [
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-9B",
    ]
    expected_inference_call = (
        {"messages": [{"role": "user", "content": "hello"}]},
        {
            "adapter_id": "run@step-1.sha",
            "deployment_generation": "generation-1",
        },
        "step-1",
        generation_id,
        deadline,
    )
    assert generate_calls == [expected_inference_call]
    assert stream_calls == [expected_inference_call]
    assert register_calls == [
        (
            {
                "adapter_id": "run@step-1.sha",
                "deployment_generation": "generation-1",
            },
            "generation-1",
        )
    ]


def test_modal_stream_returns_channel_directly_and_close_reaches_channel(
    modal_app_module, monkeypatch
) -> None:
    closed = 0
    captured: dict[str, Any] = {}
    stream_call_method = types.SimpleNamespace(spawn=types.SimpleNamespace(aio=None))

    class _Channel:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> dict[str, str]:
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return {"type": "ready"}

        async def aclose(self) -> None:
            nonlocal closed
            closed += 1

    class _Engine:
        stream_generate_call = stream_call_method

    monkeypatch.setattr(
        "flash.serving.src.stream_channel.client.CancellableStreamChannel",
        _Channel,
    )
    monkeypatch.setattr(modal_app_module, "_engine_cls_for", lambda _base_model: lambda: _Engine())
    deadline = time.time() + 60
    payload = types.SimpleNamespace(
        generation_id="fsgen-00000000000000000000000000000001",
        _pre_header_dispatch_deadline=deadline,
        model_dump=lambda **_kwargs: {"adapter_id": "revision"},
    )
    record = types.SimpleNamespace(
        deployment_generation="generation-1",
        model_dump=lambda **_kwargs: {"adapter_id": "revision"},
    )

    stream = modal_app_module._ModalEnginePool().stream_generate(
        "Qwen/Qwen3.5-9B", payload, record, expected_checkpoint="run/step-1"
    )
    assert isinstance(stream, _Channel)

    async def scenario() -> None:
        assert await anext(stream) == {"type": "ready"}
        await stream.aclose()

    asyncio.run(scenario())

    assert closed == 1
    assert captured == {
        "spawn_method": stream_call_method,
        "payload_dict": {"adapter_id": "revision"},
        "record_dict": {
            "adapter_id": "revision",
            "deployment_generation": "generation-1",
        },
        "expected_checkpoint": "run/step-1",
        "generation_id": payload.generation_id,
        "dispatch_deadline_unix": deadline,
        "invocation_nonce": captured["invocation_nonce"],
    }
    assert captured["invocation_nonce"]


@pytest.mark.parametrize("generation_id", [None, "", " spaced ", "x" * 513])
def test_modal_stream_rejects_missing_or_invalid_generation_id(
    modal_app_module, generation_id: Any
) -> None:
    payload = types.SimpleNamespace(
        generation_id=generation_id,
        _pre_header_dispatch_deadline=time.time() + 60,
    )
    with pytest.raises(RuntimeError, match="valid generation id"):
        modal_app_module._ModalEnginePool().stream_generate("Qwen/Qwen3.5-9B", payload, object())


@pytest.mark.parametrize("deadline", [None, True, float("nan"), float("inf"), "123"])
def test_modal_stream_rejects_missing_or_invalid_deadline(modal_app_module, deadline: Any) -> None:
    payload = types.SimpleNamespace(
        generation_id="fsgen-00000000000000000000000000000001",
        _pre_header_dispatch_deadline=deadline,
    )
    with pytest.raises(RuntimeError, match="valid pre-header dispatch deadline"):
        modal_app_module._ModalEnginePool().stream_generate("Qwen/Qwen3.5-9B", payload, object())


def test_blocked_modal_spawn_expires_with_absolute_deadline(modal_app_module) -> None:
    spawn_cancelled = False

    async def blocked_spawn() -> Any:
        nonlocal spawn_cancelled
        try:
            await asyncio.Event().wait()
        finally:
            spawn_cancelled = True

    method = types.SimpleNamespace(spawn=types.SimpleNamespace(aio=blocked_spawn))

    async def scenario() -> None:
        with pytest.raises(modal_app_module.PreHeaderDispatchExpired):
            await modal_app_module._spawn_modal_call(method, time.time() + 0.01)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert spawn_cancelled is True


def test_blocked_modal_result_expires_and_cancels_exact_call(modal_app_module) -> None:
    class _Call:
        def __init__(self) -> None:
            self.cancelled = 0
            self.get = types.SimpleNamespace(aio=self._get)
            self.cancel = types.SimpleNamespace(aio=self._cancel)

        async def _get(self) -> dict[str, bool]:
            await asyncio.Event().wait()
            return {"ok": True}

        async def _cancel(self) -> None:
            self.cancelled += 1

    call = _Call()

    async def scenario() -> None:
        with pytest.raises(modal_app_module.PreHeaderDispatchExpired):
            await modal_app_module._await_modal_call(call, time.time() + 0.01)

    asyncio.run(scenario())
    assert call.cancelled == 1


def test_modal_cancel_cleanup_failure_does_not_mask_primary_failures(modal_app_module) -> None:
    class _ModalFailure(RuntimeError):
        pass

    class _Call:
        def __init__(self, get: Any) -> None:
            self.cancelled = 0
            self.get = types.SimpleNamespace(aio=get)
            self.cancel = types.SimpleNamespace(aio=self._cancel)

        async def _cancel(self) -> None:
            self.cancelled += 1
            raise RuntimeError("cleanup failed")

    async def blocked() -> dict[str, bool]:
        await asyncio.Event().wait()
        return {"ok": True}

    async def modal_failure() -> dict[str, bool]:
        raise _ModalFailure("modal failed")

    async def scenario() -> tuple[int, int, int]:
        timed_out = _Call(blocked)
        with pytest.raises(modal_app_module.PreHeaderDispatchExpired):
            await modal_app_module._await_modal_call(timed_out, time.time() + 0.01)

        failed = _Call(modal_failure)
        with pytest.raises(_ModalFailure, match="modal failed"):
            await modal_app_module._await_modal_call(failed, time.time() + 60)

        cancelled = _Call(blocked)
        task = asyncio.create_task(modal_app_module._await_modal_call(cancelled, time.time() + 60))
        await asyncio.sleep(0)
        task.cancel()
        result = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        return timed_out.cancelled, failed.cancelled, cancelled.cancelled

    assert asyncio.run(scenario()) == (1, 1, 1)


def test_completed_modal_call_is_not_cancelled(modal_app_module) -> None:
    class _Call:
        def __init__(self) -> None:
            self.cancelled = 0
            self.get = types.SimpleNamespace(aio=lambda: asyncio.sleep(0, result={"ok": True}))
            self.cancel = types.SimpleNamespace(aio=self._cancel)

        async def _cancel(self) -> None:
            self.cancelled += 1

    call = _Call()
    result = asyncio.run(modal_app_module._await_modal_call(call, time.time() + 60))

    assert result == {"ok": True}
    assert call.cancelled == 0


# ---- Functional: actually run _load() and capture the AsyncEngineArgs (vLLM/tokenizer stubbed) ----
# Stronger than the AST checks in test_fp8_config: this exercises the real per-model override
# resolution (_ov.get) + the _arg_supported guard, proving the FP8 + moe_backend wiring end-to-end.


def _load_engine_and_args(
    modal_app_module,
    monkeypatch,
    tmp_path,
    base_model: str,
    *,
    engine_type: type | None = None,
) -> tuple[Any, Any]:
    """Run _LoraEngineImpl._load() for ``base_model`` with the tokenizer + vLLM engine stubbed, and
    return the engine instance plus the AsyncEngineArgs it was constructed with."""
    import transformers

    fake_tok = types.SimpleNamespace(pad_token="<pad>", eos_token="<eos>", eos_token_id=0)
    fake_processor = types.SimpleNamespace(tokenizer=fake_tok)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: fake_tok)
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: fake_processor
    )
    monkeypatch.setattr("flash.serving.src.store.settings.ADAPTER_CACHE_DIR", tmp_path / "adapters")

    import vllm  # conftest stub when real vLLM is absent

    captured: dict[str, Any] = {}

    def _capture(engine_args: Any) -> Any:
        captured["args"] = engine_args
        return types.SimpleNamespace(engine_args=engine_args)

    monkeypatch.setattr(vllm.AsyncLLMEngine, "from_engine_args", staticmethod(_capture))

    engine = object.__new__(engine_type or modal_app_module._LoraEngineImpl)
    engine.base_model = base_model
    asyncio.run(engine._load())
    return engine, captured["args"]


def _capture_engine_args(modal_app_module, monkeypatch, tmp_path, base_model: str) -> Any:
    return _load_engine_and_args(modal_app_module, monkeypatch, tmp_path, base_model)[1]


@pytest.mark.parametrize("base_model", base_models())
def test_every_catalog_model_forwards_its_reasoning_parser(
    modal_app_module, monkeypatch, tmp_path, base_model: str
):
    args = _capture_engine_args(modal_app_module, monkeypatch, tmp_path, base_model)
    assert args.reasoning_parser == "qwen3"


def test_lora_pinning_only_when_hot_pool_covers_cpu_pool(modal_app_module, monkeypatch, tmp_path):
    """Pinned LoRAs cannot be LRU-evicted, so capped hot pools must stay unpinned if 256 adapters
    are deployable through max_cpu_loras."""
    engine, args = _load_engine_and_args(modal_app_module, monkeypatch, tmp_path, "Qwen/Qwen3.5-9B")
    assert args.max_loras == 16
    assert args.max_cpu_loras == 256
    assert engine._pin_loras is False

    for model, max_loras in (
        ("Qwen/Qwen3.5-9B", 16),
        ("Qwen/Qwen3.6-35B-A3B", 6),
    ):
        engine, args = _load_engine_and_args(modal_app_module, monkeypatch, tmp_path, model)
        assert args.max_loras == max_loras
        assert args.max_cpu_loras == 256
        assert engine._pin_loras is False


def test_load_prequant_checkpoint_for_9b(modal_app_module, monkeypatch, tmp_path):
    """The 9B loads a pre-quantized FP8 checkpoint with FP8 KV and the L40S-validated 32k context."""
    args = _capture_engine_args(modal_app_module, monkeypatch, tmp_path, "Qwen/Qwen3.5-9B")
    assert args.model == "Freesolo-Co/Qwen3.5-9B-FP8"  # owned pre-quant checkpoint
    assert args.quantization is None  # auto-detected from the checkpoint, not online-quantized
    assert args.kv_cache_dtype == "fp8"
    assert args.max_loras == 16
    assert args.max_lora_rank == 128
    assert args.max_model_len == 32768
    assert args.max_num_seqs == 8
    assert args.gpu_memory_utilization == 0.90  # 0.90 leaves CUDA-graph capture headroom (was 0.98)
    assert args.enforce_eager is False  # CUDA graphs ON: ~10x faster decode on the hybrid GDN model
    assert getattr(args, "max_num_batched_tokens", None) is None


def test_qwen38_candidate_immutable_args_fail_closed_when_vllm_drops_revision_support():
    from flash.serving.src.engine import boot
    from flash.serving.src.engine.model_config import _QWEN38_HOSTED_CANDIDATE

    with pytest.raises(RuntimeError, match=r"cannot pin.*missing engine args"):
        boot._required_immutable_args(
            "Qwen/Qwen3.8-27B",
            _QWEN38_HOSTED_CANDIDATE["engine"],
            {"model"},
        )


def test_load_bf16_base_with_full_experts_for_35b(modal_app_module, monkeypatch, tmp_path):
    """The 35B MoE loads the BASE bf16 weights (not the FP8 checkpoint) on the H200 with CUDA graphs
    and full all-expert LoRA — the only config where experts AND graphs coexist. quantization=None
    here means bf16 (explicit), not FP8 auto-detection."""
    args = _capture_engine_args(modal_app_module, monkeypatch, tmp_path, "Qwen/Qwen3.6-35B-A3B")
    assert args.model == "Qwen/Qwen3.6-35B-A3B"  # bf16 base, NOT the -FP8 checkpoint
    assert args.quantization is None  # explicit bf16 (no online quant)
    assert args.kv_cache_dtype == "fp8"
    assert getattr(args, "moe_backend", None) in (None, "auto")
    assert args.max_loras == 6
    assert args.max_lora_rank == 64
    assert args.max_model_len == 32768
    assert args.max_num_batched_tokens == 4096
    assert args.gpu_memory_utilization == 0.90
    assert args.enforce_eager is False  # CUDA graphs ON
    assert args.max_num_seqs == 8
    # language_model_only is NOT enabled: the full VL model (vision encoder included) is loaded so
    # flash adapters' vision-tower LoRA keys have real modules to bind to.
    assert not args.language_model_only
