"""Per-model GPU tier routing: each base model's engine runs on the GPU class from the catalog.

Modal fixes a class's GPU and concurrency at decoration time, so the serving app registers one
``LoraEngine`` ``@app.cls`` per distinct (GPU tier, max_inputs) key and dispatches each base model to
its class (4B -> L4, 9B -> L40S, 27B -> H100, small models -> L4, 35B -> H200). modal_app imports the ``modal`` SDK
at module top, which isn't installed offline, so we stub it just enough to import the module and
reach the built engine classes.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from flash.serving.src.engine.model_config import base_models, gpu_for


def _passthrough_decorator(*_a: Any, **_k: Any):
    def deco(obj: Any) -> Any:
        return obj

    return deco


@pytest.fixture(scope="module")
def modal_app_module():
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
    _MISSING = object()
    prev_modal = sys.modules.get("modal", _MISSING)
    prev_modal_app = sys.modules.get("flash.serving.app.modal_app", _MISSING)
    sys.modules["modal"] = modal_stub
    # Force a fresh import UNDER the stub: if another test imported modal_app earlier (without this
    # stub), Python would reuse the cached module and the stub wouldn't apply, making this fixture
    # order-dependent. Drop the cached module first; the finally block restores the prior entry.
    sys.modules.pop("flash.serving.app.modal_app", None)
    import flash.serving.app.modal_app as modal_app  # imported after the stub is installed

    try:
        yield modal_app
    finally:
        for name, prev in (("modal", prev_modal), ("modal_app", prev_modal_app)):
            if prev is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


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


def test_deployment_mode_unset_defaults_to_production_with_zero_floor() -> None:
    payload = _successful_import_payload(_probe_modal_app_import())
    assert payload == {
        "mode": "production",
        "custom_domain": "",
        "asgi_custom_domains": None,
        "min_containers": 0,
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
        "min_containers": 0,
    }


def test_development_mode_accepts_exact_custom_domain_with_zero_floor() -> None:
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
        "min_containers": 0,
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
        "base_model": "Qwen/Qwen3.5-4B",
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
assert "flash.serving.src.multimodal" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_one_engine_class_per_distinct_engine_key(modal_app_module):
    """A LoraEngine @app.cls is built for each distinct (GPU tier, max_inputs) key. The small models
    share (L4, 64); 4B uses (L4, 16), 9B uses (L40S, 16), 27B uses (H100, 16), and 35B uses (H200, 16)."""
    assert set(modal_app_module.ENGINE_BY_KEY) == {
        ("L4", 64),
        ("L4", 16),
        ("L40S", 16),
        ("H100", 16),
        ("H200", 16),
    }


def test_engine_concurrency_rejects_malformed_catalog_values(modal_app_module, monkeypatch):
    mod = modal_app_module

    monkeypatch.setattr(mod, "engine_overrides_for", lambda _bm: {"max_num_seqs": 8})
    assert mod._engine_concurrency("valid") == (16, 12)

    monkeypatch.setattr(mod, "engine_overrides_for", lambda _bm: {})
    assert mod._engine_concurrency("defaulted") == (64, 48)

    monkeypatch.setattr(mod, "engine_overrides_for", lambda _bm: {"max_num_seqs": "invalid"})
    monkeypatch.setattr(mod, "base_models", lambda: ["malformed"])
    monkeypatch.setattr(mod, "gpu_for", lambda _bm: "L4")
    with pytest.raises(ValueError, match="invalid literal"):
        mod._distinct_engine_keys()


def test_class_names_are_distinct_and_modal_safe(modal_app_module):
    """Each tier registers under an app-unique, alnum/underscore-only class name that encodes both
    the GPU and its concurrency (max_inputs is part of the class identity)."""
    assert modal_app_module._engine_class_name("L4", 64) == "LoraEngine_L4_c64"
    assert modal_app_module._engine_class_name("L4", 16) == "LoraEngine_L4_c16"
    assert modal_app_module._engine_class_name("A100-80GB", 16) == "LoraEngine_A100_80GB_c16"


def test_small_models_route_to_l4(modal_app_module):
    """Small dense models (owned pre-quant checkpoints) keep the cheap FP8-capable L4 tier."""
    by_key = modal_app_module.ENGINE_BY_KEY
    for bm in (
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
    ):
        assert modal_app_module._engine_cls_for(bm) is by_key[("L4", 64)]
        assert gpu_for(bm) == "L4"


def test_4b_routes_to_l4(modal_app_module):
    """Rank-128 LoRA serving for 4B uses the L4 tier (8-seq -> (L4, 16))."""
    by_key = modal_app_module.ENGINE_BY_KEY
    assert gpu_for("Qwen/Qwen3.5-4B") == "L4"
    assert modal_app_module._engine_cls_for("Qwen/Qwen3.5-4B") is by_key[("L4", 16)]


def test_9b_routes_to_l40s(modal_app_module):
    """Rank-128 LoRA serving for 9B uses the L40S tier (8-seq -> (L40S, 16))."""
    by_key = modal_app_module.ENGINE_BY_KEY
    assert gpu_for("Qwen/Qwen3.5-9B") == "L40S"
    assert modal_app_module._engine_cls_for("Qwen/Qwen3.5-9B") is by_key[("L40S", 16)]
    assert by_key[("L40S", 16)].__name__ == "LoraEngine_L40S_c16"


def test_27b_routes_to_h100(modal_app_module):
    """rank-64 LoRA serving for 27B uses the H100 tier (8-seq -> (H100, 16))."""
    by_key = modal_app_module.ENGINE_BY_KEY
    assert gpu_for("Qwen/Qwen3.6-27B") == "H100"
    assert modal_app_module._engine_cls_for("Qwen/Qwen3.6-27B") is by_key[("H100", 16)]


def test_35b_moe_routes_to_h200(modal_app_module):
    """The 35B-A3B MoE runs bf16 on the H200 tier ((H200, 16))."""
    by_key = modal_app_module.ENGINE_BY_KEY
    assert gpu_for("Qwen/Qwen3.6-35B-A3B") == "H200"
    assert modal_app_module._engine_cls_for("Qwen/Qwen3.6-35B-A3B") is by_key[("H200", 16)]
    assert by_key[("H200", 16)].pinned_gpu == "H200"


def test_unknown_base_model_is_rejected_before_engine_dispatch(modal_app_module):
    """An unseen base model must not silently dispatch to the L4 default tier."""
    with pytest.raises(ValueError, match="Unsupported base model"):
        modal_app_module._engine_cls_for("Qwen/Qwen3.5-99B")


def test_tier_classes_inherit_the_shared_impl(modal_app_module):
    """Each tier class subclasses _LoraEngineImpl (so _load/_generate/etc resolve) and defines the
    public Modal entrypoints itself (so Modal collects them per class)."""
    l4 = modal_app_module.ENGINE_BY_KEY[("L4", 64)]
    assert issubclass(l4, modal_app_module._LoraEngineImpl)
    for impl in ("_load", "_register", "_generate", "_stream_generate", "_unregister", "_health"):
        assert hasattr(l4, impl)
    for entry in ("load", "register", "generate", "stream_generate", "unregister", "health"):
        assert entry in l4.__dict__


def test_each_tier_class_records_its_pinned_gpu(modal_app_module):
    """Each per-tier class records the GPU it was actually pinned to in _build_engine, so health/ops
    can detect a base model misrouted onto the wrong tier instead of trusting the derived lookup."""
    by_key = modal_app_module.ENGINE_BY_KEY
    # Every class records the GPU half of its (gpu, max_inputs) key.
    for (gpu, _max_inputs), cls in by_key.items():
        assert cls.pinned_gpu == gpu
    assert by_key[("L4", 64)].pinned_gpu == "L4"
    assert by_key[("L4", 16)].pinned_gpu == "L4"
    assert by_key[("L40S", 16)].pinned_gpu == "L40S"
    assert by_key[("H100", 16)].pinned_gpu == "H100"
    assert by_key[("H200", 16)].pinned_gpu == "H200"


def test_tier_class_identity_is_fixed_before_decoration(modal_app_module):
    """Each tier class carries its distinct, module-level identity: a clean ``__name__``/``__qualname__``
    (no ``<locals>``) plus a module attribute under that name. The rename + global binding happen on the
    REAL class BEFORE ``@modal.concurrent`` wraps it — otherwise (renaming after the decorator) every tier
    would register under ``_Engine`` and the ``<locals>`` qualname would fail Modal's global-scope check."""
    for (gpu, max_inputs), cls in modal_app_module.ENGINE_BY_KEY.items():
        class_name = modal_app_module._engine_class_name(gpu, max_inputs)
        assert cls.__name__ == class_name
        assert cls.__qualname__ == class_name  # clean — no `_build_engine.<locals>.` prefix
        assert getattr(modal_app_module, class_name) is cls  # reachable as a module global by name


def test_health_reports_pinned_gpu_over_derived_tier(modal_app_module):
    """_health() reports the class's ACTUAL pinned tier, not gpu_for(base_model): a base model
    instantiated on the wrong tier's class (here an A100-pinned class holding a 4B model whose
    expected tier is L4) must surface as the real A100 tier, otherwise misrouting is masked."""
    impl = modal_app_module._LoraEngineImpl

    class _Fake:
        pinned_gpu = "A100-80GB"
        base_model = "Qwen/Qwen3.5-4B"  # gpu_for -> L4 (the EXPECTED tier)
        registry = type("R", (), {"list_ready": lambda self: []})()

    health = impl._health(_Fake())
    assert (
        health["configured_gpu"] == "A100-80GB"
    )  # actual pinned tier, not the derived catalog tier

    class _Bare:
        # No pinned_gpu (the shared impl used directly) -> fall back to the expected tier.
        base_model = "Qwen/Qwen3.5-4B"
        registry = type("R", (), {"list_ready": lambda self: []})()

    assert impl._health(_Bare())["configured_gpu"] == "L4"  # fallback to gpu_for(base_model)


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
        base_model = "Qwen/Qwen3.5-2B"  # per-model 32k override
        registry = type("R", (), {"list_ready": lambda self: []})()

    assert impl._health(_Dense())["max_model_len"] == 32768
    assert impl._health(_Dense())["max_model_len"] != cfg.MAX_MODEL_LEN


def test_start_all_raises_after_any_engine_fails(modal_app_module, monkeypatch):
    from flash.serving.src.engine import model_config

    mod = modal_app_module
    monkeypatch.setattr(model_config, "base_models", lambda: ["ok", "boom"])
    monkeypatch.setattr(model_config, "gpu_for", lambda _bm: "L4")
    monkeypatch.setattr(mod, "engine_overrides_for", lambda _bm: {})
    spawned: list[str] = []

    class _Handle:
        def __init__(self, base_model: str) -> None:
            self.base_model = base_model

        def get(self, timeout: int = 0) -> str:
            if self.base_model == "boom":
                raise RuntimeError("cold start failed")
            return "ok"

    def _from_name(_app_name: str, _cls_name: str):
        def _factory(base_model: str):
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


def test_scale_to_zero_pool_dispatches_inference_and_registration(modal_app_module, monkeypatch):
    """The pool never updates autoscaling and still dispatches demand-driven remote calls."""
    mod = modal_app_module
    bound_models: list[str] = []
    generate_calls: list[tuple[dict, dict, str | None]] = []
    stream_calls: list[tuple[dict, dict, str | None]] = []
    register_calls: list[tuple[dict, str | None]] = []

    class _Dump:
        def __init__(
            self,
            value: dict,
            *,
            deployment_generation: str | None = None,
        ) -> None:
            self.value = value
            self.deployment_generation = deployment_generation

        def model_dump(self, *, by_alias: bool) -> dict:
            assert by_alias is True
            return self.value

    async def _generate(payload: dict, record: dict, checkpoint: str | None) -> dict:
        generate_calls.append((payload, record, checkpoint))
        return {"ok": True}

    async def _stream_generate(payload: dict, record: dict, checkpoint: str | None):
        stream_calls.append((payload, record, checkpoint))
        yield {"delta": "hello"}
        yield {"delta": " world"}

    async def _register(record: dict, deployment_generation: str | None) -> None:
        register_calls.append((record, deployment_generation))

    class _FakeEngine:
        generate = types.SimpleNamespace(remote=types.SimpleNamespace(aio=_generate))
        stream_generate = types.SimpleNamespace(
            remote_gen=types.SimpleNamespace(aio=_stream_generate)
        )
        register = types.SimpleNamespace(remote=types.SimpleNamespace(aio=_register))

        @property
        def update_autoscaler(self):
            raise AssertionError("zero-floor engines must not update the autoscaler")

    engine = _FakeEngine()

    def _bind(*, base_model: str):
        bound_models.append(base_model)
        return engine

    monkeypatch.setattr(mod, "_engine_cls_for", lambda _base_model: _bind)
    pool = mod._ModalEnginePool()
    payload = _Dump({"messages": [{"role": "user", "content": "hello"}]})
    record = _Dump(
        {"adapter_id": "run@step-1.sha"},
        deployment_generation="generation-1",
    )

    result = asyncio.run(
        pool.generate(
            "Qwen/Qwen3.5-4B",
            payload,
            record,
            expected_checkpoint="step-1",
        )
    )

    async def _collect_stream() -> list[dict]:
        return [
            event
            async for event in pool.stream_generate(
                "Qwen/Qwen3.5-4B",
                payload,
                record,
                expected_checkpoint="step-1",
            )
        ]

    stream_events = asyncio.run(_collect_stream())
    asyncio.run(pool.register("Qwen/Qwen3.5-4B", record))

    assert result == {"ok": True}
    assert stream_events == [{"delta": "hello"}, {"delta": " world"}]
    assert bound_models == [
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-4B",
    ]
    expected_inference_call = (
        {"messages": [{"role": "user", "content": "hello"}]},
        {
            "adapter_id": "run@step-1.sha",
            "deployment_generation": "generation-1",
        },
        "step-1",
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


# ---- Functional: actually run _load() and capture the AsyncEngineArgs (vLLM/tokenizer stubbed) ----
# Stronger than the AST checks in test_fp8_config: this exercises the real per-model override
# resolution (_ov.get) + the _arg_supported guard, proving the FP8 + moe_backend wiring end-to-end.


def _load_engine_and_args(
    modal_app_module, monkeypatch, tmp_path, base_model: str
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

    engine = object.__new__(modal_app_module._LoraEngineImpl)
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


def test_load_prequant_fp8_checkpoint_for_dense_small(modal_app_module, monkeypatch, tmp_path):
    """A small dense base (2B) now loads its owned PRE-QUANTIZED FP8 checkpoint: model =
    serve_model_id, quantization = None (vLLM auto-detects the checkpoint's FP8 — no online quant),
    FP8 KV, rank-128 / max_loras-16 buffers, NO moe_backend (that's MoE-only)."""
    args = _capture_engine_args(modal_app_module, monkeypatch, tmp_path, "Qwen/Qwen3.5-2B")
    assert args.model == "Freesolo-Co/Qwen3.5-2B-FP8"  # owned pre-quant checkpoint
    assert args.quantization is None  # auto-detected from the checkpoint, not online-quantized
    assert args.kv_cache_dtype == "fp8"
    assert args.enable_lora is True
    assert args.max_lora_rank == 128
    assert args.max_loras == 16
    assert args.max_model_len == 32768
    assert args.gpu_memory_utilization == 0.98  # small L4 tiers pin 0.98, not the global 0.90
    assert (
        args.max_num_seqs == 64
    )  # capped to the MAX_INPUTS concurrency ceiling, not vLLM's default
    assert getattr(args, "moe_backend", None) in (None, "auto")
    assert args.limit_mm_per_prompt == {"image": 4}
    assert args.mm_processor_cache_gb == 0
    assert args.enable_tower_connector_lora is True
    assert getattr(args, "calculate_kv_scales", None) in (None, False)  # never enabled (GDN-safe)


def test_lora_pinning_only_when_hot_pool_covers_cpu_pool(modal_app_module, monkeypatch, tmp_path):
    """Pinned LoRAs cannot be LRU-evicted, so capped hot pools must stay unpinned if 256 adapters
    are deployable through max_cpu_loras."""
    engine, args = _load_engine_and_args(modal_app_module, monkeypatch, tmp_path, "Qwen/Qwen3.5-2B")
    assert args.max_loras == 16
    assert args.max_cpu_loras == 256
    assert engine._pin_loras is False

    for model, max_loras in (
        ("Qwen/Qwen3.5-4B", 16),
        ("Qwen/Qwen3.5-9B", 16),
        ("Qwen/Qwen3.6-27B", 16),
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


def test_load_owned_fp8_checkpoint_for_27b(modal_app_module, monkeypatch, tmp_path):
    args = _capture_engine_args(modal_app_module, monkeypatch, tmp_path, "Qwen/Qwen3.6-27B")
    assert args.model == "Freesolo-Co/Qwen3.6-27B-FP8"  # owned pre-quant checkpoint
    assert args.tensor_parallel_size == 1
    assert args.quantization is None
    assert args.kv_cache_dtype == "fp8"
    assert args.max_loras == 16
    assert args.max_lora_rank == 64
    assert args.max_model_len == 32768
    assert args.max_num_seqs == 8
    assert args.gpu_memory_utilization == 0.90  # 0.90 leaves CUDA-graph capture headroom (was 0.98)
    assert args.enforce_eager is False  # CUDA graphs ON: ~7x faster decode on the hybrid GDN model
    assert args.reasoning_parser == "qwen3"
    assert args.limit_mm_per_prompt == {"image": 4}
    assert args.enable_tower_connector_lora is True
    assert not args.language_model_only
    assert getattr(args, "max_num_batched_tokens", None) is None


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
