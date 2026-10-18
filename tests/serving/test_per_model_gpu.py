"""Per-model GPU tier routing: each base model's engine runs on the GPU class from the catalog.

Modal fixes a class's GPU and concurrency at decoration time, so the serving app registers one
``LoraEngine`` ``@app.cls`` per distinct (GPU tier, max_inputs) key and dispatches each base model to
its class (all active tiers -> b200). modal_app imports the ``modal`` sdk
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

from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serving.src.engine.model_config import base_models, gpu_for
from flash.serving.src.io.schemas import AdapterRecord, internal_adapter_payload


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

    run_id = "flash-1234567890-abcdef12"
    adapter_id = f"{run_id}/step-1"
    record_values = {
        "adapter_id": adapter_id,
        "repo_id": "org/private-adapter",
        "base_model": "Qwen/Qwen3.5-9B",
        "subfolder": "checkpoints/step-1/adapter",
        "repo_type": "dataset",
        "org_id": "org-1",
        "checkpoint": adapter_id,
        "private": True,
        "thinking": False,
        "status": "ready",
        "run_id": run_id,
        "checkpoint_step": 1,
        "artifact_revision": "a" * 40,
        "artifact_digest": "b" * 64,
        "lora_rank": 32,
    }
    record_values["artifact_fingerprint"] = immutable_binding_fingerprint(record_values)
    record_dict = internal_adapter_payload(AdapterRecord.model_validate(record_values))
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
    assert engine.registry.get("org-1", adapter_id) == record


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


def test_one_engine_class_per_distinct_engine_key(modal_app_module):
    """a loraengine class is built for each active gpu tier and concurrency key."""
    # all three active tiers serve on B200 with the same concurrency, so they legitimately share
    # ONE engine class: the key is (gpu, max_inputs) and each model still gets its own container
    # and its own engine args.
    assert set(modal_app_module.ENGINE_BY_KEY) == {("B200", 8)}


def test_engine_concurrency_rejects_malformed_catalog_values(modal_app_module, monkeypatch):
    mod = modal_app_module

    monkeypatch.setattr(mod, "engine_overrides_for", lambda _bm: {"max_num_seqs": 8})
    assert mod._engine_concurrency("valid") == (8, 6)

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


def test_9b_routes_to_b200(modal_app_module):
    """Rank-128 LoRA serving for 9B uses the B200 tier (8-seq -> (B200, 8))."""
    by_key = modal_app_module.ENGINE_BY_KEY
    assert gpu_for("Qwen/Qwen3.5-9B") == "B200"
    assert modal_app_module._engine_cls_for("Qwen/Qwen3.5-9B") is by_key[("B200", 8)]
    assert by_key[("B200", 8)].__name__ == "LoraEngine_B200_c8"


def test_27b_routes_to_b200(modal_app_module):
    """The dense 27B runs its FP8 checkpoint on the B200 tier (8-seq -> (B200, 8))."""
    by_key = modal_app_module.ENGINE_BY_KEY
    assert gpu_for("Qwen/Qwen3.8-27B") == "B200"
    assert modal_app_module._engine_cls_for("Qwen/Qwen3.8-27B") is by_key[("B200", 8)]
    assert by_key[("B200", 8)].pinned_gpu == "B200"


def test_35b_moe_routes_to_b200(modal_app_module):
    """The 35B-A3B MoE runs bf16 on the B200 tier ((B200, 8))."""
    by_key = modal_app_module.ENGINE_BY_KEY
    assert gpu_for("Qwen/Qwen3.6-35B-A3B") == "B200"
    assert modal_app_module._engine_cls_for("Qwen/Qwen3.6-35B-A3B") is by_key[("B200", 8)]
    assert by_key[("B200", 8)].pinned_gpu == "B200"


def test_unknown_base_model_is_rejected_before_engine_dispatch(modal_app_module):
    """An unseen base model must not silently dispatch to the L4 default tier."""
    with pytest.raises(ValueError, match="Unsupported base model"):
        modal_app_module._engine_cls_for("Qwen/Qwen3.5-99B")


def test_tier_classes_inherit_the_shared_impl(modal_app_module):
    """Each tier class subclasses _LoraEngineImpl (so _load/_generate/etc resolve) and defines the
    public Modal entrypoints itself (so Modal collects them per class)."""
    b200 = modal_app_module.ENGINE_BY_KEY[("B200", 8)]
    assert issubclass(b200, modal_app_module._LoraEngineImpl)
    for impl in ("_load", "_register", "_generate", "_stream_generate", "_unregister", "_health"):
        assert hasattr(b200, impl)
    for entry in ("load", "register", "generate", "stream_generate", "unregister", "health"):
        assert entry in b200.__dict__


def test_each_tier_class_records_its_pinned_gpu(modal_app_module):
    """Each per-tier class records the GPU it was actually pinned to in _build_engine, so health/ops
    can detect a base model misrouted onto the wrong tier instead of trusting the derived lookup."""
    by_key = modal_app_module.ENGINE_BY_KEY
    # Every class records the GPU half of its (gpu, max_inputs) key.
    for (gpu, _max_inputs), cls in by_key.items():
        assert cls.pinned_gpu == gpu
    assert by_key[("B200", 8)].pinned_gpu == "B200"


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

    assert impl._health(_Bare())["configured_gpu"] == "B200"


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
    generate_calls: list[tuple[dict, dict, str | None, str]] = []
    stream_calls: list[tuple[dict, dict, str | None, str]] = []
    register_calls: list[tuple[dict, str | None]] = []

    class _Dump:
        def __init__(
            self,
            value: dict,
            *,
            deployment_generation: str | None = None,
            generation_id: str | None = None,
        ) -> None:
            self.value = value
            self.deployment_generation = deployment_generation
            self.generation_id = generation_id

        def model_dump(self, *, by_alias: bool) -> dict:
            assert by_alias is True
            return self.value

    async def _generate(
        payload: dict, record: dict, checkpoint: str | None, generation_id: str
    ) -> dict:
        generate_calls.append((payload, record, checkpoint, generation_id))
        return {"ok": True}

    async def _stream_generate(
        payload: dict, record: dict, checkpoint: str | None, generation_id: str
    ):
        stream_calls.append((payload, record, checkpoint, generation_id))
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
    generation_id = "fsgen-00000000000000000000000000000001"
    payload = _Dump(
        {"messages": [{"role": "user", "content": "hello"}]},
        generation_id=generation_id,
    )
    checkpoint_id = "flash-1234567890-abcdef12/step-1"
    record_values = {
        "adapter_id": checkpoint_id,
        "repo_id": "org/run",
        "base_model": "Qwen/Qwen3.5-9B",
        "subfolder": "checkpoints/step-1/adapter",
        "repo_type": "dataset",
        "org_id": "org-1",
        "checkpoint": checkpoint_id,
        "thinking": False,
        "deployment_generation": "generation-1",
        "run_id": "flash-1234567890-abcdef12",
        "checkpoint_step": 1,
        "artifact_revision": "a" * 40,
        "artifact_digest": "b" * 64,
        "lora_rank": 32,
    }
    record_values["artifact_fingerprint"] = immutable_binding_fingerprint(record_values)
    record = AdapterRecord.model_validate(record_values)

    result = asyncio.run(
        pool.generate(
            "Qwen/Qwen3.5-9B",
            payload,
            record,
            expected_checkpoint=checkpoint_id,
        )
    )

    async def _collect_stream() -> list[dict]:
        return [
            event
            async for event in pool.stream_generate(
                "Qwen/Qwen3.5-9B",
                payload,
                record,
                expected_checkpoint=checkpoint_id,
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
    forwarded_record = internal_adapter_payload(record)
    assert AdapterRecord.model_validate(forwarded_record) == record
    expected_inference_call = (
        {"messages": [{"role": "user", "content": "hello"}]},
        forwarded_record,
        checkpoint_id,
        generation_id,
    )
    assert generate_calls == [expected_inference_call]
    assert stream_calls == [expected_inference_call]
    assert register_calls == [(forwarded_record, "generation-1")]


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
    """The 9B loads a pre-quantized FP8 checkpoint with FP8 KV and the validated 32k context."""
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


def test_qwen38_immutable_args_fail_closed_when_vllm_drops_revision_support():
    from flash.serving.src.engine import boot
    from flash.serving.src.engine.model_config import engine_overrides_for

    with pytest.raises(RuntimeError, match=r"cannot pin.*missing engine args"):
        boot._required_immutable_args(
            "Qwen/Qwen3.8-27B",
            engine_overrides_for("Qwen/Qwen3.8-27B"),
            {"model"},
        )


def test_load_bf16_base_with_full_experts_for_35b(modal_app_module, monkeypatch, tmp_path):
    """The 35B MoE loads the BASE bf16 weights (not the FP8 checkpoint) on the B200 with CUDA graphs
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
