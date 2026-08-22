"""Serving->backend usage reporter gating: a serving deployment must only post metered usage to
the billing API it was EXPLICITLY pointed at. Regression guard for the bug where backend_url
defaulted to production and reporting turned on as soon as FREESOLO_INTERNAL_KEY was present, so a
non-prod deployment that merely had that key could silently bill real orgs for dev traffic.

modal_app imports the `modal` SDK at module top (decorators run at import), which isn't installed
in the offline test env, so we stub it just enough to import and reach _build_usage_reporter.
"""

from __future__ import annotations

import asyncio
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from flash.serving.src import engine_support
from flash.serving.src.settings import Settings


def _passthrough_decorator(*_a: Any, **_k: Any):
    # Modal's class/function/method decorators must return the decorated object so the module's
    # class + function bodies survive import.
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
    # Save + restore the prior sys.modules entries so this stub doesn't leak into other tests
    # (which would hide a real `modal`/`modal_app` import error or make the suite order-dependent).
    # _MISSING marks "was not present" so teardown removes our stub instead of reinserting a stale
    # one. We also drop the imported `modal_app` so it is re-imported fresh against the real module.
    _MISSING = object()
    prev_modal = sys.modules.get("modal", _MISSING)
    prev_modal_app = sys.modules.get("flash.serving.modal_app", _MISSING)
    sys.modules["modal"] = modal_stub

    import flash.serving.modal_app as modal_app  # imported after the stub is installed

    try:
        yield modal_app
    finally:
        for name, prev in (("modal", prev_modal), ("modal_app", prev_modal_app)):
            if prev is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _settings(*, backend_url: str | None = None, internal_key: str | None = None) -> Settings:
    # Build Settings without reading a stray on-disk .env. The fields use validation_alias, so we
    # set them via their env-var aliases (which is also the real configuration path). Omit an alias
    # entirely to exercise its field default.
    kwargs: dict[str, Any] = {"_env_file": None}
    if backend_url is not None:
        kwargs["PLATFORM_BACKEND_URL"] = backend_url
    if internal_key is not None:
        kwargs["FREESOLO_INTERNAL_KEY"] = internal_key
    return Settings(**kwargs)


def test_stream_text_delta_keeps_native_delta_chunks(modal_app_module):
    previous = ""
    deltas = []

    for text in ("Hel", "lo", " hello"):
        delta, previous = engine_support._stream_text_delta(text, previous, cumulative_output=False)
        deltas.append(delta)

    assert deltas == ["Hel", "lo", " hello"]
    assert previous == "Hello hello"


def test_stream_text_delta_diffs_cumulative_chunks_without_token_ids(modal_app_module):
    previous = ""
    deltas = []

    for text in ("Hel", "Hello", "Hello", "Hello!"):
        delta, previous = engine_support._stream_text_delta(text, previous, cumulative_output=None)
        deltas.append(delta)

    assert deltas == ["Hel", "lo", "", "!"]
    assert previous == "Hello!"


def test_stream_text_delta_keeps_text_when_cumulative_hint_is_not_a_prefix(modal_app_module):
    delta, previous = engine_support._stream_text_delta("world", "hello ", cumulative_output=True)

    assert delta == "world"
    assert previous == "hello world"


def test_backend_url_has_no_production_default():
    # The crux of the fix: backend_url must NOT default to a real (prod) URL. An unset
    # PLATFORM_BACKEND_URL must leave it empty so reporting stays off until wired deliberately.
    s = Settings(_env_file=None)
    assert s.backend_url == ""


def test_lora_engine_scales_to_zero_by_default(modal_app_module):
    # every gpu engine class scales to zero, while the cpu router stays available to trigger them.
    assert modal_app_module.MIN_CONTAINERS == 0

    cls_calls = [call.kwargs for call in modal_app_module.app.cls.call_args_list]
    assert len(cls_calls) == len(modal_app_module.ENGINE_BY_KEY)
    assert all("min_containers" not in kwargs for kwargs in cls_calls)
    # Each engine class is pinned to ITS gpu tier's window, not one flat value.
    assert {kwargs["gpu"]: kwargs["scaledown_window"] for kwargs in cls_calls} == {
        gpu: modal_app_module.scaledown_window_for(gpu)
        for gpu, _mi in modal_app_module.ENGINE_BY_KEY
    }

    assert modal_app_module.app.function.call_count == 1
    assert modal_app_module.app.function.call_args.kwargs["min_containers"] == 1


def test_scaledown_window_is_per_tier_and_cheaper_tiers_release_sooner(modal_app_module):
    # The whole point of the table: an idle container bills at the full gpu rate, so a cheap
    # fast-booting tier must not hold a card as long as the 35B's ~1010s-boot H200 does.
    from flash.serving.src.model_config import base_models, gpu_for

    window_for = modal_app_module.scaledown_window_for
    default = modal_app_module.DEFAULT_SCALEDOWN_WINDOW_SECONDS

    assert window_for("L4") < window_for("L40S") < window_for("H100") < window_for("H200")
    # The H200 keeps the full legacy hold (boot is ~17 min; a miss stalls the user that long).
    assert window_for("H200") == default == 1800
    # Every catalog tier resolves to a positive window, so no model falls back by accident.
    assert all(window_for(gpu_for(bm)) > 0 for bm in base_models())
    # An unknown tier falls back to the safe (longest) default rather than releasing early.
    assert window_for("B200") == default


def test_modal_concurrency_is_per_engine_key(modal_app_module):
    # Each engine class caps max_inputs near its engine's real max_num_seqs (see _engine_concurrency)
    # so Modal autoscales instead of over-packing one container; target_inputs = 3/4 of that. The
    # router keeps the global 64/48. MAX_INPUTS 64 came from a real-GPU sweep (near-linear to 128).
    assert modal_app_module.MAX_INPUTS == 64
    assert modal_app_module.TARGET_INPUTS == 48
    # One @modal.concurrent per distinct engine key plus one on the router.
    calls = [call.kwargs for call in modal_app_module.modal.concurrent.call_args_list]
    assert len(calls) == len(modal_app_module.ENGINE_BY_KEY) + 1
    seen = {(c["max_inputs"], c["target_inputs"]) for c in calls}
    # Router carries the global sizing; each engine key carries max_inputs with target = 3/4 of it.
    expected = {(mi, max(1, mi * 3 // 4)) for (_gpu, mi) in modal_app_module.ENGINE_BY_KEY}
    expected.add((modal_app_module.MAX_INPUTS, modal_app_module.TARGET_INPUTS))
    assert seen == expected


def test_lora_engine_replica_identifier_is_stable_per_instance(modal_app_module):
    first = object.__new__(modal_app_module._LoraEngineImpl)
    second = object.__new__(modal_app_module._LoraEngineImpl)

    first_id = first._replica_identifier()
    assert first._replica_identifier() == first_id
    assert second._replica_identifier() != first_id
    assert len(first_id) == 32


def test_cached_token_telemetry_distinguishes_zero_from_absent(modal_app_module):
    class _Absent:
        pass

    class _Zero:
        num_cached_tokens = 0

    class _Invalid:
        num_cached_tokens = "invalid"

    class _Negative:
        num_cached_tokens = -1

    assert engine_support._num_cached_tokens(_Absent()) == 0
    assert engine_support._cached_tokens_reported(_Absent()) is False
    assert engine_support._num_cached_tokens(_Zero()) == 0
    assert engine_support._cached_tokens_reported(_Zero()) is True
    assert engine_support._num_cached_tokens(_Invalid()) == 0
    assert engine_support._cached_tokens_reported(_Invalid()) is False
    assert engine_support._num_cached_tokens(_Negative()) == 0
    assert engine_support._cached_tokens_reported(_Negative()) is False


def test_lora_engine_builds_tokenized_chat_prompt(modal_app_module):
    engine = object.__new__(modal_app_module._LoraEngineImpl)

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "hi"}]
            # A caller that sends no chat_template_kwargs still gets an explicit enable_thinking
            # from the resolved adapter's stored `thinking` flag.
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": False,
                "enable_thinking": True,
            }
            return [1, 2, 3]

    class _Payload:
        messages: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "hi"}]
        prompt = None

    engine.tokenizer = _Tokenizer()

    assert asyncio.run(engine._prompt_input(_Payload(), thinking_default=True)) == {
        "prompt_token_ids": [1, 2, 3]
    }


def test_lora_engine_builds_tokenized_raw_prompt(modal_app_module):
    engine = object.__new__(modal_app_module._LoraEngineImpl)

    class _Tokenizer:
        def encode(self, prompt: str, **kwargs):
            assert prompt == "raw"
            assert kwargs == {"add_special_tokens": False}
            return [4, 5]

    class _Payload:
        messages = None
        prompt = "raw"

    engine.tokenizer = _Tokenizer()

    assert asyncio.run(engine._prompt_input(_Payload())) == {"prompt_token_ids": [4, 5]}


def test_lora_engine_caches_exact_prompt_tokens(modal_app_module):
    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine._prompt_cache_size = 2  # prompt-cache size is an instance attr (constant at load)
    engine._prompt_token_cache = OrderedDict()
    calls: list[str] = []

    class _Tokenizer:
        def encode(self, prompt: str, **kwargs):
            calls.append(prompt)
            return [len(prompt)]

    class _Payload:
        messages = None

        def __init__(self, prompt: str) -> None:
            self.prompt = prompt

    engine.tokenizer = _Tokenizer()

    assert asyncio.run(engine._prompt_input(_Payload("first"))) == {"prompt_token_ids": [5]}
    assert asyncio.run(engine._prompt_input(_Payload("first"))) == {"prompt_token_ids": [5]}
    assert calls == ["first"]

    asyncio.run(engine._prompt_input(_Payload("second")))
    asyncio.run(engine._prompt_input(_Payload("third")))
    asyncio.run(engine._prompt_input(_Payload("first")))

    assert calls == ["first", "second", "third", "first"]


def test_lora_engine_filters_reserved_chat_template_kwargs(modal_app_module):
    # A caller's chat_template_kwargs must not re-supply args we already pass explicitly (that
    # raises "got multiple values for keyword argument" -> a 500). Reserved keys are dropped;
    # enable_thinking is forced from the adapter default, not the caller payload.
    engine = object.__new__(modal_app_module._LoraEngineImpl)
    seen: dict = {}

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.update(kwargs)
            return [9]

    class _Payload:
        messages: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "hi"}]
        prompt = None
        chat_template_kwargs: ClassVar[dict[str, object]] = {
            "enable_thinking": False,
            "add_generation_prompt": False,  # reserved -> dropped, not duplicated
            "tokenize": False,
            "return_dict": True,
        }

    engine.tokenizer = _Tokenizer()
    assert asyncio.run(engine._prompt_input(_Payload(), thinking_default=True)) == {
        "prompt_token_ids": [9]
    }
    assert seen == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": False,
        "enable_thinking": True,
    }


def test_lora_engine_cache_key_uses_adapter_thinking_default(modal_app_module):
    # Caller-supplied enable_thinking is ignored: same messages with different caller values collide
    # under the same adapter default. Different adapter defaults still split the cache because they
    # render different prompts.
    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine._prompt_cache_size = 4
    engine._prompt_token_cache = OrderedDict()
    calls: list = []

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs.get("enable_thinking"))
            return [1] if kwargs.get("enable_thinking") else [2]

    class _Payload:
        messages: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "hi"}]
        prompt = None

        def __init__(self, thinking: bool) -> None:
            self.chat_template_kwargs = {"enable_thinking": thinking}

    engine.tokenizer = _Tokenizer()
    assert asyncio.run(engine._prompt_input(_Payload(True), thinking_default=True)) == {
        "prompt_token_ids": [1]
    }
    assert asyncio.run(engine._prompt_input(_Payload(False), thinking_default=True)) == {
        "prompt_token_ids": [1]
    }
    assert asyncio.run(engine._prompt_input(_Payload(True), thinking_default=False)) == {
        "prompt_token_ids": [2]
    }
    assert calls == [True, False]


def test_lora_engine_requires_trained_thinking_default(modal_app_module):
    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine._prompt_cache_size = 4
    engine._prompt_token_cache = OrderedDict()

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1]

    class _Payload:
        messages: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "hi"}]
        prompt = None

        def __init__(self, thinking: bool) -> None:
            self.chat_template_kwargs = {"enable_thinking": thinking}

    engine.tokenizer = _Tokenizer()
    with pytest.raises(ValueError, match="adapter thinking default is required"):
        asyncio.run(engine._prompt_input(_Payload(False)))


def test_lora_engine_drops_return_shape_chat_template_kwargs(modal_app_module):
    # Return-shape-altering keys (e.g. return_tensors) would make apply_chat_template return a
    # tensor/mapping and break list(prompt_token_ids) -> a 500. They must be dropped, and a
    # non-dict chat_template_kwargs must be ignored (not .items()-ed) rather than crash.
    engine = object.__new__(modal_app_module._LoraEngineImpl)
    seen: dict = {}

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.clear()
            seen.update(kwargs)
            return [7]

    class _Payload:
        messages: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "hi"}]
        prompt = None

        def __init__(self, ctk) -> None:
            self.chat_template_kwargs = ctk

    engine.tokenizer = _Tokenizer()

    # return_tensors / padding / truncation / max_length are reserved -> dropped; enable_thinking is
    # forced from the adapter default, not the caller.
    assert asyncio.run(
        engine._prompt_input(
            _Payload(
                {
                    "enable_thinking": False,
                    "return_tensors": "pt",
                    "padding": True,
                    "truncation": True,
                    "max_length": 8,
                }
            ),
            thinking_default=True,
        )
    ) == {"prompt_token_ids": [7]}
    assert seen == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": False,
        "enable_thinking": True,
    }

    # A non-dict chat_template_kwargs is ignored entirely (no crash). enable_thinking still comes
    # from the adapter default.
    assert asyncio.run(
        engine._prompt_input(_Payload(["not", "a", "dict"]), thinking_default=True)
    ) == {"prompt_token_ids": [7]}
    assert seen == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": False,
        "enable_thinking": True,
    }


def test_lora_engine_cache_key_ignores_reserved_chat_template_kwargs(modal_app_module):
    # Comment #4: the cache key uses the SAME sanitized kwargs as tokenization, so a reserved /
    # ignored key (which doesn't change the rendered prompt) must NOT cause a cache miss.
    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine._prompt_cache_size = 4
    engine._prompt_token_cache = OrderedDict()
    calls: list = []

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs.get("enable_thinking"))
            return [3]

    class _Payload:
        messages: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "hi"}]
        prompt = None

        def __init__(self, ctk) -> None:
            self.chat_template_kwargs = ctk

    engine.tokenizer = _Tokenizer()
    # First render with a genuine control; second adds only a reserved key (same rendered prompt).
    assert asyncio.run(
        engine._prompt_input(_Payload({"enable_thinking": True}), thinking_default=True)
    ) == {"prompt_token_ids": [3]}
    assert asyncio.run(
        engine._prompt_input(
            _Payload({"enable_thinking": True, "return_tensors": "pt"}), thinking_default=True
        )
    ) == {"prompt_token_ids": [3]}
    # Only one tokenize call: the reserved key did not split the cache entry.
    assert calls == [True]


def test_lora_engine_health_reports_served_model_and_baked_config(modal_app_module, monkeypatch):
    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine.base_model = "Qwen/Qwen3.5-2B"
    engine._prompt_cache_size = 2048

    class _Registry:
        def list_ready(self):
            return [object(), object()]

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA H100"

    class _Torch:
        cuda = _Cuda()

    engine.registry = _Registry()
    engine._prompt_token_cache = OrderedDict([(("prompt", "cached"), (1, 2, 3))])
    monkeypatch.setitem(sys.modules, "torch", _Torch())

    # Health reports the served PRE-QUANTIZED checkpoint (owned FP8 for 2B) + baked-in config.
    assert engine._health() == {
        "ok": True,
        # No engine built on this bare instance -> not dead -> ok stays True.
        "engine_dead": False,
        "base_model": "Qwen/Qwen3.5-2B",
        "served_model": "Freesolo-Co/Qwen3.5-2B-FP8",
        # A pre-quant FP8 checkpoint carries fp8 weights (auto-detected) + the global fp8 KV cache.
        "quantization": "fp8",
        "kv_cache_dtype": "fp8",
        "adapters": 2,
        # configured_gpu now reflects the per-model GPU tier (2B -> L4), not a single global.
        "configured_gpu": modal_app_module.gpu_for("Qwen/Qwen3.5-2B"),
        "cuda_available": True,
        "device_name": "NVIDIA H100",
        "enable_prefix_caching": True,
        "prompt_token_cache_size": 2048,
        "prompt_token_cache_entries": 1,
        "max_model_len": 32768,
    }


def test_engine_pool_applies_warm_floor_to_parameterized_instance(modal_app_module, monkeypatch):
    # modal rejects min_containers on parameterized class decorators. a positive floor is applied to
    # each loraengine(base_model=...) instance instead.
    import asyncio

    monkeypatch.setattr(modal_app_module, "MIN_CONTAINERS", 1)
    calls: list[tuple[str, dict[str, int]]] = []

    class _FakeEngine:
        def __init__(self, base_model: str) -> None:
            self.base_model = base_model

        async def update_autoscaler(self, **kwargs: int) -> None:
            calls.append((self.base_model, kwargs))

    def _fake_lora_engine(*, base_model: str) -> _FakeEngine:
        return _FakeEngine(base_model)

    monkeypatch.setattr(modal_app_module, "_engine_cls_for", lambda _bm: _fake_lora_engine)
    pool = modal_app_module._ModalEnginePool()

    first = asyncio.run(pool._engine("Qwen/Qwen3.5-4B"))
    second = asyncio.run(pool._engine("Qwen/Qwen3.5-4B"))

    assert first.base_model == "Qwen/Qwen3.5-4B"
    assert second.base_model == "Qwen/Qwen3.5-4B"
    assert calls == [
        (
            "Qwen/Qwen3.5-4B",
            {
                "min_containers": 1,
                "scaledown_window": modal_app_module.scaledown_window_for("L4"),
            },
        )
    ]


def test_engine_pool_applies_max_container_cap_when_configured(modal_app_module, monkeypatch):
    import asyncio

    monkeypatch.setattr(modal_app_module, "MIN_CONTAINERS", 1)
    calls: list[dict[str, int]] = []

    class _FakeEngine:
        async def update_autoscaler(self, **kwargs: int) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(modal_app_module, "MAX_CONTAINERS", 1)
    monkeypatch.setattr(
        modal_app_module, "_engine_cls_for", lambda _bm: lambda *, base_model: _FakeEngine()
    )

    asyncio.run(modal_app_module._ModalEnginePool()._engine("Qwen/Qwen3.5-4B"))

    assert calls == [
        {
            "min_containers": 1,
            "scaledown_window": modal_app_module.scaledown_window_for("L4"),
            "max_containers": 1,
        }
    ]


def test_engine_pool_applies_autoscaler_buffer_when_configured(modal_app_module, monkeypatch):
    import asyncio

    monkeypatch.setattr(modal_app_module, "MIN_CONTAINERS", 1)
    calls: list[dict[str, int]] = []

    class _FakeEngine:
        async def update_autoscaler(self, **kwargs: int) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(modal_app_module, "BUFFER_CONTAINERS", 2)
    monkeypatch.setattr(
        modal_app_module, "_engine_cls_for", lambda _bm: lambda *, base_model: _FakeEngine()
    )

    asyncio.run(modal_app_module._ModalEnginePool()._engine("Qwen/Qwen3.5-4B"))

    assert calls == [
        {
            "min_containers": 1,
            "scaledown_window": modal_app_module.scaledown_window_for("L4"),
            "buffer_containers": 2,
        }
    ]


def test_engine_pool_prefers_async_autoscaler_api(modal_app_module, monkeypatch):
    import asyncio

    monkeypatch.setattr(modal_app_module, "MIN_CONTAINERS", 1)
    calls: list[dict[str, int]] = []

    class _AsyncUpdate:
        async def aio(self, **kwargs: int) -> None:
            calls.append(kwargs)

        def __call__(self, **_kwargs: int) -> None:
            raise AssertionError("sync update_autoscaler should not be used when .aio exists")

    class _FakeEngine:
        update_autoscaler = _AsyncUpdate()

    monkeypatch.setattr(
        modal_app_module, "_engine_cls_for", lambda _bm: lambda *, base_model: _FakeEngine()
    )

    asyncio.run(modal_app_module._ModalEnginePool()._engine("Qwen/Qwen3.5-4B"))

    assert calls == [
        {
            "min_containers": 1,
            "scaledown_window": modal_app_module.scaledown_window_for("L4"),
        }
    ]


def test_runtime_secret_exports_hf_alias_and_deployment_identity(modal_app_module, monkeypatch):
    names = (
        "HF_API_KEY",
        "HF_TOKEN",
        "SERVING_DEPLOYMENT_MODE",
        "SERVING_CUSTOM_DOMAIN",
        "PLATFORM_BACKEND_URL",
        "FREESOLO_INTERNAL_KEY",
        "FREESOLO_DEPLOYMENT_SHA",
        "FREESOLO_DEPLOYMENT_ID",
        "SUPABASE_PROJECT_REF",
        "SUPABASE_PROJECT_REF_DEV",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "HOSTING_ADAPTER_TABLE",
        "HOSTING_MIN_CONTAINERS",
        "HOSTING_MAX_CONTAINERS",
        "HOSTING_BUFFER_CONTAINERS",
        "HOSTING_SCALEDOWN_WINDOW_SECONDS",
        "HOSTING_ROUTER_MIN",
        "HOSTING_MAX_MODEL_LEN",
        "HOSTING_TRUST_REMOTE_CODE",
        "HOSTING_TP_SIZE",
        "HOSTING_MAX_LORAS",
        "HOSTING_MAX_LORA_RANK",
        "HOSTING_MAX_CPU_LORAS",
        "HOSTING_SPECIALIZE_ACTIVE_LORA",
        "HOSTING_PRELOAD_CACHED_LORAS",
        "HOSTING_PROMPT_TOKEN_CACHE_SIZE",
        "HOSTING_DTYPE",
        "HOSTING_GPU_MEMORY_UTILIZATION",
        "HOSTING_KV_CACHE_DTYPE",
        "HOSTING_KV_CACHE_MEMORY_BYTES",
        "HOSTING_CALCULATE_KV_SCALES",
        "HOSTING_ENABLE_PREFIX_CACHING",
        "HOSTING_PREFIX_CACHING_HASH_ALGO",
        "HOSTING_PERFORMANCE_MODE",
        "HOSTING_BLOCK_SIZE",
        "HOSTING_MAX_CUDAGRAPH_CAPTURE_SIZE",
        "HOSTING_CUDAGRAPH_CAPTURE_SIZES",
        "HOSTING_DISABLE_LOG_STATS",
        "HOSTING_STREAM_INTERVAL",
        "HOSTING_SPECULATIVE_CONFIG",
        "HOSTING_ENFORCE_EAGER",
        "HOSTING_QUANTIZATION",
        "HOSTING_LOAD_FORMAT",
        "HOSTING_MAX_NUM_SEQS",
        "HOSTING_MAX_NUM_BATCHED_TOKENS",
        "HOSTING_MAX_NUM_PARTIAL_PREFILLS",
        "HOSTING_MAX_LONG_PARTIAL_PREFILLS",
        "HOSTING_LONG_PREFILL_TOKEN_THRESHOLD",
        "HOSTING_ASYNC_SCHEDULING",
        "HOSTING_SCHEDULING_POLICY",
        "HOSTING_ENABLE_CHUNKED_PREFILL",
        "HOSTING_GDN_PREFILL_BACKEND",
        "HOSTING_SCHEDULER_DELAY_FACTOR",
        "HOSTING_MAX_SEQ_LEN_TO_CAPTURE",
        "HOSTING_PREEMPTION_MODE",
        "HOSTING_RELOAD_INTERVAL_SECONDS",
        "NEXT_PUBLIC_SUPABASE_URL",
        "SUPABASE_SECRET_KEY",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_API_KEY", "hf_secret")
    monkeypatch.setenv("SERVING_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("SERVING_CUSTOM_DOMAIN", "serve-dev.freesolo.co")
    monkeypatch.setenv("FREESOLO_DEPLOYMENT_SHA", "abc123")
    monkeypatch.setenv("FREESOLO_DEPLOYMENT_ID", "456-2")
    monkeypatch.setenv("SUPABASE_PROJECT_REF", "production-project-ref")
    monkeypatch.setenv("SUPABASE_PROJECT_REF_DEV", "dev-project-ref")
    modal_app_module.modal.Secret.from_dict.reset_mock()

    modal_app_module._runtime_secret()

    assert modal_app_module.modal.Secret.from_dict.call_args.args[0] == {
        "HF_API_KEY": "hf_secret",
        "HF_TOKEN": "hf_secret",
        "SERVING_DEPLOYMENT_MODE": "development",
        "SERVING_CUSTOM_DOMAIN": "serve-dev.freesolo.co",
        "FREESOLO_DEPLOYMENT_SHA": "abc123",
        "FREESOLO_DEPLOYMENT_ID": "456-2",
        "SUPABASE_PROJECT_REF": "production-project-ref",
        "SUPABASE_PROJECT_REF_DEV": "dev-project-ref",
    }


def test_load_adapters_for_base_filters_records(modal_app_module, monkeypatch):
    class _Record:
        def __init__(self, base_model: str) -> None:
            self.base_model = base_model
            self.status = "ready"
            self.is_revision = True

    records = [_Record("Qwen/Qwen3.5-4B"), _Record("Qwen/Qwen3.5-0.8B")]
    monkeypatch.setattr("flash.serving.src.persistence.load_adapters", lambda settings: records)

    assert engine_support._load_adapters_for_base(object(), "Qwen/Qwen3.5-0.8B") == [records[1]]


def test_load_adapters_for_base_skips_hydration_failures(modal_app_module, monkeypatch, capsys):
    def _boom(_settings):
        raise TimeoutError("supabase timeout")

    monkeypatch.setattr("flash.serving.src.persistence.load_adapters", _boom)

    assert engine_support._load_adapters_for_base(object(), "Qwen/Qwen3.5-0.8B") == []
    assert "adapter hydration skipped" in capsys.readouterr().out


def test_adapter_source_cache_dir_ignores_adapter_id(modal_app_module):
    from flash.serving.src.schemas import AdapterRecord

    first = AdapterRecord.model_validate(
        {
            "adapter_id": "a",
            "repo_id": "org/run",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "subfolder": "sft/run/seed0/adapter",
            "repo_type": "dataset",
            "thinking": True,
        }
    )
    second = first.model_copy(update={"adapter_id": "b"})
    changed = first.model_copy(update={"subfolder": "sft/other/seed0/adapter"})

    root = Path("/cache/adapters")
    assert engine_support._adapter_source_cache_dir(root, first) == (
        engine_support._adapter_source_cache_dir(root, second)
    )
    assert engine_support._adapter_source_cache_dir(root, first) != (
        engine_support._adapter_source_cache_dir(root, changed)
    )


def test_ensure_adapter_local_reuses_same_source_download(modal_app_module, monkeypatch, tmp_path):
    import asyncio

    from flash.serving.src.schemas import AdapterRecord

    calls: list[dict[str, Any]] = []

    def _snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        adapter_dir = local_dir / "sft/run/seed0/adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
        return str(local_dir)

    class _Registry:
        def __init__(self) -> None:
            self.paths: dict[str, Path] = {}

        def local_path_is_stale(self, _record: Any) -> bool:
            return False

        def local_path(self, record: Any) -> Path | None:
            return self.paths.get(record.adapter_id)

        def set_local_path(self, record: Any, path: Path) -> None:
            self.paths[record.adapter_id] = path

    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)
    monkeypatch.setattr("flash.serving.src.settings.ADAPTER_CACHE_DIR", tmp_path / "adapters")

    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine.registry = _Registry()
    engine._source_locks = {}
    engine._source_locks_guard = asyncio.Lock()
    engine._source_paths = {}
    engine.settings = type("_Settings", (), {"hf_api_key": "hf_secret"})()

    first = AdapterRecord.model_validate(
        {
            "adapter_id": "a",
            "repo_id": "org/run",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "subfolder": "sft/run/seed0/adapter",
            "repo_type": "dataset",
            "thinking": True,
        }
    )
    second = first.model_copy(update={"adapter_id": "b"})

    first_path = asyncio.run(engine._ensure_adapter_local_locked(first))
    second_path = asyncio.run(engine._ensure_adapter_local_locked(second))

    assert first_path == second_path
    assert engine.registry.paths == {"a": first_path, "b": first_path}
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "org/run"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["allow_patterns"] == [
        "sft/run/seed0/adapter/**",
        "sft/run/seed0/adapter/*",
    ]


def test_ensure_adapter_local_uses_existing_volume_cache(modal_app_module, monkeypatch, tmp_path):
    import asyncio

    from flash.serving.src.schemas import AdapterRecord

    class _Registry:
        def __init__(self) -> None:
            self.paths: dict[str, Path] = {}

        def local_path_is_stale(self, _record: Any) -> bool:
            return False

        def local_path(self, record: Any) -> Path | None:
            return self.paths.get(record.adapter_id)

        def set_local_path(self, record: Any, path: Path) -> None:
            self.paths[record.adapter_id] = path

    def _snapshot_download(**_kwargs: Any) -> str:
        raise AssertionError("cached adapter should not call Hugging Face")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)
    monkeypatch.setattr("flash.serving.src.settings.ADAPTER_CACHE_DIR", tmp_path / "adapters")

    record = AdapterRecord.model_validate(
        {
            "adapter_id": "a",
            "repo_id": "org/run",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "subfolder": "sft/run/seed0/adapter",
            "repo_type": "dataset",
            "thinking": True,
        }
    )
    cached_path = engine_support._adapter_source_cache_dir(tmp_path / "adapters", record)
    cached_path = cached_path / "sft/run/seed0/adapter"
    cached_path.mkdir(parents=True)
    (cached_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (cached_path / "adapter_model.safetensors").write_bytes(b"weights")

    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine.registry = _Registry()
    engine._source_locks = {}
    engine._source_locks_guard = asyncio.Lock()
    engine._source_paths = {}
    engine.settings = type("_Settings", (), {"hf_api_key": "hf_secret"})()

    path = asyncio.run(engine._ensure_adapter_local_locked(record))

    assert path == cached_path
    assert engine.registry.paths == {"a": cached_path}


def test_ensure_adapter_local_redownloads_partial_volume_cache(
    modal_app_module, monkeypatch, tmp_path
):
    import asyncio

    from flash.serving.src.schemas import AdapterRecord

    class _Registry:
        def __init__(self) -> None:
            self.paths: dict[str, Path] = {}

        def local_path_is_stale(self, _record: Any) -> bool:
            return False

        def local_path(self, record: Any) -> Path | None:
            return self.paths.get(record.adapter_id)

        def set_local_path(self, record: Any, path: Path) -> None:
            self.paths[record.adapter_id] = path

    record = AdapterRecord.model_validate(
        {
            "adapter_id": "a",
            "repo_id": "org/run",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "subfolder": "sft/run/seed0/adapter",
            "repo_type": "dataset",
            "thinking": True,
        }
    )
    partial_path = engine_support._adapter_source_cache_dir(tmp_path / "adapters", record)
    partial_adapter_path = partial_path / "sft/run/seed0/adapter"
    partial_adapter_path.mkdir(parents=True)
    (partial_adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def _snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        adapter_dir = local_dir / "sft/run/seed0/adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
        return str(local_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)
    monkeypatch.setattr("flash.serving.src.settings.ADAPTER_CACHE_DIR", tmp_path / "adapters")

    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine.registry = _Registry()
    engine._source_locks = {}
    engine._source_locks_guard = asyncio.Lock()
    engine._source_paths = {}
    engine.settings = type("_Settings", (), {"hf_api_key": "hf_secret"})()

    path = asyncio.run(engine._ensure_adapter_local_locked(record))

    assert path == partial_adapter_path
    assert len(calls) == 1
    assert (path / "adapter_model.safetensors").read_bytes() == b"weights"
    assert engine.registry.paths == {"a": partial_adapter_path}


def test_preload_cached_loras_adds_only_volume_cached_adapters(
    modal_app_module, monkeypatch, tmp_path
):
    import asyncio

    from flash.serving.src.registry import AdapterRegistry, lora_int_id
    from flash.serving.src.schemas import AdapterRecord

    cached = AdapterRecord.model_validate(
        {
            "adapter_id": "cached",
            "repo_id": "org/cached",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "subfolder": "sft/cached/seed0/adapter",
            "repo_type": "dataset",
            "thinking": True,
        }
    )
    missing = cached.model_copy(update={"adapter_id": "missing", "repo_id": "org/missing"})
    cache_path = engine_support._adapter_source_cache_dir(tmp_path / "adapters", cached)
    cache_path = cache_path / "sft/cached/seed0/adapter"
    cache_path.mkdir(parents=True)
    (cache_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (cache_path / "adapter_model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr("flash.serving.src.settings.ADAPTER_CACHE_DIR", tmp_path / "adapters")

    class _Engine:
        def __init__(self) -> None:
            self.added: list[Any] = []
            self.pinned: list[int] = []

        async def add_lora(self, lora_request: Any) -> None:
            self.added.append(lora_request)

        async def pin_lora(self, lora_id: int) -> None:
            self.pinned.append(lora_id)

    registry = AdapterRegistry()
    registry.hydrate([cached, missing])
    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine.registry = registry
    engine.engine = _Engine()
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    engine._source_paths = {}
    engine._lora_requests = {}
    # _load() normally sets this; the object.__new__ engine here must too, or _add_lora_locked's
    # `if self._pin_loras` raises AttributeError (swallowed by the preload) and pinning never runs.
    engine._pin_loras = True

    asyncio.run(engine._preload_cached_loras())

    assert [request.lora_name for request in engine.engine.added] == ["cached"]
    assert engine.engine.pinned == [lora_int_id("cached")]
    assert registry.local_path(cached) == cache_path
    assert registry.local_path(missing) is None


def test_cached_lora_request_probes_on_int_id_collision(modal_app_module, monkeypatch):
    # Two distinct adapters whose sha1 masks collide to the same vLLM int id must NOT share it —
    # that cross-wires two orgs' LoRAs on one base-model engine. The second one linear-probes to the
    # next free id.
    from flash.serving.src import registry as registry_mod
    from flash.serving.src.schemas import AdapterRecord

    monkeypatch.setattr(registry_mod, "lora_int_id", lambda adapter_id: 42)

    def _rec(adapter_id: str, repo: str) -> AdapterRecord:
        return AdapterRecord.model_validate(
            {
                "adapter_id": adapter_id,
                "repo_id": repo,
                "base_model": "Qwen/Qwen3.5-0.8B",
                "thinking": True,
            }
        )

    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine._lora_requests = {}

    req_a = engine._cached_lora_request_locked(_rec("a", "org/a"), Path("/tmp/a"))
    req_b = engine._cached_lora_request_locked(_rec("b", "org/b"), Path("/tmp/b"))

    assert req_a.lora_int_id == 42
    assert req_b.lora_int_id == 43
    assert req_a.lora_int_id != req_b.lora_int_id
    # Re-resolving an already-cached adapter (same source) returns the same request/id, unchanged.
    assert engine._cached_lora_request_locked(_rec("a", "org/a"), Path("/tmp/a")) is req_a


def test_evict_uncached_alias_does_not_remove_a_colliding_adapter(modal_app_module, monkeypatch):
    from flash.serving.src import registry as registry_mod

    monkeypatch.setattr(registry_mod, "lora_int_id", lambda _adapter_id: 42)

    class _Request:
        lora_int_id = 42

    class _Engine:
        def __init__(self) -> None:
            self.removed: list[int] = []

        async def remove_lora(self, int_id: int) -> None:
            self.removed.append(int_id)

    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine.engine = _Engine()
    engine._lora_requests = {"tenant-b": (("org/b", "model", "sha", None), _Request())}

    asyncio.run(engine._evict_loaded_lora("tenant-a"))

    assert engine.engine.removed == []
    assert "tenant-b" in engine._lora_requests


def test_reporter_disabled_when_only_internal_key_is_set(modal_app_module):
    # The exact bug: a non-prod deployment has FREESOLO_INTERNAL_KEY but never set
    # PLATFORM_BACKEND_URL. With the prod default gone, reporting must be DISABLED (None), not
    # silently pointed at production billing.
    s = _settings(internal_key="secret-key")
    assert s.backend_url == ""
    assert modal_app_module._build_usage_reporter(s) is None


def test_reporter_disabled_when_key_unset_even_with_url(modal_app_module):
    s = _settings(backend_url="https://staging.example.com")
    assert modal_app_module._build_usage_reporter(s) is None


def test_reporter_enabled_only_when_both_url_and_key_set_explicitly(modal_app_module):
    # Reporting turns on only when BOTH are explicitly configured -> a real reporter callable.
    s = _settings(
        backend_url="https://staging.example.com",
        internal_key="secret-key",
    )
    reporter = modal_app_module._build_usage_reporter(s)
    assert reporter is not None
    assert callable(reporter)


def test_reporter_posts_to_the_configured_backend(modal_app_module, monkeypatch):
    # When enabled, it POSTs usage to {backend_url}/api/billing/serving-usage with the internal key
    # -- and to the configured backend, never a hardcoded prod fallback.
    # The reporter now uses a single persistent httpx.AsyncClient (headers set at init time,
    # not per call) to avoid TCP/TLS handshake overhead across bursts of usage reports.
    import asyncio

    captured: dict[str, Any] = {}

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            # Headers are set at client construction time (not per-call) in the persistent client.
            captured["init_headers"] = _k.get("headers", {})

        async def post(self, url: str, json: Any) -> _FakeResp:
            captured["url"] = url
            captured["json"] = json
            return _FakeResp()

        async def aclose(self) -> None:
            captured["closed"] = True

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    s = _settings(
        backend_url="https://staging.example.com/",  # trailing slash must be normalized
        internal_key="secret-key",
    )
    reporter = modal_app_module._build_usage_reporter(s)
    assert reporter is not None

    usage = {"adapter_id": "a1", "base_model": "Qwen/Qwen3.5-0.8B", "promptTokens": 1}
    asyncio.run(reporter(usage))

    assert captured["url"] == "https://staging.example.com/api/billing/serving-usage"
    assert captured["json"] == usage
    assert captured["init_headers"]["Authorization"] == "Bearer secret-key"


def test_reporter_reuses_single_client_across_calls(modal_app_module, monkeypatch):
    # The persistent client must be created ONCE per reporter, not once per report() call —
    # otherwise the TCP/TLS handshake cost is paid on every generation.
    import asyncio

    init_count = {"n": 0}

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            init_count["n"] += 1

        async def post(self, url: str, json: Any) -> _FakeResp:
            return _FakeResp()

        async def aclose(self) -> None:
            return None

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    s = _settings(backend_url="https://staging.example.com", internal_key="secret-key")
    reporter = modal_app_module._build_usage_reporter(s)
    assert reporter is not None

    async def _run():
        for _ in range(5):
            await reporter({"adapter_id": "a", "promptTokens": 1})

    asyncio.run(_run())
    assert init_count["n"] == 1  # one client for all five calls


@pytest.mark.parametrize("status_code", [408, 429, 500])
def test_reporter_retries_retryable_statuses_three_times(
    modal_app_module, monkeypatch, status_code: int
):
    import httpx

    calls: list[dict[str, Any]] = []

    class _FakeClient:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        async def post(self, url: str, json: Any) -> httpx.Response:
            calls.append(json)
            response_status = status_code if len(calls) <= 3 else 204
            return httpx.Response(response_status, request=httpx.Request("POST", url))

        async def aclose(self) -> None:
            pass

    async def _no_sleep(_delay: float) -> None:
        pass

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(modal_app_module.asyncio, "sleep", _no_sleep)
    reporter = modal_app_module._build_usage_reporter(
        _settings(backend_url="https://staging.example.com", internal_key="secret-key")
    )
    usage = {"requestId": "generation-1", "promptTokens": 1}

    asyncio.run(reporter(usage))

    assert calls == [usage, usage, usage, usage]


def test_reporter_retries_transport_failures_three_times(modal_app_module, monkeypatch):
    import httpx

    calls = 0

    class _FakeClient:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        async def post(self, url: str, json: Any) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls <= 3:
                raise httpx.ConnectError("backend unavailable")
            return httpx.Response(204, request=httpx.Request("POST", url))

        async def aclose(self) -> None:
            pass

    async def _no_sleep(_delay: float) -> None:
        pass

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(modal_app_module.asyncio, "sleep", _no_sleep)
    reporter = modal_app_module._build_usage_reporter(
        _settings(backend_url="https://staging.example.com", internal_key="secret-key")
    )

    asyncio.run(reporter({"requestId": "generation-1", "promptTokens": 1}))

    assert calls == 4


@pytest.mark.parametrize(
    ("usage", "status_code"),
    [
        ({"requestId": "generation-1", "promptTokens": 1}, 402),
        ({"promptTokens": 1}, 500),
    ],
)
def test_reporter_does_not_retry_non_idempotent_or_non_retryable_failures(
    modal_app_module, monkeypatch, usage: dict[str, Any], status_code: int
):
    import httpx

    calls = 0

    class _FakeClient:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        async def post(self, url: str, json: Any) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(status_code, request=httpx.Request("POST", url))

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    reporter = modal_app_module._build_usage_reporter(
        _settings(backend_url="https://staging.example.com", internal_key="secret-key")
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(reporter(usage))

    assert calls == 1
