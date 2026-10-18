"""Engine-side token telemetry, adapter cache, and Modal construction regressions."""

from __future__ import annotations

import asyncio
import hashlib
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from flash.serving.src.accounting.usage_outbox import DurableUsageOutbox, UsageOutboxError
from flash.serving.src.engine import support as engine_support
from flash.serving.src.store.settings import Settings


def _passthrough_decorator(*_a: Any, **_k: Any):
    # Modal's class/function/method decorators must return the decorated object so the module's
    # class + function bodies survive import.
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


def test_hosted_usage_outbox_fails_closed_on_partial_configuration(modal_app_module):
    settings = Settings(
        _env_file=None,
        PLATFORM_BACKEND_URL="https://api.example.com",
        FREESOLO_INTERNAL_KEY="internal-key",
        FREESOLO_DEPLOYMENT_ID="deployment-1",
    )

    with pytest.raises(UsageOutboxError, match="durable_usage_outbox_not_configured"):
        modal_app_module._build_usage_outbox(settings)


def test_hosted_usage_outbox_builds_only_with_complete_configuration(modal_app_module):
    settings = Settings(
        _env_file=None,
        PLATFORM_BACKEND_URL="https://api.example.com",
        FREESOLO_INTERNAL_KEY="internal-key",
        FREESOLO_DEPLOYMENT_ID="deployment-1",
        SUPABASE_URL="https://project.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="service-role",
    )

    outbox = modal_app_module._build_usage_outbox(settings)

    assert isinstance(outbox, DurableUsageOutbox)
    asyncio.run(outbox.aclose())


def test_hosted_usage_outbox_worker_id_is_unique_per_instance(modal_app_module):
    settings = Settings(
        _env_file=None,
        PLATFORM_BACKEND_URL="https://api.example.com",
        FREESOLO_INTERNAL_KEY="internal-key",
        FREESOLO_DEPLOYMENT_ID="deployment-1",
        SUPABASE_URL="https://project.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="service-role",
    )

    first = modal_app_module._build_usage_outbox(settings)
    second = modal_app_module._build_usage_outbox(settings)

    assert first._worker_id != second._worker_id
    assert first._generation_owner_id == second._generation_owner_id
    assert first._generation_owner_epoch != second._generation_owner_epoch
    asyncio.run(first.aclose())
    asyncio.run(second.aclose())


def test_stream_text_delta_keeps_native_delta_chunks(modal_app_module):
    previous = ""
    deltas = []

    for text in ("Hel", "lo", " hello"):
        delta, previous = engine_support._stream_text_delta(text, previous)
        deltas.append(delta)

    assert deltas == ["Hel", "lo", " hello"]
    assert previous == "Hello hello"


def test_lora_engine_scales_to_zero_by_default(modal_app_module):
    # every gpu engine class scales to zero, while the cpu router stays available to trigger them.
    assert modal_app_module.MIN_CONTAINERS == 0

    cls_calls = [call.kwargs for call in modal_app_module.app.cls.call_args_list]
    assert len(cls_calls) == len(modal_app_module.ENGINE_BY_KEY)
    # asserted on the kwargs modal is actually CALLED with, not on the constant: the floor is only
    # real if it reaches `app.cls`. `AutoscalerSettings.min_containers` is an `optional` proto field
    # (has_presence), so omitting it and sending 0 are distinguishable on the wire -- an omitted
    # field leaves the floor to whatever the server defaults to, which is not ours to assume.
    assert all(kwargs["min_containers"] == modal_app_module.MIN_CONTAINERS for kwargs in cls_calls)
    # Each engine class is pinned to ITS gpu tier's window, not one flat value.
    assert {kwargs["gpu"]: kwargs["scaledown_window"] for kwargs in cls_calls} == {
        gpu: modal_app_module.scaledown_window_for(gpu)
        for gpu, _mi in modal_app_module.ENGINE_BY_KEY
    }

    assert modal_app_module.app.function.call_count == 1
    assert modal_app_module.app.function.call_args.kwargs["min_containers"] == 1


def test_no_gpu_engine_is_capped_and_each_keeps_one_buffer(modal_app_module):
    # `base_model` is a modal.parameter(), so Modal gives each distinct value its own container pool
    # with its own autoscaling accounting. A fixed cap therefore ceilings a SINGLE model's capacity
    # rather than bounding total spend -- sustained load on one model cannot borrow headroom from an
    # idle one, so the cap converts demand into queueing on the hot tier. Spend is bounded by workspace
    # quotas and billing alerts instead. Asserted on the kwargs modal is actually CALLED with, because
    # a module constant that never reaches `app.cls` governs nothing.
    assert modal_app_module.MAX_CONTAINERS is None
    assert modal_app_module.BUFFER_CONTAINERS == 1

    cls_calls = [call.kwargs for call in modal_app_module.app.cls.call_args_list]
    assert len(cls_calls) == len(modal_app_module.ENGINE_BY_KEY)
    assert all(kwargs["max_containers"] is None for kwargs in cls_calls)
    # One spare warm container per engine absorbs a burst past TARGET_INPUTS without a cold boot.
    # `buffer_containers` only provisions while the Function is ACTIVE, so this preserves scale-to-zero
    # (MIN_CONTAINERS stays 0) rather than paying for an idle gpu.
    assert all(kwargs["buffer_containers"] == 1 for kwargs in cls_calls)
    assert all(kwargs["min_containers"] == 0 for kwargs in cls_calls)

    # The cpu router is likewise uncapped: it is the front door that triggers cold engines, and
    # capping it would throttle every model at once rather than bounding gpu spend per model. It
    # carries the same buffer, so a burst does not queue behind the front door it just cleared.
    router_kwargs = modal_app_module.app.function.call_args.kwargs
    assert "max_containers" not in router_kwargs
    assert router_kwargs["buffer_containers"] == 1


def test_scaledown_window_is_per_tier_and_cheaper_tiers_release_sooner(modal_app_module):
    # The whole point of the table: an idle container bills at the full gpu rate, so a cheap
    # fast-booting tier must not hold a card as long as the 35B's ~1010s-boot H200 does.
    from flash.serving.src.engine.model_config import base_models, gpu_for

    window_for = modal_app_module.scaledown_window_for

    assert window_for("L4") < window_for("L40S") < window_for("H100") < window_for("H200")
    # The H200 keeps the full legacy hold (boot is ~17 min; a miss stalls the user that long).
    assert window_for("H200") == 1800
    # Every catalog tier resolves to a positive window, so no model falls back by accident.
    assert all(window_for(gpu_for(bm)) > 0 for bm in base_models())
    # B200 now carries all three shipped tiers and its window is MEASURED, not inherited: the
    # slowest cold boot on the card was the 27B at 1821s, so it holds longer than the H200 it
    # replaced. B300 ships nothing and keeps the unmeasured placeholder.
    assert window_for("B200") == 2100
    assert window_for("B300") == 1800
    # Every tier that actually serves holds at least as long as its slowest measured cold boot.
    assert all(window_for(gpu_for(bm)) >= 2100 for bm in base_models())


def test_unknown_gpu_tier_is_rejected_not_silently_defaulted(modal_app_module):
    # `gpu` in the serving catalog is a plain string. A typo or an unvalidated new card used to fall
    # through `dict.get` to the default window and DEPLOY, billing that card's real hourly rate on a
    # tier nobody qualified. Every one of these once returned 1800 silently.
    from flash.serving.src.engine.model_config import base_models, gpu_for

    window_for = modal_app_module.scaledown_window_for

    for bogus in ("b200", "B2OO", "H2OO", "A100", "TOTALLY_FAKE", ""):
        with pytest.raises(ValueError, match="Unsupported serving GPU tier"):
            window_for(bogus)

    # The gate is membership in the shipped table, so the two stay in sync by construction.
    assert frozenset(modal_app_module.SCALEDOWN_WINDOW_SECONDS_BY_GPU) == (
        modal_app_module.SUPPORTED_GPUS
    )
    # Every tier a cataloged model actually routes to must be supported.
    assert {gpu_for(bm) for bm in base_models()} <= modal_app_module.SUPPORTED_GPUS


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
                "preserve_thinking": False,
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
        "preserve_thinking": False,
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


def test_hosted_prompt_cache_key_utf8_encodes_accepted_tool_declarations(modal_app_module):
    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine._prompt_cache_size = 1

    class _Payload:
        messages: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "weather"}]
        prompt = None
        chat_template_kwargs: ClassVar[dict[str, object]] = {}
        tool_choice = "auto"
        tools: ClassVar[list[dict[str, object]]] = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "prévisions météo",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "object",
                                "properties": {
                                    "forecast_🌦": {
                                        "type": "string",
                                        "description": "réponse détaillée",
                                        "enum": ["ensoleillé", "nuageux ☁"],
                                    }
                                },
                                "required": ["forecast_🌦"],
                                "additionalProperties": False,
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    key = engine._prompt_cache_key(_Payload(), thinking_default=False)

    assert key is not None
    assert key == engine._prompt_cache_key(_Payload(), thinking_default=False)


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
        "preserve_thinking": False,
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
        "preserve_thinking": False,
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
    engine.base_model = "Qwen/Qwen3.5-9B"
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

    # health reports the served pre-quantized checkpoint and baked-in 9b config.
    assert engine._health() == {
        "ok": True,
        # No engine built on this bare instance -> not dead -> ok stays True.
        "engine_dead": False,
        "base_model": "Qwen/Qwen3.5-9B",
        "served_model": "Freesolo-Co/Qwen3.5-9B-FP8",
        "immutable_identity": None,
        # A pre-quant FP8 checkpoint carries fp8 weights (auto-detected) + the global fp8 KV cache.
        "quantization": "fp8",
        "kv_cache_dtype": "fp8",
        "adapters": 2,
        # configured_gpu reflects the active model's per-model gpu tier.
        "configured_gpu": modal_app_module.gpu_for("Qwen/Qwen3.5-9B"),
        "cuda_available": True,
        "device_name": "NVIDIA H100",
        "enable_prefix_caching": True,
        "prompt_token_cache_size": 2048,
        "prompt_token_cache_entries": 1,
        "max_model_len": 32768,
    }


def test_runtime_secret_exports_canonical_hf_token_and_deployment_identity(
    modal_app_module, monkeypatch
):
    names = (
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
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("SERVING_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("SERVING_CUSTOM_DOMAIN", "serve-dev.freesolo.co")
    monkeypatch.setenv("FREESOLO_DEPLOYMENT_SHA", "abc123")
    monkeypatch.setenv("FREESOLO_DEPLOYMENT_ID", "456-2")
    monkeypatch.setenv("SUPABASE_PROJECT_REF", "production-project-ref")
    monkeypatch.setenv("SUPABASE_PROJECT_REF_DEV", "dev-project-ref")
    modal_app_module.modal.Secret.from_dict.reset_mock()

    modal_app_module._runtime_secret()

    assert modal_app_module.modal.Secret.from_dict.call_args.args[0] == {
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
            self.is_checkpoint = True

    records = [_Record("Qwen/Qwen3.8-27B"), _Record("Qwen/Qwen3.5-9B")]
    monkeypatch.setattr(
        "flash.serving.src.store.persistence.load_adapters", lambda settings: records
    )

    assert engine_support._load_adapters_for_base(object(), "Qwen/Qwen3.5-9B") == [records[1]]


def test_load_adapters_for_base_skips_hydration_failures(modal_app_module, monkeypatch, capsys):
    def _boom(_settings):
        raise TimeoutError("supabase timeout")

    monkeypatch.setattr("flash.serving.src.store.persistence.load_adapters", _boom)

    assert engine_support._load_adapters_for_base(object(), "Qwen/Qwen3.5-9B") == []
    assert "adapter hydration skipped" in capsys.readouterr().out


def _checkpoint_record(
    run_id: str,
    repo_id: str,
    *,
    subfolder: str | None = None,
) -> dict[str, Any]:
    checkpoint_id = f"{run_id}/final"
    return {
        "adapter_id": checkpoint_id,
        "repo_id": repo_id,
        "org_id": "org-1",
        "base_model": "Qwen/Qwen3.5-9B",
        "checkpoint": checkpoint_id,
        "run_id": run_id,
        "checkpoint_step": None,
        "artifact_revision": hashlib.sha1(run_id.encode()).hexdigest(),
        "artifact_digest": hashlib.sha256(f"{run_id}-artifact".encode()).hexdigest(),
        "artifact_fingerprint": hashlib.sha256(f"{run_id}-binding".encode()).hexdigest(),
        "lora_rank": 16,
        "subfolder": subfolder,
        "repo_type": "dataset",
        "thinking": True,
    }


def test_adapter_source_cache_dir_ignores_adapter_id(modal_app_module):
    from flash.serving.src.io.schemas import AdapterRecord

    first = AdapterRecord.model_validate(
        _checkpoint_record("a", "org/run", subfolder="sft/run/seed0/adapter")
    )
    second = AdapterRecord.model_validate(
        {
            **_checkpoint_record("b", "org/run", subfolder="sft/run/seed0/adapter"),
            "artifact_revision": first.artifact_revision,
            "artifact_digest": first.artifact_digest,
            "artifact_fingerprint": first.artifact_fingerprint,
        }
    )
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

    from flash.serving.src.io.schemas import AdapterRecord

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
    monkeypatch.setattr("flash.serving.src.store.settings.ADAPTER_CACHE_DIR", tmp_path / "adapters")

    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine.registry = _Registry()
    engine._source_locks = {}
    engine._source_locks_guard = asyncio.Lock()
    engine._source_paths = {}
    engine.settings = type("_Settings", (), {"hf_api_key": "hf_secret"})()

    first = AdapterRecord.model_validate(
        _checkpoint_record("a", "org/run", subfolder="sft/run/seed0/adapter")
    )
    second = AdapterRecord.model_validate(
        {
            **_checkpoint_record("b", "org/run", subfolder="sft/run/seed0/adapter"),
            "artifact_revision": first.artifact_revision,
            "artifact_digest": first.artifact_digest,
            "artifact_fingerprint": first.artifact_fingerprint,
        }
    )

    first_path = asyncio.run(engine._ensure_adapter_local_locked(first))
    second_path = asyncio.run(engine._ensure_adapter_local_locked(second))

    assert first_path == second_path
    assert engine.registry.paths == {first.adapter_id: first_path, second.adapter_id: first_path}
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "org/run"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["allow_patterns"] == [
        "sft/run/seed0/adapter/**",
        "sft/run/seed0/adapter/*",
    ]


def test_ensure_adapter_local_uses_existing_volume_cache(modal_app_module, monkeypatch, tmp_path):
    import asyncio

    from flash.serving.src.io.schemas import AdapterRecord

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
    monkeypatch.setattr("flash.serving.src.store.settings.ADAPTER_CACHE_DIR", tmp_path / "adapters")

    record = AdapterRecord.model_validate(
        _checkpoint_record("a", "org/run", subfolder="sft/run/seed0/adapter")
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
    assert engine.registry.paths == {record.adapter_id: cached_path}


def test_ensure_adapter_local_redownloads_partial_volume_cache(
    modal_app_module, monkeypatch, tmp_path
):
    import asyncio

    from flash.serving.src.io.schemas import AdapterRecord

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
        _checkpoint_record("a", "org/run", subfolder="sft/run/seed0/adapter")
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
    monkeypatch.setattr("flash.serving.src.store.settings.ADAPTER_CACHE_DIR", tmp_path / "adapters")

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
    assert engine.registry.paths == {record.adapter_id: partial_adapter_path}


def test_preload_cached_loras_adds_only_volume_cached_adapters(
    modal_app_module, monkeypatch, tmp_path
):
    import asyncio

    from flash.serve.contract.provenance import engine_adapter_name
    from flash.serving.src.io.schemas import AdapterRecord
    from flash.serving.src.store.registry import AdapterRegistry, lora_int_id

    cached = AdapterRecord.model_validate(
        _checkpoint_record("cached", "org/cached", subfolder="sft/cached/seed0/adapter")
    )
    missing = AdapterRecord.model_validate(
        _checkpoint_record("missing", "org/missing", subfolder="sft/missing/seed0/adapter")
    )
    cache_path = engine_support._adapter_source_cache_dir(tmp_path / "adapters", cached)
    cache_path = cache_path / "sft/cached/seed0/adapter"
    cache_path.mkdir(parents=True)
    (cache_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (cache_path / "adapter_model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr("flash.serving.src.store.settings.ADAPTER_CACHE_DIR", tmp_path / "adapters")

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
    engine._lora_entries = {}
    # _load() normally sets this; the object.__new__ engine here must too, or _add_lora_locked's
    # `if self._pin_loras` raises AttributeError (swallowed by the preload) and pinning never runs.
    engine._pin_loras = True

    asyncio.run(engine._preload_cached_loras())

    adapter_name = engine_adapter_name(cached.org_id, cached.adapter_id)
    assert [request.lora_name for request in engine.engine.added] == [adapter_name]
    assert engine.engine.pinned == [lora_int_id(adapter_name)]
    assert registry.local_path(cached) == cache_path
    assert registry.local_path(missing) is None


def test_cached_lora_request_probes_on_int_id_collision(modal_app_module, monkeypatch, tmp_path):
    # two distinct adapters whose sha1 masks collide to the same vllm int id must not share it.
    # unconfirmed entries still occupy their id because vllm may retain the corresponding weights.
    from flash.serving.src.engine.lora_engine import _LoraEntry
    from flash.serving.src.io.schemas import AdapterRecord
    from flash.serving.src.store import registry as registry_mod

    monkeypatch.setattr(registry_mod, "lora_int_id", lambda adapter_id: 42)

    def _rec(adapter_id: str, repo: str) -> AdapterRecord:
        return AdapterRecord.model_validate(_checkpoint_record(adapter_id, repo))

    engine = object.__new__(modal_app_module._LoraEngineImpl)
    assert not hasattr(engine, "_lora_entries")

    record_a = _rec("a", "org/a")
    req_a = engine._cached_lora_request_locked(record_a, tmp_path / "a")
    req_b = engine._cached_lora_request_locked(_rec("b", "org/b"), tmp_path / "b")

    assert req_a.lora_int_id == 42
    assert req_b.lora_int_id == 43
    assert req_a.lora_int_id != req_b.lora_int_id
    # re-resolving an already-cached adapter returns the same request and id.
    assert engine._cached_lora_request_locked(record_a, tmp_path / "a") is req_a
    key_a = (record_a.org_id, record_a.adapter_id)
    entry_a = engine._lora_entries[key_a]
    engine._lora_entries[key_a] = _LoraEntry(entry_a.source_ident, req_a, "unconfirmed")
    req_c = engine._cached_lora_request_locked(_rec("c", "org/c"), tmp_path / "c")
    assert req_c.lora_int_id == 44


def test_evict_uncached_alias_does_not_remove_a_colliding_adapter(modal_app_module, monkeypatch):
    from flash.serving.src.engine.lora_engine import _LoraEntry
    from flash.serving.src.store import registry as registry_mod

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
    engine._lora_entries = {
        "tenant-b": _LoraEntry(("org/b", "model", "sha", None), _Request(), "loaded")
    }

    asyncio.run(engine._evict_loaded_lora("tenant-a"))

    assert engine.engine.removed == []
    assert "tenant-b" in engine._lora_entries
