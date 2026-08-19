"""engine construction, generation, streaming, accounting, images, and death handling."""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from flash.serve.runtime import (
    AdapterSpec,
    EngineConfig,
    EngineDeadError,
    GenerationRequest,
    RuntimeNotReadyError,
    StreamDelta,
    StreamFinished,
    StreamReady,
    VllmLoraRuntime,
)
from flash.serve.runtime import engine as engine_module

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}
MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40


class _Tokenizer:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def __init__(self) -> None:
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 9
        self.template_calls: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any):
        cls.calls.append((model, kwargs))
        return cls()

    def apply_chat_template(self, _messages, **kwargs: Any):
        self.template_calls.append(kwargs)
        return [10, 11, 12]

    def encode(self, _prompt: str, **_kwargs: Any):
        return [20, 21]


class _Processor:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.template_calls: list[tuple[Any, dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any):
        cls.calls.append((model, kwargs))
        return cls()

    def apply_chat_template(self, messages: Any, **kwargs: Any):
        self.template_calls.append((messages, kwargs))
        return "rendered image prompt"


@dataclass
class _AsyncEngineArgs:
    model: str = ""
    revision: str | None = None
    tokenizer: str = ""
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    enable_lora: bool = False
    max_loras: int = 0
    max_lora_rank: int = 0
    max_cpu_loras: int = 0
    reasoning_parser: str | None = None
    limit_mm_per_prompt: dict[str, int] | None = None
    mm_processor_cache_gb: int | None = None
    enable_tower_connector_lora: bool | None = None
    dtype: str | None = None
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    max_model_len: int | None = None
    nested: Any = None


class _LoRARequest:
    def __init__(self, name: str, lora_int_id: int, path: str) -> None:
        self.lora_name = name
        self.lora_int_id = lora_int_id
        self.lora_path = path


class _SamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _StructuredOutputsParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _RequestOutputKind:
    FINAL_ONLY = "final-only"
    DELTA = "delta"


class _Engine:
    latest: _Engine | None = None
    args: _AsyncEngineArgs | None = None

    def __init__(self) -> None:
        self.errored = False
        self.responses: list[Any] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.added: list[_LoRARequest] = []
        self.pinned: list[int] = []
        self.removed: list[int] = []
        self.shutdown_called = False
        self.seen_images: list[Any] = []
        type(self).latest = self

    @classmethod
    def from_engine_args(cls, args: _AsyncEngineArgs):
        cls.args = args
        return cls()

    async def add_lora(self, request: _LoRARequest) -> None:
        self.added.append(request)

    async def pin_lora(self, int_id: int) -> None:
        self.pinned.append(int_id)

    async def remove_lora(self, int_id: int) -> None:
        self.removed.append(int_id)

    async def shutdown(self) -> None:
        self.shutdown_called = True

    async def generate(
        self,
        prompt: dict[str, Any],
        sampling: _SamplingParams,
        request_id: str,
        *,
        lora_request: _LoRARequest | None,
        reasoning_ended: bool | None,
        reasoning_parser_kwargs: dict[str, Any] | None,
    ):
        self.generate_calls.append(
            {
                "prompt": prompt,
                "sampling": sampling,
                "request_id": request_id,
                "lora_request": lora_request,
                "reasoning_ended": reasoning_ended,
                "reasoning_parser_kwargs": reasoning_parser_kwargs,
            }
        )
        image = (prompt.get("multi_modal_data") or {}).get("image")
        if image is not None:
            self.seen_images.append(image)
        scenario = self.responses.pop(0)
        if isinstance(scenario, BaseException):
            raise scenario
        for output in scenario:
            yield output


@pytest.fixture(autouse=True)
def _no_hub_credential(monkeypatch):
    """pin the loader to the credentialless case unless a test says otherwise.

    whether the loader may reach the hub depends on a real ambient token, so without this the
    result would differ between a developer machine with HF_TOKEN exported and CI without one.
    the packaged serving container is credentialless, so that is the default here; the test that
    owns the credentialed path overrides this.
    """

    monkeypatch.setattr(engine_module, "_has_hub_credential", lambda _token: False)


@pytest.fixture(autouse=True)
def _fake_modules(monkeypatch):
    _Tokenizer.calls = []
    _Processor.calls = []
    _Engine.latest = None
    _Engine.args = None

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _Tokenizer
    transformers.AutoProcessor = _Processor

    vllm = types.ModuleType("vllm")
    vllm.AsyncEngineArgs = _AsyncEngineArgs
    vllm.AsyncLLMEngine = _Engine
    vllm.SamplingParams = _SamplingParams

    sampling = types.ModuleType("vllm.sampling_params")
    sampling.RequestOutputKind = _RequestOutputKind
    sampling.StructuredOutputsParams = _StructuredOutputsParams

    lora = types.ModuleType("vllm.lora")
    lora_request = types.ModuleType("vllm.lora.request")
    lora_request.LoRARequest = _LoRARequest

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_request)


@pytest.fixture
def adapter_dir(tmp_path: Path) -> Path:
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return path


def _output(
    text: str,
    token_ids: list[int],
    *,
    prompt_tokens: int = 3,
    cached_tokens: Any = 0,
    finish_reason: str | None = "stop",
):
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                text=text,
                token_ids=token_ids,
                finish_reason=finish_reason,
            )
        ],
        prompt_token_ids=list(range(prompt_tokens)),
        num_cached_tokens=cached_tokens,
    )


def test_exact_engine_args_and_processor_construction() -> None:
    runtime = VllmLoraRuntime(
        EngineConfig(
            model="logical/model",
            served_model="served/model",
            tokenizer_model="tokenizer/model",
            hf_token="secret",
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            trust_remote_code=True,
            max_loras=6,
            max_lora_rank=128,
            max_cpu_loras=12,
            image_limit=4,
            mm_processor_cache_gb=1.5,
            enable_tower_connector_lora=True,
            reasoning_parser="qwen3",
            engine_args={
                "dtype": "bfloat16",
                "quantization": None,
                "kv_cache_dtype": "fp8",
                "max_model_len": 32768,
            },
        )
    )
    asyncio.run(runtime.start())

    assert _Engine.args == _AsyncEngineArgs(
        model="served/model",
        revision=MODEL_REVISION,
        tokenizer="tokenizer/model",
        tokenizer_revision=TOKENIZER_REVISION,
        trust_remote_code=True,
        enable_lora=True,
        max_loras=6,
        max_lora_rank=128,
        max_cpu_loras=12,
        reasoning_parser="qwen3",
        limit_mm_per_prompt={"image": 4},
        mm_processor_cache_gb=1.5,
        enable_tower_connector_lora=True,
        dtype="bfloat16",
        quantization=None,
        kv_cache_dtype="fp8",
        max_model_len=32768,
    )
    assert _Processor.calls == [
        (
            "tokenizer/model",
            {
                "token": "secret",
                "trust_remote_code": True,
                "local_files_only": True,
                "revision": TOKENIZER_REVISION,
            },
        )
    ]
    assert _Tokenizer.calls == []
    assert runtime.health().served_model == "served/model"
    asyncio.run(runtime.close())


def test_text_tokenizer_receives_exact_revision() -> None:
    runtime = VllmLoraRuntime(
        EngineConfig(
            model="served/model",
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            tokenizer_kwargs={"use_fast": True},
        )
    )
    asyncio.run(runtime.start())

    assert _Tokenizer.calls == [
        (
            "served/model",
            {
                "token": None,
                "trust_remote_code": False,
                "local_files_only": True,
                "revision": TOKENIZER_REVISION,
                "use_fast": True,
            },
        )
    ]
    assert _Processor.calls == []
    assert _Engine.args is not None
    assert _Engine.args.revision == MODEL_REVISION
    assert _Engine.args.tokenizer_revision == TOKENIZER_REVISION
    asyncio.run(runtime.close())


@pytest.mark.parametrize(
    ("image_limit", "recorder", "other"),
    [
        (None, _Tokenizer, _Processor),
        (4, _Processor, _Tokenizer),
    ],
    ids=["text", "multimodal"],
)
def test_a_credentialless_loader_never_reaches_the_network(
    image_limit: int | None, recorder: Any, other: Any
) -> None:
    """without a credential both loaders must resolve from the hydrated cache alone.

    the served base model is private and the artifact token is deleted at the end of bootstrap,
    so every later start of the packaged container is credentialless. transformers enumerates
    the repo's additional_chat_templates/ over the network unless told otherwise, and a 401
    there surfaces as RepositoryNotFoundError, which it deliberately re-raises instead of
    falling back to the cache. so a populated cache is not enough on its own.

    setting HF_HUB_OFFLINE cannot substitute for this: huggingface_hub binds that constant when
    it is imported, and hydration imports it earlier in the same process, so a later env write
    is silently inert.
    """

    runtime = VllmLoraRuntime(
        EngineConfig(
            model="served/model",
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            image_limit=image_limit,
        )
    )
    asyncio.run(runtime.start())

    assert other.calls == []
    assert len(recorder.calls) == 1
    assert recorder.calls[0][1]["local_files_only"] is True
    asyncio.run(runtime.close())


@pytest.mark.parametrize(
    ("image_limit", "recorder"),
    [(None, _Tokenizer), (4, _Processor)],
    ids=["text", "multimodal"],
)
def test_a_credentialed_loader_may_still_populate_a_cold_cache(
    monkeypatch, image_limit: int | None, recorder: Any
) -> None:
    """with a credential the loader must stay allowed to download.

    the generated app from `flash serve setup` is the other consumer of this runtime. it starts
    against an empty volume and fetches the base model with HF_TOKEN, so forcing local-only
    unconditionally would make its very first start fail closed instead of populating the cache.
    """

    monkeypatch.setattr(engine_module, "_has_hub_credential", lambda _token: True)
    runtime = VllmLoraRuntime(
        EngineConfig(
            model="served/model",
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            image_limit=image_limit,
        )
    )
    asyncio.run(runtime.start())

    assert len(recorder.calls) == 1
    assert recorder.calls[0][1]["local_files_only"] is False
    asyncio.run(runtime.close())


def test_an_ambient_token_counts_as_a_hub_credential(monkeypatch) -> None:
    """the generated app authenticates through the environment, not through EngineConfig.

    it passes no hf_token and relies on HF_TOKEN being present, so a check that only read
    config.hf_token would report "no credential" for a container that can in fact download.
    huggingface_hub is asked directly because its own order also covers HUGGING_FACE_HUB_TOKEN,
    the cached login file, and OIDC exchange.
    """

    # the autouse fixture replaces this function for every other test; undo that here so the
    # real implementation is the thing under test rather than the stub standing in for it.
    monkeypatch.undo()
    from flash.serve.runtime.engine import _has_hub_credential

    assert _has_hub_credential("explicit-token") is True

    hub = types.ModuleType("huggingface_hub")
    hub.get_token = lambda: "ambient-token"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert _has_hub_credential(None) is True

    hub.get_token = lambda: None
    assert _has_hub_credential(None) is False

    def _explode() -> str:
        raise RuntimeError("hub unavailable")

    hub.get_token = _explode
    assert _has_hub_credential(None) is False


def test_runtime_mappings_thaw_to_detached_shape_preserving_builtin_containers() -> None:
    nested = {
        "mapping": {"value": 1},
        "list": [2],
        "tuple": (3,),
        "set": {4},
        "frozenset": frozenset({5}),
    }
    text_runtime = VllmLoraRuntime(
        EngineConfig(
            model="model",
            engine_args={"nested": nested},
            tokenizer_kwargs={"nested": nested},
        )
    )
    asyncio.run(text_runtime.start())

    assert _Engine.args is not None
    engine_nested = _Engine.args.nested
    tokenizer_nested = _Tokenizer.calls[0][1]["nested"]
    for thawed in (engine_nested, tokenizer_nested):
        assert type(thawed) is dict
        assert type(thawed["mapping"]) is dict
        assert type(thawed["list"]) is list
        assert type(thawed["tuple"]) is tuple
        assert type(thawed["set"]) is set
        assert type(thawed["frozenset"]) is frozenset
    engine_nested["mapping"]["value"] = 10
    engine_nested["list"].append(20)
    assert text_runtime.config.engine_args["nested"]["mapping"]["value"] == 1
    assert tuple(text_runtime.config.engine_args["nested"]["list"]) == (2,)
    asyncio.run(text_runtime.close())

    image_runtime = VllmLoraRuntime(
        EngineConfig(
            model="model",
            image_limit=1,
            processor_kwargs={"nested": nested},
        )
    )
    asyncio.run(image_runtime.start())
    processor_nested = _Processor.calls[0][1]["nested"]
    assert type(processor_nested) is dict
    assert type(processor_nested["list"]) is list
    assert type(processor_nested["tuple"]) is tuple
    assert type(processor_nested["set"]) is set
    assert type(processor_nested["frozenset"]) is frozenset
    processor_nested["set"].add(40)
    assert image_runtime.config.processor_kwargs["nested"]["set"] == frozenset({4})
    asyncio.run(image_runtime.close())


def test_nonstream_generation_binds_thinking_structured_outputs_and_accounting(
    adapter_dir: Path,
) -> None:
    runtime = VllmLoraRuntime(
        EngineConfig(model="model", reasoning_parser="qwen3", prompt_cache_size=4)
    )
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    spec = AdapterSpec(
        adapter_id="adapter",
        path=str(adapter_dir),
        incarnation="incarnation-1",
        thinking=False,
        structured_outputs=SCHEMA,
    )
    asyncio.run(runtime.register_adapter(spec))
    engine.responses.extend(
        [
            [_output('{"answer":"yes"}', [7, 8], prompt_tokens=5, cached_tokens=2)],
            [_output("again", [9], prompt_tokens=5, cached_tokens=None)],
        ]
    )
    request = GenerationRequest(
        adapter_id="adapter",
        expected_incarnation="incarnation-1",
        messages=[{"role": "user", "content": "answer"}],
        chat_template_kwargs={"enable_thinking": True, "return_tensors": "pt"},
    )

    result = asyncio.run(runtime.generate(request))
    second = asyncio.run(runtime.generate(request))

    assert result.text == '{"answer":"yes"}'
    assert result.token_ids == (7, 8)
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 2
    assert result.cached_tokens == 2
    assert result.cached_tokens_reported is True
    assert result.thinking is False
    assert result.incarnation == "incarnation-1"
    assert second.cached_tokens == 0
    assert second.cached_tokens_reported is False
    tokenizer = runtime._tokenizer
    assert tokenizer.template_calls == [
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": False,
            "enable_thinking": False,
        }
    ]
    call = engine.generate_calls[0]
    assert call["sampling"].kwargs["output_kind"] == "final-only"
    assert call["sampling"].kwargs["structured_outputs"].kwargs == {"json": SCHEMA}
    assert call["reasoning_ended"] is True
    assert call["reasoning_parser_kwargs"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert call["lora_request"] is engine.added[0]
    assert runtime.health().prompt_cache_entries == 1
    asyncio.run(runtime.close())


def test_adapterless_structured_generation_binds_default_thinking_false() -> None:
    runtime = VllmLoraRuntime(EngineConfig(model="model", reasoning_parser="qwen3"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.responses.append([_output('{"answer":"yes"}', [1])])
    request = GenerationRequest(
        messages=[{"role": "user", "content": "answer"}],
        chat_template_kwargs={"enable_thinking": True},
        structured_outputs=SCHEMA,
    )

    result = asyncio.run(runtime.generate(request))

    assert result.thinking is False
    tokenizer = runtime._tokenizer
    assert tokenizer.template_calls[0]["enable_thinking"] is False
    call = engine.generate_calls[0]
    assert call["reasoning_ended"] is True
    assert call["reasoning_parser_kwargs"] == {"chat_template_kwargs": {"enable_thinking": False}}
    asyncio.run(runtime.close())


def test_stream_trusts_pinned_delta_output_and_counts_chunks() -> None:
    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    # no delta reports a cache count, so the finished event must report cache state as unreported.
    engine.responses.append(
        [
            _output("he", [1], prompt_tokens=4, cached_tokens=None),
            _output("llo", [2, 3], prompt_tokens=4, cached_tokens=None, finish_reason=None),
            _output("!", [4], prompt_tokens=4, cached_tokens=None),
        ]
    )

    async def collect():
        return [event async for event in runtime.stream(GenerationRequest(prompt="hello"))]

    events = asyncio.run(collect())
    assert isinstance(events[0], StreamReady)
    assert [event.text for event in events if isinstance(event, StreamDelta)] == ["he", "llo", "!"]
    final = events[-1]
    assert isinstance(final, StreamFinished)
    assert final.text == "hello!"
    assert final.prompt_tokens == 4
    assert final.completion_tokens == 4
    assert final.cached_tokens == 0
    assert final.cached_tokens_reported is False
    assert engine.generate_calls[0]["sampling"].kwargs["output_kind"] == "delta"
    asyncio.run(runtime.close())


def _delta_without_prompt_metadata(text: str, token_ids: list[int], finish_reason: str | None):
    """a delta carrying no prompt or cache metadata, as vllm emits after the first one."""
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(text=text, token_ids=token_ids, finish_reason=finish_reason),
        ],
    )


def test_stream_accounting_reads_metadata_from_the_first_reporting_delta() -> None:
    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    # vllm reports prompt and cache metadata on the first delta only; later deltas omit it.
    engine.responses.append(
        [
            _output("he", [1], prompt_tokens=7, cached_tokens=5, finish_reason=None),
            _delta_without_prompt_metadata("llo", [2, 3], None),
            _delta_without_prompt_metadata("!", [4], "stop"),
        ]
    )

    async def collect():
        return [event async for event in runtime.stream(GenerationRequest(prompt="hello"))]

    events = asyncio.run(collect())
    final = events[-1]
    assert isinstance(final, StreamFinished)
    assert final.text == "hello!"
    assert final.finish_reason == "stop"
    assert final.prompt_tokens == 7
    assert final.cached_tokens == 5
    assert final.cached_tokens_reported is True
    assert final.completion_tokens == 4
    asyncio.run(runtime.close())


def test_stream_requires_some_delta_to_report_prompt_tokens() -> None:
    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.responses.append(
        [
            _delta_without_prompt_metadata("he", [1], None),
            _delta_without_prompt_metadata("llo", [2], "stop"),
        ]
    )

    async def collect():
        return [event async for event in runtime.stream(GenerationRequest(prompt="hello"))]

    with pytest.raises(RuntimeNotReadyError, match="expanded prompt token count"):
        asyncio.run(collect())
    asyncio.run(runtime.close())


def test_stop_sequences_reach_sampling_params_in_both_paths() -> None:
    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.responses.append([_output("hi", [1])])
    asyncio.run(runtime.generate(GenerationRequest(prompt="hello", stop=["END", "STOP"])))
    assert engine.generate_calls[0]["sampling"].kwargs["stop"] == ["END", "STOP"]

    engine.responses.append([_output("hi", [1])])

    async def collect():
        return [
            event async for event in runtime.stream(GenerationRequest(prompt="hello", stop="DONE"))
        ]

    asyncio.run(collect())
    assert engine.generate_calls[1]["sampling"].kwargs["stop"] == ["DONE"]

    # absent stop sequences must reach vllm as none, not as an empty list.
    engine.responses.append([_output("hi", [1])])
    asyncio.run(runtime.generate(GenerationRequest(prompt="hello")))
    assert engine.generate_calls[2]["sampling"].kwargs["stop"] is None
    asyncio.run(runtime.close())


def test_stream_preserves_repeated_prefix_delta_chunks() -> None:
    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.responses.append(
        [
            _output("a", [1], prompt_tokens=2, finish_reason=None),
            _output("ab", [1, 2], prompt_tokens=2),
        ]
    )

    async def collect():
        return [event async for event in runtime.stream(GenerationRequest(prompt="hello"))]

    events = asyncio.run(collect())
    assert [event.text for event in events if isinstance(event, StreamDelta)] == ["a", "ab"]
    final = events[-1]
    assert isinstance(final, StreamFinished)
    assert final.text == "aab"
    assert final.completion_tokens == 3
    asyncio.run(runtime.close())


def test_stream_engine_error_occurs_before_ready_event() -> None:
    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.responses.append(ValueError("context length exceeded"))

    async def first_event():
        return await anext(runtime.stream(GenerationRequest(prompt="hello")))

    with pytest.raises(ValueError, match="context length"):
        asyncio.run(first_event())
    asyncio.run(runtime.close())


def test_multimodal_generation_closes_images(monkeypatch) -> None:
    closed: list[bool] = []

    class _Image:
        def close(self) -> None:
            closed.append(True)

    image = _Image()
    monkeypatch.setattr(
        "flash.serve.runtime.prompt.prepare_multimodal_request",
        lambda *_args, **_kwargs: (
            [{"role": "user", "content": [{"type": "image"}]}],
            [image],
        ),
    )
    runtime = VllmLoraRuntime(EngineConfig(model="model", image_limit=1))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.responses.append([_output("ok", [1])])
    request = GenerationRequest(
        messages=[
            {
                "role": "user",
                "content": [{"type": "image", "image": "data:image/png;base64,ignored"}],
            }
        ],
        thinking=False,
    )

    result = asyncio.run(runtime.generate(request))
    assert result.text == "ok"
    assert engine.seen_images == [image]
    assert closed == [True]
    processor = runtime._processor
    assert processor.template_calls[0][1]["enable_thinking"] is False
    asyncio.run(runtime.close())


def test_engine_death_callback_receives_runtime_health_once() -> None:
    reports = []
    runtime = VllmLoraRuntime(
        EngineConfig(model="model", liveness_interval_seconds=60),
        on_engine_death=reports.append,
    )
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.errored = True

    with pytest.raises(EngineDeadError):
        asyncio.run(runtime.generate(GenerationRequest(prompt="hello")))
    with pytest.raises(EngineDeadError):
        asyncio.run(runtime.generate(GenerationRequest(prompt="hello")))

    assert len(reports) == 1
    assert reports[0].engine_dead is True
    assert reports[0].ok is False
    assert reports[0].runtime_id == runtime.runtime_id
    asyncio.run(runtime.close())


def test_engine_death_callback_retries_after_callback_failure() -> None:
    attempts = 0
    reports = []

    def notify(health) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("host unavailable")
        reports.append(health)

    runtime = VllmLoraRuntime(
        EngineConfig(model="model", liveness_interval_seconds=60),
        on_engine_death=notify,
    )
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.errored = True

    with pytest.raises(EngineDeadError):
        asyncio.run(runtime.generate(GenerationRequest(prompt="hello")))
    with pytest.raises(EngineDeadError):
        asyncio.run(runtime.generate(GenerationRequest(prompt="hello")))

    assert attempts == 2
    assert len(reports) == 1
    assert reports[0].runtime_id == runtime.runtime_id
    asyncio.run(runtime.close())
