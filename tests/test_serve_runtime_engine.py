"""engine construction, generation, streaming, accounting, images, and death handling."""

from __future__ import annotations

import asyncio
import inspect
import sys
import types
from collections.abc import Iterable
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
    PromptError,
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


class _Logprob:
    def __init__(self, value: float, token: str) -> None:
        self.logprob = value
        self.decoded_token = token


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
        self.abort_calls: list[tuple[str | Iterable[str], bool]] = []
        self.abort_outcomes: list[Any] = []
        self.active: dict[str, Any] = {}
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

    async def abort(
        self,
        request_id: str | Iterable[str],
        internal: bool = False,
    ) -> None:
        self.abort_calls.append((request_id, internal))
        outcome = self.abort_outcomes.pop(0) if self.abort_outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        if inspect.isawaitable(outcome):
            await outcome
        active = self.active.get(request_id) if isinstance(request_id, str) else None
        release = getattr(active, "set", None)
        if release is not None and outcome is not False:
            release()
        return outcome

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
        if hasattr(scenario, "wait"):
            self.active[request_id] = scenario
            try:
                await scenario.wait()
            finally:
                self.active.pop(request_id, None)
            return
        if hasattr(scenario, "__aiter__"):
            async for output in scenario:
                yield output
            return
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
    index: int = 0,
    logprobs: Any = None,
):
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                index=index,
                text=text,
                token_ids=token_ids,
                finish_reason=finish_reason,
                logprobs=logprobs,
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

    the packaged app deployed by `flash serve deploy` is the other consumer of this runtime. it
    starts against an empty volume and fetches the base model with HF_TOKEN, so forcing local-only
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


def test_packaged_engine_threads_top_logprobs_through_buffered_and_streaming() -> None:
    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    candidates = {2: _Logprob(-1.0, "b"), 1: _Logprob(-0.1, "a")}
    engine.responses.append([_output("a", [1], logprobs=[candidates])])

    result = asyncio.run(
        runtime.generate(GenerationRequest(prompt="hello", logprobs=True, top_logprobs=1))
    )
    assert result.choices[0].logprobs is not None
    assert [record["token"] for record in result.choices[0].logprobs[0]["top_logprobs"]] == ["b"]

    engine.responses.append([_output("a", [1], logprobs=[candidates])])

    async def collect():
        return [
            event
            async for event in runtime.stream(
                GenerationRequest(prompt="hello", logprobs=True, top_logprobs=0)
            )
        ]

    events = asyncio.run(collect())
    delta = next(event for event in events if isinstance(event, StreamDelta))
    assert delta.logprobs is not None
    assert delta.logprobs[0]["top_logprobs"] == []
    asyncio.run(runtime.close())


def test_stream_trusts_pinned_delta_output_and_counts_chunks() -> None:
    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    # no delta reports a cache count, so the finished event must report cache state as unreported.
    engine.responses.append(
        [
            _output("he", [1], prompt_tokens=4, cached_tokens=None, finish_reason=None),
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
            SimpleNamespace(
                index=0,
                text=text,
                token_ids=token_ids,
                finish_reason=finish_reason,
            ),
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
    """a rejected request fails before `StreamReady`, as a caller error carrying vllm's message.

    `PromptError` rather than a bare `ValueError`: vllm rejects an over-length prompt with a plain
    `ValueError`, and an unclassified exception is answered 503 by the http layer -- telling the
    client to retry a request that cannot ever succeed.
    """

    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.responses.append(ValueError("context length exceeded"))

    async def first_event():
        return await anext(runtime.stream(GenerationRequest(prompt="hello")))

    with pytest.raises(PromptError, match="context length"):
        asyncio.run(first_event())
    asyncio.run(runtime.close())


def test_generate_rejection_is_a_prompt_error_not_an_engine_failure() -> None:
    """the non-streaming path classifies a rejected request the same way the stream path does."""

    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.responses.append(ValueError("This model's maximum context length is 32768 tokens"))

    with pytest.raises(PromptError, match="maximum context length"):
        asyncio.run(runtime.generate(GenerationRequest(prompt="hello")))
    asyncio.run(runtime.close())


def test_engine_failure_is_not_reclassified_as_a_prompt_error() -> None:
    """only `ValueError` is rewritten: an engine defect must not be blamed on the caller."""

    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.responses.append(TypeError("engine-side defect"))

    with pytest.raises(TypeError, match="engine-side defect"):
        asyncio.run(runtime.generate(GenerationRequest(prompt="hello")))
    assert isinstance(engine.responses, list)
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


@pytest.mark.parametrize("streaming", [False, True])
def test_output_stream_close_failure_still_closes_images_once(monkeypatch, streaming: bool) -> None:
    images: list[Any] = []

    class _Image:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    def prepare(*_args, **_kwargs):
        image = _Image()
        images.append(image)
        return ([{"role": "user", "content": [{"type": "image"}]}], [image])

    class FailingCloseStream:
        def __init__(self, output: Any) -> None:
            self.output = output
            self.sent = False
            self.close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return self.output

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError("output close failure")

    monkeypatch.setattr(
        "flash.serve.runtime.prompt.prepare_multimodal_request",
        prepare,
    )

    async def scenario() -> tuple[int, int]:
        runtime = VllmLoraRuntime(EngineConfig(model="model", image_limit=1))
        await runtime.start()
        output_stream = FailingCloseStream(_output("ok", [1]))
        monkeypatch.setattr(runtime, "_generate_stream", lambda *_args: output_stream)
        request = GenerationRequest(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "ignored"}],
                }
            ],
        )
        if streaming:
            with pytest.raises(RuntimeError, match="output close failure"):
                _ = [event async for event in runtime.stream(request)]
        else:
            with pytest.raises(RuntimeError, match="output close failure"):
                await runtime.generate(request)
        assert runtime.health().owned_request_ids == ()
        await runtime.close()
        return output_stream.close_calls, images[0].close_calls

    assert asyncio.run(scenario()) == (1, 1)


def test_generate_owned_id_rejects_stream_before_image_or_iterator_preparation(
    monkeypatch,
) -> None:
    images: list[Any] = []

    class _Image:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    def prepare(*_args, **_kwargs):
        image = _Image()
        images.append(image)
        return ([{"role": "user", "content": [{"type": "image"}]}], [image])

    monkeypatch.setattr(
        "flash.serve.runtime.prompt.prepare_multimodal_request",
        prepare,
    )

    async def scenario() -> tuple[int, int, list[int]]:
        runtime = VllmLoraRuntime(EngineConfig(model="model", image_limit=1))
        await runtime.start()
        engine = _Engine.latest
        assert engine is not None
        engine.responses.append(asyncio.Event())
        request = GenerationRequest(
            request_id="generate-stream-collision",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "ignored"}],
                }
            ],
        )
        active = asyncio.create_task(runtime.generate(request))
        await _wait_for_generate_call(engine)
        rejected = runtime.stream(request)
        with pytest.raises(RuntimeNotReadyError, match="already owned"):
            await anext(rejected)
        await rejected.aclose()
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        await runtime.close()
        return len(images), len(engine.generate_calls), [image.close_calls for image in images]

    assert asyncio.run(scenario()) == (1, 1, [1])


def test_stream_owned_id_rejects_generate_and_closes_image_and_iterator(monkeypatch) -> None:
    images: list[Any] = []

    class _Image:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    def prepare(*_args, **_kwargs):
        image = _Image()
        images.append(image)
        return ([{"role": "user", "content": [{"type": "image"}]}], [image])

    monkeypatch.setattr(
        "flash.serve.runtime.prompt.prepare_multimodal_request",
        prepare,
    )

    async def scenario() -> tuple[int, int, list[int], bool]:
        runtime = VllmLoraRuntime(EngineConfig(model="model", image_limit=1))
        await runtime.start()
        engine = _Engine.latest
        assert engine is not None
        release = asyncio.Event()
        iterator_waiting = asyncio.Event()
        iterator_closed = asyncio.Event()

        async def outputs():
            try:
                yield _output("partial", [1], finish_reason=None)
                engine.active["stream-generate-collision"] = release
                iterator_waiting.set()
                await release.wait()
            finally:
                engine.active.pop("stream-generate-collision", None)
                iterator_closed.set()

        engine.responses.append(outputs())
        request = GenerationRequest(
            request_id="stream-generate-collision",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "ignored"}],
                }
            ],
        )
        stream = runtime.stream(request)
        assert isinstance(await anext(stream), StreamReady)
        assert isinstance(await anext(stream), StreamDelta)
        next_event = asyncio.create_task(anext(stream))
        await iterator_waiting.wait()
        with pytest.raises(RuntimeNotReadyError, match="already owned"):
            await runtime.generate(request)
        next_event.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_event
        await stream.aclose()
        await runtime.close()
        return (
            len(images),
            len(engine.generate_calls),
            [image.close_calls for image in images],
            iterator_closed.is_set(),
        )

    assert asyncio.run(scenario()) == (1, 1, [1], True)


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


def _set_short_cancellation_grace(monkeypatch) -> None:
    monkeypatch.setattr("flash.serve.runtime.cancellation._ABORT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("flash.serve.runtime.cancellation._DRAIN_TIMEOUT_SECONDS", 0.01)


async def _wait_for_generate_call(engine: _Engine) -> None:
    for _ in range(100):
        if engine.generate_calls:
            return
        await asyncio.sleep(0)
    raise AssertionError("generation did not reach the engine")


def test_request_owns_exact_engine_id_and_normal_completion_does_not_abort() -> None:
    runtime = VllmLoraRuntime(EngineConfig(model="model"))
    asyncio.run(runtime.start())
    engine = _Engine.latest
    assert engine is not None
    engine.responses.append([_output("ok", [1])])
    request = GenerationRequest(prompt="hello", request_id="request-owned-id")

    result = asyncio.run(runtime.generate(request))

    assert result.request_id == request.request_id
    assert engine.generate_calls[0]["request_id"] == request.request_id
    assert engine.abort_calls == []
    assert runtime.health().owned_request_ids == ()
    asyncio.run(runtime.close())


def test_cancel_before_first_token_aborts_exact_request_once() -> None:
    async def scenario() -> tuple[list[tuple[str | Iterable[str], bool]], tuple[str, ...]]:
        runtime = VllmLoraRuntime(EngineConfig(model="model"))
        await runtime.start()
        engine = _Engine.latest
        assert engine is not None
        engine.responses.append(asyncio.Event())
        request = GenerationRequest(prompt="hello", request_id="request-before-first")
        task = asyncio.create_task(runtime.generate(request))
        await _wait_for_generate_call(engine)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        owned = runtime.health().owned_request_ids
        await runtime.close()
        return engine.abort_calls, owned

    abort_calls, owned = asyncio.run(scenario())
    assert abort_calls == [("request-before-first", False)]
    assert owned == ()


def test_midstream_disconnect_aborts_exact_request_once() -> None:
    async def scenario() -> list[tuple[str | Iterable[str], bool]]:
        runtime = VllmLoraRuntime(EngineConfig(model="model"))
        await runtime.start()
        engine = _Engine.latest
        assert engine is not None
        release = asyncio.Event()

        async def outputs():
            yield _output("partial", [1], finish_reason=None)
            engine.active["request-midstream"] = release
            try:
                await release.wait()
            finally:
                engine.active.pop("request-midstream", None)

        engine.responses.append(outputs())
        stream = runtime.stream(GenerationRequest(prompt="hello", request_id="request-midstream"))
        assert isinstance(await anext(stream), StreamReady)
        assert isinstance(await anext(stream), StreamDelta)
        next_event = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        next_event.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_event
        await stream.aclose()
        await runtime.close()
        return engine.abort_calls

    assert asyncio.run(scenario()) == [("request-midstream", False)]


@pytest.mark.parametrize("abort_outcome", [False, RuntimeError("abort failed")])
def test_unconfirmed_abort_marks_runtime_unhealthy_and_rejects_new_work(
    monkeypatch, abort_outcome: Any
) -> None:
    _set_short_cancellation_grace(monkeypatch)

    async def scenario() -> tuple[Any, list[tuple[str | Iterable[str], bool]]]:
        runtime = VllmLoraRuntime(EngineConfig(model="model"))
        await runtime.start()
        engine = _Engine.latest
        assert engine is not None
        engine.responses.append(asyncio.Event())
        engine.abort_outcomes.append(abort_outcome)
        task = asyncio.create_task(
            runtime.generate(GenerationRequest(prompt="hello", request_id="request-unconfirmed"))
        )
        await _wait_for_generate_call(engine)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        health = runtime.health()
        with pytest.raises(RuntimeNotReadyError, match="unhealthy"):
            await runtime.generate(GenerationRequest(prompt="new"))
        await runtime.close()
        return health, engine.abort_calls

    health, abort_calls = asyncio.run(scenario())
    assert health.ok is False
    assert health.unhealthy_reason is not None
    assert health.owned_request_ids == ("request-unconfirmed",)
    assert abort_calls == [("request-unconfirmed", False)]


def test_unconfirmed_sibling_abort_does_not_leak_a_healthy_request(monkeypatch) -> None:
    """a concurrent unconfirmed abort must not skip another request's own cleanup."""

    _set_short_cancellation_grace(monkeypatch)
    images: list[Any] = []

    class _Image:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    def prepare(*_args, **_kwargs):
        image = _Image()
        images.append(image)
        return ([{"role": "user", "content": [{"type": "image"}]}], [image])

    class _GatedStream:
        def __init__(self, output: Any) -> None:
            self.output = output
            self.gate = asyncio.Event()
            self.started = asyncio.Event()
            self.sent = False
            self.close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.started.set()
            await self.gate.wait()
            self.sent = True
            return self.output

        async def aclose(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr("flash.serve.runtime.prompt.prepare_multimodal_request", prepare)

    def _image_request(request_id: str) -> GenerationRequest:
        return GenerationRequest(
            messages=[{"role": "user", "content": [{"type": "image", "image": "ignored"}]}],
            request_id=request_id,
        )

    async def scenario() -> tuple[int, int, tuple[str, ...]]:
        runtime = VllmLoraRuntime(EngineConfig(model="model", image_limit=1))
        await runtime.start()
        engine = _Engine.latest
        assert engine is not None
        engine.abort_outcomes.append(False)
        cancelled_stream = _GatedStream(_output("cancelled", [1]))
        healthy_stream = _GatedStream(_output("healthy", [2]))
        streams = [cancelled_stream, healthy_stream]
        monkeypatch.setattr(runtime, "_generate_stream", lambda *_args: streams.pop(0))

        cancelled_task = asyncio.create_task(runtime.generate(_image_request("sibling-cancelled")))
        await cancelled_stream.started.wait()
        healthy_task = asyncio.create_task(runtime.generate(_image_request("sibling-healthy")))
        await healthy_stream.started.wait()

        cancelled_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_task
        assert runtime.health().unhealthy_reason is not None

        healthy_stream.gate.set()
        result = await healthy_task
        assert result.text == "healthy"
        owned = runtime.health().owned_request_ids
        await runtime.close()
        return healthy_stream.close_calls, images[1].close_calls, owned

    stream_closes, image_closes, owned = asyncio.run(scenario())
    assert stream_closes == 1
    assert image_closes == 1
    assert "sibling-healthy" not in owned


def test_abort_timeout_and_cancellation_resistant_generation_are_bounded(monkeypatch) -> None:
    _set_short_cancellation_grace(monkeypatch)

    async def scenario() -> tuple[Any, list[tuple[str | Iterable[str], bool]]]:
        runtime = VllmLoraRuntime(EngineConfig(model="model"))
        await runtime.start()
        engine = _Engine.latest
        assert engine is not None
        engine.responses.append(asyncio.Event())
        engine.abort_outcomes.append(asyncio.Event().wait())
        task = asyncio.create_task(
            runtime.generate(GenerationRequest(prompt="hello", request_id="request-timeout"))
        )
        await _wait_for_generate_call(engine)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.1)
        health = runtime.health()
        await runtime.close()
        return health, engine.abort_calls

    health, abort_calls = asyncio.run(scenario())
    assert health.ok is False
    assert health.owned_request_ids == ("request-timeout",)
    assert abort_calls == [("request-timeout", False)]


def test_detached_abort_exception_is_consumed(monkeypatch) -> None:
    _set_short_cancellation_grace(monkeypatch)

    async def scenario() -> list[dict[str, Any]]:
        reported: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: reported.append(context))
        runtime = VllmLoraRuntime(EngineConfig(model="model"))
        await runtime.start()
        engine = _Engine.latest
        assert engine is not None
        generation_release = asyncio.Event()
        abort_release = asyncio.Event()
        engine.responses.append(generation_release)

        async def late_abort_failure() -> None:
            await abort_release.wait()
            raise RuntimeError("late abort failure")

        engine.abort_outcomes.append(late_abort_failure())
        task = asyncio.create_task(
            runtime.generate(GenerationRequest(prompt="hello", request_id="request-detached"))
        )
        await _wait_for_generate_call(engine)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        abort_release.set()
        generation_release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await runtime.close()
        return reported

    assert asyncio.run(scenario()) == []


def test_generation_owner_deduplicates_concurrent_abort_calls() -> None:
    from flash.serve.runtime.cancellation import GenerationOwner

    async def scenario() -> list[tuple[str | Iterable[str], bool]]:
        engine = _Engine()
        release = asyncio.Event()
        engine.active["request-deduplicated"] = release
        owner = GenerationOwner(
            engine,
            "request-deduplicated",
            mark_unhealthy=lambda *_args: None,
            detach=lambda *_args: None,
        )
        first, second = await asyncio.gather(owner.cancel(), owner.cancel())
        assert first.abort_confirmed is True
        assert second.abort_confirmed is True
        return engine.abort_calls

    assert asyncio.run(scenario()) == [("request-deduplicated", False)]
