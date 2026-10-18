"""import purity, public types, prompts, images, and structured outputs."""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import sys
import time
from collections import UserDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from PIL import Image

import flash.serve.runtime.tool_calls as runtime_tool_calls
from flash.serve.request.openai import parse_chat_request
from flash.serve.request.tool_calls import (
    FunctionTool,
    detached_template_messages,
    normalize_tools,
)
from flash.serve.request.validation import MAX_MESSAGE_NODES
from flash.serve.runtime import (
    AdapterSpec,
    EngineConfig,
    GenerationRequest,
    PromptError,
    RuntimeConfigurationError,
    RuntimeNotReadyError,
    StructuredOutputsError,
    normalize_structured_outputs,
)
from flash.serve.runtime.multimodal import prepare_multimodal_request
from flash.serve.runtime.prompt import PromptPreparer, resolve_thinking
from flash.serve.runtime.tool_calls import parse_qwen3_coder_output

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}
MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40


def test_runtime_imports_without_heavy_serving_packages() -> None:
    probe = r"""
import builtins
import sys

blocked = ("vllm", "transformers", "torch", "PIL", "modal", "freesolo")
real_import = builtins.__import__


def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == blocked or name.startswith(tuple(item + "." for item in blocked)):
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded
from flash.serve.runtime.types import AdapterSpec, EngineConfig, GenerationRequest
import flash.serve.runtime

assert EngineConfig(model="model")
assert AdapterSpec(adapter_id="a", path="/tmp/a", incarnation="opaque")
assert GenerationRequest(prompt="hello")
for name in blocked:
    assert name not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", probe],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_engine_config_rejects_runtime_owned_engine_args() -> None:
    with pytest.raises(RuntimeConfigurationError, match="runtime-owned"):
        EngineConfig(model="model", engine_args={"model": "other"})
    with pytest.raises(RuntimeConfigurationError, match="max_cpu_loras"):
        EngineConfig(model="model", max_loras=4, max_cpu_loras=2)
    for kwargs_name in ("tokenizer_kwargs", "processor_kwargs"):
        with pytest.raises(RuntimeConfigurationError, match="runtime-owned"):
            EngineConfig(model="model", **{kwargs_name: {"token": "other"}})


def test_engine_config_uses_first_class_exact_revisions() -> None:
    # revisions are keyword-only in practice, so a config built without them pins nothing and
    # the effective names still come from the model overrides.
    unpinned = EngineConfig("model", "served/model", "tokenizer/model")
    assert unpinned.model_revision is None
    assert unpinned.tokenizer_revision is None
    assert unpinned.effective_served_model == "served/model"
    assert unpinned.effective_tokenizer_model == "tokenizer/model"

    pinned = EngineConfig(
        model="model",
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    assert pinned.model_revision == MODEL_REVISION
    assert pinned.tokenizer_revision == TOKENIZER_REVISION


@pytest.mark.parametrize(
    "kwargs",
    [
        {"engine_args": {"revision": MODEL_REVISION}},
        {"engine_args": {"tokenizer_revision": TOKENIZER_REVISION}},
        {"tokenizer_kwargs": {"revision": TOKENIZER_REVISION}},
        {"processor_kwargs": {"revision": TOKENIZER_REVISION}},
    ],
)
def test_engine_config_rejects_revision_keys_in_arbitrary_mappings(kwargs) -> None:
    with pytest.raises(RuntimeConfigurationError, match=r"runtime-owned.*revision"):
        EngineConfig(model="model", **kwargs)


def test_engine_config_recursively_detaches_and_freezes_runtime_mappings() -> None:
    engine_nested = {"mapping": {"value": 1}, "list": [2]}
    tokenizer_nested = {"mapping": {"value": 3}, "tuple": (4,)}
    processor_nested = {"set": {5}, "frozenset": frozenset({6})}
    config = EngineConfig(
        model="model",
        engine_args={"nested": engine_nested},
        tokenizer_kwargs={"nested": tokenizer_nested},
        processor_kwargs={"nested": processor_nested},
    )

    engine_nested["mapping"]["value"] = 10
    engine_nested["list"].append(20)
    tokenizer_nested["mapping"]["value"] = 30
    processor_nested["set"].add(50)
    assert config.engine_args["nested"]["mapping"]["value"] == 1
    assert tuple(config.engine_args["nested"]["list"]) == (2,)
    assert config.tokenizer_kwargs["nested"]["mapping"]["value"] == 3
    assert config.processor_kwargs["nested"]["set"] == frozenset({5})

    for mapping in (config.engine_args, config.tokenizer_kwargs, config.processor_kwargs):
        with pytest.raises(TypeError):
            mapping["new"] = True
        with pytest.raises(TypeError):
            mapping["nested"]["new"] = True
    with pytest.raises(AttributeError):
        config.engine_args["nested"]["list"].append(3)
    with pytest.raises(AttributeError):
        config.processor_kwargs["nested"]["set"].add(7)
    frozen_sets = (
        config.processor_kwargs["nested"]["set"],
        config.processor_kwargs["nested"]["frozenset"],
    )
    for frozen in frozen_sets:
        assert not hasattr(frozen, "__dict__")
        with pytest.raises(AttributeError):
            object.__setattr__(frozen, "writable_state", True)


def test_engine_config_rejects_recursive_and_mutable_runtime_values() -> None:
    recursive: dict[str, object] = {}
    recursive["self"] = recursive
    with pytest.raises(RuntimeConfigurationError, match="recursive containers"):
        EngineConfig(model="model", engine_args=recursive)

    class CopyableMutableLeaf:
        def __deepcopy__(self, _memo):
            return CopyableMutableLeaf()

    mutable_values = (bytearray(b"mutable"), CopyableMutableLeaf())
    for field_name in ("engine_args", "tokenizer_kwargs", "processor_kwargs"):
        for value in mutable_values:
            with pytest.raises(RuntimeConfigurationError, match="immutable scalar"):
                EngineConfig(model="model", **{field_name: {"value": value}})


@pytest.mark.parametrize(
    "revision",
    ["main", "release", "A" * 40, "a" * 39, "a" * 41, " " + "a" * 40, "a" * 40 + " "],
)
def test_engine_config_rejects_mutable_or_noncanonical_revisions(revision: str) -> None:
    for field_name in ("model_revision", "tokenizer_revision"):
        with pytest.raises(RuntimeConfigurationError, match="40-character lowercase hex"):
            EngineConfig(model="model", **{field_name: revision})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_loras": 1.5},
        {"max_lora_rank": "64"},
        {"max_cpu_loras": 16.0},
        {"image_limit": 1.0},
        {"prompt_cache_size": 1.0},
        {"trust_remote_code": "false"},
        {"pin_loras": "true"},
        {"enable_tower_connector_lora": "true"},
        {"mm_processor_cache_gb": -1},
        {"mm_processor_cache_gb": float("nan")},
        {"liveness_interval_seconds": float("nan")},
        {"liveness_interval_seconds": float("inf")},
        {"liveness_interval_seconds": True},
    ],
)
def test_engine_config_rejects_inexact_public_scalar_types(kwargs) -> None:
    with pytest.raises(RuntimeConfigurationError):
        EngineConfig(model="model", **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens", 1.0),
        ("max_tokens", "1"),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", "0"),
        ("top_p", float("nan")),
        ("top_p", True),
        ("thinking", "false"),
    ],
)
def test_generation_request_rejects_inexact_public_scalar_types(field, value) -> None:
    with pytest.raises(RuntimeConfigurationError):
        GenerationRequest(prompt="x", **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [("thinking", "false"), ("thinking", 0), ("pin", "true"), ("pin", 1)],
)
def test_adapter_spec_rejects_inexact_public_boolean_types(field, value) -> None:
    with pytest.raises(RuntimeConfigurationError):
        AdapterSpec(
            adapter_id="adapter",
            path="/tmp/adapter",
            incarnation="one",
            **{field: value},
        )


def test_generation_request_requires_exactly_one_prompt_form() -> None:
    with pytest.raises(RuntimeConfigurationError, match="exactly one"):
        GenerationRequest()
    with pytest.raises(RuntimeConfigurationError, match="exactly one"):
        GenerationRequest(prompt="x", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(RuntimeConfigurationError, match="requires"):
        GenerationRequest(prompt="x", expected_incarnation="old")
    with pytest.raises(RuntimeConfigurationError, match="requires expected_incarnation"):
        GenerationRequest(adapter_id="adapter", prompt="x")


def test_stop_sequences_normalize_to_a_tuple() -> None:
    # none and an empty sequence both mean "no stop sequences".
    assert GenerationRequest(prompt="x").stop == ()
    assert GenerationRequest(prompt="x", stop=None).stop == ()
    assert GenerationRequest(prompt="x", stop=[]).stop == ()
    # a bare string is one sequence, not one entry per character.
    assert GenerationRequest(prompt="x", stop="END").stop == ("END",)
    assert GenerationRequest(prompt="x", stop=["a", "b"]).stop == ("a", "b")
    assert GenerationRequest(prompt="x", stop=("a", "b")).stop == ("a", "b")


@pytest.mark.parametrize("value", ["\n\n", " END", "   ", "\t"])
def test_stop_sequences_preserve_whitespace(value) -> None:
    """Stop sequences are generation delimiters, so their content must survive verbatim.

    Trimming them -- correct for identifiers like `model` or `adapter_id` -- either rejects a
    valid stop ("\n\n" strips to empty) or silently changes it (" END" becomes "END", which
    stops on a different string than the caller asked for). The hosted serving validator
    preserves them, and the two paths must agree.
    """
    assert GenerationRequest(prompt="x", stop=value).stop == (value,)
    assert GenerationRequest(prompt="x", stop=["ok", value]).stop == ("ok", value)


@pytest.mark.parametrize("value", ["", ["ok", ""], [1], ["ok", None], 7, {}])
def test_stop_sequences_reject_unusable_values(value) -> None:
    with pytest.raises(RuntimeConfigurationError):
        GenerationRequest(prompt="x", stop=value)


def test_structured_outputs_normalization_and_explicit_off() -> None:
    assert normalize_structured_outputs(SCHEMA) == {"json": SCHEMA}
    assert normalize_structured_outputs("json_object") == {"json_object": True}
    assert normalize_structured_outputs({"choices": ["a", "b"]}) == {"choice": ["a", "b"]}
    assert normalize_structured_outputs(False) == {}
    assert GenerationRequest(prompt="x").structured_outputs is None
    assert GenerationRequest(prompt="x", structured_outputs=False).structured_outputs == {}
    with pytest.raises(StructuredOutputsError, match="exactly one"):
        normalize_structured_outputs({"json": SCHEMA, "regex": "x"})


def test_adapter_thinking_and_structured_defaults_are_normalized() -> None:
    spec = AdapterSpec(
        adapter_id="adapter",
        path="/tmp/adapter",
        incarnation="opaque",
        thinking=False,
        structured_outputs=SCHEMA,
    )
    request = GenerationRequest(
        messages=[{"role": "user", "content": "hello"}],
        chat_template_kwargs={"enable_thinking": True},
    )
    assert spec.structured_outputs == {"json": SCHEMA}
    assert resolve_thinking(request, spec) is False
    assert resolve_thinking(request, None) is False
    assert (
        resolve_thinking(
            GenerationRequest(
                messages=[{"role": "user", "content": "hello"}],
                thinking=True,
                chat_template_kwargs={"enable_thinking": False},
            ),
            None,
        )
        is True
    )


class _Tokenizer:
    def __init__(self) -> None:
        self.template_calls: list[dict] = []
        self.rendered: list[Any] = []
        self.encode_calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        self.rendered.append(messages)
        return [1, 2, 3]

    def encode(self, _prompt, **_kwargs):
        self.encode_calls += 1
        return [4, 5]


@pytest.mark.parametrize(
    ("authored", "canonical"),
    [
        (
            [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        ),
        (
            [{"role": "developer", "content": "be terse"}],
            [{"role": "system", "content": "be terse"}],
        ),
    ],
)
def test_accepted_aliases_are_rendered_in_their_canonical_form(authored, canonical) -> None:
    """the boundary accepts these spellings, so the template must be handed what it understands.

    `parse_chat_request` validates a normalized copy and used to forward the original, and only
    image-bearing requests were normalized before rendering. A real Qwen3.5 template raises
    `Unexpected message role` on `developer` -- answered 503, "retry later", for a request that can
    never succeed -- and templates that read only canonical `text` blocks silently drop an
    `input_text` block, generating from the wrong prompt. Asserting against the canonical
    rendering, not just "it did not raise", is what pins the alias to the same prompt.
    """

    tokenizer = _Tokenizer()
    preparer = PromptPreparer(EngineConfig(model="model", prompt_cache_size=0), tokenizer, None)

    import asyncio

    asyncio.run(preparer.prepare(GenerationRequest(messages=authored), False))
    asyncio.run(preparer.prepare(GenerationRequest(messages=canonical), False))

    assert tokenizer.rendered[0] == canonical
    assert tokenizer.rendered[0] == tokenizer.rendered[1]


@pytest.mark.parametrize("rejection", [TypeError("str + dict"), ValueError("bad shape")])
def test_a_template_rejecting_the_request_is_a_client_error_not_an_unavailable_engine(
    rejection: Exception,
) -> None:
    """the template, not the engine, is what refused -- and only the request chose that shape.

    Strict history validation accepts this complete OpenAI lifecycle, but a tokenizer may still
    reject a template-specific shape. Preparation runs before
    `_rejection_as_prompt_error`, so an unclassified failure here answered 503 and invited a retry
    that must fail identically, while the engine was healthy the whole time.
    """

    class _RejectingTokenizer(_Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise rejection

    preparer = PromptPreparer(
        EngineConfig(model="model", prompt_cache_size=0), _RejectingTokenizer(), None
    )

    import asyncio

    with pytest.raises(PromptError) as raised:
        asyncio.run(
            preparer.prepare(
                GenerationRequest(
                    messages=[
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": "{}"},
                                },
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                    ]
                ),
                False,
            )
        )

    # `PromptError` is what the http layer maps to 400; a bare ValueError would not survive the
    # trip, and a TypeError would not be classified at all.
    assert isinstance(raised.value, ValueError)
    assert "chat template rejected" in str(raised.value)


def test_an_engine_fault_surfacing_through_the_template_is_not_blamed_on_the_caller() -> None:
    # the mapping must name the caller only when the caller is the cause. a runtime fault that
    # happens to surface inside the template keeps its own meaning, or a dead engine would be
    # reported as a bad request and never trigger the liveness path.
    class _FaultingTokenizer(_Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise RuntimeNotReadyError("engine died")

    preparer = PromptPreparer(
        EngineConfig(model="model", prompt_cache_size=0), _FaultingTokenizer(), None
    )

    import asyncio

    with pytest.raises(RuntimeNotReadyError):
        asyncio.run(
            preparer.prepare(GenerationRequest(messages=[{"role": "user", "content": "hi"}]), False)
        )


def test_an_alias_shares_the_prompt_cache_entry_with_its_canonical_spelling() -> None:
    tokenizer = _Tokenizer()
    preparer = PromptPreparer(EngineConfig(model="model", prompt_cache_size=4), tokenizer, None)
    alias = [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
    canonical = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    import asyncio

    asyncio.run(preparer.prepare(GenerationRequest(messages=alias), False))
    asyncio.run(preparer.prepare(GenerationRequest(messages=canonical), False))

    # keying on the authored messages would tokenize the same rendered prompt twice.
    assert len(tokenizer.template_calls) == 1
    assert preparer.cache_entries == 1


def test_text_prompt_cache_is_bounded_and_keys_thinking() -> None:
    tokenizer = _Tokenizer()
    preparer = PromptPreparer(
        EngineConfig(model="model", prompt_cache_size=1),
        tokenizer,
        None,
    )
    first = GenerationRequest(messages=[{"role": "user", "content": "hello"}])
    second = GenerationRequest(messages=[{"role": "user", "content": "other"}])

    import asyncio

    one = asyncio.run(preparer.prepare(first, False))
    again = asyncio.run(preparer.prepare(first, False))
    asyncio.run(preparer.prepare(first, True))
    asyncio.run(preparer.prepare(second, False))

    assert one.value == again.value == {"prompt_token_ids": [1, 2, 3]}
    assert len(tokenizer.template_calls) == 3
    assert tokenizer.template_calls[0]["enable_thinking"] is False
    assert tokenizer.template_calls[1]["enable_thinking"] is True
    assert preparer.cache_entries == 1


def _data_uri() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_multimodal_preparation_preserves_order_and_closes_cleanly() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": _data_uri()}},
                {"type": "input_text", "text": "after"},
            ],
        }
    ]
    template_messages, images = prepare_multimodal_request(messages, image_limit=4)
    try:
        assert template_messages[0]["content"] == [
            {"type": "text", "text": "before"},
            {"type": "image"},
            {"type": "text", "text": "after"},
        ]
        assert images[0].mode == "RGB"
        assert images[0].getpixel((0, 0)) == (10, 20, 30)
    finally:
        for image in images:
            image.close()


def test_multimodal_prompt_failure_closes_decoded_images(monkeypatch) -> None:
    closed: list[bool] = []

    class _Image:
        def close(self) -> None:
            closed.append(True)

    class _Processor:
        tokenizer = SimpleNamespace()

        def apply_chat_template(self, *_args, **_kwargs):
            raise RuntimeError("render failed")

    monkeypatch.setattr(
        "flash.serve.runtime.prompt.prepare_multimodal_request",
        lambda *_args, **_kwargs: ([{"role": "user", "content": [{"type": "image"}]}], [_Image()]),
    )
    preparer = PromptPreparer(
        EngineConfig(model="model", image_limit=1),
        SimpleNamespace(),
        _Processor(),
    )
    request = GenerationRequest(
        messages=[
            {
                "role": "user",
                "content": [{"type": "image", "image": _data_uri()}],
            }
        ]
    )

    import asyncio

    # this test is about releasing decoded images, and the raise is only the trigger. the render
    # failure is now classified as a `PromptError` (a `RuntimeError` subclass) that quotes the
    # original text, so match on the underlying cause rather than the wrapper's type alone.
    with pytest.raises(RuntimeError, match="render failed"):
        asyncio.run(preparer.prepare(request, False))
    assert closed == [True]


def _unicode_property_tools():
    return normalize_tools(
        [
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
    )


def _runtime_tool_history(call_id: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "weather"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "ok"},
    ]


@pytest.mark.parametrize("field", ["call", "result"])
def test_packaged_tool_history_rejects_surrogate_call_ids_before_cache(field: str) -> None:
    messages = _runtime_tool_history("call_1")
    if field == "call":
        messages[1]["tool_calls"][0]["id"] = "call_\ud800"
    else:
        messages[2]["tool_call_id"] = "call_\ud800"

    with pytest.raises(RuntimeConfigurationError, match="cannot contain an unpaired surrogate"):
        GenerationRequest(messages=messages)


class _BrokenMessage(Mapping):
    def __len__(self) -> int:
        return 1

    def __iter__(self):
        raise TypeError("broken mapping")

    def __getitem__(self, key):
        raise KeyError(key)


class _BrokenMetadata(Mapping):
    def __len__(self) -> int:
        return 1

    def __iter__(self):
        return iter(("value",))

    def __getitem__(self, key):
        raise TypeError("broken value access")


class _CountedSequence(Sequence):
    def __init__(self, values) -> None:
        self.values = values
        self.yielded = 0

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __iter__(self):
        for value in self.values:
            self.yielded += 1
            yield value


class _FailingSequence(_CountedSequence):
    def __iter__(self):
        raise TypeError("broken sequence")
        yield


class _WideList(list):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.yielded = 0

    def __iter__(self):
        for value in range(self.width):
            self.yielded += 1
            yield value


class _WideMessage(Mapping):
    def __init__(self, width: int) -> None:
        self.width = width
        self.yielded = 0

    def __len__(self) -> int:
        return self.width + 2

    def __iter__(self):
        for key in ("role", "content"):
            self.yielded += 1
            yield key
        for index in range(self.width):
            self.yielded += 1
            yield str(index)

    def __getitem__(self, key):
        return "user" if key == "role" else "hello" if key == "content" else key


class _WideMetadata(Mapping):
    def __init__(self, width: int) -> None:
        self.width = width
        self.yielded = 0

    def __len__(self) -> int:
        return self.width

    def __iter__(self):
        for key in range(self.width):
            self.yielded += 1
            yield str(key)

    def __getitem__(self, key):
        return key


def test_generation_request_wide_metadata_stops_at_the_complexity_bound() -> None:
    metadata = _WideMetadata(100_000)

    with pytest.raises(RuntimeConfigurationError, match="messages exceed the supported complexity"):
        GenerationRequest(messages=[{"role": "user", "content": "hello", "metadata": metadata}])

    assert metadata.yielded <= MAX_MESSAGE_NODES


def test_generation_request_wide_message_stops_at_the_complexity_bound() -> None:
    message = _WideMessage(100_000)

    with pytest.raises(RuntimeConfigurationError, match="messages exceed the supported complexity"):
        GenerationRequest(messages=[message])

    assert message.yielded <= MAX_MESSAGE_NODES


def test_generation_request_wide_message_sequence_stops_at_the_bound() -> None:
    messages = _CountedSequence([{"role": "user", "content": "hello"}] * 100_000)

    with pytest.raises(RuntimeConfigurationError, match="messages exceed the supported complexity"):
        GenerationRequest(messages=messages)

    assert messages.yielded <= MAX_MESSAGE_NODES


def test_generation_request_wide_content_stops_at_the_complexity_bound() -> None:
    content = _WideList(100_000)

    with pytest.raises(RuntimeConfigurationError, match="messages exceed the supported complexity"):
        GenerationRequest(messages=[{"role": "user", "content": content}])

    assert content.yielded <= MAX_MESSAGE_NODES


def test_generation_request_accepts_and_detaches_mapping_messages() -> None:
    authored = {"role": "user", "content": "hello", "metadata": {"value": 1}}
    for message in (UserDict(authored), MappingProxyType(authored)):
        request = GenerationRequest(messages=[message])
        authored["metadata"]["value"] = 2
        assert type(request.messages[0]) is dict
        assert request.messages[0]["metadata"] == {"value": 1}
        authored["metadata"]["value"] = 1


def test_generation_request_translates_failing_message_sequence() -> None:
    with pytest.raises(RuntimeConfigurationError, match="messages contain an unsupported value"):
        GenerationRequest(messages=_FailingSequence([]))


def test_generation_request_translates_mapping_copy_failures() -> None:
    with pytest.raises(RuntimeConfigurationError, match="messages contain an unsupported value"):
        GenerationRequest(messages=[_BrokenMessage()])
    with pytest.raises(RuntimeConfigurationError, match="messages contain an unsupported value"):
        GenerationRequest(
            messages=[{"role": "user", "content": "hello", "metadata": _BrokenMetadata()}]
        )


def test_generation_request_recursively_detaches_caller_messages() -> None:
    shared = {"value": 1}
    metadata = {"left": shared, "right": shared}
    content = [{"type": "text", "text": "hello"}]
    messages = [{"role": "user", "content": content, "metadata": metadata}]

    request = GenerationRequest(messages=messages)
    shared["value"] = 2
    content[0]["text"] = "changed"

    assert isinstance(request.messages, tuple)
    assert isinstance(request.messages[0], dict)
    assert request.messages[0]["metadata"] == {
        "left": {"value": 1},
        "right": {"value": 1},
    }
    assert request.messages[0]["content"] == [{"type": "text", "text": "hello"}]


def test_generation_request_rejects_recursive_and_over_complex_metadata() -> None:
    recursive: dict[str, object] = {}
    recursive["self"] = recursive
    with pytest.raises(RuntimeConfigurationError, match="recursive containers"):
        GenerationRequest(messages=[{"role": "user", "content": "hello", "metadata": recursive}])

    with pytest.raises(RuntimeConfigurationError, match="messages exceed the supported complexity"):
        GenerationRequest(
            messages=[{"role": "user", "content": "hello", "metadata": {"values": [0] * 4096}}]
        )


def test_generation_request_accepts_and_detaches_normal_tool_history() -> None:
    messages = _runtime_tool_history("call_1")

    request = GenerationRequest(messages=messages)
    messages[1]["tool_calls"][0]["function"]["name"] = "changed"

    assert request.messages[1]["tool_calls"][0]["function"]["name"] == "weather"
    assert request.messages[2]["tool_call_id"] == "call_1"


@pytest.mark.parametrize(
    "content",
    ["bad\ud800", [{"type": "text", "text": "bad\ud800"}]],
    ids=["string", "text-block"],
)
def test_packaged_tool_result_surrogates_are_rejected_before_cache(content) -> None:
    messages = _runtime_tool_history("call_1")
    messages[2]["content"] = content

    with pytest.raises(RuntimeConfigurationError, match="tool result content cannot contain"):
        GenerationRequest(messages=messages)


def test_packaged_prompt_cache_accepts_non_bmp_tool_result_text() -> None:
    messages = _runtime_tool_history("call_1")
    messages[2]["content"] = [{"type": "text", "text": "sunny ☀"}]
    request = GenerationRequest(messages=messages)
    preparer = PromptPreparer(EngineConfig(model="model", prompt_cache_size=1), _Tokenizer(), None)

    assert preparer._cache_key(request, request.messages, False) is not None


def test_packaged_prompt_cache_key_accepts_non_bmp_tool_call_ids() -> None:
    messages = _runtime_tool_history("call_🌦")
    request = GenerationRequest(messages=messages)
    preparer = PromptPreparer(EngineConfig(model="model", prompt_cache_size=1), _Tokenizer(), None)

    key = preparer._cache_key(request, request.messages, False)

    assert key is not None
    assert key == preparer._cache_key(request, request.messages, False)
    assert messages == _runtime_tool_history("call_🌦")


def test_packaged_prompt_cache_key_utf8_encodes_accepted_tool_declarations() -> None:
    request = GenerationRequest(
        messages=[{"role": "user", "content": "weather"}],
        tools=_unicode_property_tools(),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    preparer = PromptPreparer(
        EngineConfig(model="model", prompt_cache_size=1),
        _Tokenizer(),
        None,
    )

    key = preparer._cache_key(request, request.messages, False)

    assert key is not None
    assert key == preparer._cache_key(request, request.messages, False)


def test_packaged_prompt_cache_keys_active_tool_schema_and_inactive_choice() -> None:
    first = _runtime_tools()[0].wire()
    second = _runtime_tools()[0].wire()
    second["function"]["parameters"]["properties"]["country"] = {"type": "string"}
    messages = [{"role": "user", "content": "weather"}]
    active_first = GenerationRequest(
        messages=messages,
        tools=[first],
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    active_second = GenerationRequest(
        messages=messages,
        tools=[second],
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    inactive = GenerationRequest(
        messages=messages,
        tools=[first],
        tool_choice="none",
        parallel_tool_calls=True,
    )
    preparer = PromptPreparer(EngineConfig(model="model", prompt_cache_size=4), _Tokenizer(), None)

    keys = {
        preparer._cache_key(request, request.messages, False)
        for request in (active_first, active_second, inactive)
    }

    assert None not in keys
    assert len(keys) == 3


def _runtime_tools():
    return normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                        },
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )


def _container_tools():
    return normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "record",
                    "parameters": {
                        "type": "object",
                        "properties": {"rows": {"type": "array", "items": {"type": "string"}}},
                        "required": ["rows"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )


def _container_call(argument: str) -> str:
    return (
        "<tool_call>\n<function=record>\n<parameter=rows>\n"
        f"{argument}\n</parameter>\n</function>\n</tool_call>"
    )


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ('["plain"]', ["plain"]),
        # the end token inside a string literal does not close the value, so the parse must run
        # past it to the real one. a scan that hops between quotes has to keep this property.
        ('["</parameter>"]', ["</parameter>"]),
        ('["a", "</parameter>", "b"]', ["a", "</parameter>", "b"]),
        # an escaped quote does not leave the string, so the token after it is still quoted.
        (r'["a\"</parameter>"]', ['a"</parameter>']),
        # the escape is itself escaped, so the next quote does close the string.
        (r'["a\\", "</parameter>"]', ["a\\", "</parameter>"]),
        # many short strings before the token, which is where a per-segment research would
        # degrade even though the verdict stays correct.
        ('["x", "x", "x", "x", "x", "x", "x", "x"]', ["x"] * 8),
    ],
    ids=["plain", "quoted-token", "token-between", "escaped-quote", "escaped-escape", "segments"],
)
def test_a_json_container_argument_ends_at_the_unquoted_token(
    argument: str, expected: object
) -> None:
    """the end token counts only outside a string literal, whatever the scan strategy is."""
    result = parse_qwen3_coder_output(_container_call(argument), _container_tools())

    assert [call.name for call in result.calls] == ["record"]
    assert json.loads(result.calls[0].arguments) == {"rows": expected}


@pytest.mark.parametrize("pairs", range(6))
def test_an_even_backslash_run_leaves_the_quote_closing(pairs: int) -> None:
    """every backslash is escaped, so the quote after them still ends the string.

    the run is measured natively rather than stepped through, so the parity has to stay exact at
    every length where a bounded measurement and a per-character walk could disagree.
    """
    run = "\\\\" * pairs
    argument = '["a' + run + '", "</parameter>"]'

    result = parse_qwen3_coder_output(_container_call(argument), _container_tools())

    assert [call.name for call in result.calls] == ["record"]
    assert json.loads(result.calls[0].arguments) == {"rows": json.loads(argument)}


@pytest.mark.parametrize("pairs", range(6))
def test_an_odd_backslash_run_escapes_the_quote_that_follows(pairs: int) -> None:
    """the last backslash escapes the quote, so the string runs on and swallows the token."""
    # the trailing quote is escaped, so the value only ends at the second `"` and the end token
    # between them is quoted. a parity that miscounted by one would end the value early instead.
    run = "\\\\" * pairs
    argument = '["a' + run + '\\"</parameter>b"]'

    result = parse_qwen3_coder_output(_container_call(argument), _container_tools())

    assert [call.name for call in result.calls] == ["record"]
    assert json.loads(result.calls[0].arguments) == {"rows": json.loads(argument)}


def test_locating_a_container_end_does_not_research_the_token_per_string() -> None:
    """the end token is located once, not once per string segment in the argument.

    researching it per segment is semantically identical and only costs time, so no verdict can
    detect it. an argument of many short strings then pays one scan of the remaining text per
    string, which is the quadratic shape this bound exists to prevent.
    """
    segments = 200
    argument = "[" + ", ".join(f'"{index}"' for index in range(segments)) + "]"
    searches = 0

    class _CountingText(str):
        """the scanned text, counting searches for the end token. `str` cannot be patched."""

        __slots__ = ()

        def find(self, sub: str, *args: int) -> int:
            nonlocal searches
            if sub == runtime_tool_calls._PARAMETER_END:
                searches += 1
            return str.find(self, sub, *args)

    scanned = _CountingText(argument + runtime_tool_calls._PARAMETER_END)
    end = runtime_tool_calls._find_json_container_end(scanned, 0, [10**9])

    assert end == len(argument)
    # a handful of searches for the whole argument, not one per string segment.
    assert searches < segments


def test_an_unterminated_escaped_string_settles_without_backtracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """an unterminated string is the regex failure path and must stay single-pass.

    the ordinary run and the escape are both possessive and start on disjoint characters, so
    there is one way to match at each position. written without those, the engine retries every
    split of the run to prove failure and a thousand characters already run for seconds, which a
    client reaches just by opening a string and never closing it.

    the argument needs a quote after the opening one, or the scan settles on the missing quote
    and never consults the pattern at all, which would leave this asserting nothing.
    """
    pattern = runtime_tool_calls._STRING_BODY_RE
    # the property, not the spelling: either possessive quantifier alone already makes this
    # single-pass, so pinning the exact text would fail a valid equivalent. what must hold is that
    # an unterminated body settles in time that stays linear rather than exploding on the split.
    scaling = []
    for width in (2_000, 4_000, 8_000):
        unterminated = '"' + "a" * width + '\\"' + "b" * width
        start = time.perf_counter()
        assert pattern.match(unterminated, 1) is None
        scaling.append(time.perf_counter() - start)
    # a backtracking spelling is superlinear here and blows past this long before the last width.
    assert max(scaling) < 0.5, scaling

    matches = 0

    class _CountingPattern:
        """the string-body pattern, counting the matches the scan actually asks for."""

        pattern = runtime_tool_calls._STRING_BODY_RE.pattern

        def match(self, text: str, position: int) -> re.Match[str] | None:
            nonlocal matches
            matches += 1
            return pattern.match(text, position)

    monkeypatch.setattr(runtime_tool_calls, "_STRING_BODY_RE", _CountingPattern())

    # the escaped quote keeps the string open, so the body is what has to prove it never closes.
    argument = '["' + ("a" * 500_000) + '\\"' + ("b" * 500_000)
    text = argument + runtime_tool_calls._PARAMETER_END
    end = runtime_tool_calls._find_json_container_end(text, 0, [10**9])

    # the end token sits inside the unterminated string, so it cannot close the value.
    assert end == -1
    # without this the payload never reaches the pattern and the test proves nothing.
    assert matches == 1, matches


@pytest.mark.parametrize(
    "argument",
    ['["unterminated', '["a", "b"', '["a"] trailing'],
    ids=["unterminated-string", "unterminated-array", "no-end-token"],
)
def test_a_json_container_argument_without_an_unquoted_token_is_not_a_call(argument: str) -> None:
    """no end token outside a string means no complete call, so the text survives verbatim."""
    text = _container_call(argument).removesuffix("\n</parameter>\n</function>\n</tool_call>")

    result = parse_qwen3_coder_output(text, _container_tools())

    assert result.calls == ()
    assert result.content == text


@pytest.mark.parametrize(
    "enum_value",
    [("tuple",), {1: "numeric key"}, {1: "collision", "1": "string key"}],
    ids=["tuple", "non-string-key", "coercive-key-collision"],
)
def test_generation_request_rejects_nonexact_tool_enum_json(enum_value: object) -> None:
    declaration = _runtime_tools()[0].wire()
    declaration["function"]["parameters"]["properties"]["city"]["enum"] = [enum_value]

    with pytest.raises(
        RuntimeConfigurationError,
        match=r"exact JSON values|string-keyed JSON objects",
    ):
        GenerationRequest(
            messages=[{"role": "user", "content": "weather"}],
            tools=[declaration],
            tool_choice="auto",
            parallel_tool_calls=True,
        )


def test_generation_request_translates_non_string_schema_keywords() -> None:
    declaration = _runtime_tools()[0].wire()
    declaration["function"]["parameters"][5] = "boom"

    with pytest.raises(RuntimeConfigurationError, match=r"unsupported schema keyword\(s\): 5"):
        GenerationRequest(
            messages=[{"role": "user", "content": "weather"}],
            tools=[declaration],
            tool_choice="auto",
            parallel_tool_calls=True,
        )


def test_packaged_generation_request_rejects_active_tool_whitespace_stop() -> None:
    with pytest.raises(RuntimeConfigurationError, match="whitespace separators"):
        GenerationRequest(
            messages=[{"role": "user", "content": "weather"}],
            tools=_runtime_tools(),
            tool_choice="auto",
            parallel_tool_calls=True,
            stop="\t",
        )


def test_generation_request_revalidates_tools_and_rejects_tool_images() -> None:
    with pytest.raises(RuntimeConfigurationError, match="tools must be a nonempty array"):
        GenerationRequest(
            messages=[{"role": "user", "content": "weather"}],
            tools=(),
            tool_choice="auto",
            parallel_tool_calls=True,
        )
    with pytest.raises(RuntimeConfigurationError, match="additionalProperties must be false"):
        GenerationRequest(
            messages=[{"role": "user", "content": "weather"}],
            tools=(
                FunctionTool(
                    "weather",
                    None,
                    {"type": "object", "properties": {}, "required": []},
                ),
            ),
            tool_choice="auto",
            parallel_tool_calls=True,
        )
    with pytest.raises(RuntimeConfigurationError, match="image messages"):
        GenerationRequest(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "image": _data_uri()}],
                }
            ],
            tools=_runtime_tools(),
            tool_choice="auto",
            parallel_tool_calls=True,
        )
    with pytest.raises(RuntimeConfigurationError, match=r"grammar markers.*tool_choice='auto'"):
        GenerationRequest(
            messages=[{"role": "user", "content": "weather"}],
            tools=_runtime_tools(),
            tool_choice="auto",
            parallel_tool_calls=True,
            stop="</tool_call>",
        )
    request = GenerationRequest(
        messages=[
            {
                "role": "user",
                "content": [{"type": "image", "image": _data_uri()}],
            }
        ],
        tools=_runtime_tools(),
        tool_choice="none",
        parallel_tool_calls=True,
        stop="</tool_call>",
        logprobs=True,
        top_logprobs=1,
        structured_outputs=SCHEMA,
    )
    assert request.stop == ("</tool_call>",)
    assert request.logprobs is True
    assert request.structured_outputs == {"json": SCHEMA}


def test_generation_request_validates_tool_history_without_current_tools() -> None:
    messages = _runtime_tool_history("call_1")
    messages[1]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"city": "</parameter><parameter=other>spoof", "other": "actual"}
    )

    with pytest.raises(RuntimeConfigurationError, match="cannot be replayed exactly"):
        GenerationRequest(messages=messages)


@pytest.mark.parametrize(
    ("left", "right", "accepted"), [(255, 255, True), (255, 256, True), (256, 256, False)]
)
def test_generation_request_enforces_the_replay_declaration_ceiling(
    left: int, right: int, accepted: bool
) -> None:
    # the runtime type is publicly constructible, so it is a validation entry point in its own
    # right. two same-name calls with disjoint integer keys union to 510, 511, or 512 properties,
    # walking right up to the root-property budget a declared schema is normalized to and one past
    # it. the exact 511 row is what pins the boundary itself rather than a range that contains it.
    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": "f",
                "arguments": json.dumps({f"c{index}_p{key}": key for key in range(fields)}),
            },
        }
        for index, fields in enumerate((left, right))
    ]
    messages = [
        {"role": "assistant", "content": None, "tool_calls": calls},
        *({"role": "tool", "tool_call_id": item["id"], "content": "ok"} for item in calls),
    ]

    if not accepted:
        with pytest.raises(RuntimeConfigurationError, match="cannot be replayed exactly"):
            GenerationRequest(messages=messages)
        return

    request = GenerationRequest(messages=messages)

    assert [item["function"]["name"] for item in request.messages[0]["tool_calls"]] == ["f", "f"]


def test_generation_request_ignores_inactive_tools_during_history_replay() -> None:
    declaration = _runtime_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {"new": {"type": "string"}}
    declaration["function"]["parameters"]["required"] = []
    messages = _runtime_tool_history("call_1")
    messages[1]["tool_calls"][0]["function"] = {
        "name": "weather",
        "arguments": '{"old":"x"}',
    }

    without_tools = GenerationRequest(messages=messages)
    inactive = GenerationRequest(
        messages=messages,
        tools=[declaration],
        tool_choice="none",
        parallel_tool_calls=True,
    )

    assert inactive.messages == without_tools.messages


def test_historical_integer_lexeme_limit_detaches_exactly_and_rejects_overflow() -> None:
    literal = "9" * 1024
    arguments = (
        '{"direct":' + literal + ',"nested":{"value":' + literal + '},"values":[' + literal + "]}"
    )
    messages = _runtime_tool_history("call_1")
    messages[1]["tool_calls"][0]["function"]["arguments"] = arguments
    request = GenerationRequest(messages=messages)

    converted = detached_template_messages(request.messages)
    detached = converted[1]["tool_calls"][0]["function"]["arguments"]

    assert detached == {
        "direct": int(literal),
        "nested": {"value": int(literal)},
        "values": [int(literal)],
    }
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == arguments

    for digits in (1025, 5000):
        rejected = _runtime_tool_history("call_1")
        rejected[1]["tool_calls"][0]["function"]["arguments"] = '{"value":' + "9" * digits + "}"
        with pytest.raises(RuntimeConfigurationError) as raised:
            GenerationRequest(messages=rejected)
        assert "1024-digit limit" in str(raised.value)
        assert "Exceeds the limit (4300 digits)" not in str(raised.value)


@pytest.mark.parametrize(
    "arguments",
    [
        '{"direct":1e1024}',
        '{"nested":{"value":1e1024}}',
        '{"values":[1e1024]}',
    ],
    ids=["direct", "nested", "list"],
)
def test_compact_historical_exponent_is_accepted_without_expanding(
    arguments: str,
) -> None:
    """the runtime accepts what the template can render, so a compact exponent survives.

    the significand bound still rejects an oversized literal; only the expanded-integer
    conversion is gone, because the template renders these compactly and expanding them
    would inflate the prompt and eventually hit python's integer-to-string limit.
    """
    messages = _runtime_tool_history("call_1")
    messages[1]["tool_calls"][0]["function"]["arguments"] = arguments

    request = GenerationRequest(messages=messages)

    assert request.messages[1]["tool_calls"][0]["function"]["arguments"] == arguments


def test_detached_history_arguments_are_objects_without_caller_mutation() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        }
    ]
    converted = detached_template_messages(messages)
    assert converted[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"city":"Paris"}'


def test_detached_tool_result_text_parts_flatten_only_for_templates() -> None:
    content = [
        {"type": "input_text", "text": "sun"},
        {"type": "text", "text": "ny"},
    ]
    messages = [{"role": "tool", "tool_call_id": "call_1", "content": content}]

    converted = detached_template_messages(messages)

    assert converted[0]["content"] == "sunny"
    assert messages[0]["content"] == content


def test_detached_history_arguments_are_native_json_values_without_stringified_containers() -> None:
    arguments = (
        '{"direct":1.25,"integral":9007199254740993.0,'
        '"nested":{"value":2.5e1},"values":[6.25e-1,1e2],'
        '"text":"exact","enabled":true,"empty":null}'
    )
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": arguments},
                }
            ],
        }
    ]

    converted = detached_template_messages(messages)
    detached = converted[0]["tool_calls"][0]["function"]["arguments"]

    # a top-level bool or null is pre-rendered because the template spells a scalar with
    # ``string``, which would emit python's ``True`` and ``None``. inside a container
    # ``tojson`` already spells them correctly, so those stay native.
    assert detached == {
        "direct": 1.25,
        "integral": 9007199254740993,
        "nested": {"value": 25},
        "values": [0.625, 100],
        "text": "exact",
        "enabled": "true",
        "empty": "null",
    }
    json.dumps(detached, allow_nan=False)
    assert type(detached["nested"]) is dict
    assert type(detached["values"]) is list
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == arguments


def test_parsed_history_uses_cached_qwen35_template_with_native_numeric_shapes() -> None:
    import asyncio

    from transformers import AutoTokenizer

    integer = "9" * 1024
    expanded = "1" + "0" * 1023
    arguments = (
        '{"direct":1.25,"exponent":1e2,"expanded":1e1023,"integer":'
        + integer
        + ',"nested":{"decimal":2.5,"exponent":2.5e1,"expanded":1e1023,"integer":'
        + integer
        + '},"values":[0.625,6.25e-1,1e1023,'
        + integer
        + "]}"
    )
    messages = _runtime_tool_history("call_1")
    messages[1]["tool_calls"][0]["function"]["arguments"] = arguments
    original = json.loads(json.dumps(messages))
    normalized = parse_chat_request(
        {"messages": messages},
        require_model=False,
        allow_managed_selectors=True,
    )
    request = GenerationRequest(
        messages=normalized.messages,
        tools=normalized.tools,
        tool_choice=normalized.tool_choice,
        parallel_tool_calls=normalized.parallel_tool_calls,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5 tokenizer is not cached locally")
    preparer = PromptPreparer(
        EngineConfig(model="Qwen/Qwen3.5-9B", prompt_cache_size=1),
        tokenizer,
        None,
    )

    first = asyncio.run(preparer.prepare(request, False))
    second = asyncio.run(preparer.prepare(request, False))
    template_messages = detached_template_messages(request.messages)
    rendered = tokenizer.apply_chat_template(
        template_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    detached = template_messages[1]["tool_calls"][0]["function"]["arguments"]

    assert first.value == second.value
    assert preparer.cache_entries == 1
    assert detached == {
        "direct": 1.25,
        "exponent": 100,
        "expanded": int(expanded),
        "integer": int(integer),
        "nested": {
            "decimal": 2.5,
            "exponent": 25,
            "expanded": int(expanded),
            "integer": int(integer),
        },
        "values": [0.625, 0.625, int(expanded), int(integer)],
    }
    assert "<parameter=direct>\n1.25\n</parameter>" in rendered
    assert "<parameter=exponent>\n100\n</parameter>" in rendered
    assert f"<parameter=expanded>\n{expanded}\n</parameter>" in rendered
    assert '"decimal": 2.5' in rendered
    assert f'"expanded": {expanded}' in rendered
    assert "<parameter=values>\n[0.625, 0.625," in rendered
    assert expanded in rendered
    assert integer in rendered
    assert messages == original
    assert request.messages[1]["tool_calls"][0]["function"]["arguments"] == arguments


def test_a_multimodal_template_rejection_is_a_client_error_not_an_unavailable_engine(
    monkeypatch,
) -> None:
    """the image branch renders through the processor, so it needed the same translation.

    Its only wrapper was the handler that closes decoded images and re-raises, which is about
    freeing memory rather than classifying failures. So an unrenderable multimodal request escaped
    unclassified and answered 503 exactly as the text path did before it was fixed.
    """

    class _Image:
        def close(self) -> None:
            return None

    class _Processor:
        tokenizer = SimpleNamespace()

        def apply_chat_template(self, *_args, **_kwargs):
            raise TypeError('can only concatenate str (not "dict") to str')

    monkeypatch.setattr(
        "flash.serve.runtime.prompt.prepare_multimodal_request",
        lambda *_args, **_kwargs: ([{"role": "user", "content": [{"type": "image"}]}], [_Image()]),
    )
    preparer = PromptPreparer(
        EngineConfig(model="model", image_limit=1),
        SimpleNamespace(),
        _Processor(),
    )
    request = GenerationRequest(
        messages=[{"role": "user", "content": [{"type": "image", "image": _data_uri()}]}]
    )

    import asyncio

    with pytest.raises(PromptError) as raised:
        asyncio.run(preparer.prepare(request, False))

    assert isinstance(raised.value, ValueError)
    assert "chat template rejected" in str(raised.value)
