"""import purity, public types, prompts, images, and structured outputs."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

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
from flash.serve.runtime.tool_calls import (
    FunctionTool,
    ToolCallStreamParser,
    detached_template_messages,
    normalize_tools,
    parse_qwen3_coder_output,
)

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
        messages=[{"role": "user", "content": "weather"}],
        tools=_runtime_tools(),
        tool_choice="none",
        parallel_tool_calls=True,
        stop="</tool_call>",
    )
    assert request.stop == ("</tool_call>",)


def test_qwen3_coder_parser_validates_schema_and_exact_fallback() -> None:
    valid = (
        "<tool_call>\n<function=weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "<parameter=days>\n2\n</parameter>\n"
        "</function>\n</tool_call>  \n\t"
    )
    result = parse_qwen3_coder_output(valid, _runtime_tools(), id_factory=lambda: "call_fixed")
    assert result.content is None
    assert result.calls[0].wire() == {
        "id": "call_fixed",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city":"Paris","days":2}'},
    }
    malformed = valid.replace("<parameter=days>", "<parameter=unknown>")
    assert parse_qwen3_coder_output(malformed, _runtime_tools()).content == malformed


def _delimiter_value_tools():
    return normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "string_value": {"type": "string"},
                            "object_value": {
                                "type": "object",
                                "properties": {
                                    "nested": {
                                        "type": "object",
                                        "properties": {"text": {"type": "string"}},
                                        "required": ["text"],
                                        "additionalProperties": False,
                                    }
                                },
                                "required": ["nested"],
                                "additionalProperties": False,
                            },
                            "array_value": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "count": {"type": "integer"},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )


@pytest.mark.parametrize(
    ("parameter_name", "raw_value", "expected"),
    [
        ("string_value", "before </parameter> after", "before </parameter> after"),
        (
            "object_value",
            '{"nested":{"text":"before </parameter> after"}}',
            {"nested": {"text": "before </parameter> after"}},
        ),
        (
            "array_value",
            '[["before </parameter> after"]]',
            [["before </parameter> after"]],
        ),
    ],
    ids=["string", "nested-object", "nested-array"],
)
def test_qwen3_coder_parser_preserves_parameter_delimiters_inside_values(
    parameter_name: str,
    raw_value: str,
    expected: object,
) -> None:
    text = (
        "<tool_call><function=store>"
        f"<parameter={parameter_name}>{raw_value}</parameter>"
        "</function></tool_call>"
    )
    result = parse_qwen3_coder_output(
        text,
        _delimiter_value_tools(),
        id_factory=lambda: "call_fixed",
    )
    assert json.loads(result.calls[0].arguments) == {parameter_name: expected}


def test_qwen3_coder_parser_preserves_embedded_delimiter_before_another_parameter() -> None:
    text = (
        "<tool_call><function=store>"
        "<parameter=string_value>before </parameter> after</parameter>"
        "<parameter=count>2</parameter>"
        "</function></tool_call>"
    )
    result = parse_qwen3_coder_output(
        text,
        _delimiter_value_tools(),
        id_factory=lambda: "call_fixed",
    )
    assert json.loads(result.calls[0].arguments) == {
        "string_value": "before </parameter> after",
        "count": 2,
    }


def test_qwen3_coder_stream_parser_handles_split_embedded_and_structural_closes() -> None:
    parser = ToolCallStreamParser(_delimiter_value_tools(), id_factory=lambda: "call_fixed")
    pieces = [
        "<tool_call><function=store><parameter=string_value>before </para",
        "meter> after</para",
        "meter><parameter=count>2</parameter></function></tool_call>",
    ]
    assert [parser.feed(piece) for piece in pieces] == ["", "", ""]
    result = parser.finish()
    assert json.loads(result.calls[0].arguments) == {
        "string_value": "before </parameter> after",
        "count": 2,
    }


@pytest.mark.parametrize(
    "malformed",
    [
        (
            "<tool_call><function=store>"
            '<parameter=object_value>{"nested":{"text":"before </parameter> after"}'
            "</parameter></function></tool_call>"
        ),
        (
            "<tool_call><function=store>"
            "<parameter=string_value>before </parameter>"
            "<parameter=count>not-an-int</parameter> after</parameter>"
            "</function></tool_call>"
        ),
    ],
    ids=["malformed-json", "ambiguous-grammar"],
)
def test_qwen3_coder_parser_returns_malformed_delimiter_candidate_exactly(
    malformed: str,
) -> None:
    result = parse_qwen3_coder_output(malformed, _delimiter_value_tools())
    assert result.content == malformed
    assert result.calls == ()


def test_qwen3_coder_parser_accepts_boundary_property_names() -> None:
    longest = "x" * 64
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "boundaries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string"},
                            longest: {"type": "integer"},
                        },
                        "required": ["a", longest],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    text = (
        "<tool_call><function=boundaries>"
        "<parameter=a>first</parameter>"
        f"<parameter={longest}>2</parameter>"
        "</function></tool_call>"
    )
    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")
    assert result.calls[0].arguments == json.dumps(
        {"a": "first", longest: 2}, separators=(",", ":")
    )


def test_qwen3_coder_number_preserves_exact_integer_lexemes() -> None:
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "measure",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "exact": {"type": "number"},
                            "decimal": {"type": "number"},
                            "exponent": {"type": "number"},
                        },
                        "required": ["exact", "decimal", "exponent"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    text = (
        "<tool_call><function=measure>"
        "<parameter=exact>9007199254740993</parameter>"
        "<parameter=decimal>1.25</parameter>"
        "<parameter=exponent>1e3</parameter>"
        "</function></tool_call>"
    )
    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")
    assert result.calls[0].arguments == (
        '{"exact":9007199254740993,"decimal":1.25,"exponent":1000}'
    )


def test_qwen3_coder_stream_parser_buffers_candidates_and_falls_back_exactly() -> None:
    parser = ToolCallStreamParser(_runtime_tools(), id_factory=lambda: "call_fixed")
    pieces = [
        "prefix <tool",
        "_call>\n<function=weather>\n<parameter=city>\nParis",
        "\n</parameter>\n</function>\n</tool_call>  \n",
    ]
    assert [parser.feed(piece) for piece in pieces] == ["prefix ", "", ""]
    parsed = parser.finish()
    assert parsed.content is None
    assert parsed.calls[0].name == "weather"


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
