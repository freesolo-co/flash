"""import purity, public types, prompts, images, and structured outputs."""

from __future__ import annotations

import base64
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from flash.serve.runtime import (
    AdapterSpec,
    EngineConfig,
    GenerationRequest,
    RuntimeConfigurationError,
    StructuredOutputsError,
    normalize_structured_outputs,
)
from flash.serve.runtime.multimodal import prepare_multimodal_request
from flash.serve.runtime.prompt import PromptPreparer, resolve_thinking

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}


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
    with pytest.raises(RuntimeConfigurationError, match="runtime-owned"):
        EngineConfig(model="model", tokenizer_kwargs={"token": "other"})


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
        self.encode_calls = 0

    def apply_chat_template(self, _messages, **kwargs):
        self.template_calls.append(kwargs)
        return [1, 2, 3]

    def encode(self, _prompt, **_kwargs):
        self.encode_calls += 1
        return [4, 5]


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

    with pytest.raises(RuntimeError, match="render failed"):
        asyncio.run(preparer.prepare(request, False))
    assert closed == [True]
