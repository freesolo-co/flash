"""Per-adapter thinking default applied at prompt render time.

A Qwen3.5 run trained ``thinking=false`` must serve without an open ``<think>`` block even when a
raw chat client samples it with no ``chat_template_kwargs`` — otherwise the tokenizer's chat
template runs at its built-in default (thinking ON for Qwen3.5) and the model emits a reasoning
preamble ("…</think>{json}") that breaks JSON-only outputs. ``_LoraEngineImpl._tokenize_prompt`` fixes
``enable_thinking`` to the adapter's stored ``thinking`` flag; caller-supplied values are ignored.

modal_app imports the ``modal`` SDK at module top (decorators run at import), which isn't installed
in the offline test env, so we stub it just enough to import the module + reach the engine class.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from flash.serving.src.io.schemas import GenerateRequest
from flash.serving.src.store.registry import AdapterRegistry
from tests.serving.checkpoint_fixtures import checkpoint_record


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


_OPEN_THINK_SUFFIX = "<think>\n"
_CLOSED_THINK_SUFFIX = "<think>\n\n</think>\n\n"


class _StubTokenizer:
    """Stand-in for Qwen chat templates with adapter-controlled thinking.

    ``apply_chat_template(tokenize=True)`` returns the rendered prompt encoded one int per
    character (codepoints), so ``decode`` round-trips exactly and the test can assert whether
    the rendered text ends with an open or closed ``<think>`` block.
    """

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        enable_thinking = kwargs.get("enable_thinking", True)
        body = "".join(str(m.get("content", "")) for m in messages)
        text = f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"
        text += _OPEN_THINK_SUFFIX if enable_thinking else _CLOSED_THINK_SUFFIX
        return [ord(c) for c in text] if tokenize else text

    def decode(self, ids: list[int], **_kw: Any) -> str:
        return "".join(chr(i) for i in ids)


def _engine(modal_app_module: Any, *, thinking: bool) -> Any:
    eng = modal_app_module._LoraEngineImpl()
    eng.tokenizer = _StubTokenizer()
    registry = AdapterRegistry()
    registry.upsert(checkpoint_record("r1", "Qwen/Qwen3.5-9B", thinking=thinking))
    eng.registry = registry
    return eng


def _render(eng: Any, payload: GenerateRequest) -> str:
    record = eng.registry.get("org-1", payload.adapter_id)
    assert record is not None
    return eng.tokenizer.decode(eng._tokenize_prompt(payload, thinking_default=record.thinking))


def test_no_think_adapter_default_path_has_no_open_think(modal_app_module):
    """A no-think adapter sampled with NO chat_template_kwargs must not render an open <think>."""
    eng = _engine(modal_app_module, thinking=False)
    payload = GenerateRequest(
        adapter_id="r1/final", messages=[{"role": "user", "content": "ping"}]
    )  # no chat_template_kwargs at all
    rendered = _render(eng, payload)
    assert rendered.endswith(_CLOSED_THINK_SUFFIX)


def test_thinking_adapter_default_path_keeps_open_think(modal_app_module):
    """A thinking adapter (the default) on the same no-kwarg path still renders the <think> block,
    proving the per-adapter default is what's applied (not a blanket disable)."""
    eng = _engine(modal_app_module, thinking=True)
    payload = GenerateRequest(adapter_id="r1/final", messages=[{"role": "user", "content": "ping"}])
    rendered = _render(eng, payload)
    assert rendered.endswith(_OPEN_THINK_SUFFIX)


def test_caller_enable_thinking_does_not_override_adapter_default(modal_app_module):
    """Caller-supplied enable_thinking is not authoritative: a no-think adapter still renders in
    no-think mode even when the request explicitly asks for thinking."""
    eng = _engine(modal_app_module, thinking=False)
    payload = GenerateRequest(
        adapter_id="r1/final",
        messages=[{"role": "user", "content": "ping"}],
        chat_template_kwargs={"enable_thinking": True},
    )
    rendered = _render(eng, payload)
    assert rendered.endswith(_CLOSED_THINK_SUFFIX)


def test_null_enable_thinking_uses_adapter_default(modal_app_module):
    """An explicit ``enable_thinking: null`` is ignored so the adapter default still wins."""
    eng = _engine(modal_app_module, thinking=False)
    payload = GenerateRequest(
        adapter_id="r1/final",
        messages=[{"role": "user", "content": "ping"}],
        chat_template_kwargs={"enable_thinking": None},
    )
    rendered = _render(eng, payload)
    assert rendered.endswith(_CLOSED_THINK_SUFFIX)


def test_resolved_thinking_default_wins_over_registry(modal_app_module):
    """The thinking default RESOLVED with the LoRA weights (carried out of _lora_request) must take
    precedence over a fresh registry read: during a same-id redeploy the registry record can change
    between weight resolution and prompt render, so binding to the resolved record prevents rendering
    one record's weights with another's thinking default. Here the registry says thinking=True but the
    caller threads thinking_default=False (the record the weights came from) -> no open <think>."""
    eng = _engine(modal_app_module, thinking=True)  # registry record: thinking ON
    payload = GenerateRequest(adapter_id="r1/final", messages=[{"role": "user", "content": "ping"}])
    rendered = eng.tokenizer.decode(eng._tokenize_prompt(payload, thinking_default=False))
    assert rendered.endswith(_CLOSED_THINK_SUFFIX)
