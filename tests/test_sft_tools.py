"""SFT rows that carry ``tools`` train on the prompt the served adapter actually sees.

Serving renders a chat request's ``tools`` into the prompt (``text_tokenizer`` passes
``tools_wire(normalize_tools(...))`` to the chat template). An SFT row rendered without them trains
on a different prompt than the deployed adapter receives; for Qwen3.5 the difference is a whole
``# Tools`` block at the top of the system turn.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import ClassVar

import pytest

from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
from flash.engine.profiling.sft_workload import prepare_sft_workload
from flash.engine.profiling.workload_profile import sft_profile_input_digest
from flash.serve.request.tool_calls import normalize_tools, tools_wire
from tests.test_sft_workload import FakeTokenizer


def _spec(max_context_tokens: int = 256) -> JobSpec:
    # a tool block is long; the shared fixture's 32-token cap would truncate every target away
    spec = JobSpec(
        model="test/model",
        model_revision="a" * 40,
        algorithm="sft",
        environment=EnvironmentSpec(id="team/example", resolved_sha="b" * 40),
        train=TrainSpec(epochs=1, batch_size=1, max_context_tokens=max_context_tokens),
        seed=7,
    )
    digest = sft_profile_input_digest(
        spec, tokenizer_revision=spec.model_revision, producer_version="1.2.3"
    )
    return replace(
        spec, workload_profile_input_digest=digest, workload_profile_producer_version="1.2.3"
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_user_id_by_email",
            "description": "Find a user id by email.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string"}},
                "required": ["email"],
                "additionalProperties": False,
            },
        },
    }
]


class RecordingToolTokenizer(FakeTokenizer):
    """Renders declared tools into the text and records every render's tools argument."""

    calls: ClassVar[list[object]] = []

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking, **kwargs
    ):
        tools = kwargs.get("tools", "<absent>")
        type(self).calls.append(tools)
        text = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        if tools != "<absent>" and tools is not None:
            names = ",".join(tool["function"]["name"] for tool in tools)
            text = f"[tools:{names}]" + text
        return text


class ToolRowEnvironment:
    multi_turn = False
    package_root = None

    def __init__(self, rows):
        self._rows = rows

    def dataset(self):
        return list(self._rows)

    def prompt_messages(self, row):
        raise AssertionError("stored prompt_messages must be used")

    def sft_completion(self, row):
        return [{"role": "assistant", "content": row["answer"]}]


def _row(**extra):
    return {
        "input": "cancel",
        "prompt_messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "cancel #W1"},
        ],
        "answer": "email?",
        **extra,
    }


def _prepare(rows):
    RecordingToolTokenizer.calls = []
    return prepare_sft_workload(
        _spec(),
        ToolRowEnvironment(rows),
        tokenizer_loader=lambda *_: RecordingToolTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda *_: ("pure-attention", True),
    )


def test_a_row_with_tools_renders_them_in_every_training_view() -> None:
    prepared = _prepare([_row(tools=TOOLS)])

    expected = tools_wire(normalize_tools(TOOLS))
    # transcript, prompt boundary and any mask probe: none may render the row without its tools
    assert RecordingToolTokenizer.calls
    assert all(call == expected for call in RecordingToolTokenizer.calls)
    assert prepared.sampled_texts == ["[tools:find_user_id_by_email]policycancel #W1email?"]
    assert prepared.profile.retained_examples == 1
    # the tool block is prompt, not target: it lengthens the row but adds no supervised token
    plain = _prepare([_row()])
    assert len(prepared.rows[0]["input_ids"]) > len(plain.rows[0]["input_ids"])
    assert sum(prepared.rows[0]["loss_mask"]) == sum(plain.rows[0]["loss_mask"])


def test_a_row_without_tools_renders_exactly_as_before() -> None:
    prepared = _prepare([_row()])

    assert RecordingToolTokenizer.calls
    assert all(call == "<absent>" for call in RecordingToolTokenizer.calls)
    assert prepared.sampled_texts == ["policycancel #W1email?"]


def test_explicit_null_tools_is_the_same_as_no_tools() -> None:
    _prepare([_row(tools=None)])

    assert all(call == "<absent>" for call in RecordingToolTokenizer.calls)


def test_tools_are_normalized_as_serving_normalizes_a_request() -> None:
    # key order in the authored declaration must not change the rendered (and trained) prompt
    reordered = [
        {
            "function": {
                "parameters": TOOLS[0]["function"]["parameters"],
                "description": TOOLS[0]["function"]["description"],
                "name": TOOLS[0]["function"]["name"],
            },
            "type": "function",
        }
    ]
    _prepare([_row(tools=reordered)])
    reordered_calls = list(RecordingToolTokenizer.calls)
    _prepare([_row(tools=TOOLS)])

    assert json.dumps(reordered_calls) == json.dumps(RecordingToolTokenizer.calls)


@pytest.mark.parametrize(
    "tools",
    [
        [],
        "find_user_id_by_email",
        [{"type": "function"}],
        [{"type": "retrieval", "function": TOOLS[0]["function"]}],
    ],
)
def test_malformed_tools_fail_the_profile(tools) -> None:
    with pytest.raises(ValueError, match="sft tools"):
        _prepare([_row(tools=tools)])


def test_tools_on_an_image_row_are_refused(monkeypatch) -> None:
    import flash.content.multimodal as multimodal

    monkeypatch.setattr(multimodal, "record_has_images", lambda *_: True)
    with pytest.raises(ValueError, match="not supported on image-bearing rows"):
        _prepare([_row(tools=TOOLS)])


@pytest.mark.skipif(
    os.environ.get("FLASH_SFT_TOOLS_TEMPLATE_LIVE") != "1",
    reason="real Qwen3.5 template parity disabled; set FLASH_SFT_TOOLS_TEMPLATE_LIVE=1",
)
def test_qwen35_training_text_is_the_served_prompt_plus_the_target() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    rows = [
        _row(tools=TOOLS),
        {
            **_row(tools=TOOLS),
            "answer": None,
            "output": {
                "messages": [
                    {"role": "assistant", "content": "What is your email?"},
                    {"role": "user", "content": "my email is a@b.c"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "find_user_id_by_email",
                                    "arguments": {"email": "a@b.c"},
                                },
                            }
                        ],
                    },
                ]
            },
        },
    ]

    class Env(ToolRowEnvironment):
        def sft_completion(self, row):
            if row.get("output"):
                return row["output"]["messages"]
            return super().sft_completion(row)

    prepared = prepare_sft_workload(
        _spec(max_context_tokens=4096),
        Env(rows),
        tokenizer_loader=lambda *_: tokenizer,
        producer_version="1.2.3",
        packing_support=lambda *_: ("pure-attention", True),
    )
    served_tools = tools_wire(normalize_tools(TOOLS))
    for row, text in zip(rows, prepared.sampled_texts, strict=True):
        completion = Env(rows).sft_completion(row)
        served = tokenizer.apply_chat_template(
            [*row["prompt_messages"], *completion],
            tools=served_tools,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        assert text == served
        assert text.startswith("<|im_start|>system\n# Tools")
    # the multi-turn row's loss covers the assistant turns (text and tool call) and nothing else:
    # not the tool block, the policy or the customer's turn
    multiturn = prepared.rows[1]
    supervised = tokenizer.decode(
        [t for t, m in zip(multiturn["input_ids"], multiturn["loss_mask"], strict=True) if m]
    )
    assert "What is your email?" in supervised
    assert "find_user_id_by_email" in supervised
    assert "# Tools" not in supervised
    assert "policy" not in supervised
    assert "my email is" not in supervised
