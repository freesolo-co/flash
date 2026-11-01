"""Reasoning text is never executable; answer tool parsing keeps streaming parity."""

import json

import pytest

from flash.serve.app.openai import ReasoningDeltaSplitter, split_reasoning
from flash.serve.request.openai import parse_chat_request
from flash.serve.request.tool_calls import normalize_tools
from flash.serve.runtime.reasoning_tools import ReasoningToolStreamParser, parse_tool_output

TOOLS = normalize_tools(
    [
        {
            "type": "function",
            "function": {
                "name": "write_range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": ["string", "number", "boolean", "null"]},
                            },
                        }
                    },
                    "required": ["values"],
                    "additionalProperties": False,
                },
            },
        }
    ]
)
CALL = '<tool_call><function=write_range><parameter=values>[["12",12,true,null]]</parameter></function></tool_call>'


@pytest.mark.parametrize("prefix", ["", "plan", "<think>plan", CALL, "<think>" + CALL])
@pytest.mark.parametrize(
    "answer", [CALL, "I will write. " + CALL, "ordinary answer", "<tool_call>broken"]
)
def test_every_split_preserves_reasoning_and_only_parses_answer(prefix, answer):
    text = prefix + "</think>" + answer
    expected = parse_tool_output(text, TOOLS, thinking=True, id_factory=lambda: "call_fixed")
    reasoning, content = split_reasoning(expected.content or "", thinking=True)
    assert bool(expected.calls) == answer.endswith(CALL)
    for boundary in range(len(text) + 1):
        parser = ReasoningToolStreamParser(TOOLS, thinking=True, id_factory=lambda: "call_fixed")
        visible = parser.feed(text[:boundary]) + parser.feed(text[boundary:])
        final = parser.finish()
        visible += final.content or ""
        assert final.calls == expected.calls
        assert visible == expected.content
        splitter = ReasoningDeltaSplitter(thinking=True)
        fields = {"content": "", "reasoning_content": ""}
        for character in visible:
            for key, value in splitter.feed(character):
                fields[key] += value
        for key, value in splitter.finish():
            fields[key] += value
        assert fields == {"content": content, "reasoning_content": reasoning}


@pytest.mark.parametrize("text", [CALL, "<think>" + CALL, "plan</thin", "", "</think>"])
def test_unfinished_reasoning_cannot_call_tools(text):
    expected = parse_tool_output(text, TOOLS, thinking=True)
    assert not expected.calls
    parser = ReasoningToolStreamParser(TOOLS, thinking=True)
    visible = "".join(parser.feed(character) for character in text)
    final = parser.finish()
    assert not final.calls
    assert visible + (final.content or "") == text


def test_reasoning_tool_mentions_survive_followup_history():
    parsed = parse_tool_output(CALL + "</think>" + CALL, TOOLS, thinking=True)
    reasoning, content = split_reasoning(parsed.content, thinking=True)
    messages = [
        {"role": "user", "content": "Write the cells"},
        {
            "role": "assistant",
            "content": content or None,
            "reasoning_content": reasoning,
            "tool_calls": [call.wire() for call in parsed.calls],
        },
        {"role": "tool", "tool_call_id": parsed.calls[0].id, "content": "done"},
    ]
    request = parse_chat_request(
        {"messages": messages, "tools": [tool.wire() for tool in TOOLS]},
        require_model=False,
        allow_managed_selectors=True,
    )
    assert request.messages[1]["reasoning_content"] == CALL
    assert json.loads(parsed.calls[0].arguments)["values"] == [["12", 12, True, None]]


def test_non_thinking_does_not_treat_literal_reasoning_tags_as_a_boundary():
    text = "literal </think> " + CALL
    result = parse_tool_output(text, TOOLS)
    assert result.content == "literal </think> "
    assert len(result.calls) == 1
