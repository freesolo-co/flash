import json
from decimal import Decimal

import pytest

from flash.serve.request.openai import parse_chat_request
from flash.serve.request.tool_calls import normalize_tools, validate_tool_history_replay
from flash.serve.runtime.tool_calls import ToolCallStreamParser, parse_qwen3_coder_output


def declaration(cell_type=None):
    return [
        {
            "type": "function",
            "function": {
                "name": "store",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["values"],
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": cell_type
                                    if cell_type is not None
                                    else ["string", "number", "boolean", "null"],
                                },
                            },
                        }
                    },
                },
            },
        }
    ]


def call(payload):
    return (
        f"<tool_call><function=store><parameter=values>{payload}</parameter></function></tool_call>"
    )


def test_mixed_cells_survive_request_streaming_and_history():
    raw = declaration()
    tools = normalize_tools(raw)
    parse_chat_request(
        {"messages": [{"role": "user", "content": "write cells"}], "tools": raw},
        require_model=False,
        allow_managed_selectors=True,
    )
    payload = '[["12","=SUM(A1:A2)",true,false,null,9007199254740993.0,1e-400,1.25]]'
    text = call(payload)
    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")
    parser = ToolCallStreamParser(tools, id_factory=lambda: "call_fixed")
    assert all(parser.feed(character) == "" for character in text)
    assert parser.finish() == result
    arguments = result.calls[0].arguments
    assert json.loads(arguments, parse_float=Decimal)["values"] == json.loads(
        payload, parse_float=Decimal
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_fixed",
                    "type": "function",
                    "function": {"name": "store", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_fixed", "content": "ok"},
    ]
    validate_tool_history_replay(messages, tools)


@pytest.mark.parametrize("payload", ["[[{}]]", "[[[]]]", '[["\\ud800"]]', "[[NaN]]"])
def test_invalid_cells_are_not_executable(payload):
    assert not parse_qwen3_coder_output(call(payload), normalize_tools(declaration())).calls


@pytest.mark.parametrize(
    "types",
    [[], ["string", "string"], ["object", "null"], ["array"], ["unknown"], [None], [["string"]]],
)
def test_invalid_union_definitions_are_rejected(types):
    with pytest.raises(ValueError, match="supported scalar types"):
        normalize_tools(declaration(types))


def test_direct_union_remains_rejected():
    raw = declaration()
    raw[0]["function"]["parameters"]["properties"]["values"] = {"type": ["string", "number"]}
    with pytest.raises(ValueError, match="only inside JSON"):
        normalize_tools(raw)


def test_nested_object_union_and_enum():
    raw = declaration()
    raw[0]["function"]["parameters"]["properties"]["values"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["cell"],
        "properties": {"cell": {"type": ["boolean", "integer"], "enum": [True, 2]}},
    }
    tools = normalize_tools(raw)
    for payload in ['{"cell":true}', '{"cell":2.0}']:
        assert parse_qwen3_coder_output(call(payload), tools).calls
    for payload in ['{"cell":1}', '{"cell":false}', '{"cell":2.5}']:
        assert not parse_qwen3_coder_output(call(payload), tools).calls


def test_integer_union_preserves_number_semantics_and_copies_definition():
    raw = declaration(["integer", "null"])
    tools = normalize_tools(raw)
    raw[0]["function"]["parameters"]["properties"]["values"]["items"]["items"]["type"].append(
        "number"
    )
    assert not parse_qwen3_coder_output(call("[[1.5]]"), tools).calls
    assert not parse_qwen3_coder_output(call("[[true]]"), tools).calls
    result = parse_qwen3_coder_output(call("[[2.0,null]]"), tools)
    assert json.loads(result.calls[0].arguments)["values"] == [[2, None]]
    result = parse_qwen3_coder_output(call("[[1.5]]"), normalize_tools(raw))
    assert json.loads(result.calls[0].arguments)["values"] == [[1.5]]
