"""exact numeric and delimiter ownership regressions for tool calls."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from flash.serve.runtime.tool_calls import (
    ToolCallStreamParser,
    normalize_tools,
    parse_qwen3_coder_output,
)


def _exact_tools():
    return normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scalar": {"type": "number"},
                            "nested": {
                                "type": "object",
                                "properties": {
                                    "exact": {"type": "number"},
                                    "tiny": {"type": "number"},
                                    "text": {"type": "string"},
                                },
                                "required": ["exact", "tiny", "text"],
                                "additionalProperties": False,
                            },
                            "values": {"type": "array", "items": {"type": "number"}},
                            "selected": {"type": "number", "enum": [1.25]},
                            "count": {"type": "integer"},
                            "enabled": {"type": "boolean"},
                            "label": {"type": "string"},
                            "empty": {"type": "null"},
                        },
                        "required": [
                            "scalar",
                            "nested",
                            "values",
                            "selected",
                            "count",
                            "enabled",
                            "label",
                            "empty",
                        ],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )


def _exact_call() -> str:
    return (
        "<tool_call><function=store>"
        "<parameter=scalar>9007199254740993.0</parameter>"
        '<parameter=nested>{"exact":9007199254740993.0,"tiny":1e-400,'
        '"text":"inside </tool_call> value"}</parameter>'
        "<parameter=values>[9007199254740993.0,1e-400]</parameter>"
        "<parameter=selected>1.25</parameter>"
        "<parameter=count>2</parameter>"
        "<parameter=enabled>true</parameter>"
        "<parameter=label>ok</parameter>"
        "<parameter=empty>null</parameter>"
        "</function></tool_call>"
    )


def test_tool_call_numbers_round_trip_exactly_across_scalar_and_nested_values() -> None:
    tools = _exact_tools()
    result = parse_qwen3_coder_output(_exact_call(), tools, id_factory=lambda: "call_fixed")

    assert result.content is None
    assert result.calls[0].arguments == (
        '{"scalar":9007199254740993.0,'
        '"nested":{"exact":9007199254740993.0,"tiny":1e-400,'
        '"text":"inside </tool_call> value"},'
        '"values":[9007199254740993.0,1e-400],"selected":1.25,'
        '"count":2,"enabled":true,"label":"ok","empty":null}'
    )
    decoded = json.loads(result.calls[0].arguments, parse_float=Decimal)
    assert decoded["scalar"] == Decimal("9007199254740993.0")
    assert decoded["nested"]["tiny"] == Decimal("1e-400")
    assert decoded["values"] == [Decimal("9007199254740993.0"), Decimal("1e-400")]
    json.dumps(tools[0].parameters, allow_nan=False)


def test_tool_call_exact_numbers_survive_character_at_a_time_streaming() -> None:
    parser = ToolCallStreamParser(_exact_tools(), id_factory=lambda: "call_fixed")

    assert all(parser.feed(character) == "" for character in _exact_call())
    result = parser.finish()

    assert "9007199254740993.0" in result.calls[0].arguments
    assert "1e-400" in result.calls[0].arguments


def _huge_exponent_call() -> str:
    return _exact_call().replace("9007199254740993.0</parameter>", "1e100000</parameter>", 1)


def test_tool_call_huge_positive_exponent_stays_compact_and_exact() -> None:
    result = parse_qwen3_coder_output(
        _huge_exponent_call(),
        _exact_tools(),
        id_factory=lambda: "call_fixed",
    )

    arguments = result.calls[0].arguments
    assert len(arguments) < 512
    assert '"scalar":1e+100000' in arguments
    assert json.loads(arguments, parse_float=Decimal)["scalar"] == Decimal("1e100000")


def test_tool_call_huge_positive_exponent_survives_character_at_a_time_streaming() -> None:
    parser = ToolCallStreamParser(_exact_tools(), id_factory=lambda: "call_fixed")

    assert all(parser.feed(character) == "" for character in _huge_exponent_call())
    arguments = parser.finish().calls[0].arguments

    assert len(arguments) < 512
    assert json.loads(arguments, parse_float=Decimal)["scalar"] == Decimal("1e100000")


@pytest.mark.parametrize(
    "raw_value",
    ["NaN", "Infinity", "-Infinity", '{"exact":NaN,"tiny":1,"text":"x"}'],
)
def test_tool_call_numbers_reject_non_finite_values(raw_value: str) -> None:
    text = _exact_call()
    if raw_value.startswith("{"):
        text = text.replace(
            '{"exact":9007199254740993.0,"tiny":1e-400,"text":"inside </tool_call> value"}',
            raw_value,
        )
    else:
        text = text.replace("9007199254740993.0</parameter>", f"{raw_value}</parameter>", 1)

    result = parse_qwen3_coder_output(text, _exact_tools())

    assert result.content == text
    assert result.calls == ()


@pytest.mark.parametrize(
    "enum_value",
    [
        "value </parameter><parameter=next>",
        "value </parameter>\n</function>",
    ],
    ids=["next-parameter", "function-close"],
)
def test_string_enum_rejects_unrepresentable_structural_delimiter(enum_value: str) -> None:
    declaration = _exact_tools()[0].wire()
    declaration["function"]["parameters"]["properties"]["label"]["enum"] = [enum_value]

    with pytest.raises(ValueError, match="unrepresentable tool grammar delimiter"):
        normalize_tools([declaration])


def test_string_enum_allows_nonstructural_parameter_delimiter_text() -> None:
    declaration = _exact_tools()[0].wire()
    enum_value = "before </parameter> after"
    declaration["function"]["parameters"]["properties"]["label"]["enum"] = [enum_value]
    tools = normalize_tools([declaration])
    text = _exact_call().replace(
        "<parameter=label>ok</parameter>", f"<parameter=label>{enum_value}</parameter>"
    )

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert json.loads(result.calls[0].arguments)["label"] == enum_value


def _delimiter_tools():
    return normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scalar": {"type": "string"},
                            "nested": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                                "additionalProperties": False,
                            },
                            "values": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )


def _delimiter_calls() -> str:
    return (
        "<tool_call><function=store>"
        "<parameter=scalar>before </tool_call> after</parameter>"
        '<parameter=nested>{"text":"nested </tool_call> value"}</parameter>'
        '<parameter=values>["array </tool_call> value"]</parameter>'
        "</function></tool_call>"
        " <tool_call><function=store>"
        "<parameter=scalar>second </tool_call> value</parameter>"
        "</function></tool_call>"
    )


def test_tool_call_outer_closer_is_owned_after_function_grammar() -> None:
    text = "visible " + _delimiter_calls()
    result = parse_qwen3_coder_output(text, _delimiter_tools(), id_factory=lambda: "call_fixed")

    assert result.content == "visible "
    assert len(result.calls) == 2
    assert json.loads(result.calls[0].arguments) == {
        "scalar": "before </tool_call> after",
        "nested": {"text": "nested </tool_call> value"},
        "values": ["array </tool_call> value"],
    }
    assert json.loads(result.calls[1].arguments) == {"scalar": "second </tool_call> value"}


def test_tool_call_outer_closer_survives_arbitrary_stream_splits_and_multiple_calls() -> None:
    parser = ToolCallStreamParser(_delimiter_tools(), id_factory=lambda: "call_fixed")
    emitted = "".join(parser.feed(character) for character in "visible " + _delimiter_calls())
    result = parser.finish()

    assert emitted == "visible "
    assert result.content is None
    assert len(result.calls) == 2
    assert "before </tool_call> after" in result.calls[0].arguments
    assert "second </tool_call> value" in result.calls[1].arguments
