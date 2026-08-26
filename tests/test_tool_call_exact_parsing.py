"""exact numeric and delimiter ownership regressions for tool calls."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

import flash.serve.runtime.tool_calls as tool_calls_module
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
    ("raw_value", "expected"),
    [
        ("1.0", "1"),
        ("1e3", "1000"),
        ("9007199254740993.0", "9007199254740993"),
    ],
)
def test_integer_schema_accepts_exact_integral_json_numbers(
    raw_value: str,
    expected: str,
) -> None:
    text = _exact_call().replace(
        "<parameter=count>2</parameter>", f"<parameter=count>{raw_value}</parameter>"
    )

    result = parse_qwen3_coder_output(text, _exact_tools(), id_factory=lambda: "call_fixed")

    assert f'"count":{expected}' in result.calls[0].arguments


def test_integer_schema_rejects_fractional_json_numbers_exactly() -> None:
    text = _exact_call().replace(
        "<parameter=count>2</parameter>", "<parameter=count>1.5</parameter>"
    )

    result = parse_qwen3_coder_output(text, _exact_tools())

    assert result.content == text
    assert result.calls == ()


def test_nested_integer_values_serialize_canonically() -> None:
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {
                                "type": "object",
                                "properties": {
                                    "count": {"type": "integer"},
                                    "values": {"type": "array", "items": {"type": "integer"}},
                                },
                                "required": ["count", "values"],
                                "additionalProperties": False,
                            }
                        },
                        "required": ["payload"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    text = (
        "<tool_call><function=store>"
        '<parameter=payload>{"count":1.0,"values":[1e3,9007199254740993.0]}</parameter>'
        "</function></tool_call>"
    )

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert result.calls[0].arguments == '{"payload":{"count":1,"values":[1000,9007199254740993]}}'


def test_integer_enum_uses_json_schema_mathematical_integer_semantics() -> None:
    declaration = _exact_tools()[0].wire()
    declaration["function"]["parameters"]["properties"]["count"]["enum"] = [1.0]

    tools = normalize_tools([declaration])

    assert tools[0].parameters["properties"]["count"]["enum"] == [1.0]


@pytest.mark.parametrize(
    ("raw_value", "accepted"),
    [("null", True), ("NULL", False), ("Null", False)],
)
def test_null_schema_accepts_only_the_exact_json_literal_buffered(
    raw_value: str,
    accepted: bool,
) -> None:
    text = _exact_call().replace(
        "<parameter=empty>null</parameter>",
        f"<parameter=empty>{raw_value}</parameter>",
    )

    result = parse_qwen3_coder_output(text, _exact_tools(), id_factory=lambda: "call_fixed")

    if accepted:
        assert result.content is None
        assert json.loads(result.calls[0].arguments)["empty"] is None
    else:
        assert result.content == text
        assert result.calls == ()


@pytest.mark.parametrize(
    ("raw_value", "accepted"),
    [("null", True), ("NULL", False), ("Null", False)],
)
def test_null_schema_casing_survives_arbitrary_stream_splits(
    raw_value: str,
    accepted: bool,
) -> None:
    text = _exact_call().replace(
        "<parameter=empty>null</parameter>",
        f"<parameter=empty>{raw_value}</parameter>",
    )

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(_exact_tools(), id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        if accepted:
            assert result.content is None
            assert json.loads(result.calls[0].arguments)["empty"] is None
        else:
            assert result.content == text
            assert result.calls == ()


@pytest.mark.parametrize("raw_value", ["1", "0", "TRUE", "False"])
def test_boolean_schema_rejects_non_json_boolean_literals_exactly(raw_value: str) -> None:
    text = _exact_call().replace(
        "<parameter=enabled>true</parameter>",
        f"<parameter=enabled>{raw_value}</parameter>",
    )

    result = parse_qwen3_coder_output(text, _exact_tools())

    assert result.content == text
    assert result.calls == ()


@pytest.mark.parametrize("raw_value", ["1e99999999999999999999", "1e-99999999999999999999"])
def test_extreme_decimal_exponents_fall_back_exactly(raw_value: str) -> None:
    text = _exact_call().replace("9007199254740993.0</parameter>", f"{raw_value}</parameter>", 1)

    result = parse_qwen3_coder_output(text, _exact_tools())

    assert result.content == text
    assert result.calls == ()


def test_generated_arguments_with_duplicate_object_keys_fall_back_exactly() -> None:
    text = _exact_call().replace(
        '"exact":9007199254740993.0,',
        '"exact":1,"exact":2,',
    )

    result = parse_qwen3_coder_output(text, _exact_tools())

    assert result.content == text
    assert result.calls == ()

    parser = ToolCallStreamParser(_exact_tools())
    assert all(parser.feed(character) == "" for character in text)
    streamed = parser.finish()
    assert streamed.content == text
    assert streamed.calls == ()


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


def _enum_tool(enum: list[object]) -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "array",
                            "items": {"type": "number"},
                            "enum": enum,
                        }
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def test_enum_fingerprint_detects_exact_numeric_duplicates_without_pairwise_comparison(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tool_calls_module,
        "_json_values_equal",
        lambda *_args: pytest.fail("enum uniqueness must not use pairwise recursive comparison"),
    )

    with pytest.raises(ValueError, match="enum values must be unique"):
        normalize_tools(_enum_tool([[1, 2], [1.0, 2.0]]))


def test_enum_fingerprint_uses_exact_json_number_equality() -> None:
    fingerprint = tool_calls_module._json_value_fingerprint

    assert fingerprint(1) == fingerprint(1.0) == fingerprint(Decimal("1.00"))
    assert fingerprint(9007199254740992) != fingerprint(9007199254740993)
    assert fingerprint(Decimal("1e-400")) != fingerprint(Decimal("0"))


def test_enum_fingerprint_accepts_bounded_distinct_nested_values() -> None:
    enum = [[*range(255), index] for index in range(128)]

    tools = normalize_tools(_enum_tool(enum))

    assert len(tools[0].parameters["properties"]["value"]["enum"]) == 128


def test_enum_member_depth_is_bounded_before_fingerprinting() -> None:
    enum_value = json.loads("[" * 600 + "0" + "]" * 600)

    with pytest.raises(ValueError, match="enum value complexity"):
        normalize_tools(_enum_tool([enum_value]))


def test_enum_aggregate_nodes_are_bounded_before_fingerprinting() -> None:
    enum = [[*range(512), index] for index in range(128)]

    with pytest.raises(ValueError, match="enum value complexity"):
        normalize_tools(_enum_tool(enum))


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
        '<parameter=nested>{"text":"nested </tool_call> value"}</parameter>'
        '<parameter=values>["array </tool_call> value"]</parameter>'
        "<parameter=scalar>before </tool_call> after</parameter>"
        "</function></tool_call>"
        " <tool_call><function=store>"
        "<parameter=scalar>second </tool_call> value</parameter>"
        "</function></tool_call>"
    )


def _structural_string_call() -> str:
    return (
        "<tool_call><function=store>"
        "<parameter=scalar>before </parameter></function> after</parameter>"
        "</function></tool_call>"
    )


def test_unconstrained_string_preserves_structural_function_close_text() -> None:
    result = parse_qwen3_coder_output(
        _structural_string_call(),
        _delimiter_tools(),
        id_factory=lambda: "call_fixed",
    )

    assert json.loads(result.calls[0].arguments) == {
        "scalar": "before </parameter></function> after"
    }


def test_unconstrained_structural_string_survives_arbitrary_stream_splits() -> None:
    parser = ToolCallStreamParser(_delimiter_tools(), id_factory=lambda: "call_fixed")

    assert all(parser.feed(character) == "" for character in _structural_string_call())
    result = parser.finish()

    assert json.loads(result.calls[0].arguments)["scalar"] == (
        "before </parameter></function> after"
    )


def test_complete_structural_candidate_inside_string_falls_back_exactly() -> None:
    malformed = _structural_string_call().replace(
        "</function> after", "</function></tool_call> after"
    )

    result = parse_qwen3_coder_output(malformed, _delimiter_tools())

    assert result.content == malformed
    assert result.calls == ()


def _optional_string_tools():
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        "a": {"type": "string"},
        "b": {"type": "string"},
    }
    declaration["function"]["parameters"]["required"] = ["a"]
    return normalize_tools([declaration])


def _optional_parameter_ambiguity() -> str:
    return (
        "<tool_call><function=store>"
        "<parameter=a>before </parameter><parameter=b>inside</parameter>"
        "</function></tool_call>"
    )


def test_unconstrained_string_before_optional_parameter_falls_back_exactly() -> None:
    text = _optional_parameter_ambiguity()

    result = parse_qwen3_coder_output(text, _optional_string_tools())

    assert result.content == text
    assert result.calls == ()


def test_optional_parameter_ambiguity_survives_arbitrary_stream_splits() -> None:
    text = _optional_parameter_ambiguity()

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(_optional_string_tools(), id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert result.content == text
        assert result.calls == ()


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


@pytest.mark.parametrize(
    ("parameter_name", "raw_value", "expected"),
    [
        (
            "nested",
            '{"text":"abc</parameter></function> and </tool_call>"}',
            {"text": "abc</parameter></function> and </tool_call>"},
        ),
        (
            "values",
            '["abc</parameter></function> and </tool_call>"]',
            ["abc</parameter></function> and </tool_call>"],
        ),
    ],
    ids=["object", "array"],
)
def test_container_values_continue_past_structural_text_inside_json_strings(
    parameter_name: str,
    raw_value: str,
    expected: object,
) -> None:
    text = (
        "<tool_call><function=store>"
        f"<parameter={parameter_name}>{raw_value}</parameter>"
        "</function></tool_call>"
    )

    result = parse_qwen3_coder_output(text, _delimiter_tools(), id_factory=lambda: "call_fixed")

    assert json.loads(result.calls[0].arguments) == {parameter_name: expected}


@pytest.mark.parametrize(
    ("parameter_name", "raw_value", "expected"),
    [
        (
            "nested",
            '{"text":"abc</parameter></function> and </tool_call>"}',
            {"text": "abc</parameter></function> and </tool_call>"},
        ),
        (
            "values",
            '["abc</parameter></function> and </tool_call>"]',
            ["abc</parameter></function> and </tool_call>"],
        ),
    ],
    ids=["object", "array"],
)
def test_container_structural_text_survives_arbitrary_stream_splits(
    parameter_name: str,
    raw_value: str,
    expected: object,
) -> None:
    text = (
        "<tool_call><function=store>"
        f"<parameter={parameter_name}>{raw_value}</parameter>"
        "</function></tool_call>"
    )

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(_delimiter_tools(), id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert json.loads(result.calls[0].arguments) == {parameter_name: expected}


@pytest.mark.parametrize(
    "raw_value",
    [
        '{"text":"abc</parameter></function>"',
        '["abc</parameter></tool_call>"',
    ],
    ids=["object", "array"],
)
def test_malformed_container_delimiter_candidates_fall_back_exactly(raw_value: str) -> None:
    parameter_name = "nested" if raw_value.startswith("{") else "values"
    text = (
        "<tool_call><function=store>"
        f"<parameter={parameter_name}>{raw_value}</parameter>"
        "</function></tool_call>"
    )

    result = parse_qwen3_coder_output(text, _delimiter_tools())

    assert result.content == text
    assert result.calls == ()


@pytest.mark.parametrize("parameter_name", ["nested", "values"])
def test_unpaired_surrogate_generated_container_arguments_fall_back_exactly(
    parameter_name: str,
) -> None:
    raw_value = '{"text":"\\ud800"}' if parameter_name == "nested" else '["\\ud800"]'
    text = (
        "<tool_call><function=store>"
        f"<parameter={parameter_name}>{raw_value}</parameter>"
        "</function></tool_call>"
    )

    result = parse_qwen3_coder_output(text, _delimiter_tools())

    assert result.content == text
    assert result.calls == ()
    json.dumps(result.content, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("parameter_name", ["nested", "values"])
def test_unpaired_surrogate_fallback_survives_arbitrary_stream_splits(
    parameter_name: str,
) -> None:
    raw_value = '{"text":"\\ud800"}' if parameter_name == "nested" else '["\\ud800"]'
    text = (
        "<tool_call><function=store>"
        f"<parameter={parameter_name}>{raw_value}</parameter>"
        "</function></tool_call>"
    )

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(_delimiter_tools(), id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert result.content == text
        assert result.calls == ()


@pytest.mark.parametrize(
    ("parameter_name", "raw_value", "expected"),
    [
        ("nested", '{"text":"\\ud83d\\ude00"}', {"text": "😀"}),
        ("values", '["\\ud83d\\ude00"]', ["😀"]),
    ],
    ids=["object", "array"],
)
def test_valid_non_bmp_surrogate_pairs_remain_structured(
    parameter_name: str,
    raw_value: str,
    expected: object,
) -> None:
    text = (
        "<tool_call><function=store>"
        f"<parameter={parameter_name}>{raw_value}</parameter>"
        "</function></tool_call>"
    )

    result = parse_qwen3_coder_output(text, _delimiter_tools(), id_factory=lambda: "call_fixed")

    assert json.loads(result.calls[0].arguments) == {parameter_name: expected}
    assert "😀" in result.calls[0].arguments
    result.calls[0].arguments.encode("utf-8")
