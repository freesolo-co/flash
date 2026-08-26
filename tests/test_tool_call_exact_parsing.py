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
    validate_tool_stop_sequences,
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
                            "selected": {"type": "number"},
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


@pytest.mark.parametrize("stop", ["\n", " ", "\t", "\r\n", " \t\r\n"])
def test_active_tool_stops_reject_parser_whitespace_separators(stop: str) -> None:
    with pytest.raises(ValueError, match="whitespace separators"):
        validate_tool_stop_sequences(stop=(stop,), tools=_exact_tools(), tool_choice="auto")


def test_active_tool_stops_accept_ordinary_text_and_inactive_whitespace() -> None:
    tools = _exact_tools()

    validate_tool_stop_sequences(stop=("END", "not whitespace"), tools=tools, tool_choice="auto")
    validate_tool_stop_sequences(stop=("\n",), tools=tools, tool_choice="none")


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
    declaration["function"]["parameters"]["properties"]["count"]["enum"] = [1]
    tools = normalize_tools([declaration])
    text = _exact_call().replace(
        "<parameter=count>2</parameter>", "<parameter=count>1.0</parameter>"
    )

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert '"count":1' in result.calls[0].arguments


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


def _property_name_tool(property_name: str, *, nested: bool) -> list[dict[str, object]]:
    property_schema: dict[str, object] = {"type": "string"}
    if nested:
        property_schema = {
            "type": "object",
            "properties": {property_name: {"type": "string"}},
            "required": [property_name],
            "additionalProperties": False,
        }
        property_name = "outer"
    return [
        {
            "type": "function",
            "function": {
                "name": "store",
                "parameters": {
                    "type": "object",
                    "properties": {property_name: property_schema},
                    "required": [property_name],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _unicode_declaration(location: str, value: str) -> list[dict[str, object]]:
    declaration = _property_name_tool("value", nested=False)
    function = declaration[0]["function"]
    parameters = function["parameters"]
    schema = parameters["properties"]["value"]
    if location == "function-description":
        function["description"] = value
    elif location == "root-schema-description":
        parameters["description"] = value
    elif location == "nested-schema-description":
        schema["description"] = value
    elif location == "string-enum":
        schema["enum"] = [value]
    elif location == "array-enum-string":
        parameters["properties"]["value"] = {
            "type": "array",
            "items": {"type": "string"},
            "enum": [[value]],
        }
    elif location == "object-enum-string":
        parameters["properties"]["value"] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
            "enum": [{"nested": value}],
        }
    elif location == "object-enum-key":
        parameters["properties"]["value"] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
            "enum": [{value: "nested"}],
        }
    else:
        return _property_name_tool(value, nested=True)
    return declaration


_UNICODE_DECLARATION_LOCATIONS = (
    "function-description",
    "root-schema-description",
    "nested-schema-description",
    "string-enum",
    "array-enum-string",
    "object-enum-string",
    "object-enum-key",
    "nested-property-key",
)


@pytest.mark.parametrize("location", _UNICODE_DECLARATION_LOCATIONS)
def test_tool_declarations_reject_unpaired_surrogates_after_normalization(location: str) -> None:
    with pytest.raises(ValueError, match="tools cannot contain an unpaired surrogate"):
        normalize_tools(_unicode_declaration(location, "\ud800"))


@pytest.mark.parametrize("location", _UNICODE_DECLARATION_LOCATIONS)
def test_tool_declarations_preserve_valid_non_bmp_unicode(location: str) -> None:
    tools = normalize_tools(_unicode_declaration(location, "forecast_🌦"))

    assert "forecast_🌦" in json.dumps(tools[0].wire(), ensure_ascii=False)


def test_root_property_surrogate_preserves_identifier_error_precedence() -> None:
    with pytest.raises(ValueError, match="properties key is invalid"):
        normalize_tools(_property_name_tool("\ud800", nested=False))


def test_surrogate_check_runs_after_structural_normalization() -> None:
    declaration = _unicode_declaration("function-description", "bad\ud800")
    del declaration[0]["function"]["parameters"]["additionalProperties"]

    with pytest.raises(ValueError, match="additionalProperties must be false"):
        normalize_tools(declaration)


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


def _numeric_enum_tool(location: str, value: object) -> list[dict[str, object]]:
    declaration = _property_name_tool("value", nested=False)
    parameters = declaration[0]["function"]["parameters"]
    if location == "direct":
        parameters["properties"]["value"] = {"type": "number", "enum": [value]}
    elif location == "root":
        parameters["enum"] = [{"value": value}]
    elif location == "nested":
        parameters["properties"]["value"] = {
            "type": "object",
            "properties": {"number": {"type": "number", "enum": [value]}},
            "required": ["number"],
            "additionalProperties": False,
        }
    elif location == "array-member":
        parameters["properties"]["value"] = {
            "type": "array",
            "items": {"type": "number"},
            "enum": [[value]],
        }
    else:
        parameters["properties"]["value"] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
            "enum": [{"number": value}],
        }
    return declaration


@pytest.mark.parametrize("location", ["direct", "root", "nested"])
@pytest.mark.parametrize("value", [1.0, Decimal("1.25")], ids=["float", "decimal"])
def test_numeric_enum_rejects_inexact_python_numbers(location: str, value: object) -> None:
    with pytest.raises(ValueError, match="numeric enum members must be JSON integers"):
        normalize_tools(_numeric_enum_tool(location, value))


@pytest.mark.parametrize("location", ["array-member", "object-member"])
def test_numeric_enum_rejects_floats_nested_in_container_members(location: str) -> None:
    with pytest.raises(ValueError, match="numeric enum members must be JSON integers"):
        normalize_tools(_numeric_enum_tool(location, 1.0))


def test_numeric_enum_accepts_arbitrarily_large_exact_integers() -> None:
    exact = 10**400 + 123

    tools = normalize_tools(_numeric_enum_tool("direct", exact))

    assert tools[0].parameters["properties"]["value"]["enum"] == [exact]


@pytest.mark.parametrize(
    ("schema_type", "enum", "extra"),
    [
        ("string", ["exact"], {}),
        ("boolean", [True, False], {}),
        ("null", [None], {}),
        ("array", [[1, 10**100]], {"items": {"type": "integer"}}),
        (
            "object",
            [{"values": ["exact"], "id": 10**100}],
            {
                "properties": {
                    "values": {"type": "array", "items": {"type": "string"}},
                    "id": {"type": "integer"},
                },
                "required": ["values", "id"],
                "additionalProperties": False,
            },
        ),
    ],
)
def test_nonnumeric_and_integer_container_enums_remain_supported(
    schema_type: str,
    enum: list[object],
    extra: dict[str, object],
) -> None:
    declaration = _property_name_tool("value", nested=False)
    declaration[0]["function"]["parameters"]["properties"]["value"] = {
        "type": schema_type,
        "enum": enum,
        **extra,
    }

    tools = normalize_tools(declaration)

    assert tools[0].parameters["properties"]["value"]["enum"] == enum


def test_enum_fingerprint_detects_exact_numeric_duplicates_without_pairwise_comparison(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tool_calls_module,
        "_json_values_equal",
        lambda *_args: pytest.fail("enum uniqueness must not use pairwise recursive comparison"),
    )

    with pytest.raises(ValueError, match="enum values must be unique"):
        normalize_tools(_enum_tool([[1, 2], [1, 2]]))


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


@pytest.mark.parametrize(
    ("parameter_name", "raw_value"),
    [
        ("nested", '{"text":' + '{"child":' * 1100 + '"leaf"' + "}" * 1100 + "}"),
        ("values", "[" * 1100 + '"leaf"' + "]" * 1100),
    ],
    ids=["object", "array"],
)
def test_generated_decoder_recursion_falls_back_exactly_buffered_and_streaming(
    parameter_name: str,
    raw_value: str,
) -> None:
    text = (
        "<tool_call><function=store>"
        f"<parameter={parameter_name}>{raw_value}</parameter>"
        "</function></tool_call>"
    )

    buffered = parse_qwen3_coder_output(text, _delimiter_tools())
    parser = ToolCallStreamParser(_delimiter_tools())
    assert all(parser.feed(character) == "" for character in text)
    streamed = parser.finish()

    assert buffered == streamed
    assert buffered.content == text
    assert buffered.calls == ()


@pytest.mark.parametrize("schema_parameter", ["scalar", "count"], ids=["number", "integer"])
def test_generated_integer_lexeme_digit_boundary_is_exact_buffered_and_streaming(
    schema_parameter: str,
) -> None:
    literal = "9" * 1024
    original = "9007199254740993.0" if schema_parameter == "scalar" else "2"
    text = _exact_call().replace(
        f"<parameter={schema_parameter}>{original}</parameter>",
        f"<parameter={schema_parameter}>{literal}</parameter>",
    )

    buffered = parse_qwen3_coder_output(text, _exact_tools(), id_factory=lambda: "call_fixed")
    parser = ToolCallStreamParser(_exact_tools(), id_factory=lambda: "call_fixed")
    assert all(parser.feed(character) == "" for character in text)
    streamed = parser.finish()

    assert buffered == streamed
    assert f'"{schema_parameter}":{literal}' in buffered.calls[0].arguments


@pytest.mark.parametrize("digits", [1025, 5000], ids=["first-over-limit", "python-cap-proof"])
def test_generated_oversize_integer_lexemes_fall_back_without_interpreter_errors(
    digits: int,
) -> None:
    literal = "9" * digits
    text = _exact_call().replace(
        "<parameter=count>2</parameter>",
        f"<parameter=count>{literal}</parameter>",
    )

    buffered = parse_qwen3_coder_output(text, _exact_tools())
    parser = ToolCallStreamParser(_exact_tools())
    assert all(parser.feed(character) == "" for character in text)
    streamed = parser.finish()

    assert buffered == streamed
    assert buffered.content == text
    assert buffered.calls == ()
