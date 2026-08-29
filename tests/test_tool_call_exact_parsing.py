"""exact numeric and delimiter ownership regressions for tool calls."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

import flash.serve.request.tool_calls as request_tool_calls_module
import flash.serve.runtime.tool_calls as tool_calls_module
from flash.serve.request.tool_calls import (
    normalize_tools,
    tools_wire,
    validate_tool_history,
    validate_tool_stop_sequences,
)
from flash.serve.runtime.tool_calls import ToolCallStreamParser, parse_qwen3_coder_output


def test_tools_wire_recursively_detaches_the_normalized_schema() -> None:
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )

    wire = tools_wire(tools)
    wire[0]["function"]["parameters"]["properties"]["value"]["type"] = "string"
    wire[0]["function"]["parameters"]["required"].clear()

    assert tools[0].parameters["properties"]["value"]["type"] == "integer"
    assert tools[0].parameters["required"] == ["value"]
    assert tools_wire(tools) != wire


@pytest.mark.parametrize(
    "enum_value",
    [
        ("tuple",),
        {1: "numeric key"},
        {1: "collision", "1": "string key"},
    ],
    ids=["tuple", "non-string-key", "coercive-key-collision"],
)
def test_tool_enum_rejects_nonexact_json_containers(enum_value: object) -> None:
    declaration = _enum_tool([enum_value])

    with pytest.raises(ValueError, match=r"exact JSON values|string-keyed JSON objects"):
        normalize_tools(declaration)


def test_parser_accepts_one_and_64_character_property_names() -> None:
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


@pytest.mark.parametrize(
    "enum_value",
    ["\nleading", "trailing\n", "a\nb"],
    ids=["leading-newline", "trailing-newline", "interior-newline"],
)
def test_string_enum_accepts_newline_members(enum_value: str) -> None:
    """every newline position round-trips, so no member needs declaration-time rejection.

    the runtime strips the grammar's newline wrapper only when BOTH sides are present, so a
    one-sided newline is the value's own and survives. these members are representable.
    """
    declaration = _exact_tools()[0].wire()
    declaration["function"]["parameters"]["properties"]["label"]["enum"] = [enum_value]

    tools = normalize_tools([declaration])

    assert tools[0].parameters["properties"]["label"]["enum"] == [enum_value]


@pytest.mark.parametrize(
    ("emitted", "decoded"),
    [
        ("\nfoo\n", "foo"),
        ("\nfoo", "\nfoo"),
        ("foo\n", "foo\n"),
        ("foo", "foo"),
        ("\n\nfoo\n", "\nfoo"),
        ("\nfoo\n\n", "foo\n"),
        ("", ""),
        ("\n", "\n"),
        ("\n\n", ""),
        ("\n\n\n", "\n"),
        ("\n\n\n\n", "\n\n"),
    ],
    ids=[
        "wrapped",
        "leading-only",
        "trailing-only",
        "bare",
        "wrapped-leading",
        "wrapped-trailing",
        "empty",
        "lone-newline",
        "wrapped-empty",
        "wrapped-lone-newline",
        "wrapped-two-newlines",
    ],
)
def test_free_string_strips_only_the_complete_newline_wrapper(emitted: str, decoded: str) -> None:
    """a one-sided newline is data, not framing, so it must survive the parse.

    stripping each side independently collapses four distinct wire forms onto one value, which
    silently invokes the tool with different arguments than the model emitted. the newline-only
    cases pin the length guard: a lone ``"\\n"`` must not be read as both halves of a wrapper,
    and a wrapped ``"\\n\\n"`` must still decode to the empty value.
    """
    text = _exact_call().replace(
        "<parameter=label>ok</parameter>", f"<parameter=label>{emitted}</parameter>"
    )

    result = parse_qwen3_coder_output(text, _exact_tools(), id_factory=lambda: "call_fixed")

    assert json.loads(result.calls[0].arguments)["label"] == decoded


@pytest.mark.parametrize(
    ("member", "emitted"),
    [
        ("\nfoo", "\nfoo"),
        ("foo\n", "foo\n"),
        ("\nfoo", "\n\nfoo\n"),
        ("a\nb", "\na\nb\n"),
    ],
    ids=["bare-leading", "bare-trailing", "wrapped-leading", "wrapped-interior"],
)
def test_enum_member_with_newline_round_trips_through_the_grammar(
    member: str, emitted: str
) -> None:
    """an enum member carrying a newline stays selectable, so it needs no declaration-time ban.

    enums take the schema-directed value path rather than the free-string path, so the wrapper
    rule has to hold there too or the member decodes to a different one and fails validation.
    """
    declaration = _exact_tools()[0].wire()
    declaration["function"]["parameters"]["properties"]["label"]["enum"] = [member]
    tools = normalize_tools([declaration])
    text = _exact_call().replace(
        "<parameter=label>ok</parameter>", f"<parameter=label>{emitted}</parameter>"
    )

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert json.loads(result.calls[0].arguments)["label"] == member


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


@pytest.mark.parametrize("location", ["direct", "nested"])
def test_numeric_enum_accepts_1024_digit_integers(location: str) -> None:
    exact = int("9" * 1024)

    tools = normalize_tools(_numeric_enum_tool(location, exact))

    if location == "direct":
        enum = tools[0].parameters["properties"]["value"]["enum"]
    else:
        enum = tools[0].parameters["properties"]["value"]["properties"]["number"]["enum"]
    assert enum == [exact]


@pytest.mark.parametrize("location", ["direct", "nested"])
def test_numeric_enum_rejects_1025_digit_integers_before_copy(location: str) -> None:
    exact = int("9" * 1025)

    with pytest.raises(ValueError, match="1024-digit limit"):
        normalize_tools(_numeric_enum_tool(location, exact))


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
    fingerprint = request_tool_calls_module._json_value_fingerprint

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


@pytest.mark.parametrize(("width", "accepted"), [(510, True), (511, False)])
def test_enum_nodes_are_bounded_across_tool_declarations(width: int, accepted: bool) -> None:
    first_enum = [[*range(width), 1_000_000 + index] for index in range(64)]
    second_enum = [[*range(width), 2_000_000 + index] for index in range(64)]
    declarations = _enum_tool(first_enum) + _enum_tool(second_enum)
    declarations[1]["function"]["name"] = "store_two"

    if accepted:
        assert len(normalize_tools(declarations)) == 2
    else:
        with pytest.raises(ValueError, match="enum value complexity"):
            normalize_tools(declarations)


def test_enum_limit_precedes_rejected_declaration_copy_and_fingerprint(monkeypatch) -> None:
    first_enum = [[*range(511), 1_000_000 + index] for index in range(64)]
    second_enum = [[*range(511), 2_000_000 + index] for index in range(64)]
    declarations = _enum_tool(first_enum) + _enum_tool(second_enum)
    declarations[1]["function"]["name"] = "store_two"
    copied_first = fingerprinted_first = False
    original_copy = request_tool_calls_module._json_copy
    original_fingerprint = request_tool_calls_module._json_value_fingerprint

    def tracked_copy(value, *args):
        nonlocal copied_first
        if value is second_enum:
            pytest.fail("rejected enum must not be copied")
        copied_first |= value is first_enum
        return original_copy(value, *args)

    def tracked_fingerprint(value):
        nonlocal fingerprinted_first
        if type(value) is list and value and type(value[-1]) is int:
            if value[-1] >= 2_000_000:
                pytest.fail("rejected enum must not be fingerprinted")
            fingerprinted_first |= value[-1] >= 1_000_000
        return original_fingerprint(value)

    monkeypatch.setattr(request_tool_calls_module, "_json_copy", tracked_copy)
    monkeypatch.setattr(request_tool_calls_module, "_json_value_fingerprint", tracked_fingerprint)
    with pytest.raises(ValueError, match="enum value complexity"):
        normalize_tools(declarations)
    assert copied_first
    assert fingerprinted_first


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


def _candidate_string_tools():
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        "a": {"type": "string"},
        "b": {"type": "integer", "enum": [1]},
        "d": {"type": "integer", "enum": [2]},
        "c": {"type": "string"},
    }
    declaration["function"]["parameters"]["required"] = ["a", "c"]
    return normalize_tools([declaration])


def _candidate_call(parameters: str) -> str:
    return f"<tool_call><function=store>{parameters}</function></tool_call>"


def _absorbing_optional_string_tools():
    declaration = _candidate_string_tools()[0].wire()
    declaration["function"]["parameters"]["properties"]["b"] = {"type": "string"}
    return normalize_tools([declaration])


_EMBEDDED_PARAMETER_CASES = [
    (
        "unknown",
        _candidate_call(
            "<parameter=a>before </parameter><parameter=unknown>inside</parameter> after</parameter>"
            "<parameter=c>done</parameter>"
        ),
        {"a": "before </parameter><parameter=unknown>inside</parameter> after", "c": "done"},
    ),
    (
        "consumed-required-before-current",
        _candidate_call(
            "<parameter=c>done</parameter>"
            "<parameter=a>before </parameter><parameter=c>inside</parameter> after</parameter>"
        ),
        {"c": "done", "a": "before </parameter><parameter=c>inside</parameter> after"},
    ),
    (
        "current-parameter",
        _candidate_call(
            "<parameter=a>before </parameter><parameter=a>inside</parameter> after</parameter>"
            "<parameter=c>done</parameter>"
        ),
        {"a": "before </parameter><parameter=a>inside</parameter> after", "c": "done"},
    ),
    (
        "invalid-optional-syntax",
        _candidate_call(
            "<parameter=a>before </parameter><parameter=b>nope</parameter> after</parameter>"
            "<parameter=c>done</parameter>"
        ),
        {"a": "before </parameter><parameter=b>nope</parameter> after", "c": "done"},
    ),
    (
        "invalid-optional-enum",
        _candidate_call(
            "<parameter=a>before </parameter><parameter=b>2</parameter> after</parameter>"
            "<parameter=c>done</parameter>"
        ),
        {"a": "before </parameter><parameter=b>2</parameter> after", "c": "done"},
    ),
    (
        "optional-leaves-required-missing",
        _candidate_call(
            "<parameter=a>before </parameter><parameter=b>1</parameter>"
            "</function></tool_call> after</parameter><parameter=c>done</parameter>"
        ),
        {
            "a": "before </parameter><parameter=b>1</parameter></function></tool_call> after",
            "c": "done",
        },
    ),
    (
        "multiple-optionals-leave-required-missing",
        _candidate_call(
            "<parameter=a>before </parameter><parameter=b>1</parameter>"
            "<parameter=d>2</parameter></function></tool_call> after</parameter>"
            "<parameter=c>done</parameter>"
        ),
        {
            "a": (
                "before </parameter><parameter=b>1</parameter>"
                "<parameter=d>2</parameter></function></tool_call> after"
            ),
            "c": "done",
        },
    ),
]


@pytest.mark.parametrize(
    ("_case", "text", "expected"),
    _EMBEDDED_PARAMETER_CASES,
    ids=[case[0] for case in _EMBEDDED_PARAMETER_CASES],
)
def test_nonstructural_parameter_openers_remain_string_content_buffered(
    _case: str,
    text: str,
    expected: dict[str, object],
) -> None:
    result = parse_qwen3_coder_output(
        text, _candidate_string_tools(), id_factory=lambda: "call_fixed"
    )

    assert json.loads(result.calls[0].arguments) == expected


@pytest.mark.parametrize(
    ("_case", "text", "expected"),
    _EMBEDDED_PARAMETER_CASES,
    ids=[case[0] for case in _EMBEDDED_PARAMETER_CASES],
)
def test_nonstructural_parameter_openers_survive_arbitrary_stream_splits(
    _case: str,
    text: str,
    expected: dict[str, object],
) -> None:
    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(_candidate_string_tools(), id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert json.loads(result.calls[0].arguments) == expected


def test_required_parameter_opener_remains_structural_across_stream_splits() -> None:
    text = _candidate_call("<parameter=a>before</parameter><parameter=c>inside</parameter>")

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(_candidate_string_tools(), id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        assert json.loads(parser.finish().calls[0].arguments) == {"a": "before", "c": "inside"}


def _optional_free_string_absorption_case() -> tuple[str, dict[str, str]]:
    text = _candidate_call(
        "<parameter=a>before </parameter><parameter=b>inside</parameter>"
        "</function></tool_call> boundary <parameter=c>embedded</parameter> after</parameter>"
        "<parameter=c>done</parameter>"
    )
    expected = {
        "a": (
            "before </parameter><parameter=b>inside</parameter>"
            "</function></tool_call> boundary <parameter=c>embedded</parameter> after"
        ),
        "c": "done",
    }
    return text, expected


def _required_free_string_absorption_case():
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        name: {"type": "string"} for name in "abcd"
    }
    declaration["function"]["parameters"]["required"] = list("abcd")
    tools = normalize_tools([declaration])
    text = _candidate_call(
        "<parameter=a>alpha</parameter>"
        "<parameter=b>before </parameter><parameter=c>fake</parameter>"
        "</function></tool_call> after</parameter>"
        "<parameter=c>real-c</parameter><parameter=d>real-d</parameter>"
    )
    expected = {
        "a": "alpha",
        "b": ("before </parameter><parameter=c>fake</parameter></function></tool_call> after"),
        "c": "real-c",
        "d": "real-d",
    }
    return tools, text, expected


def test_required_free_string_absorbs_fake_complete_continuation_buffered() -> None:
    tools, text, expected = _required_free_string_absorption_case()

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert json.loads(result.calls[0].arguments) == expected


def test_required_free_string_absorbs_fake_complete_continuation_across_splits() -> None:
    tools, text, expected = _required_free_string_absorption_case()

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools, id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        assert json.loads(parser.finish().calls[0].arguments) == expected


def test_optional_free_string_cannot_absorb_required_fields_buffered(monkeypatch) -> None:
    text, expected = _optional_free_string_absorption_case()
    original = tool_calls_module._parse_parameter_value
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tool_calls_module, "_parse_parameter_value", counted)
    result = parse_qwen3_coder_output(
        text, _absorbing_optional_string_tools(), id_factory=lambda: "call_fixed"
    )

    assert json.loads(result.calls[0].arguments) == expected
    assert calls <= 4


def test_optional_free_string_cannot_absorb_required_fields_across_splits() -> None:
    text, expected = _optional_free_string_absorption_case()

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(
            _absorbing_optional_string_tools(), id_factory=lambda: "call_fixed"
        )
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        assert json.loads(parser.finish().calls[0].arguments) == expected


def test_complete_optional_free_string_continuation_remains_ambiguous_across_splits() -> None:
    text = _candidate_call(
        "<parameter=c>done</parameter><parameter=a>before </parameter>"
        "<parameter=b>inside</parameter>"
    )

    buffered = parse_qwen3_coder_output(text, _absorbing_optional_string_tools())
    assert buffered.content == text
    assert buffered.calls == ()
    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(_absorbing_optional_string_tools())
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert result.content == text
        assert result.calls == ()


def test_parameter_continuation_probe_has_bounded_linear_work(monkeypatch) -> None:
    count = 128
    declaration = _delimiter_tools()[0].wire()
    properties = {f"p{index}": {"type": "string"} for index in range(count)}
    declaration["function"]["parameters"]["properties"] = properties
    declaration["function"]["parameters"]["required"] = list(properties)
    tools = normalize_tools([declaration])
    text = _candidate_call("".join(f"<parameter={name}>value</parameter>" for name in properties))
    original = tool_calls_module._parse_parameter_value
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tool_calls_module, "_parse_parameter_value", counted)

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert len(result.calls) == 1
    assert calls <= 2 * count


def test_required_impossible_multi_optional_probe_has_bounded_linear_work(monkeypatch) -> None:
    count = 64
    declaration = _delimiter_tools()[0].wire()
    properties = {"a": {"type": "string"}, "c": {"type": "string"}}
    properties.update({f"p{index}": {"type": "integer"} for index in range(count)})
    declaration["function"]["parameters"]["properties"] = properties
    declaration["function"]["parameters"]["required"] = ["a", "c"]
    tools = normalize_tools([declaration])
    embedded = "".join(f"<parameter=p{index}>{index}</parameter>" for index in range(count))
    text = _candidate_call(
        f"<parameter=a>before </parameter>{embedded}</function></tool_call> after</parameter>"
        "<parameter=c>done</parameter>"
    )
    original = tool_calls_module._parse_parameter_value
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tool_calls_module, "_parse_parameter_value", counted)
    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert len(result.calls) == 1
    assert calls <= 2 * count


def _string_field_tools(properties: str, required: str):
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        name: {"type": "string"} for name in properties
    }
    declaration["function"]["parameters"]["required"] = list(required)
    return normalize_tools([declaration])


def _ownership_cases():
    all_required = _string_field_tools("abcde", "abcde")
    all_required_text = _candidate_call(
        "<parameter=a>A</parameter><parameter=b>B "
        "</parameter><parameter=c>fake-c</parameter>"
        "<parameter=d>fake-d</parameter></function></tool_call> tail</parameter>"
        "<parameter=c>real-c</parameter><parameter=d>real-d</parameter>"
        "<parameter=e>real-e</parameter>"
    )
    optional_b = _string_field_tools("bcd", "cd")
    optional_b_text = _candidate_call(
        "<parameter=b>B </parameter><parameter=c>fake-c</parameter>"
        "</function></tool_call> tail</parameter>"
        "<parameter=c>real-c</parameter><parameter=d>real-d</parameter>"
    )
    optional_a = _string_field_tools("abc", "bc")
    ambiguous_text = _candidate_call(
        "<parameter=a>A </parameter><parameter=b>fake-b</parameter>"
        "<parameter=c>fake-c</parameter></function></tool_call> tail</parameter>"
        "<parameter=b>real-b</parameter><parameter=c>real-c</parameter>"
    )
    required_abcd = _string_field_tools("abcd", "abcd")
    required_abcd_text = _candidate_call(
        "<parameter=a>A</parameter><parameter=b>B "
        "</parameter><parameter=c>fake-c</parameter>"
        "</function></tool_call> tail</parameter>"
        "<parameter=c>real-c</parameter><parameter=d>real-d</parameter>"
    )
    required_abc = _string_field_tools("abc", "abc")
    resumed_missing_text = _candidate_call(
        "<parameter=a></parameter>"
        "<parameter=b></parameter></function></tool_call></parameter>"
        "<parameter=c></parameter><parameter=b></parameter>"
    )
    return [
        (
            "all-required-fake-cd",
            all_required,
            all_required_text,
            {
                "a": "A",
                "b": (
                    "B </parameter><parameter=c>fake-c</parameter>"
                    "<parameter=d>fake-d</parameter></function></tool_call> tail"
                ),
                "c": "real-c",
                "d": "real-d",
                "e": "real-e",
            },
        ),
        (
            "optional-b-incomplete-close",
            optional_b,
            optional_b_text,
            {
                "b": ("B </parameter><parameter=c>fake-c</parameter></function></tool_call> tail"),
                "c": "real-c",
                "d": "real-d",
            },
        ),
        ("optional-a-two-valid-assignments", optional_a, ambiguous_text, None),
        (
            "required-b-incomplete-close",
            required_abcd,
            required_abcd_text,
            {
                "a": "A",
                "b": ("B </parameter><parameter=c>fake-c</parameter></function></tool_call> tail"),
                "c": "real-c",
                "d": "real-d",
            },
        ),
        ("required-b-resumes-after-nested-close", required_abc, resumed_missing_text, None),
    ]


@pytest.mark.parametrize(
    ("_case", "tools", "text", "expected"),
    _ownership_cases(),
    ids=[case[0] for case in _ownership_cases()],
)
def test_free_string_delimiter_ownership_is_exact_buffered(
    _case: str,
    tools,
    text: str,
    expected: dict[str, str] | None,
) -> None:
    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    if expected is None:
        assert result.content == text
        assert result.calls == ()
    else:
        assert result.content is None
        assert json.loads(result.calls[0].arguments) == expected


@pytest.mark.parametrize(
    ("_case", "tools", "text", "expected"),
    _ownership_cases(),
    ids=[case[0] for case in _ownership_cases()],
)
def test_free_string_delimiter_ownership_survives_every_two_chunk_split(
    _case: str,
    tools,
    text: str,
    expected: dict[str, str] | None,
) -> None:
    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools, id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        if expected is None:
            assert result.content == text
            assert result.calls == ()
        else:
            assert result.content is None
            assert json.loads(result.calls[0].arguments) == expected


def test_third_nested_ownership_decision_falls_back_exactly() -> None:
    tools = _string_field_tools("abcde", "abcde")
    text = _candidate_call(
        "<parameter=a>A </parameter><parameter=b>fake-b "
        "</parameter><parameter=c>fake-c "
        "</parameter><parameter=d>fake-d</parameter>"
        "</function></tool_call> fake-c-tail</parameter>"
        "<parameter=d>nested-real-d</parameter><parameter=e>nested-e</parameter>"
        "</function></tool_call> fake-b-tail</parameter>"
        "<parameter=c>real-c</parameter><parameter=d>real-d</parameter>"
        "<parameter=e>real-e</parameter></function></tool_call> fake-a-tail</parameter>"
        "<parameter=b>real-b</parameter><parameter=c>outer-c</parameter>"
        "<parameter=d>outer-d</parameter><parameter=e>outer-e</parameter>"
    )

    result = parse_qwen3_coder_output(text, tools)

    assert result.content == text
    assert result.calls == ()


def _repeated_fake_continuation_case(repeats: int):
    tools = _string_field_tools("abcde", "abcde")
    fake = "".join(
        f"chunk-{index}</parameter><parameter=c>fake-c-{index}</parameter>"
        f"<parameter=d>fake-d-{index}</parameter></function></tool_call> tail-{index} "
        for index in range(repeats)
    )
    text = _candidate_call(
        f"<parameter=a>A</parameter><parameter=b>{fake}</parameter>"
        "<parameter=c>real-c</parameter><parameter=d>real-d</parameter>"
        "<parameter=e>real-e</parameter>"
    )
    return tools, text, fake


def test_repeated_complete_fake_continuations_preserve_exact_string_bytes() -> None:
    tools, text, fake = _repeated_fake_continuation_case(12)

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert json.loads(result.calls[0].arguments) == {
        "a": "A",
        "b": fake,
        "c": "real-c",
        "d": "real-d",
        "e": "real-e",
    }


def test_repeated_fake_continuation_work_is_linear_without_prefix_materialization(
    monkeypatch,
) -> None:
    originals = {
        name: getattr(tool_calls_module, name)
        for name in (
            "_find_parameter_end",
            "_materialize_span",
            "_coerce_value",
            "_validate_value",
            "_parse_parameter_value",
        )
    }
    counters = dict.fromkeys(originals, 0)

    def counted(name):
        def wrapper(*args, **kwargs):
            counters[name] += 1
            return originals[name](*args, **kwargs)

        return wrapper

    for name in originals:
        monkeypatch.setattr(tool_calls_module, name, counted(name))

    measurements = {}
    for repeats in (32, 64):
        counters.update(dict.fromkeys(counters, 0))
        tools, text, fake = _repeated_fake_continuation_case(repeats)
        result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")
        assert json.loads(result.calls[0].arguments)["b"] == fake
        measurements[repeats] = dict(counters)

    for measured in measurements.values():
        assert measured["_coerce_value"] == 0
        assert measured["_materialize_span"] == 5
        assert measured["_validate_value"] == 6
    work_32 = measurements[32]["_find_parameter_end"] + measurements[32]["_parse_parameter_value"]
    work_64 = measurements[64]["_find_parameter_end"] + measurements[64]["_parse_parameter_value"]
    assert work_64 <= 2 * work_32


def _wide_array_tools():
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        "values": {"type": "array", "items": {"type": "integer"}}
    }
    declaration["function"]["parameters"]["required"] = ["values"]
    return normalize_tools([declaration])


def _wide_array_call(count: int) -> str:
    values = json.dumps(list(range(count)), separators=(",", ":"))
    return _candidate_call(f"<parameter=values>{values}</parameter>")


def _round_trip_history(call) -> list[dict[str, object]]:
    return [
        {"role": "assistant", "content": None, "tool_calls": [call.wire()]},
        {"role": "tool", "tool_call_id": call.id, "content": "ok"},
    ]


def test_generated_tool_arguments_at_history_complexity_boundary_round_trip() -> None:
    result = parse_qwen3_coder_output(
        _wide_array_call(510), _wide_array_tools(), id_factory=lambda: "call_fixed"
    )

    assert result.content is None
    validate_tool_history(_round_trip_history(result.calls[0]))


def test_generated_tool_arguments_over_history_complexity_fall_back_exactly() -> None:
    text = _wide_array_call(600)

    result = parse_qwen3_coder_output(text, _wide_array_tools())

    assert result.content == text
    assert result.calls == ()


def test_over_complex_generated_tool_arguments_fall_back_across_stream_splits() -> None:
    text = _wide_array_call(600)

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(_wide_array_tools(), id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert result.content == text
        assert result.calls == ()


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


def test_complete_multi_optional_continuation_remains_ambiguous_across_splits() -> None:
    text = _candidate_call(
        "<parameter=c>done</parameter><parameter=a>before </parameter>"
        "<parameter=b>1</parameter><parameter=d>2</parameter>"
    )

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(_candidate_string_tools())
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert result.content == text
        assert result.calls == ()


def _multi_call_scope_case(*, malformed_first: bool) -> tuple[object, str]:
    declaration = _string_field_tools("abz", "abz")[0].wire()
    declaration["function"]["parameters"]["properties"]["z"] = {"type": "integer"}
    tools = normalize_tools([declaration])
    first = "<parameter=a>first</parameter><parameter=b>first-b</parameter>"
    if not malformed_first:
        first += "<parameter=z>1</parameter>"
    second = (
        "<parameter=z>2</parameter><parameter=a>second-a</parameter>"
        "<parameter=b>second-b</parameter>"
    )
    text = _candidate_call(first) + _candidate_call(second)
    return tools, text


def test_multiple_calls_use_only_their_own_parameter_openers_buffered() -> None:
    tools, text = _multi_call_scope_case(malformed_first=False)

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert [json.loads(call.arguments) for call in result.calls] == [
        {"a": "first", "b": "first-b", "z": 1},
        {"z": 2, "a": "second-a", "b": "second-b"},
    ]


def test_multiple_calls_use_only_their_own_parameter_openers_across_splits() -> None:
    tools, text = _multi_call_scope_case(malformed_first=False)

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools, id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        assert len(parser.finish().calls) == 2


def _one_or_two_call_ambiguity_case() -> tuple[object, str]:
    tools = _string_field_tools("ab", "ab")
    text = _candidate_call(
        "<parameter=a>before </parameter><parameter=b>fake-b</parameter>"
    ) + _candidate_call("<parameter=a>apparent-a</parameter><parameter=b>real-b</parameter>")
    return tools, text


def test_valid_later_call_and_merged_prefix_ambiguity_falls_back_buffered() -> None:
    tools, text = _one_or_two_call_ambiguity_case()

    result = parse_qwen3_coder_output(text, tools)

    assert result.content == text
    assert result.calls == ()


def test_valid_later_call_and_merged_prefix_ambiguity_falls_back_across_splits() -> None:
    tools, text = _one_or_two_call_ambiguity_case()

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools)
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert result.content == text
        assert result.calls == ()


def _three_call_wider_scope_ambiguity_case() -> tuple[object, str]:
    def declaration(name: str, properties: dict[str, object], required: list[str]):
        return {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    edge_properties = {
        "a": {"type": "string"},
        "z": {"type": "integer", "enum": [1]},
    }
    tools = normalize_tools(
        [
            declaration("outer", edge_properties, ["a", "z"]),
            declaration("middle", {"m": {"type": "integer"}}, ["m"]),
            declaration("last", edge_properties, ["a", "z"]),
        ]
    )
    text = (
        "<tool_call><function=outer><parameter=a>outer</parameter>"
        "<parameter=z>1</parameter></function></tool_call>"
        "<tool_call><function=middle><parameter=m>2</parameter></function></tool_call>"
        "<tool_call><function=last><parameter=a>last</parameter>"
        "<parameter=z>1</parameter></function></tool_call>"
    )
    return tools, text


def test_first_call_absorbing_two_later_calls_falls_back_buffered() -> None:
    tools, text = _three_call_wider_scope_ambiguity_case()

    result = parse_qwen3_coder_output(text, tools)

    assert result.content == text
    assert result.calls == ()


def test_first_call_absorbing_two_later_calls_falls_back_in_every_two_chunk_split() -> None:
    tools, text = _three_call_wider_scope_ambiguity_case()

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools)
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert result.content == text
        assert result.calls == ()


def _whitespace_separated_multi_call_case() -> tuple[object, str]:
    tools, text = _multi_call_scope_case(malformed_first=False)
    return tools, text.replace(
        "</function></tool_call><tool_call>",
        "</function> \n </tool_call> \n <tool_call>",
        1,
    )


def test_multiple_calls_allow_whitespace_between_closing_tags_buffered() -> None:
    tools, text = _whitespace_separated_multi_call_case()

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert [json.loads(call.arguments) for call in result.calls] == [
        {"a": "first", "b": "first-b", "z": 1},
        {"z": 2, "a": "second-a", "b": "second-b"},
    ]


def test_multiple_calls_allow_whitespace_between_closing_tags_in_every_two_chunk_split() -> None:
    tools, text = _whitespace_separated_multi_call_case()

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools, id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        assert [json.loads(call.arguments) for call in parser.finish().calls] == [
            {"a": "first", "b": "first-b", "z": 1},
            {"z": 2, "a": "second-a", "b": "second-b"},
        ]


_EMBEDDED_TOOL_MARKERS = (
    "<tool_call><function=store> inside",
    "</function></tool_call><tool_call> inside",
    "</function></tool_call><tool_call><function=store inside",
    "</function></tool_call><tool_call><function=store> inside",
    "</function></tool_call><tool_call><function=unknown> inside",
)


def _embedded_tool_call_marker_case(marker: str) -> tuple[object, str]:
    tools = _string_field_tools("abc", "abc")
    text = _candidate_call(
        f"<parameter=a>before {marker}</parameter>"
        "<parameter=b>real-b</parameter><parameter=c>real-c</parameter>"
    )
    return tools, text


@pytest.mark.parametrize("marker", _EMBEDDED_TOOL_MARKERS)
def test_embedded_tool_call_marker_remains_free_string_content_buffered(marker: str) -> None:
    tools, text = _embedded_tool_call_marker_case(marker)

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert json.loads(result.calls[0].arguments) == {
        "a": f"before {marker}",
        "b": "real-b",
        "c": "real-c",
    }


@pytest.mark.parametrize("marker", _EMBEDDED_TOOL_MARKERS)
def test_embedded_tool_call_marker_remains_free_string_content_across_splits(marker: str) -> None:
    tools, text = _embedded_tool_call_marker_case(marker)

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools, id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        assert f"before {marker}" in parser.finish().calls[0].arguments


_INVALID_DECLARED_CALLS = (
    "<parameter=unknown>x</parameter></function></tool_call> fake-tail",
    "<parameter=b>incomplete",
    "</function></tool_call> fake-tail",
)


def _invalid_declared_call_in_string_case(fake_call: str) -> tuple[object, str, str]:
    tools = _string_field_tools("abc", "abc")
    marker = "</function> \n </tool_call> \n <tool_call><function=store>" + fake_call
    expected = f"before {marker}"
    text = _candidate_call(
        f"<parameter=a>{expected}</parameter>"
        "<parameter=b>real-b</parameter><parameter=c>real-c</parameter>"
    )
    return tools, text, expected


@pytest.mark.parametrize("fake_call", _INVALID_DECLARED_CALLS)
def test_invalid_declared_call_remains_free_string_content_buffered(fake_call: str) -> None:
    tools, text, expected = _invalid_declared_call_in_string_case(fake_call)

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert len(result.calls) == 1
    assert json.loads(result.calls[0].arguments)["a"] == expected


@pytest.mark.parametrize("fake_call", _INVALID_DECLARED_CALLS)
def test_invalid_declared_call_remains_free_string_content_in_every_two_chunk_split(
    fake_call: str,
) -> None:
    tools, text, expected = _invalid_declared_call_in_string_case(fake_call)

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools, id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        assert json.loads(parser.finish().calls[0].arguments)["a"] == expected


def test_later_call_cannot_complete_a_malformed_first_call_buffered() -> None:
    tools, text = _multi_call_scope_case(malformed_first=True)

    result = parse_qwen3_coder_output(text, tools)

    assert result.content == text
    assert result.calls == ()


def test_later_call_cannot_complete_a_malformed_first_call_across_splits() -> None:
    tools, text = _multi_call_scope_case(malformed_first=True)

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools)
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert result.content == text
        assert result.calls == ()


def _optional_consumption_ambiguity_case() -> tuple[object, str]:
    tools = _string_field_tools("abc", "ab")
    text = _candidate_call(
        "<parameter=a></parameter><parameter=c></parameter><parameter=b><parameter=b></parameter>"
    )
    return tools, text


def test_optional_consumption_enumerates_both_valid_assignments_buffered() -> None:
    tools, text = _optional_consumption_ambiguity_case()

    result = parse_qwen3_coder_output(text, tools)

    assert result.content == text
    assert result.calls == ()


def test_optional_consumption_enumerates_both_valid_assignments_across_splits() -> None:
    tools, text = _optional_consumption_ambiguity_case()

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools)
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert result.content == text
        assert result.calls == ()


def test_many_calls_scan_only_bounded_call_ranges(monkeypatch) -> None:
    declaration = _string_field_tools("ab", "ab")[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        "a": {"type": "integer"},
        "b": {"type": "integer"},
    }
    tools = normalize_tools([declaration])
    original = tool_calls_module._PARAMETER_OPEN_RE
    examined = 0

    class MeasuredOpeners:
        def finditer(self, text: str, start: int, end: int):
            nonlocal examined
            examined += end - start
            return original.finditer(text, start, end)

    monkeypatch.setattr(tool_calls_module, "_PARAMETER_OPEN_RE", MeasuredOpeners())
    measurements = {}
    for count in (32, 64, 128):
        examined = 0
        text = "".join(
            _candidate_call(f"<parameter=a>{index}</parameter><parameter=b>{index}</parameter>")
            for index in range(count)
        )
        result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")
        assert len(result.calls) == count
        measurements[count] = examined

    assert measurements[64] <= 2 * measurements[32] + 64
    assert measurements[128] <= 2 * measurements[64] + 128


def _integer_calls(count: int) -> tuple[object, str]:
    declaration = _string_field_tools("a", "a")[0].wire()
    declaration["function"]["parameters"]["properties"] = {"a": {"type": "integer"}}
    tools = normalize_tools([declaration])
    text = "".join(_candidate_call(f"<parameter=a>{index}</parameter>") for index in range(count))
    return tools, text


def test_exactly_512_calls_remain_supported() -> None:
    tools, text = _integer_calls(512)
    next_id = iter(f"call_{index}" for index in range(512)).__next__

    result = parse_qwen3_coder_output(text, tools, id_factory=next_id)

    assert result.content is None
    assert len(result.calls) == 512
    assert result.calls[-1].id == "call_511"


def test_513_calls_reject_before_call_id_creation() -> None:
    tools, text = _integer_calls(513)
    id_calls = 0

    def make_id() -> str:
        nonlocal id_calls
        id_calls += 1
        return "unexpected"

    result = parse_qwen3_coder_output(text, tools, id_factory=make_id)

    assert result.content == text
    assert result.calls == ()
    assert id_calls == 0


def test_dense_invalid_declared_candidates_have_bounded_segmentation_work(monkeypatch) -> None:
    tools = _string_field_tools("abc", "abc")
    examined = 0
    original = tool_calls_module._parse_tool_call

    def measured(text, start, scope_end, tool_map, opener_positions, work):
        nonlocal examined
        examined += scope_end - start
        return original(text, start, scope_end, tool_map, opener_positions, work)

    monkeypatch.setattr(tool_calls_module, "_parse_tool_call", measured)
    fake = "".join(
        "chunk</function></tool_call><tool_call><function=store>"
        "<parameter=unknown>x</parameter></function></tool_call>tail"
        for _ in range(128)
    )
    text = _candidate_call(
        f"<parameter=a>{fake}</parameter><parameter=b>b</parameter><parameter=c>c</parameter>"
    )

    result = parse_qwen3_coder_output(text, tools)

    assert result.content == text
    assert result.calls == ()
    assert examined <= 4 * len(text) + len(text)


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
