"""exact numeric and delimiter ownership regressions for tool calls."""

from __future__ import annotations

import base64
import io
import json
import random
import re
import sys
import time
import tracemalloc
from array import array
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from PIL import Image
from pydantic import ValidationError

import flash.serve.request.openai as openai_module
import flash.serve.request.text_scan as text_scan
import flash.serve.request.tool_calls as request_tool_calls_module
import flash.serve.runtime.tool_calls as tool_calls_module
from flash.serve.contract.protocol import MAX_CHAT_REQUEST_BYTES
from flash.serve.request.openai import OpenAIRequestError, parse_chat_request
from flash.serve.request.tool_calls import (
    FunctionTool,
    detached_template_messages,
    normalize_tools,
    tools_wire,
    validate_tool_history,
    validate_tool_history_replay,
    validate_tool_stop_sequences,
)
from flash.serve.request.validation import detached_messages
from flash.serve.runtime.tool_calls import ToolCallStreamParser, parse_qwen3_coder_output
from flash.serving.src.io.openai_request import OpenAIGenerateRequest


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


def test_non_string_schema_keyword_uses_the_requested_validation_error() -> None:
    declaration = _exact_tools()[0].wire()
    declaration["function"]["parameters"][5] = "boom"

    with pytest.raises(ValueError, match=r"unsupported schema keyword\(s\): 5"):
        normalize_tools([declaration])


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


def test_integer_schema_preserves_generated_negative_zero_without_leaking_float() -> None:
    text = _exact_call().replace(
        "<parameter=count>2</parameter>", "<parameter=count>-0</parameter>"
    )

    result = parse_qwen3_coder_output(text, _exact_tools(), id_factory=lambda: "call_fixed")
    parser = ToolCallStreamParser(_exact_tools(), id_factory=lambda: "call_fixed")
    assert all(parser.feed(character) == "" for character in text)
    streamed = parser.finish()

    assert streamed == result
    assert '"count":-0' in result.calls[0].arguments
    decoded = request_tool_calls_module._load_exact_json(result.calls[0].arguments)
    assert type(decoded["count"]) is Decimal
    assert decoded["count"].is_zero()
    assert decoded["count"].is_signed()


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


@pytest.mark.parametrize("raw_value", ["1e1000001", "1e-1000001"])
def test_out_of_contract_decimal_exponents_fall_back_exactly(raw_value: str) -> None:
    text = _exact_call().replace("9007199254740993.0</parameter>", f"{raw_value}</parameter>", 1)

    result = parse_qwen3_coder_output(text, _exact_tools())

    assert result.content == text
    assert result.calls == ()


@pytest.mark.parametrize("raw_value", ["1e1000000", "1e-1000000"])
def test_decimal_exponent_magnitude_boundary_remains_structured(raw_value: str) -> None:
    text = _exact_call().replace("9007199254740993.0</parameter>", f"{raw_value}</parameter>", 1)

    result = parse_qwen3_coder_output(text, _exact_tools(), id_factory=lambda: "call_fixed")

    assert result.content is None
    assert json.loads(result.calls[0].arguments, parse_float=Decimal)["scalar"] == Decimal(
        raw_value
    )


def test_serving_contract_documents_the_numeric_exponent_safety_bound() -> None:
    contract = (Path(__file__).resolve().parents[1] / "docs/serving-contract.md").read_text()

    assert "exponent\nmagnitude at most 1,000,000" in contract
    assert "history outside either bound is rejected" in contract
    assert "Exponent magnitude is not\npart of this bound" not in contract


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


@pytest.mark.parametrize(
    "codepoint",
    # the surrogate range and the codepoint on either side of it. the check is one native scan
    # for exactly the range utf-8 refuses, so the equivalence has to hold at those precise edges.
    # these are codepoints rather than literals because a lone surrogate cannot be written as a
    # distinguishable source character.
    [0xD7FF, 0xD800, 0xDBFF, 0xDC00, 0xDFFF, 0xE000],
)
def test_the_surrogate_boundary_matches_what_utf_8_refuses_to_encode(codepoint: int) -> None:
    declaration = _unicode_declaration("function-description", f"before{chr(codepoint)}after")

    if 0xD800 <= codepoint <= 0xDFFF:
        with pytest.raises(ValueError, match="tools cannot contain an unpaired surrogate"):
            normalize_tools(declaration)
    else:
        assert normalize_tools(declaration)


def test_an_ascii_declaration_is_settled_without_scanning_for_surrogates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the ascii flag settles the common declaration, so the range scan never runs on it.

    the verdict alone cannot show this: scanning every argument returns the same answer, only
    slower, so without a spy the fast path could be dropped and every test would still pass.
    """
    searched: list[str] = []
    real = request_tool_calls_module._SURROGATE_RANGE

    class _CountingRange:
        def search(self, value: str) -> object:
            searched.append(value)
            return real.search(value)

    monkeypatch.setattr(request_tool_calls_module, "_SURROGATE_RANGE", _CountingRange())

    assert normalize_tools(_unicode_declaration("function-description", "plain ascii only"))
    assert searched == []

    # a non-ascii declaration still has to be scanned, or the guard would be skipping real work.
    assert normalize_tools(_unicode_declaration("function-description", "réponse détaillée"))
    assert "réponse détaillée" in searched


@pytest.mark.parametrize("order", [(0xD800, 0xDC00), (0xDC00, 0xD800)])
def test_adjacent_raw_surrogate_code_units_are_still_unpaired(order: tuple[int, int]) -> None:
    # two raw surrogate code units side by side are not a non-bmp character: the pair only exists
    # as an escape that json decoding turns into one real codepoint. an implementation that read
    # an adjacent high and low unit as "paired" would pass every single-codepoint row above while
    # letting a string through that cannot be encoded or rendered.
    adjacent = "".join(chr(unit) for unit in order)
    declaration = _unicode_declaration("function-description", f"before{adjacent}after")

    with pytest.raises(ValueError, match="tools cannot contain an unpaired surrogate"):
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


def _repeated_boundary_call(repeats: int) -> str:
    """one valid call whose own argument repeats the full call-boundary text."""
    boundary = "</function></tool_call><tool_call><function=store>"
    argument = json.dumps({"text": "x" + boundary * repeats})
    return (
        f"<tool_call><function=store><parameter=nested>{argument}</parameter>"
        "</function></tool_call>"
    )


@pytest.mark.parametrize("repeats", [1, 6, 7, 20, 100, 400])
def test_boundary_text_inside_an_argument_parses_with_actual_work_accounting(
    repeats: int,
) -> None:
    """embedded boundary text is cheap when each apparent call body is immediately invalid."""
    text = _repeated_boundary_call(repeats)

    result = parse_qwen3_coder_output(text, _delimiter_tools(), id_factory=lambda: "call_fixed")

    boundary = "</function></tool_call><tool_call><function=store>"
    assert result.content is None
    assert json.loads(result.calls[0].arguments) == {"nested": {"text": "x" + boundary * repeats}}


def test_whitespace_runs_charge_the_shared_parser_budget(monkeypatch) -> None:
    tools = _delimiter_tools()
    calls, spaces = 64, 1024
    text = (f"<tool_call><function=store>{' ' * spaces}</function></tool_call>") * calls
    charged = 0
    original = tool_calls_module._consume_work

    def measured(work: list[int], amount: int) -> bool:
        nonlocal charged
        charged += amount
        return original(work, amount)

    monkeypatch.setattr(tool_calls_module, "_consume_work", measured)
    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    declared = len(tools[0].parameters["properties"])
    budget = 4 * len(text) + calls * declared
    assert result.content == text
    assert result.calls == ()
    assert calls * spaces <= charged <= budget + 1


def test_unterminated_parameter_openers_charge_before_large_parse(monkeypatch) -> None:
    # observe the public parse path, so moving opener discovery outside the shared budget
    # fails on missing accounting rather than scheduler-dependent elapsed time.
    count = 24_000
    text = "<tool_call><function=store>" + "<parameter=" * count
    charges: list[int] = []
    original = tool_calls_module._consume_work

    def measured(work: list[int], amount: int) -> bool:
        charges.append(amount)
        return original(work, amount)

    monkeypatch.setattr(tool_calls_module, "_consume_work", measured)
    result = parse_qwen3_coder_output(text, _delimiter_tools())

    assert result.content == text
    assert result.calls == ()
    assert charges[0] == len(text)
    assert sum(charges) <= 4 * len(text) + 1


def test_property_opener_rebuilds_stop_at_the_shared_parser_budget(monkeypatch) -> None:
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        f"p{index}": {"type": "string"} for index in range(128)
    }
    tools = normalize_tools([declaration])
    text = "<tool_call><function=store></function></tool_call>" * 64
    charged = searches = 0
    original_bisect = tool_calls_module.bisect_left
    original_consume = tool_calls_module._consume_work

    def measured_bisect(positions, scope_end):
        nonlocal searches
        searches += 1
        return original_bisect(positions, scope_end)

    def measured_consume(work: list[int], amount: int) -> bool:
        nonlocal charged
        charged += amount
        return original_consume(work, amount)

    monkeypatch.setattr(tool_calls_module, "bisect_left", measured_bisect)
    monkeypatch.setattr(tool_calls_module, "_consume_work", measured_consume)
    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert result.content == text
    assert result.calls == ()
    # every call scans the declaration once, so the ceiling scales with calls, not with one copy.
    declared = len(declaration["function"]["parameters"]["properties"])
    assert searches <= charged <= 4 * len(text) + 64 * declared + 1


def _history_replay_tools():
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        "x": {"type": "string"},
        "y": {"type": "string"},
    }
    declaration["function"]["parameters"]["required"] = ["x"]
    return normalize_tools([declaration])


def _history_replay_messages(value: str) -> list[dict[str, Any]]:
    return _history_replay_arguments({"x": value})


def _history_replay_arguments(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_fixed",
                    "type": "function",
                    "function": {"name": "store", "arguments": json.dumps(arguments)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_fixed", "content": "ok"},
    ]


def test_long_trivial_history_is_not_rejected_by_the_generation_work_cap() -> None:
    messages = _history_replay_arguments({"x": {"s": "a" * 8_500_000}})

    request = parse_chat_request(
        {"messages": messages},
        require_model=False,
        allow_managed_selectors=True,
    )

    assert len(request.messages[0]["tool_calls"][0]["function"]["arguments"]) > 8_500_000


def test_history_that_renders_far_larger_than_the_request_is_rejected() -> None:
    # a six-character literal renders as 1024 digits, so a request well under the transport cap
    # can describe a rendered turn hundreds of megabytes wide. the ceiling must be measured on
    # the rendered text, not on the request that carried it.
    arguments = "{" + ",".join(f'"f{index}":1e1023' for index in range(511)) + "}"
    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": f"fn{index}", "arguments": arguments},
        }
        for index in range(408)
    ]
    messages = [
        {"role": "assistant", "content": None, "tool_calls": calls},
        *(
            {"role": "tool", "tool_call_id": f"call_{index}", "content": "ok"}
            for index in range(408)
        ),
    ]

    assert len(json.dumps(messages)) < MAX_CHAT_REQUEST_BYTES
    with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
        parse_chat_request(
            {"messages": messages},
            require_model=False,
            allow_managed_selectors=True,
        )


@pytest.mark.parametrize("declared", [10, 149, 150, 300, 511])
def test_minimal_call_parses_under_any_declared_property_count(declared: int) -> None:
    # each call scans its declared parameters once, which the generated text does not pay for.
    # a schema wide enough to outweigh a short call must not silently degrade it to raw text.
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            f"p{index}": {"type": "string"} for index in range(declared)
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )

    result = parse_qwen3_coder_output("<tool_call><function=f></function></tool_call>", tools)

    assert result.tools_called
    assert [call.name for call in result.calls] == ["f"]


@pytest.mark.parametrize("declared", [10, 139, 140, 300, 511])
def test_replay_accepts_every_call_the_parser_emits_under_wide_declarations(
    declared: int,
) -> None:
    # closure: a call the generation parser is willing to emit must be accepted back as history.
    # the replay budget therefore has to grant the same declaration work generation grants.
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            f"p{index}": {"type": "string"} for index in range(declared)
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    emitted = parse_qwen3_coder_output("<tool_call><function=f></function></tool_call>", tools)
    assert emitted.tools_called

    request = parse_chat_request(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fixed",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_fixed", "content": "ok"},
            ],
            "tools": tools_wire(tools),
            "tool_choice": "auto",
        },
        require_model=False,
        allow_managed_selectors=True,
    )

    assert request.messages[0]["tool_calls"][0]["function"]["name"] == "f"


@pytest.mark.parametrize("declared", [10, 128, 300, 511])
@pytest.mark.parametrize("occurrences", [1, 5, 50, 200])
def test_replay_accepts_a_call_whose_argument_quotes_call_boundaries(
    declared: int, occurrences: int
) -> None:
    # closure again, from the other direction: the candidate scan counts boundaries quoted inside
    # this call's own argument, so generation grants declaration work for each of them. replay
    # renders exactly one call, so a replay budget derived from the call count rather than from
    # the text would be the narrower side and would reject what generation just emitted.
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {"type": "string"},
                            **{f"p{index}": {"type": "string"} for index in range(declared - 1)},
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    value = "x" + "</function></tool_call><tool_call><function=store>" * occurrences + "y"
    arguments = json.dumps({"payload": value})

    request = parse_chat_request(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_quoted",
                            "type": "function",
                            "function": {"name": "store", "arguments": arguments},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_quoted", "content": "ok"},
            ],
            "tools": tools_wire(tools),
            "tool_choice": "auto",
        },
        require_model=False,
        allow_managed_selectors=True,
    )

    assert json.loads(request.messages[0]["tool_calls"][0]["function"]["arguments"]) == {
        "payload": value
    }


def test_merged_self_derived_declarations_stay_within_the_schema_budget() -> None:
    # a declared schema is normalized against the node budget, so generation can never expose more
    # than 511 root properties for one function. repeating one name with disjoint keys would
    # otherwise union self-derived probes into a declaration orders of magnitude wider, and the
    # parser charges a declaration scan per call against it.
    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": "f",
                "arguments": json.dumps({f"c{index}_p{field}": field for field in range(255)}),
            },
        }
        for index in range(204)
    ]
    messages = [
        {"role": "assistant", "content": None, "tool_calls": calls},
        *(
            {"role": "tool", "tool_call_id": f"call_{index}", "content": "ok"}
            for index in range(204)
        ),
    ]

    # integer values render unambiguously, so this 52,020-property union is accepted without the
    # cap and rejected with it. that makes the public request path enough to pin the behavior.
    with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
        parse_chat_request(
            {"messages": messages},
            require_model=False,
            allow_managed_selectors=True,
        )


@pytest.mark.parametrize(
    ("content", "accepted"),
    [
        (None, True),
        ("", True),
        ("done", True),
        # only a complete start marker steals the parse. a partial prefix or a stray end tag
        # renders as ordinary text, so rejecting them would refuse turns the model can replay.
        ("<tool_c", True),
        ("</tool_call>", True),
        ("<tool_call>", False),
        ("a <tool_call> b", False),
        # the template splits content on `</think>` and keeps only what follows the last one, so a
        # marker it discards must not reject a turn whose real render replays correctly.
        ("before</think><tool_call></think>after", True),
        ("<think>plan</think><tool_call>", False),
        # supported text blocks are concatenated into that same prefix, including across a split.
        ([{"type": "text", "text": "done"}], True),
        ([{"type": "text", "text": "<tool_call>"}], False),
        ([{"type": "text", "text": "<tool_"}, {"type": "input_text", "text": "call>"}], False),
    ],
)
def test_assistant_content_markers_are_replayed_with_the_call_blocks(
    content: Any, accepted: bool
) -> None:
    # the qwen template renders assistant content immediately before the call blocks, so a start
    # marker in content becomes the parser's first candidate and swallows the whole turn as text.
    # a probe built from the call blocks alone would validate a string the model never sees.
    messages = _history_replay_messages("value")
    messages[0]["content"] = content

    if not accepted:
        with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
            parse_chat_request(
                {"messages": messages}, require_model=False, allow_managed_selectors=True
            )
        return

    request = parse_chat_request(
        {"messages": messages}, require_model=False, allow_managed_selectors=True
    )

    assert request.messages[0]["content"] == content


@pytest.mark.parametrize(("reasoning", "accepted"), [("plan", True), ("<tool_call>", False)])
def test_assistant_reasoning_markers_are_replayed_with_the_call_blocks(
    reasoning: str, accepted: bool
) -> None:
    # the template renders reasoning inside `<think>` ahead of both the answer and the calls, so a
    # marker there steals the parse exactly as one in content does. the leading query matters: the
    # block is rendered only for turns after it, and without one the template refuses the history.
    messages = [{"role": "user", "content": "q"}, *_history_replay_messages("value")]
    messages[1]["content"] = "done"
    messages[1]["reasoning_content"] = reasoning

    if not accepted:
        with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
            parse_chat_request(
                {"messages": messages}, require_model=False, allow_managed_selectors=True
            )
        return

    parse_chat_request({"messages": messages}, require_model=False, allow_managed_selectors=True)


@pytest.mark.parametrize("reasoning", [["<tool_call>"], {"a": "<tool_call>"}, 5, True])
def test_non_string_reasoning_falls_back_to_the_template_content_split(reasoning: Any) -> None:
    # the template gates on `reasoning_content is string`, so it discards a non-string value
    # instead of rendering it and splits the content itself. keying the fallback on absence would
    # skip that split whenever the field merely exists, rejecting a turn the model replays.
    messages = [{"role": "user", "content": "q"}, *_history_replay_messages("value")]
    messages[1]["content"] = "a</think><tool_call></think>b"
    messages[1]["reasoning_content"] = reasoning

    request = parse_chat_request(
        {"messages": messages}, require_model=False, allow_managed_selectors=True
    )

    assert request.messages[1]["reasoning_content"] == reasoning


def test_reasoning_before_the_last_query_does_not_reject_a_replayable_turn() -> None:
    # the template renders the `<think>` block only for turns after the last ordinary user query.
    # an earlier turn's reasoning is dropped, so a marker there cannot steal the parse and must
    # not be validated as though the model would see it.
    earlier = _history_replay_messages("value")
    earlier[0]["content"] = "fine"
    earlier[0]["reasoning_content"] = "<tool_call>"
    later = _history_replay_arguments({"x": "second"})
    later[0]["tool_calls"][0]["id"] = later[1]["tool_call_id"] = "call_second"
    messages = [{"role": "user", "content": "q1"}, *earlier, {"role": "user", "content": "q2"}]
    messages.extend(later)

    request = parse_chat_request(
        {"messages": messages}, require_model=False, allow_managed_selectors=True
    )

    assert request.messages[1]["reasoning_content"] == "<tool_call>"


def _png_data_uri() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _image_block(spelling: str) -> dict[str, Any]:
    if spelling == "image_url":
        return {"type": "image_url", "image_url": {"url": _png_data_uri()}}
    return {"type": spelling, "image": _png_data_uri()}


def _query_span_messages(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """a history whose middle user turn decides whether the first turn's reasoning renders."""
    earlier = _history_replay_messages("value")
    earlier[0]["content"] = "fine"
    earlier[0]["reasoning_content"] = "<tool_call>"
    later = _history_replay_arguments({"x": "second"})
    later[0]["tool_calls"][0]["id"] = later[1]["tool_call_id"] = "call_second"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "q1"},
        *earlier,
        {"role": "user", "content": blocks},
    ]
    messages.extend(later)
    return messages


@pytest.mark.parametrize("spelling", ["image_url", "input_image", "image"])
def test_an_image_after_a_tool_response_keeps_the_turn_an_ordinary_query(spelling: str) -> None:
    # the template renders an image as a placeholder rather than dropping it, so a turn whose text
    # block looks like a synthesized tool response but that ends with an image does not end with
    # `</tool_response>` and still closes the query span. reading only the text blocks would move
    # the span earlier and reject the preceding turn for reasoning the model never sees. every
    # accepted spelling reaches the same placeholder, so each one has to close the span.
    messages = _query_span_messages(
        [
            {"type": "text", "text": "<tool_response>x</tool_response>"},
            _image_block(spelling),
        ]
    )

    request = parse_chat_request(
        {"messages": messages}, require_model=False, allow_managed_selectors=True
    )

    assert request.messages[1]["reasoning_content"] == "<tool_call>"


@pytest.mark.parametrize("spelling", ["image_url", "input_image", "image"])
def test_an_image_between_the_tool_response_tags_does_not_close_the_query_span(
    spelling: str,
) -> None:
    # the macro emits blocks in list order, so a placeholder BETWEEN the tags leaves the turn
    # still starting and ending with them: it stays a synthesized tool response and the earlier
    # reasoning marker does render, which the parser cannot replay. an implementation that
    # rendered the text first and appended images would read this as an ordinary query and accept
    # a turn that really parses to one call rather than two. every spelling canonicalizes to the
    # same placeholder, so order has to hold for each one rather than for the first one tested.
    messages = _query_span_messages(
        [
            {"type": "text", "text": "<tool_response>"},
            _image_block(spelling),
            {"type": "text", "text": "</tool_response>"},
        ]
    )

    with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
        parse_chat_request(
            {"messages": messages}, require_model=False, allow_managed_selectors=True
        )


@pytest.mark.parametrize(
    ("left", "right", "accepted"), [(255, 255, True), (255, 256, True), (256, 256, False)]
)
def test_merged_self_derived_declarations_stay_under_the_property_ceiling(
    left: int, right: int, accepted: bool
) -> None:
    # the union is capped at the same root-property budget a declared schema is normalized to, so
    # a turn that stays under it still replays and one that crosses it does not. integer values
    # render unambiguously, which is what lets a two-call turn walk the boundary through the public
    # request path rather than reaching for the merge helper directly.
    def call(index: int, fields: int) -> dict[str, Any]:
        arguments = {f"c{index}_p{field}": field for field in range(fields)}
        return {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": "f", "arguments": json.dumps(arguments)},
        }

    calls = [call(0, left), call(1, right)]
    payload = {
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": calls},
            *({"role": "tool", "tool_call_id": item["id"], "content": "ok"} for item in calls),
        ]
    }

    if not accepted:
        with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
            parse_chat_request(payload, require_model=False, allow_managed_selectors=True)
        return

    request = parse_chat_request(payload, require_model=False, allow_managed_selectors=True)

    assert [item["function"]["name"] for item in request.messages[0]["tool_calls"]] == ["f", "f"]

    # the hosted envelope is a second entry into the same replay validation, so pin the ceiling
    # there too rather than assuming one caller speaks for both.
    hosted = OpenAIGenerateRequest(
        adapter_id="adapter", generation_id="generation", messages=payload["messages"]
    )

    assert [item["function"]["name"] for item in hosted.messages[0]["tool_calls"]] == ["f", "f"]


def test_hosted_envelope_enforces_the_replay_declaration_ceiling() -> None:
    # the canonical path rejects a union above the schema budget; the hosted envelope must reach
    # the same verdict, so a caller-specific uncapped route cannot appear unnoticed.
    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": "f",
                "arguments": json.dumps({f"c{index}_p{field}": field for field in range(256)}),
            },
        }
        for index in range(2)
    ]

    with pytest.raises(ValidationError, match="cannot be replayed exactly"):
        OpenAIGenerateRequest(
            adapter_id="adapter",
            generation_id="generation",
            messages=[
                {"role": "assistant", "content": None, "tool_calls": calls},
                *({"role": "tool", "tool_call_id": item["id"], "content": "ok"} for item in calls),
            ],
        )


def test_boundaries_quoted_inside_an_argument_do_not_trip_the_call_cap() -> None:
    # the candidate scan is context blind, so a boundary quoted inside a json string looks like a
    # call. the emitted-call ceiling must count real calls, not that estimate.
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "z",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                        "required": ["data"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
    )
    quoted = json.dumps("</function></tool_call><tool_call><function=z>" * 500)
    text = f"<tool_call><function=store><parameter=data>\n{quoted}\n</parameter></function></tool_call>"

    result = parse_qwen3_coder_output(text, tools)

    assert [call.name for call in result.calls] == ["store"]


@pytest.mark.parametrize(("calls", "emitted"), [(408, 408), (409, 0)])
def test_genuine_call_runs_still_stop_at_the_emitted_call_ceiling(calls: int, emitted: int) -> None:
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "z",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )

    result = parse_qwen3_coder_output(
        "<tool_call><function=z></function></tool_call>" * calls, tools
    )

    assert len(result.calls) == emitted


def test_a_run_past_the_call_ceiling_stops_parsing_instead_of_finishing_the_chain() -> None:
    """once the ceiling is passed the verdict is fixed, so the rest must not be parsed.

    `parsed` only grows, so a chain longer than the ceiling is already rejected by its first
    `_MAX_POTENTIALLY_REPLAYABLE_CALLS + 1` calls, and parsing the remainder is work spent on a
    result that cannot change. counting confirmed parses rather than candidate boundaries is what
    makes stopping sound: a single call whose argument quotes many boundaries confirms once.
    """
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "z",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    chain = "<tool_call><function=z></function></tool_call>" * 5_000

    parses = [0]
    real = tool_calls_module._parse_tool_call

    def counting(*args, **kwargs):
        parses[0] += 1
        return real(*args, **kwargs)

    with mock.patch.object(tool_calls_module, "_parse_tool_call", counting):
        result = parse_qwen3_coder_output(chain, tools, _work_limit=None)

    assert result.calls == ()
    assert result.content == chain
    # one parse per call up to the ceiling, then one more that passes it and ends the loop.
    assert parses[0] == tool_calls_module._MAX_POTENTIALLY_REPLAYABLE_CALLS + 1, parses[0]


def _hosted_tool_request(tool_choice: str) -> OpenAIGenerateRequest:
    # history whose replay verdict genuinely depends on the declaration: the argument carries
    # structural delimiters that are ambiguous only when "b" is a declared sibling parameter.
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_fixed",
                    "type": "function",
                    "function": {
                        "name": "store",
                        "arguments": '{"a": "</parameter><parameter=b>v"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_fixed", "content": "ok"},
    ]
    declarations = tools_wire(
        (
            FunctionTool(
                "store",
                None,
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                },
            ),
        )
    )
    return OpenAIGenerateRequest(
        adapter_id="adapter",
        generation_id="generation",
        messages=messages,
        tools=declarations,
        tool_choice=tool_choice,
        parallel_tool_calls=True,
    )


def test_hosted_request_replays_history_without_inactive_declarations() -> None:
    # the hosted envelope must reach the same verdict as the canonical path: a declaration that
    # tool_choice has switched off cannot decide whether past calls replay.
    assert _hosted_tool_request("none").tool_choice == "none"


def test_hosted_request_applies_active_declarations_to_history_replay() -> None:
    # the companion negative case: with the same declaration active, replay is ambiguous and must
    # be rejected. without this, an envelope that ignored declarations entirely would still pass.
    with pytest.raises(ValidationError, match="cannot be replayed exactly"):
        _hosted_tool_request("auto")


def test_hosted_request_accepts_unambiguous_history_under_active_declarations() -> None:
    # the pair above is satisfied by an envelope that blanket-rejects every active tool history,
    # so pin the accepting direction too: an unambiguous call must survive with tools active.
    declarations = tools_wire(
        (
            FunctionTool(
                "store",
                None,
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                    "additionalProperties": False,
                },
            ),
        )
    )

    request = OpenAIGenerateRequest(
        adapter_id="adapter",
        generation_id="generation",
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_plain",
                        "type": "function",
                        "function": {"name": "store", "arguments": '{"a": "v"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_plain", "content": "ok"},
        ],
        tools=declarations,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    assert request.tool_choice == "auto"
    assert request.messages[0]["tool_calls"][0]["function"]["arguments"] == '{"a": "v"}'


@pytest.mark.parametrize(
    "value",
    [
        "plain",
        "a</function></tool_call>b",
        "</parameter>",
        "<parameter=y>nested",
        "a</parameter>b",
        "</parameter><parameter=x>same",
        "</parameter><parameter=unknown>ignored",
    ],
)
def test_scalar_history_accepts_only_structural_text_that_roundtrips(value: str) -> None:
    tools = _history_replay_tools()
    request = parse_chat_request(
        {"messages": _history_replay_messages(value), "tools": tools_wire(tools)},
        require_model=False,
        allow_managed_selectors=True,
    )

    rendered = detached_template_messages(request.messages)[0]["tool_calls"][0]["function"][
        "arguments"
    ]["x"]
    text = f"<tool_call><function=store><parameter=x>{rendered}</parameter></function></tool_call>"
    reparsed = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")
    assert json.loads(reparsed.calls[0].arguments)["x"] == value


def test_scalar_history_accepts_declared_parameter_text_after_that_parameter_was_consumed() -> None:
    value = "</parameter><parameter=y>spoof"
    tools = _history_replay_tools()
    messages = _history_replay_messages(value)
    messages[0]["tool_calls"][0]["function"]["arguments"] = json.dumps({"y": "already", "x": value})

    request = parse_chat_request(
        {"messages": messages, "tools": tools_wire(tools)},
        require_model=False,
        allow_managed_selectors=True,
    )

    rendered = detached_template_messages(request.messages)[0]["tool_calls"][0]["function"][
        "arguments"
    ]
    text = (
        f"<tool_call><function=store><parameter=y>{rendered['y']}</parameter>"
        f"<parameter=x>{rendered['x']}</parameter></function></tool_call>"
    )
    reparsed = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")
    assert json.loads(reparsed.calls[0].arguments) == {"y": "already", "x": value}


def test_scalar_history_roundtrip_is_independent_of_argument_order() -> None:
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        "x": {"type": "string"},
        "z": {"type": "integer"},
    }
    declaration["function"]["parameters"]["required"] = ["x", "z"]
    tools = normalize_tools([declaration])

    for arguments in (
        {"x": "a</parameter>b", "z": 1},
        {"z": 1, "x": "a</parameter>b"},
    ):
        messages = _history_replay_messages("unused")
        messages[0]["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
        request = parse_chat_request(
            {"messages": messages, "tools": tools_wire(tools)},
            require_model=False,
            allow_managed_selectors=True,
        )
        assert request.messages[0]["tool_calls"][0]["function"]["arguments"] == json.dumps(
            arguments
        )


def test_scalar_history_rejects_declared_parameter_injection_that_does_not_roundtrip() -> None:
    value = "</parameter><parameter=y>spoof"
    tools = _history_replay_tools()
    text = f"<tool_call><function=store><parameter=x>{value}</parameter></function></tool_call>"
    reparsed = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")
    assert reparsed.calls == ()
    assert reparsed.content == text

    with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
        parse_chat_request(
            {"messages": _history_replay_messages(value), "tools": tools_wire(tools)},
            require_model=False,
            allow_managed_selectors=True,
        )


def _measure_history_replay_validation(monkeypatch) -> list[None]:
    # patch the public entry point where the request path binds it, rather than a private
    # helper, so reverting the production change fails these cases on replay behavior instead
    # of on a renamed private symbol.
    calls: list[None] = []
    original = openai_module.validate_tool_history_replay

    def measured(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(openai_module, "validate_tool_history_replay", measured)
    return calls


@pytest.mark.parametrize("declaration", ["none", "matched", "unmatched"])
def test_parser_emitted_scalar_history_replays_with_any_current_declaration(
    declaration: str,
    monkeypatch,
) -> None:
    original = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    text = (
        "<tool_call><function=store><parameter=x>"
        "a</function></tool_call>b</parameter></function></tool_call>"
    )
    parsed = parse_qwen3_coder_output(text, original, id_factory=lambda: "call_fixed")
    messages = _round_trip_history(parsed.calls[0])
    payload: dict[str, Any] = {"messages": messages}
    if declaration == "matched":
        payload["tools"] = tools_wire(_history_replay_tools())
    elif declaration == "unmatched":
        unmatched = _history_replay_tools()[0].wire()
        unmatched["function"]["name"] = "other"
        payload["tools"] = [unmatched]

    validation_calls = _measure_history_replay_validation(monkeypatch)
    request = parse_chat_request(
        payload,
        require_model=False,
        allow_managed_selectors=True,
    )

    assert validation_calls == [None]
    assert json.loads(request.messages[0]["tool_calls"][0]["function"]["arguments"]) == {
        "x": "a</function></tool_call>b"
    }


def test_tools_none_rejects_history_that_fails_its_self_derived_probe() -> None:
    with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
        parse_chat_request(
            {
                "messages": _history_replay_arguments(
                    {"x": "</parameter><parameter=y>spoof", "y": "actual"}
                )
            },
            require_model=False,
            allow_managed_selectors=True,
        )


def test_tool_choice_none_replays_history_without_inactive_declarations() -> None:
    declaration = _history_replay_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {"new": {"type": "string"}}
    declaration["function"]["parameters"]["required"] = []
    messages = _history_replay_arguments({"old": "x"})

    without_tools = parse_chat_request(
        {"messages": messages},
        require_model=False,
        allow_managed_selectors=True,
    )
    inactive = parse_chat_request(
        {
            "messages": messages,
            "tools": [declaration],
            "tool_choice": "none",
            "parallel_tool_calls": True,
        },
        require_model=False,
        allow_managed_selectors=True,
    )

    assert inactive.messages == without_tools.messages


@pytest.mark.parametrize(
    "value",
    ["plain", "</parameter>", "<parameter=y>nested", "a</parameter>b"],
)
@pytest.mark.parametrize("declaration", ["none", "matched", "unmatched"])
def test_scalar_history_replay_closure_with_any_current_declaration(
    value: str,
    declaration: str,
    monkeypatch,
) -> None:
    payload: dict[str, Any] = {"messages": _history_replay_messages(value)}
    if declaration == "matched":
        payload["tools"] = tools_wire(_history_replay_tools())
    elif declaration == "unmatched":
        unmatched = _history_replay_tools()[0].wire()
        unmatched["function"]["name"] = "other"
        payload["tools"] = [unmatched]

    validation_calls = _measure_history_replay_validation(monkeypatch)
    request = parse_chat_request(
        payload,
        require_model=False,
        allow_managed_selectors=True,
    )

    assert validation_calls == [None]
    assert json.loads(request.messages[0]["tool_calls"][0]["function"]["arguments"]) == {"x": value}


def _repeated_call_messages(count: int, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": "store", "arguments": json.dumps(arguments)},
        }
        for index in range(count)
    ]
    return [
        {"role": "assistant", "content": None, "tool_calls": calls},
        *({"role": "tool", "tool_call_id": call["id"], "content": "ok"} for call in calls),
    ]


@pytest.mark.parametrize("declaration", ["none", "matched"])
def test_repeated_calls_that_the_template_renders_ambiguously_are_rejected(
    declaration: str,
) -> None:
    # two identical two-parameter calls render as one block whose second call supplies a
    # competing assignment for the first, so the parser recovers no calls at all. validating
    # each call on its own cannot see that, because the ambiguity only exists across calls.
    payload: dict[str, Any] = {"messages": _repeated_call_messages(2, {"x": "plain", "y": "plain"})}
    if declaration == "matched":
        payload["tools"] = tools_wire(_history_replay_tools())

    with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
        parse_chat_request(payload, require_model=False, allow_managed_selectors=True)


@pytest.mark.parametrize("count", [2, 3, 5, 10])
@pytest.mark.parametrize("declaration", ["none", "unmatched", "matched"])
def test_repeated_unambiguous_calls_still_replay(count: int, declaration: str) -> None:
    # the turn-level probe must reject only genuine ambiguity. a single declared parameter
    # leaves no competing scope, so the parser reproduces every call and replay closure holds.
    messages = _repeated_call_messages(count, {"x": "plain"})
    payload: dict[str, Any] = {"messages": messages}
    if declaration == "matched":
        declared = _history_replay_tools()[0].wire()
        declared["function"]["parameters"]["properties"].pop("y", None)
        declared["function"]["parameters"]["required"] = ["x"]
        payload["tools"] = [declared]
    elif declaration == "unmatched":
        payload["tools"] = tools_wire(_unmatched_replay_tools())

    request = parse_chat_request(payload, require_model=False, allow_managed_selectors=True)

    assert len(request.messages[0]["tool_calls"]) == count


def test_repeated_calls_with_different_optional_arguments_replay() -> None:
    # one turn may call the same function twice with different optional parameters. each call
    # derives its own probe, so the turn-level parse needs both key sets merged under that
    # name. keeping one probe per name would reject a turn the parser emits.
    declared = _history_replay_tools()[0].wire()
    declared["function"]["parameters"]["required"] = []
    tools = normalize_tools([declared])
    text = (
        "<tool_call><function=store><parameter=x>\none\n</parameter></function></tool_call>"
        "<tool_call><function=store><parameter=y>\ntwo\n</parameter></function></tool_call>"
    )
    emitted = parse_qwen3_coder_output(text, tools)
    assert [call.arguments for call in emitted.calls] == ['{"x":"one"}', '{"y":"two"}']

    request = _replay_emitted_calls(emitted.calls, tools=None)

    assert len(request.messages[0]["tool_calls"]) == 2


def _replay_emitted_calls(calls, *, tools=None):
    history_calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
        for index, call in enumerate(calls)
    ]
    messages = [
        {"role": "assistant", "content": None, "tool_calls": history_calls},
        *({"role": "tool", "tool_call_id": call["id"], "content": "ok"} for call in history_calls),
    ]
    payload: dict[str, Any] = {"messages": messages}
    if tools is not None:
        payload.update(tools=tools_wire(tools), tool_choice="auto", parallel_tool_calls=True)
    return parse_chat_request(
        payload,
        require_model=False,
        allow_managed_selectors=True,
    )


def _unmatched_replay_tools():
    declaration = _history_replay_tools()[0].wire()
    declaration["function"]["name"] = "other"
    return normalize_tools([declaration])


@pytest.mark.parametrize("reverse", [False, True], ids=["emitted-order", "reversed-order"])
@pytest.mark.parametrize("declaration", ["none", "unmatched", "matched"])
def test_same_property_integer_and_number_probes_widen_without_order_dependence(
    reverse: bool,
    declaration: str,
) -> None:
    declared = _history_replay_tools()[0].wire()
    declared["function"]["parameters"]["properties"] = {"x": {"type": "number"}}
    declared["function"]["parameters"]["required"] = ["x"]
    tools = normalize_tools([declared])
    text = (
        "<tool_call><function=store><parameter=x>1.5</parameter></function></tool_call>"
        "<tool_call><function=store><parameter=x>1</parameter></function></tool_call>"
    )
    emitted = parse_qwen3_coder_output(text, tools)
    calls = emitted.calls[::-1] if reverse else emitted.calls
    current = (
        None
        if declaration == "none"
        else _unmatched_replay_tools()
        if declaration == "unmatched"
        else tools
    )

    request = _replay_emitted_calls(calls, tools=current)

    assert [
        json.loads(call["function"]["arguments"])["x"] for call in request.messages[0]["tool_calls"]
    ] == ([1, 1.5] if reverse else [1.5, 1])


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1, "text"),
        (1, True),
        (1, None),
        (1, {"nested": 1}),
        (1.5, "text"),
        (1.5, True),
        (1.5, None),
        (1.5, {"nested": 1}),
        ("text", True),
        ("text", None),
        ("text", {"nested": 1}),
        (True, None),
        (True, {"nested": 1}),
        (None, {"nested": 1}),
    ],
)
def test_incompatible_self_derived_property_types_are_rejected(left: Any, right: Any) -> None:
    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": "store",
                "arguments": json.dumps({"x": value}, separators=(",", ":")),
            },
        }
        for index, value in enumerate((left, right))
    ]
    messages = [
        {"role": "assistant", "content": None, "tool_calls": calls},
        *({"role": "tool", "tool_call_id": call["id"], "content": "ok"} for call in calls),
    ]

    with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
        parse_chat_request(
            {"messages": messages},
            require_model=False,
            allow_managed_selectors=True,
        )


@pytest.mark.parametrize("reverse", [False, True], ids=["emitted-order", "reversed-order"])
@pytest.mark.parametrize("declaration", ["none", "unmatched", "matched"])
def test_merged_probe_keeps_properties_required_when_every_call_contains_them(
    reverse: bool,
    declaration: str,
) -> None:
    tools = _history_replay_tools()
    text = (
        "<tool_call><function=store><parameter=x>plain</parameter></function></tool_call>"
        "<tool_call><function=store><parameter=y>a</parameter>b</parameter>"
        "<parameter=x>a</function></tool_call>b</parameter></function></tool_call>"
    )
    emitted = parse_qwen3_coder_output(text, tools)
    assert len(emitted.calls) == 2
    calls = emitted.calls[::-1] if reverse else emitted.calls
    current = (
        None
        if declaration == "none"
        else _unmatched_replay_tools()
        if declaration == "unmatched"
        else tools
    )

    request = _replay_emitted_calls(calls, tools=current)

    assert len(request.messages[0]["tool_calls"]) == 2


def test_mixed_required_optional_probe_merge_fuzz_preserves_emitted_turns() -> None:
    declaration = _history_replay_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        "x": {"type": "number"},
        "y": {"type": "integer"},
        "z": {"type": "boolean"},
    }
    declaration["function"]["parameters"]["required"] = ["x"]
    tools = normalize_tools([declaration])
    unmatched = _unmatched_replay_tools()
    generator = random.Random(70880)

    for count in (2, 3, 5, 10):
        for _ in range(5):
            parts = []
            for index in range(count):
                fields = [f"<parameter=x>{'1.5' if generator.getrandbits(1) else '1'}</parameter>"]
                if generator.getrandbits(1):
                    fields.append(f"<parameter=y>{index}</parameter>")
                if generator.getrandbits(1):
                    fields.append(f"<parameter=z>{'true' if index % 2 else 'false'}</parameter>")
                parts.append(
                    "<tool_call><function=store>" + "".join(fields) + "</function></tool_call>"
                )
            emitted = parse_qwen3_coder_output("".join(parts), tools)
            assert len(emitted.calls) == count
            calls = list(emitted.calls)
            generator.shuffle(calls)
            for current in (None, unmatched, tools):
                request = _replay_emitted_calls(calls, tools=current)
                assert len(request.messages[0]["tool_calls"]) == count


def test_undeclared_history_uses_a_self_derived_replay_probe() -> None:
    value = "</parameter><parameter=y>spoof"

    for tools in (None, tools_wire(_history_replay_tools())):
        messages = _history_replay_messages(value)
        if tools is not None:
            messages[0]["tool_calls"][0]["function"]["name"] = "missing"
        payload = {"messages": messages}
        if tools is not None:
            payload["tools"] = tools
        request = parse_chat_request(
            payload,
            require_model=False,
            allow_managed_selectors=True,
        )
        assert json.loads(request.messages[0]["tool_calls"][0]["function"]["arguments"]) == {
            "x": value
        }

    with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
        parse_chat_request(
            {
                "messages": _history_replay_arguments(
                    {"x": "</parameter><parameter=y>spoof", "y": "actual"}
                )
            },
            require_model=False,
            allow_managed_selectors=True,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"x": "plain", "z": "</parameter><parameter=x>spoof"},
        {"z": "</parameter><parameter=x>spoof", "x": "plain"},
    ],
)
def test_structural_text_in_wrong_typed_declared_field_rejects_history(
    arguments: dict[str, Any],
) -> None:
    messages = _history_replay_arguments(arguments)
    declaration = _history_replay_tools()[0].wire()
    declaration["function"]["parameters"]["properties"]["z"] = {"type": "integer"}
    declaration["function"]["parameters"]["required"] = ["x", "z"]

    with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
        parse_chat_request(
            {"messages": messages, "tools": [declaration]},
            require_model=False,
            allow_managed_selectors=True,
        )


@pytest.mark.parametrize(
    "value",
    ["plain", "</parameter><parameter=x>spoof"],
    ids=["plain", "structural"],
)
def test_unknown_declared_property_rejects_history(value: str) -> None:
    messages = _history_replay_arguments({"x": "plain", "unknown": value})

    with pytest.raises(OpenAIRequestError, match="cannot be replayed exactly"):
        parse_chat_request(
            {"messages": messages, "tools": tools_wire(_history_replay_tools())},
            require_model=False,
            allow_managed_selectors=True,
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
    # the interior opener reopens ``a`` itself, so optional ``b`` never offers a competing
    # assignment and absorption is the single valid parse. an interior ``b`` would make the
    # body ambiguous, which belongs with the fail-closed ownership cases instead.
    text = _candidate_call(
        "<parameter=a>before </parameter><parameter=a>inside</parameter>"
        "</function></tool_call> boundary <parameter=c>embedded</parameter> after</parameter>"
        "<parameter=c>done</parameter>"
    )
    expected = {
        "a": (
            "before </parameter><parameter=a>inside</parameter>"
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
    # the fake continuation reopens ``a``, which is already assigned, so no competing
    # assignment exists and absorption is the single valid parse. naming a still-missing
    # parameter here would make the body genuinely ambiguous and must fall back to text.
    text = _candidate_call(
        "<parameter=a>alpha</parameter>"
        "<parameter=b>before </parameter><parameter=a>fake</parameter>"
        "</function></tool_call> after</parameter>"
        "<parameter=c>real-c</parameter><parameter=d>real-d</parameter>"
    )
    expected = {
        "a": "alpha",
        "b": ("before </parameter><parameter=a>fake</parameter></function></tool_call> after"),
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
    # every case below admits more than one valid parameter assignment, so the only
    # answer that cannot invoke a tool with arguments the model never emitted is the
    # exact-text fallback. a nested close does not settle ownership: the competing
    # assignment can resume several value boundaries later.
    return [
        ("all-required-fake-cd", all_required, all_required_text, None),
        ("optional-b-incomplete-close", optional_b, optional_b_text, None),
        ("optional-a-two-valid-assignments", optional_a, ambiguous_text, None),
        ("required-b-incomplete-close", required_abcd, required_abcd_text, None),
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
    """repeated fake continuations stay exact text because each one adds an assignment.

    every repeat reopens ``c`` and ``d`` while both are still missing, so the body admits a
    combinatorial number of valid assignments. emitting any single one would invoke the tool
    with arguments the model never chose, so the whole candidate has to survive as its bytes.
    """
    tools, text, _fake = _repeated_fake_continuation_case(12)

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert result.content == text
    assert result.calls == ()


def test_repeated_fake_continuation_work_is_linear_without_prefix_materialization(
    monkeypatch,
) -> None:
    """the boundary scan must stay linear in characters read, not merely in calls made.

    counting invocations alone would certify a parser that makes ``2n`` calls while each
    one scans an ``O(n)`` suffix, so the aggregate distance every ``find`` actually
    traverses is what gets measured here.

    both bounds observe the module-level ``_find_parameter_end`` alias, so a rescan that
    called ``str.find`` directly would stay invisible to them. reroute the scan through the
    alias rather than loosening these assertions if that ever becomes reachable.
    """
    originals = {
        name: getattr(tool_calls_module, name)
        for name in (
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

    find_parameter_end = tool_calls_module._find_parameter_end
    scanned = {"distance": 0, "finds": 0}

    def measured_find(text, needle, start):
        found = find_parameter_end(text, needle, start)
        # a match reads the needle it matched, and a miss reads to the end of the text.
        scanned["distance"] += len(text) - start if found < 0 else found - start + len(needle)
        scanned["finds"] += 1
        return found

    monkeypatch.setattr(tool_calls_module, "_find_parameter_end", measured_find)

    measurements = {}
    for repeats in (32, 64, 128, 256):
        counters.update(dict.fromkeys(counters, 0))
        scanned.update({"distance": 0, "finds": 0})
        tools, text, _fake = _repeated_fake_continuation_case(repeats)
        result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")
        assert result.content == text
        assert result.calls == ()
        # an ambiguous candidate never reaches value materialization, so the only work that
        # may grow with the input is the boundary scan itself.
        assert counters["_coerce_value"] == 0
        assert counters["_materialize_span"] == 0
        assert counters["_validate_value"] == 0
        calls = scanned["finds"] + counters["_parse_parameter_value"]
        measurements[repeats] = (scanned["distance"], len(text), calls)

    # the growth bound is stated against input characters, not repeat count: a repeat carries
    # a long value, so doubling the repeats multiplies the text by 1.96 to 2.02 rather than
    # by two. the scan reads ``len(text) - 129`` characters at every width here, and 2.1 is
    # what makes the ratio load-bearing: an ``L * log L`` scan over these lengths grows by
    # 2.12 to 2.16, which a looser threshold reads as linear, and ``L ** 1.5`` by 2.75 to
    # 2.86. distance alone would still miss a rescan that repeats work without moving the
    # endpoint it returns, so the number of boundary calls is bounded next to it.
    for distance, length, _calls in measurements.values():
        assert distance <= length
    for narrower, wider in pairwise(sorted(measurements)):
        assert measurements[wider][0] <= 2.1 * measurements[narrower][0]
        assert measurements[wider][2] <= 2.1 * measurements[narrower][2]


_REVIEWED_OWNERSHIP_CASES = [
    (
        "cross-parameter-delimiters",
        "abc",
        ["a", "b", "c"],
        "abc",
        {
            "a": "",
            "b": "</parameter></function></tool_call></parameter>",
            "c": "</parameter><parameter=b>",
        },
    ),
    (
        "nested-optional-closure",
        "abc",
        ["a", "c"],
        "cba",
        {"c": "", "b": "</parameter></function></tool_call></parameter>", "a": ""},
    ),
]


@pytest.mark.parametrize(
    ("_case", "properties", "required", "order", "values"),
    _REVIEWED_OWNERSHIP_CASES,
    ids=[case[0] for case in _REVIEWED_OWNERSHIP_CASES],
)
def test_ambiguous_cross_parameter_ownership_falls_back_to_exact_text(
    _case: str,
    properties: str,
    required: list[str],
    order: str,
    values: dict[str, str],
) -> None:
    """delimiter bytes a model may legitimately emit must never be redistributed.

    each body below is exactly what the grammar writes for its values, yet the same bytes
    also parse as a different assignment. the first case moves ``b``'s content into ``a``;
    the second drops ``b`` entirely and absorbs its markup into ``c``. one closing sequence
    does not settle ownership, because the competing assignment can resume several value
    boundaries later, so the parser has to keep scanning before it commits to a call.
    """
    tools = _string_field_tools(properties, "".join(required))
    text = _candidate_call(
        "".join(f"<parameter={name}>{values[name]}</parameter>" for name in order)
    )

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert result.content == text
    assert result.calls == ()


@pytest.mark.parametrize(
    ("_case", "properties", "required", "order", "values"),
    _REVIEWED_OWNERSHIP_CASES,
    ids=[case[0] for case in _REVIEWED_OWNERSHIP_CASES],
)
def test_ambiguous_cross_parameter_ownership_falls_back_across_every_split(
    _case: str,
    properties: str,
    required: list[str],
    order: str,
    values: dict[str, str],
) -> None:
    """streaming must reach the same verdict, since a delta cannot be retracted later."""
    tools = _string_field_tools(properties, "".join(required))
    text = _candidate_call(
        "".join(f"<parameter={name}>{values[name]}</parameter>" for name in order)
    )

    for split in range(len(text) + 1):
        parser = ToolCallStreamParser(tools, id_factory=lambda: "call_fixed")
        assert parser.feed(text[:split]) == ""
        assert parser.feed(text[split:]) == ""
        result = parser.finish()
        assert result.content == text
        assert result.calls == ()


def _numeric_replay_tools():
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {"x": {"type": "integer"}}
    declaration["function"]["parameters"]["required"] = ["x"]
    return normalize_tools([declaration])


@pytest.mark.parametrize(
    "emitted", ["1", "1e300", "1e2000", "1e100000"], ids=["one", "native", "huge", "enormous"]
)
def test_generated_numeric_call_replays_as_valid_history(emitted: str) -> None:
    """a call flash emits must be a call flash accepts back.

    the follow-up turn resends the assistant call alongside its tool result, so a value the
    parser emits but the request layer refuses would strand the model with a call it can
    never supply a result for. compact exponents render exactly, so they stay legal history.
    """
    text = _candidate_call(f"<parameter=x>{emitted}</parameter>")
    result = parse_qwen3_coder_output(
        text, _numeric_replay_tools(), id_factory=lambda: "call_fixed"
    )
    call = result.calls[0]

    parse_chat_request(
        {
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": [call.wire()]},
                {"role": "tool", "tool_call_id": call.id, "content": "ok"},
            ]
        },
        require_model=False,
        allow_managed_selectors=True,
    )


def _empty_argument_tools():
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {}
    declaration["function"]["parameters"]["required"] = []
    return normalize_tools([declaration])


def _repeated_empty_calls(count: int) -> str:
    name = _empty_argument_tools()[0].name
    return "".join(f"<tool_call><function={name}></function></tool_call>" for _ in range(count))


@pytest.mark.parametrize("count", [408, 409], ids=["best-case-ceiling", "past-every-continuation"])
def test_generated_call_count_stops_where_no_continuation_could_carry_it(count: int) -> None:
    """the parser stops emitting once no follow-up shape could carry the batch.

    the cheapest continuation is a minimal prior message with plain string results, and even
    that exhausts the budget past 408 calls with empty arguments, so 409 could not replay
    under any shape. passing at 408 is not the converse promise: prior history and richer
    result shapes shrink the same budget well below it, which
    ``test_replay_budget_depends_on_history_and_result_shape`` pins.
    """
    tools = _empty_argument_tools()
    text = _repeated_empty_calls(count)

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    if count > 408:
        assert result.content == text
        assert result.calls == ()
        return

    assert len(result.calls) == count
    parse_chat_request(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {**call.wire(), "id": f"{call.id}{index}"}
                        for index, call in enumerate(result.calls)
                    ],
                },
                *(
                    {"role": "tool", "tool_call_id": f"{call.id}{index}", "content": "ok"}
                    for index, call in enumerate(result.calls)
                ),
            ],
            "tools": [tool.wire() for tool in tools],
        },
        require_model=False,
        allow_managed_selectors=True,
    )


def _replay_messages(count: int, *, prior: int = 0, result_shape: str = "string"):
    """the follow-up a client sends after receiving ``count`` calls."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    for index in range(prior):
        messages.append({"role": "assistant", "content": f"a{index}"})
        messages.append({"role": "user", "content": f"u{index}"})
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": "probe", "arguments": "{}"},
                }
                for index in range(count)
            ],
        }
    )
    for index in range(count):
        result: dict[str, Any] = {"role": "tool", "tool_call_id": f"call_{index}"}
        if result_shape == "named":
            result["name"] = "probe"
            result["content"] = "ok"
        elif result_shape == "block":
            result["content"] = [{"type": "text", "text": "ok"}]
        else:
            result["content"] = "ok"
        messages.append(result)
    return messages


def _replay_accepted(count: int, **kwargs) -> bool:
    try:
        detached_messages(
            _replay_messages(count, **kwargs),
            sequence_types=(list, tuple),
            sequence_error="messages must be a list",
            error_type=OpenAIRequestError,
        )
    except OpenAIRequestError as exc:
        if "complexity" in str(exc):
            return False
        raise
    return True


@pytest.mark.parametrize(
    ("kwargs", "largest"),
    [
        ({}, 408),
        ({"prior": 2}, 407),
        ({"prior": 100}, 348),
        ({"result_shape": "named"}, 371),
        ({"result_shape": "block"}, 314),
    ],
    ids=["baseline", "short-history", "long-history", "named-result", "text-block-result"],
)
def test_replay_budget_depends_on_history_and_result_shape(kwargs, largest: int) -> None:
    """the emitted-call ceiling cannot promise replayability, and this records why.

    every one of these follow-ups is well-formed and uses only supported shapes, yet each
    admits a different number of calls, all at or below the 408 the parser will emit. prior
    conversation competes for the same budget, and a result costs four nodes as a string,
    five with the optional ``name``, and seven as a single text block. a ceiling chosen for
    one shape is therefore wrong for the others, which is why the contract documents the
    follow-up as rejectable instead of claiming every emitted call replays.
    """
    assert _replay_accepted(largest, **kwargs)
    assert not _replay_accepted(largest + 1, **kwargs)


def test_each_extra_result_block_costs_three_nodes() -> None:
    """the ``+3`` per additional text block that the parser comment cites.

    the block cost is what makes a fixed ceiling unable to promise replay, since the client
    chooses how many blocks each result carries. pinning it here keeps the published figure
    honest if the walker's accounting ever changes.
    """

    def largest(blocks: int) -> int:
        low, high = 0, 512
        while low < high:
            middle = (low + high + 1) // 2
            messages = _replay_messages(middle, result_shape="string")
            for message in messages:
                if message["role"] == "tool":
                    message["content"] = [{"type": "text", "text": "ok"}] * blocks
            try:
                detached_messages(
                    messages,
                    sequence_types=(list, tuple),
                    sequence_error="messages must be a list",
                    error_type=OpenAIRequestError,
                )
            except OpenAIRequestError as exc:
                if "complexity" not in str(exc):
                    raise
                high = middle - 1
            else:
                low = middle
        return low

    # a call plus a one-block result costs 13 nodes and a two-block result 16, so the same
    # budget admits 314 and 255 calls. that difference is the third block node made visible.
    assert (largest(1), largest(2)) == (314, 255)


def test_argument_content_does_not_change_the_replay_budget() -> None:
    """``arguments`` is a JSON string, so its size never moves the boundary.

    this is why the cap is stated structurally rather than against argument complexity, and
    it is the claim that made the original empty-``{}`` measurement generalize at all.
    """
    for arguments in ("{}", '{"query":"x"}', '{"blob":"' + "x" * 100_000 + '"}'):
        messages = _replay_messages(408)
        for message in messages:
            for call in message.get("tool_calls", ()):
                call["function"]["arguments"] = arguments
        detached_messages(
            messages,
            sequence_types=(list, tuple),
            sequence_error="messages must be a list",
            error_type=OpenAIRequestError,
        )


def test_long_history_leaves_no_room_for_even_one_call() -> None:
    """the strongest form: an accepted request can admit zero replayable calls.

    a history flash accepts on its own can already consume so much of the budget that
    replaying a single call overflows it. no positive ceiling in the parser can prevent
    this, so the parser must not advertise a replay guarantee it cannot keep.
    """
    history = [{"role": "user", "content": [{"type": "text", "text": "x"} for _ in range(1364)]}]
    detached_messages(
        history,
        sequence_types=(list, tuple),
        sequence_error="messages must be a list",
        error_type=OpenAIRequestError,
    )

    follow_up = [
        *history,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "probe", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "ok"},
    ]
    with pytest.raises(OpenAIRequestError, match="complexity"):
        detached_messages(
            follow_up,
            sequence_types=(list, tuple),
            sequence_error="messages must be a list",
            error_type=OpenAIRequestError,
        )


def _wide_optional_string_tools(count: int):
    declaration = _delimiter_tools()[0].wire()
    declaration["function"]["parameters"]["properties"] = {
        f"p{index}": {"type": "string"} for index in range(count)
    }
    declaration["function"]["parameters"]["required"] = []
    return normalize_tools([declaration])


def _wide_optional_string_call(count: int) -> str:
    return _candidate_call("".join(f"<parameter=p{index}>v</parameter>" for index in range(count)))


def test_recursion_during_classification_falls_back_to_exact_text(monkeypatch) -> None:
    """the RecursionError handler itself is pinned, not the width that happens to trip it.

    the width at which the classifier exhausts the stack depends on the ambient recursion
    limit, so a raised limit in ci or a future interpreter would leave the width-based
    cases green with the handler deleted. injecting the error directly keeps the branch
    covered no matter how deep the interpreter lets the parser descend.
    """
    # both parameters are required so the body admits exactly one assignment. an all-optional
    # body of the same shape falls back on its own merits, which would hide the injection.
    declaration = _wide_optional_string_tools(2)[0].wire()
    declaration["function"]["parameters"]["required"] = ["p0", "p1"]
    tools = normalize_tools([declaration])
    text = _wide_optional_string_call(2)
    assert parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed").calls != ()

    original = tool_calls_module._parse_parameters
    calls = {"count": 0}

    def exhausting(*args, **kwargs):
        calls["count"] += 1
        raise RecursionError("injected")

    monkeypatch.setattr(tool_calls_module, "_parse_parameters", exhausting)

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")
    assert result.content == text
    assert result.calls == ()
    assert calls["count"] > 0

    buffered_calls = calls["count"]
    parser = ToolCallStreamParser(tools, id_factory=lambda: "call_fixed")
    assert parser.feed(text) == ""
    streamed = parser.finish()
    assert streamed.content == text
    assert streamed.calls == ()
    # the streamed subcase has to reach classification too, or it would pass just as well
    # against a finish() that returned exact text without ever descending.
    assert calls["count"] > buffered_calls

    # the injected failure raises before touching anything, so this only confirms the fixture
    # and the patch are restored, not that a genuine stack exhaustion leaves no residue. the
    # real evidence for that is structural: `work` is the only caller-owned object mutated
    # before the descent, and both callers discard it when `_EXHAUSTED` comes back.
    monkeypatch.setattr(tool_calls_module, "_parse_parameters", original)
    assert parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed").calls != ()


@pytest.mark.parametrize("count", [332, 511], ids=["first-leaking-width", "widest-declarable"])
def test_wide_optional_parameter_run_falls_back_instead_of_exhausting_the_stack(
    count: int,
) -> None:
    """a candidate too deep to classify keeps its bytes rather than raising.

    the parser descends once per parameter value, so a schema wide enough to declare
    hundreds of optional strings can exhaust the interpreter stack before the work budget
    notices. both widths here are legal declarations, so a RecursionError escaping to the
    caller would turn a servable request into a 500 instead of ordinary assistant text.
    """
    tools = _wide_optional_string_tools(count)
    text = _wide_optional_string_call(count)

    result = parse_qwen3_coder_output(text, tools, id_factory=lambda: "call_fixed")

    assert result.content == text
    assert result.calls == ()

    parser = ToolCallStreamParser(tools, id_factory=lambda: "call_fixed")
    assert parser.feed(text) == ""
    streamed = parser.finish()
    assert streamed.content == text
    assert streamed.calls == ()


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
    original = tool_calls_module._index_parameter_openers
    examined = 0

    def measured(text: str, start: int, declared, work):
        nonlocal examined
        examined += len(text) - start
        return original(text, start, declared, work)

    monkeypatch.setattr(tool_calls_module, "_index_parameter_openers", measured)
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


def test_exactly_the_best_case_ceiling_remains_supported() -> None:
    """the largest emittable batch is the largest any continuation could carry back.

    this used to allow 512, but replaying 409 calls with their tool results already exceeds
    the message budget under the cheapest continuation, so the parser was handing back
    responses flash then rejected. emitting 408 does not promise this batch replays; it
    promises only that no smaller ceiling is required to rule out every continuation.
    """
    tools, text = _integer_calls(408)
    next_id = iter(f"call_{index}" for index in range(408)).__next__

    result = parse_qwen3_coder_output(text, tools, id_factory=next_id)

    assert result.content is None
    assert len(result.calls) == 408
    assert result.calls[-1].id == "call_407"


def test_one_call_past_the_best_case_ceiling_rejects_before_call_id_creation() -> None:
    tools, text = _integer_calls(409)
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


def test_whitespace_scanning_matches_stepping_and_stays_native() -> None:
    """the whitespace scan must measure a run natively and keep the stepping semantics exactly.

    a tool declaration reaches the megabytes on the request path, so stepping per character holds
    the event loop for seconds on a run of spaces. the replacement has to agree with stepping on
    every input, including the past-the-end cursor that must come back unchanged rather than
    clamped to the length, because callers compare the returned offset against their own bounds.
    """

    def stepping(text: str, cursor: int) -> int:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor

    # `\s` and `str.isspace` have to accept the same set, or the scan silently reclassifies a
    # separator. exhaustive over every codepoint rather than sampled.
    assert not [
        code
        for code in range(0x110000)
        if bool(text_scan._LEADING_WHITESPACE_RE.match(chr(code)).end()) != chr(code).isspace()
    ]

    alphabet = " \t\n\r\f\v\u00a0\u2003\u3000xy"
    random.seed(20260830)
    cases = [("", 0), ("", 5), ("  ", 0), ("  ", 2), ("  ", 9), ("x", 1), ("x", 99), ("  a", 0)]
    cases += [
        (
            "".join(random.choice(alphabet) for _ in range(random.randint(0, 14))),
            random.randint(0, 16),
        )
        for _ in range(20_000)
    ]
    assert not [
        (text, cursor)
        for text, cursor in cases
        if text_scan.skip_whitespace(text, cursor) != stepping(text, cursor)
    ]

    # a cursor past the end returns the cursor, not the length. stepping never moved it, and a
    # clamp here would silently rewrite an out-of-range offset into a valid-looking one.
    assert text_scan.skip_whitespace("ab", 7) == 7

    # the run is measured in one match rather than one per character, which is the regression
    # this guards: stepping over this many spaces costs seconds on the request path.
    calls = 0
    pattern = text_scan._LEADING_WHITESPACE_RE

    class _CountingPattern:
        def match(self, text: str, position: int):
            nonlocal calls
            calls += 1
            return pattern.match(text, position)

    with mock.patch.object(text_scan, "_LEADING_WHITESPACE_RE", _CountingPattern()):
        assert text_scan.skip_whitespace(" " * 2_000_000 + "x", 0) == 2_000_000
    assert calls == 1, calls


def test_stop_overlap_scanning_does_not_grow_with_the_declared_marker_count() -> None:
    """one stop against a wide catalog must not cost a slice per marker per size.

    every declared parameter contributes a marker, so the allowed 128 tools reach tens of
    thousands of them. comparing each against a stop by rebuilding a slice for every shared-run
    length is millions of allocations on the synchronous request path, and the complexity ceiling
    passes it because that formula counts characters rather than comparisons.
    """
    names = [f"{'n' * 60}{index:04d}" for index in range(120)]
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": f"t{index}",
                    "parameters": {
                        "type": "object",
                        "properties": {name: {"type": "string"} for name in names},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
            for index in range(16)
        ]
    )
    markers = ["<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>"]  # fmt: skip
    for tool in tools:
        markers.append(f"<function={tool.name}>")
        markers.extend(f"<parameter={name}>" for name in tool.parameters["properties"])

    slices = 0
    real_overlaps = text_scan.overlaps_any

    def counting(value: str, candidates):
        nonlocal slices
        materialized = list(candidates)
        slices += len(materialized)
        return real_overlaps(value, materialized)

    # a stop that shares no run with any marker is the worst case: nothing short-circuits.
    stop = "z" * 75
    with mock.patch.object(text_scan, "overlaps_any", counting):
        validate_tool_stop_sequences(
            [stop], tools=tools, tool_choice="auto", error_type=OpenAIRequestError
        )
    # the scan visits each marker once, rather than once per shared-run length per marker.
    assert slices == len(markers), (slices, len(markers))

    # and the verdict is unchanged from the per-pair predicate it replaced, on both sides.
    assert not text_scan.overlaps_any(stop, markers)
    for overlapping in ("zzz<", ">zzz", f"{names[0]}>", "<tool_call>"):
        assert text_scan.overlaps_any(overlapping, markers), overlapping

    # enumerating every run of the stop rather than only those a marker could share costs the
    # square of its length. the complexity ceiling counts characters, so it admits a stop of
    # millions of them, whose full run set is terabytes: the memory has to be bounded by the
    # longest marker, not by the stop. measured rather than reasoned, because the allocation is
    # inside a comprehension that no call count can see.
    def peak_bytes(length: int) -> int:
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            validate_tool_stop_sequences(
                ["z" * length], tools=tools, tool_choice="auto", error_type=OpenAIRequestError
            )
            return tracemalloc.get_traced_memory()[1] - before
        finally:
            tracemalloc.stop()

    # a stop eight times longer would cost sixty-four times the memory if the runs were unbounded.
    small, large = peak_bytes(1_000), peak_bytes(8_000)
    assert large < 4 * max(small, 64 * 1024), (small, large)


def test_nested_string_enums_may_carry_grammar_delimiters() -> None:
    """only a value the grammar writes between its own delimiters can be broken by carrying one.

    a root string property is written as a bare parameter value, so a viable closer inside it
    genuinely cannot be read back. a string nested in a container is written as part of that
    container's json, which is delimited by brace depth and quoting instead, so the same
    characters round trip and rejecting them refuses a schema the parser handles correctly.
    """
    hostile = "</parameter><parameter=x>spoof"

    def declaration(parameters: dict[str, object]) -> list[dict[str, object]]:
        return [{"type": "function", "function": {"name": "store", "parameters": parameters}}]

    root = {
        "type": "object",
        "properties": {"tag": {"type": "string", "enum": [hostile]}},
        "required": [],
        "additionalProperties": False,
    }
    with pytest.raises(OpenAIRequestError, match="unrepresentable tool grammar delimiter"):
        normalize_tools(declaration(root), error_type=OpenAIRequestError)

    nested_object = {
        "type": "object",
        "properties": {
            "o": {
                "type": "object",
                "properties": {"tag": {"type": "string", "enum": [hostile]}},
                "required": [],
                "additionalProperties": False,
            }
        },
        "required": [],
        "additionalProperties": False,
    }
    nested_array = {
        "type": "object",
        "properties": {"a": {"type": "array", "items": {"type": "string", "enum": [hostile]}}},
        "required": [],
        "additionalProperties": False,
    }
    tools = normalize_tools(declaration(nested_object), error_type=OpenAIRequestError)
    normalize_tools(declaration(nested_array), error_type=OpenAIRequestError)

    # the accepted nested value is not merely tolerated: the parser reproduces it exactly.
    payload = json.dumps({"tag": hostile})
    text = (
        f"<tool_call>\n<function=store>\n<parameter=o>\n{payload}\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    result = parse_qwen3_coder_output(text, tools, _work_limit=None)
    assert [call.name for call in result.calls] == ["store"]
    assert json.loads(result.calls[0].arguments) == {"o": {"tag": hostile}}


def test_the_narrowed_closer_scan_finds_every_declared_name_and_no_other() -> None:
    """filtering names outside the pattern must not change which closers are viable.

    a name that prefixes another must not be accepted for it, and the scan must resume far enough
    past a rejected opener to stay linear while still finding a closer that begins inside the span
    it skipped. the declared set comes from an untrusted declaration, so a name carrying regex
    syntax must be compared literally rather than reaching an engine as syntax.
    """
    find = text_scan.find_viable_parameter_end
    shadowing = [{"a", "ab"}, {"x", "xy", "xyz"}, {"dat", "data"}, {"n", "nn", "nnn"}]
    for names in shadowing:
        for name in names:
            assert find(f"</parameter> <parameter={name}>", 0, names) == 0, (names, name)
        for absent in ("abc", "xyzz", "datax", "nnnn", "zz"):
            if absent not in names:
                assert find(f"</parameter> <parameter={absent}>", 0, names) == -1, (names, absent)
        # the function end is always viable, whatever the declaration names.
        assert find("</parameter>  </function>", 0, names) == 0, names

    # with nothing declared only the function end can continue a call.
    assert find("</parameter> </function>", 0, frozenset()) == 0
    assert find("</parameter> <parameter=anything>", 0, frozenset()) == -1

    # a rejected pairing is skipped whole, so the closer that opens inside it must still be found:
    # resuming even one character further would step past this one and report no viable closer.
    nested = "</parameter> <parameter=undeclared></parameter> <parameter=a>"
    assert find(nested, 0, frozenset({"a"})) == nested.index("</parameter> <parameter=a>")

    # a name carrying regex syntax reaches the scan through a replay probe, whose names are the
    # keys of a historical call rather than a validated declaration. it must be compared as literal
    # text: matching its own spelling, and never matching a name it would match as a pattern.
    for hostile in ("a|b", "a.b", "(a)", "a*"):
        assert find(f"</parameter> <parameter={hostile}>", 0, frozenset({hostile})) == 0, hostile
        assert find("</parameter> <parameter=a>", 0, frozenset({hostile})) == -1, hostile
    for hostile in ("a.b", "(a)", "a*", "a|b"):
        with pytest.raises(OpenAIRequestError, match="key is invalid"):
            normalize_tools(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "store",
                            "parameters": {
                                "type": "object",
                                "properties": {hostile: {"type": "string"}},
                                "required": [],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                error_type=OpenAIRequestError,
            )


def test_a_replay_probe_sees_openers_named_outside_the_declaration_charset() -> None:
    """the closer scan's name run is the grammar's, not a public declaration's.

    a replay probe derives its parameter names from the keys of a historical call, which never pass
    `_identifier_name`. narrowing the scan to the charset a declaration may hold made an opener
    named for such a key invisible, so a closer that genuinely hands off to it read as inert and a
    replayable history was rejected as unreplayable.
    """
    # each of these keys is outside what `_identifier_name` admits, and `<` is outside even the
    # grammar's own markers, yet the parser ends a name at the first `>` and nothing narrower. a
    # scan that stops the run at any of these characters cannot see the opener the key spells.
    for key in ("bad.name", "bad<name", "bad name", "", "é", "x" * 200):
        arguments = {"good": "prefix</parameter>x", key: "b"}
        history = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "hist", "arguments": json.dumps(arguments)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        # no current declaration, so the probe's names come straight from the historical keys.
        validate_tool_history_replay(history, None, error_type=OpenAIRequestError)

        # the scan itself must find the boundary that hands off to the key's opener.
        assert (
            text_scan.find_viable_parameter_end(
                f"</parameter>\n<parameter={key}>", 0, frozenset(arguments)
            )
            == 0
        ), key
        # and must still reject a name no side declares, whatever characters it carries.
        assert (
            text_scan.find_viable_parameter_end(
                f"</parameter>\n<parameter=other{key}z>", 0, frozenset(arguments)
            )
            == -1
        ), key

    # a key carrying `>` is a different case and must stay rejected: the parser reads the name as
    # ending at that `>`, so the template cannot spell the key back and the history is genuinely
    # unreplayable. widening the run past the delimiter would claim a replay the parser cannot do.
    with pytest.raises(OpenAIRequestError):
        validate_tool_history_replay(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "hist",
                                "arguments": json.dumps({"a>b": "v"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
            None,
            error_type=OpenAIRequestError,
        )


def test_a_wide_catalog_costs_no_per_declaration_pattern_construction() -> None:
    """the closer scan must not be built from the declared names.

    the node budget is per declaration, so one legal request may carry `_MAX_TOOLS` declarations of
    `_MAX_SCHEMA_NODES` properties each. spelling those names into a pattern makes each declaration
    cost a compilation of its whole name list, and nothing charges the parser for it: replaying a
    history with one call per tool spent seconds before rejecting. filtering the names against the
    schema instead keeps the scan one fixed pattern, so the cost cannot scale with the catalog.
    """
    from flash.serve.request.tool_calls import _MAX_SCHEMA_NODES, _MAX_TOOLS

    names = [f"p{index}_{'a' * 40}" for index in range(_MAX_SCHEMA_NODES - 1)]
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": f"f{index}",
                    "parameters": {
                        "type": "object",
                        "properties": {name: {"type": "string"} for name in names},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
            for index in range(_MAX_TOOLS)
        ],
        error_type=OpenAIRequestError,
    )
    # one candidate per declaration, each naming a different tool. a pattern built from the declared
    # names is built once per candidate, so this is the shape that pays for it.
    text = "".join(
        f"<tool_call><function=f{index}><parameter={names[0]}>\nv\n"
        "</parameter></function></tool_call>"
        for index in range(_MAX_TOOLS)
    )

    compiles = 0
    real_compile = re.compile

    def counting_compile(*args, **kwargs):
        nonlocal compiles
        compiles += 1
        return real_compile(*args, **kwargs)

    # counted rather than timed: a pattern built from the names is expensive whether it is cached
    # or rebuilt, and a cached one is only slow on its first request, so a stopwatch reports either
    # as noise on a warm process. compiling nothing at all is the property that holds regardless.
    with mock.patch.object(re, "compile", counting_compile):
        parse_qwen3_coder_output(text, tools)
    assert compiles == 0, compiles


def test_schema_node_budget_is_per_declaration_and_bounded_by_the_tool_maximum() -> None:
    """the node ceiling is per declaration, and the tool maximum bounds the list.

    a single shared budget across the list would reject ordinary multi-integration catalogs whose
    declarations are individually tiny, and the two existing ceilings already cap a request at
    `_MAX_TOOLS` x `_MAX_SCHEMA_NODES` nodes, whose full normalize-wire-renormalize round trip
    measures a fraction of a second. so the pair is the bound, with no third smaller ceiling.
    """
    from flash.serve.request.tool_calls import _MAX_SCHEMA_NODES, _MAX_TOOLS

    def declarations(count: int, properties: int) -> list[dict]:
        schema = {
            "type": "object",
            "properties": {f"p{index}": {"type": "string"} for index in range(properties)},
            "required": [],
            "additionalProperties": False,
        }
        return [
            {"type": "function", "function": {"name": f"t{index}", "parameters": schema}}
            for index in range(count)
        ]

    # the shapes an ordinary multi-integration caller sends: many small tools, a few larger ones.
    # a single shared budget across the list rejected every one of these, so each is load-bearing.
    for count, properties in ((_MAX_TOOLS, 1), (_MAX_TOOLS, 4), (_MAX_TOOLS, 16), (64, 8), (32, 16)):  # fmt: skip
        normalize_tools(declarations(count, properties), error_type=OpenAIRequestError)

    # the largest list the two ceilings permit is accepted, since nothing smaller bounds the sum.
    # `- 1` leaves room for the root object node, which counts against the per-declaration ceiling.
    largest = declarations(_MAX_TOOLS, _MAX_SCHEMA_NODES - 1)
    normalize_tools(largest, error_type=OpenAIRequestError)

    # the per-declaration ceiling is what rejects an oversized declaration, and it is charged per
    # declaration rather than against a running total, so the first one over the line is the one
    # named. one property past the ceiling is enough, with no dependence on how many tools precede.
    with pytest.raises(OpenAIRequestError, match="exceeds"):
        normalize_tools(declarations(1, _MAX_SCHEMA_NODES), error_type=OpenAIRequestError)
    oversized = declarations(1, _MAX_SCHEMA_NODES)[0]
    oversized["function"]["name"] = "late"
    with pytest.raises(OpenAIRequestError, match=r"tools\[3\].*exceeds"):
        normalize_tools([*declarations(3, 4), oversized], error_type=OpenAIRequestError)


def test_inert_free_string_closers_are_skipped_natively() -> None:
    """an argument full of closers that cannot end it must not cost one python step each.

    only a ``</parameter>`` followed, after whitespace, by the next parameter or the function end
    can close a free-string value. replay parses with no fixed work cap, so stepping to each inert
    closer let a single accepted request hold the event loop for seconds.
    """
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    # each closer carries a whitespace run, so a skip that stepped per closer would show up both in
    # the search count below and in the work charged further down.
    body = "</parameter>   x" * 200_000
    text = (
        f"<tool_call><function=store><parameter=data>\n{body}\n</parameter></function></tool_call>"
    )

    searches = 0
    real_find = text_scan.find_viable_parameter_end

    def counting_find(text: str, cursor: int, declared):
        nonlocal searches
        searches += 1
        return real_find(text, cursor, declared)

    with mock.patch.object(tool_calls_module.text_scan, "find_viable_parameter_end", counting_find):
        result = parse_qwen3_coder_output(text, tools, _work_limit=None)

    # the value still parses exactly, closers and all.
    assert [call.name for call in result.calls] == ["store"]
    assert json.loads(result.calls[0].arguments)["data"] == body
    # one native search settles the whole inert run rather than 200k python iterations.
    assert searches == 1, searches

    # the skip must stay proportional to the span it settles, so the work charged for a value grows
    # with its length rather than with how many closers happen to sit inside it. an inert run and an
    # ordinary run of the same size must exhaust at the same point, or the charge is measuring the
    # delimiters rather than the text and an untrusted value can be priced by its punctuation.
    def min_work(value: str) -> int:
        probe = (
            f"<tool_call><function=store><parameter=data>\n{value}\n"
            "</parameter></function></tool_call>"
        )

        def parses(limit: int) -> bool:
            return bool(parse_qwen3_coder_output(probe, tools, _work_limit=limit).calls)

        low, high = 1, 200_000
        assert parses(high)
        while low < high:
            middle = (low + high) // 2
            if parses(middle):
                high = middle
            else:
                low = middle + 1
        return low

    length = 12_000
    plain = min_work("z" * length)
    # the same number of characters, carrying one inert closer or six hundred of them. if the skip
    # charged per closer the cost would climb with the count; instead it settles toward the cost of
    # ordinary text of that length, because what is charged is the span the native scan measured.
    charges = {
        closers: min_work(
            ("</parameter>" + " " * (length // closers - len("</parameter>") - 1) + "x") * closers
        )
        for closers in (6, 60, 600)
    }
    assert max(charges.values()) < 1.1 * plain, charges
    assert charges[600] < charges[60] < charges[6], charges


def test_undeclared_openers_do_not_each_open_a_parse_branch() -> None:
    """a closer followed by an opener the schema never declared cannot end the value.

    `_parse_parameters` rejects an undeclared name on sight, so treating each one as a boundary
    spent a recursive branch per occurrence to reach that same answer. narrowing the closer scan
    to declared names lets the engine skip them instead: a value quoting one a million times is
    settled by the same single native search as a value quoting none.
    """
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                        "required": ["data"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )

    def passes(body: str) -> tuple[int, int]:
        """(regex engine entries, parse branches) spent on a value with this body."""
        text = (
            f"<tool_call><function=store><parameter=data>\n{body}\n"
            "</parameter></function></tool_call>"
        )
        searches, branches = [0], [0]
        real_pattern = text_scan._NAMED_PARAMETER_END_RE
        real = tool_calls_module._parse_parameters

        class _CountingPattern:
            """counts entries into the regex engine, not calls to the helper wrapping it.

            counting the helper hides what this test bounds: it is entered once per value either
            way, so a scan that spends a branch on every quoted opener and one that skips them all
            are indistinguishable from outside it. only the one method the scan uses is defined,
            so a rewrite onto a different engine entry point fails here rather than being counted
            under a rule that does not describe it.
            """

            def search(self, text: str, position: int):
                searches[0] += 1
                return real_pattern.search(text, position)

        def counting_parse(*args, **kwargs):
            branches[0] += 1
            return real(*args, **kwargs)

        with (
            mock.patch.object(text_scan, "_NAMED_PARAMETER_END_RE", _CountingPattern()),
            mock.patch.object(tool_calls_module, "_parse_parameters", counting_parse),
        ):
            result = parse_qwen3_coder_output(text, tools, _work_limit=None)
        # the value is unchanged: whatever it quotes is still part of its text.
        assert [call.name for call in result.calls] == ["store"]
        assert json.loads(result.calls[0].arguments)["data"] == body
        return searches[0], branches[0]

    # the recursive descent may not scale with how many times the value quotes an opener the schema
    # never declared: treating each as a boundary spent a rejected branch per occurrence, which is
    # what made an 8 MiB argument take seconds. the branch count is therefore flat.
    quoted = "</parameter><parameter=nope>"
    one, thousand, many = passes(quoted), passes(quoted * 1_000), passes(quoted * 100_000)
    assert one[1] == thousand[1] == many[1] == 2, (one, thousand, many)

    # the engine still visits each quoted pairing to reject it, so its entries do scale. that cost
    # is one match against a fixed pattern rather than a python frame, and it is charged as the
    # span it settles. asserting the entries are flat would be asserting something untrue, so what
    # is pinned is that the visit stays one entry per occurrence and the value it settles on is
    # exact: a value quoting nothing must not enter the engine per character either.
    assert (one[0], thousand[0], many[0]) == (1, 1_000, 100_000), (one, thousand, many)

    # a rejected pairing can contain the start of a real one, because a name runs to the first `>`
    # and so may swallow a whole `</parameter>`. the scan therefore resumes one character into the
    # span it rejected rather than past it. settling the whole scan with one non-overlapping
    # `finditer` pass would resume at the match end, step over the swallowed closer, and report no
    # boundary at all, so the resume is asserted here rather than through the parser: the parser
    # reaches the same text either way, by falling back to the end of input.
    swallowed = (
        "<tool_call><function=store><parameter=data>x"
        "</parameter><parameter=n</parameter></function></tool_call>"
    )
    value_start = swallowed.index(">", swallowed.index("<parameter=")) + 1
    boundary = text_scan.find_viable_parameter_end(swallowed, value_start, {"data"})
    assert swallowed[boundary:] == "</parameter></function></tool_call>", (boundary, swallowed)


def test_opener_index_retains_only_declared_parameter_names() -> None:
    """openers naming something the schema never declares are never read, so none are kept.

    both readers look positions up by a declared parameter name. retaining the rest let an
    untrusted argument spend hundreds of megabytes on offsets nothing consults.
    """
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    # both an undeclared name, which is never consulted and so must not be retained at all, and a
    # declared one, whose every offset is genuinely reachable and so must be retained compactly.
    quoted = ("<parameter=nope>" + "<parameter=data>") * 50_000
    text = (
        f"<tool_call><function=store><parameter=data>\n{quoted}\n"
        "</parameter></function></tool_call>"
    )

    captured: dict[str, object] = {}
    original = tool_calls_module._index_parameter_openers

    def spy(text: str, start: int, declared, work):
        result = original(text, start, declared, work)
        if isinstance(result, dict):
            captured.update(result)
        return result

    with mock.patch.object(tool_calls_module, "_index_parameter_openers", spy):
        result = parse_qwen3_coder_output(text, tools, _work_limit=None)

    # parsing is unchanged: the quoted openers are still part of the value's text.
    assert [call.name for call in result.calls] == ["store"]
    assert json.loads(result.calls[0].arguments)["data"] == quoted
    # and not one of the 50k undeclared openers was retained.
    assert "nope" not in captured, sorted(captured)
    assert set(captured) == {"data"}, sorted(captured)
    # the declared offsets cannot be dropped or collapsed, because a lookup asks for the greatest
    # one below a scope end not known until the call is parsed. so the only saving available is how
    # they are held: a compact integer array rather than a list of boxed python ints.
    offsets = captured["data"]
    assert isinstance(offsets, array), type(offsets)
    assert len(offsets) == 50_001, len(offsets)
    assert list(offsets) == sorted(offsets), "bisect requires increasing order"
    # the saving is the boxed integers, which `getsizeof` does not charge to the list holding them,
    # so the equivalent list is weighed with its elements rather than by its pointer array alone.
    boxed = list(offsets)
    assert sys.getsizeof(offsets) * 4 < sys.getsizeof(boxed) + sum(
        sys.getsizeof(offset) for offset in boxed
    ), sys.getsizeof(offsets)


def test_the_opener_index_sees_every_name_a_replay_probe_can_declare() -> None:
    """the index's name run is the parser's, not a public declaration's.

    a replay probe's parameter names are the keys of a historical call, which never pass
    `_identifier_name`. indexing only the charset a declaration may hold left the opener such a key
    spells invisible, so a value quoting it read as never handing off and a history the template
    renders exactly was rejected as unreplayable. the index is filtered against the declared names
    either way, so seeing these costs no offsets a reader cannot ask for.
    """
    for key in ("weird key", "bad.name", "bad<name", "", "é", "x" * 200, "slash/key", "a\tb"):
        # `a`'s value quotes `a`'s own opener, so the index must hold both names to settle the
        # boundary. the second key is what the narrow charset dropped, and dropping it left the
        # value's quoted opener looking like a handoff to a name nothing declared.
        arguments = {"a": "</parameter><parameter=a>", key: "z"}
        history = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "hist", "arguments": json.dumps(arguments)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        # no current declaration, so the probe's names come straight from the historical keys.
        validate_tool_history_replay(history, None, error_type=OpenAIRequestError)
        assert tool_calls_module._index_parameter_openers(
            f"<parameter={key}>", 0, frozenset({key}), [10**9]
        ) == {key: array("q", [0])}, key

    # a name carrying `>` is still not one the parser can read back: it ends the name at that `>`.
    # indexing the longer run would record an opener no lookup can ever match.
    assert (
        tool_calls_module._index_parameter_openers(
            "<parameter=a>b>", 0, frozenset({"a>b"}), [10**9]
        )
        == {}
    )

    # a name runs to the first `>`, so one opener's name can span a later opener's marker. the
    # inner one is a real position a reader may ask for, so the scan resumes after this opener's
    # own marker rather than after the name it took. resuming past the name drops the inner offset
    # while leaving every parse result unchanged, which is why it is pinned on the index directly.
    for nested, offset in (("<parameter=x<parameter=a>", 12), ("<parameter=<parameter=a>", 11)):
        assert tool_calls_module._index_parameter_openers(nested, 0, frozenset({"a"}), [10**9]) == {
            "a": array("q", [offset])
        }, nested

    # the same nesting where the inner name is empty, so the delimiter sits exactly at the name
    # start. a scan that treats a zero-width name as nothing to record drops a declarable opener.
    assert tool_calls_module._index_parameter_openers(
        "<parameter=<parameter=>", 0, frozenset({""}), [10**9]
    ) == {"": array("q", [11])}


def test_unterminated_openers_do_not_rescan_the_tail_for_each_one() -> None:
    """reading the name by its delimiter must not retry the run at every opener it cannot end.

    the index reads a name the way the parser does, to the next `>`. expressing that as a pattern
    makes the engine retry the whole remaining tail from every `<parameter=` that never terminates,
    which is quadratic: a 132 KiB run of them cost seconds. this runs synchronously on model output
    in both serving paths, and the work budget charges the span once before the scan, so it cannot
    stop it. a malformed generation must stay proportional to its length.
    """
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                        "required": ["data"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )

    def elapsed(count: int, tail: str) -> float:
        text = "<tool_call><function=store>" + "<parameter=" * count + tail
        start = time.perf_counter()
        assert parse_qwen3_coder_output(text, tools).calls == ()
        return time.perf_counter() - start

    # both shapes, because they exercise different halves of the scan. with no delimiter the first
    # search fails and the loop stops, so it cannot show a delimiter search that restarts; with one
    # far delimiter every opener shares it, which is what re-finding it per opener makes quadratic.
    for tail in ("", ">"):
        # quadratic growth quadruples for each doubling, so a linear scan is far inside this bound.
        # the ratio is asserted rather than an absolute time, which would only measure the runner.
        small, large = elapsed(4_000, tail), elapsed(16_000, tail)
        assert large < 8 * max(small, 0.001), (tail, small, large)


def test_a_declared_name_as_long_as_the_run_does_not_widen_the_scan() -> None:
    """a declaration is untrusted too, so no opener may be copied out to be compared.

    a replay probe declares the keys of a historical call, and a key carries anything, so one key
    can be as long as the whole generation. bounding the copy by the longest declared name is
    therefore no bound at all: such a key makes every opener in a run wide enough to cut out, and a
    run sharing one far delimiter spells out a name per opener that shrinks by a constant. copying
    them costs the square of the run's length before a single comparison, on the synchronous parse
    path, and the work budget charges the span once beforehand so it cannot stop it.
    """

    def elapsed(count: int) -> float:
        # the key is the run itself, so it grows with the text. a key of some fixed width would
        # leave only the openers within that width of the delimiter wide enough to cut out, which
        # is a constant number of them however long the run gets, and would measure nothing.
        text = "<parameter=" * count + ">"
        key = text[len("<parameter=") : -1]
        start = time.perf_counter()
        # the first opener is the only one whose name is the whole key; the rest are shorter.
        assert tool_calls_module._index_parameter_openers(text, 0, frozenset({key}), [10**9]) == {
            key: array("q", [0])
        }
        return time.perf_counter() - start

    # quadratic growth quadruples for each doubling; reading only declared widths stays inside it.
    small, large = elapsed(8_000), elapsed(32_000)
    assert large < 8 * max(small, 0.001), (small, large)


def test_a_wide_schema_does_not_cost_its_width_at_every_opener() -> None:
    """settling an opener must not walk the declared names that happen to share its width.

    a schema may declare hundreds of parameters, and nothing stops them being the same length, so
    comparing an opener against each name of its width costs their count at every opener. that is
    the declaration's width times the generation's length, which both an ordinary wide schema and
    an untrusted generation reach. one hash of the name settles it against all of them at once.
    """
    # one width, so a per-name comparison has no narrowing left to do. the openers match nothing,
    # which is the worst case for it: every name of the width is compared before the miss.
    declared = frozenset(f"{index:012d}" for index in range(2_000))
    text = "<parameter=zzzzzzzzzzzz>" * 20_000

    def elapsed(names: frozenset[str]) -> float:
        start = time.perf_counter()
        assert tool_calls_module._index_parameter_openers(text, 0, names, [10**9]) == {}
        return time.perf_counter() - start

    # the same text against a schema 100x narrower. a scan that walks the width's names grows with
    # the declaration; hashing the name once does not, so the two stay within an order of each other.
    narrow, wide = elapsed(frozenset(sorted(declared)[:20])), elapsed(declared)
    assert wide < 10 * max(narrow, 0.001), (narrow, wide)


def test_bounding_the_read_names_does_not_spend_the_parse_budget() -> None:
    """holding the copying down must not charge the caller for the span it already charged.

    the index charges its whole span once, up front. a value quoting openers whose widths a wide
    declaration happens to hold makes it read a name at each of them, and charging those reads
    again bills the same characters twice: a call well inside the budget then reports exhausted
    and falls back to text, silently losing a tool call the model emitted correctly.
    """
    # widths the declaration holds, so every quoted opener in the value is read rather than skipped.
    names = ["data", *("a" * width for width in (11, 22, 33, 44, 55))]
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {name: {"type": "string"} for name in names},
                        "required": ["data"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    # each group of six markers shares the `>` that ends it, so the widths run 11 through 66 and
    # five of those six are declared. the value is ordinary text to the parser, and must survive.
    value = ("<parameter=" * 6 + ">") * 8
    text = f"<tool_call><function=store><parameter=data>{value}</parameter></function></tool_call>"

    parsed = parse_qwen3_coder_output(text, tools)

    assert len(parsed.calls) == 1, parsed
    assert json.loads(parsed.calls[0].arguments) == {"data": value}


def test_inert_enum_delimiters_are_not_stepped_one_at_a_time() -> None:
    """an enum value full of delimiters that cannot close it must not cost one step each.

    only a ``</parameter>`` followed, after whitespace, by the next parameter or the function end
    can close a value. `normalize_tools` runs synchronously on the packaged and hosted request
    paths, so stepping to every inert delimiter in a megabyte-scale enum holds the event loop.
    """
    from flash.serve.request.tool_calls import _string_enum_conflicts_with_tool_grammar

    # a delimiter run that never becomes viable: each one is followed by an ordinary character.
    inert = "</parameter>x" * 100_000
    assert not _string_enum_conflicts_with_tool_grammar(inert)

    # the same run is rejected as soon as one delimiter is actually followed by a parameter, so
    # the fast path did not buy its speed by giving up the check.
    assert _string_enum_conflicts_with_tool_grammar(inert + "</parameter><parameter=a>")
    assert _string_enum_conflicts_with_tool_grammar(inert + "</parameter>  </function>")

    # whitespace between the delimiter and the marker still counts, and a partial marker does not.
    assert _string_enum_conflicts_with_tool_grammar("</parameter>\n\t<parameter=a>")
    assert not _string_enum_conflicts_with_tool_grammar("</parameter>x<parameter=a>")
    assert not _string_enum_conflicts_with_tool_grammar("</parameter>")

    # a declaration carrying the inert run normalizes rather than being rejected or hanging.
    normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {"mode": {"type": "string", "enum": [inert]}},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        error_type=OpenAIRequestError,
    )


def test_streamed_candidate_accumulates_without_recopying_the_buffer() -> None:
    """a buffered candidate must not be recopied once per delta.

    a candidate is retained whole because a malformed later fragment falls back to the exact
    text, so the buffer grows to the length of the call. concatenating onto a string per delta
    copies the whole buffer each time, which costs work in the square of the candidate length.
    """
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "store",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                        "required": ["data"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )

    parser = ToolCallStreamParser(tools)
    parser.feed("<tool_call><function=store><parameter=data>\n")
    chunk = "a" * 40
    for _ in range(2_000):
        parser.feed(chunk)

    # timing this would be flaky, so the structure is asserted instead: each delta is retained as
    # its own part rather than concatenated onto one growing string. a string buffer would show a
    # single part holding everything, which is exactly the shape that copies per delta.
    assert len(parser._parts) == 2_001, len(parser._parts)
    assert sum(len(part) for part in parser._parts) == 44 + 2_000 * 40

    # the joined text is still exact, so buffering by parts changed no observable content.
    assert parser._pending == "<tool_call><function=store><parameter=data>\n" + chunk * 2_000

    # and the retained text is still exact, so the fallback path is unaffected by the buffering.
    parser = ToolCallStreamParser(tools)
    emitted = "".join(parser.feed(part) for part in ("before <tool_", "call> not a call"))
    assert emitted == "before "
    assert parser.finish().content == "<tool_call> not a call"
