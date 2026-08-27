"""Structured-outputs spec normalization (src/structured_outputs.py) and the schema fields that
carry it.

Every entry point funnels caller specs through ``normalize_structured_outputs``, so the accepted
forms, the aliasing, the explicit-off markers, and the error cases are all pinned here. The
normalizer must also be IDEMPOTENT on its own outputs: the router validates a GenerateRequest
(which normalizes), forwards it over Modal RPC as a dict, and the engine re-validates that dict —
a canonical spec has to survive the second pass unchanged.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from flash.serving.src.io.schemas import AdapterRecord, GenerateRequest
from flash.serving.src.io.structured_outputs import (
    StructuredOutputsError,
    normalize_structured_outputs,
)
from tests.serving.checkpoint_fixtures import checkpoint_record

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}


# --- not specified vs explicitly off ---------------------------------------------------------


def test_none_means_not_specified():
    assert normalize_structured_outputs(None) is None


@pytest.mark.parametrize(
    "off",
    [False, "", "none", "NONE", " none ", "text", "TEXT", " text ", {}],
)
def test_explicit_off_forms_normalize_to_empty_dict(off):
    # {} (not None) so a per-call "off" can override a per-adapter default downstream.
    assert normalize_structured_outputs(off) == {}


def test_dict_of_only_nulls_is_off():
    # Null-valued keys are "unset" (the StructuredOutputsParams defaults), so nothing remains.
    assert normalize_structured_outputs({"json": None, "regex": None}) == {}


# --- string shorthands ------------------------------------------------------------------------


@pytest.mark.parametrize("s", ["json", "json_object", "JSON", " Json_Object "])
def test_json_strings_mean_json_object_mode(s):
    assert normalize_structured_outputs(s) == {"json_object": True}


def test_json_string_spec_is_parsed_and_recursed():
    assert normalize_structured_outputs(json.dumps(SCHEMA)) == {"json": SCHEMA}
    assert normalize_structured_outputs('{"choice": ["a", "b"]}') == {"choice": ["a", "b"]}


def test_unparseable_string_is_an_error():
    with pytest.raises(StructuredOutputsError, match="valid JSON"):
        normalize_structured_outputs("not json {")


@pytest.mark.parametrize(
    "value",
    [
        '{"type": "number", "minimum": NaN}',
        {"json": '{"type": "number", "minimum": Infinity}'},
    ],
)
def test_json_strings_reject_non_finite_constants(value):
    with pytest.raises(StructuredOutputsError, match="json does not define"):
        normalize_structured_outputs(value)


@pytest.mark.parametrize(
    "value",
    [
        {"type": "number", "minimum": float("nan")},
        {"json": {"type": "number", "maximum": float("inf")}},
    ],
    ids=["raw-schema", "canonical-json"],
)
def test_decoded_dicts_reject_nested_non_finite_constants(value):
    with pytest.raises(StructuredOutputsError, match="json does not define"):
        normalize_structured_outputs(value)


@pytest.mark.parametrize("s", ["[1, 2]", "42", "null", '"json-ish"'])
def test_json_string_must_decode_to_an_object(s):
    with pytest.raises(StructuredOutputsError, match="decode to an object"):
        normalize_structured_outputs(s)


# --- canonical dicts and aliases --------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"json": SCHEMA}, {"json": SCHEMA}),
        ({"json": json.dumps(SCHEMA)}, {"json": SCHEMA}),  # schema given as a JSON string
        ({"json_schema": SCHEMA}, {"json": SCHEMA}),
        ({"schema": SCHEMA}, {"json": SCHEMA}),
        ({"regex": r"\d+"}, {"regex": r"\d+"}),
        ({"choice": ["a", "b"]}, {"choice": ["a", "b"]}),
        ({"choices": ["a", "b"]}, {"choice": ["a", "b"]}),
        ({"json_object": True}, {"json_object": True}),
    ],
)
def test_canonical_constraints_and_aliases(spec, expected):
    assert normalize_structured_outputs(spec) == expected


def test_options_pass_through_with_the_constraint():
    spec = {
        "json": SCHEMA,
        "disable_any_whitespace": True,
        "disable_additional_properties": True,
        "whitespace_pattern": r"[\n\t ]*",
    }
    assert normalize_structured_outputs(spec) == spec


def test_null_constraint_alongside_a_real_one_is_dropped():
    assert normalize_structured_outputs({"json": None, "regex": r"\d+"}) == {"regex": r"\d+"}


def test_stray_type_key_next_to_a_constraint_is_rejected():
    # With the OpenAI response_format form gone, "type" is no longer special: passing it next to a
    # real constraint is an unknown key (parity with the flash normalizer).
    with pytest.raises(StructuredOutputsError, match="unknown structured outputs key"):
        normalize_structured_outputs({"type": "spec", "regex": "a+"})


# --- raw JSON schema dicts --------------------------------------------------------------------


def test_raw_json_schema_dict_becomes_a_json_constraint():
    # No constraint keys, so the whole dict lands in the raw-schema branch.
    assert normalize_structured_outputs(SCHEMA) == {"json": SCHEMA}


def test_raw_schema_keeps_null_values_inside_the_schema():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "default": None}
    assert normalize_structured_outputs(schema) == {"json": schema}


def test_enum_schema_is_a_json_constraint():
    # An enum inside the schema is ordinary schema content: it rides through untouched as a json
    # constraint for vLLM/xgrammar to enforce at decode time.
    enum_schema = {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
        "required": ["color"],
    }
    assert normalize_structured_outputs(enum_schema) == {"json": enum_schema}
    # A bare enum ({"enum": [...]}) is itself a valid JSON Schema (no constraint keys -> raw schema).
    assert normalize_structured_outputs({"enum": ["yes", "no"]}) == {
        "json": {"enum": ["yes", "no"]}
    }


def test_former_openai_json_object_mode_is_now_a_raw_schema():
    # The OpenAI response_format {"type": "json_object"} mode is no longer detected; with no
    # constraint keys the dict is just wrapped as a raw JSON schema.
    assert normalize_structured_outputs({"type": "json_object"}) == {
        "json": {"type": "json_object"}
    }


# --- error cases ------------------------------------------------------------------------------


def test_two_constraints_is_an_error():
    with pytest.raises(StructuredOutputsError, match="exactly one constraint"):
        normalize_structured_outputs({"json": SCHEMA, "regex": r"\d+"})


def test_aliased_duplicate_constraint_is_an_error():
    with pytest.raises(StructuredOutputsError, match="twice"):
        normalize_structured_outputs({"json": SCHEMA, "json_schema": SCHEMA})


def test_option_without_a_constraint_is_an_error():
    with pytest.raises(StructuredOutputsError, match="exactly one constraint"):
        normalize_structured_outputs({"disable_any_whitespace": True})


def test_empty_choice_list_is_an_error():
    with pytest.raises(StructuredOutputsError, match="non-empty list"):
        normalize_structured_outputs({"choice": []})


def test_non_string_choice_entries_are_an_error():
    with pytest.raises(StructuredOutputsError, match="strings"):
        normalize_structured_outputs({"choice": ["a", 1]})


def test_non_string_pattern_constraints_are_an_error():
    with pytest.raises(StructuredOutputsError, match="non-empty string"):
        normalize_structured_outputs({"regex": 42})
    with pytest.raises(StructuredOutputsError, match="non-empty string"):
        normalize_structured_outputs({"regex": "  "})


@pytest.mark.parametrize("bad", [{"grammar": "root ::= digit"}, {"structural_tag": '{"x": 1}'}])
def test_grammar_and_structural_tag_are_rejected(bad):
    # Unsupported constraints must be rejected, not silently treated as a raw JSON schema.
    with pytest.raises(StructuredOutputsError, match="not supported"):
        normalize_structured_outputs(bad)


def test_unknown_key_error_lists_allowed_keys():
    with pytest.raises(StructuredOutputsError, match=r"allowed keys.*json_object"):
        normalize_structured_outputs({"json": SCHEMA, "bogus": 1})


def test_json_object_false_is_an_error():
    with pytest.raises(StructuredOutputsError, match="json_object"):
        normalize_structured_outputs({"json_object": False})


def test_bare_true_is_an_error():
    with pytest.raises(StructuredOutputsError, match="cannot be `true`"):
        normalize_structured_outputs(True)


def test_json_constraint_of_wrong_type_is_an_error():
    with pytest.raises(StructuredOutputsError, match="JSON schema object"):
        normalize_structured_outputs({"json": 42})
    with pytest.raises(StructuredOutputsError, match="decode to a JSON schema object"):
        normalize_structured_outputs({"json": "[1, 2]"})


def test_bad_option_types_are_an_error():
    with pytest.raises(StructuredOutputsError, match="disable_any_whitespace"):
        normalize_structured_outputs({"json": SCHEMA, "disable_any_whitespace": "yes"})
    with pytest.raises(StructuredOutputsError, match="whitespace_pattern"):
        normalize_structured_outputs({"json": SCHEMA, "whitespace_pattern": 1})


@pytest.mark.parametrize("value", [42, 3.14, ["a"], ("a",)])
def test_unsupported_top_level_types_are_an_error(value):
    with pytest.raises(StructuredOutputsError, match="must be a dict"):
        normalize_structured_outputs(value)


# --- idempotency (router normalizes -> RPC dict -> engine re-normalizes) ----------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        SCHEMA,
        {"choices": ["a", "b"]},
        {"json": SCHEMA, "disable_any_whitespace": True},
        "json",
    ],
)
def test_normalization_is_idempotent(value):
    once = normalize_structured_outputs(value)
    assert normalize_structured_outputs(once) == once


# --- schema fields: GenerateRequest -----------------------------------------------------------


def test_generate_request_normalizes_structured_outputs():
    req = GenerateRequest.model_validate(
        {"adapter_id": "a", "prompt": "hi", "structured_outputs": "json_object"}
    )
    assert req.structured_outputs == {"json_object": True}


def test_generate_request_accepts_a_raw_json_schema():
    # `structured_outputs` is the only accepted key -- `response_format` was dropped as an alias
    # (see the test below). This was parametrized over two spellings until the alias was removed,
    # after which both cases named the same key and the second proved nothing.
    req = GenerateRequest.model_validate(
        {"adapter_id": "a", "prompt": "hi", "structured_outputs": SCHEMA}
    )
    assert req.structured_outputs == {"json": SCHEMA}


def test_generate_request_rejects_a_response_format_key():
    # response_format belongs only to the strict openai chat boundary. raw generation must reject
    # it rather than silently dispatching an unconstrained request.
    with pytest.raises(ValueError, match="extra_forbidden"):
        GenerateRequest.model_validate(
            {"adapter_id": "a", "prompt": "hi", "response_format": {"type": "json_object"}}
        )


def test_generate_request_keeps_explicit_off_distinct_from_absent():
    absent = GenerateRequest.model_validate({"adapter_id": "a", "prompt": "hi"})
    off = GenerateRequest.model_validate(
        {"adapter_id": "a", "prompt": "hi", "structured_outputs": False}
    )
    assert absent.structured_outputs is None  # inherit the adapter default
    assert off.structured_outputs == {}  # explicitly unconstrained, overrides the default


def test_generate_request_serializes_camel_case_for_the_rpc():
    # The router forwards model_dump(by_alias=True) over Modal RPC; the engine re-validates that
    # dict, so the canonical spec must ride the "structured_outputs" alias and survive re-parsing.
    req = GenerateRequest.model_validate(
        {"adapter_id": "a", "prompt": "hi", "structured_outputs": SCHEMA}
    )
    dumped = req.model_dump(by_alias=True)
    assert dumped["structured_outputs"] == {"json": SCHEMA}
    assert GenerateRequest.model_validate(dumped).structured_outputs == {"json": SCHEMA}


def test_generate_request_invalid_spec_is_a_validation_error():
    with pytest.raises(ValidationError, match="exactly one constraint"):
        GenerateRequest.model_validate(
            {
                "adapter_id": "a",
                "prompt": "hi",
                "structured_outputs": {"json": SCHEMA, "regex": "x"},
            }
        )


# --- schema fields: AdapterRecord -------------------------------------------------------------


def _record(**extra) -> AdapterRecord:
    base_model = extra.pop("base_model", "Qwen/Qwen3.5-9B")
    thinking = extra.pop("thinking", True)
    return checkpoint_record("a", base_model, thinking=thinking, **extra)


def test_adapter_record_normalizes_structured_outputs_default():
    # A bare JSON schema is normalized to the canonical {"json": schema} on the way in.
    rec = _record(structured_outputs=SCHEMA)
    assert rec.structured_outputs == {"json": SCHEMA}
    assert rec.model_dump(by_alias=True)["structured_outputs"] == {"json": SCHEMA}


def test_adapter_record_defaults_to_no_structured_outputs():
    assert _record().structured_outputs is None


@pytest.mark.parametrize("off", [False, "none", {}])
def test_adapter_record_collapses_explicit_off_to_none(off):
    # "no default" and "default: unconstrained" are the same thing for a record; None keeps the
    # persisted metadata free of a meaningless {} marker.
    assert _record(structured_outputs=off).structured_outputs is None


def test_adapter_record_invalid_spec_is_a_validation_error():
    with pytest.raises(ValidationError, match="non-empty list"):
        _record(structured_outputs={"choice": []})


def test_thinking_structured_default_requires_model_parser():
    with pytest.raises(ValidationError, match="base model with a reasoning parser"):
        _record(base_model="example/parserless-model", structured_outputs=SCHEMA)


def test_parserless_non_thinking_structured_default_is_allowed():
    record = _record(
        base_model="example/parserless-model",
        thinking=False,
        structured_outputs=SCHEMA,
    )
    assert record.structured_outputs == {"json": SCHEMA}
