"""Option lowering must never silently weaken or contradict the user's schema."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import jsonschema
import pytest

from flash.content.structured_options import _compact_json_grammar, vllm_structured_kwargs


@pytest.mark.parametrize("nested", [False, True])
def test_compact_json_rejects_backend_fallback_before_compiling(monkeypatch, nested):
    schema = {"type": "number", "multipleOf": 0.5}
    if nested:
        schema = {"type": "object", "properties": {"value": schema}}
    check = Mock(side_effect=_recursive_backend_check)
    compile_schema = Mock(side_effect=AssertionError("unsupported schema reached xgrammar"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.structured_output.backend_xgrammar",
        SimpleNamespace(has_xgrammar_unsupported_json_features=check),
    )
    monkeypatch.setitem(
        sys.modules,
        "xgrammar",
        SimpleNamespace(Grammar=SimpleNamespace(from_json_schema=compile_schema)),
    )
    _compact_json_grammar.cache_clear()
    try:
        with pytest.raises(ValueError, match="disable_any_whitespace requires an xgrammar"):
            vllm_structured_kwargs({"json": schema, "disable_any_whitespace": True})
        check.assert_any_call({"type": "number", "multipleOf": 0.5})
        compile_schema.assert_not_called()
        # Requests without compact formatting still reach vLLM as JSON so it can choose a backend.
        assert vllm_structured_kwargs({"json": schema}) == {"json": schema}
    finally:
        _compact_json_grammar.cache_clear()


_UNSAFE_SCHEMAS = [
    (
        {
            "type": "object",
            "properties": {"x": {}, "extra": {}},
            "not": {"properties": {"x": {"const": 1}}},
        },
        {"x": 1, "extra": 2},
        False,
        "not",
    ),
    (
        {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]},
            ]
        },
        {"a": "x", "b": "y"},
        True,
        "allOf",
    ),
    (
        {
            "type": "object",
            "properties": {"kind": {}, "other": {}, "answer": {}},
            "if": {"properties": {"kind": {"const": "x"}}},
            "then": {"required": ["answer"]},
        },
        {"kind": "x", "other": 1},
        False,
        "if",
    ),
]


@pytest.mark.parametrize(("schema", "instance", "valid", "keyword"), _UNSAFE_SCHEMAS)
@pytest.mark.parametrize("nested", [False, True])
def test_unsafe_property_rewrites_are_rejected_at_public_and_runtime_boundaries(
    schema, instance, valid, keyword, nested
):
    from flash.schema import spec_from_dict
    from flash.serving.src.io.structured_outputs import normalize_structured_outputs
    from tests.test_structured_outputs import _raw

    assert jsonschema.Draft202012Validator(schema).is_valid(instance) is valid
    if nested:
        schema = {"type": "object", "properties": {"nested": schema}}
    original = copy.deepcopy(schema)
    for compact in (False, True):
        spec = {
            "json": schema,
            "disable_additional_properties": True,
            "disable_any_whitespace": compact,
        }
        message = f"cannot safely rewrite schemas containing {keyword}"
        with pytest.raises(ValueError, match=message):
            vllm_structured_kwargs(spec)
        with pytest.raises(ValueError, match=message):
            normalize_structured_outputs(spec)
        with pytest.raises(ValueError, match=message):
            spec_from_dict(_raw(spec))
        with pytest.raises(ValueError, match=message):
            vllm_structured_kwargs({**spec, "json": json.dumps(schema)})
    assert schema == original
    assert vllm_structured_kwargs({"json": schema, "disable_additional_properties": False}) == {
        "json": schema
    }


@pytest.mark.parametrize(
    "keyword",
    [
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "dependencies",
        "$ref",
        "$dynamicRef",
    ],
)
def test_composition_keywords_in_property_names_and_data_are_not_schema_keywords(keyword):
    schema = {
        "type": "object",
        "properties": {keyword: {"type": "string"}},
        "default": {keyword: {}},
        "examples": [{keyword: {}}],
        "enum": [{keyword: "value"}],
    }
    result = vllm_structured_kwargs({"json": schema, "disable_additional_properties": True})
    assert result["json"] == {**schema, "additionalProperties": False}


def test_simple_referenced_objects_still_close_without_mutating_input():
    schema = {
        "type": "object",
        "properties": {"nested": {"$ref": "#/$defs/entry"}},
        "$defs": {"entry": {"type": "object", "properties": {"x": {"type": "number"}}}},
    }
    result = vllm_structured_kwargs({"json": schema, "disable_additional_properties": True})
    validator = jsonschema.Draft202012Validator(result["json"])
    assert validator.is_valid({"nested": {"x": 1}})
    assert not validator.is_valid({"nested": {"x": 1, "extra": 2}})
    assert not validator.is_valid({"nested": {"x": 1}, "extra": 2})
    assert "additionalProperties" not in schema["$defs"]["entry"]


@pytest.mark.parametrize("reference", ["$ref", "$dynamicRef"])
@pytest.mark.parametrize(
    "constraint", [{"type": "object"}, {"properties": {}}, {"required": ["a"]}]
)
@pytest.mark.parametrize("nested", [False, True])
def test_reference_with_local_object_constraints_is_rejected(reference, constraint, nested):
    from flash.schema import spec_from_dict
    from flash.serving.src.io.structured_outputs import normalize_structured_outputs
    from tests.test_structured_outputs import _raw

    node = {reference: "#/$defs/entry", **constraint}
    schema = {"type": "object", "properties": {"nested": node}} if nested else node.copy()
    schema["$defs"] = {"entry": {"type": "object", "properties": {"a": {"type": "string"}}}}
    instance = {"nested": {"a": "value"}} if nested else {"a": "value"}
    assert jsonschema.Draft202012Validator(schema).is_valid(instance)
    original = copy.deepcopy(schema)
    spec = {"json": schema, "disable_additional_properties": True}
    for validate in (vllm_structured_kwargs, normalize_structured_outputs):
        with pytest.raises(ValueError, match="cannot safely rewrite"):
            validate(spec)
    with pytest.raises(ValueError, match="cannot safely rewrite"):
        spec_from_dict(_raw(spec))
    with pytest.raises(ValueError, match="cannot safely rewrite"):
        vllm_structured_kwargs({**spec, "json": json.dumps(schema)})
    assert schema == original


@pytest.mark.parametrize(
    ("constraint", "instance"),
    [
        ({"required": ["id"]}, {"id": "value", "extra": "value"}),
        ({"minProperties": 1}, {"extra": "value"}),
        ({"maxProperties": 2}, {"extra": "value"}),
        ({"propertyNames": {"pattern": "^id"}}, {"id_extra": "value"}),
        ({"dependentRequired": {"id": ["other"]}}, {"extra": "value"}),
        ({"unevaluatedProperties": {"type": "string"}}, {"extra": "value"}),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_implicit_objects_forbid_additional_properties(constraint, instance, nested):
    schema = {"type": "object", "properties": {"nested": constraint}} if nested else constraint
    instance = {"nested": instance} if nested else instance
    original = copy.deepcopy(schema)
    assert jsonschema.Draft202012Validator(schema).is_valid(instance)
    lowered = vllm_structured_kwargs({"json": schema, "disable_additional_properties": True})
    assert not jsonschema.Draft202012Validator(lowered["json"]).is_valid(instance)
    assert schema == original


def _recursive_backend_check(value):
    # model the installed backend's recursive scan, including its literal-data false positives.
    if isinstance(value, dict):
        if value.get("type") == "number" and "multipleOf" in value:
            return True
        if value.get("type") == "array" and "contains" in value:
            return True
        if value.get("type") == "object" and "propertyNames" in value:
            return True
        return any(_recursive_backend_check(child) for child in value.values())
    if isinstance(value, list):
        return any(_recursive_backend_check(child) for child in value)
    return False


@pytest.fixture
def compact_backend(monkeypatch):
    check = Mock(side_effect=_recursive_backend_check)
    compile_schema = Mock(return_value='root ::= "{}"')
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.structured_output.backend_xgrammar",
        SimpleNamespace(has_xgrammar_unsupported_json_features=check),
    )
    monkeypatch.setitem(
        sys.modules,
        "xgrammar",
        SimpleNamespace(Grammar=SimpleNamespace(from_json_schema=compile_schema)),
    )
    _compact_json_grammar.cache_clear()
    yield check, compile_schema
    _compact_json_grammar.cache_clear()


@pytest.mark.parametrize("keyword", ["default", "examples", "enum", "const", "x-metadata"])
@pytest.mark.parametrize("nested", [False, True])
def test_compact_json_does_not_check_literal_data_as_schemas(compact_backend, keyword, nested):
    literal = {"type": "number", "multipleOf": 0.5}
    schema = {
        "type": "object",
        "properties": {"type": {"type": "string"}, "multipleOf": {"type": "number"}},
        keyword: [literal] if keyword in {"examples", "enum"} else literal,
    }
    if nested:
        schema = {"type": "object", "properties": {"nested": schema}}
    original = copy.deepcopy(schema)
    assert _recursive_backend_check(schema)
    result = vllm_structured_kwargs({"json": schema, "disable_any_whitespace": True})
    assert result == {"grammar": 'root ::= "{}"'}
    _, compile_schema = compact_backend
    assert json.loads(compile_schema.call_args.args[0]) == original
    assert schema == original


@pytest.mark.parametrize(
    "schema",
    [
        {"properties": {"value": {"type": "number", "multipleOf": 0.5}}},
        {"$defs": {"value": {"type": "number", "multipleOf": 0.5}}},
        {"items": {"type": "number", "multipleOf": 0.5}},
        {"items": [{"type": "number", "multipleOf": 0.5}]},
        {"allOf": [{"type": "number", "multipleOf": 0.5}]},
        {"dependencies": {"value": {"type": "number", "multipleOf": 0.5}}},
        {"type": "array", "contains": {"type": "integer"}},
        {"type": "object", "propertyNames": {"pattern": "^id"}},
    ],
)
def test_compact_json_still_checks_actual_nested_schemas(compact_backend, schema):
    with pytest.raises(ValueError, match="requires an xgrammar-compatible JSON schema"):
        vllm_structured_kwargs({"json": schema, "disable_any_whitespace": True})
    _, compile_schema = compact_backend
    compile_schema.assert_not_called()
