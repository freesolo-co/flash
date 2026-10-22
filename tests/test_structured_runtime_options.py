"""request options must affect the engine grammar, not ignored vllm fields."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from flash.content.structured_options import (
    _compact_json_grammar,
    validate_structured_options,
    vllm_structured_kwargs,
)


def test_compact_json_is_lowered_to_cached_explicit_grammar(monkeypatch):
    calls = []

    def compile_schema(schema, **options):
        calls.append((json.loads(schema), options))
        return 'root ::= "{}"'

    _compact_json_grammar.cache_clear()
    monkeypatch.setitem(
        sys.modules,
        "xgrammar",
        SimpleNamespace(Grammar=SimpleNamespace(from_json_schema=compile_schema)),
    )
    spec = {"json": {"type": "object"}, "disable_any_whitespace": True}
    original = copy.deepcopy(spec)
    assert vllm_structured_kwargs(spec) == {"grammar": 'root ::= "{}"'}
    assert vllm_structured_kwargs(spec) == {"grammar": 'root ::= "{}"'}
    assert calls == [
        (
            {"type": "object"},
            {
                "any_whitespace": False,
                "indent": None,
                "separators": (",", ":"),
            },
        )
    ]
    assert spec == original
    _compact_json_grammar.cache_clear()


def test_additional_properties_are_lowered_only_in_schema_locations():
    schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "nested": {"type": "object", "properties": {"x": {"type": "number"}}},
            "items": {"type": "array", "items": {"type": "object", "properties": {}}},
        },
        "additionalProperties": True,
        "$defs": {"entry": {"type": ["object", "null"], "properties": {}}},
        "default": {"type": "object", "properties": {"customer": "data"}},
    }
    original = copy.deepcopy(schema)
    lowered = vllm_structured_kwargs({"json": schema, "disable_additional_properties": True})
    result = lowered["json"]
    assert result["additionalProperties"] is False
    assert result["properties"]["nested"]["additionalProperties"] is False
    assert result["properties"]["items"]["items"]["additionalProperties"] is False
    assert result["$defs"]["entry"]["additionalProperties"] is False
    assert result["properties"]["type"] == {"type": "string"}
    assert result["default"] == original["default"]
    assert schema == original


def test_json_object_can_forbid_undeclared_properties():
    assert vllm_structured_kwargs(
        {
            "json_object": True,
            "disable_additional_properties": True,
        }
    ) == {"json": {"type": "object", "additionalProperties": False}}


@pytest.mark.parametrize("constraint", [{"choice": ["yes", "no"]}, {"regex": "[a-z]+"}])
def test_inapplicable_options_are_rejected_instead_of_ignored(constraint):
    for option in ("disable_any_whitespace", "disable_additional_properties"):
        with pytest.raises(ValueError, match="require a JSON constraint"):
            vllm_structured_kwargs({**constraint, option: True})
    assert vllm_structured_kwargs({**constraint, "disable_any_whitespace": False}) == constraint


def test_custom_whitespace_is_rejected_at_both_public_validation_boundaries():
    from flash.schema import spec_from_dict
    from flash.serving.src.io.structured_outputs import normalize_structured_outputs
    from tests.test_structured_outputs import _raw

    spec = {"json_object": True, "whitespace_pattern": " ?"}
    with pytest.raises(ValueError, match="whitespace_pattern is not supported"):
        validate_structured_options(spec)
    with pytest.raises(ValueError, match="whitespace_pattern is not supported"):
        normalize_structured_outputs(spec)
    with pytest.raises(ValueError, match="whitespace_pattern is not supported"):
        spec_from_dict(_raw(spec))


def test_compiler_errors_remain_request_errors(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("invalid schema")

    _compact_json_grammar.cache_clear()
    monkeypatch.setitem(
        sys.modules, "xgrammar", SimpleNamespace(Grammar=SimpleNamespace(from_json_schema=fail))
    )
    with pytest.raises(ValueError, match="invalid compact JSON constraint"):
        vllm_structured_kwargs({"json": {"type": "invalid"}, "disable_any_whitespace": True})


def test_nested_runtime_configuration_becomes_plain_json():
    from collections import UserDict, UserList

    schema = UserDict(
        {
            "type": "object",
            "properties": UserDict(
                {"answer": UserDict({"type": "string", "enum": UserList(["red", "blue"])})}
            ),
            "required": UserList(["answer"]),
        }
    )
    lowered = vllm_structured_kwargs(UserDict({"json": schema}))
    assert json.loads(json.dumps(lowered)) == {
        "json": {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["red", "blue"]}},
            "required": ["answer"],
        }
    }
    assert isinstance(schema["properties"], UserDict)
    assert vllm_structured_kwargs({"choice": UserList(["red", "blue"])}) == {
        "choice": ["red", "blue"]
    }
