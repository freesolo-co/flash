"""request options must affect the engine grammar, not ignored vllm fields."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flash.content.structured_options import (
    _compact_json_grammar,
    validate_structured_options,
    vllm_structured_kwargs,
)


@pytest.fixture
def backend_feature_check(monkeypatch):
    check = Mock(return_value=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.structured_output.backend_xgrammar",
        SimpleNamespace(has_xgrammar_unsupported_json_features=check),
    )
    _compact_json_grammar.cache_clear()
    yield check
    _compact_json_grammar.cache_clear()


def test_compact_json_is_lowered_to_cached_explicit_grammar(monkeypatch, backend_feature_check):
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
    backend_feature_check.assert_called_once_with(spec["json"])
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


def test_compiler_errors_remain_request_errors(monkeypatch, backend_feature_check):
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


# xgrammar (<= 0.2.9) prints a length-limited string's repetition unit without an escape branch.
_LIMITED_GRAMMAR = (
    'root ::= (("{" [ \\n\\t]* "\\"c\\"" [ \\n\\t]* ":" [ \\n\\t]* root_prop_0 [ \\n\\t]* "}"))\n'
    'root_prop_0 ::= (("\\"" root_prop_0_characters{1, -1} "\\""))\n'
    'root_prop_0_characters ::= (([^\\"\\\\\\r\\n]))\n'
)


def _limited(prop=None):
    prop = prop or {"type": "string", "minLength": 1}
    return {"type": "object", "properties": {"c": prop}, "required": ["c"]}


@pytest.fixture
def fake_xgrammar(monkeypatch, backend_feature_check):
    from flash.content.structured_options import _escape_safe_json_grammar

    calls = []

    def compile_schema(schema, **options):
        calls.append((json.loads(schema), options))
        return _LIMITED_GRAMMAR

    monkeypatch.setitem(
        sys.modules,
        "xgrammar",
        SimpleNamespace(Grammar=SimpleNamespace(from_json_schema=compile_schema)),
    )
    _escape_safe_json_grammar.cache_clear()
    yield calls
    _escape_safe_json_grammar.cache_clear()


def test_length_limited_strings_keep_json_escapes(fake_xgrammar):
    """minLength/maxLength strings must still decode escapes (e.g. code inside a JSON answer)."""
    spec = {"json": _limited()}
    original = copy.deepcopy(spec)
    lowered = vllm_structured_kwargs(spec)
    grammar = lowered["grammar"]
    assert set(lowered) == {"grammar"}
    assert 'root_prop_0_characters ::= (([^\\"\\\\\\r\\n]) | ("\\\\" flash_json_escape))' in grammar
    assert grammar.rstrip().endswith(
        'flash_json_escape ::= (([\\"\\\\/bfnrt]) | '
        '("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))'
    )
    # vLLM's default JSON compilation (any whitespace) is kept; only the character unit changes.
    assert fake_xgrammar == [(spec["json"], {"any_whitespace": True})]
    assert vllm_structured_kwargs(spec) == lowered  # cached
    assert len(fake_xgrammar) == 1
    assert spec == original


@pytest.mark.parametrize(
    "prop",
    [
        {"type": "string", "maxLength": 64},
        {"type": ["string", "null"], "minLength": 1},
    ],
)
def test_any_length_limited_string_type_is_lowered(fake_xgrammar, prop):
    assert "flash_json_escape" in vllm_structured_kwargs({"json": _limited(prop)})["grammar"]


def test_unlimited_strings_stay_plain_json(fake_xgrammar):
    schema = _limited({"type": "string"})
    assert vllm_structured_kwargs({"json": schema}) == {"json": schema}
    assert fake_xgrammar == []


def test_length_limited_schema_xgrammar_cannot_compile_is_left_to_vllm(
    fake_xgrammar, backend_feature_check
):
    backend_feature_check.return_value = True
    schema = _limited()
    assert vllm_structured_kwargs({"json": schema}) == {"json": schema}
    assert fake_xgrammar == []


def test_compact_length_limited_strings_also_keep_escapes(fake_xgrammar):
    lowered = vllm_structured_kwargs({"json": _limited(), "disable_any_whitespace": True})
    assert "flash_json_escape" in lowered["grammar"]
    assert fake_xgrammar[0][1] == {
        "any_whitespace": False,
        "indent": None,
        "separators": (",", ":"),
    }
