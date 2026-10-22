import pytest

from flash.engine.worker.train.opd.orchestration.validation import (
    _require_exact_replay_constraint,
)


@pytest.mark.parametrize("name", ["type", "format", "multipleOf", "uniqueItems", "propertyNames"])
@pytest.mark.parametrize("container", ["properties", "$defs", "definitions"])
def test_schema_member_names_are_not_interpreted_as_keywords(name, container):
    schema = {"type": "object", container: {name: {"type": "string"}}}
    _require_exact_replay_constraint({"json": schema})


@pytest.mark.parametrize("keyword", ["default", "const", "examples", "enum"])
def test_minimum_supported_vllm_literal_value_limitation_remains_rejected(keyword):
    value = {"type": "string", "format": "date", "multipleOf": 3}
    schema = {"type": "object", keyword: [value] if keyword in ("examples", "enum") else value}
    # The supported vLLM floor, 0.11.0, also inspects literal values. Accepting these in preflight would
    # move the same rejection into the rented worker rather than make the schema usable.
    with pytest.raises(ValueError, match="guidance fallback"):
        _require_exact_replay_constraint({"json": schema})


@pytest.mark.parametrize(
    "keyword",
    [
        "properties",
        "$defs",
        "dependentSchemas",
        "dependencies",
        "additionalProperties",
        "items",
        "prefixItems",
        "allOf",
        "anyOf",
        "oneOf",
        "if",
        "then",
        "else",
        "not",
    ],
)
def test_unsupported_features_in_real_subschemas_remain_rejected(keyword):
    nested = {"type": "number", "multipleOf": 3}
    if keyword in ("properties", "$defs", "dependentSchemas", "dependencies"):
        nested = {"value": nested}
    elif keyword in ("prefixItems", "allOf", "anyOf", "oneOf"):
        nested = [nested]
    with pytest.raises(ValueError, match="guidance fallback"):
        _require_exact_replay_constraint({"json": {keyword: nested}})
