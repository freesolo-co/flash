"""Structured-output edge coverage verifies tuple choices and malformed schema strings.

These direct normalizer tests pin the remaining pure-Python branches without loading an engine.
"""

from __future__ import annotations

import pytest

from flash.serving.src.io.structured_outputs import (
    StructuredOutputsError,
    normalize_structured_outputs,
)


def test_choice_tuple_normalizes_to_list() -> None:
    assert normalize_structured_outputs({"choice": ("a", "b")}) == {"choice": ["a", "b"]}


def test_json_constraint_rejects_invalid_json_string() -> None:
    with pytest.raises(StructuredOutputsError, match="string is not valid JSON"):
        normalize_structured_outputs({"json": "not valid JSON {"})
