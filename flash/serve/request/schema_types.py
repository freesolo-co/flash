"""Exact JSON Schema type matching for tool declarations and arguments."""

import math
from decimal import Decimal
from typing import Any


def _matches_type(value: Any, schema_type: str | list[str]) -> bool:
    if type(schema_type) is list:
        return any(_matches_type(value, member) for member in schema_type)
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return type(value) is bool
    if schema_type == "integer":
        return (
            type(value) is int
            or (type(value) is Decimal and _decimal_is_integral(value))
            or (type(value) is float and math.isfinite(value) and value.is_integer())
        )
    if schema_type == "number":
        if type(value) is Decimal:
            return value.is_finite()
        return type(value) is int or (type(value) is float and math.isfinite(value))
    if schema_type == "string":
        return type(value) is str
    if schema_type == "array":
        return type(value) is list
    return type(value) is dict


def _decimal_is_integral(value: Decimal) -> bool:
    if not value.is_finite():
        return False
    digits, exponent = value.as_tuple().digits, value.as_tuple().exponent
    return exponent >= 0 or all(digit == 0 for digit in digits[exponent:])
