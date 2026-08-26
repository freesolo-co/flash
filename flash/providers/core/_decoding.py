"""Strict decoding for numeric fields returned by external providers."""

from __future__ import annotations

import math
from typing import Final


class MalformedProviderFieldError(ValueError):
    """A provider response carried a present field that cannot be trusted."""

    def __init__(self, provider: str, field: str, expected: str) -> None:
        super().__init__(f"malformed {provider} field {field!r}: expected {expected}")
        self.provider = provider
        self.field = field


class _MissingProviderField:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING_PROVIDER_FIELD"


MISSING_PROVIDER_FIELD: Final = _MissingProviderField()


def decode_finite_number(
    value: object,
    *,
    provider: str,
    field: str,
) -> float | _MissingProviderField | None:
    """Decode a finite JSON number while preserving missing and explicit null."""
    if value is MISSING_PROVIDER_FIELD or value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalformedProviderFieldError(provider, field, "a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise MalformedProviderFieldError(provider, field, "a finite number") from exc
    if not math.isfinite(number):
        raise MalformedProviderFieldError(provider, field, "a finite number")
    return number


def decode_positive_int(
    value: object,
    *,
    provider: str,
    field: str,
) -> int | _MissingProviderField | None:
    """Decode an exact positive JSON integer."""
    if value is MISSING_PROVIDER_FIELD or value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MalformedProviderFieldError(provider, field, "a positive integer")
    return value
