"""validation for provider http statuses safe to expose across teacher boundaries."""

from __future__ import annotations


def validated_provider_status(value: object) -> int | None:
    if type(value) is int and 100 <= value <= 599:
        return value
    return None
