"""Typed outcomes for one destructive provider operation."""

from __future__ import annotations

from enum import StrEnum


class DestructiveOperationOutcome(StrEnum):
    """Why one destructive operation stopped."""

    DELETED = "deleted"
    NOT_CONFIRMED = "not_confirmed"
    HALTED = "halted"
