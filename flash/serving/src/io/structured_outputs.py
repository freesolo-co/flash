"""Structured-output normalization shared by the CPU router and GPU engine.

Every entry point normalizes to ``StructuredOutputsParams`` keyword arguments before engine work.
The result is idempotent because the router validates before RPC and the engine validates again.
``{}`` therefore remains the explicit-off marker instead of becoming an empty JSON schema.
"""

from __future__ import annotations

from typing import Any

from flash.serve.request import validation as _shared


class StructuredOutputsError(ValueError):
    """Invalid structured-output spec surfaced by pydantic as a client-visible 422."""


def normalize_structured_outputs(value: Any) -> dict[str, Any] | None:
    """Return canonical structured-output kwargs, an explicit-off dict, or None."""
    return _shared.normalize_structured_outputs(
        value,
        error_type=StructuredOutputsError,
        validate_decoded_dicts=True,
    )
