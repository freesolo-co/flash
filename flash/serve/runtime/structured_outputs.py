"""pure-python normalization for vllm structured-output constraints."""

from __future__ import annotations

from typing import Any

from flash.serve.request import validation as _shared

from .errors import StructuredOutputsError


def normalize_structured_outputs(value: Any) -> dict[str, Any] | None:
    """return canonical structured-output kwargs, an explicit-off dict, or none."""
    return _shared.normalize_structured_outputs(
        value,
        error_type=StructuredOutputsError,
        validate_decoded_dicts=False,
    )
