"""Private Verl strategy policy shared by control-plane and worker paths."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

FsdpGeneration = Literal[1, 2]


def _resolve_fsdp_generation(
    algorithm: str,
    target_parameters: Sequence[str] | None,
) -> FsdpGeneration:
    """Return the only validated FSDP generation for one resolved workload."""
    if algorithm == "sft" or algorithm == "grpo":
        return 2
    if algorithm == "opd":
        return 2 if target_parameters else 1
    raise ValueError(f"unsupported Verl algorithm for FSDP policy: {algorithm!r}")
