"""Shared GRPO rollout-shape and managed host-thread contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

DEFAULT_GRPO_PROMPTS_PER_STEP = 64
DEFAULT_GRPO_GROUP_SIZE = 8
SUPPORTED_GRPO_GROUP_SIZES = (2, 4, 8, 16)
MAX_GRPO_COMPLETIONS_PER_STEP = 512

GRPO_NATIVE_THREAD_ENV: Mapping[str, str] = MappingProxyType(
    {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
)
GRPO_CONTROL_PLANE_OWNED_ENV_KEYS = frozenset(GRPO_NATIVE_THREAD_ENV)


@dataclass(frozen=True)
class GrpoRolloutShape:
    prompts_per_step: int
    group_size: int

    @property
    def completions_per_step(self) -> int:
        return self.prompts_per_step * self.group_size


def resolve_grpo_rollout_shape(
    prompts_per_step: Any = None,
    group_size: Any = None,
) -> GrpoRolloutShape:
    """Resolve omitted GRPO values and reject every unsupported authored shape."""
    prompts = DEFAULT_GRPO_PROMPTS_PER_STEP if prompts_per_step is None else prompts_per_step
    group = DEFAULT_GRPO_GROUP_SIZE if group_size is None else group_size

    if isinstance(prompts, bool) or not isinstance(prompts, int):
        raise TypeError("train.prompts_per_step must be an integer or omitted for GRPO")
    if prompts <= 0:
        raise ValueError("train.prompts_per_step must be positive for GRPO")
    if isinstance(group, bool) or not isinstance(group, int):
        raise TypeError("train.group_size must be an integer or omitted for GRPO")
    if group not in SUPPORTED_GRPO_GROUP_SIZES:
        allowed = ", ".join(str(value) for value in SUPPORTED_GRPO_GROUP_SIZES)
        raise ValueError(f"train.group_size must be one of {{{allowed}}} for GRPO")

    shape = GrpoRolloutShape(prompts, group)
    if shape.completions_per_step > MAX_GRPO_COMPLETIONS_PER_STEP:
        raise ValueError(
            "train.prompts_per_step * train.group_size must be <= "
            f"{MAX_GRPO_COMPLETIONS_PER_STEP} for GRPO "
            f"(got {prompts} * {group} = {shape.completions_per_step})"
        )
    return shape
