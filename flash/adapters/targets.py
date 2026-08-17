"""Resolve the exact LoRA module surface for one training job."""

from __future__ import annotations

import re
from dataclasses import dataclass

from flash.adapters.fused_experts import lora_target_parameters
from flash.core.catalog import validate_model_for_algorithm


@dataclass(frozen=True)
class LoraTargeting:
    target_modules: str
    target_parameters: list[str] | None
    exclude_modules: str | None


def resolve_lora_targeting(model_id: str, *, algorithm: str, multimodal: bool) -> LoraTargeting:
    """Resolve fresh-training targets without changing an authored warm-start adapter."""
    model = validate_model_for_algorithm(model_id, algorithm)
    exclude_modules = None
    if not multimodal:
        prefix = model.lora_language_prefix.strip(".")
        if not prefix:
            raise ValueError(f"{model_id} has no cataloged text-only LoRA language module prefix")
        escaped = re.escape(prefix)
        exclude_modules = rf"^(?!{escaped}(?:\.|$)).*$"
    return LoraTargeting(
        target_modules="all-linear",
        target_parameters=lora_target_parameters(model_id),
        exclude_modules=exclude_modules,
    )
