"""Resolve the exact LoRA module surface for one training job."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flash.adapters.fused_experts import lora_target_parameters
from flash.core.catalog import validate_model_for_algorithm


@dataclass(frozen=True)
class LoraTargeting:
    target_modules: str
    target_parameters: list[str] | None
    exclude_modules: str | None


def require_modality_marker(config: Mapping[str, Any], *, source: str) -> None:
    """Reject an adapter config that carries no ``exclude_modules`` modality marker.

    Homed here, beside the resolver that WRITES the marker, so the control plane and the worker
    decide this identically -- and called from both. The control plane already downloads this exact
    config during warm-start preparation, so rejecting there costs a submit-time error, while the
    worker-only check that used to be the sole gate could not fire until a GPU had been allocated
    and the adapter re-downloaded. Same rejection, no paid allocation.

    The worker keeps its own call because it validates the config it actually loads: the two
    processes never share a call stack, so a marker present at preparation is not proof of one in
    the bytes the trainer opens.

    Every supported install writes the key -- ``peft>=0.19`` serializes ``exclude_modules`` on every
    ``LoraConfig``, as null for a multimodal run -- so an absent key means the artifact predates the
    marker entirely and its modality is unknowable. Guessing it is what this rejects: a text-only
    adapter continued as multimodal (or the reverse) trains the wrong module surface and silently
    produces an adapter that serves as a no-op.
    """
    if "exclude_modules" not in config:
        raise ValueError(
            f"{source} is missing the required exclude_modules modality marker; "
            "unmarked artifacts are unsupported"
        )


def config_targets_images(config: Mapping[str, Any]) -> bool:
    """Whether an exported adapter targeted the image stack, per its own modality marker.

    Homed beside `resolve_lora_targeting`, which WRITES the marker, so producer and consumer stay
    one definition. `require_modality_marker` rejects an absent key wherever the answer must be
    trustworthy -- a warm start trains the wrong module surface if it guesses. This reader is for
    callers that must stay non-fatal instead, so it answers False for an absent or non-null marker
    and reserves True for the producer's explicit json null.

    That asymmetry is deliberate and is why this is not `config.get("exclude_modules") is None`:
    that spelling reads an UNMARKED config as multimodal, which is the safe default for a warm
    start (mismatch raises) but the unsafe one for a deployment smoke, where claiming multimodal
    asks a text-only adapter an image question it was never trained for and fails the deploy.
    """
    return "exclude_modules" in config and config["exclude_modules"] is None


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
