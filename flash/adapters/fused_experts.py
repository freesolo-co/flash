"""Fused-expert LoRA targeting and legacy-export normalization.

PEFT adapts fused routed experts through ``target_parameters`` because the experts are direct
``nn.Parameter`` objects rather than ordinary modules. verl's exporter currently loses that field
and derives synthetic ``target_modules`` names from PEFT's nested wrapper tensor paths. This module
is the single definition of those adapter-shape rules so submit, export, and worker recovery cannot
drift into accepting different artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_QWEN35_EXPERT_TARGET_PARAMETERS = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)


def lora_target_parameters(model_id: str | None) -> list[str] | None:
    """Return direct parameter targets required by the model's fused expert layout."""
    if model_id == "Qwen/Qwen3.6-35B-A3B":
        return list(_QWEN35_EXPERT_TARGET_PARAMETERS)
    return None


def expected_fused_expert_modules(model_id: str | None) -> set[str]:
    """Return the synthetic ``target_modules`` names verl emits for the expert wrappers."""
    targets = lora_target_parameters(model_id)
    if not targets:
        return set()
    return {target.split(".")[-2] for target in targets if "." in target} | {"base_layer"}


def restore_fused_expert_targets(config: dict[str, Any], model_id: str) -> None:
    """Restore the fused-expert targets verl drops and remove its synthetic module names."""
    targets = lora_target_parameters(model_id)
    if not targets:
        return
    config["target_parameters"] = list(targets)
    modules = config.get("target_modules")
    if isinstance(modules, list):
        config["target_modules"] = sorted(
            {str(module) for module in modules} - expected_fused_expert_modules(model_id)
        )


def legacy_fused_expert_config_is_recoverable(config: Mapping[str, Any], model_id: str) -> bool:
    """Return whether config has the complete fingerprint of verl's lossy legacy export.

    This is eligibility for tensor-backed recovery, not proof that the weights are complete. The
    worker must still validate the tensor keys before restoring the fields.
    """
    fingerprint = expected_fused_expert_modules(model_id)
    if not fingerprint or config.get("target_parameters"):
        return False
    modules = config.get("target_modules")
    return isinstance(modules, (list, tuple)) and fingerprint <= {str(module) for module in modules}


def has_complete_fused_expert_tensors(keys: Iterable[str], model_id: str) -> bool:
    """Return whether every fused-expert instance has every PEFT wrapper and LoRA factor.

    PEFT names a wrapped ``nn.Parameter`` after its owning module, not after the parameter itself.
    Wrapping another parameter on that module nests another wrapper under ``base_layer``. Thus N
    targeted parameters yield a fixed ladder of N wrapper paths, while each trained LoRA wrapper
    contributes both ``lora_A`` and ``lora_B`` tensors.

    Coverage is checked per concrete module instance and per transformer layer. Pooling wrapper
    paths across the adapter would accept two incomplete layers whose union looks complete. Matching
    only the owner's final segment would also count an unrelated module such as ``router.experts``.
    """
    targets = lora_target_parameters(model_id)
    if not targets:
        return False
    tensor_keys = list(keys)
    expected_layers = {
        prefix
        for key in tensor_keys
        if ".lora_" in key
        for prefix in [_layer_prefix(key.partition(".lora_")[0])]
        if prefix is not None
    }
    required_per_owner: dict[str, int] = {}
    for target in targets:
        if "." in target:
            owner = target.rsplit(".", 1)[0]
            required_per_owner[owner] = required_per_owner.get(owner, 0) + 1
    if not required_per_owner:
        return False
    for owner, needed in required_per_owner.items():
        ladder = tuple(".".join(["base_layer"] * depth) for depth in range(needed))
        factors: dict[str, dict[str, set[str]]] = {}
        for key in tensor_keys:
            if ".lora_" not in key:
                continue
            path, _, tail = key.partition(".lora_")
            factor = tail.split(".")[0]
            marker = f".{owner}."
            if path.endswith(f".{owner}"):
                instance, suffix = path[: -len(owner) - 1], ""
            elif marker in path:
                instance, suffix = path.rsplit(marker, 1)
            else:
                continue
            if suffix not in ladder:
                continue
            full_instance = f"{instance}.{owner}"
            factors.setdefault(full_instance, {}).setdefault(suffix, set()).add(factor)
        if not factors:
            return False
        for seen in factors.values():
            if any(seen.get(rung, set()) < {"A", "B"} for rung in ladder):
                return False
        if expected_layers:
            expert_layers = {
                prefix
                for instance in factors
                for prefix in [_layer_prefix(instance)]
                if prefix is not None
            }
            if not expected_layers <= expert_layers:
                return False
    return True


def _layer_prefix(path: str) -> str | None:
    """Return the concrete ``...layers.N`` prefix in a tensor path, if present."""
    segments = path.split(".")
    for index in range(len(segments) - 2, -1, -1):
        if segments[index] == "layers" and segments[index + 1].isdigit():
            return ".".join(segments[: index + 2])
    return None
