"""Fused-expert LoRA targeting, validation, and export normalization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from flash.core.catalog import get_model

_QWEN35_EXPERT_TARGET_PARAMETERS = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)
_FUSED_EXPERT_SYNTHETIC_MODULES = frozenset({"experts", "base_layer"})
_FUSED_EXPERT_WRAPPER_MODULES = ("mlp.experts", "mlp.experts.base_layer")


def lora_target_parameters(model_id: str | None) -> list[str] | None:
    """Return direct parameter targets required by the model's fused expert layout."""
    if model_id == "Qwen/Qwen3.6-35B-A3B":
        return list(_QWEN35_EXPERT_TARGET_PARAMETERS)
    return None


def validate_fused_expert_adapter_config(config: Mapping[str, Any], model_id: str) -> None:
    """Validate the exact current adapter config required by a fused-expert model."""
    required = lora_target_parameters(model_id)
    if not required:
        return

    targets = config.get("target_parameters")
    if not isinstance(targets, list):
        raise ValueError(
            f"adapter for {model_id} omits required expert targets; "
            "target_parameters must be a list of strings"
        )
    if any(not isinstance(target, str) for target in targets):
        raise ValueError(
            f"adapter for {model_id} must declare target_parameters as a list of strings"
        )
    if len(targets) != len(set(targets)):
        raise ValueError(f"adapter for {model_id} must declare unique target_parameters")
    if set(targets) != set(required):
        raise ValueError(
            f"adapter for {model_id} must declare exactly the fused expert targets {required}"
        )

    modules = config.get("target_modules")
    if modules is None:
        return
    if isinstance(modules, str):
        if modules != "all-linear":
            raise ValueError(f"adapter for {model_id} string target_modules must be 'all-linear'")
        return
    if not isinstance(modules, list) or any(
        not isinstance(module, str) or not module for module in modules
    ):
        raise ValueError(
            f"adapter for {model_id} target_modules must be null, 'all-linear', or a list of "
            "non-empty strings"
        )
    synthetic = [module for module in modules if _targets_fused_expert_wrapper(module)]
    if synthetic:
        raise ValueError(
            f"adapter for {model_id} contains invalid synthetic target_modules {sorted(synthetic)}"
        )


def _targets_fused_expert_wrapper(module: str) -> bool:
    """Return whether PEFT suffix matching binds this target to an expert wrapper."""
    return any(
        module == wrapper or module.endswith(f".{wrapper}") or wrapper.endswith(f".{module}")
        for wrapper in _FUSED_EXPERT_WRAPPER_MODULES
    )


def normalize_verl_fused_expert_export(config: dict[str, Any], model_id: str) -> None:
    """Write the canonical fused targets into a verl-exported adapter config."""
    targets = lora_target_parameters(model_id)
    if not targets:
        return
    config["target_parameters"] = targets
    modules = config.get("target_modules")
    if isinstance(modules, list):
        config["target_modules"] = [
            module
            for module in modules
            if not isinstance(module, str) or module not in _FUSED_EXPERT_SYNTHETIC_MODULES
        ]


def has_complete_fused_expert_tensors(keys: Iterable[str], model_id: str) -> bool:
    """Return whether every fused-expert instance has every PEFT wrapper and LoRA factor.

    PEFT names a wrapped ``nn.Parameter`` after its owning module, not after the parameter itself.
    Wrapping another parameter on that module nests another wrapper under ``base_layer``. Thus N
    targeted parameters yield a fixed ladder of N wrapper paths, while each trained LoRA wrapper
    contributes both ``lora_A`` and ``lora_B`` tensors.

    Coverage is checked per concrete module instance and against the model catalog's authoritative
    transformer-layer count. Deriving expected layers from the artifact itself would accept a
    truncated file that omitted every LoRA tensor for one layer. Pooling wrapper paths across the
    adapter would likewise accept two incomplete layers whose union looks complete. Matching only
    the owner's final segment would also count an unrelated module such as ``router.experts``.
    """
    targets = lora_target_parameters(model_id)
    if not targets:
        return False
    tensor_keys = list(keys)
    expected_layer_count = get_model(model_id).num_layers
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
        expert_layers = {
            prefix
            for instance in factors
            for prefix in [_layer_prefix(instance)]
            if prefix is not None
        }
        layer_roots = {prefix.rsplit(".", 1)[0] for prefix in expert_layers}
        if len(layer_roots) != 1:
            return False
        layer_root = next(iter(layer_roots))
        expected_layers = {f"{layer_root}.{index}" for index in range(expected_layer_count)}
        if expert_layers != expected_layers:
            return False
    return True


def _layer_prefix(path: str) -> str | None:
    """Return the concrete ``...layers.N`` prefix in a tensor path, if present."""
    segments = path.split(".")
    for index in range(len(segments) - 2, -1, -1):
        if segments[index] == "layers" and segments[index + 1].isdigit():
            return ".".join(segments[: index + 2])
    return None
