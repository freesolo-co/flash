"""Fused-expert LoRA targeting, validation, and export normalization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from flash.adapters.lora_rank import declared_lora_ranks
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
    rank_pattern = config.get("rank_pattern")
    if rank_pattern is not None and rank_pattern != {}:
        raise ValueError(
            f"adapter for {model_id} uses unsupported rank_pattern for fused target_parameters"
        )

    modules = config.get("target_modules")
    if modules is None:
        # peft uses null for valid direct-parameter-only configs.
        return
    if isinstance(modules, str):
        if modules != "all-linear":
            raise ValueError(f"adapter for {model_id} string target_modules must be 'all-linear'")
        return
    if (
        not isinstance(modules, list)
        or not modules
        or any(not isinstance(module, str) or not module for module in modules)
    ):
        raise ValueError(
            f"adapter for {model_id} target_modules must be null, 'all-linear', or a non-empty "
            "list of non-empty strings"
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
    if modules is None:
        raise ValueError(
            f"exported adapter for {model_id} must retain ordinary LoRA target_modules"
        )
    if isinstance(modules, list):
        config["target_modules"] = [
            module
            for module in modules
            if not isinstance(module, str) or module not in _FUSED_EXPERT_SYNTHETIC_MODULES
        ]


def has_complete_fused_expert_tensors(
    tensors: Mapping[str, tuple[int, ...]], config: Mapping[str, Any], model_id: str
) -> bool:
    """Return whether every fused-expert owner has the exact catalog-backed PEFT factor shapes.

    PEFT serializes each fused target parameter as one 2-D A/B pair under its owning module. Multiple
    targets on that owner form the nested ``base_layer`` ladder, but PEFT does not promise which target
    occupies which rung. Each concrete owner must therefore expose the multiset of target-specific
    pairs derived from the adapter rank, catalog expert count, and catalog target dimensions.
    """
    targets = lora_target_parameters(model_id)
    expected_pairs = _expected_fused_expert_pairs(config, model_id, len(targets or ()))
    if not targets or expected_pairs is None:
        return False
    model = get_model(model_id)
    required_per_owner: dict[str, int] = {}
    for target in targets:
        if "." in target:
            owner = target.rsplit(".", 1)[0]
            required_per_owner[owner] = required_per_owner.get(owner, 0) + 1
    if not required_per_owner:
        return False
    for owner, needed in required_per_owner.items():
        ladder = tuple(".".join(["base_layer"] * depth) for depth in range(needed))
        factors: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {}
        for key, shape in tensors.items():
            if ".lora_" not in key:
                continue
            path, _, tail = key.partition(".lora_")
            leaf = tail.split(".")
            if len(leaf) != 3:
                continue
            factor, adapter_name, parameter = leaf
            if factor not in {"A", "B"} or not adapter_name or parameter != "weight":
                continue
            marker = f".{owner}."
            if path.endswith(f".{owner}"):
                instance, suffix = path[: -len(owner) - 1], ""
            elif marker in path:
                instance, suffix = path.rsplit(marker, 1)
            else:
                continue
            if suffix not in ladder:
                continue
            layer_prefix = _layer_prefix(instance)
            if layer_prefix is None or instance != layer_prefix:
                return False
            full_instance = f"{instance}.{owner}"
            factors.setdefault(full_instance, {}).setdefault(suffix, {})[factor] = shape
        if not factors:
            return False
        for seen in factors.values():
            observed = Counter(_lora_pair_shapes(seen.get(rung, {})) for rung in ladder)
            if None in observed or observed != expected_pairs:
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
        expected_layers = {f"{layer_root}.{index}" for index in range(model.num_layers)}
        if expert_layers != expected_layers:
            return False
    return True


def _expected_fused_expert_pairs(
    config: Mapping[str, Any], model_id: str, target_count: int
) -> Counter[tuple[tuple[int, ...], tuple[int, ...]]] | None:
    """Return the exact unordered factor pairs PEFT must serialize for one fused owner."""
    rank_pattern = config.get("rank_pattern")
    if rank_pattern is not None and rank_pattern != {}:
        return None
    rank = declared_lora_ranks(config).default
    model = get_model(model_id)
    experts = model.lora_expert_count
    if rank is None or experts <= 0 or model.num_layers <= 0:
        return None
    fused_count = model.num_layers * experts
    dimensions = [
        (input_dim, output_dim)
        for input_dim, output_dim, count in model.lora_target_shapes
        if count == fused_count
    ]
    if len(dimensions) != target_count:
        return None
    stacked_rank = rank * experts
    return Counter(
        (
            ((stacked_rank, input_dim), (output_dim, stacked_rank))
            for input_dim, output_dim in dimensions
        )
    )


def _lora_pair_shapes(
    factors: Mapping[str, tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return one complete 2-D A/B shape pair without claiming model compatibility."""
    if not {"A", "B"} <= factors.keys():
        return None
    shape_a = factors["A"]
    shape_b = factors["B"]
    if len(shape_a) != 2 or len(shape_b) != 2:
        return None
    return shape_a, shape_b


def _layer_prefix(path: str) -> str | None:
    """Return the concrete ``...layers.N`` prefix in a tensor path, if present."""
    segments = path.split(".")
    for index in range(len(segments) - 2, -1, -1):
        if segments[index] == "layers" and segments[index + 1].isdigit():
            return ".".join(segments[: index + 2])
    return None
