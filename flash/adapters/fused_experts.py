"""Fused-expert LoRA targeting, validation, and export normalization."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from flash.adapters.lora_rank import (
    _rank_for_module,
    lora_tensor_rank_disagrees,
    strict_declared_lora_ranks,
)
from flash.core.catalog import get_model

_LORA_FACTOR_PATTERN = re.compile(r"\.lora_[AB]\.")
_LORA_PARAMETER = "weight"
_QWEN36_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
_QWEN36_EXPERT_TARGET_PARAMETERS = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)
_QWEN36_FUSED_TARGET_DIMENSIONS = {
    "mlp.experts.gate_up_proj": (2048, 1024),
    "mlp.experts.down_proj": (512, 2048),
}
_QWEN36_FUSED_TARGET_RUNGS = {
    "mlp.experts.down_proj": "",
    "mlp.experts.gate_up_proj": "base_layer",
}
_FUSED_EXPERT_SYNTHETIC_MODULES = frozenset({"experts", "base_layer"})
_FUSED_EXPERT_WRAPPER_MODULES = ("mlp.experts", "mlp.experts.base_layer")
_NON_LANGUAGE_LORA_SEGMENTS = frozenset(
    {
        "mtp",
        "multi_modal_projector",
        "patch_embed",
        "visual",
        "vision",
        "vision_encoder",
        "vision_model",
        "vision_tower",
    }
)
_INVALID_FUSED_LOCATION = object()

_LoraTensor = tuple[str, str, str, tuple[int, ...]]
_LoraPair = tuple[tuple[int, ...], tuple[int, ...]]
_LoraPairKeys = tuple[str, str]


def lora_target_parameters(model_id: str | None) -> list[str] | None:
    """Return direct parameter targets required by the model's fused expert layout."""
    if model_id == _QWEN36_MODEL_ID:
        return list(_QWEN36_EXPERT_TARGET_PARAMETERS)
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

    declared = strict_declared_lora_ranks(config, source=f"adapter for {model_id}")
    unresolved = [target for target in required if _rank_for_module(target, declared) is None]
    if unresolved:
        raise ValueError(
            f"adapter for {model_id} has no resolved LoRA rank for fused targets {unresolved}"
        )

    modules = config.get("target_modules")
    if isinstance(modules, str):
        if modules != "all-linear":
            raise ValueError(f"adapter for {model_id} string target_modules must be 'all-linear'")
        # all-linear names no modules, so only a default rank can resolve every ordinary target.
        if declared.default is None:
            raise ValueError(
                f"adapter for {model_id} has no resolved default LoRA rank for 'all-linear' "
                "target_modules"
            )
        return
    if (
        not isinstance(modules, list)
        or not modules
        or any(not isinstance(module, str) or not module for module in modules)
    ):
        raise ValueError(
            f"adapter for {model_id} target_modules must be 'all-linear' or a non-empty "
            "list of non-empty strings"
        )
    synthetic = [module for module in modules if _targets_fused_expert_wrapper(module)]
    if synthetic:
        raise ValueError(
            f"adapter for {model_id} contains invalid synthetic target_modules {sorted(synthetic)}"
        )
    unresolved = [module for module in modules if _rank_for_module(module, declared) is None]
    if unresolved:
        raise ValueError(
            f"adapter for {model_id} has no resolved LoRA rank for ordinary target_modules "
            f"{unresolved}"
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


def is_non_language_lora_key(key: str) -> bool:
    """whether a LoRA tensor key names a non-language (vision, projector, mtp) module."""
    return bool(set(key.lower().split(".")) & _NON_LANGUAGE_LORA_SEGMENTS)


def fused_expert_lora_tensor_pairs(
    tensors: Mapping[str, tuple[int, ...]], config: Mapping[str, Any], model_id: str
) -> dict[str, _LoraPairKeys] | None:
    """Return complete canonical pair keys when fused and ordinary topology is exact."""
    expected = _expected_fused_expert_rungs(config, model_id)
    if expected is None:
        return None
    lora_keys = {key: shape for key, shape in tensors.items() if _LORA_FACTOR_PATTERN.search(key)}
    parsed = [
        tensor
        for key, shape in lora_keys.items()
        if (tensor := _parse_lora_tensor(key, shape)) is not None
    ]
    if not parsed or len(parsed) != len(lora_keys):
        return None
    if not _has_complete_fused_rungs(parsed, expected, model_id) or not _has_ordinary_evidence(
        parsed, config, expected
    ):
        return None
    return _complete_lora_pair_keys(parsed)


def has_complete_fused_expert_tensors(
    tensors: Mapping[str, tuple[int, ...]], config: Mapping[str, Any], model_id: str
) -> bool:
    """Return whether fused and ordinary LoRA tensors match the declared PEFT topology."""
    expected = _expected_fused_expert_rungs(config, model_id)
    if expected is None:
        return False
    lora_keys = {key: shape for key, shape in tensors.items() if _LORA_FACTOR_PATTERN.search(key)}
    parsed = [
        tensor
        for key, shape in lora_keys.items()
        if (tensor := _parse_lora_tensor(key, shape)) is not None
    ]
    if len(parsed) != len(lora_keys):
        return False
    return _has_complete_fused_rungs(parsed, expected, model_id) and _has_ordinary_evidence(
        parsed, config, expected
    )


def _parse_lora_tensor(key: str, shape: tuple[int, ...]) -> _LoraTensor | None:
    """Parse one canonical PEFT LoRA tensor key.

    Accept exactly the grammar PEFT writes: a bare ``lora_A.weight`` leaf with no adapter
    namespace. ``get_peft_model_state_dict`` documents stripping the namespace on save, verl's
    merger reproduces that strip, and ``_insert_adapter_name_into_state_dict`` re-inserts it on
    load, so the stripped form is the only spelling that round-trips. Verified against peft 0.19.1:
    ``...lora_A.weight`` loads as ``...lora_A.default.weight`` and matches the live module, while a
    file that already carries the namespace loads as ``...lora_A.default.default.weight`` and
    matches nothing. Since one grammar is accepted, the adapter namespace is no longer part of the
    parsed identity.
    """
    matches = [
        (factor, infix) for factor, infix in (("A", ".lora_A."), ("B", ".lora_B.")) if infix in key
    ]
    if len(matches) != 1:
        return None
    factor, infix = matches[0]
    module_path, _, leaf = key.partition(infix)
    if not module_path or module_path.endswith(".") or leaf != _LORA_PARAMETER:
        return None
    return module_path, factor, key, shape


def _expected_fused_expert_rungs(
    config: Mapping[str, Any], model_id: str
) -> dict[str, dict[str, _LoraPair]] | None:
    """Return the exact target-specific PEFT rung geometry for each fused owner."""
    targets = lora_target_parameters(model_id)
    if model_id != _QWEN36_MODEL_ID or not targets:
        return None
    model = get_model(model_id)
    experts = model.lora_expert_count
    if experts <= 0 or model.num_layers <= 0:
        return None
    fused_count = model.num_layers * experts
    catalog_dimensions = Counter(
        (input_dim, output_dim)
        for input_dim, output_dim, count in model.lora_target_shapes
        if count == fused_count
    )
    if catalog_dimensions != Counter(_QWEN36_FUSED_TARGET_DIMENSIONS.values()):
        return None

    declared = strict_declared_lora_ranks(config)
    expected: dict[str, dict[str, _LoraPair]] = {}
    for target in targets:
        rank = _rank_for_module(target, declared)
        dimensions = _QWEN36_FUSED_TARGET_DIMENSIONS.get(target)
        rung = _QWEN36_FUSED_TARGET_RUNGS.get(target)
        owner, separator, _ = target.rpartition(".")
        if rank is None or dimensions is None or rung is None or not separator or not owner:
            return None
        input_dim, output_dim = dimensions
        stacked_rank = rank * experts
        expected.setdefault(owner, {})[rung] = (
            (stacked_rank, input_dim),
            (output_dim, stacked_rank),
        )
    return expected


def _has_complete_fused_rungs(
    tensors: list[_LoraTensor], expected: Mapping[str, Mapping[str, _LoraPair]], model_id: str
) -> bool:
    """Validate every concrete fused owner, rung, and factor shape."""
    model = get_model(model_id)
    for owner, expected_rungs in expected.items():
        factors: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {}
        for module_path, factor, _key, shape in tensors:
            location = _fused_tensor_location(module_path, owner, frozenset(expected_rungs))
            if location is _INVALID_FUSED_LOCATION:
                return False
            if location is None:
                continue
            instance, rung = location
            factors.setdefault(instance, {}).setdefault(rung, {})[factor] = shape
        if not factors:
            return False
        for seen_rungs in factors.values():
            if set(seen_rungs) != set(expected_rungs):
                return False
            for rung, expected_pair in expected_rungs.items():
                if _lora_pair_shapes(seen_rungs[rung]) != expected_pair:
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


def _fused_tensor_location(
    module_path: str, owner: str, rungs: frozenset[str]
) -> tuple[str, str] | object | None:
    """Locate one tensor on an exact concrete fused owner rung."""
    if module_path.endswith(f".{owner}"):
        instance, rung = module_path[: -len(owner) - 1], ""
    else:
        marker = f".{owner}."
        if marker not in module_path:
            return None
        instance, rung = module_path.rsplit(marker, 1)
    if rung not in rungs:
        return _INVALID_FUSED_LOCATION
    layer_prefix = _layer_prefix(instance)
    if layer_prefix is None or instance != layer_prefix:
        return _INVALID_FUSED_LOCATION
    return f"{instance}.{owner}", rung


def _has_ordinary_evidence(
    tensors: list[_LoraTensor],
    config: Mapping[str, Any],
    fused_rungs: Mapping[str, Mapping[str, _LoraPair]],
) -> bool:
    """Validate concrete ordinary target evidence with per-module ranks."""
    modules = config.get("target_modules")
    all_linear = modules == "all-linear"
    targets = () if all_linear else tuple(modules) if isinstance(modules, list) else ()
    groups: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    evidence: set[str] = set()
    for module_path, factor, key, shape in tensors:
        if _is_fused_rung(module_path, fused_rungs):
            continue
        if not module_path.startswith("base_model.model."):
            return False
        matched = tuple(target for target in targets if _anchored_suffix_match(module_path, target))
        if not all_linear and not matched:
            return False
        evidence.update(matched)
        groups.setdefault(module_path, {})[factor] = (key, shape)
    if not groups or (not all_linear and evidence != set(targets)):
        return False

    declared = strict_declared_lora_ranks(config)
    for module_path, factors in groups.items():
        if _rank_for_module(module_path, declared) is None:
            return False
        if set(factors) != {"A", "B"}:
            return False
        for key, shape in factors.values():
            if not _is_positive_2d(shape) or lora_tensor_rank_disagrees(key, shape, declared):
                return False
    return True


def _complete_lora_pair_keys(
    tensors: list[_LoraTensor],
) -> dict[str, _LoraPairKeys] | None:
    """Return every canonical module pair without orphan factors."""
    groups: dict[str, dict[str, str]] = {}
    for module_path, factor, key, _shape in tensors:
        factors = groups.setdefault(module_path, {})
        if factor in factors:
            return None
        factors[factor] = key
    if not groups or any(set(factors) != {"A", "B"} for factors in groups.values()):
        return None
    return {module_path: (factors["A"], factors["B"]) for module_path, factors in groups.items()}


def _is_fused_rung(module_path: str, fused_rungs: Mapping[str, Mapping[str, _LoraPair]]) -> bool:
    """Return whether a module path is one of the exact fused wrapper rungs."""
    return any(
        _fused_tensor_location(module_path, owner, frozenset(rungs))
        not in (None, _INVALID_FUSED_LOCATION)
        for owner, rungs in fused_rungs.items()
    )


def _anchored_suffix_match(module_path: str, target: str) -> bool:
    """Mirror PEFT list-target matching without accepting partial segment suffixes."""
    return module_path == target or module_path.endswith(f".{target}")


def _is_positive_2d(shape: tuple[int, ...]) -> bool:
    """Return whether a tensor shape is positive, integral, and two-dimensional."""
    return len(shape) == 2 and all(
        isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0
        for dimension in shape
    )


def _lora_pair_shapes(factors: Mapping[str, tuple[int, ...]]) -> _LoraPair | None:
    """Return one complete positive 2-D A/B shape pair."""
    if set(factors) != {"A", "B"}:
        return None
    shape_a = factors["A"]
    shape_b = factors["B"]
    if not _is_positive_2d(shape_a) or not _is_positive_2d(shape_b):
        return None
    return shape_a, shape_b


def _layer_prefix(path: str) -> str | None:
    """Return the concrete ``...layers.N`` prefix in a tensor path, if present."""
    segments = path.split(".")
    for index in range(len(segments) - 2, -1, -1):
        if segments[index] == "layers" and segments[index + 1].isdigit():
            return ".".join(segments[: index + 2])
    return None
