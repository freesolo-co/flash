"""Decoding rules for PERSISTED job-spec records, as distinct from authored configuration.

`JobSpec` serves four different contracts, and this module owns exactly one of them: reading back
bytes that were already written to `~/.flash/runs/<run_id>.json`. That is not the same job as
validating a config a user just wrote, and the two must not be confused:

- authored configuration is parsed by `flash.schema.spec_from_dict`, which is STRICT. an unknown key
  is a typo in a file the author can still fix, so it fails loudly.
- a persisted record is parsed by `JobSpec.from_dict`, which applies the looser scalar coercions
  here. It reads a record this build wrote, in this build's shape; an unknown key is fatal in both.

Serialized-byte warning: `flash/runner/preparation.py::_preparation_digest` sha256-hashes the
canonical JSON of `to_dict()` and `to_internal_dict()` output. Key ORDER is safe there
(`sort_keys=True`), but key PRESENCE is not -- adding, dropping, or renaming a key in a decode path
changes the digest and fails integrity validation for a run whose snapshot was already written.

These functions take plain dicts and return plain dicts and scalars. That is the rule that keeps the
module importable by `spec.py` rather than the other way round, and it is why `validated_section`
takes its allowed-key set as an argument instead of reading `fields(TrainSpec)` itself.

What is deliberately NOT here: `_coerce_wandb` and `_parse_persisted_gpu_types` are also called only
from `JobSpec.from_dict`, so they belong to this contract by usage -- but they construct `WandbSpec`
and reach `_validated_gpu_type`, which `__post_init__` needs too. Moving them would make this module
import the dataclasses that import it. The boundary is therefore drawn at "persisted-decoding logic
that does not depend on the spec types themselves".
"""

from __future__ import annotations

from typing import Any


def validated_section(data: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    """Read one nested block (``[train]``, ``[gpu]``) of a persisted record, rejecting unknown keys.

    A missing block and an explicit null are the same state -- both mean "authored nothing" -- so both
    normalize to empty rather than one of them raising. An unknown key is fatal.
    """
    section = data.get(name)
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise TypeError(f"{name} must be an object")
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"{name} has unknown key(s): {', '.join(unknown)}")
    return section


def validated_persisted_providers(
    gpu: dict[str, Any], gpu_type: str, gpu_type_fallbacks: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """``(provider, providers)`` from a persisted ``[gpu]`` block, cross-checked against its classes.

    ``GpuSpec.__post_init__`` runs the same both-set check but cannot replace this one: ``allow_empty``
    here keys off whether the record CARRIED a ``providers`` key, and that presence is exactly what the
    dataclass has already lost by the time it validates. An explicitly empty preference and an absent
    one are different states in a stored record.
    """
    from flash.providers import PROVIDER_NAMES, validated_provider_preferences

    provider = gpu.get("provider", "")
    if not isinstance(provider, str):
        raise TypeError("gpu.provider must be a string")
    provider = provider.strip().lower()
    providers = validated_provider_preferences(
        gpu.get("providers", ()), allow_empty="providers" not in gpu
    )
    if provider and providers:
        raise ValueError("gpu.provider and gpu.providers cannot both be set")
    if provider or providers or gpu_type:
        from flash.providers.base import providers_for

        if provider and provider not in PROVIDER_NAMES:
            raise ValueError(f"unknown gpu.provider {provider!r}")
        # a hard provider pin must carry every acceptable class. provider preferences stay soft: an
        # ineligible named provider contributes no candidate, while eligible named or unnamed
        # configured providers remain available for failover.
        for candidate in (gpu_type, *gpu_type_fallbacks):
            if candidate and provider and provider not in providers_for(candidate):
                raise ValueError(
                    f"gpu.provider {provider!r} cannot provision gpu.type {candidate!r}"
                )
    return provider, providers


def str_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a string-or-list knob to a tuple of strings; a bare string is one element, not iterated."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(s for s in (str(x) for x in value) if s)


def volume_gb(value: Any, default: int = 100) -> int:
    """Parse volume size in GB; non-positive / non-numeric / missing values return default."""
    if isinstance(value, bool):
        # bool is an int subclass; reject to avoid int(True)==1 becoming a 1 GB volume.
        return default
    try:
        gb = int(value)
    except (TypeError, ValueError):
        return default
    return gb if gb > 0 else default


def opt_int(value: Any) -> int | None:
    """Parse optional int; rejects bools (bool is int subclass — int(True)==1 is a footgun)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return int(value)


def opt_float(value: Any) -> float | None:
    """Parse optional float; rejects bools (mirrors ``opt_int``)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return float(value)
