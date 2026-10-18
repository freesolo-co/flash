"""Strict decoding helpers for persisted job-spec records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PREPARATION_ENVELOPE_VERSION = 1


def validate_persisted_spec_envelope(snapshot: object) -> int:
    """Validate and return the current persisted preparation version."""
    if not isinstance(snapshot, Mapping):
        raise ValueError("persisted effective preparation is malformed")
    # an ABSENT version is the pre-envelope shape and reads as version 1; only a PRESENT one is
    # type-checked. the writer that stamps this key landed in 1.2.59 (five days before this branch),
    # so runs prepared by an older build are still in flight, and `reallocation_spec_from_status`
    # is what the retry path calls -- rejecting them there marks a live run `unrecoverable` instead
    # of retrying it. a malformed value is still rejected: absence is a known shape, a bad type
    # is not.
    version = snapshot.get("version", PREPARATION_ENVELOPE_VERSION)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("persisted preparation envelope version must be a positive integer")
    if version != PREPARATION_ENVELOPE_VERSION:
        raise ValueError(f"unsupported persisted preparation envelope version {version}")
    return version


def validated_section(
    data: dict[str, Any],
    name: str,
    allowed: set[str],
) -> dict[str, Any]:
    """Read one nested persisted block and reject unknown keys."""
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
    """Return persisted provider preferences cross-checked against gpu classes."""
    from flash.providers.core.registry import PROVIDER_NAMES, validated_provider_preferences

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
        from flash.providers.core.base import providers_for

        if provider and provider not in PROVIDER_NAMES:
            raise ValueError(f"unknown gpu.provider {provider!r}")
        for candidate in (gpu_type, *gpu_type_fallbacks):
            if candidate and provider and provider not in providers_for(candidate):
                raise ValueError(
                    f"gpu.provider {provider!r} cannot provision gpu.type {candidate!r}"
                )
    return provider, providers


def str_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a string-or-list knob to a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(s for s in (str(x) for x in value) if s)


def volume_gb(value: Any, default: int = 100) -> int:
    """Parse a positive volume size in gb, returning the default otherwise."""
    if isinstance(value, bool):
        return default
    try:
        gb = int(value)
    except (TypeError, ValueError):
        return default
    return gb if gb > 0 else default


def opt_int(value: Any) -> int | None:
    """Parse an optional integer while rejecting booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return int(value)


def opt_float(value: Any) -> float | None:
    """Parse an optional float while rejecting booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return float(value)
