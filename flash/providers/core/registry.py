"""Modal training substrate registry."""

from __future__ import annotations

from functools import cache

from flash.providers.core.base import Provider

# registry order is not a primary selection preference. it affects only exact full-key allocation
# ties through Python's stable sort.
PROVIDER_NAMES: tuple[str, ...] = ("modal",)


def validated_provider_preferences(value, *, allow_empty: bool = False) -> tuple[str, ...]:
    """normalize an ordered provider preference, preserving the first occurrence."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TypeError("gpu.providers must be a list of provider names")
    if not value and not allow_empty:
        raise ValueError("gpu.providers must name at least one provider")
    result = []
    seen = set()
    for raw in value:
        if not isinstance(raw, str):
            raise TypeError("gpu.providers entries must be strings")
        name = raw.strip().lower()
        if name not in PROVIDER_NAMES:
            raise ValueError(
                f"gpu.providers entry {raw!r} must be one of {', '.join(PROVIDER_NAMES)}"
            )
        if name not in seen:
            result.append(name)
            seen.add(name)
    return tuple(result)


INSTANCE_PROVIDERS: tuple[str, ...] = ("modal",)


def get_provider(name: str) -> Provider:
    """Return the ``Provider`` singleton for a registered name (raises on unknown)."""
    return _get_provider((name or "").strip().lower())


@cache
def _get_provider(key: str) -> Provider:
    if key == "modal":
        from flash.providers.modal.provider import PROVIDER

        return PROVIDER
    raise KeyError(f"unknown provider {key!r}; Flash uses Modal")


def available_providers() -> tuple[str, ...]:
    """provider names enabled by Provider.is_configured(); network and capacity are not checked."""
    return tuple(n for n in PROVIDER_NAMES if get_provider(n).is_configured())


def configured_providers() -> list[Provider]:
    """enabled Provider objects in registered order (see available_providers)."""
    return [get_provider(n) for n in available_providers()]
