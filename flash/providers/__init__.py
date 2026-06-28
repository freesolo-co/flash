"""Provider registry: ``get_provider(name)`` / ``PROVIDER_NAMES``."""

from __future__ import annotations

from functools import cache

from flash.providers.base import Provider

PROVIDER_NAMES: tuple[str, ...] = ("runpod", "lambda")


def get_provider(name: str) -> Provider:
    """Return the ``Provider`` singleton for a registered name (raises on unknown)."""
    return _get_provider((name or "").strip().lower())


@cache
def _get_provider(key: str) -> Provider:
    if key == "runpod":
        from flash.providers.runpod import PROVIDER

        return PROVIDER
    if key == "lambda":
        from flash.providers.lambdalabs import PROVIDER

        return PROVIDER
    raise KeyError(f"unknown provider {key!r} (known: {', '.join(PROVIDER_NAMES)})")


def available_providers() -> tuple[str, ...]:
    """Provider names whose credentials are present on this control plane."""
    return tuple(n for n in PROVIDER_NAMES if get_provider(n).is_configured())


def configured_providers() -> list[Provider]:
    """The ``Provider`` objects available right now (see ``available_providers``)."""
    return [get_provider(n) for n in available_providers()]
