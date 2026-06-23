"""Pluggable GPU substrates.

The training worker (``flash.engine.worker``) reads a JobSpec from the environment, pulls code
from the HF dataset repo, and streams artifacts/heartbeats/metrics back to it. The provider
owns pricing, provisioning, polling, cancellation, and teardown.

  runpod  serverless Flash endpoints

This module is the registry: ``get_provider(name)`` / ``PROVIDER_NAMES``.
``allocator.allocate`` iterates the active provider list below.
"""

from __future__ import annotations

from functools import cache

from flash.providers.base import Provider

# Active provider order is also the tie-break preference.
PROVIDER_NAMES: tuple[str, ...] = ("runpod",)


def get_provider(name: str) -> Provider:
    """The ``Provider`` singleton for a registered name (raises on unknown)."""
    # Normalize BEFORE the cache so "RunPod"/"runpod"/" runpod " share one cache entry.
    return _get_provider((name or "").strip().lower())


@cache
def _get_provider(key: str) -> Provider:
    if key == "runpod":
        from flash.providers.runpod import PROVIDER

        return PROVIDER
    raise KeyError(f"unknown provider {key!r} (known: {', '.join(PROVIDER_NAMES)})")


def available_providers() -> tuple[str, ...]:
    """Provider NAMES usable from this control plane right now: a provider is available when it
    ``is_configured()`` (creds present). RunPod is the only active substrate right now."""
    return tuple(n for n in PROVIDER_NAMES if get_provider(n).is_configured())


def configured_providers() -> list[Provider]:
    """The ``Provider`` objects available right now (see ``available_providers``)."""
    return [get_provider(n) for n in available_providers()]
