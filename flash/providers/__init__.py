"""Pluggable GPU substrates.

The training worker (``flash.engine.worker``) reads a JobSpec from the environment, pulls code
from the HF dataset repo, and streams artifacts/heartbeats/metrics back to it. The provider
owns pricing, provisioning, polling, cancellation, and teardown.

  runpod      serverless Flash endpoints (always on)
  lambda      Lambda Cloud GPU instances (instance-based complement; iff LAMBDA_API_KEY set)
  hyperstack  Hyperstack GPU VMs (instance-based complement; iff HYPERSTACK_API_KEY set)

This module is the registry: ``get_provider(name)`` / ``PROVIDER_NAMES``.
``allocator.allocate`` iterates the active provider list below; ``available_providers`` narrows
it to the ones configured on THIS control plane (Lambda/Hyperstack are opt-in via their operator
keys, so a box without them silently behaves exactly as the RunPod-only setup).
"""

from __future__ import annotations

from functools import cache

from flash.providers.base import Provider

# Active provider order is also the tie-break preference (RunPod wins price ties, then Lambda).
PROVIDER_NAMES: tuple[str, ...] = ("runpod", "lambda", "hyperstack")


def get_provider(name: str) -> Provider:
    """The ``Provider`` singleton for a registered name (raises on unknown)."""
    # Normalize BEFORE the cache so "RunPod"/"runpod"/" runpod " share one cache entry.
    return _get_provider((name or "").strip().lower())


@cache
def _get_provider(key: str) -> Provider:
    if key == "runpod":
        from flash.providers.runpod import PROVIDER

        return PROVIDER
    if key == "lambda":
        from flash.providers.lambdalabs import PROVIDER

        return PROVIDER
    if key == "hyperstack":
        from flash.providers.hyperstack import PROVIDER

        return PROVIDER
    raise KeyError(f"unknown provider {key!r} (known: {', '.join(PROVIDER_NAMES)})")


def available_providers() -> tuple[str, ...]:
    """Provider NAMES usable from this control plane right now: a provider is available when it
    ``is_configured()`` (creds present). RunPod is always on; Lambda/Hyperstack join only when
    their operator keys are present."""
    return tuple(n for n in PROVIDER_NAMES if get_provider(n).is_configured())


def configured_providers() -> list[Provider]:
    """The ``Provider`` objects available right now (see ``available_providers``)."""
    return [get_provider(n) for n in available_providers()]
