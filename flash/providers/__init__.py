"""Pluggable GPU substrates.

The training worker (``flash.engine.worker``) reads a JobSpec from the environment, pulls code
from the HF dataset repo, and streams artifacts/heartbeats/metrics back to it. The provider
owns pricing, provisioning, polling, cancellation, and teardown.

  runpod      serverless Flash endpoints (always on)
  lambda      Lambda Cloud GPU instances (instance-based complement; iff LAMBDA_API_KEY set)
  vast        Vast.ai verified-datacenter containers (live-market complement; iff VAST_API_KEY set)

This module is the registry: ``get_provider(name)`` / ``PROVIDER_NAMES``.
``allocator.allocate`` iterates the active provider list below; ``available_providers`` narrows
it to the ones configured on THIS control plane (Lambda/Vast are opt-in via their operator
keys, so a box without them silently behaves exactly as the RunPod-only setup).
"""

from __future__ import annotations

from functools import cache

from flash.providers.base import Provider

# Registry / iteration order only — NOT a selection preference. Allocation ranks candidates purely
# by price (see allocator.allocate), so runpod/lambda/vast get no tie-break edge from this order.
PROVIDER_NAMES: tuple[str, ...] = ("runpod", "lambda", "vast")

# Instance-billed providers: they rent a VM/container that BILLS UNTIL TERMINATED, so they need the
# generic instance-cleanup paths (retry teardown, gc-by-label) that endpoint-based RunPod doesn't.
# Single source of truth — keep cleanup/realization sites pointed here so a new instance provider is
# wired into every reaper by adding ONE name (not by hunting hardcoded ("lambda", "vast") tuples).
INSTANCE_PROVIDERS: tuple[str, ...] = ("lambda", "vast")


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
    if key == "vast":
        from flash.providers.vast import PROVIDER

        return PROVIDER
    raise KeyError(f"unknown provider {key!r} (known: {', '.join(PROVIDER_NAMES)})")


def available_providers() -> tuple[str, ...]:
    """Provider names whose credentials are present on this control plane."""
    return tuple(n for n in PROVIDER_NAMES if get_provider(n).is_configured())


def configured_providers() -> list[Provider]:
    """The ``Provider`` objects available right now (see ``available_providers``)."""
    return [get_provider(n) for n in available_providers()]
