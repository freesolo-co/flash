"""Pluggable GPU substrates.

The training worker (``flash.engine.worker``) reads a JobSpec from the environment, pulls code
from the HF dataset repo, and streams artifacts/heartbeats/metrics back to it. The provider
owns pricing, provisioning, polling, cancellation, and teardown.

  runpod      serverless Flash endpoints (always on)
  lambda      Lambda Cloud GPU instances (instance-based complement; iff LAMBDA_API_KEY set)
  vast        Vast.ai verified-datacenter containers (live-market complement; iff VAST_API_KEY set)

this module is the registry: ``get_provider(name)`` / ``PROVIDER_NAMES``.
``allocator.allocate`` iterates providers enabled by ``available_providers()`` in registered
``PROVIDER_NAMES`` order. ``Provider.is_configured()`` controls control-plane enablement; preflight
separately checks required credentials and configuration. neither check implies network reachability
or live capacity. Lambda/Vast opt in with operator keys; RunPod reports missing credentials at preflight.
"""

from __future__ import annotations

from functools import cache

from flash.providers.base import Provider

# registry order is not a primary selection preference. it affects only exact full-key allocation
# ties through Python's stable sort.
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
    """provider names enabled by Provider.is_configured(); network and capacity are not checked."""
    return tuple(n for n in PROVIDER_NAMES if get_provider(n).is_configured())


def configured_providers() -> list[Provider]:
    """enabled Provider objects in registered order (see available_providers)."""
    return [get_provider(n) for n in available_providers()]
