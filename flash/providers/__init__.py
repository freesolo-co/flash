"""Pluggable GPU substrates (RunPod Flash + Vast.ai verified datacenters).

The training worker (``flash.engine.worker``) is substrate-neutral — it reads a
JobSpec from the environment, pulls code from the HF dataset repo, and streams
artifacts/heartbeats/metrics back to it. Providers differ only in HOW a GPU is priced,
provisioned, and torn down. Every provider implements the SAME ``base.Provider``
protocol — that protocol, not the file set, is what makes them interchangeable — and
each shares a broadly similar module layout (``providers/<name>/{api,auth,pricing,
gpus,jobs,train,preflight}.py``), with provider-specific additions where needed (e.g.
``vast/_bootstrap.py``, which has no RunPod analog):

  runpod  serverless Flash endpoints (the original substrate)
  vast    verified-datacenter instances (REST only)

This module is the registry: ``get_provider(name)`` / ``PROVIDER_NAMES``.
``allocator.allocate`` is the cross-provider "cheapest GPU that fits" policy that
iterates every registered provider.
"""

from __future__ import annotations

from functools import cache

from flash.providers.base import Provider

# Registry order is also the tie-break preference (runpod is the longest-validated
# substrate, so an equal-priced tie prefers it — see allocator.py).
PROVIDER_NAMES: tuple[str, ...] = ("runpod", "vast")


def get_provider(name: str) -> Provider:
    """The ``Provider`` singleton for a registered name (raises on unknown)."""
    # Normalize BEFORE the cache so "RunPod"/"runpod"/" runpod " share one cache entry.
    return _get_provider((name or "").strip().lower())


@cache
def _get_provider(key: str) -> Provider:
    if key == "runpod":
        from flash.providers.runpod import PROVIDER

        return PROVIDER
    if key == "vast":
        from flash.providers.vast import PROVIDER

        return PROVIDER
    raise KeyError(f"unknown provider {key!r} (known: {', '.join(PROVIDER_NAMES)})")


def available_providers() -> tuple[str, ...]:
    """Provider NAMES usable from this control plane right now: a provider is available when it
    ``is_configured()`` (creds present). RunPod is the always-on default; Vast needs
    ``VAST_API_KEY`` (without it, allocation stays on RunPod's static catalog)."""
    return tuple(n for n in PROVIDER_NAMES if get_provider(n).is_configured())


def configured_providers() -> list[Provider]:
    """The ``Provider`` objects available right now (see ``available_providers``)."""
    return [get_provider(n) for n in available_providers()]
