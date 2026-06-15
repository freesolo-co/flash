"""Pluggable GPU substrates (RunPod Flash + Vast.ai verified datacenters).

The training worker (``autoslm.engine.worker``) is substrate-neutral — it reads a
JobSpec from the environment, pulls code from the HF dataset repo, and streams
artifacts/heartbeats/metrics back to it. Providers differ only in HOW a GPU is priced,
provisioned, and torn down, and every provider implements the SAME ``base.Provider``
interface behind the SAME module layout (``providers/<name>/{api,auth,pricing,gpus,
jobs,train,preflight}.py``), so they are interchangeable:

  runpod  serverless Flash endpoints (the original substrate)
  vast    verified-datacenter instances (REST only)

This module is the registry: ``get_provider(name)`` / ``PROVIDER_NAMES``.
``allocator.allocate`` is the cross-provider "cheapest GPU that fits" policy that
iterates every registered provider.
"""

from __future__ import annotations

import os
from functools import cache

from autoslm.providers.base import Provider

# Registry order is also the tie-break preference (runpod is the longest-validated
# substrate, so an equal-priced tie prefers it — see allocator.py).
PROVIDER_NAMES: tuple[str, ...] = ("runpod", "vast")

# Back-compat alias for the historical name.
PROVIDERS = PROVIDER_NAMES


def get_provider(name: str) -> Provider:
    """The ``Provider`` singleton for a registered name (raises on unknown)."""
    # Normalize BEFORE the cache so "RunPod"/"runpod"/" runpod " share one cache entry.
    return _get_provider((name or "").strip().lower())


@cache
def _get_provider(key: str) -> Provider:
    if key == "runpod":
        from autoslm.providers.runpod import PROVIDER

        return PROVIDER
    if key == "vast":
        from autoslm.providers.vast import PROVIDER

        return PROVIDER
    raise KeyError(f"unknown provider {key!r} (known: {', '.join(PROVIDER_NAMES)})")


def available_providers() -> tuple[str, ...]:
    """Provider NAMES usable from this control plane right now.

    A provider is available when it ``is_configured()`` (creds present + net path).
    ``AUTOSLM_PROVIDERS`` pins the set explicitly (comma-separated) for operators who
    want to disable a substrate without unsetting its key. RunPod is the always-on
    default; Vast needs ``VAST_API_KEY`` (and AUTOSLM_SKIP_NET disables both live
    paths, keeping offline allocation deterministic).
    """
    pinned = os.environ.get("AUTOSLM_PROVIDERS")
    if pinned:
        names = {p.strip().lower() for p in pinned.split(",") if p.strip()}
        return tuple(n for n in PROVIDER_NAMES if n in names and get_provider(n).is_configured())
    return tuple(n for n in PROVIDER_NAMES if get_provider(n).is_configured())


def configured_providers() -> list[Provider]:
    """The ``Provider`` objects available right now (see ``available_providers``)."""
    return [get_provider(n) for n in available_providers()]
