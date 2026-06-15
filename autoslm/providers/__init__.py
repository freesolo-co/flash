"""Pluggable GPU substrates (RunPod Flash + Vast.ai verified datacenters).

The training worker (``autoslm.engine.worker``) is substrate-neutral — it reads a
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

import os
from functools import cache

from autoslm.providers.base import Provider

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
        from autoslm.providers.runpod import PROVIDER

        return PROVIDER
    if key == "vast":
        from autoslm.providers.vast import PROVIDER

        return PROVIDER
    raise KeyError(f"unknown provider {key!r} (known: {', '.join(PROVIDER_NAMES)})")


def pinned_provider_names() -> set[str] | None:
    """The ``AUTOSLM_PROVIDERS`` pin parsed to a lowercased name set, or None when unset.

    Single source of truth for the comma-separated pin parse shared by
    ``available_providers`` and ``providers.preflight._preflight_provider_names``.

    A pin is authoritative downstream: ``available_providers`` intersects with it and
    ``_preflight_provider_names`` derives its credential-check set from it. A value that
    parses to NO known provider names (blank/whitespace, comma-only, or all-typo) is
    treated as UNSET — we return None, the documented "no pin / default" contract.
    Returning the full set instead would wrongly force EVERY provider's credentials in
    preflight (e.g. demand a Vast key on a RunPod-only run), and returning an empty set
    would disable every provider; None makes both consumers fall back to their defaults.
    """
    pinned = os.environ.get("AUTOSLM_PROVIDERS")
    if not pinned:
        return None
    names = {p.strip().lower() for p in pinned.split(",") if p.strip()}
    names &= set(PROVIDER_NAMES)
    if not names:
        return None
    return names


def available_providers() -> tuple[str, ...]:
    """Provider NAMES usable from this control plane right now.

    A provider is available when it ``is_configured()`` (creds present + net path).
    ``AUTOSLM_PROVIDERS`` pins the set explicitly (comma-separated) for operators who
    want to disable a substrate without unsetting its key. RunPod is the always-on
    default; Vast needs ``VAST_API_KEY`` (and AUTOSLM_SKIP_NET disables both live
    paths, keeping offline allocation deterministic).
    """
    names = pinned_provider_names()
    if names is not None:
        return tuple(n for n in PROVIDER_NAMES if n in names and get_provider(n).is_configured())
    return tuple(n for n in PROVIDER_NAMES if get_provider(n).is_configured())


def configured_providers() -> list[Provider]:
    """The ``Provider`` objects available right now (see ``available_providers``)."""
    return [get_provider(n) for n in available_providers()]
