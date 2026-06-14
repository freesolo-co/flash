"""Multi-provider GPU substrates (RunPod Flash + Vast.ai verified datacenters).

The training worker (``autoslm.engine.worker``) is substrate-neutral — it reads a
JobSpec from the environment, pulls code from the HF dataset repo, and streams
artifacts/heartbeats/metrics back to it. Providers differ only in how a GPU is
priced, provisioned, and torn down:

  runpod  flash/ (serverless Flash endpoints; the original substrate)
  vast    providers/vast.py (verified-datacenter instances, REST only)

``allocator.allocate`` is the cross-provider "cheapest GPU that fits" policy.
"""

from __future__ import annotations

import os

PROVIDERS = ("runpod", "vast")


def available_providers() -> tuple[str, ...]:
    """Providers usable from this control plane right now.

    RunPod is the always-on substrate. Vast needs its operator key and a network
    path (AUTOSLM_SKIP_NET — the offline test/CI marker — disables it, keeping
    offline allocation deterministic). AUTOSLM_PROVIDERS pins the set explicitly
    (comma-separated) for operators who want to disable a substrate without
    unsetting its key.
    """
    pinned = os.environ.get("AUTOSLM_PROVIDERS")
    if pinned:
        names = tuple(p.strip().lower() for p in pinned.split(",") if p.strip())
        return tuple(p for p in PROVIDERS if p in names)
    out = ["runpod"]
    if os.environ.get("VAST_API_KEY") and not os.environ.get("AUTOSLM_SKIP_NET"):
        out.append("vast")
    return tuple(out)
