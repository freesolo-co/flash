"""GPU substrate (RunPod Flash).

The training worker (``autoslm.engine.worker``) is substrate-neutral — it reads a
JobSpec from the environment, pulls code from the HF dataset repo, and streams
artifacts/heartbeats/metrics back to it. RunPod is the sole substrate: GPUs are
priced, provisioned, and torn down via serverless Flash endpoints (``flash/``).

``allocator.allocate`` is the "cheapest GPU that fits" policy.
"""

from __future__ import annotations

PROVIDERS = ("runpod",)


def available_providers() -> tuple[str, ...]:
    """Providers usable from this control plane right now (RunPod only)."""
    return PROVIDERS
