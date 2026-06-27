"""Vast.ai credential handling (operator-side), mirroring the RunPod auth module.

The Vast REST client authenticates via the ``VAST_API_KEY`` environment variable, set
by the **operator** on the control-plane host. Env-only by
design, exactly like ``RUNPOD_API_KEY``: it is never written to config files or shipped
to workers (the instance self-destroy backstop uses the Vast-injected, instance-scoped
``CONTAINER_API_KEY`` instead).
"""

from __future__ import annotations

from .._auth import load_provider_key

_ENV_VAR = "VAST_API_KEY"


def load_api_key() -> str | None:
    """API key from the environment (operator configuration)."""
    return load_provider_key(_ENV_VAR)
