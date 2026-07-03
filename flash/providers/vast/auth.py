"""Vast.ai credential handling (operator-side), mirroring the RunPod auth module.

``VAST_API_KEY`` is env-only and set by the operator on the control-plane host; it is never
shipped to workers (the instance self-destroy backstop uses the injected ``CONTAINER_API_KEY``).
"""

from __future__ import annotations

from .._auth import load_provider_key

_ENV_VAR = "VAST_API_KEY"


def load_api_key() -> str | None:
    """API key from the environment (operator configuration)."""
    return load_provider_key(_ENV_VAR)
