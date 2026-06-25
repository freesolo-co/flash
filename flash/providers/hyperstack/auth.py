"""Hyperstack credential handling (operator-side), mirroring the RunPod/Lambda auth modules.

The Hyperstack REST client authenticates via the ``HYPERSTACK_API_KEY`` environment variable, set
by the **operator** on the control-plane host. Env-only by design. Hyperstack presents the key in a
bare ``api_key`` header (not ``Authorization: Bearer``).
"""

from __future__ import annotations

from .._auth import load_provider_key

_ENV_VAR = "HYPERSTACK_API_KEY"


def load_api_key() -> str | None:
    """API key from the environment (operator configuration)."""
    return load_provider_key(_ENV_VAR)
