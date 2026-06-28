"""Lambda Cloud credential handling (operator-side).

Operator key is NOT shipped to the worker box — teardown is control-plane-side only.
"""

from __future__ import annotations

from .._auth import load_provider_key

_ENV_VAR = "LAMBDA_API_KEY"


def load_api_key() -> str | None:
    """Return API key from environment."""
    return load_provider_key(_ENV_VAR)
