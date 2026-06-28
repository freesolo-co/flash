"""RunPod credential handling (operator-side; end users never provide provider keys)."""

from __future__ import annotations

from .._auth import load_provider_key
from . import keys as _keys

_ENV_VAR = "RUNPOD_API_KEY"


def load_api_key() -> str | None:
    """API key from the environment (operator configuration)."""
    return load_provider_key(_ENV_VAR)


def ensure_auth() -> str:
    """Collapse RUNPOD_API_KEY to the active key — the SDK sends the raw env var verbatim, so a comma-separated pool would 401."""
    key = _keys.select_active()
    if not key:
        raise RuntimeError(
            "no RunPod API key found; set RUNPOD_API_KEY on the control-plane host"
        )
    return key
