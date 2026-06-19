"""RunPod credential handling for the managed Flash backend (operator-side).

The Flash SDK authenticates via the ``RUNPOD_API_KEY`` environment variable, set by
the **operator** on the control-plane host. End users never
provide provider credentials — they authenticate to the control plane with an Flash
key. Deliberately env-only: ``~/.flash/config.json`` holds the *Flash* key, which
must never be mistaken for a RunPod key.
"""

from __future__ import annotations

from .._auth import ensure_provider_auth, load_provider_key

_ENV_VAR = "RUNPOD_API_KEY"


def load_api_key() -> str | None:
    """API key from the environment (operator configuration)."""
    return load_provider_key(_ENV_VAR)


def ensure_auth() -> str:
    """Ensure ``RUNPOD_API_KEY`` is set; raise if unavailable."""
    return ensure_provider_auth(_ENV_VAR, "no RunPod API key found; set RUNPOD_API_KEY on the control-plane host")
