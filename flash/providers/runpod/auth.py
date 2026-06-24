"""RunPod credential handling for the managed Flash backend (operator-side).

The Flash SDK authenticates via the ``RUNPOD_API_KEY`` environment variable, set by
the **operator** on the control-plane host. End users never
provide provider credentials — they authenticate to the control plane with an Flash
key. Deliberately env-only: ``~/.flash/config.json`` holds the *Flash* key, which
must never be mistaken for a RunPod key.
"""

from __future__ import annotations

from .._auth import load_provider_key
from . import keys as _keys

_ENV_VAR = "RUNPOD_API_KEY"


def load_api_key() -> str | None:
    """API key from the environment (operator configuration)."""
    return load_provider_key(_ENV_VAR)


def ensure_auth() -> str:
    """Select the active account and collapse ``RUNPOD_API_KEY`` to that single key.

    ``RUNPOD_API_KEY`` may be a comma-separated pool of per-account keys (see
    ``runpod.keys``). The runpod_flash SDK reads the raw env var, so a multi-key value
    would be sent as one bearer token (a 401); ``select_active`` collapses it to the
    single active key while the cached pool keeps the rest for failover. Raises if no
    key is configured.
    """
    key = _keys.select_active()
    if not key:
        raise RuntimeError(
            "no RunPod API key found; set RUNPOD_API_KEY on the control-plane host"
        )
    return key
