"""RunPod credential handling for the managed Flash backend (operator-side).

The Flash SDK authenticates via the ``RUNPOD_API_KEY`` environment variable, set by
the **operator** on the control-plane host. End users never
provide provider credentials — they authenticate to the control plane with an AutoSLM
key. Deliberately env-only: ``~/.autoslm/config.json`` holds the *AutoSLM* key, which
must never be mistaken for a RunPod key.
"""

from __future__ import annotations

import os


def load_api_key() -> str | None:
    """RunPod API key from the environment (operator configuration)."""
    return os.environ.get("RUNPOD_API_KEY") or None


def ensure_auth() -> str:
    """Ensure ``RUNPOD_API_KEY`` is set for the Flash SDK; raise if unavailable."""
    key = load_api_key()
    if not key:
        raise RuntimeError("no RunPod API key found; set RUNPOD_API_KEY on the control-plane host")
    return key
