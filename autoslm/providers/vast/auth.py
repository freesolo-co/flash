"""Vast.ai credential handling (operator-side), mirroring the RunPod auth module.

The Vast REST client authenticates via the ``VAST_API_KEY`` environment variable, set
by the **operator** on the control-plane host. Env-only by
design, exactly like ``RUNPOD_API_KEY``: it is never written to config files or shipped
to workers (the instance self-destroy backstop uses the Vast-injected, instance-scoped
``CONTAINER_API_KEY`` instead).
"""

from __future__ import annotations

import os


def load_api_key() -> str | None:
    """Vast API key from the environment (operator configuration)."""
    return os.environ.get("VAST_API_KEY") or None


def ensure_auth() -> str:
    """Ensure ``VAST_API_KEY`` is set; raise if unavailable."""
    key = load_api_key()
    if not key:
        raise RuntimeError("no Vast API key found; set VAST_API_KEY on the control-plane host")
    return key
