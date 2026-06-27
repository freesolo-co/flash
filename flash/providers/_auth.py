"""Shared operator-credential helpers for the GPU providers.

Every provider authenticates the same way: a single API key read ONLY from an
environment variable on the control-plane host (never config files, never shipped to
workers). The per-provider ``auth.py`` modules wrap these with their own env-var name
and error message.
"""

from __future__ import annotations

import os


def load_provider_key(env_var: str) -> str | None:
    """Provider API key from ``env_var`` (operator configuration), or None."""
    return os.environ.get(env_var) or None
