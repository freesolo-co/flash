"""Shared operator-credential helpers for the GPU providers."""

from __future__ import annotations

import os


def load_provider_key(env_var: str) -> str | None:
    """Provider API key from ``env_var`` (operator configuration), or None."""
    return os.environ.get(env_var) or None
