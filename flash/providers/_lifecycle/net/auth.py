"""Shared operator-credential helpers for the GPU providers."""

from __future__ import annotations

import os


def load_provider_key(env_var: str) -> str | None:
    """Provider API key from ``env_var`` (operator configuration), or None when unset OR blank.

    Whitespace-stripped, and a whitespace-only value reads as unset. This is the single point
    every instance provider loads its key through, so ``is_configured()``, the startup preflight
    and the outbound API calls all agree about what "configured" means. Without the strip, a key
    that is only whitespace (a stray newline in an env file) makes the provider advertise itself,
    the preflight pass on it as the plane's only substrate, and every allocation then fail on an
    unusable credential. Matches how the RunPod key pool and ``internal_key()`` normalize.
    """
    return (os.environ.get(env_var) or "").strip() or None
