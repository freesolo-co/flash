"""Fail-fast credential checks for the Vast.ai substrate (operator-side).

Mirrors ``providers/runpod/preflight.py``: surfaces missing operator config as a clear
problem list the control plane aggregates into one startup error. Vast is opt-in (it is
only required when a run pins ``gpu.provider = "vast"`` or the operator enables it), so
the only Vast-specific requirement is ``VAST_API_KEY``; HF_REPO/HUGGINGFACE_TOKEN are
shared run requirements checked once by the RunPod preflight.
"""

from __future__ import annotations

from autoslm.providers.vast.auth import load_api_key


def missing_credentials(require_hf: bool = True) -> list[str]:
    """Vast-related operator config that is missing (empty list == ready)."""
    problems: list[str] = []
    if not load_api_key():
        problems.append("  - VAST_API_KEY: the operator's Vast.ai API key (for the vast provider)")
    return problems
