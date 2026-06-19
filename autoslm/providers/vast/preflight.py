"""Fail-fast credential checks for the Vast.ai substrate (operator-side).

Mirrors ``providers/runpod/preflight.py``: surfaces missing operator config as a clear
problem list the control plane aggregates into one startup error. Vast is opt-in (it is
only required when a run pins ``gpu.provider = "vast"`` or the operator enables it), so
the only Vast-specific requirement is ``VAST_API_KEY``; HF_TOKEN is a shared run
requirement checked once by the RunPod preflight (the HF dataset repo is per-run via
``[train] hf_repo``).
"""

from __future__ import annotations

from autoslm.providers.vast.auth import load_api_key


def missing_credentials(require_hf: bool = True) -> list[str]:
    """Vast-related operator config that is missing (empty list == ready).

    ``require_hf`` is accepted only for signature parity with the RunPod check and is
    intentionally ignored: Vast has no provider-owned HF requirement (the shared
    HF_TOKEN is checked once centrally in ``providers.preflight``).
    """
    problems: list[str] = []
    if not load_api_key():
        problems.append("  - VAST_API_KEY: the operator's Vast.ai API key (for the vast provider)")
    return problems
