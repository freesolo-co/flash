"""Fail-fast credential checks for the Vast.ai substrate (operator-side).

Mirrors ``providers/runpod/preflight.py``: the only Vast-specific requirement is ``VAST_API_KEY``
(HF_TOKEN is a shared run requirement checked centrally in ``flash/providers/preflight.py``).
"""

from __future__ import annotations

from flash.providers.vast.auth import load_api_key


def missing_credentials(require_hf: bool = True) -> list[str]:
    """Missing Vast operator config; empty list means ready. ``require_hf`` ignored — HF checked centrally."""
    problems: list[str] = []
    if not load_api_key():
        problems.append("  - VAST_API_KEY: the operator's Vast.ai API key (for the vast provider)")
    return problems
