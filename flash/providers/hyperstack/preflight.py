"""Fail-fast credential checks for the Hyperstack substrate (operator-side).

Mirrors ``providers/lambdalabs/preflight.py``. Hyperstack is OPT-IN (the allocator only reaches for
it when ``HYPERSTACK_API_KEY`` is set), so the only Hyperstack-specific requirement is the key;
HF_TOKEN is the shared run requirement checked once centrally.
"""

from __future__ import annotations

from flash.providers.hyperstack.auth import load_api_key


def missing_credentials(require_hf: bool = True) -> list[str]:
    """Hyperstack-related operator config that is missing (empty list == ready).

    ``require_hf`` is accepted only for signature parity with the RunPod check and is ignored.
    """
    problems: list[str] = []
    if not load_api_key():
        problems.append(
            "  - HYPERSTACK_API_KEY: the operator's Hyperstack API key (for the hyperstack provider)"
        )
    return problems
