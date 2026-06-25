"""Fail-fast credential checks for the Lambda Cloud substrate (operator-side).

Mirrors ``providers/runpod/preflight.py``. Lambda is OPT-IN (the allocator only reaches for it
when ``LAMBDA_API_KEY`` is set), so the only Lambda-specific requirement is ``LAMBDA_API_KEY``;
HF_TOKEN is a shared run requirement checked once centrally by the cross-provider preflight
(``flash/providers/preflight.py``), which calls each provider-specific check with
``require_hf=False`` so HF is never double-reported.
"""

from __future__ import annotations

from flash.providers.lambdalabs.auth import load_api_key


def missing_credentials(require_hf: bool = True) -> list[str]:
    """Lambda-related operator config that is missing (empty list == ready).

    ``require_hf`` is accepted only for signature parity with the RunPod check and is
    intentionally ignored: Lambda has no provider-owned HF requirement (the shared HF_TOKEN is
    checked once centrally in ``providers.preflight``).
    """
    problems: list[str] = []
    if not load_api_key():
        problems.append(
            "  - LAMBDA_API_KEY: the operator's Lambda Cloud API key (for the lambda provider)"
        )
    return problems
