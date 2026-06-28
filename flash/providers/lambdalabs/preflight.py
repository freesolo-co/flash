"""Fail-fast credential checks for the Lambda Cloud substrate (operator-side)."""

from __future__ import annotations

from flash.providers.lambdalabs.auth import load_api_key


def missing_credentials(require_hf: bool = True) -> list[str]:
    """Return missing Lambda credentials; empty list means ready. ``require_hf`` ignored — HF checked centrally."""
    problems: list[str] = []
    if not load_api_key():
        problems.append(
            "  - LAMBDA_API_KEY: the operator's Lambda Cloud API key (for the lambda provider)"
        )
    return problems
