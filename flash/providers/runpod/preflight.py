"""Fail-fast credential checks for the RunPod substrate (operator-side)."""

from __future__ import annotations

import os

from flash.providers.runpod.auth import load_api_key


class PreflightError(RuntimeError):
    """Raised when required operator credentials/configuration are missing."""


def missing_credentials(require_hf: bool = True) -> list[str]:
    """RunPod-related operator config that is missing (empty list == ready)."""
    problems: list[str] = []
    if not load_api_key():
        problems.append("  - RUNPOD_API_KEY: the operator's RunPod API key")
    if require_hf and not os.environ.get("HF_TOKEN"):
        problems.append(
            "  - HF_TOKEN: a token with write access to each run's "
            "`[train] hf_repo`, e.g. `export HF_TOKEN=hf_...`"
        )
    return problems
