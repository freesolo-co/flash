"""Fail-fast credential checks for the RunPod substrate (operator-side).

These run when the AutoSLM server starts (and before any RunPod Flash provisioning) so
missing operator configuration produces one clear, actionable error instead of a
partial run that dies mid-provisioning. End users never see these — their preflight is
client-side ("do I have an AutoSLM key?", see autoslm/client).
"""

from __future__ import annotations

import os

from autoslm.providers.runpod.auth import load_api_key


class PreflightError(RuntimeError):
    """Raised when required operator credentials/configuration are missing."""


def missing_credentials(require_hf: bool = True) -> list[str]:
    """RunPod-related operator config that is missing (empty list == ready)."""
    problems: list[str] = []
    if not load_api_key():
        problems.append("  - RUNPOD_API_KEY: the operator's RunPod API key")
    if require_hf and not os.environ.get("HUGGINGFACE_TOKEN"):
        problems.append(
            "  - HUGGINGFACE_TOKEN: a token with write access to each run's "
            "`[train] hf_repo`, e.g. `export HUGGINGFACE_TOKEN=hf_...`"
        )
    return problems
