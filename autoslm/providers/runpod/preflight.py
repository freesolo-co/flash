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
    if require_hf:
        if not os.environ.get("HF_REPO"):
            problems.append(
                "  - HF_REPO: a Hugging Face *dataset* repo for adapters/checkpoints, e.g. "
                "`export HF_REPO=your-org/autoslm-runs`"
            )
        if not os.environ.get("HUGGINGFACE_TOKEN"):
            problems.append(
                "  - HUGGINGFACE_TOKEN: a token with write access to HF_REPO, e.g. "
                "`export HUGGINGFACE_TOKEN=hf_...`"
            )
    return problems


# Historical private name kept for callers/tests that import it.
_missing_credentials = missing_credentials


def check_run_preflight(require_hf: bool = True) -> None:
    """Validate that everything needed to provision managed GPU runs is present.

    Raises ``PreflightError`` listing every missing item (``RUNPOD_API_KEY``,
    ``HF_REPO``, ``HUGGINGFACE_TOKEN``) and how to set it. No-op when nothing is
    missing. See docs/self-hosting.md.
    """
    problems = missing_credentials(require_hf=require_hf)
    if problems:
        raise PreflightError(
            "the AutoSLM control plane is missing required operator configuration:\n"
            + "\n".join(problems)
            + "\n\nSet these on the control-plane host (docs/self-hosting.md)."
        )
