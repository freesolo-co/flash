"""Fail-fast credential checks for the control plane (operator-side).

These run when the AutoSLM server starts (and before any RunPod Flash provisioning) so
missing operator configuration produces one clear, actionable error instead of a
partial run that dies mid-provisioning. End users never see these — their preflight is
client-side ("do I have an AutoSLM key?", see autoslm/client).
"""

from __future__ import annotations

import os

from autoslm.flash.auth import load_api_key


class PreflightError(RuntimeError):
    """Raised when required operator credentials/configuration are missing."""


def _missing_credentials(require_hf: bool = True) -> list[str]:
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


def check_run_preflight(require_hf: bool = True) -> None:
    """Validate that everything needed to provision managed GPU runs is present.

    Raises ``PreflightError`` listing every missing item (``RUNPOD_API_KEY``,
    ``HF_REPO``, ``HUGGINGFACE_TOKEN``) and how to set it. No-op when nothing is
    missing. ``VAST_API_KEY`` is OPTIONAL — absent just disables the Vast substrate
    (one log line); present, a cheap auth probe catches a bad key at startup
    instead of at first allocation. See docs/self-hosting.md.
    """
    problems = _missing_credentials(require_hf=require_hf)
    if problems:
        raise PreflightError(
            "the AutoSLM control plane is missing required operator configuration:\n"
            + "\n".join(problems)
            + "\n\nSet these on the control-plane host (docs/self-hosting.md)."
        )
    _check_vast()


def _check_vast() -> None:
    from autoslm._logging import get_logger

    logger = get_logger(__name__)
    if not os.environ.get("VAST_API_KEY"):
        logger.info("VAST_API_KEY not set; the vast substrate is disabled (runpod only)")
        return
    if os.environ.get("AUTOSLM_SKIP_NET"):
        return
    try:
        from autoslm.providers import vast_api

        vast_api.list_instances()
    except Exception as exc:
        raise PreflightError(
            f"VAST_API_KEY is set but the Vast API rejected it: {exc}\n"
            "Fix or unset it (unset = runpod-only) on the control-plane host."
        ) from exc
