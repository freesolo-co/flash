"""RunPod startup preflight.

``check_run_preflight`` aggregates RunPod's missing-config problems plus the shared Hugging Face
dataset-repo requirements, so a single startup error lists everything missing.
"""

from __future__ import annotations

import os

from flash.providers.runpod.preflight import (
    PreflightError,
    missing_credentials,
)

__all__ = [
    "PreflightError",
    "check_run_preflight",
]


def _missing_hf_credentials() -> list[str]:
    """Shared run infra every substrate needs."""
    problems: list[str] = []
    if not os.environ.get("FLASH_ENV_BLOB_CONNECTION_STRING"):
        problems.append(
            "  - FLASH_ENV_BLOB_CONNECTION_STRING: Azure Storage connection string for the "
            "managed environment package store (Azure Blob)"
        )
    if not os.environ.get("FLASH_ENV_PG_URL"):
        problems.append(
            "  - FLASH_ENV_PG_URL: Azure PostgreSQL URL for the environment package index"
        )
    if not os.environ.get("HF_TOKEN"):
        problems.append(
            "  - HF_TOKEN: a token with write access to each run's "
            "`[train] hf_repo`, e.g. `export HF_TOKEN=hf_...`"
        )
    return problems


def _preflight_provider_names() -> set[str]:
    """The providers whose operator config this control plane must satisfy."""
    return {"runpod"}


def check_run_preflight(require_hf: bool = True) -> None:
    """Validate RunPod operator config; raise on missing."""
    selected = _preflight_provider_names()
    problems: list[str] = []
    # The HF write token is shared run infra and is checked once so it isn't double-reported.
    # The HF dataset repo itself is per-run (``[train] hf_repo``).
    if "runpod" in selected:
        problems += missing_credentials(require_hf=False)
    if require_hf:
        problems += _missing_hf_credentials()
    if problems:
        raise PreflightError(
            "the Flash control plane is missing required operator configuration:\n"
            + "\n".join(problems)
            + "\n\nSet these on the control-plane host."
        )
