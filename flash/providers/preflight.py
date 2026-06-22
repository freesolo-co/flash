"""Cross-provider startup preflight.

``check_run_preflight`` aggregates EVERY selected provider's missing-config problems
(RunPod is the default substrate; Vast only when configured/pinned) plus the shared
Hugging Face dataset-repo requirements, so a single startup error lists everything
missing. The per-provider key checks live in ``flash.providers.<name>.preflight``.
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
    """Shared run infra every substrate needs: the HF write token, plus PRIME_API_KEY (the
    worker installs the run's published environment regardless of the GPU provider). The HF
    dataset repo is per-run (``[train] hf_repo``), not an operator var."""
    problems: list[str] = []
    if not os.environ.get("PRIME_API_KEY"):
        problems.append(
            "  - PRIME_API_KEY: environment install credentials for the GPU worker, e.g. "
            "`export PRIME_API_KEY=pit_...`"
        )
    if not os.environ.get("HF_TOKEN"):
        problems.append(
            "  - HF_TOKEN: a token with write access to each run's "
            "`[train] hf_repo`, e.g. `export HF_TOKEN=hf_...`"
        )
    return problems


def _preflight_provider_names() -> set[str]:
    """The providers whose operator config this control plane must satisfy. RunPod is always
    required (the default substrate); Vast is opt-in (preflighted only when VAST_API_KEY signals
    intent)."""
    names = {"runpod"}  # always-on default substrate
    if os.environ.get("VAST_API_KEY"):
        names.add("vast")  # opt-in: a partial vast config signals intent
    return names


def check_run_preflight(require_hf: bool = True) -> None:
    """Validate operator config across the configured providers; raise on missing.

    Only the providers this control plane actually uses are checked: RunPod's requirements
    (RUNPOD_API_KEY + the shared PRIME_API_KEY/HF_TOKEN) are always checked, and a configured
    Vast key (VAST_API_KEY) adds its own check. The HF dataset repo is per-run
    (``[train] hf_repo``), not an operator var.
    """
    selected = _preflight_provider_names()
    problems: list[str] = []
    # The HF write token is SHARED run infra (every substrate streams artifacts through HF),
    # so it is checked once regardless of which providers are selected — a Vast-only plane
    # still needs it. Each provider check is asked for its keys only (require_hf=False) so HF
    # isn't double-reported. The HF dataset repo itself is per-run (``[train] hf_repo``).
    if "runpod" in selected:
        problems += missing_credentials(require_hf=False)
    if "vast" in selected:
        from flash.providers.vast.preflight import missing_credentials as vast_missing

        problems += vast_missing(require_hf=False)
    if require_hf:
        problems += _missing_hf_credentials()
    if problems:
        raise PreflightError(
            "the Flash control plane is missing required operator configuration:\n"
            + "\n".join(problems)
            + "\n\nSet these on the control-plane host."
        )
