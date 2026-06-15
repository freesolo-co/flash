"""Cross-provider startup preflight.

``check_run_preflight`` aggregates EVERY selected provider's missing-config problems
(RunPod is the default substrate; Vast only when configured/pinned) plus the shared
Hugging Face dataset-repo requirements, so a single startup error lists everything
missing. The per-provider key checks live in ``autoslm.providers.<name>.preflight``.
"""

from __future__ import annotations

import os

from autoslm.providers.runpod.preflight import (
    PreflightError,
    missing_credentials,
)

__all__ = [
    "PreflightError",
    "check_run_preflight",
]


def _missing_hf_credentials() -> list[str]:
    """Shared run infra every substrate needs: the HF write token, plus PRIME_API_KEY (the
    worker ``prime env install``s the run's Hub env regardless of the GPU provider). The HF
    dataset repo is per-run (``[train] hf_repo``), not an operator var."""
    problems: list[str] = []
    if not os.environ.get("PRIME_API_KEY"):
        problems.append(
            "  - PRIME_API_KEY: a Prime Intellect API key; the GPU worker uses it to "
            "`prime env install` the run's Hub environment (public + private), e.g. "
            "`export PRIME_API_KEY=pit_...`"
        )
    if not os.environ.get("HUGGINGFACE_TOKEN"):
        problems.append(
            "  - HUGGINGFACE_TOKEN: a token with write access to each run's "
            "`[train] hf_repo`, e.g. `export HUGGINGFACE_TOKEN=hf_...`"
        )
    return problems


def _preflight_provider_names() -> set[str]:
    """The providers whose operator config this control plane must satisfy.

    Honors the ``AUTOSLM_PROVIDERS`` pin: a Vast-only control plane
    (``AUTOSLM_PROVIDERS=vast``) must NOT demand RUNPOD_API_KEY, and conversely.
    Without a pin, RunPod is always required (the default substrate) and Vast is opt-in
    (preflighted only when VAST_API_KEY signals intent)."""
    from autoslm.providers import PROVIDER_NAMES, pinned_provider_names

    pinned = pinned_provider_names()
    if pinned is not None:
        return {n for n in PROVIDER_NAMES if n in pinned}
    names = {"runpod"}  # always-on default substrate
    if os.environ.get("VAST_API_KEY"):
        names.add("vast")  # opt-in: a partial vast config signals intent
    return names


def check_run_preflight(require_hf: bool = True) -> None:
    """Validate operator config across the configured providers; raise on missing.

    Only the providers this control plane actually uses are checked: the
    ``AUTOSLM_PROVIDERS`` pin selects the substrate set, so a Vast-only deployment never
    fails on a missing RUNPOD_API_KEY (and vice versa). Unpinned, RunPod's requirements
    (RUNPOD_API_KEY + the shared PRIME_API_KEY/HUGGINGFACE_TOKEN) are always checked
    and a configured Vast key adds its own check. The HF dataset repo is per-run
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
        from autoslm.providers.vast.preflight import missing_credentials as vast_missing

        problems += vast_missing(require_hf=False)
    if require_hf:
        problems += _missing_hf_credentials()
    if problems:
        raise PreflightError(
            "the AutoSLM control plane is missing required operator configuration:\n"
            + "\n".join(problems)
            + "\n\nSet these on the control-plane host."
        )
