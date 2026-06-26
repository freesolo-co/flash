"""Control-plane startup preflight.

``check_run_preflight`` aggregates every missing piece of REQUIRED operator configuration — the
RunPod multi-account pool, the Lambda + Hyperstack provider keys, the shared HF/GitHub tokens, and
the Freesolo backend internal key — into one startup error, so a half-configured plane fails fast
at deploy instead of degrading silently in production.
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
    if not os.environ.get("GITHUB_TOKEN"):
        problems.append("  - GITHUB_TOKEN: server token with access to managed Freesolo environments")
    if not os.environ.get("HF_TOKEN"):
        problems.append(
            "  - HF_TOKEN: a token with write access to each run's "
            "`[train] hf_repo`, e.g. `export HF_TOKEN=hf_...`"
        )
    return problems


def _preflight_provider_names() -> set[str]:
    """The providers whose operator config this control plane must satisfy."""
    return {"runpod"}


# A managed control plane provisions across ALL THREE GPU substrates (RunPod multi-account pool +
# Lambda + Hyperstack) and authenticates to the Freesolo backend, so a complete operator config is
# mandatory — a half-configured plane must fail fast at startup instead of degrading silently in
# production. This guards the leak that motivated the gate: a RunPod pool launched with a SINGLE
# account key never reaps (or fails over) across the second account, so that account's idle
# endpoints pile up unseen. Requiring >= 2 RunPod keys makes that exact misconfiguration
# undeployable.
_REQUIRED_RUNPOD_ACCOUNTS = 2


def _missing_operator_infra() -> list[str]:
    """Operator config a managed deployment MUST have, beyond a bare single RunPod key."""
    from flash.providers.runpod import keys as runpod_keys

    problems: list[str] = []
    # >= 2 RunPod account keys (comma-separated pool). A wholly-absent key is already reported as
    # "missing" by missing_credentials(); here we only flag a present-but-too-small pool so the
    # RUNPOD_API_KEY line isn't double-listed.
    n = runpod_keys.key_count()
    if 0 < n < _REQUIRED_RUNPOD_ACCOUNTS:
        problems.append(
            f"  - RUNPOD_API_KEY: needs >= {_REQUIRED_RUNPOD_ACCOUNTS} comma-separated account keys "
            f"(found {n}) — a single-account pool can't reap or fail over across accounts"
        )
    if not os.environ.get("LAMBDA_API_KEY"):
        problems.append("  - LAMBDA_API_KEY: the operator's Lambda Cloud API key")
    if not os.environ.get("HYPERSTACK_API_KEY"):
        problems.append("  - HYPERSTACK_API_KEY: the operator's Hyperstack API key")
    if not os.environ.get("FREESOLO_INTERNAL_KEY"):
        problems.append(
            "  - FREESOLO_INTERNAL_KEY: the control-plane <-> Freesolo backend internal auth key"
        )
    return problems


def check_run_preflight(require_hf: bool = True) -> None:
    """Validate the FULL operator config for a managed control-plane deployment; raise on missing.

    Beyond a bare RunPod key this requires >= 2 RunPod account keys, the Lambda + Hyperstack provider
    keys, the shared HF/GitHub tokens, and the Freesolo internal key — every problem is aggregated so
    one startup error lists everything that's missing.
    """
    selected = _preflight_provider_names()
    problems: list[str] = []
    # The HF write token is shared run infra and is checked once so it isn't double-reported.
    # The HF dataset repo itself is per-run (``[train] hf_repo``).
    if "runpod" in selected:
        problems += missing_credentials(require_hf=False)
    problems += _missing_operator_infra()
    if require_hf:
        problems += _missing_hf_credentials()
    if problems:
        raise PreflightError(
            "the Flash control plane is missing required operator configuration:\n"
            + "\n".join(problems)
            + "\n\nSet these on the control-plane host."
        )
