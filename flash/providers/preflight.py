"""Control-plane startup preflight.

``check_run_preflight`` aggregates every missing piece of REQUIRED operator configuration — the
RunPod multi-account pool, the Lambda + Hyperstack provider keys, the shared HF/GitHub tokens, and
the Freesolo backend internal key — into one startup error, so a half-configured plane fails fast
at deploy instead of degrading silently in production.
"""

from __future__ import annotations

import os

from flash.providers.runpod.preflight import PreflightError

__all__ = [
    "PreflightError",
    "check_run_preflight",
]

# A managed control plane provisions across ALL THREE GPU substrates (RunPod multi-account pool +
# Lambda + Hyperstack) and authenticates to the Freesolo backend, so a complete operator config is
# mandatory. Requiring >= 2 RunPod account keys guards the leak that motivated this gate: a pool
# launched with a SINGLE key never reaps (or fails over) across the second account, so that
# account's idle endpoints pile up unseen.
_REQUIRED_RUNPOD_ACCOUNTS = 2


def check_run_preflight(require_hf: bool = True) -> None:
    """Validate the FULL operator config for a managed control-plane deployment; raise on missing.

    One self-contained check: a RunPod pool of >= 2 account keys, the Lambda + Hyperstack provider
    keys, the Freesolo backend internal key, and (when ``require_hf``) the shared GitHub + HF tokens.
    Every missing piece is aggregated into a single, actionable startup error.
    """
    from flash.providers.runpod import keys as runpod_keys

    problems: list[str] = []

    # RunPod: a comma-separated pool of >= 2 account keys. key_count() is 0 when RUNPOD_API_KEY is
    # unset/empty, so the two branches cover "missing" and "too few" without double-listing the line.
    n = runpod_keys.key_count()
    if n == 0:
        problems.append(
            f"  - RUNPOD_API_KEY: the operator's RunPod API key "
            f"(>= {_REQUIRED_RUNPOD_ACCOUNTS} comma-separated account keys)"
        )
    elif n < _REQUIRED_RUNPOD_ACCOUNTS:
        problems.append(
            f"  - RUNPOD_API_KEY: needs >= {_REQUIRED_RUNPOD_ACCOUNTS} comma-separated account keys "
            f"(found {n}) — a single-account pool can't reap or fail over across accounts"
        )

    # The other two GPU substrates and the control-plane <-> backend auth key.
    if not os.environ.get("LAMBDA_API_KEY"):
        problems.append("  - LAMBDA_API_KEY: the operator's Lambda Cloud API key")
    if not os.environ.get("HYPERSTACK_API_KEY"):
        problems.append("  - HYPERSTACK_API_KEY: the operator's Hyperstack API key")
    if not os.environ.get("FREESOLO_INTERNAL_KEY"):
        problems.append(
            "  - FREESOLO_INTERNAL_KEY: the control-plane <-> Freesolo backend internal auth key"
        )

    # Shared run infra (the HF dataset repo itself is per-run, ``[train] hf_repo``, not checked here).
    if require_hf:
        if not os.environ.get("GITHUB_TOKEN"):
            problems.append(
                "  - GITHUB_TOKEN: server token with access to managed Freesolo environments"
            )
        if not os.environ.get("HF_TOKEN"):
            problems.append(
                "  - HF_TOKEN: a token with write access to each run's "
                "`[train] hf_repo`, e.g. `export HF_TOKEN=hf_...`"
            )

    if problems:
        raise PreflightError(
            "the Flash control plane is missing required operator configuration:\n"
            + "\n".join(problems)
            + "\n\nSet these on the control-plane host."
        )
