"""Control-plane startup preflight - fail fast on missing operator config.

The rule is "enough to run a job", not "everything Freesolo runs in production". A
self-hosted plane picks its own GPU substrate, so the provider check is a floor (at
least ONE of RunPod/Lambda/Vast configured), not a roll call. Configuration that only
degrades an optional capability warns instead of refusing to boot - a plane that cannot
fail over between RunPod accounts still trains runs, and refusing to start teaches the
operator nothing it could not learn from a log line.
"""

from __future__ import annotations

import os

from flash._logging import get_logger
from flash.providers.runpod.preflight import PreflightError

__all__ = [
    "PreflightError",
    "check_run_preflight",
]

logger = get_logger(__name__)

# Cross-account reap/failover needs >= 2 distinct RunPod accounts. WARN-only: a single-account
# pool still provisions, still reaps its own idle endpoints (the sweep lists per key), and only
# loses the ability to ride out one account's quota/credit exhaustion by moving to another.
# That is an availability property, not a correctness one, so it must not block a self-hosted
# plane whose operator has exactly one RunPod account.
_RECOMMENDED_RUNPOD_ACCOUNTS = 2

# How an operator turns each provider on. Reported together when NONE is configured, so the
# error names every acceptable way out rather than the one Freesolo happens to use.
_PROVIDER_SETUP = (
    ("runpod", "RUNPOD_API_KEY", "one or more comma-separated RunPod account keys"),
    ("lambda", "LAMBDA_API_KEY", "a Lambda Cloud API key"),
    ("vast", "VAST_API_KEY", "a Vast.ai API key"),
)


def _present(var: str) -> bool:
    """True iff env var is set to a non-empty, non-whitespace value."""
    return bool((os.environ.get(var) or "").strip())


def check_run_preflight() -> None:
    """Raise PreflightError if the plane cannot run a job; warn on degraded-but-workable config."""
    from flash.providers import available_providers
    from flash.providers.runpod import keys as runpod_keys

    problems: list[str] = []

    configured = available_providers()
    if not configured:
        problems.append(
            "  - no GPU provider is configured. Set at least ONE of:\n"
            + "\n".join(
                f"      {var} - {desc} (enables the {name!r} provider)"
                for name, var, desc in _PROVIDER_SETUP
            )
        )

    if not _present("HF_TOKEN"):
        problems.append(
            "  - HF_TOKEN: a token with write access to each run's "
            "`[train] hf_repo`, e.g. `export HF_TOKEN=hf_...`. Flash streams code, "
            "checkpoints and adapters through HuggingFace dataset repos, so every run "
            "needs it regardless of GPU provider"
        )
    if not _present("FREESOLO_INTERNAL_KEY"):
        problems.append(
            "  - FREESOLO_INTERNAL_KEY: the control plane's own auth key. On a self-hosted "
            "plane this is the credential your clients present (`flash login --api-url "
            "<your-plane>`); without it the plane can only authenticate keys issued by a "
            "Freesolo-compatible backend at FREESOLO_BASE_URL. Any high-entropy string works: "
            "`export FREESOLO_INTERNAL_KEY=$(openssl rand -hex 32)`"
        )

    if problems:
        raise PreflightError(
            "the Flash control plane is missing required operator configuration:\n"
            + "\n".join(problems)
            + "\n\nSet these on the control-plane host. See SELF_HOSTING.md."
        )

    _warn_degraded(configured, runpod_keys)


def _warn_degraded(configured: tuple[str, ...], runpod_keys) -> None:
    """Log capabilities this config gives up. Never raises: the plane can run every job type."""
    if "runpod" in configured:
        distinct = len(set(runpod_keys.keys()))
        if distinct < _RECOMMENDED_RUNPOD_ACCOUNTS:
            logger.warning(
                "RUNPOD_API_KEY holds %d distinct account key(s); >= %d comma-separated keys are "
                "recommended so a run can fail over when one account hits its quota or credit "
                "limit. Runs still allocate and idle endpoints are still reaped within the "
                "account you did configure.",
                distinct,
                _RECOMMENDED_RUNPOD_ACCOUNTS,
            )

    if not _present("GITHUB_TOKEN"):
        logger.warning(
            "GITHUB_TOKEN is not set. Environments hosted on PUBLIC GitHub repos still load "
            "(subject to GitHub's unauthenticated rate limit), but private repos and "
            "`flash env push` are unavailable."
        )

    logger.info("GPU provider(s) configured: %s", ", ".join(configured))
