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

from flash._internal.logging import get_logger
from flash.providers.runpod.client.preflight import PreflightError

__all__ = [
    "PreflightError",
    "check_run_preflight",
    "require_operator_config",
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


def _artifact_namespace_problem(value: str) -> str:
    """Why ``value`` cannot serve as an artifact namespace, or "" if it can.

    A namespace is ONE hugging face id segment. ``managed_hf_repo_for_environment`` appends the
    repo name to it, so an operator who writes the natural-looking ``owner/repo`` produces a
    three-segment id that hugging face rejects -- and rejects at the first submit, long after
    preflight called the plane healthy. Validating the assembled id rather than the segment alone
    keeps this honest about what actually gets created.
    """
    if "/" in value:
        return (
            f"{value!r} contains '/', but this is a single namespace (a HuggingFace user or org), "
            "not a repo id. Flash appends the repo name itself, so this would build "
            f"'{value}/flashrun-...', which HuggingFace rejects as an id"
        )
    from huggingface_hub.utils import validate_repo_id

    from flash.runner.accounting.artifacts import _environment_artifact_repo_name

    # validate the id that WILL be created, not the segment in isolation: the repo half is
    # generated and already known-good, so anything rejected here is the operator's namespace.
    candidate = f"{value}/{_environment_artifact_repo_name('preflight-probe')}"
    try:
        validate_repo_id(candidate)
    except Exception as exc:
        return f"{value!r} is not a usable HuggingFace namespace: {exc}"
    return ""


# Credentials the plane reads straight from ``os.environ`` at many call sites rather than through
# one accessor. Normalized in place at startup so the value consumers read is the value preflight
# accepted -- see ``_normalize_operator_credentials``.
_NORMALIZED_IN_PLACE = ("HF_TOKEN", "GITHUB_TOKEN")


def _normalize_operator_credentials() -> None:
    """Strip surrounding whitespace off operator credentials, in the process environment.

    ``_present`` strips before deciding a credential is set, so a token with a trailing newline
    (routine in a copied ``.env``) passes startup. The consumers then read ``os.environ`` RAW:
    ``HfApi(token=...)`` does not strip either, so the padding reaches the wire as
    ``Authorization: Bearer <sp><sp>hf_...`` and HF rejects it. The plane reported healthy and the
    first submit died creating the artifact repo, after the operator thought setup was done.

    Fixed here rather than at ~40 call sites because this is where the two halves disagree: the
    check that ACCEPTS the padded value is the right place to make it true. Every other operator
    credential already normalizes at its single accessor (``load_provider_key``, ``internal_key``,
    ``artifact_namespace``, the RunPod key pool), so this brings the odd ones out into line rather
    than adding a new convention.

    Stripping and accepting, not rejecting: a stray newline is a typo, and every other credential
    in this plane tolerates one. Runs before the checks below, and before anything is served, so
    no consumer can observe the unnormalized value.
    """
    for var in _NORMALIZED_IN_PLACE:
        raw = os.environ.get(var)
        if raw is None:
            continue
        stripped = raw.strip()
        if stripped != raw:
            # a whitespace-ONLY value becomes "", which every consumer's `or None` / truthiness
            # check already reads as absent -- the same meaning `_present` gives it.
            os.environ[var] = stripped
            logger.warning("%s had surrounding whitespace; using the stripped value.", var)


def require_operator_config() -> None:
    """Raise PreflightError if the plane is missing configuration it needs to run a job.

    The refusing half of the preflight, on its own because it runs twice: ``run_server`` calls it
    before uvicorn so a misconfigured operator gets a clean message instead of an ASGI startup
    traceback, and the lifespan calls the full ``check_run_preflight`` for the plane that another
    ASGI server built through ``create_app()``. Pure reads, apart from re-stripping already-stripped
    env vars, so running it twice changes nothing.
    """
    from flash.providers.core.registry import available_providers
    from flash.server.platform.auth import standalone

    _normalize_operator_credentials()

    problems: list[str] = []

    configured = available_providers()
    if "vast" in configured or "FLASH_VAST_RESULT_ORIGINS" in os.environ:
        from flash.providers.vast.client.result import configured_result_origins

        try:
            configured_result_origins()
        except ValueError as exc:
            problems.append(f"  - {exc}")
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
    if standalone() and not _present("FLASH_HF_NAMESPACE"):
        # Only in standalone. The managed plane's HF_TOKEN owns the default namespace; a
        # self-hoster's does not, so leaving it unset defers a certain failure to the first
        # submit, where it surfaces as an HF permission error on repo creation long after
        # preflight reported the plane healthy.
        problems.append(
            "  - FLASH_HF_NAMESPACE: the HuggingFace user or org that run artifacts are "
            "created under. Required with FLASH_STANDALONE=1: it defaults to Freesolo's "
            "namespace, which your HF_TOKEN cannot write to, so every run would fail at "
            "artifact upload before training starts. Set it to a namespace your HF_TOKEN "
            "owns, e.g. `export FLASH_HF_NAMESPACE=your-hf-username`"
        )
    elif _present("FLASH_HF_NAMESPACE"):
        # set, but not necessarily usable. presence alone was the whole check, so `owner/repo` --
        # the natural spelling for anyone used to HF ids -- passed preflight and then failed every
        # submit while creating the artifact repo, with the plane reported healthy.
        namespace_problem = _artifact_namespace_problem(
            (os.environ.get("FLASH_HF_NAMESPACE") or "").strip()
        )
        if namespace_problem:
            problems.append(f"  - FLASH_HF_NAMESPACE: {namespace_problem}")
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


def check_run_preflight() -> None:
    """Raise PreflightError if the plane cannot run a job; warn on degraded-but-workable config."""
    from flash.providers.core.registry import available_providers
    from flash.providers.runpod.client import auth as runpod_keys

    require_operator_config()
    _warn_degraded(available_providers(), runpod_keys)


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
