"""Validate Modal credentials, artifact access, and control-plane authentication at startup."""

from __future__ import annotations

import os

from flash._internal.logging import get_logger


class PreflightError(RuntimeError):
    """Required operator configuration is missing."""


__all__ = [
    "PreflightError",
    "check_run_preflight",
    "require_operator_config",
]

logger = get_logger(__name__)

_PROVIDER_SETUP = (("modal", "MODAL_TOKEN_ID + MODAL_TOKEN_SECRET", "Modal workspace credentials"),)


def _present(var: str) -> bool:
    """True iff env var is set to a non-empty, non-whitespace value."""
    return bool((os.environ.get(var) or "").strip())


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
    ``artifact_namespace``), so this brings the odd ones out into line rather
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

    _normalize_operator_credentials()

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

    from flash.artifacts.config import configured

    if not configured():
        problems.append(
            "  - configure private Azure artifact storage (FLASH_ARTIFACT_BLOB_ACCOUNT_URL or FLASH_ARTIFACT_BLOB_CONNECTION_STRING)"
        )
    import re

    namespace = os.environ.get("FLASH_ARTIFACT_NAMESPACE", "Freesolo-Co").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", namespace):
        problems.append("  - FLASH_ARTIFACT_NAMESPACE must be one path-safe namespace")
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
    require_operator_config()
    if not _present("GITHUB_TOKEN"):
        logger.warning("GITHUB_TOKEN is not set; private environment repositories are unavailable.")
    logger.info("GPU provider configured: modal")
