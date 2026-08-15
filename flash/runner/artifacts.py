"""Artifact repository naming and environment pinning."""

from __future__ import annotations

import hashlib
import os
import re

import flash.runner as runner
from flash.core.spec import JobSpec
from flash.envs.identity import GitHubEnvironmentRef


def artifact_namespace() -> str:
    """The HuggingFace namespace run artifacts are created under.

    Flash streams code, checkpoints and adapters through HF dataset repos that the control plane
    CREATES, so the namespace has to be one the operator's ``HF_TOKEN`` can write to. Hardcoding
    Freesolo's made self-hosting impossible: ``_assign_managed_hf_repo`` runs on every submit, and a
    self-hoster's token cannot create ``Freesolo-Co/flashrun-*``, so the run failed at upload before
    any training started.
    """
    return (
        os.environ.get("FLASH_HF_NAMESPACE") or ""
    ).strip() or runner._DEFAULT_ARTIFACT_NAMESPACE


def _environment_artifact_repo_name(env_id: str) -> str:
    """Stable HF dataset repo name for all runs of one environment."""
    raw = (env_id or "default-environment").strip() or "default-environment"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw.lower()).strip("-") or "environment"
    budget = runner._ARTIFACT_REPO_NAME_MAX - len(runner._ARTIFACT_REPO_PREFIX) - len(digest) - 1
    slug = slug[:budget].rstrip("-") or "environment"
    return f"{runner._ARTIFACT_REPO_PREFIX}{slug}-{digest}"


def managed_hf_repo_for_environment(env_id: str) -> str:
    """Private HF dataset repo shared by runs that use the same environment id."""
    return f"{runner.artifact_namespace()}/{runner._environment_artifact_repo_name(env_id)}"


def _file_digest(path: str, digest) -> None:
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)


def flash_code_prefix() -> str:
    """Content-addressed HF path for the current ``flash`` package snapshot."""
    import flash

    pkg_dir = os.path.realpath(os.path.dirname(os.path.abspath(flash.__file__)))
    digest = hashlib.sha1()
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, pkg_dir).replace(os.sep, "/")
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            runner._file_digest(path, digest)
            digest.update(b"\0")
    return f"code/{digest.hexdigest()[:32]}/flash"


def _assign_managed_hf_repo(spec: JobSpec) -> JobSpec:
    """Assign the environment-scoped HF artifact repo (platform-managed, never user-set)."""
    if not spec.run_id or spec.run_id == "local":
        raise ValueError("run_id must be finalized before assigning the artifact repo")
    repo = runner.managed_hf_repo_for_environment(spec.environment.id)
    d = spec.to_internal_dict()
    d["train"] = {**d["train"], "hf_repo": repo}
    return runner.JobSpec.from_dict(d)


def _github_environment_ref(spec: JobSpec) -> GitHubEnvironmentRef | None:
    """parse the spec's resolvable github environment ref without making a request."""
    env_id = spec.environment.id
    if not env_id or spec.environment.resolved_sha:
        return None
    try:
        from flash.envs.loader import (
            _parse_github_environment_ref,
            is_managed_environment_slug,
            managed_slug_to_github_ref,
        )

        ref_str = (
            managed_slug_to_github_ref(env_id) if is_managed_environment_slug(env_id) else env_id
        )
        return _parse_github_environment_ref(ref_str)
    except Exception:
        return None


def _resolve_environment_sha_once(
    spec: JobSpec, parsed: GitHubEnvironmentRef, *, timeout: float
) -> tuple[JobSpec, Exception | None]:
    """resolve and pin one environment ref, preserving the typed failure for its caller."""
    try:
        from flash.envs.loader import _resolve_ref_sha

        sha = _resolve_ref_sha(parsed, timeout=timeout, max_rate_limit_retries=0)
    except Exception as exc:
        return spec, exc
    if not sha:
        return spec, None
    d = spec.to_internal_dict()
    d["environment"] = {**d["environment"], "resolved_sha": sha}
    return runner.JobSpec.from_dict(d), None


def _pin_env_sha_with_reason(spec: JobSpec) -> tuple[JobSpec, str]:
    """best-effort pinning with the exact failure reason used by sft diagnostics."""
    import logging

    parsed = _github_environment_ref(spec)
    if parsed is None:
        return spec, ""
    pinned, error = _resolve_environment_sha_once(spec, parsed, timeout=10.0)
    if error is None:
        return pinned, ""
    logging.getLogger(runner.__name__).warning(
        "resolve-once: could not pin env ref->sha for %r (%s); worker will resolve",
        spec.environment.id,
        error,
    )
    return spec, str(error).strip()


def _assign_resolved_env_sha(spec: JobSpec) -> JobSpec:
    """Pin env ref->SHA once so N workers don't fan-out N GitHub API calls (secondary rate-limit). Best-effort."""
    return _pin_env_sha_with_reason(spec)[0]


def preflight_validate_environment_ref(spec: JobSpec) -> tuple[JobSpec, bool]:
    """reject permanent refs and report whether github-dependent work must defer.

    transient and unclassified failures keep the existing worker deferral. tokenless planes skip
    the request because github also returns 404 for private repositories an anonymous caller cannot
    read. a successful resolve is retained on the returned worker spec.
    """
    from flash.envs.identity import GitHubPermanentError
    from flash.envs.loader import _github_token

    parsed = _github_environment_ref(spec)
    if parsed is None:
        return spec, False
    if not _github_token():
        return spec, True

    pinned, error = _resolve_environment_sha_once(spec, parsed, timeout=4.0)
    if isinstance(error, GitHubPermanentError):
        env_id = spec.environment.id
        raise runner.EnvironmentRefNotFound(
            f"environment {env_id!r} could not be resolved on GitHub: {error}. Verify the repository "
            "and ref exist and that the plane's GitHub token can read them"
        ) from error
    return pinned, not bool(pinned.environment.resolved_sha)
