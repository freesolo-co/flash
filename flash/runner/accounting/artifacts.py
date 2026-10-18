"""Artifact repository naming and environment pinning."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path

from flash.core.spec import JobSpec
from flash.envs.meta.identity import GitHubEnvironmentRef

_DEFAULT_ARTIFACT_NAMESPACE = "Freesolo-Co"
_ARTIFACT_REPO_PREFIX = "flashrun-"
_ARTIFACT_REPO_NAME_MAX = 96


def artifact_namespace() -> str:
    """The HuggingFace namespace run artifacts are created under.

    Flash streams code, checkpoints and adapters through HF dataset repos that the control plane
    CREATES, so the namespace has to be one the operator's ``HF_TOKEN`` can write to. Hardcoding
    Freesolo's made self-hosting impossible: ``_assign_managed_hf_repo`` runs on every submit, and a
    self-hoster's token cannot create ``Freesolo-Co/flashrun-*``, so the run failed at upload before
    any training started.
    """
    return (os.environ.get("FLASH_HF_NAMESPACE") or "").strip() or _DEFAULT_ARTIFACT_NAMESPACE


def _environment_artifact_repo_name(env_id: str) -> str:
    """Stable HF dataset repo name for all runs of one environment."""
    raw = (env_id or "default-environment").strip() or "default-environment"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw.lower()).strip("-") or "environment"
    budget = _ARTIFACT_REPO_NAME_MAX - len(_ARTIFACT_REPO_PREFIX) - len(digest) - 1
    slug = slug[:budget].rstrip("-") or "environment"
    return f"{_ARTIFACT_REPO_PREFIX}{slug}-{digest}"


def managed_hf_repo_for_environment(env_id: str) -> str:
    """Private HF dataset repo shared by runs that use the same environment id."""
    return f"{artifact_namespace()}/{_environment_artifact_repo_name(env_id)}"


def _file_digest(path: str, digest) -> None:
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)


def _assign_managed_hf_repo(spec: JobSpec) -> JobSpec:
    """Assign the environment-scoped HF artifact repo (platform-managed, never user-set)."""
    if not spec.run_id or spec.run_id == "local":
        raise ValueError("run_id must be finalized before assigning the artifact repo")
    repo = managed_hf_repo_for_environment(spec.environment.id)
    d = spec.to_internal_dict()
    d["train"] = {**d["train"], "hf_repo": repo}
    return JobSpec.from_dict(d)


def stage_environment_package(
    spec: JobSpec,
    *,
    deadline_at: float | None = None,
) -> JobSpec:
    """Stage or verify one exact environment package before provider allocation."""
    if not spec.environment.id:
        return spec
    if not spec.train.hf_repo:
        raise RuntimeError("hf_repo must be assigned before staging the environment package")
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("staging the environment package requires HF_TOKEN")

    from flash.envs.loading.staged import (
        StagedEnvironmentTransientError,
        archive_path_for_digest,
        encode_manifest,
        is_staged_environment_transient_error,
        manifest_path_for_digest,
        manifest_payload,
        resolve_environment_source,
        verify_staged_environment,
        write_environment_archive,
    )

    if spec.environment.package is not None:
        verify_staged_environment(
            spec.environment,
            hf_repo=spec.train.hf_repo,
            token=token,
        )
        return spec

    from huggingface_hub import HfApi

    from flash.core.spec import EnvironmentPackageSpec
    from flash.providers._lifecycle.net.worker import _ensure_private_artifact_repo, _hf_call

    source = resolve_environment_source(
        spec.environment.id,
        spec.environment.resolved_sha,
        deadline_at=deadline_at,
    )
    try:
        api = HfApi(token=token)
        _ensure_private_artifact_repo(api, spec.train.hf_repo, deadline_at=deadline_at)
        with tempfile.TemporaryDirectory(prefix="flash-stage-env-") as tmp:
            archive_file = Path(tmp) / "package.tar.gz"
            archive_sha256 = write_environment_archive(source, archive_file)
            archive_path = archive_path_for_digest(archive_sha256)
            manifest_bytes = encode_manifest(manifest_payload(source, archive_sha256, archive_path))
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            manifest_path = manifest_path_for_digest(manifest_sha256)
            _hf_call(
                lambda: api.upload_file(
                    path_or_fileobj=str(archive_file),
                    path_in_repo=archive_path,
                    repo_id=spec.train.hf_repo,
                    repo_type="dataset",
                ),
                f"upload environment package {spec.train.hf_repo}:{archive_path}",
                deadline_at=deadline_at,
            )
            commit = _hf_call(
                lambda: api.upload_file(
                    path_or_fileobj=BytesIO(manifest_bytes),
                    path_in_repo=manifest_path,
                    repo_id=spec.train.hf_repo,
                    repo_type="dataset",
                ),
                f"complete environment package {spec.train.hf_repo}:{manifest_path}",
                deadline_at=deadline_at,
            )
        package = EnvironmentPackageSpec(
            artifact_revision=str(getattr(commit, "oid", "") or "").lower(),
            archive_sha256=archive_sha256,
            manifest_sha256=manifest_sha256,
        )
        staged_spec = replace(
            spec,
            environment=replace(
                spec.environment,
                resolved_sha=source.resolved_sha,
                package=package,
            ),
        )
        verify_staged_environment(
            staged_spec.environment,
            hf_repo=staged_spec.train.hf_repo,
            token=token,
        )
        return staged_spec
    except Exception as exc:
        if is_staged_environment_transient_error(exc):
            raise StagedEnvironmentTransientError(
                "environment package staging is temporarily unavailable"
            ) from exc
        raise
    finally:
        if source.staging_root is not None:
            shutil.rmtree(source.staging_root, ignore_errors=True)


def _github_environment_ref(spec: JobSpec) -> GitHubEnvironmentRef | None:
    """parse the spec's resolvable github environment ref without making a request."""
    env_id = spec.environment.id
    if not env_id or spec.environment.resolved_sha:
        return None
    try:
        from flash.envs.loading.loader import (
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
        from flash.envs.loading.loader import _resolve_ref_sha

        sha = _resolve_ref_sha(parsed, timeout=timeout, max_rate_limit_retries=0)
    except Exception as exc:
        return spec, exc
    if not sha:
        return spec, None
    d = spec.to_internal_dict()
    d["environment"] = {**d["environment"], "resolved_sha": sha}
    return JobSpec.from_dict(d), None


def _pin_env_sha_with_reason(spec: JobSpec) -> tuple[JobSpec, str]:
    """best-effort pinning with the exact failure reason used by sft diagnostics."""
    import logging

    parsed = _github_environment_ref(spec)
    if parsed is None:
        return spec, ""
    pinned, error = _resolve_environment_sha_once(spec, parsed, timeout=10.0)
    if error is None:
        return pinned, ""
    logging.getLogger(__name__).warning(
        "resolve-once: could not pin env ref->sha for %r (%s); controller staging will retry",
        spec.environment.id,
        error,
    )
    return spec, str(error).strip()


def _assign_resolved_env_sha(spec: JobSpec) -> JobSpec:
    """Best-effort preflight pin for diagnostics before authoritative controller staging."""
    return _pin_env_sha_with_reason(spec)[0]


def preflight_validate_environment_ref(spec: JobSpec) -> tuple[JobSpec, bool]:
    """reject permanent refs and report whether github-dependent work must defer.

    transient and unclassified failures defer to controller staging before provider allocation.
    tokenless planes skip the request because github also returns 404 for private repositories an
    anonymous caller cannot read. a successful resolve is retained on the returned worker spec.
    """
    from flash.envs.loading.loader import _github_token
    from flash.envs.meta.identity import GitHubPermanentError

    parsed = _github_environment_ref(spec)
    if parsed is None:
        return spec, False
    if not _github_token():
        return spec, True

    pinned, error = _resolve_environment_sha_once(spec, parsed, timeout=4.0)
    if isinstance(error, GitHubPermanentError):
        env_id = spec.environment.id
        from flash.runner.lifecycle.preparation import EnvironmentRefNotFound

        raise EnvironmentRefNotFound(
            f"environment {env_id!r} could not be resolved on GitHub: {error}. Verify the repository "
            "and ref exist and that the plane's GitHub token can read them"
        ) from error
    return pinned, not bool(pinned.environment.resolved_sha)
