"""Controller-staged immutable environment package transport."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from flash.core.spec import EnvironmentPackageSpec, EnvironmentSpec
from flash.envs.package.limits import LimitedArchiveReader, archive_stream_limit
from flash.envs.package.unpack import extract_validated_archive_members

_MANIFEST_VERSION = 1
_MANIFEST_MAX_BYTES = 64 * 1024


class StagedEnvironmentTransientError(RuntimeError):
    """A temporary failure while acquiring or verifying a staged environment."""


@dataclass(frozen=True)
class ResolvedEnvironmentSource:
    canonical_id: str
    resolved_sha: str
    root: Path
    entrypoint: str
    staging_root: Path | None = None


@dataclass(frozen=True)
class VerifiedStagedEnvironment:
    manifest_file: Path
    archive_file: Path
    entrypoint: str


@dataclass
class StagedEnvironmentMaterialization:
    root: Path
    entrypoint: Path
    _cleaned: bool = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        shutil.rmtree(self.root, ignore_errors=True)
        self._cleaned = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.cleanup()


def _reference_for_environment_id(env_id: str) -> tuple[str, Any]:
    from flash.envs.meta.identity import (
        _parse_github_environment_ref,
        canonical_environment_id,
        is_managed_environment_slug,
        managed_slug_to_github_ref,
    )

    canonical_id = canonical_environment_id(env_id)
    reference = (
        managed_slug_to_github_ref(canonical_id)
        if is_managed_environment_slug(canonical_id)
        else canonical_id
    )
    parsed = _parse_github_environment_ref(reference)
    if parsed is None:
        raise ValueError(f"environment {env_id!r} is not a GitHub environment reference")
    return canonical_id, parsed


def is_staged_environment_transient_error(exc: BaseException) -> bool:
    """Classify timeout, connection, rate-limit, and server failures across clients."""
    from flash.envs.meta.identity import GitHubTransientError
    from flash.providers.artifacts.hf import hf_status_code

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (StagedEnvironmentTransientError, GitHubTransientError)):
            return True
        status = hf_status_code(current)
        if status == 429 or (status is not None and status >= 500):
            return True
        name = type(current).__name__.lower()
        if isinstance(current, (TimeoutError, ConnectionError)) or any(
            marker in name for marker in ("timeout", "connect", "temporar")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _require_staging_deadline(deadline_at: float | None) -> None:
    if deadline_at is not None and float(deadline_at) - time.time() <= 0:
        raise StagedEnvironmentTransientError(
            "environment staging exceeded the authoritative run deadline"
        )


def resolve_environment_source(
    env_id: str,
    resolved_sha: str = "",
    *,
    deadline_at: float | None = None,
) -> ResolvedEnvironmentSource:
    """Download one fresh exact package tree without touching the executable environment cache."""
    from flash.envs.loading import loader
    from flash.envs.meta.identity import GitHubEnvironmentRef, is_commit_sha

    canonical_id, parsed = _reference_for_environment_id(env_id)
    _require_staging_deadline(deadline_at)
    try:
        request_options = {"deadline_at": deadline_at} if deadline_at is not None else {}
        exact_sha = loader._resolve_ref_sha(
            parsed,
            pinned_sha=resolved_sha or None,
            **request_options,
        ).lower()
    except Exception as exc:
        if is_staged_environment_transient_error(exc):
            raise StagedEnvironmentTransientError(
                "environment source is temporarily unavailable"
            ) from exc
        raise
    if not is_commit_sha(exact_sha):
        raise RuntimeError("resolved environment revision is not an immutable commit")
    exact_ref = GitHubEnvironmentRef(parsed.owner, parsed.repo, exact_sha, parsed.path)
    _require_staging_deadline(deadline_at)
    staging_root = Path(tempfile.mkdtemp(prefix="flash-stage-env-source-"))
    try:
        package_root = loader._managed_hub_package_root(exact_ref)
        if exact_ref.repo_full_name.lower() == loader._DEFAULT_MANAGED_ENV_REPO.lower():
            if not package_root:
                raise ValueError(
                    "managed environment hub refs must include a namespace/project/name environment path"
                )
            root = loader._download_github_directory(
                exact_ref,
                package_root,
                staging_root,
                **request_options,
            )
        else:
            root = loader._extract_github_tarball(
                exact_ref,
                staging_root,
                **request_options,
            )
        candidate = root / parsed.path
        if candidate.is_dir():
            candidate = candidate / loader._DEFAULT_ENVIRONMENT_PATH
        entrypoint = candidate.relative_to(root).as_posix()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"environment archive did not contain required entrypoint {entrypoint!r}"
            )
        return ResolvedEnvironmentSource(
            canonical_id,
            exact_sha,
            root,
            entrypoint,
            staging_root,
        )
    except BaseException as exc:
        shutil.rmtree(staging_root, ignore_errors=True)
        if is_staged_environment_transient_error(exc):
            raise StagedEnvironmentTransientError(
                "environment source is temporarily unavailable"
            ) from exc
        raise


def _normalized_tar_info(path: Path, root: Path) -> tarfile.TarInfo:
    info = tarfile.TarInfo(path.relative_to(root).as_posix())
    stat_result = path.stat()
    info.mode = 0o755 if path.is_dir() or stat_result.st_mode & 0o111 else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if path.is_dir():
        info.type = tarfile.DIRTYPE
        info.size = 0
    elif path.is_file():
        info.type = tarfile.REGTYPE
        info.size = stat_result.st_size
    else:
        raise RuntimeError(f"unsupported entry in resolved environment package: {path}")
    return info


def write_environment_archive(source: ResolvedEnvironmentSource, destination: Path) -> str:
    """Write a deterministic bounded package archive and return its sha256."""
    from flash.envs.loading import loader

    paths = sorted(
        source.root.rglob("*"), key=lambda path: path.relative_to(source.root).as_posix()
    )
    if len(paths) > loader._MAX_ARCHIVE_MEMBERS:
        raise RuntimeError(
            f"env package has too many members (limit {loader._MAX_ARCHIVE_MEMBERS})"
        )
    total_bytes = 0
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"unsupported entry in resolved environment package: {path}")
        if path.is_file():
            total_bytes += path.stat().st_size
            if total_bytes > loader._MAX_ARCHIVE_BYTES:
                raise RuntimeError(
                    "environment archive is too large uncompressed "
                    f"({total_bytes} bytes; limit {loader._MAX_ARCHIVE_BYTES} bytes)"
                )
        elif not path.is_dir():
            raise RuntimeError(f"unsupported entry in resolved environment package: {path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for path in paths:
            info = _normalized_tar_info(path, source.root)
            if path.is_file():
                with path.open("rb") as content:
                    archive.addfile(info, content)
            else:
                archive.addfile(info)
    if destination.stat().st_size > loader._MAX_TARBALL_BYTES:
        raise RuntimeError(
            f"environment archive is too large compressed (limit {loader._MAX_TARBALL_BYTES} bytes)"
        )
    digest = hashlib.sha256()
    with destination.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_path_for_digest(archive_sha256: str) -> str:
    return f"environment-packages/archives/sha256/{archive_sha256}/package.tar.gz"


def manifest_path_for_digest(manifest_sha256: str) -> str:
    return f"environment-packages/manifests/sha256/{manifest_sha256}/manifest.json"


def manifest_payload(
    source: ResolvedEnvironmentSource,
    archive_sha256: str,
    archive_path: str,
) -> dict[str, Any]:
    return {
        "version": _MANIFEST_VERSION,
        "canonical_environment_id": source.canonical_id,
        "resolved_sha": source.resolved_sha,
        "entrypoint": source.entrypoint,
        "archive_sha256": archive_sha256,
        "archive_path": archive_path,
    }


def encode_manifest(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _read_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.stat().st_size > _MANIFEST_MAX_BYTES:
        raise RuntimeError("staged environment completion manifest is too large")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("staged environment completion manifest is invalid") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(
            "staged environment completion manifest digest does not match worker spec"
        )
    if not isinstance(payload, dict):
        raise RuntimeError("staged environment completion manifest is invalid")
    return payload


def _validated_manifest_entrypoint(
    environment: EnvironmentSpec,
    package: EnvironmentPackageSpec,
    manifest: dict[str, Any],
) -> str:
    from flash.envs.meta.identity import _normalize_env_path, canonical_environment_id

    expected_keys = {
        "version",
        "canonical_environment_id",
        "resolved_sha",
        "entrypoint",
        "archive_sha256",
        "archive_path",
    }
    if set(manifest) != expected_keys:
        raise RuntimeError("staged environment completion manifest has unexpected fields")
    expected_identity = canonical_environment_id(environment.id)
    expected_archive_path = archive_path_for_digest(package.archive_sha256)
    if (
        manifest.get("version") != _MANIFEST_VERSION
        or manifest.get("canonical_environment_id") != expected_identity
        or manifest.get("resolved_sha") != environment.resolved_sha
        or manifest.get("archive_sha256") != package.archive_sha256
        or manifest.get("archive_path") != expected_archive_path
    ):
        raise RuntimeError(
            "staged environment completion manifest does not match parent environment"
        )
    entrypoint = manifest.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise RuntimeError("staged environment completion manifest entrypoint is invalid")
    try:
        normalized = _normalize_env_path(entrypoint)
    except ValueError as exc:
        raise RuntimeError("staged environment completion manifest entrypoint is invalid") from exc
    if normalized != entrypoint:
        raise RuntimeError("staged environment completion manifest entrypoint is not canonical")
    return entrypoint


def _archive_digest(path: Path) -> str:
    from flash.envs.loading import loader

    if path.stat().st_size > loader._MAX_TARBALL_BYTES:
        raise RuntimeError(
            f"environment archive is too large compressed (limit {loader._MAX_TARBALL_BYTES} bytes)"
        )
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_staged_archive(
    archive_path: Path,
    entrypoint: str,
) -> StagedEnvironmentMaterialization:
    from flash.envs.loading import loader

    owned_root = Path(tempfile.mkdtemp(prefix="flash-staged-env-"))
    root = owned_root / "package"
    try:
        root.mkdir(parents=True, exist_ok=False)
        with archive_path.open("rb") as raw:
            reader = LimitedArchiveReader(
                gzip.GzipFile(fileobj=raw),
                archive_stream_limit(loader._MAX_ARCHIVE_BYTES, loader._MAX_ARCHIVE_MEMBERS),
                lambda: RuntimeError(
                    "environment archive is too large uncompressed "
                    f"(limit {loader._MAX_ARCHIVE_BYTES} bytes)"
                ),
            )
            extract_validated_archive_members(
                reader,
                extract_base=root,
                content_byte_limit=loader._MAX_ARCHIVE_BYTES,
                extracted_member_limit=loader._MAX_ARCHIVE_MEMBERS,
                scanned_member_limit=loader._MAX_ARCHIVE_SCAN_MEMBERS,
            )
        resolved_entrypoint = (root / entrypoint).resolve()
        if root.resolve() not in resolved_entrypoint.parents or not resolved_entrypoint.is_file():
            raise RuntimeError(
                f"staged environment archive did not contain required entrypoint {entrypoint!r}"
            )
        return StagedEnvironmentMaterialization(owned_root, resolved_entrypoint)
    except BaseException:
        shutil.rmtree(owned_root, ignore_errors=True)
        raise


def _download_staged_file(
    *,
    repo_id: str,
    artifact_revision: str,
    filename: str,
    token: str,
) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                revision=artifact_revision,
                token=token,
                force_download=True,
            )
        )
    except Exception as exc:
        if is_staged_environment_transient_error(exc):
            raise StagedEnvironmentTransientError(
                "staged environment artifact store is temporarily unavailable"
            ) from exc
        raise RuntimeError(
            "staged environment artifact is unavailable at its immutable revision"
        ) from exc


def verify_staged_environment(
    environment: EnvironmentSpec,
    *,
    hf_repo: str,
    token: str | None = None,
) -> VerifiedStagedEnvironment:
    """Verify the exact immutable manifest and archive without importing environment code."""
    package = environment.package
    if package is None:
        raise RuntimeError("worker job spec has no staged environment package")
    token = (token if token is not None else os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("staged environment package requires HF_TOKEN")
    manifest_path = manifest_path_for_digest(package.manifest_sha256)
    manifest_file = _download_staged_file(
        repo_id=hf_repo,
        artifact_revision=package.artifact_revision,
        filename=manifest_path,
        token=token,
    )
    manifest = _read_manifest(manifest_file, package.manifest_sha256)
    entrypoint = _validated_manifest_entrypoint(environment, package, manifest)
    archive_path = archive_path_for_digest(package.archive_sha256)
    archive_file = _download_staged_file(
        repo_id=hf_repo,
        artifact_revision=package.artifact_revision,
        filename=archive_path,
        token=token,
    )
    if _archive_digest(archive_file) != package.archive_sha256:
        raise RuntimeError("staged environment archive digest does not match worker spec")
    return VerifiedStagedEnvironment(manifest_file, archive_file, entrypoint)


def materialize_staged_environment(
    environment: EnvironmentSpec,
    *,
    hf_repo: str,
    token: str | None = None,
) -> StagedEnvironmentMaterialization:
    """Download, verify, and safely extract the spec-bound immutable package."""
    verified = verify_staged_environment(
        environment,
        hf_repo=hf_repo,
        token=token,
    )
    return _extract_staged_archive(verified.archive_file, verified.entrypoint)


def load_staged_freesolo_environment(
    environment: EnvironmentSpec,
    params: dict[str, Any] | None,
    *,
    hf_repo: str,
) -> tuple[Any, StagedEnvironmentMaterialization]:
    """Load the verified local package and return its explicit cleanup owner."""
    from flash.envs.loading.loader import _load_resolved_freesolo_environment

    materialization = materialize_staged_environment(environment, hf_repo=hf_repo)
    try:
        loaded = _load_resolved_freesolo_environment(
            environment.id,
            str(materialization.entrypoint),
            dict(params or {}),
        )
    except BaseException:
        materialization.cleanup()
        raise
    return loaded, materialization
