"""Managed Freesolo environment publishing."""

from __future__ import annotations

import base64
import gzip
import io
import re
import tarfile
import tempfile
from pathlib import Path

from flash.envs.package.direct_tokens import DirectTokenScanError, package_contains_direct_token
from flash.envs.package.limits import (
    ARCHIVE_MEMBER_LIMIT,
    ARCHIVE_SCAN_MEMBER_LIMIT,
    TAR_METADATA_TYPES,
    LimitedArchiveReader,
    archive_stream_limit,
    tar_member_segments,
)
from flash.server.domain.registry import blob_envs

_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_MEMBERS = ARCHIVE_MEMBER_LIMIT
_MAX_SCAN_MEMBERS = ARCHIVE_SCAN_MEMBER_LIMIT
_DEFAULT_ENVIRONMENT_FILE = "environment.py"
# Reserved namespaces and archive paths remain invalid after the storage migration.
_REPO_CONTROL_TOP_LEVEL_PATHS = {
    ".github",
    ".git",
}
# Top-level segments barred from env-package CONTENTS by `_safe_extract`. A strict SUPERSET of the
# repo-control set: it also bars a top-level `source/` dir inside a package. `source` is intentionally
# NOT a repo-control namespace, so the delete validator must reject only
# `_REPO_CONTROL_TOP_LEVEL_PATHS`, not this set, or `source/<name>` envs become undeletable.
_BLOCKED_TOP_LEVEL_PATHS = _REPO_CONTROL_TOP_LEVEL_PATHS | {"source"}
_NAMESPACE_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")


def _human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB"


class EnvPublishError(Exception):
    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.status = status


def namespace_for(key: dict) -> str:
    org = key.get("org") if isinstance(key.get("org"), dict) else {}
    raw = key.get("org_slug") or org.get("slug")
    slug = str(raw or "").strip()
    if not slug:
        raise EnvPublishError(
            "authenticated Freesolo key must include an org slug (used to derive the hub namespace) — "
            "publish with a key created at https://freesolo.co/sign-in (`flash login`)"
        )
    if not _NAMESPACE_RE.fullmatch(slug):
        raise EnvPublishError("authenticated Freesolo key has an invalid org slug")
    return slug


def _sanitize_name(name: str) -> str:
    from flash.schema import normalize_env_name_segment

    return normalize_env_name_segment(name) or "env"


def publish_slug_for_name(name: str, key: dict, project_slug: str) -> tuple[str, str, str]:
    """The ``(namespace, project, name)`` a publish of ``name`` by ``key`` would write.

    Environment names are unique per PROJECT, so the owning project is part of the published
    identity: the destination is ``<org-slug>/<project-slug>/<name>``, which is also the
    key prefix in the managed registry.

    Pure and side-effect free, so the publish route can resolve the destination slug before
    uploading anything. Accepts a bare name, or a fully qualified id whose namespace and project
    segments must match the caller's org and the project being published to -- a qualified id
    that disagrees is refused rather than silently redirected.
    """
    caller_namespace = namespace_for(key)
    project = str(project_slug or "").strip()
    if not project:
        # The route resolves the slug through `require_project_access_slug`, which already refuses
        # an empty one and says WHY (standalone has no project directory; a validation response
        # missing it is an upstream fault). Neither cause is the caller's key, so this no longer
        # blames one -- it stays as the last guard for a direct domain-level caller, which is why
        # it is a 500: reaching here means a caller skipped the resolution that would have
        # explained it.
        raise EnvPublishError(
            "the project's slug was not resolved before publishing, so the environment's "
            "destination is unknown",
            status=500,
        )
    if not _NAMESPACE_RE.fullmatch(project):
        raise EnvPublishError("project slug must match [a-z0-9][a-z0-9._-]*")
    raw = str(name or "").strip()
    if "/" not in raw:
        return caller_namespace, project, _sanitize_name(raw)
    parts = [part.strip() for part in raw.split("/")]
    if len(parts) != 3 or not all(parts):
        raise EnvPublishError("env name with namespace must be 'namespace/project/name'")
    namespace, given_project = parts[0], parts[1]
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise EnvPublishError("env namespace must match [a-z0-9][a-z0-9._-]*")
    clean_name = _sanitize_name(parts[2])
    if namespace != caller_namespace:
        raise EnvPublishError(
            "env namespace must match your Freesolo org namespace "
            f"({caller_namespace}/...); got {namespace}/{given_project}/{clean_name}",
            status=403,
        )
    if given_project != project:
        raise EnvPublishError(
            "env project segment must match the project you are publishing to "
            f"({caller_namespace}/{project}/...); got {namespace}/{given_project}/{clean_name}",
            status=403,
        )
    return namespace, project, clean_name


def _safe_extract(tar_bytes: bytes, dest: Path) -> None:
    root = dest.resolve()
    try:
        reader = LimitedArchiveReader(
            gzip.GzipFile(fileobj=io.BytesIO(tar_bytes)),
            archive_stream_limit(_MAX_UNCOMPRESSED_BYTES, _MAX_MEMBERS),
            lambda: EnvPublishError(
                f"env package is too large uncompressed (limit {_human_mb(_MAX_UNCOMPRESSED_BYTES)})"
            ),
        )
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            total = 0
            scanned = 0
            extracted = 0
            for member in tar:
                scanned += 1
                if scanned > _MAX_SCAN_MEMBERS:
                    raise EnvPublishError(
                        f"env package has too many entries to scan (limit {_MAX_SCAN_MEMBERS})"
                    )
                if member.type in TAR_METADATA_TYPES:
                    continue
                segments = tar_member_segments(
                    member.name,
                    unsafe_error=lambda name: EnvPublishError(
                        f"unsafe path in env package: {name!r}"
                    ),
                )
                if not segments:
                    continue
                normalized_name = "/".join(segments)
                target = (dest / normalized_name).resolve()
                if target != root and root not in target.parents:
                    raise EnvPublishError(f"unsafe path in env package: {member.name!r}")
                if segments[0] in _BLOCKED_TOP_LEVEL_PATHS:
                    raise EnvPublishError(
                        "env packages must not contain repo-control or source top-level paths"
                    )
                if member.islnk() or member.issym():
                    raise EnvPublishError(f"links are not allowed in env packages: {member.name!r}")
                if not (member.isreg() or member.isdir()):
                    raise EnvPublishError(
                        f"only regular files and directories are allowed in env packages, "
                        f"but {member.name!r} is a special file"
                    )
                extracted += 1
                if extracted > _MAX_MEMBERS:
                    raise EnvPublishError(
                        f"env package has too many members (limit {_MAX_MEMBERS})"
                    )
                total += max(0, member.size)
                if total > _MAX_UNCOMPRESSED_BYTES:
                    raise EnvPublishError(
                        "env package is too large uncompressed "
                        f"(limit {_human_mb(_MAX_UNCOMPRESSED_BYTES)})"
                    )
                member.name = normalized_name
                tar.extract(member, dest)
    except tarfile.TarError as exc:
        raise EnvPublishError(f"env package is not a valid .tar.gz archive: {exc}") from exc
    except (OSError, EOFError) as exc:
        raise EnvPublishError(f"env package could not be extracted: {exc}") from exc


def validate_publish_inputs(*, package_b64: object, name: object) -> bytes:
    """Check a publish request's own inputs and return its decoded package.

    Pure and side-effect free, so the route can reject an unpublishable request before doing
    anything observable — no ownership lookup, no backend call, no upload. Raises the same
    :class:`EnvPublishError` (400/413) the publish itself would raise, so the failure a caller
    sees does not depend on when the check ran.
    """
    if not isinstance(name, str):
        raise EnvPublishError("env name must be a string")
    if not isinstance(package_b64, str):
        raise EnvPublishError("env package must be a base64 string")
    if not name:
        raise EnvPublishError("missing env name")
    max_encoded = ((_MAX_UPLOAD_BYTES + 2) // 3) * 4 + 3
    if len(package_b64) > max_encoded:
        raise EnvPublishError(
            f"env package upload is too large (limit {_human_mb(_MAX_UPLOAD_BYTES)} compressed)",
            status=413,
        )
    try:
        tar_bytes = base64.b64decode(package_b64, validate=True)
    except Exception as exc:
        raise EnvPublishError("env package is not valid base64") from exc
    if not tar_bytes:
        raise EnvPublishError("empty env package")
    if len(tar_bytes) > _MAX_UPLOAD_BYTES:
        raise EnvPublishError(
            f"env package upload is too large (limit {_human_mb(_MAX_UPLOAD_BYTES)} compressed)",
            status=413,
        )
    return tar_bytes


def publish_package(
    *,
    package_b64: str,
    name: str,
    key: dict,
    project_slug: str,
) -> str:
    """Publish a package to the hub under ``<namespace>/<project>/<name>``."""
    tar_bytes = validate_publish_inputs(package_b64=package_b64, name=name)
    with tempfile.TemporaryDirectory(prefix="flash-env-publish-") as tmp:
        dest = Path(tmp)
        _safe_extract(tar_bytes, dest)
        if not (dest / _DEFAULT_ENVIRONMENT_FILE).is_file():
            raise EnvPublishError("env package must contain environment.py")
        try:
            contains_direct_token = package_contains_direct_token(dest)
        except DirectTokenScanError:
            raise EnvPublishError("env package could not be scanned safely") from None
        if contains_direct_token:
            raise EnvPublishError(
                "env package contains a direct access token; remove it before publishing"
            )
        slug = "/".join(publish_slug_for_name(name, key, project_slug))
        return blob_envs.publish(slug, dest)


_SLUG_SEGMENT_RE = re.compile(r"^[a-z0-9._-]+$")


def _validate_slug(slug: str) -> tuple[str, str, str]:
    """Validate a ``namespace/project/name`` environment id and return its three segments.

    Rejects anything that isn't exactly three non-empty path-safe segments — in particular
    ``..`` and stray separators — so the slug can be used directly as a git pathspec / on-disk
    publish root without traversal risk (mirrors the guarantees `_sanitize_name` gives publish).
    """
    if not isinstance(slug, str):
        raise EnvPublishError("env id must be a string")
    # reject rather than normalize: whitespace or stray separators must not delete the canonical
    # slug
    # while returning a non-canonical id. validate the raw segments.
    parts = slug.split("/")
    if len(parts) != 3 or not all(parts):
        raise EnvPublishError("env id must be 'namespace/project/name'")
    for segment in parts:
        if segment in {".", ".."} or not _SLUG_SEGMENT_RE.fullmatch(segment):
            raise EnvPublishError(f"invalid env id segment: {segment!r}")
    namespace, project, name = parts
    # Preserve the existing namespace contract, including the valid source namespace.
    if namespace in _REPO_CONTROL_TOP_LEVEL_PATHS:
        raise EnvPublishError(f"invalid env id segment: {namespace!r}")
    return namespace, project, name


def canonical_env_id(slug: str) -> str:
    """Validate ``slug`` and return the canonical ``namespace/project/name`` id.

    Public wrapper over :func:`_validate_slug` so the route can normalize the id ONCE up front and
    use the same canonical value for deletion, the metadata-mirror drop, and the response — never a
    non-canonical variant. Raises :class:`EnvPublishError` (400) for anything that isn't already
    canonical, so a padded / stray-separator id is rejected rather than silently trimmed.
    """
    namespace, project, name = _validate_slug(slug)
    return f"{namespace}/{project}/{name}"


def _require_namespace_access(canonical: str, key: dict, *, action: str) -> None:
    namespace = canonical.split("/", 1)[0]
    caller_namespace = None if key.get("auth_kind") == "internal" else namespace_for(key)
    if caller_namespace is not None and namespace != caller_namespace:
        raise EnvPublishError(
            f"you can only {action} environments in your own namespace "
            f"({caller_namespace}/...); got {canonical!r}",
            status=403,
        )


def download_package(*, slug: str, key: dict) -> bytes:
    """Download the current package from the private managed registry."""
    canonical = canonical_env_id(slug)
    _require_namespace_access(canonical, key, action="download")
    return blob_envs.download(canonical)


def list_namespace_slugs(*, key: dict) -> list[str]:
    """List packages belonging to the authenticated organization."""
    return blob_envs.list_slugs(namespace_for(key))


def delete_package(*, slug: str, key: dict) -> bool:
    """Remove the current head, retaining immutable revisions for pinned jobs."""
    canonical = canonical_env_id(slug)
    _require_namespace_access(canonical, key, action="delete")
    return blob_envs.delete(canonical)
