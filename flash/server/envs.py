"""Managed Freesolo environment publishing.

``POST /v1/envs`` accepts a packaged Freesolo environment and stores it as a single ``.tar.gz``
blob in Azure Blob Storage, indexed by slug (``namespace/name``) in Azure PostgreSQL. The returned
id is the slug, which Flash resolves internally: at run submission the control plane looks the slug
up, mints a short-lived read-only SAS URL for the blob, and threads it to the worker — so the GPU
worker downloads the package over plain HTTPS with no Azure credentials (see ``flash/envs/adapter``).
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import tarfile
import tempfile
from pathlib import Path

_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_MEMBERS = 5000
_DEFAULT_ENVIRONMENT_FILE = "environment.py"
# Blob key layout: one tarball per environment slug.
_BLOB_KEY_PREFIX = "flash-envs"
_BLOCKED_TOP_LEVEL_PATHS = {
    ".github",
    ".git",
    "source",
}


def _human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB"


class EnvPublishError(Exception):
    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.status = status


# Reserved hub namespace for the operator/internal service key. Its /v1/me identity is synthetic
# and shared (no per-user email), so it can't derive a namespace from an email like a user key does —
# but it IS a trusted identity that can submit runs and read everything, so it must be able to
# publish environments too. Matches the slug `internal@freesolo.co` would yield (see
# flash.server.db.ensure_internal_key) so the namespace is stable regardless of the row's email.
_INTERNAL_NAMESPACE = "internal-freesolo-co"


def namespace_for(key: dict) -> str:
    # Special case RESERVED for the internal service key ONLY (auth_kind == "internal"): give it a
    # fixed namespace instead of requiring an email. Every other key is a freesolo USER key and still
    # must carry a real email — we never loosen that, so two different users can't collide on one
    # namespace.
    if key.get("auth_kind") == "internal":
        return _INTERNAL_NAMESPACE
    email = str(key.get("email") or "")
    if "@" not in email:
        raise EnvPublishError(
            "authenticated Freesolo key must include an email (used to derive the hub namespace) — "
            "publish with a key created at https://freesolo.co/sign-in (`flash login`)"
        )
    slug = re.sub(r"[^a-z0-9]+", "-", email.lower()).strip("-")
    return slug or "user"


def _sanitize_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")
    if slug in {".", ".."} or not re.search(r"[a-z0-9]", slug):
        return "env"
    return slug or "env"


def blob_key_for(slug: str) -> str:
    """The deterministic blob key for a published environment slug."""
    return f"{_BLOB_KEY_PREFIX}/{slug}/package.tar.gz"


def _safe_extract(tar_bytes: bytes, dest: Path) -> None:
    root = dest.resolve()
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            total = 0
            for count, member in enumerate(tar, start=1):
                if count > _MAX_MEMBERS:
                    raise EnvPublishError(
                        f"env package has too many members (limit {_MAX_MEMBERS})"
                    )
                segments: list[str] = []
                for segment in member.name.replace("\\", "/").split("/"):
                    if not segment or segment == ".":
                        continue
                    if segment == "..":
                        raise EnvPublishError(f"unsafe path in env package: {member.name!r}")
                    segments.append(segment)
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
    except OSError as exc:
        raise EnvPublishError(f"env package could not be extracted: {exc}") from exc


def _environment_file_relative_path(root: Path) -> str:
    canonical = root / _DEFAULT_ENVIRONMENT_FILE
    if canonical.is_file():
        return _DEFAULT_ENVIRONMENT_FILE
    raise EnvPublishError("env package must contain environment.py")


def _azure_publish(*, slug: str, namespace: str, name: str, tar_bytes: bytes) -> None:
    """Upload the package tarball to Azure Blob and index the pointer in Azure Postgres.

    Translates configuration / backend failures into the right HTTP status: 503 when the control
    plane isn't configured for Azure storage, 502 when the upload or index write fails.
    """
    from flash.server import azure_blob, environment_store

    blob_key = blob_key_for(slug)
    sha = hashlib.sha256(tar_bytes).hexdigest()
    try:
        azure_blob.upload_package(blob_key, tar_bytes)
    except azure_blob.AzureBlobNotConfigured as exc:
        raise EnvPublishError(str(exc), status=503) from exc
    except Exception as exc:
        raise EnvPublishError(
            f"failed to upload environment package to Azure Blob storage: {exc}", status=502
        ) from exc
    try:
        environment_store.upsert(
            slug=slug,
            namespace=namespace,
            name=name,
            blob_container=azure_blob.container_name(),
            blob_key=blob_key,
            package_sha256=sha,
            size_bytes=len(tar_bytes),
        )
    except environment_store.EnvironmentStoreNotConfigured as exc:
        raise EnvPublishError(str(exc), status=503) from exc
    except Exception as exc:
        raise EnvPublishError(
            f"failed to index environment package in Azure Postgres: {exc}", status=502
        ) from exc


def publish_package(*, package_b64: str, name: str, key: dict) -> str:
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
    ns = namespace_for(key)
    clean = _sanitize_name(name)
    slug = f"{ns}/{clean}"
    with tempfile.TemporaryDirectory(prefix="flash-env-publish-") as tmp:
        dest = Path(tmp)
        _safe_extract(tar_bytes, dest)
        _environment_file_relative_path(dest)
        if not any(path.is_file() for path in dest.rglob("*")):
            raise EnvPublishError("env package contains no files")
    _azure_publish(slug=slug, namespace=ns, name=clean, tar_bytes=tar_bytes)
    return slug
