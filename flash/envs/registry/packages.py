"""Validated archive transport between the managed registry and environment loaders."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from flash.envs.meta.identity import canonical_managed_environment_slug
from flash.envs.registry.azure import RegistryError, enabled, open_registry


def managed_slug(value: str) -> str | None:
    return canonical_managed_environment_slug(value) if enabled() else None


def resolve_revision(slug: str, revision: str = "", *, deadline_at: float | None = None) -> str:
    with open_registry(deadline_at=deadline_at) as registry:
        return registry.resolve(slug, revision).revision


def resolve_source(slug: str, revision: str = "", *, deadline_at: float | None = None):
    from flash.envs.loading.pull import _extract_environment_package_archive
    from flash.envs.loading.staged import ResolvedEnvironmentSource

    with open_registry(deadline_at=deadline_at) as registry:
        record = registry.resolve(slug, revision)
        archive = registry.download(record)
    staging = Path(tempfile.mkdtemp(prefix="flash-env-blob-"))
    try:
        root = _extract_environment_package_archive(archive, staging)
        if not (root / "environment.py").is_file():
            raise RegistryError("environment archive has no environment.py")
        return ResolvedEnvironmentSource(slug, record.revision, root, "environment.py", staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def cached_entrypoint(slug: str, revision: str = "") -> Path:
    from flash.envs.loading import loader
    from flash.envs.meta.cache_storage import find_cached_entry, publish_cache_entry

    revision = resolve_revision(slug, revision)
    # Registry revision IDs are content-addressed, so a cache hit cannot cross contents.
    import hashlib

    cache_key = hashlib.sha256(f"azure:{slug}:{revision}".encode()).hexdigest()[:24]
    destination = loader._ensure_cache_root() / cache_key
    cached = find_cached_entry(destination, "environment.py", "environment.py")
    if cached is not None:
        return cached
    source = resolve_source(slug, revision)
    try:
        publish_cache_entry(source.root, destination, source.entrypoint)
        loader._evict_env_cache(keep=destination)
        return destination / source.entrypoint
    finally:
        shutil.rmtree(source.staging_root, ignore_errors=True)


def publish_directory(
    slug: str, root: Path, *, create_only: bool = False, source_revision: str = ""
):
    from flash.envs.loading.staged import ResolvedEnvironmentSource, write_environment_archive

    if not (root / "environment.py").is_file():
        raise RegistryError("environment package must contain environment.py", status=400)
    with tempfile.TemporaryDirectory(prefix="flash-env-upload-") as tmp:
        archive = Path(tmp) / "package.tar.gz"
        write_environment_archive(
            ResolvedEnvironmentSource(slug, "", root, "environment.py"), archive
        )
        with open_registry() as registry:
            return registry.publish(
                slug, archive.read_bytes(), create_only=create_only, source_revision=source_revision
            )
