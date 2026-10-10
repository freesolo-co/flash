"""Download published Freesolo environments to local disk."""

from __future__ import annotations

import contextlib
import gzip
import io
import os
import shutil
import tempfile
from pathlib import Path

from flash.envs.loading import loader
from flash.envs.package.limits import (
    LimitedArchiveReader,
    archive_stream_limit,
)
from flash.envs.package.unpack import extract_validated_archive_members


def _safe_repo_relative_path(path: str) -> str:
    raw = (path or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/"):
        raise ValueError(f"invalid environment file path: {path!r}")
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"invalid environment file path: {path!r}")
    return "/".join(parts)


def environment_local_dirname(env_ref: str) -> str:
    """The local directory a pull of ``env_ref`` writes into: the env NAME, not its namespace.

    The slug is ``(namespace, project, name)``, so the name is the LAST segment -- pulling
    `acme/checkout-bot/math` writes `math/`, not a directory named after the project.
    """
    slug = loader._parse_managed_environment_slug(env_ref)
    if slug is None:
        raise ValueError(f"not a managed Freesolo environment slug: {env_ref!r}")
    return slug[2]


def _cwd_is_inside(path: Path) -> bool:
    try:
        cwd = Path.cwd().resolve()
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == cwd or resolved in cwd.parents


def _populate_empty_dir(source: Path, dest: Path) -> None:
    staging_parent = Path(tempfile.mkdtemp(prefix=".flash-env-pull-", dir=dest))
    moved: list[Path] = []
    try:
        staging = staging_parent / "contents"
        shutil.copytree(source, staging)
        if any(child.name != staging_parent.name for child in dest.iterdir()):
            raise FileExistsError(
                f"destination {dest} already exists (pass overwrite=True to replace)"
            )
        for child in sorted(staging.iterdir()):
            target = dest / child.name
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    f"destination {target} already exists (pass overwrite=True to replace)"
                )
            os.replace(child, target)
            moved.append(target)
    except Exception:
        for target in reversed(moved):
            if target.exists() or target.is_symlink():
                with contextlib.suppress(OSError):
                    if target.is_symlink() or not target.is_dir():
                        target.unlink()
                    else:
                        shutil.rmtree(target)
        raise
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def _replace_with_tree(source: Path, dest: Path) -> None:
    staging_parent = Path(tempfile.mkdtemp(prefix=".flash-env-pull-", dir=dest.parent))
    keep_staging = False
    try:
        staging = staging_parent / dest.name
        backup = staging_parent / f"{dest.name}.old"
        shutil.copytree(source, staging)
        if dest.exists() or dest.is_symlink():
            os.replace(dest, backup)
            keep_staging = True
            try:
                os.replace(staging, dest)
            except Exception:
                try:
                    os.replace(backup, dest)
                    keep_staging = False
                except OSError as restore_exc:
                    raise RuntimeError(
                        f"failed to replace {dest}; original retained at {backup}"
                    ) from restore_exc
                raise
            keep_staging = False
        else:
            os.replace(staging, dest)
    finally:
        if not keep_staging:
            shutil.rmtree(staging_parent, ignore_errors=True)


def ensure_environment_pull_destination_available(
    dest: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    dest_path = Path(dest)
    if (
        dest_path.is_symlink()
        or (dest_path.exists() and (not dest_path.is_dir() or any(dest_path.iterdir())))
    ) and not overwrite:
        raise FileExistsError(
            f"destination {dest_path} already exists (pass overwrite=True to replace)"
        )
    if (
        overwrite
        and dest_path.is_dir()
        and not dest_path.is_symlink()
        and _cwd_is_inside(dest_path)
    ):
        raise RuntimeError(
            f"refusing to overwrite {dest_path} because it contains the current working directory; "
            "choose a separate output path"
        )
    return dest_path


def _copy_environment_source(source: Path, dest_path: Path, *, overwrite: bool = False) -> None:
    ensure_environment_pull_destination_available(dest_path, overwrite=overwrite)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_is_real_dir = dest_path.is_dir() and not dest_path.is_symlink()
    dest_is_empty_dir = dest_is_real_dir and not any(dest_path.iterdir())
    if not dest_path.exists() and not dest_path.is_symlink():
        shutil.copytree(source, dest_path)
    elif dest_is_empty_dir:
        _populate_empty_dir(source, dest_path)
    elif dest_is_real_dir and _cwd_is_inside(dest_path):
        raise RuntimeError(
            f"refusing to overwrite {dest_path} because it contains the current working directory; "
            "choose a separate output path"
        )
    else:
        if not overwrite:
            raise FileExistsError(
                f"destination {dest_path} already exists (pass overwrite=True to replace)"
            )
        _replace_with_tree(source, dest_path)


def _extract_environment_package_archive(package: bytes | bytearray, dest: Path) -> Path:
    """Extract a flat environment package tarball into ``dest`` and return its root."""
    if len(package) > loader._MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"environment archive is too large compressed (limit {loader._MAX_ARCHIVE_BYTES} bytes)"
        )

    root = (dest / "package").resolve()
    root.mkdir(parents=True, exist_ok=True)
    reader = LimitedArchiveReader(
        gzip.GzipFile(fileobj=io.BytesIO(package)),
        archive_stream_limit(loader._MAX_ARCHIVE_BYTES, loader._MAX_ARCHIVE_MEMBERS),
        lambda: RuntimeError(
            f"environment archive is too large uncompressed (limit {loader._MAX_ARCHIVE_BYTES} bytes)"
        ),
    )
    extract_validated_archive_members(
        reader,
        extract_base=root,
        content_byte_limit=loader._MAX_ARCHIVE_BYTES,
        extracted_member_limit=loader._MAX_ARCHIVE_MEMBERS,
        scanned_member_limit=loader._MAX_ARCHIVE_SCAN_MEMBERS,
    )
    return root


def pull_environment_package_from_archive(
    package: bytes | bytearray,
    dest: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Extract a backend-returned managed environment package to ``dest``."""
    dest_path = ensure_environment_pull_destination_available(dest, overwrite=overwrite)

    tmp_parent = Path(tempfile.mkdtemp(prefix="flash-env-pull-"))
    try:
        source = _extract_environment_package_archive(package, tmp_parent)
        if not (source / loader._DEFAULT_ENVIRONMENT_PATH).is_file():
            raise FileNotFoundError(
                f"environment entrypoint {loader._DEFAULT_ENVIRONMENT_PATH!r} not found in package"
            )
        _copy_environment_source(source, dest_path, overwrite=overwrite)
        return dest_path
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)


def download_environment_file_from_archive(
    package: bytes | bytearray,
    rel_path: str,
) -> bytes:
    """Read one file from a backend-returned managed environment package."""
    safe_rel = _safe_repo_relative_path(rel_path)
    tmp_parent = Path(tempfile.mkdtemp(prefix="flash-env-pull-"))
    try:
        source = _extract_environment_package_archive(package, tmp_parent)
        target = source / safe_rel
        if not target.is_file():
            raise FileNotFoundError(f"environment file {safe_rel!r} not found in package")
        return target.read_bytes()
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)
