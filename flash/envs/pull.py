"""Download published Freesolo environments to local disk."""

from __future__ import annotations

import contextlib
import gzip
import io
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from flash.envs import loader as adapter
from flash.envs.archive_policy import (
    LimitedArchiveReader,
    archive_stream_limit,
    tar_member_segments,
)


def _coerce_environment_github_ref(env_ref: str) -> adapter.GitHubEnvironmentRef:
    candidate = env_ref
    if adapter.is_managed_environment_slug(candidate):
        candidate = adapter.managed_slug_to_github_ref(candidate)
    parsed = adapter._parse_github_environment_ref(candidate)
    if parsed is None:
        raise ValueError(f"not a Freesolo or GitHub environment reference: {env_ref!r}")
    return parsed


def _safe_repo_relative_path(path: str) -> str:
    raw = (path or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/"):
        raise ValueError(f"invalid environment file path: {path!r}")
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"invalid environment file path: {path!r}")
    return "/".join(parts)


def _environment_dir(env_path: str) -> str:
    parts = [part for part in env_path.split("/") if part]
    if parts and parts[-1].endswith(".py"):
        parts = parts[:-1]
    return "/".join(parts)


def _environment_entrypoint(env_path: str) -> str:
    parts = [part for part in env_path.split("/") if part]
    if parts and parts[-1].endswith(".py"):
        return parts[-1]
    return adapter._DEFAULT_ENVIRONMENT_PATH


def environment_local_dirname(env_ref: str) -> str:
    slug = adapter._parse_managed_environment_slug(env_ref)
    if slug is not None:
        return slug[1]
    ref = _coerce_environment_github_ref(env_ref)
    env_dir = _environment_dir(ref.path)
    return env_dir.rsplit("/", 1)[-1] if env_dir else ref.repo


def download_environment_file(env_ref: str, rel_path: str, *, timeout: float = 120.0) -> bytes:
    """Download one file from an environment using GitHub's raw media type."""
    ref = _coerce_environment_github_ref(env_ref)
    safe_rel = _safe_repo_relative_path(rel_path)
    env_dir = _environment_dir(ref.path)
    full_path = f"{env_dir}/{safe_rel}" if env_dir else safe_rel
    quoted_path = "/".join(urllib.parse.quote(part, safe="") for part in full_path.split("/"))
    url = (
        f"https://api.github.com/repos/{ref.repo_full_name}/contents/{quoted_path}"
        f"?ref={urllib.parse.quote(ref.ref, safe='')}"
    )
    headers = {"Accept": "application/vnd.github.raw", "User-Agent": "freesolo-flash"}
    token = adapter._github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return adapter._urlopen(
        urllib.request.Request(url, headers=headers),
        timeout=timeout,
        max_bytes=adapter._MAX_ARCHIVE_BYTES,
    )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _path_occupied(path: Path) -> bool:
    return path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir())))


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
                    _remove_path(target)
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


def _copy_environment_source(source: Path, dest_path: Path, *, overwrite: bool = False) -> None:
    if _path_occupied(dest_path) and not overwrite:
        raise FileExistsError(
            f"destination {dest_path} already exists (pass overwrite=True to replace)"
        )

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
    if len(package) > adapter._MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"environment archive is too large compressed (limit {adapter._MAX_ARCHIVE_BYTES} bytes)"
        )

    root = (dest / "package").resolve()
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    extracted = 0
    scanned = 0
    reader = LimitedArchiveReader(
        gzip.GzipFile(fileobj=io.BytesIO(package)),
        archive_stream_limit(adapter._MAX_ARCHIVE_BYTES, adapter._MAX_ARCHIVE_MEMBERS),
        lambda: RuntimeError(
            f"environment archive is too large uncompressed (limit {adapter._MAX_ARCHIVE_BYTES} bytes)"
        ),
    )
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            scanned += 1
            if scanned > adapter._MAX_ARCHIVE_SCAN_MEMBERS:
                raise RuntimeError(
                    "env package has too many entries to scan "
                    f"(limit {adapter._MAX_ARCHIVE_SCAN_MEMBERS})"
                )
            if member.type in adapter._TAR_METADATA_TYPES:
                continue
            raw = tar_member_segments(
                member.name,
                unsafe_error=lambda name: RuntimeError(
                    f"unsafe path in environment archive: {name!r}"
                ),
            )
            if not raw:
                continue
            normalized_name = "/".join(raw)
            target = (root / normalized_name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe path in environment archive: {member.name!r}")
            if member.islnk() or member.issym() or not (member.isreg() or member.isdir()):
                continue
            extracted += 1
            if extracted > adapter._MAX_ARCHIVE_MEMBERS:
                raise RuntimeError(
                    f"env package has too many members (limit {adapter._MAX_ARCHIVE_MEMBERS})"
                )
            total += max(0, member.size)
            if total > adapter._MAX_ARCHIVE_BYTES:
                raise RuntimeError(
                    "environment archive is too large uncompressed "
                    f"({total} bytes; limit {adapter._MAX_ARCHIVE_BYTES} bytes)"
                )
            member.name = normalized_name
            tar.extract(member, root)
    return root


def pull_environment_package_from_archive(
    package: bytes | bytearray,
    dest: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Extract a backend-returned managed environment package to ``dest``."""
    dest_path = Path(dest)
    if _path_occupied(dest_path) and not overwrite:
        raise FileExistsError(
            f"destination {dest_path} already exists (pass overwrite=True to replace)"
        )

    tmp_parent = Path(tempfile.mkdtemp(prefix="flash-env-pull-"))
    try:
        source = _extract_environment_package_archive(package, tmp_parent)
        if not (source / adapter._DEFAULT_ENVIRONMENT_PATH).is_file():
            raise FileNotFoundError(
                f"environment entrypoint {adapter._DEFAULT_ENVIRONMENT_PATH!r} not found in package"
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


def pull_environment_package(env_ref: str, dest: str | Path, *, overwrite: bool = False) -> Path:
    """Download a published environment directory to ``dest``."""
    ref = _coerce_environment_github_ref(env_ref)
    env_dir = _environment_dir(ref.path)
    if not env_dir and ref.repo_full_name.lower() == adapter._DEFAULT_MANAGED_ENV_REPO.lower():
        raise ValueError(
            f"refusing to pull the whole shared environment hub ({adapter._DEFAULT_MANAGED_ENV_REPO}); "
            "specify a managed environment id (namespace/name) or a github ref to its subdirectory"
        )

    dest_path = Path(dest)
    if _path_occupied(dest_path) and not overwrite:
        raise FileExistsError(
            f"destination {dest_path} already exists (pass overwrite=True to replace)"
        )

    tmp_parent = Path(tempfile.mkdtemp(prefix="flash-env-pull-"))
    try:
        extracted = adapter._extract_github_tarball(ref, tmp_parent, subdir=env_dir)
        source = extracted / env_dir if env_dir else extracted
        if not source.is_dir():
            raise FileNotFoundError(
                f"environment directory {env_dir or '.'!r} not found in {ref.repo_full_name}@{ref.ref}"
            )

        entrypoint = _environment_entrypoint(ref.path)
        if not (source / entrypoint).is_file():
            raise FileNotFoundError(
                f"environment entrypoint {entrypoint!r} not found in "
                f"{env_dir or '.'!r} of {ref.repo_full_name}@{ref.ref}"
            )

        _copy_environment_source(source, dest_path, overwrite=overwrite)
        return dest_path
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)
