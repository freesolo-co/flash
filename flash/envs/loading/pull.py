"""Download published Freesolo environments to local disk."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import gzip
import io
import os
import secrets
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from flash.envs.loading import loader
from flash.envs.package.limits import (
    LimitedArchiveReader,
    archive_stream_limit,
)
from flash.envs.package.unpack import extract_validated_archive_members

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_TRANSACTION_ERROR = "environment pull could not safely complete because filesystem state changed"
_UNSUPPORTED_ERROR = "atomic environment publication is unsupported on this platform"


@dataclass(frozen=True)
class _TreeSnapshot:
    identity: tuple[int, int, int]
    children: tuple[tuple[str, _TreeSnapshot], ...] = ()


@dataclass
class _StagingDirectory:
    parent_fd: int
    name: str
    fd: int
    identity: tuple[int, int, int]


_transaction_sync_hook: Callable[[str], None] | None = None


def _sync_transaction(stage: str) -> None:
    hook = _transaction_sync_hook
    if hook is not None:
        hook(stage)


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


def _changed() -> RuntimeError:
    return RuntimeError(_TRANSACTION_ERROR)


def _identity(result: os.stat_result) -> tuple[int, int, int]:
    return result.st_dev, result.st_ino, stat.S_IFMT(result.st_mode)


def _lstat_identity(path: Path) -> tuple[int, int, int]:
    return _identity(path.lstat())


def _entry_identity(parent_fd: int, name: str) -> tuple[int, int, int]:
    return _identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))


def _fd_identity(fd: int) -> tuple[int, int, int]:
    return _identity(os.fstat(fd))


def _open_directory_at(parent_fd: int, name: str) -> int:
    return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)


def _fd_path(fd: int) -> Path:
    path = Path(f"/proc/self/fd/{fd}")
    if not path.exists():
        raise RuntimeError(_UNSUPPORTED_ERROR)
    return path


def _require_transaction_support() -> None:
    required_dir_fd = {os.open, os.mkdir, os.rename, os.rmdir, os.stat, os.unlink}
    if (
        not sys.platform.startswith("linux")
        or not required_dir_fd.issubset(os.supports_dir_fd)
        or os.stat not in os.supports_follow_symlinks
        or not shutil.rmtree.avoids_symlink_attacks
        or not Path("/proc/self/fd").is_dir()
    ):
        raise RuntimeError(_UNSUPPORTED_ERROR)


def _rename_no_replace_at(
    source_parent_fd: int,
    source_name: str,
    target_parent_fd: int,
    target_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(_UNSUPPORTED_ERROR)
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            source_parent_fd,
            os.fsencode(source_name),
            target_parent_fd,
            os.fsencode(target_name),
            _RENAME_NOREPLACE,
        )
        == 0
    ):
        return
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
        raise RuntimeError(_UNSUPPORTED_ERROR)
    raise OSError(error, os.strerror(error), target_name)


def _new_name(prefix: str) -> str:
    return f".{prefix}-{secrets.token_hex(12)}"


def _mkdir_unique(parent_fd: int, prefix: str) -> str:
    for _ in range(32):
        name = _new_name(prefix)
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        return name
    raise RuntimeError(_TRANSACTION_ERROR)


def _probe_no_replace(parent_fd: int) -> None:
    probe_name = _mkdir_unique(parent_fd, "flash-env-rename-probe")
    probe_fd = _open_directory_at(parent_fd, probe_name)
    probe_identity = _fd_identity(probe_fd)
    source = "source"
    occupied = "occupied"
    installed = "installed"
    try:
        os.mkdir(source, mode=0o700, dir_fd=probe_fd)
        source_identity = _entry_identity(probe_fd, source)
        os.mkdir(occupied, mode=0o700, dir_fd=probe_fd)
        occupied_identity = _entry_identity(probe_fd, occupied)
        try:
            _rename_no_replace_at(probe_fd, source, probe_fd, occupied)
        except FileExistsError:
            pass
        else:
            raise RuntimeError(_UNSUPPORTED_ERROR)
        _rename_no_replace_at(probe_fd, source, probe_fd, installed)
        if _entry_identity(probe_fd, installed) != source_identity:
            raise _changed()
        _remove_owned_entry(
            probe_fd,
            installed,
            source_identity,
            sync_stage="probe_entry_cleanup_ready",
        )
        _remove_owned_entry(
            probe_fd,
            occupied,
            occupied_identity,
            sync_stage="probe_entry_cleanup_ready",
        )
        _remove_owned_entry(
            parent_fd,
            probe_name,
            probe_identity,
            sync_stage="probe_root_cleanup_ready",
        )
    finally:
        os.close(probe_fd)


def _create_staging(parent_fd: int) -> _StagingDirectory:
    name = _mkdir_unique(parent_fd, "flash-env-pull")
    fd = _open_directory_at(parent_fd, name)
    return _StagingDirectory(parent_fd, name, fd, _fd_identity(fd))


def _snapshot_entry(parent_fd: int, name: str) -> _TreeSnapshot:
    identity = _entry_identity(parent_fd, name)
    if not stat.S_ISDIR(identity[2]):
        return _TreeSnapshot(identity)
    child_fd = _open_directory_at(parent_fd, name)
    try:
        children = tuple(
            (child_name, _snapshot_entry(child_fd, child_name))
            for child_name in sorted(os.listdir(child_fd))
        )
    finally:
        os.close(child_fd)
    return _TreeSnapshot(identity, children)


def _restore_claim(parent_fd: int, claim_name: str, original_name: str) -> None:
    with contextlib.suppress(OSError):
        _rename_no_replace_at(parent_fd, claim_name, parent_fd, original_name)


def _verify_snapshot(parent_fd: int, name: str, snapshot: _TreeSnapshot) -> None:
    if _entry_identity(parent_fd, name) != snapshot.identity:
        raise _changed()
    if not stat.S_ISDIR(snapshot.identity[2]):
        return
    child_fd = _open_directory_at(parent_fd, name)
    try:
        if set(os.listdir(child_fd)) != {child_name for child_name, _ in snapshot.children}:
            raise _changed()
        for child_name, child_snapshot in snapshot.children:
            _verify_snapshot(child_fd, child_name, child_snapshot)
    finally:
        os.close(child_fd)


def _claim_entry(parent_fd: int, name: str, snapshot: _TreeSnapshot) -> str:
    claim_name = _new_name("flash-env-owned")
    _rename_no_replace_at(parent_fd, name, parent_fd, claim_name)
    try:
        _verify_snapshot(parent_fd, claim_name, snapshot)
    except Exception:
        _restore_claim(parent_fd, claim_name, name)
        raise
    return claim_name


def _claim_tree_for_move(parent_fd: int, name: str, snapshot: _TreeSnapshot) -> str:
    claim_name = _new_name("flash-env-restore")
    _rename_no_replace_at(parent_fd, name, parent_fd, claim_name)
    try:
        if _snapshot_entry(parent_fd, claim_name) != snapshot:
            raise _changed()
    except Exception:
        _restore_claim(parent_fd, claim_name, name)
        raise
    return claim_name


def _claim_identity(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int, int],
) -> str:
    claim_name = _new_name("flash-env-final")
    _rename_no_replace_at(parent_fd, name, parent_fd, claim_name)
    try:
        if _entry_identity(parent_fd, claim_name) != expected_identity:
            raise _changed()
    except Exception:
        _restore_claim(parent_fd, claim_name, name)
        raise
    return claim_name


def _final_claim(
    parent_fd: int,
    name: str,
    snapshot: _TreeSnapshot,
    *,
    sync_stage: str,
    safety_check: Callable[[], None] | None,
) -> str:
    _sync_transaction(sync_stage)
    if safety_check is not None:
        safety_check()
    if _entry_identity(parent_fd, name) != snapshot.identity:
        raise _changed()
    return _claim_identity(parent_fd, name, snapshot.identity)


def _delete_claimed(
    parent_fd: int,
    name: str,
    snapshot: _TreeSnapshot,
    safety_check: Callable[[], None] | None = None,
) -> None:
    if _entry_identity(parent_fd, name) != snapshot.identity:
        raise _changed()
    if stat.S_ISDIR(snapshot.identity[2]):
        child_fd = _open_directory_at(parent_fd, name)
        try:
            if _fd_identity(child_fd) != snapshot.identity:
                raise _changed()
            for child_name, child_snapshot in snapshot.children:
                claimed_child = _claim_entry(child_fd, child_name, child_snapshot)
                _delete_claimed(child_fd, claimed_child, child_snapshot, safety_check)
            if os.listdir(child_fd):
                raise _changed()
        finally:
            os.close(child_fd)
        final_name = _final_claim(
            parent_fd,
            name,
            snapshot,
            sync_stage="claimed_directory_remove_ready",
            safety_check=safety_check,
        )
        os.rmdir(final_name, dir_fd=parent_fd)
        return
    final_name = _final_claim(
        parent_fd,
        name,
        snapshot,
        sync_stage="claimed_file_unlink_ready",
        safety_check=safety_check,
    )
    os.unlink(final_name, dir_fd=parent_fd)


def _remove_owned_entry(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int, int],
    *,
    sync_stage: str | None = None,
    verify_after_sync: Callable[[], None] | None = None,
) -> None:
    snapshot = _snapshot_entry(parent_fd, name)
    if snapshot.identity != expected_identity:
        raise _changed()
    if sync_stage is not None:
        _sync_transaction(sync_stage)
    if verify_after_sync is not None:
        verify_after_sync()
    claimed_name = _claim_entry(parent_fd, name, snapshot)
    _delete_claimed(parent_fd, claimed_name, snapshot, verify_after_sync)


def _remove_empty_staging(staging: _StagingDirectory) -> None:
    if os.listdir(staging.fd):
        raise _changed()
    _remove_owned_entry(staging.parent_fd, staging.name, staging.identity)


def _close_staging(staging: _StagingDirectory) -> None:
    os.close(staging.fd)


def _copy_tree_into_staging(source: Path, staging: _StagingDirectory, name: str) -> int:
    shutil.copytree(source, _fd_path(staging.fd) / name)
    return _open_directory_at(staging.fd, name)


def _path_matches_fd(path: Path, fd: int) -> bool:
    try:
        return _lstat_identity(path) == _fd_identity(fd)
    except FileNotFoundError:
        return False


def _rollback_moves(
    source_fd: int,
    target_fd: int,
    moved: list[tuple[str, _TreeSnapshot]],
) -> None:
    for name, snapshot in reversed(moved):
        claimed_name = _claim_tree_for_move(target_fd, name, snapshot)
        _sync_transaction("empty_rollback_move_ready")
        if _snapshot_entry(target_fd, claimed_name) != snapshot:
            raise _changed()
        try:
            _rename_no_replace_at(target_fd, claimed_name, source_fd, name)
        except OSError as exc:
            _restore_claim(target_fd, claimed_name, name)
            raise _changed() from exc
        if _snapshot_entry(source_fd, name) != snapshot:
            raise _changed()


def _populate_empty_dir(
    source: Path,
    dest: Path,
    expected_identity: tuple[int, int, int],
) -> None:
    _require_transaction_support()
    parent_fd = os.open(dest.parent, _DIRECTORY_FLAGS)
    dest_fd = -1
    staging: _StagingDirectory | None = None
    payload_fd = -1
    payload_identity: tuple[int, int, int] | None = None
    cleanup_staging = True
    moved: list[tuple[str, _TreeSnapshot]] = []
    try:
        if _entry_identity(parent_fd, dest.name) != expected_identity:
            raise _changed()
        dest_fd = _open_directory_at(parent_fd, dest.name)
        if _fd_identity(dest_fd) != expected_identity:
            raise _changed()
        _probe_no_replace(dest_fd)
        staging = _create_staging(dest_fd)
        payload_fd = _copy_tree_into_staging(source, staging, "contents")
        payload_identity = _fd_identity(payload_fd)
        if set(os.listdir(dest_fd)) != {staging.name}:
            raise FileExistsError(
                f"destination {dest} already exists (pass overwrite=True to replace)"
            )

        _sync_transaction("empty_publish_ready")
        if (
            not _path_matches_fd(dest.parent, parent_fd)
            or _entry_identity(parent_fd, dest.name) != expected_identity
            or _entry_identity(dest_fd, staging.name) != staging.identity
        ):
            raise _changed()

        for name in sorted(os.listdir(payload_fd)):
            child_snapshot = _snapshot_entry(payload_fd, name)
            try:
                _rename_no_replace_at(payload_fd, name, dest_fd, name)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"destination {dest / name} already exists (pass overwrite=True to replace)"
                ) from exc
            if _snapshot_entry(dest_fd, name) != child_snapshot:
                raise _changed()
            moved.append((name, child_snapshot))

        if (
            not _path_matches_fd(dest.parent, parent_fd)
            or _entry_identity(parent_fd, dest.name) != expected_identity
            or _entry_identity(dest_fd, staging.name) != staging.identity
        ):
            raise _changed()
    except Exception as exc:
        try:
            if moved and payload_fd >= 0 and dest_fd >= 0:
                _rollback_moves(payload_fd, dest_fd, moved)
        except Exception:
            cleanup_staging = False
            raise
        if isinstance(exc, RuntimeError) and str(exc) == _TRANSACTION_ERROR:
            cleanup_staging = False
        raise
    finally:
        if payload_fd >= 0:
            os.close(payload_fd)
        if staging is not None:
            try:
                if cleanup_staging:
                    names = os.listdir(staging.fd)
                    if not names:
                        _remove_empty_staging(staging)
                    elif names == ["contents"] and payload_identity is not None:
                        _remove_owned_entry(staging.fd, "contents", payload_identity)
                        _remove_empty_staging(staging)
            finally:
                _close_staging(staging)
        if dest_fd >= 0:
            os.close(dest_fd)
        os.close(parent_fd)


def _replace_with_tree_at_identity(
    source: Path,
    dest: Path,
    expected_identity: tuple[int, int, int] | None,
) -> None:
    _require_transaction_support()
    parent_fd = os.open(dest.parent, _DIRECTORY_FLAGS)
    try:
        _probe_no_replace(parent_fd)
    except Exception:
        os.close(parent_fd)
        raise
    staging = _create_staging(parent_fd)
    payload_fd = -1
    backup_identity: tuple[int, int, int] | None = None
    try:
        payload_fd = _copy_tree_into_staging(source, staging, "new")
        if expected_identity is None:
            try:
                _rename_no_replace_at(staging.fd, "new", parent_fd, dest.name)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"destination {dest} was created while the environment was being pulled"
                ) from exc
            if not _path_matches_fd(dest.parent, parent_fd):
                _rename_no_replace_at(parent_fd, dest.name, staging.fd, "new")
                raise _changed()
        else:
            if _entry_identity(parent_fd, dest.name) != expected_identity:
                raise _changed()
            _rename_no_replace_at(parent_fd, dest.name, staging.fd, "old")
            backup_identity = _entry_identity(staging.fd, "old")
            if backup_identity != expected_identity:
                raise _changed()
            _sync_transaction("publish_ready")
            try:
                _rename_no_replace_at(staging.fd, "new", parent_fd, dest.name)
            except Exception as publish_exc:
                backup_snapshot = _snapshot_entry(staging.fd, "old")
                if backup_snapshot.identity != expected_identity:
                    raise _changed() from publish_exc
                _sync_transaction("rollback_ready")
                claimed_backup = _claim_tree_for_move(staging.fd, "old", backup_snapshot)
                _sync_transaction("rollback_restore_ready")
                restore_name = _claim_tree_for_move(staging.fd, claimed_backup, backup_snapshot)
                try:
                    _rename_no_replace_at(staging.fd, restore_name, parent_fd, dest.name)
                except OSError as restore_exc:
                    _restore_claim(staging.fd, restore_name, "old")
                    raise _changed() from restore_exc
                if _snapshot_entry(parent_fd, dest.name) != backup_snapshot:
                    raise _changed() from publish_exc
                backup_identity = None
                if os.listdir(staging.fd) != ["new"]:
                    raise _changed() from publish_exc
                _remove_owned_entry(
                    staging.fd,
                    "new",
                    _fd_identity(payload_fd),
                    sync_stage="rollback_staging_cleanup_ready",
                )
                os.close(payload_fd)
                payload_fd = -1
                _remove_empty_staging(staging)
                if isinstance(publish_exc, FileExistsError):
                    raise FileExistsError(
                        f"destination {dest} was created while the environment was being pulled"
                    ) from publish_exc
                raise publish_exc

            def verify_publication() -> None:
                if not _path_matches_fd(dest.parent, parent_fd) or _entry_identity(
                    parent_fd, dest.name
                ) != _fd_identity(payload_fd):
                    raise _changed()

            _remove_owned_entry(
                staging.fd,
                "old",
                backup_identity,
                sync_stage="backup_cleanup_ready",
                verify_after_sync=verify_publication,
            )
            backup_identity = None

        if payload_fd >= 0:
            os.close(payload_fd)
            payload_fd = -1
        if os.listdir(staging.fd):
            raise _changed()
        _remove_empty_staging(staging)
    finally:
        if payload_fd >= 0:
            os.close(payload_fd)
        _close_staging(staging)
        os.close(parent_fd)


def _replace_with_tree(source: Path, dest: Path) -> None:
    try:
        expected_identity = _lstat_identity(dest)
    except FileNotFoundError:
        expected_identity = None
    _replace_with_tree_at_identity(source, dest, expected_identity)


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
    try:
        initial_identity = _lstat_identity(dest_path)
    except FileNotFoundError:
        initial_identity = None
    ensure_environment_pull_destination_available(dest_path, overwrite=overwrite)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if initial_identity is None:
        _replace_with_tree_at_identity(source, dest_path, None)
        return

    try:
        current_identity = _lstat_identity(dest_path)
    except FileNotFoundError as exc:
        raise _changed() from exc
    if current_identity != initial_identity:
        raise _changed()

    dest_is_real_dir = stat.S_ISDIR(current_identity[2]) and not dest_path.is_symlink()
    dest_is_empty_dir = dest_is_real_dir and not any(dest_path.iterdir())
    if dest_is_empty_dir:
        _populate_empty_dir(source, dest_path, initial_identity)
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
        _replace_with_tree_at_identity(source, dest_path, initial_identity)


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
