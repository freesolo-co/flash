"""descriptor-bound access to one local publication snapshot."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO


class UnsafeLocalSnapshotError(RuntimeError):
    """the local snapshot contains a symlink or non-regular entry."""


class SnapshotContentMismatchError(RuntimeError):
    """immutable remote content differs from the bound local snapshot."""


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
if hasattr(os, "O_NOFOLLOW"):
    _DIRECTORY_FLAGS |= os.O_NOFOLLOW
    _FILE_FLAGS |= os.O_NOFOLLOW

_InodeIdentity = tuple[int, int]
_StatIdentity = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class SnapshotFileIdentity:
    """local descriptor identity plus both remote content identities."""

    stat_identity: _StatIdentity
    size: int
    sha256: str
    blob_id: str


@dataclass
class LocalSnapshot:
    """scanned files and retained descriptors for every containing directory."""

    paths: list[str]
    directory_fds: dict[tuple[str, ...], int]
    directory_identities: dict[tuple[str, ...], _InodeIdentity]
    file_identities: dict[str, _StatIdentity]

    def close(self) -> None:
        for directory_fd in reversed(tuple(self.directory_fds.values())):
            with contextlib.suppress(OSError):
                os.close(directory_fd)
        self.directory_fds.clear()


def _open_directory_path(local_dir: str) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise UnsafeLocalSnapshotError("no-follow local snapshot access is unavailable")
    absolute = Path(os.path.abspath(os.path.expanduser(local_dir)))
    parts = absolute.parts
    try:
        directory_fd = os.open(parts[0], _DIRECTORY_FLAGS)
        for part in parts[1:]:
            child_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
    except OSError as error:
        with contextlib.suppress(UnboundLocalError):
            os.close(directory_fd)
        raise UnsafeLocalSnapshotError(
            "local adapter snapshot contains a symlink or unsafe path component"
        ) from error
    return directory_fd


def _open_regular_file(directory_fd: int, name: str, relative: str) -> _StatIdentity:
    scanned = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    file_fd = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
    try:
        opened = os.fstat(file_fd)
        if (scanned.st_dev, scanned.st_ino) != (opened.st_dev, opened.st_ino) or not stat.S_ISREG(
            opened.st_mode
        ):
            raise UnsafeLocalSnapshotError(
                f"local adapter snapshot contains changed or non-regular file {relative}"
            )
        return (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
    finally:
        os.close(file_fd)


def _scan_regular_files(
    snapshot: LocalSnapshot,
    directory_fd: int,
    prefix: tuple[str, ...] = (),
) -> None:
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                relative_parts = (*prefix, entry.name)
                relative = "/".join(relative_parts)
                if entry.is_symlink():
                    raise UnsafeLocalSnapshotError(
                        f"local adapter snapshot contains symlink {relative}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    scanned = entry.stat(follow_symlinks=False)
                    child_fd = os.open(entry.name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                    opened = os.fstat(child_fd)
                    if (scanned.st_dev, scanned.st_ino) != (opened.st_dev, opened.st_ino):
                        os.close(child_fd)
                        raise UnsafeLocalSnapshotError(
                            f"local adapter snapshot directory changed while scanning {relative}"
                        )
                    snapshot.directory_fds[relative_parts] = child_fd
                    snapshot.directory_identities[relative_parts] = (opened.st_dev, opened.st_ino)
                    _scan_regular_files(snapshot, child_fd, relative_parts)
                else:
                    snapshot.file_identities[relative] = _open_regular_file(
                        directory_fd, entry.name, relative
                    )
                    snapshot.paths.append(relative)
    except UnsafeLocalSnapshotError:
        raise
    except OSError as error:
        raise UnsafeLocalSnapshotError(
            "local adapter snapshot could not be scanned safely"
        ) from error


@contextlib.contextmanager
def local_snapshot_root(local_dir: str) -> Iterator[LocalSnapshot]:
    """yield scanned paths with every directory descriptor retained."""
    root_fd = _open_directory_path(local_dir)
    root_stat = os.fstat(root_fd)
    snapshot = LocalSnapshot(
        paths=[],
        directory_fds={(): root_fd},
        directory_identities={(): (root_stat.st_dev, root_stat.st_ino)},
        file_identities={},
    )
    try:
        _scan_regular_files(snapshot, root_fd)
        snapshot.paths.sort()
        yield snapshot
    finally:
        snapshot.close()


def _identity(stat_result: os.stat_result) -> _StatIdentity:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_mode,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def snapshot_file_identity(file: BinaryIO, relative: str) -> SnapshotFileIdentity:
    """hash one bound file descriptor and restore its upload position."""
    file.seek(0)
    before = os.fstat(file.fileno())
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(usedforsecurity=False)
    blob.update(f"blob {before.st_size}\0".encode())
    read_size = 0
    while chunk := file.read(1024 * 1024):
        read_size += len(chunk)
        sha256.update(chunk)
        blob.update(chunk)
    after = os.fstat(file.fileno())
    file.seek(0)
    if _identity(before) != _identity(after) or read_size != before.st_size:
        raise UnsafeLocalSnapshotError(f"local adapter snapshot changed while hashing {relative}")
    return SnapshotFileIdentity(
        stat_identity=_identity(before),
        size=before.st_size,
        sha256=sha256.hexdigest(),
        blob_id=blob.hexdigest(),
    )


def revalidate_snapshot_file(
    file: BinaryIO,
    relative: str,
    identity: SnapshotFileIdentity,
) -> None:
    """confirm the same open handle still has its bound metadata identity."""
    try:
        current = os.fstat(file.fileno())
    except (OSError, ValueError) as error:
        raise UnsafeLocalSnapshotError(
            f"local adapter snapshot file became unavailable: {relative}"
        ) from error
    if _identity(current) != identity.stat_identity:
        raise UnsafeLocalSnapshotError(f"local adapter snapshot changed after hashing {relative}")


def snapshot_upload_paths(paths: list[str], ignore_patterns: tuple[str, ...]) -> list[str]:
    """apply hub-compatible exclusions to scanned snapshot paths."""
    from huggingface_hub.utils import filter_repo_objects

    ignored = [
        ".git",
        ".git/*",
        "*/.git",
        "**/.git/**",
        ".cache/huggingface",
        ".cache/huggingface/*",
        "*/.cache/huggingface",
        "**/.cache/huggingface/**",
        *ignore_patterns,
    ]
    return list(filter_repo_objects(paths, ignore_patterns=ignored))


def revalidate_snapshot_directories(snapshot: LocalSnapshot) -> None:
    """confirm every retained nested directory still names its bound inode."""
    nested_paths = sorted(
        (parts for parts in snapshot.directory_fds if parts),
        key=lambda parts: (len(parts), parts),
    )
    for parts in nested_paths:
        relative = "/".join(parts)
        parent_fd = snapshot.directory_fds.get(parts[:-1])
        directory_fd = snapshot.directory_fds[parts]
        expected = snapshot.directory_identities.get(parts)
        if parent_fd is None or expected is None:
            raise UnsafeLocalSnapshotError(
                f"local adapter snapshot directory became unavailable: {relative}"
            )
        try:
            live = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(directory_fd)
        except OSError as error:
            raise UnsafeLocalSnapshotError(
                f"local adapter snapshot directory became unavailable: {relative}"
            ) from error
        if (
            not stat.S_ISDIR(live.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or (live.st_dev, live.st_ino) != expected
            or (opened.st_dev, opened.st_ino) != expected
        ):
            raise UnsafeLocalSnapshotError(
                f"local adapter snapshot directory changed after scanning: {relative}"
            )


def revalidate_snapshot_files(
    files: dict[str, BinaryIO], identities: dict[str, SnapshotFileIdentity]
) -> None:
    """revalidate every committed descriptor against its bound identity."""
    for path, identity in identities.items():
        revalidate_snapshot_file(files[path], path, identity)


def revalidate_snapshot(
    snapshot: LocalSnapshot,
    files: dict[str, BinaryIO],
    identities: dict[str, SnapshotFileIdentity],
) -> None:
    """revalidate retained directories and committed file descriptors."""
    revalidate_snapshot_directories(snapshot)
    revalidate_snapshot_files(files, identities)


def verify_snapshot_content(
    metadata: dict[str, object],
    *,
    prefix: str,
    identities: dict[str, SnapshotFileIdentity],
) -> None:
    """match immutable hub metadata to every committed local file."""
    for relative, identity in identities.items():
        info = metadata[f"{prefix}{relative}"]
        remote_size = getattr(info, "size", None)
        if isinstance(remote_size, bool) or remote_size != identity.size:
            raise SnapshotContentMismatchError(
                f"immutable adapter commit content differs for {relative}"
            )
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            remote_sha256 = getattr(lfs, "sha256", None)
            matches = isinstance(remote_sha256, str) and remote_sha256.lower() == identity.sha256
        else:
            remote_blob_id = getattr(info, "blob_id", None)
            matches = isinstance(remote_blob_id, str) and remote_blob_id.lower() == identity.blob_id
        if not matches:
            raise SnapshotContentMismatchError(
                f"immutable adapter commit content differs for {relative}"
            )


def open_snapshot_file(snapshot: LocalSnapshot, relative: str) -> BinaryIO:
    """open one scanned file from its retained parent directory descriptor."""
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeLocalSnapshotError("local adapter snapshot contains an unsafe relative path")
    parent_fd = snapshot.directory_fds.get(path.parts[:-1])
    scanned_identity = snapshot.file_identities.get(relative)
    if parent_fd is None or scanned_identity is None:
        raise UnsafeLocalSnapshotError(
            f"local adapter snapshot file changed or became unsafe: {relative}"
        )
    try:
        file_fd = os.open(path.parts[-1], _FILE_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        raise UnsafeLocalSnapshotError(
            f"local adapter snapshot file changed or became unsafe: {relative}"
        ) from error
    try:
        opened = os.fstat(file_fd)
        if _identity(opened) != scanned_identity or not stat.S_ISREG(opened.st_mode):
            raise UnsafeLocalSnapshotError(
                f"local adapter snapshot contains changed or non-regular file {relative}"
            )
        flags = fcntl.fcntl(file_fd, fcntl.F_GETFL)
        fcntl.fcntl(file_fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        return os.fdopen(file_fd, "rb")
    except Exception:
        os.close(file_fd)
        raise
