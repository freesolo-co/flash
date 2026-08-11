"""Filesystem trust checks for the on-disk env cache.

Split out of ``flash.envs.loader`` (which owns cache layout, keying, and orchestration) to
keep that module under the file-size gate. These are checks over stat results with no
knowledge of the cache's directory structure -- the loader tells each one which path to look
at and what to do with the answer.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path


def cache_root_is_creatable(root: Path) -> bool:
    """true if this process can actually create ``root``, all of it.

    an arbitrary-uid worker container often still has ``HOME`` pointing at an existing
    directory (``/root``, ``/``) that ``home.is_dir()`` happily confirms but that this process
    cannot write under. without this check the caller would pick that path anyway and
    ``_ensure_cache_root`` would die with ``PermissionError``, never reaching the uid-suffixed
    temp fallback that exists precisely to keep those containers working.

    checking ``home`` alone is not enough, because ``home`` is not the directory the cache
    root is created in. a writable home with a root-owned ``~/.cache`` (mode 0755, left by an
    earlier privileged run or baked into the image) passes a home-only test and then fails at
    ``mkdir(~/.cache/flash)`` with the very ``PermissionError`` the fallback exists to avoid.
    so walk down to the DEEPEST EXISTING directory on the path -- that is the one ``mkdir
    -p`` will have to write into -- and judge that.

    ownership and the write bit both matter: a directory this process happens to have write
    access to but does not own is somewhere another account can interfere with the cache.
    symlinks are followed deliberately; a ``~/.cache`` symlinked onto a volume we own is
    perfectly usable, and the anti-symlink trust checks belong to ``_ensure_cache_root`` and
    ``validate_cache_root_ancestors``, which run against the root that is finally chosen.
    """
    for candidate in (root, *root.parents):
        try:
            info = os.stat(candidate)
        except FileNotFoundError:
            continue
        except OSError:
            return False
        if not stat.S_ISDIR(info.st_mode):
            return False
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            return False
        return os.access(candidate, os.W_OK)
    return False


def validate_cache_root_ancestors(root: Path) -> None:
    """refuse `root` if any ancestor of its resolved path is untrustworthy.

    checking the leaf alone (in ``_ensure_cache_root``) is not enough: with a shared parent --
    e.g. an ``XDG_CACHE_HOME`` pointing somewhere shared -- another local account can own an
    intermediate directory, let this process create a private leaf, then swap that leaf for a
    symlink or attacker tree after the leaf check passes. the later cache-hit path would import
    the swapped-in ``environment.py`` with no re-validation. resolving first (rather than
    walking the raw, possibly-symlinked path) means a legitimate system symlink such as
    macOS's ``/tmp`` -> ``/private/tmp`` is not itself flagged; ``lstat`` on each resolved
    ancestor still catches an ancestor being replaced by a symlink after resolution.

    posix-only, like the mode-bit check in ``_ensure_cache_root``: uid and mode bits do not
    carry the same meaning on Windows.
    """
    if not hasattr(os, "getuid"):
        return
    uid = os.getuid()
    for ancestor in root.resolve().parents:
        info = os.lstat(ancestor)
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(
                f"env cache root ancestor {ancestor} is not a directory; "
                f"refusing to use env cache root {root}"
            )
        if info.st_uid not in (uid, 0):
            raise RuntimeError(
                f"env cache root ancestor {ancestor} is owned by uid {info.st_uid}, not {uid} "
                f"or root; refusing to load environment code from env cache root {root}"
            )
        if (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) and not (info.st_mode & stat.S_ISVTX):
            raise RuntimeError(
                f"env cache root ancestor {ancestor} is group/other-writable without the "
                f"sticky bit (mode {info.st_mode & 0o777:04o}); refusing to load environment "
                f"code from env cache root {root} -- chmod +t it or move the cache root"
            )


def trust_cache_entry(path: Path) -> bool:
    """true if `path` exists, is not a symlink, and (posix) is owned by us.

    a cache root that was previously group/other-writable can still hold a foreign-planted
    directory or symlink at a guessable cache key (a sha of a public repo/ref/path) even after
    the owner repairs the root's own permissions -- ``chmod 700`` does not touch pre-existing
    children. a bare ``is_file``/``is_dir`` check follows symlinks and does not look at
    ownership, so it would happily hand the planted content back on a cache hit.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return False
    return not hasattr(os, "getuid") or info.st_uid == os.getuid()


def discard_untrusted_entry(path: Path) -> None:
    """delete an entry ``trust_cache_entry`` rejected, or refuse the cache key outright.

    removal can genuinely fail: a foreign-owned directory whose contents are readable but not
    writable cannot be emptied by us even when we own the cache root, so ``rmtree`` raises
    partway through. best-effort removal (``ignore_errors=True``) turns that into silence, and
    the caller then downloads the environment before ``copytree`` dies on the entry that is
    still there -- a wasted download and an opaque ``FileExistsError``, repeated on every run,
    for a cache key that has become permanently unusable.

    so removal is VERIFIED and a failure raises here, before the download. the one thing this
    must never do is fall back to using the entry: it is foreign-owned content at a guessable
    key, and the caller's next step is to import it as python.
    """
    reason = ""
    try:
        if path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)
    except OSError as exc:
        reason = f": {exc}"
    if not reason and not os.path.lexists(path):
        return
    raise RuntimeError(
        f"env cache entry {path} is not owned by this user and could not be removed{reason}; "
        "refusing to load environment code from it. delete it as its owner, or point "
        "XDG_CACHE_HOME/HOME at a directory you own, then retry"
    )
