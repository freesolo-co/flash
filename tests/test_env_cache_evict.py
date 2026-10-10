"""unit tests for the bounded, user-private on-disk env cache."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from flash.envs.loading import loader as adapter
from flash.envs.meta import cache_security


def _make_private_cache_root(root: Path) -> None:
    """Create ``root`` and its intermediates 0700, independently of the caller's umask.

    ``root.mkdir(parents=True, mode=0o700)`` applies the mode to the LEAF only -- every intermediate
    it creates gets ``0o777 & ~umask`` instead. Under CI's umask 022 that is 0755 and the ancestor
    checks pass; under a developer's umask 002 it is 0775, which is group-writable, so
    ``_ensure_cache_root`` refuses the root and the test fails on a clean checkout for a reason that
    has nothing to do with the behaviour under test. Set the mode explicitly on the way down.
    """
    root.mkdir(parents=True, exist_ok=True)
    for path in (root, root.parent):
        path.chmod(0o700)


def test_cache_config_ignores_ambient_overrides(tmp_path):
    # the on-disk env cache location and bounds are hardcoded, not env-tunable: ambient
    # FLASH_ENV_CACHE_* vars (a stray shell or CI export) must never change them. the root
    # is user-scoped off XDG_CACHE_HOME/HOME so no other local account can pre-seed it.
    expected = tmp_path / "xdg" / "flash" / "env-cache"
    script = (
        "import os; "
        "from pathlib import Path; "
        "from flash.envs.loading import loader; "
        "assert loader._CACHE_ROOT == Path(os.environ['EXPECTED_CACHE_ROOT']), "
        "loader._CACHE_ROOT; "
        "assert loader._CACHE_MAX_ENTRIES == 32; "
        "assert loader._CACHE_MAX_BYTES == 4 * 1024 * 1024 * 1024"
    )
    for cache_dir, max_entries, max_bytes in (
        ("/some/other/dir", "1", "2"),
        ("also-bogus", "invalid", "also-invalid"),
    ):
        env = os.environ.copy()
        env["XDG_CACHE_HOME"] = str(tmp_path / "xdg")
        env["EXPECTED_CACHE_ROOT"] = str(expected)
        env["FLASH_ENV_CACHE_DIR"] = cache_dir
        env["FLASH_ENV_CACHE_MAX_ENTRIES"] = max_entries
        env["FLASH_ENV_CACHE_MAX_BYTES"] = max_bytes
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            check=True,
        )


def test_cache_root_falls_back_to_uid_scoped_tmp_when_homeless(monkeypatch, tmp_path):
    # worker containers can have neither XDG_CACHE_HOME nor a resolvable home; the fallback
    # must still be per-uid rather than a shared /tmp name.
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(adapter.os.path, "expanduser", lambda _p: "~")
    monkeypatch.setattr(adapter.tempfile, "gettempdir", lambda: str(tmp_path))

    root = adapter._default_cache_root()

    # derived the same way production does: os.getuid does not exist on windows, where the
    # fallback deliberately substitutes uid 0, and collection must not die on the assert.
    uid = os.getuid() if hasattr(os, "getuid") else 0
    assert root == tmp_path / f"flash-env-cache-{uid}"


def test_default_cache_root_falls_back_when_xdg_hierarchy_not_writable(monkeypatch, tmp_path):
    # XDG_CACHE_HOME is commonly inherited from a container image (e.g. /home/app/.cache) and
    # points somewhere an arbitrary-uid worker cannot create under. taking it unconditionally,
    # unlike the HOME branch, means _ensure_cache_root dies with PermissionError on every
    # github env resolve instead of reaching the uid-scoped temp fallback.
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.setattr(adapter.os.path, "expanduser", lambda _p: "~")
    monkeypatch.setattr(adapter.os, "access", lambda _path, _mode: False)
    monkeypatch.setattr(adapter.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    root = adapter._default_cache_root()

    uid = os.getuid() if hasattr(os, "getuid") else 0
    assert root == tmp_path / "tmp" / f"flash-env-cache-{uid}"


def test_default_cache_root_uses_a_usable_xdg_hierarchy(monkeypatch, tmp_path):
    # the other side of that check: a writable XDG_CACHE_HOME we own must still win, so the
    # new usability probe rejects, never relocates a perfectly good cache root.
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.setattr(adapter.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    assert adapter._default_cache_root() == xdg / "flash" / "env-cache"


def test_default_cache_root_falls_back_when_home_not_writable(monkeypatch, tmp_path):
    # an arbitrary-uid worker container commonly has HOME pointing at an existing directory
    # (e.g. /root) that home.is_dir() confirms but this process cannot write under; selecting
    # it anyway means _ensure_cache_root dies with PermissionError instead of reaching the
    # uid-scoped temp fallback that exists precisely to keep those containers working.
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(adapter.os.path, "expanduser", lambda _p: str(home))
    monkeypatch.setattr(adapter.os, "access", lambda _path, _mode: False)
    monkeypatch.setattr(adapter.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    root = adapter._default_cache_root()

    uid = os.getuid() if hasattr(os, "getuid") else 0
    assert root == tmp_path / "tmp" / f"flash-env-cache-{uid}"


def test_default_cache_root_falls_back_when_cache_parent_denies_search(monkeypatch, tmp_path):
    # write access alone cannot create a child: mkdir also needs search (x) on the parent, as
    # with an acl that grants write but denies traversal. a W_OK-only probe accepts the home
    # root anyway and _ensure_cache_root then dies with PermissionError instead of reaching
    # the temp fallback.
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    home = tmp_path / "home"
    (home / ".cache").mkdir(parents=True)
    monkeypatch.setattr(adapter.os.path, "expanduser", lambda _p: str(home))
    monkeypatch.setattr(adapter.os, "access", lambda _path, mode: not (mode & os.X_OK))
    monkeypatch.setattr(adapter.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    root = adapter._default_cache_root()

    uid = os.getuid() if hasattr(os, "getuid") else 0
    assert root == tmp_path / "tmp" / f"flash-env-cache-{uid}"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only uid check")
def test_default_cache_root_falls_back_when_home_foreign_owned(monkeypatch, tmp_path):
    # same fallback, triggered by ownership rather than the write-bit: a home directory this
    # process happens to have write access to but does not own is just as untrustworthy.
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(adapter.os.path, "expanduser", lambda _p: str(home))
    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if Path(path) == home:
            result = os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    result.st_uid + 1,
                    result.st_gid,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )
        return result

    monkeypatch.setattr(adapter.os, "stat", fake_stat)
    monkeypatch.setattr(adapter.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    root = adapter._default_cache_root()

    assert root == tmp_path / "tmp" / f"flash-env-cache-{os.getuid()}"


def _report_foreign_owner(monkeypatch, attr, targets):
    # ownership cannot be faked for real in a test (chown needs root), so make the stat call
    # the check uses report someone else's uid for exactly these paths.
    real = getattr(os, attr)
    targets = {Path(t) for t in targets}

    def fake(path, *args, **kwargs):
        result = real(path, *args, **kwargs)
        if Path(path) in targets:
            fields = list(result)
            fields[4] = result.st_uid + 1
            return os.stat_result(tuple(fields))
        return result

    monkeypatch.setattr(adapter.os, attr, fake)


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only uid check")
def test_default_cache_root_falls_back_when_existing_dot_cache_is_foreign(monkeypatch, tmp_path):
    # HOME itself is writable and ours, but ~/.cache already exists owned by root at mode
    # 0755 (an image artifact, or an earlier privileged run). a usability check that looks
    # only at `home` selects the home-based root anyway, and _ensure_cache_root then dies
    # with PermissionError creating ~/.cache/flash -- never reaching the temp fallback that
    # exists for exactly this. the deepest EXISTING directory on the path is the one mkdir
    # has to write into, so that is the one that has to be judged.
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    home = tmp_path / "home"
    (home / ".cache").mkdir(parents=True)
    monkeypatch.setattr(adapter.os.path, "expanduser", lambda _p: str(home))
    _report_foreign_owner(monkeypatch, "stat", [home / ".cache"])
    monkeypatch.setattr(adapter.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    root = adapter._default_cache_root()

    assert root == tmp_path / "tmp" / f"flash-env-cache-{os.getuid()}"


def test_default_cache_root_uses_home_when_existing_dot_cache_is_ours(monkeypatch, tmp_path):
    # the other side of the same walk: an existing ~/.cache we own must not push the cache
    # into temp. the deepest-existing check only rejects, never relocates a usable home.
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    home = tmp_path / "home"
    (home / ".cache" / "flash").mkdir(parents=True)
    monkeypatch.setattr(adapter.os.path, "expanduser", lambda _p: str(home))
    monkeypatch.setattr(adapter.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    root = adapter._default_cache_root()

    assert root == home / ".cache" / "flash" / "env-cache"


def test_ensure_cache_root_creates_private_dir(monkeypatch, tmp_path):
    root = tmp_path / "cache" / "env-cache"
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)

    assert adapter._ensure_cache_root() == root
    # derived the same way production is gated: mkdir(mode=0o700) does not establish posix mode
    # bits on windows, where st_mode is synthetic, so only the posix run can assert 0700.
    if hasattr(os, "getuid"):
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
    # idempotent on a root that already exists.
    assert adapter._ensure_cache_root() == root


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only mode bits")
def test_ensure_cache_root_creates_private_ancestors_under_group_writable_umask(
    monkeypatch, tmp_path
):
    # mkdir(parents=True, mode=0o700) sets the mode on the LEAF only, so under umask 0002 the
    # ancestors it creates land at 0775 and the ancestor walk refuses the root that same call
    # just created. every github env resolve would fail on first use wherever private user
    # groups are the default.
    root = tmp_path / "outer" / "flash" / "env-cache"
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)
    previous_umask = os.umask(0o002)
    try:
        assert adapter._ensure_cache_root() == root
    finally:
        os.umask(previous_umask)

    for created in (root, root.parent, root.parent.parent):
        assert stat.S_IMODE(created.stat().st_mode) == 0o700, created


def test_discard_untrusted_entry_accepts_a_concurrently_removed_entry(monkeypatch, tmp_path):
    # a concurrent resolve of the same untrusted key can clear the entry between our trust check
    # and our delete, so rmtree raises FileNotFoundError for the outcome we actually wanted.
    # success is "the entry is gone", not "the delete call did not raise" -- judging by the
    # exception fails both resolves on a cache key that is now perfectly safe to refetch.
    entry = tmp_path / "entry"
    entry.mkdir()
    real_rmtree = shutil.rmtree

    def racing_rmtree(path, *args, **kwargs):
        real_rmtree(path, *args, **kwargs)
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(cache_security.shutil, "rmtree", racing_rmtree)

    cache_security.discard_untrusted_entry(entry)

    assert not entry.exists()


def test_discard_untrusted_entry_removes_a_plain_file(tmp_path):
    # a regular file squatting at a cache-key path is neither a symlink nor a directory, and
    # handing it to rmtree raises NotADirectoryError -- condemning a key that a simple unlink
    # inside our own root clears fine, and blocking the fresh download that should follow.
    entry = tmp_path / "entry"
    entry.write_text("not a directory")

    cache_security.discard_untrusted_entry(entry)

    assert not entry.exists()


def test_ensure_cache_root_refuses_foreign_owner(monkeypatch, tmp_path):
    root = tmp_path / "cache"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)
    monkeypatch.setattr(adapter.os, "getuid", lambda: os.stat(root).st_uid + 1)

    with pytest.raises(RuntimeError, match="owned by uid"):
        adapter._ensure_cache_root()


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only mode bits")
def test_ensure_cache_root_refuses_group_or_other_writable(monkeypatch, tmp_path):
    # a mode check, not an access check, so this holds for root too.
    root = tmp_path / "cache"
    root.mkdir()
    root.chmod(0o777)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)

    with pytest.raises(RuntimeError, match="accessible to group/other"):
        adapter._ensure_cache_root()


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only mode bits")
def test_ensure_cache_root_refuses_group_traversable_root(monkeypatch, tmp_path):
    # write bits are not the only hazard: a 0710 (or 0755) root lets same-group accounts
    # traverse into cached entries, whose contents can legitimately carry group-writable
    # modes (umask-dependent mkdir parents on the contents-api path, ancient 100664 tree
    # modes, both preserved by copytree). in-place tampering there keeps the victim's uid,
    # so the entry ownership vetting still trusts the modified code.
    root = tmp_path / "cache"
    root.mkdir(mode=0o710)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)

    with pytest.raises(RuntimeError, match="accessible to group/other"):
        adapter._ensure_cache_root()


def test_ensure_cache_root_refuses_symlinked_root(monkeypatch, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    root = tmp_path / "cache"
    root.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)

    with pytest.raises(RuntimeError, match="not a directory"):
        adapter._ensure_cache_root()


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only uid check")
def test_ensure_cache_root_refuses_foreign_owned_ancestor(monkeypatch, tmp_path):
    # the leaf-only check misses a shared parent (e.g. a shared XDG_CACHE_HOME): another local
    # account can own an intermediate directory, let us create a private leaf under it, then
    # swap the leaf for a symlink or attacker tree after the leaf check passes.
    outer = tmp_path / "outer"
    outer.mkdir()
    root = outer / "flash" / "env-cache"
    _make_private_cache_root(root)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)
    resolved_outer = outer.resolve()
    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        result = real_lstat(path, *args, **kwargs)
        if Path(path) == resolved_outer:
            result = os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    result.st_uid + 1,
                    result.st_gid,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )
        return result

    monkeypatch.setattr(adapter.os, "lstat", fake_lstat)

    with pytest.raises(RuntimeError) as excinfo:
        adapter._ensure_cache_root()
    assert "ancestor" in str(excinfo.value)
    assert "owned by uid" in str(excinfo.value)


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only uid check")
def test_ensure_cache_root_refuses_foreign_owned_symlink_ancestor(monkeypatch, tmp_path):
    # resolve() erases a symlink component from the ancestor chain, so a validator walking
    # only the resolved parents never examines the symlink itself. its owner can retarget it
    # after validation while every later cache operation still traverses the unresolved path,
    # so a foreign-owned symlink component has to be refused outright.
    target = tmp_path / "real"
    target.mkdir(mode=0o700)
    outer = tmp_path / "outer"
    outer.mkdir(mode=0o700)
    link = outer / "link"
    link.symlink_to(target, target_is_directory=True)
    root = link / "env-cache"
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)
    _report_foreign_owner(monkeypatch, "lstat", [link])

    with pytest.raises(RuntimeError) as excinfo:
        adapter._ensure_cache_root()
    assert "symlink" in str(excinfo.value)
    assert "owned by uid" in str(excinfo.value)


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only mode bits")
def test_ensure_cache_root_refuses_world_writable_ancestor_without_sticky_bit(
    monkeypatch, tmp_path
):
    # an ordinary world-writable ancestor (unlike /tmp, which is also sticky) lets another
    # local account replace anything beneath it, including a leaf we already validated.
    outer = tmp_path / "outer"
    outer.mkdir()
    outer.chmod(0o777)
    root = outer / "flash" / "env-cache"
    root.mkdir(parents=True, mode=0o700)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)

    with pytest.raises(RuntimeError) as excinfo:
        adapter._ensure_cache_root()
    assert "ancestor" in str(excinfo.value)
    assert "sticky bit" in str(excinfo.value)


def test_ensure_cache_root_allows_world_writable_ancestor_with_sticky_bit(monkeypatch, tmp_path):
    # the sticky bit is what makes a shared temp root (e.g. /tmp) acceptable: it stops other
    # accounts from renaming or replacing entries they do not own, even though they can create
    # their own entries in the same directory.
    outer = tmp_path / "outer"
    outer.mkdir()
    outer.chmod(0o777 | stat.S_ISVTX)
    root = outer / "flash" / "env-cache"
    _make_private_cache_root(root)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)

    assert adapter._ensure_cache_root() == root


def test_ensure_cache_root_skips_posix_checks_when_getuid_missing(monkeypatch, tmp_path):
    # windows: os.getuid does not exist, mkdir(mode=0o700) does not establish posix mode bits,
    # and os.stat commonly reports directories as group/other-writable, so both the ancestor
    # walk and the leaf mode-bit check must no-op rather than reject an otherwise-usable root.
    outer = tmp_path / "outer"
    outer.mkdir()
    outer.chmod(0o777)  # would fail the ancestor check if it ran
    root = outer / "flash" / "env-cache"
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o777)  # would fail the leaf mode-bit check if it ran
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)
    monkeypatch.delattr(adapter.os, "getuid", raising=False)

    assert adapter._ensure_cache_root() == root


def _make_entry(root: Path, name: str, *, size: int, age_seconds: float) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "environment.py").write_bytes(b"x" * size)
    mtime = time.time() - age_seconds
    os.utime(d, (mtime, mtime))
    return d


def test_evict_over_entry_cap_removes_oldest(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(adapter, "_CACHE_MAX_ENTRIES", 2)
    monkeypatch.setattr(adapter, "_CACHE_MAX_BYTES", 10**12)
    monkeypatch.setattr(adapter, "_CACHE_MIN_AGE_SECONDS", 0)

    old = _make_entry(tmp_path, "old", size=10, age_seconds=10_000)
    mid = _make_entry(tmp_path, "mid", size=10, age_seconds=5_000)
    keep = _make_entry(tmp_path, "keep", size=10, age_seconds=1)

    adapter._evict_env_cache(keep=keep)

    # over the 2-entry cap: oldest ("old") evicted, newer two retained.
    assert not old.exists()
    assert mid.exists()
    assert keep.exists()


def test_evict_over_byte_cap_removes_oldest(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(adapter, "_CACHE_MAX_ENTRIES", 100)
    monkeypatch.setattr(adapter, "_CACHE_MAX_BYTES", 25)
    monkeypatch.setattr(adapter, "_CACHE_MIN_AGE_SECONDS", 0)

    old = _make_entry(tmp_path, "old", size=10, age_seconds=10_000)
    mid = _make_entry(tmp_path, "mid", size=10, age_seconds=5_000)
    keep = _make_entry(tmp_path, "keep", size=10, age_seconds=1)

    adapter._evict_env_cache(keep=keep)

    # total 30 bytes > 25 cap: evict oldest until under. "old" removed -> 20 <= 25.
    assert not old.exists()
    assert mid.exists()
    assert keep.exists()


def test_evict_never_removes_keep_or_recent(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(adapter, "_CACHE_MAX_ENTRIES", 1)
    monkeypatch.setattr(adapter, "_CACHE_MAX_BYTES", 10**12)
    monkeypatch.setattr(adapter, "_CACHE_MIN_AGE_SECONDS", 600)

    recent = _make_entry(tmp_path, "recent", size=10, age_seconds=1)
    keep = _make_entry(tmp_path, "keep", size=10, age_seconds=1)

    adapter._evict_env_cache(keep=keep)

    # over the 1-entry cap, but both are within the min-age window -> nothing evicted.
    assert recent.exists()
    assert keep.exists()
