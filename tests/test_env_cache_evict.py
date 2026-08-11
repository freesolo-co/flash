"""unit tests for the bounded, user-private on-disk env cache."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from flash.envs import loader as adapter


def test_cache_config_ignores_ambient_overrides(tmp_path):
    # the on-disk env cache location and bounds are hardcoded, not env-tunable: ambient
    # FLASH_ENV_CACHE_* vars (a stray shell or CI export) must never change them. the root
    # is user-scoped off XDG_CACHE_HOME/HOME so no other local account can pre-seed it.
    expected = tmp_path / "xdg" / "flash" / "env-cache"
    script = (
        "import os; "
        "from pathlib import Path; "
        "from flash.envs import loader; "
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

    assert root == tmp_path / f"flash-env-cache-{os.getuid()}"


def test_ensure_cache_root_creates_private_dir(monkeypatch, tmp_path):
    root = tmp_path / "cache" / "env-cache"
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)

    assert adapter._ensure_cache_root() == root
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    # idempotent on a root that already exists.
    assert adapter._ensure_cache_root() == root


def test_ensure_cache_root_refuses_foreign_owner(monkeypatch, tmp_path):
    root = tmp_path / "cache"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)
    monkeypatch.setattr(adapter.os, "getuid", lambda: os.stat(root).st_uid + 1)

    with pytest.raises(RuntimeError, match="owned by uid"):
        adapter._ensure_cache_root()


def test_ensure_cache_root_refuses_group_or_other_writable(monkeypatch, tmp_path):
    # a mode check, not an access check, so this holds for root too.
    root = tmp_path / "cache"
    root.mkdir()
    root.chmod(0o777)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)

    with pytest.raises(RuntimeError, match="group/other-writable"):
        adapter._ensure_cache_root()


def test_ensure_cache_root_refuses_symlinked_root(monkeypatch, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    root = tmp_path / "cache"
    root.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", root)

    with pytest.raises(RuntimeError, match="not a directory"):
        adapter._ensure_cache_root()


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
