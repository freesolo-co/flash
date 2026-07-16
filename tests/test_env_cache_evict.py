"""Unit tests for the bounded on-disk env cache eviction."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from flash.envs import loader as adapter


def test_cache_config_ignores_ambient_overrides():
    # the on-disk env cache location and bounds are hardcoded, not env-tunable: ambient
    # FLASH_ENV_CACHE_* vars (a stray shell or CI export) must never change them.
    script = (
        "from pathlib import Path; "
        "from flash.envs import loader; "
        "assert loader._CACHE_ROOT == Path('/tmp/flash-env-cache'); "
        "assert loader._CACHE_MAX_ENTRIES == 32; "
        "assert loader._CACHE_MAX_BYTES == 4 * 1024 * 1024 * 1024"
    )
    for cache_dir, max_entries, max_bytes in (
        ("/some/other/dir", "1", "2"),
        ("also-bogus", "invalid", "also-invalid"),
    ):
        env = os.environ.copy()
        env["FLASH_ENV_CACHE_DIR"] = cache_dir
        env["FLASH_ENV_CACHE_MAX_ENTRIES"] = max_entries
        env["FLASH_ENV_CACHE_MAX_BYTES"] = max_bytes
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            check=True,
        )


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
