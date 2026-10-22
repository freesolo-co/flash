"""Only complete environment packages may become reusable cache entries."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from flash.envs.loading import loader
from flash.envs.meta import cache_storage
from tests.test_env_pull import _github_env_tarball


@pytest.fixture
def environment_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(loader, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)
    downloads = []

    def download(ref):
        downloads.append(ref)
        return _github_env_tarball(b"# complete environment\n")

    monkeypatch.setattr(loader, "_download_github_tarball", download)
    return downloads


def test_failed_copy_cannot_become_a_cache_hit(environment_cache, monkeypatch):
    real_copytree = loader.shutil.copytree

    def interrupted_copy(source, destination):
        destination = Path(destination)
        destination.mkdir()
        (destination / "environment.py").write_text("# incomplete package\n")
        raise OSError("copy interrupted")

    monkeypatch.setattr(loader.shutil, "copytree", interrupted_copy)
    ref = "github:owner/repo@main:environment.py"
    with pytest.raises(OSError, match="copy interrupted"):
        loader._resolve_github_environment_file(ref)
    monkeypatch.setattr(loader.shutil, "copytree", real_copytree)
    resolved = loader._resolve_github_environment_file(ref)
    assert resolved.read_text() == "# complete environment\n"
    assert len(environment_cache) == 2


def test_publisher_cannot_be_deleted_after_a_stale_miss(environment_cache, monkeypatch, tmp_path):
    source = tmp_path / "winner"
    source.mkdir()
    (source / "environment.py").write_text("# winner\n")
    checked = threading.Event()
    published = threading.Event()
    target = []
    errors = []
    real_is_file = Path.is_file
    original_download = loader._download_github_tarball

    def publish():
        try:
            assert checked.wait(timeout=5)
            cache_storage.publish_cache_entry(source, target[0].parent, "environment.py")
        except BaseException as exc:
            errors.append(exc)
        finally:
            published.set()

    def stale_miss(path):
        result = real_is_file(path)
        if (
            path.name == "environment.py"
            and path.parent.parent == loader._CACHE_ROOT
            and not target
        ):
            target.append(path)
            checked.set()
            # Without the metadata lock the other publisher can finish inside this miss check.
            # With the lock it finishes after the resolver releases the check/removal section.
            published.wait(timeout=1)
        return result

    def download(ref):
        assert published.wait(timeout=5)
        assert not errors
        assert target[0].read_text() == "# winner\n"
        return original_download(ref)

    monkeypatch.setattr(Path, "is_file", stale_miss)
    monkeypatch.setattr(loader, "_download_github_tarball", download)
    worker = threading.Thread(target=publish)
    worker.start()
    try:
        resolved = loader._resolve_github_environment_file("github:owner/repo@main:environment.py")
        assert resolved.read_text() == "# winner\n"
    finally:
        checked.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert not errors


def test_eviction_reaps_old_stage_left_by_hard_exit(environment_cache, tmp_path):
    root = loader._ensure_cache_root()
    source = tmp_path / "source"
    source.mkdir()
    (source / "environment.py").write_text("# source\n")
    script = """
import os, sys
from pathlib import Path
from flash.envs.meta import cache_storage
def crash(source, destination):
    destination.mkdir()
    (destination / 'partial').write_bytes(b'partial copy')
    os._exit(17)
cache_storage.shutil.copytree = crash
cache_storage.publish_cache_entry(Path(sys.argv[1]), Path(sys.argv[2]), 'environment.py')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(root / "entry")], check=False
    )
    assert result.returncode == 17
    stages = list(root.glob(f"{cache_storage.STAGING_PREFIX}*"))
    assert len(stages) == 1
    old = time.time() - 7 * 24 * 3600
    os.utime(stages[0], (old, old))
    loader._evict_env_cache(keep=root / "entry")
    assert not stages[0].exists()


def test_eviction_preserves_active_stage_even_when_old(environment_cache, monkeypatch, tmp_path):
    root = loader._ensure_cache_root()
    source = tmp_path / "source"
    source.mkdir()
    (source / "environment.py").write_text("# active\n")
    real_copy = loader.shutil.copytree

    def copy_and_evict(source, destination):
        old = time.time() - 7 * 24 * 3600
        os.utime(destination.parent, (old, old))
        loader._evict_env_cache(keep=root / "entry")
        assert destination.parent.exists()
        return real_copy(source, destination)

    monkeypatch.setattr(loader.shutil, "copytree", copy_and_evict)
    cache_storage.publish_cache_entry(source, root / "entry", "environment.py")
    assert (root / "entry" / "environment.py").read_text() == "# active\n"


def test_eviction_preserves_recent_abandoned_stage(environment_cache):
    root = loader._ensure_cache_root()
    stage = root / f"{cache_storage.STAGING_PREFIX}recent"
    stage.mkdir()
    loader._evict_env_cache(keep=root / "entry")
    assert stage.exists()


def test_cache_hit_also_reaps_old_abandoned_stages(environment_cache):
    ref = "github:owner/repo@main:environment.py"
    resolved = loader._resolve_github_environment_file(ref)
    stage = loader._CACHE_ROOT / f"{cache_storage.STAGING_PREFIX}abandoned"
    stage.mkdir()
    old = time.time() - 7 * 24 * 3600
    os.utime(stage, (old, old))
    assert loader._resolve_github_environment_file(ref) == resolved
    assert len(environment_cache) == 1
    assert not stage.exists()


def test_new_publication_is_recent_even_with_old_archive_timestamps(
    environment_cache, monkeypatch, tmp_path
):
    root = loader._ensure_cache_root()
    source = tmp_path / "old-source"
    source.mkdir()
    (source / "environment.py").write_text("# source\n")
    old = time.time() - 7 * 24 * 3600
    os.utime(source, (old, old))
    cache_storage.publish_cache_entry(source, root / "entry", "environment.py")
    monkeypatch.setattr(loader, "_CACHE_MAX_ENTRIES", 0)
    loader._evict_env_cache(keep=root / "another-entry")
    assert (root / "entry" / "environment.py").is_file()


def test_concurrent_publisher_keeps_the_completed_winner(environment_cache, monkeypatch):
    real_copytree = loader.shutil.copytree
    winner = []
    ref = "github:owner/repo@main:environment.py"

    def racing_copy(source, destination):
        monkeypatch.setattr(loader.shutil, "copytree", real_copytree)
        result = real_copytree(source, destination)
        published = loader._resolve_github_environment_file(ref)
        winner.append((published, published.stat().st_ino))
        return result

    monkeypatch.setattr(loader.shutil, "copytree", racing_copy)
    resolved = loader._resolve_github_environment_file(ref)
    assert len(environment_cache) == 2
    assert resolved == winner[0][0]
    assert resolved.stat().st_ino == winner[0][1]
    assert resolved.read_text() == "# complete environment\n"
