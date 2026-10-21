"""Only complete environment packages may become reusable cache entries."""

from pathlib import Path

import pytest

from flash.envs.loading import loader
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
