"""Server-side managed Freesolo env download from GitHub."""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import pytest

from flash.server import envs


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_download_package_user_can_download_own_namespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    seen: dict[str, object] = {}

    def fake_download(slug, *, token):
        seen.update(slug=slug, token=token)
        return b"package"

    monkeypatch.setattr(envs, "_github_download", fake_download)

    assert envs.download_package(slug="acme/my-env", key={"org_slug": "acme"}) == b"package"
    assert seen == {"slug": "acme/my-env", "token": "ghp-test"}


def test_download_package_rejects_other_users_namespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setattr(
        envs, "_github_download", lambda *a, **k: pytest.fail("storage must not be touched")
    )

    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.download_package(slug="someone-else/env", key={"org_slug": "acme"})

    assert excinfo.value.status == 403


def test_download_package_internal_key_can_download_any_namespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        envs,
        "_github_download",
        lambda slug, *, token: seen.update(slug=slug, token=token) or b"package",
    )

    assert envs.download_package(slug="acme/paper-foo", key={"auth_kind": "internal"}) == b"package"
    assert seen == {"slug": "acme/paper-foo", "token": "ghp-test"}


def test_download_package_requires_control_plane_github_credential(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.download_package(slug="acme/x", key={"auth_kind": "internal"})

    assert excinfo.value.status == 503
    assert "control plane" in str(excinfo.value)


def test_github_download_once_packages_slug_directory(tmp_path, monkeypatch):
    remote = tmp_path / "hub.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "--initial-branch", "main")
    _git(seed, "config", "user.name", "test")
    _git(seed, "config", "user.email", "test@example.com")
    env_dir = seed / "ns" / "env"
    env_dir.mkdir(parents=True)
    (env_dir / "environment.py").write_text("def load_environment(**k): pass\n")
    (env_dir / "datasets").mkdir()
    (env_dir / "datasets" / "train.jsonl").write_text('{"a":1}\n')
    other = seed / "other" / "env"
    other.mkdir(parents=True)
    (other / "environment.py").write_text("# other\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "init", "--bare", str(remote))
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "origin", "main")

    monkeypatch.setattr(envs, "_credentialed_repo_url", lambda repo, token: str(remote))

    package = envs._github_download_once(repo="ignored/repo", token="tok", publish_root="ns/env")

    with tarfile.open(fileobj=io.BytesIO(package), mode="r:gz") as tar:
        names = sorted(member.name for member in tar.getmembers() if member.isfile())
        env_file = tar.extractfile("environment.py")
        dataset_file = tar.extractfile("datasets/train.jsonl")
        assert env_file is not None
        assert dataset_file is not None
        assert env_file.read() == b"def load_environment(**k): pass\n"
        assert dataset_file.read() == b'{"a":1}\n'
    assert names == ["datasets/train.jsonl", "environment.py"]
