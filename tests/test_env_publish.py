"""Server-side managed Freesolo env publishing to GitHub."""

from __future__ import annotations

import base64
import io
import tarfile

import pytest

from flash.server import envs

_MINIMAL = {
    "pyproject.toml": "[project]\nname = 'e'\n",
    "freesolo/__init__.py": "",
    "freesolo/environment.py": "def load_environment(**k):\n    return None\n",
}


def _pkg_b64(files: dict[str, str]) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def test_namespace_is_stable_and_repo_safe():
    assert envs.namespace_for({"email": "Dev@Clado.ai"}) == "dev-clado-ai"
    assert envs.namespace_for({"email": "dev@clado.ai"}) == "dev-clado-ai"
    assert envs.namespace_for({"id": 7}) == "key-7"
    assert envs.namespace_for({"key_prefix": "fslo-abc"}) == "fslo-abc"
    assert envs.namespace_for({}) == "user"


def test_namespace_distinct_for_placeholder_emails():
    k1 = {"id": 11, "email": "freesolo-user", "key_prefix": "freesolo"}
    k2 = {"id": 12, "email": "freesolo-user", "key_prefix": "freesolo"}
    ns1, ns2 = envs.namespace_for(k1), envs.namespace_for(k2)
    assert ns1 != ns2
    assert "freesolo-user" not in (ns1, ns2)
    assert envs.namespace_for(dict(k1)) == ns1


def test_publish_uploads_to_github_and_returns_ref(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("FLASH_ENV_GITHUB_REPO", "freesolo-co/environment-hub")
    monkeypatch.setenv("FLASH_ENV_GITHUB_BRANCH", "dev")
    uploaded: list[tuple[str, bytes]] = []

    def fake_put_github_file(*, path, data, **_kwargs):
        uploaded.append((path, data))

    monkeypatch.setattr(envs, "_put_github_file", fake_put_github_file)
    ref = envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="My Env!",
        is_new=True,
        key={"email": "dev@clado.ai"},
    )

    root = "environments/dev-clado-ai/my-env"
    assert ref == (f"github:freesolo-co/environment-hub@dev:{root}/freesolo/environment.py")
    assert (
        f"{root}/freesolo/environment.py",
        _MINIMAL["freesolo/environment.py"].encode(),
    ) in uploaded
    assert (f"{root}/pyproject.toml", _MINIMAL["pyproject.toml"].encode()) in uploaded


def test_publish_rejects_bad_input(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setattr(envs, "_put_github_file", lambda **_kwargs: None)
    with pytest.raises(envs.EnvPublishError, match="base64"):
        envs.publish_package(package_b64="not base64!!!", name="e", is_new=True, key={})
    with pytest.raises(envs.EnvPublishError, match="empty"):
        envs.publish_package(package_b64="", name="e", is_new=True, key={})
    with pytest.raises(envs.EnvPublishError, match="name"):
        envs.publish_package(package_b64=_pkg_b64(_MINIMAL), name="", is_new=True, key={})
    with pytest.raises(envs.EnvPublishError, match=r"environment\.py"):
        envs.publish_package(
            package_b64=_pkg_b64({"pyproject.toml": "[project]\nname='e'\n"}),
            name="e",
            is_new=True,
            key={},
        )


def test_publish_requires_github_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.publish_package(package_b64=_pkg_b64(_MINIMAL), name="e", is_new=True, key={})
    assert excinfo.value.status == 503
    assert "GITHUB_TOKEN" in str(excinfo.value)


def test_safe_extract_rejects_traversal(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"pwn"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(envs.EnvPublishError, match="unsafe path"):
        envs._safe_extract(buf.getvalue(), tmp_path)
    assert not (tmp_path.parent / "escape.txt").exists()


def test_safe_extract_extracts_regular_files_and_dirs(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        d = tarfile.TarInfo("pkg")
        d.type = tarfile.DIRTYPE
        d.mode = 0o755
        tar.addfile(d)
        data = b"[project]\nname='e'\n"
        f = tarfile.TarInfo("pkg/pyproject.toml")
        f.size = len(data)
        f.mode = 0o644
        tar.addfile(f, io.BytesIO(data))
    envs._safe_extract(buf.getvalue(), tmp_path)
    assert (tmp_path / "pkg").is_dir()
    assert (tmp_path / "pkg" / "pyproject.toml").read_text() == "[project]\nname='e'\n"


def test_safe_extract_rejects_special_members(tmp_path):
    for typeflag, label in ((tarfile.FIFOTYPE, "fifo"), (tarfile.CHRTYPE, "char-device")):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(f"evil-{label}")
            info.type = typeflag
            if typeflag == tarfile.CHRTYPE:
                info.devmajor, info.devminor = 1, 3
            tar.addfile(info)
        with pytest.raises(envs.EnvPublishError, match="special file"):
            envs._safe_extract(buf.getvalue(), tmp_path)
        assert not (tmp_path / f"evil-{label}").exists()


def test_safe_extract_rejects_workflow_control_paths(tmp_path):
    for label in (".github", ".git", "./.github", "./.git"):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            d = tarfile.TarInfo(f"{label}/workflows")
            d.type = tarfile.DIRTYPE
            tar.addfile(d)
        with pytest.raises(envs.EnvPublishError, match="top-level paths"):
            envs._safe_extract(buf.getvalue(), tmp_path)


def test_put_github_file_rejects_oversized_blob():
    with pytest.raises(envs.EnvPublishError, match="exceeds GitHub Contents API limit"):
        envs._put_github_file(
            repo="owner/repo",
            branch="main",
            path="environments/sample.txt",
            data=b"x" * (envs._MAX_GITHUB_FILE_BYTES + 1),
            token="test",
            message="upload",
        )


def test_existing_file_sha_returns_none_for_404(monkeypatch):
    def fake_github_json(method: str, url: str, *, token: str, body: dict | None = None):
        raise envs._GitHubApiError("missing", status=404)

    monkeypatch.setattr(envs, "_github_json", fake_github_json)
    assert envs._existing_file_sha("owner/repo", "main", "f.txt", "token") is None


def test_existing_file_sha_propagates_non_404(monkeypatch):
    def fake_github_json(method: str, url: str, *, token: str, body: dict | None = None):
        raise envs._GitHubApiError("forbidden", status=403)

    monkeypatch.setattr(envs, "_github_json", fake_github_json)
    with pytest.raises(envs.EnvPublishError, match="forbidden"):
        envs._existing_file_sha("owner/repo", "main", "f.txt", "token")
