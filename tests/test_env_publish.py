"""Server-side managed Freesolo env publishing to GitHub."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from flash.server import envs

_MINIMAL = {
    "pyproject.toml": "[project]\nname = 'e'\n",
    "environment.py": "def load_environment(**k):\n    return None\n",
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


def test_namespace_requires_email():
    for key in ({"id": 7}, {"key_prefix": "fslo-abc"}, {}, {"email": "missing-email"}):
        with pytest.raises(envs.EnvPublishError, match="email"):
            envs.namespace_for(key)


def test_internal_key_gets_reserved_namespace_without_email():
    # The internal/operator service key has a synthetic/shared identity (no per-user email) but is
    # trusted to submit runs and read everything, so it must be able to publish too. It gets a
    # fixed reserved namespace instead of raising — regardless of whether its row carries an email.
    assert envs.namespace_for({"auth_kind": "internal"}) == envs._INTERNAL_NAMESPACE
    assert envs.namespace_for({"auth_kind": "internal", "email": ""}) == envs._INTERNAL_NAMESPACE
    assert (
        envs.namespace_for({"auth_kind": "internal", "email": "missing-email"})
        == envs._INTERNAL_NAMESPACE
    )


def test_internal_special_case_does_not_loosen_user_keys():
    # The bypass is reserved for auth_kind == "internal" ONLY: a user key (any other auth_kind, or
    # none) without a valid email must still be rejected so two users can't collide on a namespace.
    for key in ({"auth_kind": "external"}, {"auth_kind": "user", "email": "no-at"}, {"email": ""}):
        with pytest.raises(envs.EnvPublishError, match="email"):
            envs.namespace_for(key)


def test_sanitize_name_never_returns_path_segments():
    assert envs._sanitize_name("..") == "env"
    assert envs._sanitize_name(".") == "env"
    assert envs._sanitize_name("___") == "env"
    assert envs._sanitize_name("My Env!") == "my-env"


def test_publish_uploads_to_github_and_returns_slug(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    captured: dict[str, object] = {}

    def fake_publish_once(*, dest, repo, token, publish_root, message):
        captured.update(
            repo=repo,
            token=token,
            publish_root=publish_root,
            message=message,
            files=sorted(
                path.relative_to(dest).as_posix() for path in dest.rglob("*") if path.is_file()
            ),
        )

    monkeypatch.setattr(envs, "_github_publish_once", fake_publish_once)
    ref = envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="My Env!",
        key={"email": "dev@clado.ai"},
    )

    root = "dev-clado-ai/my-env"
    assert ref == root
    assert captured["repo"] == "freesolo-co/environment-hub"
    assert captured["token"] == "ghp-test"
    assert captured["publish_root"] == root
    assert captured["message"] == "Upload Flash environment dev-clado-ai/my-env"
    assert captured["files"] == ["environment.py", "pyproject.toml"]


def test_publish_rejects_bad_input(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setattr(envs, "_github_publish_once", lambda **_kwargs: None)
    with pytest.raises(envs.EnvPublishError, match="base64"):
        envs.publish_package(package_b64="not base64!!!", name="e", key={})
    with pytest.raises(envs.EnvPublishError, match="empty"):
        envs.publish_package(package_b64="", name="e", key={})
    with pytest.raises(envs.EnvPublishError, match="name"):
        envs.publish_package(package_b64=_pkg_b64(_MINIMAL), name="", key={})
    with pytest.raises(envs.EnvPublishError, match=r"environment\.py"):
        envs.publish_package(
            package_b64=_pkg_b64({"pyproject.toml": "[project]\nname='e'\n"}),
            name="e",
            key={"email": "dev@clado.ai"},
        )


def test_publish_requires_github_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.publish_package(package_b64=_pkg_b64(_MINIMAL), name="e", key={})
    assert excinfo.value.status == 503
    assert "GITHUB_TOKEN" in str(excinfo.value)


def test_publish_does_not_accept_github_pat_alias(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_PAT", "github_pat_test")
    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.publish_package(package_b64=_pkg_b64(_MINIMAL), name="e", key={})
    assert excinfo.value.status == 503
    assert "GITHUB_TOKEN" in str(excinfo.value)


def test_record_published_environment_posts_to_backend(monkeypatch):
    from flash.server import environment_registry

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://backend.test")
    seen: dict[str, object] = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = req.data
        return _Resp()

    monkeypatch.setattr(environment_registry.urllib.request, "urlopen", fake_urlopen)

    ok = environment_registry.record_published_environment(
        slug="dev-clado-ai/my-env",
        name="My Env",
        key={"org_id": "org-1", "user_id": "user-1", "api_key_id": "key-1"},
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/environments/internal"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    body = json.loads(seen["body"])
    assert body == {
        "orgId": "org-1",
        "slug": "dev-clado-ai/my-env",
        "name": "My Env",
        "hubRepo": "freesolo-co/environment-hub",
        "hubRef": "main",
        "hubPath": "dev-clado-ai/my-env/environment.py",
        "publishedByUserId": "user-1",
        "apiKeyId": "key-1",
        "metadata": {"source": "flash.env.push"},
    }


def test_record_published_environment_is_best_effort(monkeypatch):
    from flash.server import environment_registry

    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    assert (
        environment_registry.record_published_environment(
            slug="dev-clado-ai/my-env",
            name="My Env",
            key={"org_id": "org-1"},
        )
        is False
    )


def test_record_environment_use_posts_to_backend(monkeypatch):
    from flash.server import environment_registry

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://backend.test")
    seen: dict[str, object] = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = req.data
        return _Resp()

    monkeypatch.setattr(environment_registry.urllib.request, "urlopen", fake_urlopen)

    ok = environment_registry.record_environment_use(
        slug="dev-clado-ai/my-env",
        run_id="flash-1",
        key={"org_id": "org-1"},
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/environments/use/internal"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    assert json.loads(seen["body"]) == {
        "orgId": "org-1",
        "slug": "dev-clado-ai/my-env",
        "runId": "flash-1",
    }


def test_record_training_run_posts_to_backend(monkeypatch):
    from flash.runner import RunStatus
    from flash.server import run_registry

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://backend.test")
    seen: dict[str, object] = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = req.data
        return _Resp()

    monkeypatch.setattr(run_registry.urllib.request, "urlopen", fake_urlopen)

    ok = run_registry.record_training_run(
        status=RunStatus(
            run_id="flash-1",
            state="running",
            spec={
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": "grpo",
                "phase": "rl",
                "environment": {"id": "dev-clado-ai/my-env"},
                "gpu": {"type": "RTX 5090"},
            },
            platform_context={
                "org_id": "org-1",
                "user_id": "user-1",
                "api_key_id": "key-1",
            },
        )
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/runs/internal"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    body = json.loads(seen["body"])
    assert body["orgId"] == "org-1"
    assert body["runId"] == "flash-1"
    assert body["status"] == "running"
    assert body["environmentSlug"] == "dev-clado-ai/my-env"
    assert body["model"] == "Qwen/Qwen3.5-4B"


def test_record_training_checkpoint_posts_to_backend(monkeypatch, tmp_path):
    from flash import runner
    from flash.runner import RunStatus
    from flash.server import run_registry
    from flash.spec import JobSpec

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://backend.test")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    seen: dict[str, object] = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = req.data
        return _Resp()

    monkeypatch.setattr(run_registry.urllib.request, "urlopen", fake_urlopen)
    spec = JobSpec.from_dict(
        {
            "run_id": "flash-1",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {"steps": 1, "hf_repo": "Freesolo-Co/flashrun-flash-1"},
        }
    )
    runner._save_status(
        RunStatus(
            run_id="flash-1",
            state="running",
            spec=spec.to_dict(),
            platform_context={"org_id": "org-1"},
        )
    )

    ok = run_registry.record_training_checkpoint(
        spec=spec,
        metrics={"cost_usd": 0.25},
        artifact_path="/tmp/artifacts",
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/runs/checkpoints/internal"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    body = json.loads(seen["body"])
    assert body["orgId"] == "org-1"
    assert body["runId"] == "flash-1"
    assert body["checkpointId"] == "final"
    assert body["adapterRef"] == "Freesolo-Co/flashrun-flash-1:rl/flash-1"
    assert body["metrics"] == {"cost_usd": 0.25}


def test_redacts_raw_and_url_encoded_token():
    token = "ghp_test/with:special"
    encoded = "ghp_test%2Fwith%3Aspecial"
    assert envs._redact(f"https://x-access-token:{encoded}@github.com\n{token}", token) == (
        "https://x-access-token:<redacted>@github.com\n<redacted>"
    )


def test_copy_package_to_checkout_rejects_escape(tmp_path):
    source = tmp_path / "source"
    checkout = tmp_path / "checkout"
    source.mkdir()
    checkout.mkdir()
    (source / "environment.py").write_text("def load_environment(**k): pass\n")
    with pytest.raises(envs.EnvPublishError, match="unsafe"):
        envs._copy_package_to_checkout(source=source, checkout=checkout, publish_root="../escape")


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


def test_safe_extract_rejects_repo_control_and_source_paths(tmp_path):
    for label in (
        ".github",
        ".git",
        "source",
        "./.github",
        "./.git",
        "./source",
    ):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            d = tarfile.TarInfo(f"{label}/workflows")
            d.type = tarfile.DIRTYPE
            tar.addfile(d)
        with pytest.raises(envs.EnvPublishError, match="top-level paths"):
            envs._safe_extract(buf.getvalue(), tmp_path)


def test_safe_extract_allows_environment_sidecars(tmp_path):
    files = {
        "datasets/train.jsonl": '{"x": 1}\n',
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    envs._safe_extract(buf.getvalue(), tmp_path)

    for name, content in files.items():
        assert (tmp_path / name).read_text() == content


def test_push_environment_commit_rebases_before_push(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run_git(cwd, args, *, token):
        calls.append(args)
        return subprocess.CompletedProcess(["git", *args], 0, "", "")

    monkeypatch.setattr(envs, "_run_git", fake_run_git)
    envs._push_environment_commit(checkout=tmp_path, token="tok")

    assert calls == [
        ["pull", "--rebase", "origin", "main"],
        ["push", "origin", "HEAD:main"],
    ]


def test_github_publish_retries_concurrent_push(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    (tmp_path / "environment.py").write_text("def load_environment(**k): pass\n")
    calls = {"count": 0}

    def fake_publish_once(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise envs.EnvPublishError("failed to push some refs")

    monkeypatch.setattr(envs, "_github_publish_once", fake_publish_once)
    monkeypatch.setattr(envs.time, "sleep", lambda _seconds: None)

    ref = envs._github_publish(tmp_path, name="e", key={"email": "dev@clado.ai"})

    assert calls["count"] == 2
    assert ref == "dev-clado-ai/e"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_github_publish_once_commits_pull_rebases_and_pushes(tmp_path, monkeypatch):
    remote = tmp_path / "training.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "--initial-branch", "main")
    _git(seed, "config", "user.name", "test")
    _git(seed, "config", "user.email", "test@example.com")
    (seed / "README.md").write_text("hub\n")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "init", "--bare", str(remote))
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "origin", "main")

    package = tmp_path / "package"
    package.mkdir()
    (package / "environment.py").write_text("def load_environment(**k): pass\n")

    monkeypatch.setattr(envs, "_credentialed_repo_url", lambda repo, token: str(remote))

    envs._github_publish_once(
        dest=package,
        repo="ignored/repo",
        token="tok",
        publish_root="ns/env/publish-1",
        message="Upload test env",
    )

    verify = tmp_path / "verify"
    _git(tmp_path, "clone", "--branch", "main", str(remote), str(verify))
    assert (verify / "ns/env/publish-1/environment.py").read_text() == (
        "def load_environment(**k): pass\n"
    )
