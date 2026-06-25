"""Server-side managed Freesolo env publishing to Azure Blob + Postgres."""

from __future__ import annotations

import base64
import io
import json
import tarfile

import pytest

from flash.server import azure_blob, environment_store, envs

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


@pytest.fixture
def fake_azure(monkeypatch):
    """Stub Azure Blob upload + Postgres upsert; capture what publish would write."""
    captured: dict[str, object] = {}

    def fake_upload(blob_key, data):
        captured["blob_key"] = blob_key
        captured["data"] = data

    def fake_upsert(**kwargs):
        captured["upsert"] = kwargs

    monkeypatch.setattr(azure_blob, "upload_package", fake_upload)
    monkeypatch.setattr(azure_blob, "container_name", lambda: "flash-environments")
    monkeypatch.setattr(environment_store, "upsert", fake_upsert)
    return captured


def test_namespace_is_stable_and_repo_safe():
    assert envs.namespace_for({"email": "Dev@Clado.ai"}) == "dev-clado-ai"
    assert envs.namespace_for({"email": "dev@clado.ai"}) == "dev-clado-ai"


def test_namespace_requires_email():
    for key in ({"id": 7}, {"key_prefix": "fslo-abc"}, {}, {"email": "missing-email"}):
        with pytest.raises(envs.EnvPublishError, match="email"):
            envs.namespace_for(key)


def test_internal_key_gets_reserved_namespace_without_email():
    assert envs.namespace_for({"auth_kind": "internal"}) == envs._INTERNAL_NAMESPACE
    assert envs.namespace_for({"auth_kind": "internal", "email": ""}) == envs._INTERNAL_NAMESPACE
    assert (
        envs.namespace_for({"auth_kind": "internal", "email": "missing-email"})
        == envs._INTERNAL_NAMESPACE
    )


def test_internal_special_case_does_not_loosen_user_keys():
    for key in ({"auth_kind": "external"}, {"auth_kind": "user", "email": "no-at"}, {"email": ""}):
        with pytest.raises(envs.EnvPublishError, match="email"):
            envs.namespace_for(key)


def test_sanitize_name_never_returns_path_segments():
    assert envs._sanitize_name("..") == "env"
    assert envs._sanitize_name(".") == "env"
    assert envs._sanitize_name("___") == "env"
    assert envs._sanitize_name("My Env!") == "my-env"


def test_blob_key_is_deterministic_from_slug():
    assert envs.blob_key_for("dev-clado-ai/my-env") == "flash-envs/dev-clado-ai/my-env/package.tar.gz"


def test_publish_uploads_to_azure_and_returns_slug(fake_azure):
    package = _pkg_b64(_MINIMAL)
    ref = envs.publish_package(
        package_b64=package,
        name="My Env!",
        key={"email": "dev@clado.ai"},
    )

    slug = "dev-clado-ai/my-env"
    assert ref == slug
    assert fake_azure["blob_key"] == "flash-envs/dev-clado-ai/my-env/package.tar.gz"
    # The original uploaded tarball bytes are stored verbatim.
    assert fake_azure["data"] == base64.b64decode(package)
    upsert = fake_azure["upsert"]
    assert upsert["slug"] == slug
    assert upsert["namespace"] == "dev-clado-ai"
    assert upsert["name"] == "my-env"
    assert upsert["blob_container"] == "flash-environments"
    assert upsert["blob_key"] == "flash-envs/dev-clado-ai/my-env/package.tar.gz"
    assert len(upsert["package_sha256"]) == 64
    assert upsert["size_bytes"] == len(base64.b64decode(package))


def test_publish_rejects_bad_input(fake_azure):
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


def test_publish_requires_blob_config(monkeypatch):
    # No connection string configured -> AzureBlobNotConfigured -> 503 (control-plane misconfig).
    monkeypatch.delenv("FLASH_ENV_BLOB_CONNECTION_STRING", raising=False)
    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.publish_package(
            package_b64=_pkg_b64(_MINIMAL), name="e", key={"email": "dev@clado.ai"}
        )
    assert excinfo.value.status == 503
    assert "FLASH_ENV_BLOB_CONNECTION_STRING" in str(excinfo.value)


def test_record_published_environment_posts_blob_pointer(monkeypatch):
    from flash.server import environment_registry

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://backend.test")
    monkeypatch.setattr(azure_blob, "container_name", lambda: "flash-environments")
    monkeypatch.setattr(
        environment_store,
        "lookup",
        lambda slug: environment_store.EnvironmentRecord(
            slug=slug,
            namespace="dev-clado-ai",
            name="my-env",
            blob_container="flash-environments",
            blob_key="flash-envs/dev-clado-ai/my-env/package.tar.gz",
            package_sha256="a" * 64,
            size_bytes=10,
            version=1,
        ),
    )
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
        "blobContainer": "flash-environments",
        "blobKey": "flash-envs/dev-clado-ai/my-env/package.tar.gz",
        "packageSha256": "a" * 64,
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
            "train": {"steps": 1, "seeds": [0], "hf_repo": "Freesolo-Co/flashrun-flash-1"},
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
        seed=0,
        metrics={"cost_usd": 0.25},
        artifact_path="/tmp/seed0",
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/runs/checkpoints/internal"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    body = json.loads(seen["body"])
    assert body["orgId"] == "org-1"
    assert body["runId"] == "flash-1"
    assert body["checkpointId"] == "seed0"
    assert body["adapterRef"] == "Freesolo-Co/flashrun-flash-1:rl/flash-1/seed0"
    assert body["metrics"] == {"cost_usd": 0.25}


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
