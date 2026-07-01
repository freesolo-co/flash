"""Server-side managed Freesolo env publishing to GitHub."""

from __future__ import annotations

import base64
import gzip
import io
import json
import subprocess
import tarfile
import tracemalloc
from pathlib import Path

import pytest

from flash.server import envs


def _gnu_longname_bomb(name_len: int) -> bytes:
    """A tiny gzip whose GNU LONGNAME header declares ``name_len`` bytes of highly-compressible name
    payload — consumed inside ``tarfile.next()`` before any member is yielded, so per-member size
    accounting never sees it. Models a decompression bomb that would OOM the control plane."""

    def header(name: str, size: int, typeflag: str) -> bytes:
        h = bytearray(512)
        nb = name.encode()[:100]
        h[0 : len(nb)] = nb
        h[100:108] = b"0000644\0"
        h[124 : 124 + 12] = f"{size:011o}\0".encode()
        h[136:148] = b"00000000000\0"
        h[156] = ord(typeflag)
        h[257:263] = b"ustar\0"
        h[263:265] = b"00"
        chk = sum(h[:148]) + sum(h[156:]) + 32 * 8
        h[148 : 148 + 8] = f"{chk:06o}\0 ".encode()
        return bytes(h)

    longlink = header("././@LongLink", name_len, "L")
    pad = b"\0" * ((512 - name_len % 512) % 512)
    tail = header("pkg/environment.py", 1, "0") + b"x" + b"\0" * 511 + b"\0" * 1024
    buf = io.BytesIO()
    # Stream the LONGNAME payload into the gzip writer in fixed-size chunks rather than
    # materializing all ``name_len`` bytes (plus a second copy via ``bytes(raw)``) in RAM — at
    # 400 MB that peaks ~1 GB and can OOM/slow CI before the extraction code under test runs.
    # Chunked writes feed one continuous zlib stream (no flushes between), so the gzip output is
    # byte-identical to a single write of the concatenation.
    block = b"A" * min(name_len, 1 << 20)
    with gzip.GzipFile(fileobj=buf, mode="wb") as g:
        g.write(longlink)
        remaining = name_len
        while remaining > 0:
            n = min(remaining, len(block))
            g.write(block if n == len(block) else block[:n])
            remaining -= n
        g.write(pad)
        g.write(tail)
    return buf.getvalue()


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


def test_namespace_uses_org_slug():
    assert envs.namespace_for({"org_slug": "acme"}) == "acme"
    assert envs.namespace_for({"org": {"slug": "freesolo-co"}}) == "freesolo-co"
    assert envs.namespace_for({"email": "dev@clado.ai", "org_slug": "acme"}) == "acme"


def test_namespace_requires_org_slug():
    for key in (
        {"id": 7},
        {"key_prefix": "fslo-abc"},
        {},
        {"email": "dev@clado.ai"},
        {"org_name": "Acme"},
        {"org_slug": "Bad Slug"},
    ):
        with pytest.raises(envs.EnvPublishError, match="org slug"):
            envs.namespace_for(key)


def test_internal_key_does_not_get_publish_namespace_fallback():
    for key in (
        {"auth_kind": "internal"},
        {"auth_kind": "internal", "email": "internal@freesolo.co"},
    ):
        with pytest.raises(envs.EnvPublishError, match="org slug"):
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
        key={"email": "dev@clado.ai", "org_slug": "acme"},
    )

    root = "acme/my-env"
    assert ref == root
    assert captured["repo"] == "freesolo-co/environment-hub"
    assert captured["token"] == "ghp-test"
    assert captured["publish_root"] == root
    assert captured["message"] == "Upload Flash environment acme/my-env"
    assert captured["files"] == ["environment.py", "pyproject.toml"]


def test_publish_accepts_matching_explicit_namespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    captured: dict[str, object] = {}

    def fake_publish_once(*, publish_root, message, **_kwargs):
        captured.update(publish_root=publish_root, message=message)

    monkeypatch.setattr(envs, "_github_publish_once", fake_publish_once)

    ref = envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="benchmark/Math Python",
        key={"org_slug": "benchmark"},
    )

    assert ref == "benchmark/math-python"
    assert captured["publish_root"] == "benchmark/math-python"
    assert captured["message"] == "Upload Flash environment benchmark/math-python"


def test_publish_rejects_mismatched_explicit_namespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setattr(envs, "_github_publish_once", lambda **_kwargs: None)

    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.publish_package(
            package_b64=_pkg_b64(_MINIMAL),
            name="benchmark/math-python",
            key={"org_slug": "acme"},
        )

    assert excinfo.value.status == 403
    assert "namespace" in str(excinfo.value)


def test_publish_rejects_invalid_explicit_namespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setattr(envs, "_github_publish_once", lambda **_kwargs: None)

    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.publish_package(
            package_b64=_pkg_b64(_MINIMAL),
            name="!!!/math-python",
            key={"org_slug": "env"},
        )

    assert excinfo.value.status == 400
    assert "env namespace" in str(excinfo.value)


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
            key={"org_slug": "acme"},
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
        slug="acme/my-env",
        name="My Env",
        key={"org_id": "org-1", "user_id": "user-1", "api_key_id": "key-1"},
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/environments/internal"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    body = json.loads(seen["body"])
    assert body == {
        "orgId": "org-1",
        "slug": "acme/my-env",
        "name": "My Env",
        "hubRepo": "freesolo-co/environment-hub",
        "hubRef": "main",
        "hubPath": "acme/my-env/environment.py",
        "publishedByUserId": "user-1",
        "apiKeyId": "key-1",
        "metadata": {"source": "flash.env.push"},
    }


def test_record_published_environment_is_best_effort(monkeypatch):
    from flash.server import environment_registry

    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    assert (
        environment_registry.record_published_environment(
            slug="acme/my-env",
            name="My Env",
            key={"org_id": "org-1"},
        )
        is False
    )


def _capture_delete_request(monkeypatch):
    """Stub urlopen for record_deleted_environment and return the dict it records into."""
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
        seen["method"] = req.get_method()
        seen["body"] = req.data
        return _Resp()

    monkeypatch.setattr(environment_registry.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_record_deleted_environment_uses_caller_org_for_internal_key(monkeypatch):
    # The internal key is org-agnostic, so the web UI delete supplies the org explicitly.
    from flash.server import environment_registry

    seen = _capture_delete_request(monkeypatch)
    ok = environment_registry.record_deleted_environment(
        slug="acme/my-env",
        key={"auth_kind": "internal"},
        org_id="org-acme",
    )
    assert ok is True
    assert seen["method"] == "DELETE"
    assert json.loads(seen["body"]) == {"orgId": "org-acme", "slug": "acme/my-env"}


def test_record_deleted_environment_prefers_key_org_over_supplied(monkeypatch):
    # A user key carries its own org, which must win over any caller-supplied override so a
    # forged header can't drop another org's row.
    from flash.server import environment_registry

    seen = _capture_delete_request(monkeypatch)
    ok = environment_registry.record_deleted_environment(
        slug="acme/my-env",
        key={"org_id": "org-key"},
        org_id="org-other",
    )
    assert ok is True
    assert json.loads(seen["body"])["orgId"] == "org-key"


def test_record_deleted_environment_without_any_org_is_noop(monkeypatch):
    # No key org and no supplied org: nothing to target, so it must not POST.
    from flash.server import environment_registry

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("must not call the backend without an org"),
    )
    assert (
        environment_registry.record_deleted_environment(
            slug="acme/my-env",
            key={"auth_kind": "internal"},
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
        slug="acme/my-env",
        run_id="flash-1",
        key={"org_id": "org-1"},
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/environments/use/internal"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    assert json.loads(seen["body"]) == {
        "orgId": "org-1",
        "slug": "acme/my-env",
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
                "environment": {"id": "acme/my-env"},
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
    assert body["environmentSlug"] == "acme/my-env"
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


def test_safe_extract_skips_tar_metadata_entries(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        pax = tarfile.TarInfo("pax_global_header")
        pax.type = tarfile.XGLTYPE
        tar.addfile(pax)
        data = b"ok"
        info = tarfile.TarInfo("pkg/environment.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    envs._safe_extract(buf.getvalue(), tmp_path)

    assert (tmp_path / "pkg" / "environment.py").read_bytes() == b"ok"


def test_safe_extract_metadata_entries_do_not_count_toward_member_cap(monkeypatch, tmp_path):
    metadata_type = b"Z"
    monkeypatch.setattr(envs, "_MAX_MEMBERS", 2)
    monkeypatch.setattr(envs, "_TAR_METADATA_TYPES", envs._TAR_METADATA_TYPES | {metadata_type})
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for idx in range(3):
            pax = tarfile.TarInfo(f"metadata-{idx}")
            pax.type = metadata_type
            tar.addfile(pax)
        data = b"ok"
        info = tarfile.TarInfo("pkg/environment.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    envs._safe_extract(buf.getvalue(), tmp_path)

    assert (tmp_path / "pkg" / "environment.py").read_bytes() == b"ok"


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


def test_safe_extract_rejects_longname_decompression_bomb(tmp_path):
    # A ~400KB upload declaring a 400MB GNU LONGNAME header must be rejected with memory bounded near
    # the uncompressed limit (256MB), not the declared size — the header payload is read inside
    # tarfile.next() and is invisible to per-member size accounting.
    bomb = _gnu_longname_bomb(400 * 1024 * 1024)
    assert len(bomb) < 2 * 1024 * 1024
    tracemalloc.start()
    try:
        with pytest.raises(envs.EnvPublishError, match="too large uncompressed"):
            envs._safe_extract(bomb, tmp_path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 600 * 1024 * 1024, f"peak memory {peak} not bounded by the limit"


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

    ref = envs._github_publish(tmp_path, name="e", key={"org_slug": "acme"})

    assert calls["count"] == 2
    assert ref == "acme/e"


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


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_package_user_can_delete_own_namespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    seen: dict[str, object] = {}

    def fake_github_delete(slug, *, token):
        seen.update(slug=slug, token=token)
        return True

    monkeypatch.setattr(envs, "_github_delete", fake_github_delete)
    assert envs.delete_package(slug="acme/my-env", key={"org_slug": "acme"}) is True
    assert seen == {"slug": "acme/my-env", "token": "ghp-test"}


def test_delete_package_rejects_other_users_namespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setattr(
        envs, "_github_delete", lambda *a, **k: pytest.fail("storage must not be touched")
    )
    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.delete_package(slug="someone-else/env", key={"org_slug": "acme"})
    assert excinfo.value.status == 403


def test_delete_package_internal_key_can_delete_any_namespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        envs, "_github_delete", lambda slug, *, token: seen.update(slug=slug) or True
    )
    assert envs.delete_package(slug="acme/paper-foo", key={"auth_kind": "internal"}) is True
    assert seen["slug"] == "acme/paper-foo"


def test_delete_package_internal_key_rejects_repo_control_namespace(monkeypatch):
    # The internal key bypasses the namespace-ownership check, so _validate_slug is the ONLY barrier
    # before `git rm -r -- <namespace>/<name>`. A GENUINE repo-control top-level path (a dir at the
    # root of the hub checkout) must be rejected so e.g. DELETE /v1/envs/.github/workflows can't
    # remove tracked repo infrastructure. Only `.git`/`.github` qualify because org slugs can never
    # be dot-prefixed, so these are never publishable.
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setattr(
        envs, "_github_delete", lambda *a, **k: pytest.fail("storage must not be touched")
    )
    assert {".git", ".github"} == envs._REPO_CONTROL_TOP_LEVEL_PATHS
    assert "source" not in envs._REPO_CONTROL_TOP_LEVEL_PATHS
    for blocked in envs._REPO_CONTROL_TOP_LEVEL_PATHS:
        with pytest.raises(envs.EnvPublishError, match="invalid env id segment"):
            envs.delete_package(slug=f"{blocked}/workflows", key={"auth_kind": "internal"})


def test_delete_package_allows_publishable_source_namespace(monkeypatch):
    # Regression guard for publish/delete symmetry: `source` is in publish's package-CONTENT
    # blocklist (_BLOCKED_TOP_LEVEL_PATHS via _safe_extract) but is a legitimate org namespace.
    # Delete must therefore reach storage for `source/<name>` (not 400), or those envs would be
    # publishable-but-undeletable.
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    seen: dict[str, str] = {}

    def fake_delete(canonical, *, token):
        seen["slug"] = canonical
        return True

    monkeypatch.setattr(envs, "_github_delete", fake_delete)
    # A user key whose org slug is `source` deletes its OWN `source/<name>`.
    source_key = {"org_slug": "source"}
    assert envs.namespace_for(source_key) == "source"
    assert "source" in envs._BLOCKED_TOP_LEVEL_PATHS  # still barred from package CONTENTS
    assert envs.delete_package(slug="source/my-env", key=source_key) is True
    assert seen["slug"] == "source/my-env"
    # And the internal key (which may delete any namespace) reaches storage too.
    assert envs.delete_package(slug="source/other", key={"auth_kind": "internal"}) is True
    assert seen["slug"] == "source/other"


def test_delete_package_validates_slug(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setattr(
        envs, "_github_delete", lambda *a, **k: pytest.fail("storage must not be touched")
    )
    # incl. a VALID two-segment id with a trailing/leading slash (e.g. from a `:path` route capture)
    # or surrounding/embedded whitespace (e.g. an encoded `ns/env%20` decoded to `ns/env `): each
    # must be REJECTED, not silently normalized to ns/env (which would leak a non-canonical id that
    # the response / metadata mirror would then carry while deletion targets the trimmed slug).
    for bad in (
        "noslash",
        "a/b/c",
        "ns/..",
        "../escape",
        "ns/",
        "/name",
        "ns/bad name",
        "ns/env/",
        "/ns/env",
        "ns/env ",
        " ns/env",
        "ns/env\t",
        "ns /env",
        "  ns/env  ",
    ):
        with pytest.raises(envs.EnvPublishError):
            envs.delete_package(slug=bad, key={"auth_kind": "internal"})


def test_canonical_env_id_accepts_only_canonical_form():
    assert envs.canonical_env_id("acme/my-env") == "acme/my-env"
    for bad in ("ns/env/", " ns/env", "ns/env ", "ns/env%20", "Ns/Env", "noslash"):
        with pytest.raises(envs.EnvPublishError):
            envs.canonical_env_id(bad)


def test_delete_package_requires_github_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.delete_package(slug="acme/x", key={"auth_kind": "internal"})
    assert excinfo.value.status == 503
    assert "GITHUB_TOKEN" in str(excinfo.value)


def test_github_delete_once_removes_dir_and_pushes(tmp_path, monkeypatch):
    remote = tmp_path / "hub.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "--initial-branch", "main")
    _git(seed, "config", "user.name", "test")
    _git(seed, "config", "user.email", "test@example.com")
    (seed / "README.md").write_text("hub\n")
    env_dir = seed / "ns" / "env"
    env_dir.mkdir(parents=True)
    (env_dir / "environment.py").write_text("def load_environment(**k): pass\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "init", "--bare", str(remote))
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "origin", "main")

    monkeypatch.setattr(envs, "_credentialed_repo_url", lambda repo, token: str(remote))
    removed = envs._github_delete_once(
        repo="ignored/repo", token="tok", publish_root="ns/env", message="Delete test env"
    )

    assert removed is True
    verify = tmp_path / "verify"
    _git(tmp_path, "clone", "--branch", "main", str(remote), str(verify))
    assert not (verify / "ns" / "env").exists()
    # unrelated content is untouched
    assert (verify / "README.md").read_text() == "hub\n"


def test_github_delete_once_idempotent_when_absent(tmp_path, monkeypatch):
    remote = tmp_path / "hub.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "--initial-branch", "main")
    _git(seed, "config", "user.name", "test")
    _git(seed, "config", "user.email", "test@example.com")
    (seed / "README.md").write_text("hub\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "init", "--bare", str(remote))
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "origin", "main")

    monkeypatch.setattr(envs, "_credentialed_repo_url", lambda repo, token: str(remote))
    removed = envs._github_delete_once(
        repo="ignored/repo", token="tok", publish_root="ns/absent", message="Delete absent"
    )
    assert removed is False


def test_staged_has_changes_maps_exit_codes(monkeypatch, tmp_path):
    # 1 => staged changes, 0 => clean; any other code (e.g. 128 for a broken repo) is a controlled
    # error, never a "go ahead and commit/push" signal, and a missing git binary is a 503.
    def fake_run(returncode):
        def _run(*_a, **_k):
            return subprocess.CompletedProcess(["git"], returncode, "", "fatal: not a git repo")

        return _run

    monkeypatch.setattr(envs.subprocess, "run", fake_run(1))
    assert envs._staged_has_changes(tmp_path) is True
    monkeypatch.setattr(envs.subprocess, "run", fake_run(0))
    assert envs._staged_has_changes(tmp_path) is False

    monkeypatch.setattr(envs.subprocess, "run", fake_run(128))
    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs._staged_has_changes(tmp_path)
    assert excinfo.value.status == 502

    def _missing(*_a, **_k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(envs.subprocess, "run", _missing)
    with pytest.raises(envs.EnvPublishError) as exc_missing:
        envs._staged_has_changes(tmp_path)
    assert exc_missing.value.status == 503


def test_github_delete_once_reapplies_removal_after_concurrent_publish(tmp_path, monkeypatch):
    # A concurrent publish adds a NEW sidecar under the SAME slug after our checkout clones but
    # before the push. The rebase only replays our original removal, so without re-running `git rm`
    # the sidecar would survive and the slug would be only partially deleted while we report success.
    remote = tmp_path / "hub.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "--initial-branch", "main")
    _git(seed, "config", "user.name", "test")
    _git(seed, "config", "user.email", "test@example.com")
    (seed / "README.md").write_text("hub\n")
    env_dir = seed / "ns" / "env"
    env_dir.mkdir(parents=True)
    (env_dir / "environment.py").write_text("def load_environment(**k): pass\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "init", "--bare", str(remote))
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "origin", "main")

    monkeypatch.setattr(envs, "_credentialed_repo_url", lambda repo, token: str(remote))

    # Inject the concurrent publish exactly once, right as the original delete commit is staged
    # (the first `_staged_has_changes` call) — before `_push_environment_delete` rebases — by pushing
    # a sidecar under ns/env through a separate clone.
    real_staged = envs._staged_has_changes
    state = {"injected": False}

    def staged_with_injection(checkout):
        result = real_staged(checkout)
        if not state["injected"]:
            state["injected"] = True
            other = tmp_path / "other"
            _git(tmp_path, "clone", "--branch", "main", str(remote), str(other))
            _git(other, "config", "user.name", "other")
            _git(other, "config", "user.email", "other@example.com")
            (other / "ns" / "env" / "extra.py").write_text("# concurrently added sidecar\n")
            _git(other, "add", "-A")
            _git(other, "commit", "-m", "concurrent publish under same slug")
            _git(other, "push", "origin", "main")
        return result

    monkeypatch.setattr(envs, "_staged_has_changes", staged_with_injection)

    removed = envs._github_delete_once(
        repo="ignored/repo", token="tok", publish_root="ns/env", message="Delete ns/env"
    )
    assert removed is True

    verify = tmp_path / "verify"
    _git(tmp_path, "clone", "--branch", "main", str(remote), str(verify))
    # The slug directory is FULLY gone despite the concurrent re-add under it.
    assert not (verify / "ns" / "env").exists()
    assert (verify / "README.md").read_text() == "hub\n"


def test_github_delete_retries_concurrent_push(monkeypatch):
    calls = {"count": 0}

    def fake_delete_once(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise envs.EnvPublishError("failed to push some refs")
        return True

    monkeypatch.setattr(envs, "_github_delete_once", fake_delete_once)
    monkeypatch.setattr(envs.time, "sleep", lambda _seconds: None)
    assert envs._github_delete("ns/env", token="tok") is True
    assert calls["count"] == 2


def test_record_deleted_environment_sends_delete(monkeypatch):
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
        seen["method"] = req.get_method()
        seen["headers"] = dict(req.header_items())
        seen["body"] = req.data
        return _Resp()

    monkeypatch.setattr(environment_registry.urllib.request, "urlopen", fake_urlopen)

    ok = environment_registry.record_deleted_environment(
        slug="acme/my-env",
        key={"org_id": "org-1"},
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/environments/internal"
    assert seen["method"] == "DELETE"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    assert json.loads(seen["body"]) == {"orgId": "org-1", "slug": "acme/my-env"}


def test_record_deleted_environment_is_best_effort(monkeypatch):
    from flash.server import environment_registry

    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    assert (
        environment_registry.record_deleted_environment(
            slug="acme/my-env",
            key={"org_id": "org-1"},
        )
        is False
    )
