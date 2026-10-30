"""Server-side managed Freesolo env publishing to Azure."""

from __future__ import annotations

import base64
import gzip
import io
import json
import tarfile
import tracemalloc

import pytest

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
from flash.server.domain.registry import envs
from tests._helpers import deployments as deployment_fixtures
from tests._helpers.source_snapshot import valid_source_snapshot
from tests._helpers.wire_headers import sent_headers


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


def _issued_token(prefix: str) -> str:
    body_length = {"fslo_": 45, "hf_": 34, "pit_": 64}[prefix]
    seed = (
        "qR2_sT4-uV6wX8yZ0aB1cD3eF5gHjK7mN9pL"
        if prefix == "fslo_"
        else "qR2sT4uV6wX8yZ0aB1cD3eF5gHjK7mN9pL"
    )
    body = (seed * 2)[:body_length]
    if prefix == "fslo_":
        assert {"_", "-"} <= set(body)
    return prefix + body


def test_publish_rejects_direct_token_archive_before_registry_publish(monkeypatch):
    token = _issued_token("fslo_")
    path = "scripts/bootstrap.sh"
    package = {**_MINIMAL, path: f"#!/bin/sh\nexport FREESOLO_API_KEY={token}\n"}
    calls: list[str] = []
    monkeypatch.setattr(
        envs.blob_envs,
        "publish",
        lambda *_args, **_kwargs: (
            calls.append("registry") or pytest.fail("registry must not start")
        ),
    )

    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.publish_package(
            package_b64=_pkg_b64(package),
            name="direct-bypass",
            key={"org_slug": "acme"},
            project_slug="checkout-bot",
        )

    assert excinfo.value.status == 400
    error = str(excinfo.value)
    assert "direct access token" in error
    assert token not in error
    assert path not in error
    assert calls == []


def test_publish_direct_token_scan_failure_is_safe_400(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        envs,
        "package_contains_direct_token",
        lambda _dest: (_ for _ in ()).throw(envs.DirectTokenScanError()),
    )
    monkeypatch.setattr(
        envs.blob_envs,
        "publish",
        lambda *_args, **_kwargs: (
            calls.append("registry") or pytest.fail("registry must not start")
        ),
    )

    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.publish_package(
            package_b64=_pkg_b64(_MINIMAL),
            name="scan-failure",
            key={"org_slug": "acme"},
            project_slug="checkout-bot",
        )

    assert excinfo.value.status == 400
    assert str(excinfo.value) == "env package could not be scanned safely"
    assert calls == []


def test_publish_direct_token_clean_control_reaches_registry_publish(monkeypatch):
    calls: list[str] = []

    def fake_publish(slug, dest):
        calls.append("registry")
        assert (dest / "environment.py").is_file()
        return slug

    monkeypatch.setattr(envs.blob_envs, "publish", fake_publish)

    result = envs.publish_package(
        package_b64=_pkg_b64({**_MINIMAL, "assets/data.bin": "clean opaque bytes"}),
        name="clean-control",
        key={"org_slug": "acme"},
        project_slug="checkout-bot",
    )

    assert result == "acme/checkout-bot/clean-control"
    assert calls == ["registry"]


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


def test_publish_uploads_to_registry_and_returns_slug(monkeypatch):
    captured: dict[str, object] = {}

    def fake_publish_once(slug, dest):
        captured.update(
            publish_root=slug,
            files=sorted(
                path.relative_to(dest).as_posix() for path in dest.rglob("*") if path.is_file()
            ),
        )
        return slug

    monkeypatch.setattr(envs.blob_envs, "publish", fake_publish_once)
    ref = envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="My Env!",
        key={"email": "dev@clado.ai", "org_slug": "acme"},
        project_slug="checkout-bot",
    )

    root = "acme/checkout-bot/my-env"
    assert ref == root
    assert captured["publish_root"] == root
    assert captured["files"] == ["environment.py", "pyproject.toml"]


def test_same_name_in_two_projects_publishes_to_separate_roots(monkeypatch):
    published_roots: list[str] = []
    monkeypatch.setattr(
        envs.blob_envs,
        "publish",
        lambda slug, dest: published_roots.append(slug) or slug,
    )
    key = {"org_slug": "acme"}

    checkout_ref = envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="shared-name",
        key=key,
        project_slug="checkout-bot",
    )
    support_ref = envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="shared-name",
        key=key,
        project_slug="support-bot",
    )

    assert checkout_ref == "acme/checkout-bot/shared-name"
    assert support_ref == "acme/support-bot/shared-name"
    assert published_roots == [checkout_ref, support_ref]


def test_publish_accepts_matching_explicit_namespace(monkeypatch):
    captured: dict[str, object] = {}

    def fake_publish_once(slug, dest):
        captured.update(publish_root=slug)
        return slug

    monkeypatch.setattr(envs.blob_envs, "publish", fake_publish_once)

    ref = envs.publish_package(
        package_b64=_pkg_b64(_MINIMAL),
        name="benchmark/checkout-bot/Math Python",
        key={"org_slug": "benchmark"},
        project_slug="checkout-bot",
    )

    assert ref == "benchmark/checkout-bot/math-python"
    assert captured["publish_root"] == "benchmark/checkout-bot/math-python"


def test_publish_rejects_mismatched_explicit_namespace(monkeypatch):
    monkeypatch.setattr(
        envs.blob_envs, "publish", lambda *args: pytest.fail("storage must not be touched")
    )

    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.publish_package(
            package_b64=_pkg_b64(_MINIMAL),
            name="benchmark/checkout-bot/math-python",
            key={"org_slug": "acme"},
            project_slug="checkout-bot",
        )

    assert excinfo.value.status == 403
    assert "namespace" in str(excinfo.value)


def test_publish_rejects_invalid_explicit_namespace(monkeypatch):
    monkeypatch.setattr(
        envs.blob_envs, "publish", lambda *args: pytest.fail("storage must not be touched")
    )

    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.publish_package(
            package_b64=_pkg_b64(_MINIMAL),
            name="!!!/checkout-bot/math-python",
            key={"org_slug": "env"},
            project_slug="checkout-bot",
        )

    assert excinfo.value.status == 400
    assert "env namespace" in str(excinfo.value)


def test_publish_rejects_bad_input(monkeypatch):
    monkeypatch.setattr(
        envs.blob_envs, "publish", lambda *args: pytest.fail("storage must not be touched")
    )
    with pytest.raises(envs.EnvPublishError, match="base64"):
        envs.publish_package(
            package_b64="not base64!!!", name="e", key={}, project_slug="checkout-bot"
        )
    with pytest.raises(envs.EnvPublishError, match="empty"):
        envs.publish_package(package_b64="", name="e", key={}, project_slug="checkout-bot")
    with pytest.raises(envs.EnvPublishError, match="name"):
        envs.publish_package(
            package_b64=_pkg_b64(_MINIMAL), name="", key={}, project_slug="checkout-bot"
        )
    with pytest.raises(envs.EnvPublishError, match=r"environment\.py"):
        envs.publish_package(
            package_b64=_pkg_b64({"pyproject.toml": "[project]\nname='e'\n"}),
            name="e",
            key={"org_slug": "acme"},
            project_slug="checkout-bot",
        )


def test_record_published_environment_posts_to_backend(monkeypatch):
    from flash.server.domain.registry import environment_registry

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
        slug="acme/checkout-bot/my-env",
        name="My Env",
        key={"org_id": "org-1", "user_id": "user-1", "api_key_id": "key-1"},
        project_id="11111111-1111-4111-8111-111111111111",
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/environments/internal"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    body = json.loads(seen["body"])
    assert body == {
        "orgId": "org-1",
        "slug": "acme/checkout-bot/my-env",
        "name": "My Env",
        "hubRepo": "freesolo-co/environment-hub",
        "hubRef": "main",
        "hubPath": "acme/checkout-bot/my-env/environment.py",
        "publishedByUserId": "user-1",
        "apiKeyId": "key-1",
        "projectId": "11111111-1111-4111-8111-111111111111",
        "metadata": {"source": "flash.env.push"},
    }


def test_record_published_environment_sends_project_id(monkeypatch):
    """A validated project travels to the backend as `projectId`."""
    from flash.server.domain.registry import environment_registry

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
        seen["body"] = req.data
        return _Resp()

    monkeypatch.setattr(environment_registry.urllib.request, "urlopen", fake_urlopen)

    ok = environment_registry.record_published_environment(
        slug="acme/checkout-bot/my-env",
        name="My Env",
        key={"org_id": "org-1", "user_id": "user-1", "api_key_id": "key-1"},
        project_id="  11111111-1111-4111-8111-111111111111  ",
    )

    assert ok is True
    # whitespace around a pasted id is stripped, not sent through to the resolver.
    assert json.loads(seen["body"])["projectId"] == "11111111-1111-4111-8111-111111111111"


def test_record_published_environment_rejects_blank_project_id(monkeypatch):
    from flash.server.domain.registry import environment_registry

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
        seen["body"] = req.data
        return _Resp()

    monkeypatch.setattr(environment_registry.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="project_id is required"):
        environment_registry.record_published_environment(
            slug="acme/checkout-bot/my-env",
            name="My Env",
            key={"org_id": "org-1"},
            project_id="   ",
        )

    assert "body" not in seen


def test_record_published_environment_returns_false_without_internal_key(monkeypatch):
    from flash.server.domain.registry import environment_registry

    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    assert (
        environment_registry.record_published_environment(
            slug="acme/checkout-bot/my-env",
            name="My Env",
            key={"org_id": "org-1"},
            project_id="11111111-1111-4111-8111-111111111111",
        )
        is False
    )


def _capture_delete_request(monkeypatch):
    """Stub urlopen for record_deleted_environment and return the dict it records into."""
    from flash.server.domain.registry import environment_registry

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
    from flash.server.domain.registry import environment_registry

    seen = _capture_delete_request(monkeypatch)
    ok = environment_registry.record_deleted_environment(
        project_id="11111111-1111-4111-8111-111111111111",
        slug="acme/checkout-bot/my-env",
        key={"auth_kind": "internal"},
        org_id="org-acme",
    )
    assert ok is True
    assert seen["method"] == "DELETE"
    assert json.loads(seen["body"]) == {
        "orgId": "org-acme",
        "projectId": "11111111-1111-4111-8111-111111111111",
        "slug": "acme/checkout-bot/my-env",
    }


def test_record_deleted_environment_prefers_key_org_over_supplied(monkeypatch):
    # A user key carries its own org, which must win over any caller-supplied override so a
    # forged header can't drop another org's row.
    from flash.server.domain.registry import environment_registry

    seen = _capture_delete_request(monkeypatch)
    ok = environment_registry.record_deleted_environment(
        project_id="11111111-1111-4111-8111-111111111111",
        slug="acme/checkout-bot/my-env",
        key={"org_id": "org-key"},
        org_id="org-other",
    )
    assert ok is True
    assert json.loads(seen["body"])["orgId"] == "org-key"


def test_record_deleted_environment_without_any_org_is_noop(monkeypatch):
    # No key org and no supplied org: nothing to target, so it must not POST.
    from flash.server.domain.registry import environment_registry

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("must not call the backend without an org"),
    )
    assert (
        environment_registry.record_deleted_environment(
            project_id="11111111-1111-4111-8111-111111111111",
            slug="acme/checkout-bot/my-env",
            key={"auth_kind": "internal"},
        )
        is False
    )


def test_require_environment_project_posts_strict_validation(monkeypatch):
    from flash.server.domain.registry import environment_registry

    seen: dict = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def urlopen(req, timeout=None):
        seen.update(
            url=req.full_url,
            method=req.method,
            headers=sent_headers(req),
            body=json.loads(req.data),
            timeout=timeout,
        )
        return Response()

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://backend.test")
    monkeypatch.setattr(environment_registry.urllib.request, "urlopen", urlopen)

    environment_registry.require_environment_project(
        slug="acme/checkout-bot/example",
        project_id="11111111-1111-4111-8111-111111111111",
        key={"org_id": "org-A"},
    )

    assert seen == {
        "url": "https://backend.test/api/flash/environments/validate/internal",
        "method": "POST",
        "headers": {
            "Authorization": "Bearer internal-secret",
            "Content-Type": "application/json",
        },
        "body": {
            "orgId": "org-A",
            "projectId": "11111111-1111-4111-8111-111111111111",
            "slug": "acme/checkout-bot/example",
        },
        "timeout": 10.0,
    }


def test_require_environment_project_repairs_missing_legacy_environment(monkeypatch):
    import urllib.error

    from flash.server.domain.registry import environment_registry

    requests: list[dict] = []
    downloads: list[tuple[str, dict]] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(req, timeout=None):
        body = json.loads(req.data)
        requests.append({"url": req.full_url, "body": body, "timeout": timeout})
        if req.full_url.endswith("/validate/internal"):
            raise urllib.error.HTTPError(
                req.full_url,
                404,
                "not found",
                {},
                io.BytesIO(b'{"detail":"flash environment not found"}'),
            )
        return Response()

    key = {"org_id": "org-A", "org_slug": "acme", "user_id": "user-A"}
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://backend.test")
    monkeypatch.setattr(environment_registry.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(
        envs,
        "download_package",
        lambda *, slug, key: downloads.append((slug, key)) or b"package",
    )

    environment_registry.require_environment_project(
        slug="acme/checkout-bot/example",
        project_id="11111111-1111-4111-8111-111111111111",
        key=key,
        repair_missing=True,
    )

    assert downloads == [("acme/checkout-bot/example", key)]
    assert [request["url"] for request in requests] == [
        "https://backend.test/api/flash/environments/validate/internal",
        "https://backend.test/api/flash/environments/internal",
    ]
    assert requests[1]["body"] == {
        "orgId": "org-A",
        "slug": "acme/checkout-bot/example",
        "name": "example",
        "hubRepo": "freesolo-co/environment-hub",
        "hubRef": "main",
        "hubPath": "acme/checkout-bot/example/environment.py",
        "publishedByUserId": "user-A",
        "apiKeyId": None,
        "projectId": "11111111-1111-4111-8111-111111111111",
        "metadata": {"source": "flash.env.push"},
    }


def test_require_environment_project_repairs_missing_row_without_error_detail(monkeypatch):
    import urllib.error

    from flash.server.domain.registry import environment_registry

    error = urllib.error.HTTPError(
        "https://backend.test/api/flash/environments/validate/internal",
        404,
        "not found",
        {},
        io.BytesIO(b""),
    )
    downloads: list[str] = []
    records: list[dict] = []
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        envs,
        "download_package",
        lambda *, slug, key: downloads.append(slug) or b"package",
    )
    monkeypatch.setattr(
        environment_registry,
        "record_published_environment",
        lambda **kwargs: records.append(kwargs) or True,
    )

    environment_registry.require_environment_project(
        slug="acme/checkout-bot/example",
        project_id="11111111-1111-4111-8111-111111111111",
        key={"org_id": "org-A", "org_slug": "acme"},
        repair_missing=True,
    )

    assert downloads == ["acme/checkout-bot/example"]
    assert records[0]["project_id"] == "11111111-1111-4111-8111-111111111111"


def test_repairing_another_projects_environment_never_launches_the_run(monkeypatch):
    """A repair may not move an environment between projects.

    The repair path checks only the ORG segment locally, so `acme/proj-a/env` submitted under
    project B reaches `record_published_environment` carrying B's id. The recording endpoint is
    what rejects the mismatched pair (`backend/routes/flash.py` validates the slug's project
    segment against the supplied project), and a rejection there returns False rather than
    raising. So the load-bearing behavior is local: a non-True answer must abort. If it were
    ever read as success, the caller would launch a run against an environment it does not own.
    """
    import urllib.error

    from fastapi import HTTPException

    from flash.server.domain.registry import environment_registry

    error = urllib.error.HTTPError(
        "https://backend.test/api/flash/environments/validate/internal",
        404,
        "not found",
        {},
        io.BytesIO(b'{"detail":"flash environment not found"}'),
    )
    recorded: list[dict] = []
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(envs, "download_package", lambda **_kwargs: b"package")
    # the org segment matches, so the local guard passes and the backend is the one that says no.
    monkeypatch.setattr(
        environment_registry,
        "record_published_environment",
        lambda **kwargs: recorded.append(kwargs) or False,
    )

    with pytest.raises(HTTPException) as excinfo:
        environment_registry.require_environment_project(
            slug="acme/proj-a/example",
            project_id="22222222-2222-4222-8222-222222222222",
            key={"org_id": "org-A", "org_slug": "acme"},
            repair_missing=True,
        )

    # 502, not a silent return: the run must not proceed on an unrepaired association.
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == environment_registry._REPAIR_FAILURE_DETAIL
    # the mismatched pair was sent for validation rather than resolved locally to something else.
    assert recorded[0]["slug"] == "acme/proj-a/example"
    assert recorded[0]["project_id"] == "22222222-2222-4222-8222-222222222222"


def test_require_environment_project_missing_package_does_not_backfill(monkeypatch):
    import urllib.error

    from fastapi import HTTPException

    from flash.server.domain.registry import environment_registry

    error = urllib.error.HTTPError(
        "https://backend.test/api/flash/environments/validate/internal",
        404,
        "not found",
        {},
        io.BytesIO(b'{"detail":"flash environment not found"}'),
    )
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        envs,
        "download_package",
        lambda **_kwargs: (_ for _ in ()).throw(
            envs.EnvPublishError("environment package not found", status=404)
        ),
    )
    monkeypatch.setattr(
        environment_registry,
        "record_published_environment",
        lambda **_kwargs: pytest.fail("a missing package must not create a mirror row"),
    )

    with pytest.raises(HTTPException) as excinfo:
        environment_registry.require_environment_project(
            slug="acme/checkout-bot/example",
            project_id="11111111-1111-4111-8111-111111111111",
            key={"org_id": "org-A", "org_slug": "acme"},
            repair_missing=True,
        )

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "environment package not found"


def test_require_environment_project_cross_namespace_repair_preserves_404(monkeypatch):
    import urllib.error

    from fastapi import HTTPException

    from flash.server.domain.registry import environment_registry

    error = urllib.error.HTTPError(
        "https://backend.test/api/flash/environments/validate/internal",
        404,
        "not found",
        {},
        io.BytesIO(b'{"detail":"flash environment not found"}'),
    )
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        envs,
        "download_package",
        lambda **_kwargs: (_ for _ in ()).throw(
            envs.EnvPublishError(
                "you can only download environments in your own namespace",
                status=403,
            )
        ),
    )
    monkeypatch.setattr(
        environment_registry,
        "record_published_environment",
        lambda **_kwargs: pytest.fail("a cross-namespace package must not create a mirror row"),
    )

    with pytest.raises(HTTPException) as excinfo:
        environment_registry.require_environment_project(
            slug="other-org/checkout-bot/example",
            project_id="11111111-1111-4111-8111-111111111111",
            key={"org_id": "org-A", "org_slug": "acme"},
            repair_missing=True,
        )

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "flash environment not found"
    assert isinstance(excinfo.value.__cause__, envs.EnvPublishError)
    assert excinfo.value.__cause__.status == 403


@pytest.mark.parametrize("package_exists", [True, False], ids=["exists", "missing"])
def test_internal_repair_without_org_namespace_preserves_404(monkeypatch, package_exists):
    import urllib.error

    from fastapi import HTTPException

    from flash.server.domain.registry import environment_registry

    error = urllib.error.HTTPError(
        "https://backend.test/api/flash/environments/validate/internal",
        404,
        "not found",
        {},
        io.BytesIO(b'{"detail":"flash environment not found"}'),
    )
    downloads: list[str] = []
    records: list[dict] = []
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    def download_package(*, slug, key):
        downloads.append(slug)
        if package_exists:
            return b"package"
        raise envs.EnvPublishError("environment package not found", status=404)

    monkeypatch.setattr(envs, "download_package", download_package)
    monkeypatch.setattr(
        environment_registry,
        "record_published_environment",
        lambda **kwargs: records.append(kwargs) or True,
    )

    with pytest.raises(HTTPException) as excinfo:
        environment_registry.require_environment_project(
            slug="foreign-org/checkout-bot/example",
            project_id="11111111-1111-4111-8111-111111111111",
            key={"auth_kind": "internal", "org_id": "org-caller"},
            repair_missing=True,
        )

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "flash environment not found"
    assert downloads == []
    assert records == []


def test_require_environment_project_backfill_failure_is_502(monkeypatch):
    import urllib.error

    from fastapi import HTTPException

    from flash.server.domain.registry import environment_registry

    error = urllib.error.HTTPError(
        "https://backend.test/api/flash/environments/validate/internal",
        404,
        "not found",
        {},
        io.BytesIO(b'{"detail":"flash environment not found"}'),
    )
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(envs, "download_package", lambda **_kwargs: b"package")
    monkeypatch.setattr(
        environment_registry,
        "record_published_environment",
        lambda **_kwargs: False,
    )

    with pytest.raises(HTTPException) as excinfo:
        environment_registry.require_environment_project(
            slug="acme/checkout-bot/example",
            project_id="11111111-1111-4111-8111-111111111111",
            key={"org_id": "org-A", "org_slug": "acme"},
            repair_missing=True,
        )

    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == (
        "environment package exists, but its project association could not be repaired"
    )


def _validate_http_error(code: int, body: bytes):
    import urllib.error

    return urllib.error.HTTPError(
        "https://backend.test/api/flash/environments/validate/internal",
        code,
        "error",
        {},
        io.BytesIO(body),
    )


def test_record_published_environment_keeps_failures_best_effort(monkeypatch):
    """A backend 500 stays a ``False`` so the retry advice still applies."""
    from flash.server.domain.registry import environment_registry
    from flash.server.platform import internal_client

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setattr(
        internal_client.urllib.request,
        "urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(
            _validate_http_error(500, b'{"detail":"failed to persist flash environment"}')
        ),
    )

    assert (
        environment_registry.record_published_environment(
            slug="acme/checkout-bot/example",
            name="example",
            key={"org_id": "org-A"},
            project_id="22222222-2222-4222-8222-222222222222",
        )
        is False
    )


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (400, "slug namespace must match org slug"),
        (422, "invalid environment slug"),
        (409, "flash environment belongs to another project"),
    ],
)
def test_require_environment_project_preserves_input_errors(monkeypatch, status, detail):
    import urllib.error

    from fastapi import HTTPException

    from flash.server.domain.registry import environment_registry

    error = urllib.error.HTTPError(
        "https://backend.test/api/flash/environments/validate/internal",
        status,
        "invalid request",
        {},
        io.BytesIO(json.dumps({"detail": detail}).encode()),
    )
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        envs,
        "download_package",
        lambda **_kwargs: pytest.fail("an invalid request must not download the hub package"),
    )

    with pytest.raises(HTTPException) as excinfo:
        environment_registry.require_environment_project(
            slug="acme/checkout-bot/example",
            project_id="22222222-2222-4222-8222-222222222222",
            key={"org_id": "org-A"},
            repair_missing=True,
        )

    assert excinfo.value.status_code == status
    assert excinfo.value.detail == detail


def test_require_environment_project_fails_closed_without_internal_key(monkeypatch):
    from fastapi import HTTPException

    from flash.server.domain.registry import environment_registry

    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)

    with pytest.raises(HTTPException) as excinfo:
        environment_registry.require_environment_project(
            slug="acme/checkout-bot/example",
            project_id="11111111-1111-4111-8111-111111111111",
            key={"org_id": "org-A"},
        )

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Freesolo environment validation is unavailable"


def test_record_environment_use_posts_to_backend(monkeypatch):
    from flash.server.domain.registry import environment_registry

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
        project_id="11111111-1111-4111-8111-111111111111",
        slug="acme/checkout-bot/my-env",
        run_id="flash-1",
        key={"org_id": "org-1"},
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/environments/use/internal"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    assert json.loads(seen["body"]) == {
        "orgId": "org-1",
        "projectId": "11111111-1111-4111-8111-111111111111",
        "slug": "acme/checkout-bot/my-env",
        "runId": "flash-1",
    }


def test_record_training_run_posts_to_backend(monkeypatch):
    from flash.runner.lifecycle.state import RunStatus
    from flash.server.domain.registry import runs

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

    monkeypatch.setattr(runs.urllib.request, "urlopen", fake_urlopen)

    ok = runs.record_training_run(
        status=RunStatus(
            run_id="flash-1",
            report_sequence=1,
            state="running",
            spec={
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "phase": "rl",
                "environment": {"id": "acme/checkout-bot/my-env"},
                "project": "11111111-1111-4111-8111-111111111111",
                "gpu": {"type": "L40S"},
            },
            platform_context={
                "org_id": "org-1",
                "user_id": "user-1",
                "api_key_id": "key-1",
            },
            source_snapshot=valid_source_snapshot(),
            deployments=deployment_fixtures.records(
                "flash-1",
                {
                    "state": "ready",
                    "checkpoint_id": "flash-1/final",
                    "endpoint_name": "https://serve.example",
                    "adapter_hf_prefix": "private/path",
                },
            ),
            last_heartbeat={
                "attempt": 0,
                "stage": "sft_step",
                "source_provenance": {
                    "format_version": 1,
                    "sha256": "a" * 64,
                    "verified": True,
                    "verified_attempt": 0,
                },
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
    assert body["environmentSlug"] == "acme/checkout-bot/my-env"
    # the exact canonical project uuid is persisted with every managed training run.
    assert body["projectId"] == "11111111-1111-4111-8111-111111111111"
    assert body["model"] == "Qwen/Qwen3.5-9B"
    assert "checkpointId" not in body
    assert body["deployments"]["flash-1/final"] == {
        "state": "ready",
        "checkpoint_id": "flash-1/final",
        "endpoint": "https://serve.example",
    }
    assert "adapterRef" not in body
    assert body["lastHeartbeat"] == {"attempt": 0, "stage": "sft_step"}
    assert "private/path" not in json.dumps(body)
    assert "source_snapshot" not in json.dumps(body)
    assert "source_provenance" not in json.dumps(body)


def test_record_training_run_reports_the_gpu_class_actually_rented(monkeypatch):
    from flash.runner.lifecycle.state import RunStatus
    from flash.server.domain.registry import runs

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
        seen["body"] = req.data
        return _Resp()

    monkeypatch.setattr(runs.urllib.request, "urlopen", fake_urlopen)

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "sft",
        "environment": {"id": "acme/checkout-bot/my-env"},
        "project": "11111111-1111-4111-8111-111111111111",
        "gpu": {"type": ["A100 SXM", "A100 SXM"]},
    }
    context = {"org_id": "org-1", "user_id": "user-1", "api_key_id": "key-1"}

    # terminal persistence clears the remote, so the effective worker spec retains the selected class.
    runs.record_training_run(
        status=RunStatus(
            run_id="flash-1",
            report_sequence=1,
            state="cancelled",
            spec=spec,
            effective_preparation={"worker_spec": {"gpu": {"type": "A100 SXM"}}},
            platform_context=context,
        )
    )
    assert json.loads(seen["body"])["gpuType"] == "A100 SXM"

    # before allocation, the authored head remains the best available attribution.
    runs.record_training_run(
        status=RunStatus(
            run_id="flash-1",
            report_sequence=1,
            state="queued",
            spec=spec,
            platform_context=context,
        )
    )
    assert json.loads(seen["body"])["gpuType"] == "A100 SXM"


def test_record_training_checkpoint_posts_to_backend(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec
    from flash.runner.lifecycle.state import RunStatus
    from flash.server.domain.registry import runs

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://backend.test")
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
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

    monkeypatch.setattr(runs.urllib.request, "urlopen", fake_urlopen)
    spec = JobSpec.from_dict(
        {
            "run_id": "flash-1",
            "model": "Qwen/Qwen3.5-9B",
            "project": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "algorithm": "grpo",
            "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/flashrun-flash-1"},
        }
    )
    persisted_spec = spec.to_dict()
    persisted_spec["project"] = " AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA "
    runner_state._save_status(
        RunStatus(
            run_id="flash-1",
            state="running",
            spec=persisted_spec,
            updated_at=0.0,
            platform_context={"org_id": "org-1"},
        )
    )

    ok = runs.record_training_checkpoint(
        spec=spec,
        metrics={"cost_usd": 0.25, "step": 3},
        artifact_path="/tmp/artifacts",
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/runs/checkpoints/internal"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    assert json.loads(seen["body"]) == {
        "orgId": "org-1",
        "projectId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "runId": "flash-1",
        "checkpointId": "flash-1/final",
        "phase": "rl",
        "artifactPath": "/tmp/artifacts",
        "metrics": {"cost_usd": 0.25, "step": 3},
        "metadata": {"source": "flash.control_plane"},
        "updatedAt": "1970-01-01T00:00:00+00:00",
    }


@pytest.mark.parametrize("persisted_project", [None, "not-a-project-uuid"])
def test_record_training_checkpoint_rejects_invalid_persisted_project(
    monkeypatch, persisted_project
):
    from flash.core.spec import JobSpec
    from flash.runner.lifecycle.state import RunStatus
    from flash.server.domain.registry import runs

    spec = JobSpec.from_dict(
        {
            "run_id": "flash-1",
            "model": "Qwen/Qwen3.5-9B",
            "project": "11111111-1111-4111-8111-111111111111",
            "algorithm": "grpo",
            "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/flashrun-flash-1"},
        }
    )
    persisted_spec = spec.to_dict()
    if persisted_project is None:
        persisted_spec.pop("project")
    else:
        persisted_spec["project"] = persisted_project
    status = RunStatus(
        run_id="flash-1",
        state="running",
        spec=persisted_spec,
        platform_context={"org_id": "org-1"},
    )
    monkeypatch.setattr(runner_status, "get_status", lambda _run_id: status)
    monkeypatch.setattr(
        runs,
        "_post",
        lambda *_args, **_kwargs: pytest.fail("invalid project must not be reported"),
    )

    assert (
        runs.record_training_checkpoint(
            spec=spec,
            metrics={"cost_usd": 0.25},
            artifact_path="/tmp/artifacts",
        )
        is False
    )


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
    monkeypatch.setattr(envs, "TAR_METADATA_TYPES", envs.TAR_METADATA_TYPES | {metadata_type})
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


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_package_user_can_delete_own_namespace(monkeypatch):
    seen: dict[str, object] = {}

    def fake_delete(slug):
        seen.update(slug=slug)
        return True

    monkeypatch.setattr(envs.blob_envs, "delete", fake_delete)
    assert envs.delete_package(slug="acme/checkout-bot/my-env", key={"org_slug": "acme"}) is True
    assert seen == {"slug": "acme/checkout-bot/my-env"}


def test_delete_package_rejects_other_users_namespace(monkeypatch):
    monkeypatch.setattr(
        envs.blob_envs, "delete", lambda *a, **k: pytest.fail("storage must not be touched")
    )
    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.delete_package(slug="someone-else/checkout-bot/env", key={"org_slug": "acme"})
    assert excinfo.value.status == 403


def test_delete_package_internal_key_can_delete_any_namespace(monkeypatch):
    seen: dict[str, object] = {}
    monkeypatch.setattr(envs.blob_envs, "delete", lambda slug: seen.update(slug=slug) or True)
    assert (
        envs.delete_package(slug="acme/checkout-bot/paper-foo", key={"auth_kind": "internal"})
        is True
    )
    assert seen["slug"] == "acme/checkout-bot/paper-foo"


def test_delete_package_internal_key_rejects_repo_control_namespace(monkeypatch):
    # The internal key bypasses the namespace-ownership check, so _validate_slug is the ONLY barrier
    # before storage is accessed. Reserved namespaces remain invalid after the migration.
    monkeypatch.setattr(
        envs.blob_envs, "delete", lambda *a, **k: pytest.fail("storage must not be touched")
    )
    assert {".git", ".github"} == envs._REPO_CONTROL_TOP_LEVEL_PATHS
    assert "source" not in envs._REPO_CONTROL_TOP_LEVEL_PATHS
    for blocked in envs._REPO_CONTROL_TOP_LEVEL_PATHS:
        with pytest.raises(envs.EnvPublishError, match="invalid env id segment"):
            envs.delete_package(slug=f"{blocked}/project/workflows", key={"auth_kind": "internal"})


def test_delete_package_allows_publishable_source_namespace(monkeypatch):
    # Regression guard for publish/delete symmetry: `source` is in publish's package-CONTENT
    # blocklist (_BLOCKED_TOP_LEVEL_PATHS via _safe_extract) but is a legitimate org namespace.
    # Delete must therefore reach storage for `source/<project>/<name>` (not 400), or those envs would be
    # publishable-but-undeletable.
    seen: dict[str, str] = {}

    def fake_delete(canonical):
        seen["slug"] = canonical
        return True

    monkeypatch.setattr(envs.blob_envs, "delete", fake_delete)
    # A user key whose org slug is `source` deletes its OWN `source/<project>/<name>`.
    source_key = {"org_slug": "source"}
    assert envs.namespace_for(source_key) == "source"
    assert "source" in envs._BLOCKED_TOP_LEVEL_PATHS  # still barred from package CONTENTS
    assert envs.delete_package(slug="source/checkout-bot/my-env", key=source_key) is True
    assert seen["slug"] == "source/checkout-bot/my-env"
    # And the internal key (which may delete any namespace) reaches storage too.
    assert (
        envs.delete_package(slug="source/checkout-bot/other", key={"auth_kind": "internal"}) is True
    )
    assert seen["slug"] == "source/checkout-bot/other"


def test_delete_package_validates_slug(monkeypatch):
    monkeypatch.setattr(
        envs.blob_envs, "delete", lambda *a, **k: pytest.fail("storage must not be touched")
    )
    # incl. a valid three-segment id with a trailing or leading slash from a `:path` route
    # capture, or surrounding or embedded whitespace: each must be rejected rather than normalized,
    # which would leak a non-canonical id that
    # the response / metadata mirror would then carry while deletion targets the trimmed slug).
    for bad in (
        "noslash",
        "a/b",
        "a/b/c/d",
        "ns/project/..",
        "../project/escape",
        "ns/project/",
        "/project/name",
        "ns/project/bad name",
        "ns/project/env/",
        "/ns/project/env",
        "ns/project/env ",
        " ns/project/env",
        "ns/project/env\t",
        "ns /project/env",
        "  ns/project/env  ",
    ):
        with pytest.raises(envs.EnvPublishError):
            envs.delete_package(slug=bad, key={"auth_kind": "internal"})


def test_canonical_env_id_accepts_only_canonical_form():
    assert envs.canonical_env_id("acme/checkout-bot/my-env") == "acme/checkout-bot/my-env"
    for bad in (
        "ns/project/env/",
        " ns/project/env",
        "ns/project/env ",
        "ns/project/env%20",
        "Ns/Project/Env",
        "ns/env",
        "noslash",
    ):
        with pytest.raises(envs.EnvPublishError):
            envs.canonical_env_id(bad)


def test_record_deleted_environment_sends_delete(monkeypatch):
    from flash.server.domain.registry import environment_registry

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
        project_id="11111111-1111-4111-8111-111111111111",
        slug="acme/checkout-bot/my-env",
        key={"org_id": "org-1"},
    )

    assert ok is True
    assert seen["url"] == "https://backend.test/api/flash/environments/internal"
    assert seen["method"] == "DELETE"
    assert seen["headers"]["Authorization"] == "Bearer internal-test"
    assert json.loads(seen["body"]) == {
        "orgId": "org-1",
        "projectId": "11111111-1111-4111-8111-111111111111",
        "slug": "acme/checkout-bot/my-env",
    }


def test_record_deleted_environment_is_best_effort(monkeypatch):
    from flash.server.domain.registry import environment_registry

    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    assert (
        environment_registry.record_deleted_environment(
            project_id="11111111-1111-4111-8111-111111111111",
            slug="acme/checkout-bot/my-env",
            key={"org_id": "org-1"},
        )
        is False
    )
