from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from azure.core.exceptions import ResourceExistsError, ResourceModifiedError, ResourceNotFoundError

from flash.envs.registry import azure, packages
from flash.server.domain.registry import envs

SLUG = "acme/project/test"


class Container:
    def __init__(self):
        self.objects = {}
        self.sequence = 0
        self.before_write = None

    def get_blob_client(self, name):
        return Blob(self, name)

    def list_blobs(self, *, name_starts_with, **kwargs):
        return [SimpleNamespace(name=n) for n in self.objects if n.startswith(name_starts_with)]


class Blob:
    def __init__(self, container, name):
        self.container, self.name = container, name

    def download_blob(self, **kwargs):
        if self.name not in self.container.objects:
            raise ResourceNotFoundError(status_code=404)
        data, etag = self.container.objects[self.name]
        return SimpleNamespace(
            size=len(data), properties=SimpleNamespace(etag=etag), chunks=lambda: [data]
        )

    def upload_blob(self, data, *, overwrite=False, etag=None, **kwargs):
        c = self.container
        if c.before_write:
            callback, c.before_write = c.before_write, None
            callback(self.name)
        old = c.objects.get(self.name)
        if not overwrite and old:
            raise ResourceExistsError(status_code=409)
        if etag is not None and (old is None or old[1] != etag):
            raise ResourceModifiedError(status_code=412)
        c.sequence += 1
        c.objects[self.name] = bytes(data), str(c.sequence)

    def delete_blob(self, *, etag, **kwargs):
        old = self.container.objects.get(self.name)
        if old is None or old[1] != etag:
            raise ResourceModifiedError(status_code=412)
        del self.container.objects[self.name]


@pytest.fixture
def registry(monkeypatch):
    container = Container()
    store = azure.BlobRegistry(container)

    @contextmanager
    def opened(**kwargs):
        try:
            yield store
        except ResourceNotFoundError as exc:
            raise azure.RegistryError("not found", status=404) from exc

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(azure, "open_registry", opened)
    monkeypatch.setattr(packages, "open_registry", opened)
    return store


def archive(content=b"VALUE = 1\n", *, path="environment.py"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        entry = tarfile.TarInfo(path)
        entry.size = len(content)
        tar.addfile(entry, io.BytesIO(content))
    return gzip.compress(buf.getvalue(), mtime=0)


def test_publish_pin_update_delete_preserves_old_jobs(registry):
    first = registry.publish(SLUG, archive())
    second = registry.publish(SLUG, archive(b"VALUE = 2\n"))
    assert first.revision != second.revision
    assert registry.resolve(SLUG) == second
    assert registry.resolve(SLUG, first.revision) == first
    assert registry.list("acme") == [SLUG]
    assert registry.list("other") == []
    assert registry.delete(SLUG)
    assert not registry.delete(SLUG)
    assert registry.list("acme") == []
    assert registry.download(registry.resolve(SLUG, first.revision)) == archive()


def test_backfill_preserves_git_revision_and_never_overwrites_a_publish(registry):
    original = archive()
    record = registry.publish(SLUG, original, source_revision="a" * 40, create_only=True)
    assert registry.download(registry.resolve(SLUG, "a" * 40)) == original
    assert registry.publish(SLUG, archive(b"other"), create_only=True) == record


def test_failed_upload_does_not_publish_a_head(registry):
    def fail(_):
        raise ConnectionError("interrupted upload")

    registry.container.before_write = fail
    with pytest.raises(ConnectionError):
        registry.publish(SLUG, archive())
    assert registry.list("acme") == []


def test_concurrent_writer_cannot_overwrite_newer_head(registry):
    registry.publish(SLUG, archive())

    def replace_head(_):
        registry.container.objects[f"heads/{SLUG}.json"] = b"newer", "new-etag"

    registry.container.before_write = replace_head
    with pytest.raises(ResourceModifiedError):
        registry.publish(SLUG, archive(b"stale writer"))
    assert registry.container.objects[f"heads/{SLUG}.json"][0] == b"newer"


def test_corruption_is_rejected(registry):
    record = registry.publish(SLUG, archive())
    registry.container.objects[f"archives/{SLUG}/{record.sha256}.tar.gz"] = b"tampered", "x"
    with pytest.raises(azure.RegistryError, match="checksum"):
        registry.download(record)


@pytest.mark.parametrize(
    "slug", ["../project/env", "org/../env", "org/project/env/extra", "org//env"]
)
def test_invalid_slug_never_accesses_storage(registry, slug):
    with pytest.raises(azure.RegistryError):
        registry.resolve(slug)
    assert not registry.container.objects


def test_azure_staging_and_loader_do_not_call_github(registry, monkeypatch, tmp_path):
    from flash.envs.loading import loader, staged

    def no_github(*args, **kwargs):
        pytest.fail("Azure managed environment contacted GitHub")

    monkeypatch.setattr(loader, "_urlopen", no_github)
    monkeypatch.setattr(loader, "_CACHE_ROOT", tmp_path / "cache")
    record = registry.publish(SLUG, archive())
    source = staged.resolve_environment_source(SLUG)
    try:
        assert source.resolved_sha == record.revision
        assert (source.root / source.entrypoint).read_text() == "VALUE = 1\n"
        resolved = loader._resolve_environment_reference(SLUG, record.revision)
        assert resolved.endswith("environment.py")
    finally:
        import shutil

        shutil.rmtree(source.staging_root)


def test_preflight_pins_without_github_token(registry, monkeypatch):
    from flash.core.spec import JobSpec
    from flash.envs.loading import loader
    from flash.runner.accounting.artifacts import preflight_validate_environment_ref

    monkeypatch.setattr(loader, "_urlopen", lambda *a, **kw: pytest.fail("called GitHub"))
    record = registry.publish(SLUG, archive())
    spec = JobSpec.from_dict({"model": "Qwen/Qwen3.5-9B", "environment": {"id": SLUG}})
    pinned, deferred = preflight_validate_environment_ref(spec)
    assert pinned.environment.resolved_sha == record.revision
    assert not deferred


def test_api_namespace_authorization_and_delete(registry):
    registry.publish(SLUG, archive())
    with pytest.raises(envs.EnvPublishError) as exc:
        envs.download_package(slug=SLUG, key={"org_slug": "other"})
    assert exc.value.status == 403
    assert envs.list_namespace_slugs(key={"org_slug": "acme"}) == [SLUG]
    assert envs.download_package(slug=SLUG, key={"org_slug": "acme"}) == archive()
    assert envs.delete_package(slug=SLUG, key={"org_slug": "acme"})


def test_archive_traversal_is_rejected(registry):
    from flash.envs.loading.staged import resolve_environment_source

    registry.publish(SLUG, archive(path="../environment.py"))
    with pytest.raises((ValueError, RuntimeError)):
        resolve_environment_source(SLUG)


def test_full_checksum_is_kept_in_revision(registry):
    data = archive()
    record = registry.publish(SLUG, data)
    assert record.sha256 == hashlib.sha256(data).hexdigest()
    assert len(record.revision) == 40


def test_deadline_expired_before_io():
    registry = azure.BlobRegistry(Container(), deadline_at=0)
    with pytest.raises(azure.RegistryError, match="deadline"):
        registry.resolve(SLUG)


@pytest.fixture
def api(registry):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from flash.server.platform.deps import require_key
    from flash.server.routes.env_files import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_key] = lambda: {"org_slug": "acme"}
    with TestClient(app) as client:
        yield client, app, require_key


def test_file_api_enforces_scope_and_paths(registry, api):
    client, app, require_key = api
    registry.publish(SLUG, archive())
    response = client.get(f"/v1/env-files/{SLUG}")
    assert response.status_code == 200
    assert response.json()["files"] == [{"path": "environment.py", "size": 10}]
    preview = client.post(f"/v1/env-files/{SLUG}", json={"paths": ["environment.py"]})
    assert preview.json()[0]["content"] == "VALUE = 1\n"
    raw = client.get(f"/v1/env-files/{SLUG}", params={"file": "environment.py"})
    assert raw.content == b"VALUE = 1\n"
    traversal = client.get(f"/v1/env-files/{SLUG}", params={"file": "../environment.py"})
    assert traversal.status_code == 400
    assert client.post(f"/v1/env-files/{SLUG}", json={"paths": "environment.py"}).status_code == 400
    app.dependency_overrides[require_key] = lambda: {"org_slug": "other"}
    assert client.get(f"/v1/env-files/{SLUG}").status_code == 403


def test_internal_seed_is_create_only_and_user_keys_cannot_seed(registry, api):
    import base64

    client, app, require_key = api
    payload = {"package_b64": base64.b64encode(archive()).decode()}
    assert client.put(f"/v1/env-seeds/{SLUG}", json=payload).status_code == 403
    app.dependency_overrides[require_key] = lambda: {"auth_kind": "internal"}
    assert client.put(f"/v1/env-seeds/{SLUG}", json=payload).status_code == 200
    first = registry.resolve(SLUG)
    payload = {"package_b64": base64.b64encode(archive(b"CHANGED = True\n")).decode()}
    assert client.put(f"/v1/env-seeds/{SLUG}", json=payload).status_code == 200
    assert registry.resolve(SLUG) == first


def test_missing_azure_environment_is_rejected_at_preflight(registry):
    from flash.core.spec import JobSpec
    from flash.runner.accounting.artifacts import preflight_validate_environment_ref
    from flash.runner.lifecycle.preparation import EnvironmentRefNotFound

    spec = JobSpec.from_dict({"model": "Qwen/Qwen3.5-9B", "environment": {"id": SLUG}})
    with pytest.raises(EnvironmentRefNotFound):
        preflight_validate_environment_ref(spec)


def test_transient_azure_failure_remains_retryable():
    from flash.envs.loading.staged import is_staged_environment_transient_error

    assert is_staged_environment_transient_error(azure.RegistryError("unavailable", status=503))
    assert not is_staged_environment_transient_error(azure.RegistryError("missing", status=404))


def test_source_outage_uses_controller_retry_contract(registry, monkeypatch):
    from flash.envs.loading.staged import (
        StagedEnvironmentTransientError,
        resolve_environment_source,
    )

    def unavailable(*args, **kwargs):
        raise azure.RegistryError("unavailable", status=503)

    monkeypatch.setattr(packages, "resolve_source", unavailable)
    with pytest.raises(StagedEnvironmentTransientError):
        resolve_environment_source(SLUG)


@pytest.mark.parametrize(
    "reference",
    [
        SLUG,
        f"github:freesolo-co/environment-hub@main:{SLUG}/environment.py",
        f"https://github.com/freesolo-co/environment-hub/blob/main/{SLUG}/environment.py",
    ],
)
def test_legacy_managed_aliases_use_azure_without_mode_flag(
    registry, monkeypatch, tmp_path, reference
):
    from flash.envs.loading import loader

    monkeypatch.setattr(loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(loader, "_urlopen", lambda *a, **kw: pytest.fail("called GitHub"))
    registry.publish(SLUG, archive())
    assert loader._resolve_environment_reference(reference).endswith("environment.py")


def test_staging_ignores_poisoned_executable_cache(registry, monkeypatch, tmp_path):
    import shutil

    from flash.envs.loading import loader, staged

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "environment.py").write_text("POISON = True\n")
    monkeypatch.setattr(loader, "_CACHE_ROOT", cache)
    monkeypatch.setattr(
        packages, "cached_entrypoint", lambda *a: pytest.fail("staging read executable cache")
    )
    registry.publish(SLUG, archive())
    source = staged.resolve_environment_source(SLUG)
    try:
        assert source.root != cache
        assert (source.root / source.entrypoint).read_text() == "VALUE = 1\n"
    finally:
        shutil.rmtree(source.staging_root)
