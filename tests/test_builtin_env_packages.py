from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import urllib.error

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException

from flash.envs import loader
from flash.server import environment_registry

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"


def _package(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _builtin_payload(package: bytes) -> dict[str, str]:
    return {
        "sourceKind": "builtin",
        "packageBase64": base64.b64encode(package).decode("ascii"),
        "packageSha256": hashlib.sha256(package).hexdigest(),
    }


def _resolve_source(monkeypatch, payload: object) -> dict[str, str]:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://backend.test")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda _request, timeout=None: Response(),
    )
    return environment_registry.resolve_environment_package_source(
        slug="acme/example",
        project_id=_PROJECT_ID,
        key={"org_id": "org-test"},
    )


def test_package_source_accepts_exact_hub_response(monkeypatch):
    assert _resolve_source(monkeypatch, {"sourceKind": "hub"}) == {"source_kind": "hub"}


def test_package_source_preserves_project_mismatch_status_without_leak(monkeypatch):
    error = urllib.error.HTTPError(
        "https://backend.test/api/flash/environments/package/internal",
        409,
        "conflict",
        {},
        io.BytesIO(b'{"detail":"flash environment belongs to another project"}'),
    )
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setattr(
        environment_registry.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(HTTPException) as exc_info:
        environment_registry.resolve_environment_package_source(
            slug="acme/example",
            project_id=_PROJECT_ID,
            key={"org_id": "org-test"},
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "flash environment belongs to another project"


def test_package_source_accepts_and_verifies_builtin_response(monkeypatch):
    package = _package({"environment.py": b"# example\n"})
    source = _resolve_source(monkeypatch, _builtin_payload(package))

    assert source == {
        "source_kind": "builtin",
        "package_base64": base64.b64encode(package).decode("ascii"),
        "package_sha256": hashlib.sha256(package).hexdigest(),
    }


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"sourceKind": "unknown"},
        {"sourceKind": "hub", "extra": True},
        {"sourceKind": "builtin"},
        {
            "sourceKind": "builtin",
            "packageBase64": "QQ==",
            "packageSha256": "a" * 64,
            "extra": True,
        },
    ],
)
def test_package_source_rejects_malformed_shape(monkeypatch, payload):
    with pytest.raises(HTTPException) as exc_info:
        _resolve_source(monkeypatch, payload)
    assert exc_info.value.status_code == 502


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64, 123])
def test_package_source_rejects_bad_digest(monkeypatch, digest):
    payload = {
        "sourceKind": "builtin",
        "packageBase64": "QQ==",
        "packageSha256": digest,
    }
    with pytest.raises(HTTPException, match="sha256 is invalid"):
        _resolve_source(monkeypatch, payload)


def test_package_source_rejects_oversize_payload(monkeypatch):
    monkeypatch.setattr(loader, "_MAX_BUILTIN_PACKAGE_BASE64_CHARS", 4)
    payload = _builtin_payload(b"1234")
    with pytest.raises(HTTPException, match="too large"):
        _resolve_source(monkeypatch, payload)


def test_package_source_rejects_invalid_base64(monkeypatch):
    payload = {
        "sourceKind": "builtin",
        "packageBase64": "!!!!",
        "packageSha256": hashlib.sha256(b"").hexdigest(),
    }
    with pytest.raises(HTTPException, match="not valid base64"):
        _resolve_source(monkeypatch, payload)


def test_package_source_rejects_carrier_sha_mismatch(monkeypatch):
    payload = {
        "sourceKind": "builtin",
        "packageBase64": base64.b64encode(b"package").decode("ascii"),
        "packageSha256": hashlib.sha256(b"other").hexdigest(),
    }
    with pytest.raises(HTTPException, match="sha256 mismatch"):
        _resolve_source(monkeypatch, payload)


def test_builtin_loader_bypasses_all_github_paths_and_token_reads(monkeypatch, tmp_path):
    package = _package({"environment.py": b"# example\n"})
    payload = _builtin_payload(package)
    monkeypatch.setattr(loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(
        loader,
        "managed_slug_to_github_ref",
        lambda _value: pytest.fail("built-in package must not map a managed slug to github"),
    )
    monkeypatch.setattr(
        loader,
        "_resolve_github_environment_file",
        lambda *_args, **_kwargs: pytest.fail("built-in package must not access github"),
    )
    real_get = loader.os.environ.get

    def guarded_get(name, *args):
        if name == "GITHUB_TOKEN":
            pytest.fail("built-in package must not inspect GITHUB_TOKEN")
        return real_get(name, *args)

    monkeypatch.setattr(loader.os.environ, "get", guarded_get)

    resolved = loader._resolve_environment_reference(
        "acme/example",
        None,
        "builtin",
        payload["packageBase64"],
        payload["packageSha256"],
    )

    assert loader.Path(resolved).read_bytes() == b"# example\n"
    assert payload["packageSha256"] in resolved


def test_builtin_loader_rechecks_sha_immediately_before_extraction(monkeypatch, tmp_path):
    package = _package({"environment.py": b"# example\n"})
    monkeypatch.setattr(loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(loader, "_decode_builtin_environment_package", lambda *_args: package)

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        loader._resolve_builtin_environment_package("acme/example", "ignored", "a" * 64)


def test_builtin_loader_requires_environment_entrypoint(monkeypatch, tmp_path):
    package = _package({"README.md": b"missing entrypoint\n"})
    payload = _builtin_payload(package)
    monkeypatch.setattr(loader, "_CACHE_ROOT", tmp_path / "cache")

    with pytest.raises(FileNotFoundError, match=r"environment\.py"):
        loader._resolve_builtin_environment_package(
            "acme/example", payload["packageBase64"], payload["packageSha256"]
        )


def test_builtin_loader_enforces_archive_member_limit(monkeypatch, tmp_path):
    package = _package({"environment.py": b"# example\n", "extra.txt": b"x"})
    payload = _builtin_payload(package)
    monkeypatch.setattr(loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(loader, "_MAX_ARCHIVE_MEMBERS", 1)

    with pytest.raises(RuntimeError, match="too many members"):
        loader._resolve_builtin_environment_package(
            "acme/example", payload["packageBase64"], payload["packageSha256"]
        )


def test_builtin_loader_enforces_uncompressed_archive_limit(monkeypatch, tmp_path):
    package = _package({"environment.py": b"# example\n"})
    payload = _builtin_payload(package)
    monkeypatch.setattr(loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(loader, "_MAX_ARCHIVE_BYTES", 4)

    with pytest.raises(RuntimeError, match="too large uncompressed"):
        loader._resolve_builtin_environment_package(
            "acme/example", payload["packageBase64"], payload["packageSha256"]
        )
