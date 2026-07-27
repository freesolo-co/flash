"""Control-plane API for downloading managed environment packages."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

_USER_TOKEN = "fslo-user-env-download"
_PROJECT_ID = "11111111-1111-4111-8111-111111111111"


def _bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _identity_for_token(token: str) -> dict[str, str]:
    if token != _USER_TOKEN:
        return {}
    return {
        "email": "env-download@example.com",
        "key_prefix": "fslo_test",
        "org_id": "org-env-download",
        "org_slug": "acme",
    }


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test,rp-test-2")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("FLASH_DEPLOY_SYNC", "1")

    import flash.providers.runpod.keys as runpod_keys
    import flash.runner as runner
    import flash.server.auth as auth_mod
    import flash.server.db as db_mod

    runpod_keys.reset()
    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))

    import flash.providers as providers_mod
    import flash.providers.runpod.train.endpoints as rp_endpoints
    import flash.server.app as app_mod
    import flash.server.environment_registry as environment_registry
    import flash.server.projects as projects
    import flash.server.run_registry as run_registry

    importlib.reload(app_mod)
    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [], raising=False)
    monkeypatch.setattr(run_registry, "_post", lambda *a, **k: False, raising=False)
    monkeypatch.setattr(
        projects,
        "require_project_access",
        lambda *, project_id, **_kwargs: project_id,
    )
    monkeypatch.setattr(
        environment_registry,
        "resolve_environment_package_source",
        lambda **_kwargs: {"source_kind": "hub"},
    )
    monkeypatch.setattr(
        rp_endpoints, "reconcile_endpoint_slots", lambda *a, **k: None, raising=False
    )
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token == _USER_TOKEN)
    monkeypatch.setattr(auth_mod, "_cached_identity", _identity_for_token)
    with TestClient(app_mod.create_app()) as client:
        yield client


def test_download_env_package_endpoint_returns_package(api, monkeypatch):
    import flash.server.envs as envs_mod

    seen: dict = {}

    def fake_download_package(*, slug, key):
        seen.update(slug=slug, key=key)
        return b"package-bytes"

    monkeypatch.setattr(envs_mod, "download_package", fake_download_package)

    resp = api.get(
        "/v1/envs/acme/my-env/package",
        headers={**_bearer(_USER_TOKEN), "X-Freesolo-Project-Id": _PROJECT_ID},
    )

    assert resp.status_code == 200, resp.text
    assert resp.content == b"package-bytes"
    assert resp.headers["content-type"] == "application/gzip"
    assert seen["slug"] == "acme/my-env"
    assert seen["key"]["org_slug"] == "acme"

    assert api.get("/v1/envs/acme/my-env/package").status_code in (401, 403)


def test_download_builtin_env_package_never_calls_github(api, monkeypatch):
    import base64
    import hashlib

    import flash.server.environment_registry as registry
    import flash.server.envs as envs_mod

    package = b"builtin-package"
    monkeypatch.setattr(
        registry,
        "resolve_environment_package_source",
        lambda **_kwargs: {
            "source_kind": "builtin",
            "package_base64": base64.b64encode(package).decode("ascii"),
            "package_sha256": hashlib.sha256(package).hexdigest(),
        },
    )
    monkeypatch.setattr(
        envs_mod,
        "download_package",
        lambda **_kwargs: pytest.fail("built-in download must not call github storage"),
    )

    response = api.get(
        "/v1/envs/acme/example/package",
        headers={**_bearer(_USER_TOKEN), "X-Freesolo-Project-Id": _PROJECT_ID},
    )

    assert response.status_code == 200, response.text
    assert response.content == package


def test_download_env_package_requires_project_header(api):
    response = api.get("/v1/envs/acme/my-env/package", headers=_bearer(_USER_TOKEN))

    assert response.status_code == 400
    assert "X-Freesolo-Project-Id is required" in response.text


def test_download_env_package_endpoint_rejects_non_canonical_id(api, monkeypatch):
    import flash.server.envs as envs_mod

    monkeypatch.setattr(
        envs_mod, "download_package", lambda **_k: pytest.fail("storage must not be touched")
    )

    resp = api.get(
        "/v1/envs/Acme/My-Env/package",
        headers={**_bearer(_USER_TOKEN), "X-Freesolo-Project-Id": _PROJECT_ID},
    )

    assert resp.status_code == 400, resp.text
