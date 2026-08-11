"""Control-plane API for listing published environments.

Regression coverage for the gap where the CLI had no server-side list to call, so `flash env list`
enumerated local scaffold directories only and printed "no environments yet" after a *successful*
publish — which reads as "the publish silently failed" and invites re-pushing under new names.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

_USER_TOKEN = "fslo-user-env-list"
_INTERNAL_TOKEN = "fslo-internal-test"


def _bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _identity_for_token(token: str) -> dict[str, str]:
    if token != _USER_TOKEN:
        return {}
    return {
        "email": "env-list@example.com",
        "key_prefix": "fslo_test",
        "org_id": "org-env-list",
        "org_slug": "acme",
    }


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test,rp-test-2")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", _INTERNAL_TOKEN)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")

    import flash.providers.runpod.auth as runpod_keys
    import flash.runner as runner
    import flash.server.platform.auth as auth_mod
    import flash.server.platform.db as db_mod

    runpod_keys.reset()
    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))

    import flash.providers as providers_mod
    import flash.server.app as app_mod
    import flash.server.domain.run_registry as run_registry

    importlib.reload(app_mod)
    monkeypatch.setattr(providers_mod, "configured_providers", list, raising=False)
    monkeypatch.setattr(run_registry, "_post", lambda *a, **k: False, raising=False)
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token == _USER_TOKEN)
    monkeypatch.setattr(auth_mod, "_cached_identity", _identity_for_token)
    with TestClient(app_mod.create_app()) as client:
        yield client


def test_list_envs_endpoint_exists_and_returns_published_slugs(api, monkeypatch):
    """The route the CLI needs: GET /v1/envs answers with the org's published environments."""
    import flash.server.domain.envs as envs_mod

    seen: dict = {}

    def fake_list(*, key):
        seen.update(key=key)
        return ["acme/beta", "acme/my-env"]

    monkeypatch.setattr(envs_mod, "list_namespace_slugs", fake_list)

    resp = api.get("/v1/envs", headers=_bearer(_USER_TOKEN))

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"environments": [{"id": "acme/beta"}, {"id": "acme/my-env"}]}
    assert seen["key"]["org_slug"] == "acme"


def test_list_envs_endpoint_requires_authentication(api, monkeypatch):
    import flash.server.domain.envs as envs_mod

    monkeypatch.setattr(
        envs_mod, "list_namespace_slugs", lambda **_k: pytest.fail("must not reach the hub")
    )

    assert api.get("/v1/envs").status_code in (401, 403)


def test_list_envs_endpoint_reports_empty_namespace_as_empty_list(api, monkeypatch):
    import flash.server.domain.envs as envs_mod

    monkeypatch.setattr(envs_mod, "list_namespace_slugs", lambda **_k: [])

    resp = api.get("/v1/envs", headers=_bearer(_USER_TOKEN))

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"environments": []}


def test_list_envs_endpoint_propagates_hub_failure_status(api, monkeypatch):
    """A broken hub read must NOT surface as an empty list — that is the bug being fixed."""
    import flash.server.domain.envs as envs_mod

    def boom(**_kwargs):
        raise envs_mod.EnvPublishError("hub is unreachable", status=502)

    monkeypatch.setattr(envs_mod, "list_namespace_slugs", boom)

    resp = api.get("/v1/envs", headers=_bearer(_USER_TOKEN))

    assert resp.status_code == 502, resp.text
    assert "hub is unreachable" in resp.text


def test_client_parses_the_endpoint_payload_into_ids(monkeypatch):
    """The client turns the route's response into the id list the CLI prints."""
    from flash.client.http import ApiClient

    client = ApiClient(api_url="https://plane.example", api_key="k")
    monkeypatch.setattr(
        ApiClient,
        "_request",
        lambda self, method, path, **kw: {"environments": [{"id": "acme/a"}, {"id": "acme/b"}]},
    )

    assert client.list_envs() == ["acme/a", "acme/b"]


@pytest.mark.parametrize(
    "payload",
    [
        {"environments": "acme/a"},
        {},
        {"environments": [{"name": "acme/a"}]},
        {"environments": [{"id": ""}]},
    ],
)
def test_client_rejects_a_malformed_payload(monkeypatch, payload):
    """A malformed response must raise, not degrade into a short or empty list."""
    from flash.client.http import ApiClient, ClientError

    client = ApiClient(api_url="https://plane.example", api_key="k")
    monkeypatch.setattr(ApiClient, "_request", lambda self, method, path, **kw: payload)

    with pytest.raises(ClientError):
        client.list_envs()


def test_list_route_does_not_shadow_the_package_route(api, monkeypatch):
    """`GET /v1/envs` and `GET /v1/envs/{id}/package` must stay distinct after adding the list."""
    import flash.server.domain.envs as envs_mod

    monkeypatch.setattr(
        envs_mod, "list_namespace_slugs", lambda **_k: pytest.fail("package route hit the list")
    )
    monkeypatch.setattr(envs_mod, "download_package", lambda **_k: b"package-bytes")

    resp = api.get("/v1/envs/acme/my-env/package", headers=_bearer(_USER_TOKEN))

    assert resp.status_code == 200, resp.text
    assert resp.content == b"package-bytes"
