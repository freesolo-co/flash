"""Focused coverage for the server-side environment list endpoint."""

from __future__ import annotations

import pytest

from flash.client.http import (
    ENV_LIST_CLIENT_TIMEOUT_SECONDS,
    ENV_LIST_MAX_RESPONSE_BYTES,
    ApiClient,
    ClientError,
)
from flash.server.routes import envs as routes


def test_list_endpoint_returns_published_slugs(monkeypatch):
    import flash.server.domain.registry.envs as domain

    seen: dict = {}

    def fake_list(*, key):
        seen["key"] = key
        return ["acme/project/beta", "acme/project/my-env"]

    monkeypatch.setattr(domain, "list_namespace_slugs", fake_list)

    assert routes.list_envs(key={"org_slug": "acme"}) == {
        "environments": [{"id": "acme/project/beta"}, {"id": "acme/project/my-env"}]
    }
    assert seen["key"]["org_slug"] == "acme"


def test_list_endpoint_preserves_hub_failure_status(monkeypatch):
    import flash.server.domain.registry.envs as domain

    def fail(**_kwargs):
        raise domain.EnvPublishError("hub is unreachable", status=502)

    monkeypatch.setattr(domain, "list_namespace_slugs", fail)

    with pytest.raises(routes.HTTPException) as excinfo:
        routes.list_envs(key={"org_slug": "acme"})
    assert excinfo.value.status_code == 502


def test_client_parses_ids_and_uses_list_bounds(monkeypatch):
    client = ApiClient(api_url="https://plane.example", api_key="key")
    seen: dict = {}

    def fake_request(self, method, path, **kwargs):
        seen.update(method=method, path=path, **kwargs)
        return {"environments": [{"id": " acme/project/my-env "}]}

    monkeypatch.setattr(ApiClient, "_request", fake_request)

    assert client.list_envs() == ["acme/project/my-env"]
    assert seen == {
        "method": "GET",
        "path": "/v1/envs",
        "timeout": ENV_LIST_CLIENT_TIMEOUT_SECONDS,
        "max_bytes": ENV_LIST_MAX_RESPONSE_BYTES,
        "body_deadline": ENV_LIST_CLIENT_TIMEOUT_SECONDS,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"environments": "acme/project/my-env"},
        {"environments": [{"name": "acme/project/my-env"}]},
        {"environments": [{"id": ""}]},
    ],
)
def test_client_rejects_malformed_environment_lists(monkeypatch, payload):
    client = ApiClient(api_url="https://plane.example", api_key="key")
    monkeypatch.setattr(ApiClient, "_request", lambda *_args, **_kwargs: payload)

    with pytest.raises(ClientError):
        client.list_envs()


@pytest.mark.parametrize(
    "env_id", ["my-env", "acme/project/env/extra", "acme/../secrets", "acme/project/my env"]
)
def test_client_rejects_noncanonical_environment_ids(monkeypatch, env_id):
    """A nonblank id that the managed parser would reject must not be advertised as usable.

    These ids are printed for the user to paste into ``[environment]``, so accepting one here means
    advertising a value that fails later at submit -- with nothing pointing back at the list that
    supplied it.
    """
    client = ApiClient(api_url="https://plane.example", api_key="key")
    monkeypatch.setattr(
        ApiClient, "_request", lambda *_args, **_kwargs: {"environments": [{"id": env_id}]}
    )

    with pytest.raises(ClientError, match="unusable environment id"):
        client.list_envs()


def test_client_accepts_a_canonical_environment_id(monkeypatch):
    """The guard above must not reject the ids the hub actually publishes."""
    client = ApiClient(api_url="https://plane.example", api_key="key")
    monkeypatch.setattr(
        ApiClient,
        "_request",
        lambda *_args, **_kwargs: {"environments": [{"id": "acme/project/my-env"}]},
    )

    assert client.list_envs() == ["acme/project/my-env"]


def test_list_route_does_not_shadow_package_route():
    paths = {route.path for route in routes.router.routes}
    assert "/v1/envs" in paths
    assert "/v1/envs/{env_id:path}/package" in paths
