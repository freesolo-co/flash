"""GC reads durable bindings rather than this router's ready-only inventory."""

import pytest
from fastapi.testclient import TestClient

from flash.serving.src.http import adapter_routes
from flash.serving.src.http.router import build_offline_serving_app
from flash.serving.src.http.routing import AdapterRouter
from tests.serving.test_persistence import _record
from tests.serving.test_router import FakePool

HEADERS = {"X-Freesolo-Internal-Key": "test-internal-key"}


@pytest.mark.parametrize("generation", [None, "loading-generation"])
def test_protections_include_disabled_bindings_absent_from_router(monkeypatch, generation):
    record = _record(status="disabled", deployment_generation=generation)
    monkeypatch.setattr(adapter_routes, "load_checkpoint_bindings", lambda _: [record])
    router = AdapterRouter()
    with TestClient(
        build_offline_serving_app(FakePool(), router, internal_key="test-internal-key")
    ) as client:
        assert client.get("/adapters", headers=HEADERS).json()["adapters"] == []
        response = client.get("/artifact-protections", headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "bindings": [{"repo_id": record.repo_id, "subfolder": record.subfolder}],
    }
    assert router.ready_adapters() == []


def test_protections_require_internal_key_before_reading_storage(monkeypatch):
    def unexpected(_):
        pytest.fail("unauthenticated request read binding storage")

    monkeypatch.setattr(adapter_routes, "load_checkpoint_bindings", unexpected)
    with TestClient(
        build_offline_serving_app(FakePool(), AdapterRouter(), internal_key="test-internal-key")
    ) as client:
        assert client.get("/artifact-protections").status_code == 401


def test_protections_fail_closed_on_storage_errors(monkeypatch):
    def unavailable(_):
        raise RuntimeError("private storage details")

    monkeypatch.setattr(adapter_routes, "load_checkpoint_bindings", unavailable)
    with TestClient(
        build_offline_serving_app(FakePool(), AdapterRouter(), internal_key="test-internal-key")
    ) as client:
        response = client.get("/artifact-protections", headers=HEADERS)
    assert response.status_code == 503
    assert "private storage details" not in response.text
