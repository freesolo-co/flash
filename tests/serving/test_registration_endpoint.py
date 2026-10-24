"""A durable checkpoint binding needs its serving endpoint before ready promotion."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from flash.serving.deepinfer.contract.config import MODEL_ID as DEEPSEEK_MODEL_ID
from flash.serving.src.http import adapter_routes
from flash.serving.src.http.router import build_offline_serving_app
from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.schemas import ImmutableCheckpointRegistration
from tests.serving.checkpoint_fixtures import checkpoint_registration_payload
from tests.serving.test_router import QWEN, FakePool


@pytest.mark.parametrize("base_model", [DEEPSEEK_MODEL_ID, "example/unsupported"])
def test_unsupported_lora_registration_has_no_side_effects(monkeypatch, base_model):
    payload = checkpoint_registration_payload("run-unsupported", base_model)
    persist = AsyncMock()
    monkeypatch.setattr(adapter_routes, "persist_checkpoint", persist)
    pool = FakePool()
    router = AdapterRouter()
    app = build_offline_serving_app(pool, router, internal_key="test-internal-key")
    with TestClient(app) as client:
        response = client.post(
            "/adapters",
            json=payload,
            headers={
                "X-Freesolo-Internal-Key": "test-internal-key",
                "X-Freesolo-Org-Id": payload["org_id"],
            },
        )
    assert response.status_code == 400
    assert base_model in response.json()["detail"]
    persist.assert_not_called()
    assert router.get(payload["model_id"], org_id=payload["org_id"]) is None
    assert pool.registered == []


def test_registration_persists_router_endpoint_before_ready(monkeypatch):
    payload = checkpoint_registration_payload("run-1", QWEN, thinking=False)
    # The real binding RPC deliberately returns only immutable identity, with no URL.
    stored = (
        ImmutableCheckpointRegistration.model_validate(payload)
        .to_record()
        .model_copy(update={"url": None, "updated_at": "2026-10-19T00:00:00+00:00"})
    )
    updates = []

    async def persist(_record):
        return stored

    async def replace(record, *, expected_updated_at):
        nonlocal stored
        assert expected_updated_at == stored.updated_at
        # Match hosted_checkpoint_ready_url_check from the paired database migration.
        assert record.status != "ready" or (record.url and record.url.strip())
        updates.append(record)
        stored = record.model_copy(update={"updated_at": f"2026-10-19T00:00:0{len(updates)}+00:00"})
        return stored

    monkeypatch.setattr(adapter_routes, "persist_checkpoint", persist)
    monkeypatch.setattr(adapter_routes, "_replace_stored_cas", replace)
    pool = FakePool()
    router = AdapterRouter()
    app = build_offline_serving_app(pool, router, internal_key="test-internal-key")
    with TestClient(app, base_url="https://serving.example.test") as client:
        response = client.post(
            "/adapters",
            json=payload,
            headers={
                "X-Freesolo-Internal-Key": "test-internal-key",
                "X-Freesolo-Org-Id": payload["org_id"],
            },
        )
    assert response.status_code == 200
    assert [record.status for record in updates] == ["disabled", "ready"]
    assert all(
        record.url == "https://serving.example.test/v1/chat/completions" for record in updates
    )
    assert stored.status == "ready"
    assert router.get(stored.model_id, org_id=stored.org_id).url == stored.url
    assert pool.registered == [(QWEN, stored.model_id)]
