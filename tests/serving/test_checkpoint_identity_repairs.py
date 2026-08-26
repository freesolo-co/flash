from __future__ import annotations

import asyncio
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient

from flash.cli.commands.env.testing.eval import _resolve_evaluation_target
from flash.schema import format_checkpoint_ref
from flash.serve.contract.responses import matches_revision_identity
from flash.server.routes.serving_revisions import _authorized_chat_checkpoint
from flash.serving.src.engine.support import active_checkpoint_ref
from flash.serving.src.http.router import build_offline_serving_app
from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.schemas import (
    AdapterRecord,
    ImmutableCheckpointRegistration,
    internal_adapter_payload,
)
from flash.serving.src.store.identity import immutable_binding_fingerprint
from flash.serving.src.store.persistence import get_adapter, list_run_adapters
from flash.serving.src.store.settings import Settings


def _record(org_id: str, *, digest: str = "b" * 64) -> AdapterRecord:
    values = {
        "adapter_id": "shared/final",
        "repo_id": f"{org_id}/run",
        "base_model": "Qwen/Qwen3.5-9B",
        "org_id": org_id,
        "checkpoint": "shared/final",
        "thinking": False,
        "run_id": "shared",
        "checkpoint_step": None,
        "artifact_revision": "a" * 40,
        "artifact_digest": digest,
        "lora_rank": 16,
    }
    values["artifact_fingerprint"] = immutable_binding_fingerprint(values)
    return AdapterRecord.model_validate(values)


class _Pool:
    async def generate(self, _base_model, _payload, record, **_kwargs):
        return {
            "ok": True,
            "text": record.org_id,
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
            "cached_tokens_reported": True,
            "reasoning_tokens": 0,
            "inference_time_seconds": 0.01,
            "request_id": "fsgen-" + "0" * 32,
            "engine_replica_id": "test",
            "checkpoint": record.adapter_id,
            "lora_request_adapter": record.adapter_id,
        }

    async def stream_generate(self, *_args, **_kwargs):
        if False:
            yield None

    async def register(self, *_args, **_kwargs):
        return None

    async def unregister(self, *_args, **_kwargs):
        return None


def _client(*records: AdapterRecord) -> TestClient:
    by_key = {(record.org_id, record.adapter_id): record for record in records}
    app = build_offline_serving_app(
        _Pool(),
        AdapterRouter(list(records)),
        internal_key="internal",
        lookup_record=lambda org_id, checkpoint_id: by_key.get((org_id, checkpoint_id)),
    )
    return TestClient(app)


def _headers(org_id: str) -> dict[str, str]:
    return {
        "X-Freesolo-Internal-Key": "internal",
        "X-Freesolo-Org-Id": org_id,
    }


def test_exact_routes_capture_raw_and_encoded_slashes_and_absence_is_idempotent() -> None:
    record = _record("org-a")
    client = _client(record)
    for target in (record.adapter_id, quote(record.adapter_id, safe="")):
        response = client.get(f"/adapters/{target}", headers=_headers("org-a"))
        assert response.status_code == 200
        assert response.json()["adapter"]["adapter_id"] == record.adapter_id
        generated = client.post(
            f"/adapters/{target}/generate",
            json={"prompt": "hi"},
            headers=_headers("org-a"),
        )
        assert generated.status_code == 200
        assert generated.json()["text"] == "org-a"
    for _ in range(2):
        response = client.delete("/adapters/missing/final", headers=_headers("org-a"))
        assert response.status_code == 200
        assert response.json()["disabled_checkpoints"] == []


def test_same_public_checkpoint_isolated_by_organization() -> None:
    first = _record("org-a")
    second = _record("org-b", digest="c" * 64)
    router = AdapterRouter([first, second])
    assert router.resolve(first.adapter_id, org_id="org-a") == (first, first)
    assert router.resolve(second.adapter_id, org_id="org-b") == (second, second)
    assert router.resolve(first.adapter_id, org_id="org-c") is None


def test_persistence_queries_use_org_and_checkpoint_composite(monkeypatch) -> None:
    seen = []

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url, **kwargs):
            seen.append(kwargs["params"])
            return httpx.Response(200, json=[], request=httpx.Request("GET", url))

    monkeypatch.setattr(
        "flash.serving.src.store.persistence.httpx.Client",
        _Client,
    )
    settings = Settings(
        SUPABASE_URL="https://supabase.test",
        SUPABASE_SERVICE_ROLE_KEY="sb_secret_service_role",
    )
    assert get_adapter("org-a", "shared/final", settings) is None
    assert list_run_adapters("org-a", "shared", settings) == []
    assert seen[0]["org_id"] == "eq.org-a"
    assert seen[0]["checkpoint_id"] == "eq.shared/final"
    assert seen[1]["org_id"] == "eq.org-a"
    assert seen[1]["run_id"] == "eq.shared"


def test_artifact_digest_and_binding_fingerprint_are_distinct() -> None:
    record = _record("org-a")
    assert record.artifact_digest != record.artifact_fingerprint
    changed = record.model_copy(update={"base_model": "Qwen/Qwen3.5-4B"})
    assert immutable_binding_fingerprint(changed) != record.artifact_fingerprint
    payload = internal_adapter_payload(record)
    registration = {name: payload[name] for name in ImmutableCheckpointRegistration.model_fields}
    with pytest.raises(ValueError, match="immutable binding"):
        ImmutableCheckpointRegistration.model_validate(
            {**registration, "artifact_fingerprint": "d" * 64}
        )


def test_active_checkpoint_uses_only_shared_permanent_identity_grammar() -> None:
    record = _record("org-a")
    assert active_checkpoint_ref(record) == "shared/final"
    legacy = SimpleNamespace(
        adapter_id="shared",
        checkpoint=None,
        subfolder="checkpoints/step-20",
    )
    assert active_checkpoint_ref(legacy) == ""


def test_ambiguous_registration_matches_every_flat_immutable_field() -> None:
    record = _record("org-a")
    payload = {
        **record.model_dump(mode="json"),
        "org_id": record.org_id,
        "run_id": record.run_id,
        "checkpoint_step": record.checkpoint_step,
        "artifact_revision": record.artifact_revision,
        "artifact_digest": record.artifact_digest,
        "artifact_fingerprint": record.artifact_fingerprint,
        "lora_rank": record.lora_rank,
    }
    assert matches_revision_identity(payload, payload)
    assert not matches_revision_identity({**payload, "lora_rank": 32}, payload)


def test_verified_sibling_remains_authorized_while_latest_deployment_failed() -> None:
    checkpoint = format_checkpoint_ref("run-a", 20)
    assert (
        _authorized_chat_checkpoint(
            "run-a",
            {"state": "failed", "checkpoint_id": format_checkpoint_ref("run-a", 40)},
            checkpoint,
            {checkpoint},
        )
        == checkpoint
    )


def test_evaluation_accepts_verified_sibling_without_latest_deployment_equality() -> None:
    checkpoint = "run-a/step-20"

    class _Client:
        @staticmethod
        def get_run(_run_id):
            return {
                "verified_checkpoints": [checkpoint],
                "deployment": {"state": "failed", "checkpoint_id": "run-a/step-40"},
            }

    target, error = _resolve_evaluation_target(
        SimpleNamespace(target=checkpoint, debug=False),
        _Client(),
        ("run-a", 20),
        RuntimeError,
        RuntimeError,
    )
    assert target == checkpoint
    assert error is None


def test_hosted_authorizer_uses_freesolo_model_id_schema(monkeypatch) -> None:
    import httpx

    from flash.serving.app import modal_app

    seen = {}

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"orgId": "org-a"}

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        async def post(self, _url, *, json):
            seen.update(json)
            return _Response()

        async def aclose(self):
            return None

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    authorize = modal_app._build_chat_authorizer(
        SimpleNamespace(backend_url="https://backend.example", internal_key="internal")
    )
    assert asyncio.run(authorize("fs-key", "run-a/final")) == "org-a"
    assert seen == {"apiKey": "fs-key", "modelId": "run-a/final"}
