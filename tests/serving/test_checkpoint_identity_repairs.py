from __future__ import annotations

import asyncio
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient

from flash.cli.commands.env.testing.eval import _resolve_evaluation_target
from flash.runner.lifecycle.state import RunStatus
from flash.runner.results import verified_revisions
from flash.runner.supervise import transitions
from flash.schema import format_checkpoint_ref
from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serve.contract.responses import matches_revision_identity
from flash.server.routes import serving_smoke
from flash.server.routes.serving_revisions import _authorized_chat_checkpoint
from flash.serving.src.engine.support import active_checkpoint_ref
from flash.serving.src.http.router import build_offline_serving_app
from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.schemas import (
    AdapterRecord,
    ImmutableCheckpointRegistration,
    internal_adapter_payload,
)
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


@pytest.mark.parametrize("alias", ["checkpoint_id", "checkpoint"])
def test_binding_fingerprint_rejects_identity_aliases(alias: str) -> None:
    payload = internal_adapter_payload(_record("org-a"))
    payload[alias] = payload.pop("adapter_id")

    with pytest.raises(ValueError, match="inconsistent permanent identity"):
        immutable_binding_fingerprint(payload)


def test_binding_fingerprint_rejects_malformed_identity_types() -> None:
    payload = internal_adapter_payload(_record("org-a"))

    with pytest.raises(ValueError, match="requires org_id"):
        immutable_binding_fingerprint({**payload, "org_id": None})
    with pytest.raises(ValueError, match="requires run_id"):
        immutable_binding_fingerprint({**payload, "run_id": None})
    with pytest.raises(ValueError, match="invalid checkpoint_step"):
        immutable_binding_fingerprint({**payload, "checkpoint_step": True})


def test_active_checkpoint_uses_only_shared_permanent_identity_grammar() -> None:
    record = _record("org-a")
    assert active_checkpoint_ref(record) == "shared/final"


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

    target, run, error = _resolve_evaluation_target(
        SimpleNamespace(target=checkpoint, debug=False),
        _Client(),
        ("run-a", 20),
        RuntimeError,
        RuntimeError,
    )
    assert target == checkpoint
    assert run["verified_checkpoints"] == [checkpoint]
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

        async def post(self, _url, *, json, headers=None):
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


def test_serving_completion_imports_ready_states_and_control_app_constructs() -> None:
    from flash.server.asgi import app
    from flash.server.routes import serving_completion, serving_revisions

    assert serving_completion._DEPLOYMENT_READY_STATES is serving_revisions._DEPLOYMENT_READY_STATES
    assert {"ready"} == serving_revisions._DEPLOYMENT_READY_STATES
    assert app.create_app() is not None


def test_image_and_structured_smoke_paths_call_pure_answer_without_org(monkeypatch) -> None:
    result = {
        "choices": [{"message": {"content": "RED"}, "finish_reason": "stop"}],
        "freesolo": {"checkpoint_id": "run-a/final"},
        "_freesolo_headers": {"checkpoint_id": "run-a/final"},
    }
    calls: list[dict] = []

    def bounded(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(serving_smoke, "_bounded_smoke_chat", bounded)
    monkeypatch.setattr(
        serving_smoke, "_smoke_request_settings", lambda _spec: ({"json_object": True}, 8, None)
    )
    monkeypatch.setattr(serving_smoke, "_smoke_image_challenge", lambda _run_id: ("RED", []))
    monkeypatch.setattr(serving_smoke, "_validate_structured_smoke", lambda *_args, **_kwargs: None)

    output = serving_smoke._run_deployment_smoke(
        "run-a",
        SimpleNamespace(thinking=False),
        serving_model="run-a/final",
        expected_checkpoint="run-a/final",
        org_id="org-a",
        adapter_targets_images=True,
    )

    assert output["verify_turns"] == 2
    assert len(calls) == 2
    assert all(call["org_id"] == "org-a" for call in calls)


def _save_ready_status(tmp_path, monkeypatch, run_id: str, summary: str) -> tuple[str, str]:
    from flash.runner.lifecycle import state

    monkeypatch.setattr(state, "RUNS_DIR", str(tmp_path))
    sibling = f"{run_id}/step-20"
    state._save_status(
        RunStatus(
            run_id=run_id,
            state="deployed",
            spec={"run_id": run_id},
            deployment={"state": "ready", "checkpoint_id": summary, "verified_at": 1.0},
        )
    )
    for checkpoint_id in (summary, sibling):
        assert verified_revisions.add_verified_checkpoint(
            run_id,
            checkpoint_id,
            expected_generation=verified_revisions.verified_checkpoint_generation(run_id),
        )
    return summary, sibling


def test_revocation_failure_removes_only_current_chat_authority(tmp_path, monkeypatch) -> None:
    current, sibling = _save_ready_status(tmp_path, monkeypatch, "run-revoke", "run-revoke/final")

    failed = transitions.mark_deployment_revocation_failed("run-revoke", "backend unavailable")

    assert failed.deployment["state"] == "revocation_failed"
    assert verified_revisions.read_verified_checkpoints("run-revoke") == frozenset({sibling})
    with pytest.raises(Exception, match="has not passed"):
        _authorized_chat_checkpoint(
            "run-revoke",
            failed.deployment,
            current,
            set(verified_revisions.read_verified_checkpoints("run-revoke")),
        )
    assert (
        _authorized_chat_checkpoint("run-revoke", failed.deployment, sibling, {sibling}) == sibling
    )


def test_sibling_revocation_failure_removes_only_failed_checkpoint(tmp_path, monkeypatch) -> None:
    current, sibling = _save_ready_status(
        tmp_path, monkeypatch, "run-sibling-failure", "run-sibling-failure/final"
    )

    failed = transitions.mark_deployment_revocation_failed(
        "run-sibling-failure", "backend unavailable", checkpoint_id=sibling
    )

    assert failed.deployment["state"] == "ready"
    assert failed.deployment["checkpoint_id"] == current
    assert "error" not in failed.deployment
    assert verified_revisions.read_verified_checkpoints("run-sibling-failure") == frozenset(
        {current}
    )
    assert (
        _authorized_chat_checkpoint("run-sibling-failure", failed.deployment, current, {current})
        == current
    )
    retried = transitions.mark_undeployed("run-sibling-failure", sibling)
    assert retried.deployment == failed.deployment


def test_sibling_undeploy_preserves_deployed_summary_and_state(tmp_path, monkeypatch) -> None:
    current, sibling = _save_ready_status(tmp_path, monkeypatch, "run-sibling", "run-sibling/final")

    updated = transitions.mark_undeployed("run-sibling", sibling)

    assert updated.state == "deployed"
    assert updated.deployment == {
        "state": "ready",
        "checkpoint_id": current,
        "verified_at": 1.0,
    }
    assert verified_revisions.read_verified_checkpoints("run-sibling") == frozenset({current})


def test_current_undeploy_promotes_remaining_verified_sibling(tmp_path, monkeypatch) -> None:
    current, sibling = _save_ready_status(tmp_path, monkeypatch, "run-promote", "run-promote/final")

    updated = transitions.mark_undeployed("run-promote", current)

    assert updated.state == "deployed"
    assert updated.deployment == {
        "state": "ready",
        "checkpoint_id": sibling,
        "verified_at": 1.0,
    }
    assert verified_revisions.read_verified_checkpoints("run-promote") == frozenset({sibling})


def test_checkpoint_registration_rejects_bare_managed_selector() -> None:
    client = _client()
    response = client.post(
        "/adapters",
        json={
            "adapter_id": "run-a",
            "repo_id": "org/run-a",
            "base_model": "Qwen/Qwen3.5-9B",
            "org_id": "org-a",
            "checkpoint": "run-a",
            "thinking": False,
            "run_id": "run-a",
            "checkpoint_step": None,
            "artifact_revision": "a" * 40,
            "artifact_digest": "b" * 64,
            "artifact_fingerprint": "c" * 64,
            "lora_rank": 16,
        },
        headers=_headers("org-a"),
    )
    assert response.status_code == 422
