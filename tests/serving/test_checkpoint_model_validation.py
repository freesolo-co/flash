"""Request-shape validation for Flash checkpoint identifiers on OpenAI chat serving."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from flash.serving.src.http.router import AdapterRouter
from flash.serving.src.http.router import build_offline_serving_app as build_serving_app
from flash.serving.src.io.schemas import AdapterRecord

QWEN = "Qwen/Qwen3.5-9B"
INTERNAL_KEY = "fs-internal"
CHECKPOINT_MODEL = "flash-1783788692-948932a3/step-32"
BASE_RUN_ID = "flash-1783788692-948932a3"


def _rec(adapter_id: str, *, serve_base_model: bool = False) -> AdapterRecord:
    run_id, checkpoint = adapter_id.split("/", 1)
    checkpoint_step = None if checkpoint == "final" else int(checkpoint.removeprefix("step-"))
    return AdapterRecord.model_validate(
        {
            "adapter_id": adapter_id,
            "repo_id": QWEN if serve_base_model else f"org/{run_id}",
            "base_model": QWEN,
            "org_id": None if serve_base_model else "org-a",
            "status": "ready",
            "thinking": True,
            "serve_base_model": serve_base_model,
            "checkpoint": None if serve_base_model else adapter_id,
            "run_id": None if serve_base_model else run_id,
            "checkpoint_step": None if serve_base_model else checkpoint_step,
            "artifact_revision": None if serve_base_model else "a" * 40,
            "artifact_digest": None if serve_base_model else "b" * 64,
            "artifact_fingerprint": None if serve_base_model else "c" * 64,
            "lora_rank": None if serve_base_model else 16,
        }
    )


class FakePool:
    def __init__(self) -> None:
        self.generated: list[str] = []

    async def generate(self, base_model, payload, record, *, expected_checkpoint=None):
        self.generated.append(payload.adapter_id)
        return {
            "text": "hi",
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "checkpoint": "",
        }

    async def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
        self.generated.append(payload.adapter_id)
        yield {"type": "ready", "checkpoint": ""}
        yield {"type": "final", "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}

    async def register(self, base_model, record) -> None:  # pragma: no cover
        pass

    async def unregister(
        self, base_model, adapter_id, expected_generation=None
    ) -> None:  # pragma: no cover
        pass


class FakeAuthorizer:
    def __init__(self, *, error: HTTPException | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.error = error

    async def __call__(self, token: str, adapter_id: str, scope: dict | None = None) -> str:
        self.calls.append((token, adapter_id))
        if self.error is not None:
            raise self.error
        return "caller-org"


def _client(records: list[AdapterRecord], authorizer: FakeAuthorizer):
    pool = FakePool()
    app = build_serving_app(
        pool,
        AdapterRouter(records),
        internal_key=INTERNAL_KEY,
        chat_authorizer=authorizer,
    )
    return TestClient(app), pool


def _chat(client: TestClient, model: str, *, stream: bool = False, headers=None):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        },
        headers=headers or {"Authorization": "Bearer user-key"},
    )


@pytest.mark.parametrize("registered", [False, True])
def test_generated_checkpoint_reference_requires_authorization_first(registered: bool) -> None:
    auth = FakeAuthorizer(error=HTTPException(status.HTTP_403_FORBIDDEN, "hidden"))
    records = [_rec(CHECKPOINT_MODEL)] if registered else []
    client, pool = _client(records, auth)

    response = _chat(client, CHECKPOINT_MODEL)

    assert response.status_code == 403
    assert auth.calls == [("user-key", CHECKPOINT_MODEL)]
    assert pool.generated == []


def test_checkpoint_reference_whitespace_is_rejected_after_authorization() -> None:
    auth = FakeAuthorizer()
    client, pool = _client([], auth)

    response = _chat(client, f"  {CHECKPOINT_MODEL}\n")

    assert response.status_code == 422
    assert response.json() == {"detail": "model must be required"}
    assert auth.calls == [("user-key", CHECKPOINT_MODEL)]
    assert pool.generated == []


def test_unknown_checkpoint_reference_is_404_for_internal_caller() -> None:
    auth = FakeAuthorizer()
    client, pool = _client([], auth)

    response = _chat(
        client,
        CHECKPOINT_MODEL,
        headers={
            "X-Freesolo-Internal-Key": INTERNAL_KEY,
            "X-Freesolo-Org-Id": "org-a",
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": f"Unknown adapter id: {CHECKPOINT_MODEL}"}
    assert auth.calls == []
    assert pool.generated == []


def test_streaming_checkpoint_request_validates_messages_before_lookup() -> None:
    auth = FakeAuthorizer()
    client, pool = _client([], auth)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": CHECKPOINT_MODEL, "messages": [], "stream": True},
        headers={"Authorization": "Bearer user-key"},
    ) as response:
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/json")
        response.read()
        assert response.json() == {"detail": "messages must be a nonempty array of objects"}

    assert auth.calls == [("user-key", CHECKPOINT_MODEL)]
    assert pool.generated == []


@pytest.mark.parametrize(
    "model",
    [
        "flash-1783788692-948932A3/step-32",
        "flash-1783788692-948932a/step-32",
        "flash-1783788692-948932a30/step-32",
        "flash-1783788692-948932a3/steps-32",
        "flash-1783788692-948932a3/step-32/extra",
        "flash-123456789012345678901-948932a3/step-32",
        "flash-1783788692-948932a3/step-123456789012345678901",
    ],
)
def test_near_misses_keep_generic_authorization_behavior(model: str) -> None:
    auth = FakeAuthorizer(error=HTTPException(status.HTTP_403_FORBIDDEN, "hidden"))
    client, pool = _client([], auth)

    response = _chat(client, model)

    assert response.status_code == 403
    assert auth.calls == [("user-key", model)]
    assert pool.generated == []


@pytest.mark.parametrize(
    "model",
    [
        "flash-١٧٨٣٧٨٨٦٩٢-948932a3/step-32",
        "flash-1783788692-948932a3/step-٣٢",
    ],
)
def test_unicode_digits_are_not_generated_checkpoint_identifiers(model: str) -> None:
    auth = FakeAuthorizer(error=HTTPException(status.HTTP_403_FORBIDDEN, "hidden"))
    client, pool = _client([], auth)

    response = _chat(client, model)

    assert response.status_code == 403
    assert auth.calls == [("user-key", model)]
    assert pool.generated == []


def test_arbitrary_unknown_id_keeps_auth_non_enumeration_then_404_behavior() -> None:
    denied_auth = FakeAuthorizer(error=HTTPException(status.HTTP_403_FORBIDDEN, "hidden"))
    denied_client, denied_pool = _client([], denied_auth)

    denied = _chat(denied_client, "unknown/step-24")

    assert denied.status_code == 403
    assert denied_auth.calls == [("user-key", "unknown/step-24")]
    assert denied_pool.generated == []

    allowed_auth = FakeAuthorizer()
    allowed_client, allowed_pool = _client([], allowed_auth)

    missing = _chat(allowed_client, "unknown/step-24")

    assert missing.status_code == 404
    assert allowed_auth.calls == [("user-key", "unknown/step-24")]
    assert allowed_pool.generated == []
