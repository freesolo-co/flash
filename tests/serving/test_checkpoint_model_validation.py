"""Request-shape validation for Flash checkpoint identifiers on OpenAI chat serving."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from flash.serving.src.router import AdapterRouter
from flash.serving.src.router import build_offline_serving_app as build_serving_app
from flash.serving.src.schemas import AdapterRecord

QWEN = "Qwen/Qwen3.5-0.8B"
INTERNAL_KEY = "fs-internal"
CHECKPOINT_MODEL = "flash-1783788692-948932a3/step-32"
BASE_RUN_ID = "flash-1783788692-948932a3"
DETAIL = (
    "This is a checkpoint identifier, not a serving model identifier. "
    f"Deploy it first or use model {BASE_RUN_ID}."
)


def _rec(adapter_id: str, *, serve_base_model: bool = False) -> AdapterRecord:
    return AdapterRecord.model_validate(
        {
            "adapter_id": adapter_id,
            "repo_id": adapter_id if serve_base_model else f"org/{adapter_id}",
            "base_model": QWEN,
            "org_id": None if serve_base_model else "org-a",
            "status": "ready",
            "thinking": True,
            "serve_base_model": serve_base_model,
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

    async def __call__(self, token: str, adapter_id: str) -> str:
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


def test_generated_checkpoint_reference_is_normalized_before_validation() -> None:
    auth = FakeAuthorizer()
    client, pool = _client([], auth)

    response = _chat(client, f"  {CHECKPOINT_MODEL}\n")

    assert response.status_code == 400
    assert response.json() == {"detail": DETAIL}
    assert auth.calls == [("user-key", CHECKPOINT_MODEL)]
    assert pool.generated == []


def test_generated_checkpoint_reference_is_400_for_internal_caller() -> None:
    auth = FakeAuthorizer()
    client, pool = _client([], auth)

    response = _chat(
        client,
        CHECKPOINT_MODEL,
        headers={"X-Freesolo-Internal-Key": INTERNAL_KEY},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": DETAIL}
    assert auth.calls == []
    assert pool.generated == []


def test_generated_checkpoint_reference_streaming_request_is_plain_400() -> None:
    auth = FakeAuthorizer()
    client, pool = _client([], auth)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": CHECKPOINT_MODEL, "messages": [], "stream": True},
        headers={"Authorization": "Bearer user-key"},
    ) as response:
        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        response.read()
        assert response.json() == {"detail": DETAIL}

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
