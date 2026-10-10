"""External chat auth on the inference endpoints (always enforced).

The OpenAI/inference surface always requires a Freesolo API key whose org owns the adapter (the
backend authorizes via the injected ``chat_authorizer``). Trusted server-to-server callers bypass
with the internal key. Offline — a fake engine pool and a fake authorizer stand in for the real ones.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from flash.serving.src.http.router import AdapterRouter
from flash.serving.src.http.router import build_offline_serving_app as build_serving_app
from flash.serving.src.io.schemas import AdapterRecord
from tests.serving.checkpoint_fixtures import checkpoint_record
from tests.serving.conftest import attest

QWEN = "Qwen/Qwen3.5-9B"
INTERNAL_KEY = "fs-internal"


def _rec(run_id: str, *, org_id: str = "org-A") -> AdapterRecord:
    return checkpoint_record(run_id, QWEN, org_id=org_id, thinking=True)


def _id(run_id: str = "qa") -> str:
    return f"{run_id}/final"


class FakePool:
    async def generate(
        self,
        base_model: str,
        payload,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict:
        return attest(
            record,
            {
                "text": "hi",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "checkpoint": record.checkpoint or "",
            },
        )

    async def stream_generate(
        self,
        base_model: str,
        payload,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ):
        yield {"type": "ready", "checkpoint": record.checkpoint or ""}
        yield {"type": "delta", "text": "hi"}
        yield {"type": "final", "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}

    async def register(self, base_model: str, record) -> None:  # pragma: no cover - unused here
        pass

    async def unregister(
        self,
        base_model: str,
        org_id: str,
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> None:  # pragma: no cover
        pass


class FakeAuthorizer:
    """Records (token, adapter_id) calls; optionally raises to simulate a denied request."""

    def __init__(self, *, raises: HTTPException | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._raises = raises

    async def __call__(self, token: str, adapter_id: str, scope: dict | None = None) -> str:
        self.calls.append((token, adapter_id))
        if self._raises is not None:
            raise self._raises
        return "org-A"


def _client(*, authorizer=None) -> TestClient:
    revision = _rec("qa")
    router = AdapterRouter([revision])
    app = build_serving_app(
        FakePool(),
        router,
        internal_key=INTERNAL_KEY,
        chat_authorizer=authorizer,
    )
    return TestClient(app)


def _chat(client: TestClient, **headers: str):
    return client.post(
        "/v1/chat/completions",
        json={"model": _id(), "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )


def test_missing_key_is_401() -> None:
    auth = FakeAuthorizer()
    resp = _chat(_client(authorizer=auth))
    assert resp.status_code == 401
    assert auth.calls == []


def test_valid_key_owning_adapter_passes() -> None:
    auth = FakeAuthorizer()
    resp = _chat(_client(authorizer=auth), Authorization="Bearer fs-user-key")
    assert resp.status_code == 200
    assert auth.calls == [("fs-user-key", _id())]  # token + adapter forwarded to the backend


def test_denied_key_propagates_403() -> None:
    auth = FakeAuthorizer(raises=HTTPException(status.HTTP_403_FORBIDDEN, "nope"))
    resp = _chat(_client(authorizer=auth), Authorization="Bearer other-org")
    assert resp.status_code == 403


def test_internal_key_bypasses_user_auth() -> None:
    # trusted server-to-server callers (the backend /api/sample proxy, the flash control plane)
    # present the shared internal key -> bypass, no user key needed.
    auth = FakeAuthorizer()
    resp = _chat(
        _client(authorizer=auth),
        **{"X-Freesolo-Internal-Key": INTERNAL_KEY, "X-Freesolo-Org-Id": "org-A"},
    )
    assert resp.status_code == 200
    assert auth.calls == []  # trusted caller is not sent through the user-key authorizer


def test_wrong_internal_key_is_not_trusted() -> None:
    # A non-matching internal key is NOT a trusted caller: it falls through to user-key auth, which
    # rejects (no Authorization header) rather than silently serving.
    auth = FakeAuthorizer()
    resp = _chat(_client(authorizer=auth), **{"X-Freesolo-Internal-Key": "wrong"})
    assert resp.status_code == 401
    assert auth.calls == []


def test_adapter_listing_is_gated_and_never_exposes_org_id() -> None:
    # GET /adapters requires the internal key (repo_id/url would otherwise leak the adapter->tenant
    # mapping to anon callers). Even for authorized callers the org id must never be serialized
    # (defense in depth), though the record still carries it internally for the auth check.
    revision = _rec("qa", org_id="org-secret")
    router = AdapterRouter([revision])
    client = TestClient(build_serving_app(FakePool(), router, internal_key=INTERNAL_KEY))
    assert client.get("/adapters").status_code == 401
    adapters = client.get(
        "/adapters",
        headers={"X-Freesolo-Internal-Key": INTERNAL_KEY, "X-Freesolo-Org-Id": "org-secret"},
    ).json()["adapters"]
    assert adapters
    assert all("org_id" not in a and "org_id" not in a for a in adapters)
    assert router.get(_id(), org_id="org-secret").org_id == "org-secret"


def test_generate_endpoint_is_also_enforced() -> None:
    auth = FakeAuthorizer()
    client = _client(authorizer=auth)
    assert client.post("/generate", json={"adapter_id": _id(), "prompt": "hi"}).status_code == 401
    ok = client.post(
        "/generate",
        json={"adapter_id": _id(), "prompt": "hi"},
        headers={"Authorization": "Bearer fs-user-key"},
    )
    assert ok.status_code == 200
    assert auth.calls == [("fs-user-key", _id())]


def test_chat_auth_precedes_stream_validation() -> None:
    auth = FakeAuthorizer()
    response = _client(authorizer=auth).post(
        "/v1/chat/completions",
        json={
            "model": _id(),
            "messages": [{"role": "user", "content": "hi"}],
            "stream": "not-a-boolean",
        },
    )

    assert response.status_code == 401
    assert auth.calls == []


def test_per_adapter_auth_precedes_payload_validation() -> None:
    """a valid path id is authenticated before a malformed request body is validated."""
    auth = FakeAuthorizer()
    response = _client(authorizer=auth).post(
        f"/adapters/{_id()}/generate",
        json={"prompt": "hi", "top_p": "not-a-number"},
    )

    assert response.status_code == 401
    assert auth.calls == []


@pytest.mark.parametrize("encoded_adapter_id", ["%20", "%09"], ids=["space", "tab"])
def test_per_adapter_blank_path_id_is_rejected_before_authorization(
    encoded_adapter_id: str,
) -> None:
    """a whitespace-only path id is a local 422 and never reaches the authorizer."""
    auth = FakeAuthorizer()
    response = _client(authorizer=auth).post(
        f"/adapters/{encoded_adapter_id}/generate",
        json={"prompt": "hi"},
        headers={"Authorization": "Bearer fs-user-key"},
    )

    assert response.status_code == 422
    assert auth.calls == []
    assert response.json() == {"detail": "adapter_id must not be empty"}


def test_generate_authorizes_against_the_normalized_adapter_id() -> None:
    # A whitespace-padded id is stripped once (GenerateRequest validator) so /generate authorizes
    # and routes against the same value ("qa"), not "  qa  ".
    auth = FakeAuthorizer()
    client = _client(authorizer=auth)
    ok = client.post(
        "/generate",
        json={"adapter_id": f"  {_id()}  ", "prompt": "hi"},
        headers={"Authorization": "Bearer fs-user-key"},
    )
    assert ok.status_code == 200
    assert auth.calls == [("fs-user-key", _id())]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
