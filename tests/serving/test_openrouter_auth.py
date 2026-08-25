"""Focused coverage for the provisional OpenRouter credential boundary."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from flash.serving.src.model_config import base_models
from flash.serving.src.openrouter_auth import OpenRouterAuthorization
from flash.serving.src.router import AdapterRouter, build_serving_app
from flash.serving.src.schemas import AdapterRecord
from tests.serving.conftest import attest

INTERNAL_KEY = "test-internal-key"
CURRENT_TOKEN = "fsor_v1_current-test-token"
PREVIOUS_TOKEN = "fsor_v1_previous-test-token"
SETTLEMENT_ORG = "openrouter-settlement-org"
QWEN = base_models()[0]


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _authorization(
    *,
    current: str = CURRENT_TOKEN,
    previous: str | None = PREVIOUS_TOKEN,
    internal_key: str | None = INTERNAL_KEY,
) -> OpenRouterAuthorization:
    environment = {
        "OPENROUTER_INFERENCE_KEY_SHA256_CURRENT": _digest(current),
        "OPENROUTER_SETTLEMENT_ORG_ID": SETTLEMENT_ORG,
    }
    if previous is not None:
        environment["OPENROUTER_INFERENCE_KEY_SHA256_PREVIOUS"] = _digest(previous)
    authorization = OpenRouterAuthorization.from_environment(
        environment,
        internal_key=internal_key,
    )
    assert authorization is not None
    return authorization


def _base_record(model_id: str = QWEN) -> AdapterRecord:
    return AdapterRecord(
        adapter_id=model_id,
        repo_id=model_id,
        base_model=model_id,
        serve_base_model=True,
        thinking=True,
        org_id=None,
        status="ready",
    )


def _revision(run_id: str = "adapter") -> AdapterRecord:
    revision = "a" * 40
    return AdapterRecord.model_validate(
        {
            "adapter_id": f"{run_id}@final.{revision}",
            "repo_id": f"org/{run_id}",
            "base_model": QWEN,
            "org_id": "org-a",
            "checkpoint": run_id,
            "status": "ready",
            "thinking": True,
            "metadata": {
                "record_type": "revision",
                "run_id": run_id,
                "checkpoint_step": None,
                "hf_revision": revision,
            },
        }
    )


def _alias(revision: AdapterRecord) -> AdapterRecord:
    assert revision.run_id is not None
    return revision.model_copy(
        update={
            "adapter_id": revision.run_id,
            "checkpoint": None,
            "metadata": {
                "record_type": "alias",
                "run_id": revision.run_id,
                "alias_of": revision.adapter_id,
            },
        }
    )


class _Pool:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, _base_model, _payload, record, *, expected_checkpoint=None):
        self.generate_calls += 1
        return attest(
            record,
            {
                "text": "ok",
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "checkpoint": record.checkpoint or "",
            },
        )

    async def stream_generate(self, _base_model, _payload, record, *, expected_checkpoint=None):
        self.stream_calls += 1
        yield {"type": "ready", "checkpoint": record.checkpoint or ""}
        yield {"type": "delta", "text": "ok"}
        yield {
            "type": "final",
            "finish_reason": "stop",
            "prompt_tokens": 3,
            "completion_tokens": 2,
        }

    async def register(self, _base_model, _record) -> None:
        raise AssertionError("OpenRouter authorization must not register adapters")

    async def unregister(self, _base_model, _adapter_id, expected_generation=None) -> None:
        raise AssertionError("OpenRouter authorization must not unregister adapters")


class _FreesoloAuthorizer:
    def __init__(
        self,
        org_id: str = "freesolo-org",
        *,
        raises: HTTPException | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.org_id = org_id
        self.raises = raises

    async def __call__(self, token: str, adapter_id: str) -> str:
        self.calls.append((token, adapter_id))
        if self.raises is not None:
            raise self.raises
        return self.org_id


def _client(
    records: list[AdapterRecord],
    *,
    authorization: OpenRouterAuthorization | None = None,
    authorizer: _FreesoloAuthorizer | None = None,
    usage_reports: list[dict[str, Any]] | None = None,
    pool: _Pool | None = None,
) -> tuple[TestClient, _Pool, _FreesoloAuthorizer]:
    actual_pool = pool or _Pool()
    actual_authorizer = authorizer or _FreesoloAuthorizer()

    async def capture(payload: dict[str, Any]) -> None:
        assert usage_reports is not None
        usage_reports.append(payload)

    app = build_serving_app(
        actual_pool,
        AdapterRouter(records),
        internal_key=INTERNAL_KEY,
        chat_authorizer=actual_authorizer,
        openrouter_authorization=authorization,
        usage_reporter=capture if usage_reports is not None else None,
    )
    return TestClient(app), actual_pool, actual_authorizer


def _chat(client: TestClient, model: str, token: str, *, stream: bool = False):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
        },
        headers={"Authorization": f"Bearer {token}"},
    )


@pytest.mark.parametrize("token", [CURRENT_TOKEN, PREVIOUS_TOKEN], ids=["current", "previous"])
@pytest.mark.parametrize("model_id", base_models())
def test_both_key_slots_authorize_every_catalog_base(token: str, model_id: str) -> None:
    client, pool, freesolo = _client(
        [_base_record(model_id)],
        authorization=_authorization(),
    )

    response = _chat(client, model_id, token)

    assert response.status_code == 200
    assert pool.generate_calls == 1
    assert freesolo.calls == []


@pytest.mark.parametrize("previous", [PREVIOUS_TOKEN, None], ids=["enabled", "disabled"])
def test_matcher_hashes_once_and_compares_both_slots_on_every_request(
    monkeypatch, previous: str | None
) -> None:
    import flash.serving.src.openrouter_auth as module

    authorization = _authorization(previous=previous)
    real_sha256 = hashlib.sha256
    real_compare = module.hmac.compare_digest
    hashes: list[bytes] = []
    comparisons: list[tuple[bytes, bytes]] = []

    def tracked_sha256(value: bytes):
        hashes.append(value)
        return real_sha256(value)

    def tracked_compare(left: bytes, right: bytes) -> bool:
        comparisons.append((left, right))
        return real_compare(left, right)

    monkeypatch.setattr(module.hashlib, "sha256", tracked_sha256)
    monkeypatch.setattr(module.hmac, "compare_digest", tracked_compare)

    assert authorization.matches(CURRENT_TOKEN)
    assert authorization.matches(CURRENT_TOKEN)
    assert hashes == [CURRENT_TOKEN.encode(), CURRENT_TOKEN.encode()]
    assert len(comparisons) == 4
    assert [right for _left, right in comparisons] == [
        authorization.current_digest,
        authorization.previous_digest,
        authorization.current_digest,
        authorization.previous_digest,
    ]


@pytest.mark.parametrize(
    ("header", "expected_status", "expected_token"),
    [
        (None, 401, None),
        ("Basic abc", 401, None),
        ("Bearer", 401, None),
        ("Bearertoken", 401, None),
        ("Bearerx token", 401, None),
        ("Bearer ", 401, None),
        ("Bearer  ordinary", 200, "ordinary"),
        ("Bearer ordinary ", 200, "ordinary"),
        ("bEaReR ordinary", 200, "ordinary"),
        ("Bearer token, Bearer other", 400, None),
    ],
)
def test_authorization_header_syntax(
    header: str | None,
    expected_status: int,
    expected_token: str | None,
) -> None:
    client, _pool, authorizer = _client([_base_record()])
    headers = {} if header is None else {"Authorization": header}

    response = client.post(
        "/v1/chat/completions",
        json={"model": QWEN, "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )

    assert response.status_code == expected_status
    if expected_status == 401:
        assert response.headers["www-authenticate"] == "Bearer"
    if expected_token is not None:
        assert authorizer.calls == [(expected_token, QWEN)]


def test_duplicate_authorization_fields_are_400() -> None:
    client, _pool, _authorizer = _client([_base_record()])
    response = client.post(
        "/v1/chat/completions",
        json={"model": QWEN, "messages": [{"role": "user", "content": "hi"}]},
        headers=[("Authorization", "Bearer one"), ("Authorization", "Bearer two")],
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "authorization_header",
    [f"Bearer  {CURRENT_TOKEN}", f"Bearer {CURRENT_TOKEN} "],
)
def test_openrouter_matches_the_tolerantly_extracted_token(authorization_header: str) -> None:
    client, _pool, freesolo = _client([_base_record()], authorization=_authorization())

    response = client.post(
        "/v1/chat/completions",
        json={"model": QWEN, "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": authorization_header},
    )

    assert response.status_code == 200
    assert freesolo.calls == []


@pytest.mark.parametrize("path", ["/generate", "/adapters/adapter/generate"])
def test_matching_openrouter_is_denied_on_other_inference_routes(path: str) -> None:
    client, pool, freesolo = _client([_base_record()], authorization=_authorization())
    payload = {"adapter_id": QWEN, "prompt": "hi"} if path == "/generate" else {"prompt": "hi"}

    response = client.post(
        path,
        json=payload,
        headers={"Authorization": f"Bearer {CURRENT_TOKEN}"},
    )

    assert response.status_code == 403
    assert pool.generate_calls == 0
    assert freesolo.calls == []


def test_digest_match_is_openrouter_and_nonmatch_follows_freesolo() -> None:
    client, _pool, freesolo = _client([_base_record()], authorization=_authorization())
    unmatched_token = "fsor_v1_unknown-do-not-expose"

    matched = _chat(client, QWEN, CURRENT_TOKEN)
    unmatched = _chat(client, QWEN, unmatched_token)

    assert matched.status_code == 200
    assert unmatched.status_code == 200
    assert unmatched_token not in unmatched.text
    assert freesolo.calls == [(unmatched_token, QWEN)]


def test_nonmatching_token_follows_freesolo_on_legacy_inference() -> None:
    token = "fsor_v1_unknown-legacy"
    client, pool, freesolo = _client([_base_record()], authorization=_authorization())

    response = client.post(
        "/generate",
        json={"adapter_id": QWEN, "prompt": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert pool.generate_calls == 1
    assert freesolo.calls == [(token, QWEN)]


def test_nonmatching_token_follows_freesolo_when_openrouter_is_disabled() -> None:
    client, _pool, freesolo = _client([_base_record()], authorization=None)
    token = "fsor_v1_disabled"

    response = _chat(client, QWEN, token)

    assert response.status_code == 200
    assert freesolo.calls == [(token, QWEN)]


def test_matching_digest_without_namespace_authorizes_openrouter() -> None:
    token = "matching-but-unnamespaced"
    client, _pool, freesolo = _client(
        [_base_record()],
        authorization=_authorization(current=token, previous=None),
    )

    response = _chat(client, QWEN, token)

    assert response.status_code == 200
    assert freesolo.calls == []


def test_openrouter_cannot_access_adapter_lifecycle() -> None:
    client, _pool, freesolo = _client([_base_record()], authorization=_authorization())

    matched = client.get(
        "/adapters",
        headers={"Authorization": f"Bearer {CURRENT_TOKEN}"},
    )
    unmatched = client.get(
        "/adapters",
        headers={"Authorization": "Bearer fsor_v1_unknown"},
    )

    assert matched.status_code == 403
    assert unmatched.status_code == 401
    assert freesolo.calls == []


@pytest.mark.parametrize("model_id", ["unknown", "adapter", f"adapter@final.{'a' * 40}"])
def test_openrouter_denies_unknown_alias_and_revision_records(model_id: str) -> None:
    revision = _revision()
    client, pool, freesolo = _client(
        [_base_record(), revision, _alias(revision)],
        authorization=_authorization(),
    )

    response = _chat(client, model_id, CURRENT_TOKEN)

    assert response.status_code == 403
    assert pool.generate_calls == 0
    assert freesolo.calls == []


def test_missing_seeded_catalog_base_is_503_without_dispatch() -> None:
    client, pool, freesolo = _client([], authorization=_authorization())

    response = _chat(client, QWEN, CURRENT_TOKEN)

    assert response.status_code == 503
    assert pool.generate_calls == 0
    assert freesolo.calls == []


@pytest.mark.parametrize(
    "forged",
    [
        _base_record().model_copy(update={"serve_base_model": False}),
        _base_record().model_copy(update={"repo_id": "forged"}),
        _base_record().model_copy(update={"checkpoint": "step-1"}),
    ],
)
def test_forged_catalog_records_are_403_without_dispatch(forged: AdapterRecord) -> None:
    client, pool, _freesolo = _client([forged], authorization=_authorization())
    response = _chat(client, QWEN, CURRENT_TOKEN)
    assert response.status_code == 403
    assert pool.generate_calls == 0


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_usage_is_attributed_to_settlement_org_without_adapter_id(stream: bool) -> None:
    reports: list[dict[str, Any]] = []
    client, _pool, _freesolo = _client(
        [_base_record()],
        authorization=_authorization(),
        usage_reports=reports,
    )

    with client:
        response = _chat(client, QWEN, CURRENT_TOKEN, stream=stream)
        assert response.status_code == 200
        if stream:
            assert "[DONE]" in response.text

    assert len(reports) == 1
    assert reports[0]["orgId"] == SETTLEMENT_ORG
    assert reports[0]["baseModel"] == QWEN
    assert "adapterId" not in reports[0]


def test_freesolo_401_gets_bearer_challenge() -> None:
    authorizer = _FreesoloAuthorizer(
        raises=HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid key")
    )
    client, _pool, _authorizer = _client(
        [_base_record()],
        authorization=_authorization(),
        authorizer=authorizer,
    )

    response = _chat(client, QWEN, "unknown-freesolo-key")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_existing_freesolo_and_internal_auth_are_preserved() -> None:
    freesolo = _FreesoloAuthorizer(org_id="existing-org")
    reports: list[dict[str, Any]] = []
    client, pool, freesolo = _client(
        [_base_record()],
        authorization=_authorization(),
        authorizer=freesolo,
        usage_reports=reports,
    )

    user_response = _chat(client, QWEN, "existing-user-key")
    internal_response = client.post(
        "/v1/chat/completions",
        json={"model": QWEN, "messages": [{"role": "user", "content": "hi"}]},
        headers={
            "X-Freesolo-Internal-Key": INTERNAL_KEY,
            "Authorization": "Bearer malformed token",
        },
    )

    assert user_response.status_code == 200
    assert internal_response.status_code == 200
    assert freesolo.calls == [("existing-user-key", QWEN)]
    assert pool.generate_calls == 2
    assert reports[0]["orgId"] == "existing-org"


def test_configuration_absent_partial_malformed_distinct_and_collision_rules() -> None:
    assert OpenRouterAuthorization.from_environment({}, internal_key=INTERNAL_KEY) is None

    with pytest.raises(ValueError, match="OPENROUTER_SETTLEMENT_ORG_ID"):
        OpenRouterAuthorization.from_environment(
            {"OPENROUTER_INFERENCE_KEY_SHA256_CURRENT": _digest(CURRENT_TOKEN)},
            internal_key=INTERNAL_KEY,
        )
    with pytest.raises(ValueError, match="OPENROUTER_INFERENCE_KEY_SHA256_CURRENT"):
        OpenRouterAuthorization.from_environment(
            {"OPENROUTER_SETTLEMENT_ORG_ID": SETTLEMENT_ORG},
            internal_key=INTERNAL_KEY,
        )
    with pytest.raises(ValueError, match="OPENROUTER_INFERENCE_KEY_SHA256_CURRENT"):
        OpenRouterAuthorization.from_environment(
            {
                "OPENROUTER_INFERENCE_KEY_SHA256_CURRENT": "not-a-digest",
                "OPENROUTER_SETTLEMENT_ORG_ID": SETTLEMENT_ORG,
            },
            internal_key=INTERNAL_KEY,
        )
    with pytest.raises(ValueError, match="must differ"):
        OpenRouterAuthorization.from_environment(
            {
                "OPENROUTER_INFERENCE_KEY_SHA256_CURRENT": _digest(CURRENT_TOKEN),
                "OPENROUTER_INFERENCE_KEY_SHA256_PREVIOUS": _digest(CURRENT_TOKEN),
                "OPENROUTER_SETTLEMENT_ORG_ID": SETTLEMENT_ORG,
            },
            internal_key=INTERNAL_KEY,
        )
    for collision_field in (
        "OPENROUTER_INFERENCE_KEY_SHA256_CURRENT",
        "OPENROUTER_INFERENCE_KEY_SHA256_PREVIOUS",
    ):
        environment = {
            "OPENROUTER_INFERENCE_KEY_SHA256_CURRENT": _digest(CURRENT_TOKEN),
            "OPENROUTER_SETTLEMENT_ORG_ID": SETTLEMENT_ORG,
            collision_field: _digest(INTERNAL_KEY),
        }
        with pytest.raises(ValueError, match="FREESOLO_INTERNAL_KEY"):
            OpenRouterAuthorization.from_environment(
                environment,
                internal_key=INTERNAL_KEY,
            )

    without_previous = _authorization(previous=None)
    assert without_previous.previous_enabled is False
    assert without_previous.matches(CURRENT_TOKEN)
    assert not without_previous.matches(PREVIOUS_TOKEN)


def test_configuration_errors_and_repr_do_not_expose_credentials_or_digests() -> None:
    raw_secret = "fsor_v1_do-not-expose"
    digest = _digest(raw_secret)
    authorization = _authorization(current=raw_secret, previous=None)

    assert raw_secret not in repr(authorization)
    assert digest not in repr(authorization)
    with pytest.raises(ValueError, match="must differ") as exc:
        OpenRouterAuthorization.from_environment(
            {
                "OPENROUTER_INFERENCE_KEY_SHA256_CURRENT": digest,
                "OPENROUTER_INFERENCE_KEY_SHA256_PREVIOUS": digest,
                "OPENROUTER_SETTLEMENT_ORG_ID": SETTLEMENT_ORG,
            },
            internal_key=INTERNAL_KEY,
        )
    assert raw_secret not in str(exc.value)
    assert digest not in str(exc.value)
