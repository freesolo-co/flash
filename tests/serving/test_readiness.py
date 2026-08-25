from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from flash.serving.src.lookup import AdapterLookup
from flash.serving.src.readiness import (
    build_readiness_publication,
    build_runtime_readiness_evidence,
    engine_contract_sha256,
    evidence_sha256,
    load_readiness_passes,
    publish_readiness_pass,
    qualified_base_records,
)
from flash.serving.src.router import build_serving_app
from flash.serving.src.routing import AdapterRouter
from flash.serving.src.schemas import AdapterRecord
from flash.serving.src.settings import Settings
from tests.serving.conftest import attest

QWEN = "Qwen/Qwen3.5-0.8B"
QWEN_2B = "Qwen/Qwen3.5-2B"
DEPLOYMENT_SHA = "a" * 40


def _settings(**updates: Any) -> Settings:
    values = {
        "SERVING_DEPLOYMENT_MODE": "production",
        "FREESOLO_DEPLOYMENT_SHA": DEPLOYMENT_SHA,
        "FREESOLO_DEPLOYMENT_ID": "deployment-1",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "sb_secret_test",
    }
    values.update(updates)
    return Settings(**values)


def _evidence(
    model_id: str = QWEN,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or _settings()
    contract_digest = engine_contract_sha256(model_id)
    return {
        "schema_version": 1,
        "outcome": "passed",
        "deployment_mode": settings.deployment_mode,
        "deployment_sha": settings.deployment_sha,
        "deployment_id": settings.deployment_id,
        "model_id": model_id,
        "engine_contract_sha256": contract_digest,
        "health": {"passed": True, "evidence_sha256": "1" * 64},
        "non_streaming": {
            "passed": True,
            "evidence_sha256": "2" * 64,
            "request_id": "request-non-streaming",
            "engine_replica_id": "replica-1",
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "finish_reason": "stop",
            "checkpoint": model_id,
        },
        "streaming": {
            "passed": True,
            "evidence_sha256": "3" * 64,
            "request_id": "request-streaming",
            "engine_replica_id": "replica-1",
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "finish_reason": "stop",
            "checkpoint": model_id,
        },
        "provenance": {"passed": True, "evidence_sha256": "4" * 64},
        "runtime_attestation": {"passed": True, "evidence_sha256": "5" * 64},
    }


def _row(
    model_id: str = QWEN,
    *,
    settings: Settings | None = None,
    contract_digest: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or _settings()
    evidence = _evidence(model_id, settings=settings) if evidence is None else evidence
    if contract_digest is not None:
        evidence = {**evidence, "engine_contract_sha256": contract_digest}
    return {
        "deployment_mode": settings.deployment_mode,
        "deployment_sha": settings.deployment_sha,
        "deployment_id": settings.deployment_id,
        "model_id": model_id,
        "engine_contract_sha256": contract_digest or engine_contract_sha256(model_id),
        "evidence_version": 1,
        "evidence": evidence,
        "evidence_sha256": evidence_sha256(evidence),
        "passed_at": datetime.now(UTC).isoformat(),
    }


class _Client:
    def __init__(
        self,
        *,
        get: Callable[..., httpx.Response] | None = None,
        post: Callable[..., httpx.Response] | None = None,
    ) -> None:
        self._get = get
        self._post = post

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        assert self._get is not None
        return self._get(url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        assert self._post is not None
        return self._post(url, **kwargs)


def _response(status_code: int, payload: Any) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("GET", "https://example.test"),
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _Client) -> None:
    monkeypatch.setattr(
        "flash.serving.src.readiness.httpx.Client",
        lambda **_kwargs: client,
    )


def _revision(run_id: str, base_model: str = QWEN) -> AdapterRecord:
    sha = "b" * 40
    return AdapterRecord.model_validate(
        {
            "adapter_id": f"{run_id}@final.{sha}",
            "repo_id": f"org/{run_id}",
            "base_model": base_model,
            "org_id": "org-1",
            "checkpoint": run_id,
            "thinking": True,
            "status": "ready",
            "metadata": {
                "record_type": "revision",
                "run_id": run_id,
                "checkpoint_step": None,
                "hf_revision": sha,
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


def _base(model_id: str = QWEN) -> AdapterRecord:
    return AdapterRecord(
        adapter_id=model_id,
        repo_id=model_id,
        base_model=model_id,
        serve_base_model=True,
        thinking=True,
        org_id=None,
        status="ready",
    )


def test_no_evidence_and_missing_identity_yield_no_base_records(monkeypatch) -> None:
    settings = _settings()
    _patch_client(monkeypatch, _Client(get=lambda *_a, **_k: _response(200, [])))
    assert qualified_base_records(settings) == []

    monkeypatch.setattr(
        "flash.serving.src.readiness.httpx.Client",
        lambda **_kwargs: pytest.fail("missing identity must not query storage"),
    )
    assert qualified_base_records(_settings(FREESOLO_DEPLOYMENT_SHA="")) == []
    assert qualified_base_records(_settings(FREESOLO_DEPLOYMENT_ID=" padded ")) == []


def test_publication_rejects_padded_identity_before_storage(monkeypatch) -> None:
    monkeypatch.setattr(
        "flash.serving.src.readiness.httpx.Client",
        lambda **_kwargs: pytest.fail("invalid identity must not open storage"),
    )
    settings = _settings(FREESOLO_DEPLOYMENT_ID=" padded ")
    with pytest.raises(RuntimeError, match="exact deployment identity"):
        publish_readiness_pass(settings, QWEN, _evidence(settings=settings))


def test_exact_identity_and_contract_qualify_while_stale_digest_does_not(monkeypatch) -> None:
    settings = _settings()
    stale = "c" * 64
    rows = [_row(QWEN, settings=settings), _row(QWEN_2B, settings=settings, contract_digest=stale)]
    _patch_client(monkeypatch, _Client(get=lambda *_a, **_k: _response(200, rows)))

    records = qualified_base_records(settings)

    assert [record.adapter_id for record in records] == [QWEN]
    assert records[0].serve_base_model is True
    assert records[0].metadata["readiness"]["engine_contract_sha256"] == engine_contract_sha256(
        QWEN
    )


def test_mismatched_identity_isolated_from_valid_sibling(monkeypatch) -> None:
    settings = _settings()
    other_settings = _settings(FREESOLO_DEPLOYMENT_ID="other-deployment")
    stale = _row(QWEN_2B, settings=other_settings)
    rows = [_row(QWEN, settings=settings), stale]
    _patch_client(monkeypatch, _Client(get=lambda *_a, **_k: _response(200, rows)))

    assert [readiness.model_id for readiness in load_readiness_passes(settings)] == [QWEN]


@pytest.mark.parametrize(
    "evidence",
    [
        {},
        {"smoke": False},
        {"error": "engine never started"},
        {**_evidence(), "outcome": "failed"},
        {**_evidence(), "health": {"passed": False, "evidence_sha256": "1" * 64}},
    ],
)
def test_arbitrary_or_failed_evidence_never_qualifies(monkeypatch, evidence) -> None:
    settings = _settings()
    row = _row(settings=settings, evidence=evidence)
    _patch_client(monkeypatch, _Client(get=lambda *_a, **_k: _response(200, [row])))

    assert load_readiness_passes(settings) == []


def test_evidence_identity_must_match_storage_row(monkeypatch) -> None:
    settings = _settings()
    evidence = _evidence(settings=settings)
    evidence["deployment_id"] = "different-deployment"
    row = _row(settings=settings, evidence=evidence)
    _patch_client(monkeypatch, _Client(get=lambda *_a, **_k: _response(200, [row])))

    assert load_readiness_passes(settings) == []


def test_unknown_evidence_model_fails_closed(monkeypatch) -> None:
    settings = _settings()
    row = _row(settings=settings)
    row["model_id"] = "unknown/model"
    row["evidence"]["model_id"] = "unknown/model"
    row["evidence_sha256"] = evidence_sha256(row["evidence"])
    _patch_client(monkeypatch, _Client(get=lambda *_a, **_k: _response(200, [row])))

    assert load_readiness_passes(settings) == []


def test_cold_start_retains_valid_sibling_when_other_rows_are_invalid(monkeypatch) -> None:
    settings = _settings()
    malformed = _row(QWEN_2B, settings=settings)
    malformed["evidence"] = {"outcome": "failed"}
    malformed["evidence_sha256"] = evidence_sha256(malformed["evidence"])
    unknown = _row(QWEN_2B, settings=settings)
    unknown["model_id"] = "unknown/model"
    unknown["evidence"]["model_id"] = "unknown/model"
    unknown["evidence_sha256"] = evidence_sha256(unknown["evidence"])
    _patch_client(
        monkeypatch,
        _Client(get=lambda *_a, **_k: _response(200, [malformed, _row(QWEN), unknown])),
    )

    records = qualified_base_records(settings)

    assert [record.adapter_id for record in records] == [QWEN]


def test_generation_evidence_is_bound_to_exact_logical_model() -> None:
    settings = _settings()
    engine_health = {"ok": True, "engine_dead": False, "base_model": QWEN}
    deployment_health = {
        "ok": True,
        "deployment_sha": settings.deployment_sha,
        "deployment_id": settings.deployment_id,
    }
    non_streaming = {
        "request_id": "request-non-streaming",
        "engine_replica_id": "replica-1",
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "finish_reason": "stop",
        "checkpoint": QWEN,
    }
    streaming = {**non_streaming, "request_id": "request-streaming"}

    evidence = build_runtime_readiness_evidence(
        settings,
        QWEN,
        engine_health=engine_health,
        non_streaming=non_streaming,
        streaming=streaming,
        deployment_health=deployment_health,
    )
    assert evidence["non_streaming"]["checkpoint"] == QWEN
    assert evidence["streaming"]["checkpoint"] == QWEN

    with pytest.raises(ValueError, match="exact readiness model"):
        build_runtime_readiness_evidence(
            settings,
            QWEN_2B,
            engine_health={**engine_health, "base_model": QWEN_2B},
            non_streaming=non_streaming,
            streaming=streaming,
            deployment_health=deployment_health,
        )


def test_copied_cross_model_publication_is_rejected() -> None:
    settings = _settings()
    copied = _evidence(QWEN, settings=settings)
    copied["model_id"] = QWEN_2B
    copied["engine_contract_sha256"] = engine_contract_sha256(QWEN_2B)

    with pytest.raises(ValueError, match="checkpoint must match model_id"):
        build_readiness_publication(settings, QWEN_2B, copied)


def test_partial_qualification_gates_each_adapter_base() -> None:
    first = _revision("first", QWEN)
    second = _revision("second", QWEN_2B)
    router = AdapterRouter(
        [first, _alias(first), second, _alias(second), _base(QWEN)],
        require_base_qualification=True,
    )

    assert router.resolve("first") == (router.get("first"), first)
    assert router.resolve("second") is None
    assert router.is_unqualified_adapter("second") is True
    assert router.resolve(QWEN) == (router.get(QWEN), router.get(QWEN))
    assert router.resolve(QWEN_2B) is None


class _Pool:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, _base_model, _payload, record, *, expected_checkpoint=None):
        self.calls += 1
        return attest(
            record,
            {
                "text": "ok",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "checkpoint": record.checkpoint or "",
            },
        )

    async def stream_generate(self, *_args, **_kwargs):
        raise AssertionError("streaming is not expected")
        yield

    async def register(self, *_args, **_kwargs):
        raise AssertionError("registration is not expected")

    async def unregister(self, *_args, **_kwargs):
        raise AssertionError("unregister is not expected")


async def _allow(_token: str, _adapter_id: str) -> str:
    return "org-1"


def _chat(client: TestClient, model: str) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer key"},
    )


def test_valid_adapter_on_unqualified_base_returns_retryable_503_without_pool_call() -> None:
    revision = _revision("first")
    pool = _Pool()
    app = build_serving_app(
        pool,
        AdapterRouter([revision, _alias(revision)], require_base_qualification=True),
        chat_authorizer=_allow,
    )

    with TestClient(app) as client:
        response = _chat(client, "first")
        hidden_base = _chat(client, QWEN)

    assert response.status_code == 503
    assert response.json()["detail"] == "adapter base model is not qualified for this deployment"
    assert hidden_base.status_code == 404
    assert pool.calls == 0


def test_adapter_and_base_qualification_conjunction_routes_normally() -> None:
    revision = _revision("first")
    pool = _Pool()
    app = build_serving_app(
        pool,
        AdapterRouter(
            [revision, _alias(revision), _base()],
            require_base_qualification=True,
        ),
        chat_authorizer=_allow,
    )

    with TestClient(app) as client:
        assert _chat(client, "first").status_code == 200
    assert pool.calls == 1


def test_periodic_refresh_converges_two_replicas() -> None:
    revision = _revision("first")
    snapshots: list[list[AdapterRecord]] = [[revision, _alias(revision)]]
    routers = [
        AdapterRouter(snapshots[0], require_base_qualification=True),
        AdapterRouter(snapshots[0], require_base_qualification=True),
    ]
    lookups = [
        AdapterLookup(router, lambda: list(snapshots[-1]), reload_interval_seconds=0.01)
        for router in routers
    ]

    async def exercise() -> None:
        tasks = [asyncio.create_task(lookup.refresh_periodically()) for lookup in lookups]
        snapshots.append([revision, _alias(revision), _base()])
        try:
            for _ in range(100):
                if all(router.resolve("first") is not None for router in routers):
                    break
                await asyncio.sleep(0.005)
            assert all(router.resolve("first") is not None for router in routers)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(exercise())


def test_refresh_failure_preserves_complete_prior_snapshot() -> None:
    revision = _revision("first")
    router = AdapterRouter(
        [revision, _alias(revision), _base()],
        require_base_qualification=True,
    )

    def fail() -> list[AdapterRecord]:
        raise RuntimeError("storage failed after adapter fetch")

    lookup = AdapterLookup(router, fail)
    assert asyncio.run(lookup._reload_safe()) is False
    assert router.resolve("first") is not None
    assert router.resolve(QWEN) is not None


def test_health_is_in_memory_and_does_not_touch_pool_or_storage() -> None:
    calls = 0

    def reload() -> list[AdapterRecord]:
        nonlocal calls
        calls += 1
        raise AssertionError("health must not read storage")

    pool = _Pool()
    app = build_serving_app(
        pool,
        AdapterRouter([_base()], require_base_qualification=True),
        reload_records=reload,
        reload_interval_seconds=60.0,
        chat_authorizer=_allow,
    )
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["base_models"] == [QWEN]
    assert calls == 0
    assert pool.calls == 0


def test_publication_retries_then_returns_inserted_row(monkeypatch) -> None:
    settings = _settings()
    evidence = _evidence(settings=settings)
    publication = build_readiness_publication(settings, QWEN, evidence)
    responses = [
        _response(503, {"message": "unavailable"}),
        _response(
            201,
            [{**publication.model_dump(mode="json"), "passed_at": datetime.now(UTC).isoformat()}],
        ),
    ]
    posts = 0

    def post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        nonlocal posts
        response = responses[posts]
        posts += 1
        return response

    _patch_client(monkeypatch, _Client(post=post))
    row = publish_readiness_pass(settings, QWEN, evidence, retry_delays_seconds=(0.0,))
    assert row.model_id == QWEN
    assert posts == 2


def test_publication_unique_conflict_is_idempotent_only_for_exact_evidence(monkeypatch) -> None:
    settings = _settings()
    evidence = _evidence(settings=settings)
    publication = build_readiness_publication(settings, QWEN, evidence)
    existing = {
        **publication.model_dump(mode="json"),
        "passed_at": datetime.now(UTC).isoformat(),
    }
    conflict = _response(409, {"code": "23505", "message": "duplicate"})
    _patch_client(
        monkeypatch,
        _Client(
            post=lambda *_a, **_k: conflict,
            get=lambda *_a, **_k: _response(200, [existing]),
        ),
    )
    assert publish_readiness_pass(settings, QWEN, evidence).model_id == QWEN

    mismatched = dict(existing)
    mismatched["evidence"] = {
        **existing["evidence"],
        "health": {"passed": True, "evidence_sha256": "6" * 64},
    }
    mismatched["evidence_sha256"] = evidence_sha256(mismatched["evidence"])
    _patch_client(
        monkeypatch,
        _Client(
            post=lambda *_a, **_k: conflict,
            get=lambda *_a, **_k: _response(200, [mismatched]),
        ),
    )
    with pytest.raises(RuntimeError, match="conflicts with existing evidence"):
        publish_readiness_pass(settings, QWEN, evidence)
