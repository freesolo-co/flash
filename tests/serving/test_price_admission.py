from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

import flash.serving.src.http.context as context_module
import flash.serving.src.http.inference_routes as inference_routes
from flash.serving.src.accounting.usage import (
    _FREESOLO_USD_PER_MTOK,
    AuthorizedTraffic,
    build_usage_session,
    capture_authoritative_price,
    new_generation_id,
    principal_for_external_org,
)
from flash.serving.src.accounting.usage_outbox import (
    AcceptedPriceSnapshot,
    CapturedPrice,
    OfflineUsageStore,
    OpenRouterTrafficPrincipal,
    RequestIdentity,
    UsageOutboxError,
)
from flash.serving.src.http.router import AdapterRouter, build_serving_app
from flash.serving.src.io.schemas import AdapterRecord
from tests.serving.checkpoint_fixtures import checkpoint_record
from tests.serving.conftest import RecordingUsageStore, attest

BASE_MODEL = "Qwen/Qwen3.5-9B"


def _revision(*, base_model: str = BASE_MODEL) -> AdapterRecord:
    return checkpoint_record("price-admission", base_model)


class _Pool:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.generate_calls = 0
        self.stream_constructions = 0
        self.stream_advances = 0
        self.generation_id: str | None = None

    async def generate(
        self,
        base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        del base_model, expected_checkpoint
        if self.events is not None:
            self.events.append("dispatch")
        self.generate_calls += 1
        generation_id = payload.generation_id
        assert generation_id is not None
        self.generation_id = generation_id
        return attest(
            record,
            {
                "ok": True,
                "adapter_id": payload.adapter_id,
                "text": "ok",
                "finish_reason": "stop",
                "prompt_token_ids": [11, 12],
                "completion_token_ids": [21],
                "token_ids": [21],
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "cached_tokens": 1,
                "cached_tokens_reported": True,
                "reasoning_tokens": 0,
                "thinking": False,
                "inference_time_seconds": 0.01,
                "request_id": generation_id,
                "engine_replica_id": "replica-1",
                "checkpoint": record.checkpoint,
            },
        )

    def stream_generate(
        self,
        base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ):
        del base_model, expected_checkpoint
        if self.events is not None:
            self.events.append("dispatch")
        self.stream_constructions += 1
        generation_id = payload.generation_id

        async def stream():
            self.stream_advances += 1
            assert generation_id is not None
            self.generation_id = generation_id
            common = {
                "prompt_tokens": 2,
                "prompt_token_ids": [11, 12],
                "reasoning_tokens": 0,
                "thinking": False,
                "request_id": generation_id,
                "engine_replica_id": "replica-1",
                "checkpoint": record.checkpoint,
                "lora_request_adapter": record.adapter_id,
            }
            yield {"type": "ready", "completion_tokens": 0, "completion_token_ids": [], **common}
            yield {
                "type": "delta",
                "text": "ok",
                "completion_tokens": 1,
                "completion_token_ids": [21],
                **common,
            }
            yield {
                "type": "final",
                "finish_reason": "stop",
                "completion_tokens": 1,
                "completion_token_ids": [21],
                **common,
            }

        return stream()

    async def register(self, base_model: str, record: AdapterRecord) -> None:
        del base_model, record

    async def unregister(
        self,
        base_model: str,
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> None:
        del base_model, adapter_id, expected_generation


async def _authorize(_token: str, _adapter_id: str, _scope: Any = None) -> str:
    return "org-1"


def _app(
    record: AdapterRecord,
    pool: _Pool,
    *,
    store: OfflineUsageStore | None = None,
    authorizer: Any = _authorize,
):
    return build_serving_app(
        pool,
        AdapterRouter([record]),
        usage_store=store or RecordingUsageStore(),
        internal_key="internal-key",
        chat_authorizer=authorizer,
    )


def _headers(caller: str = "freesolo") -> dict[str, str]:
    if caller == "trusted_internal":
        # a trusted-internal caller addressing a checkpoint ref must scope it to an org, so the
        # registry lookup keys on (org_id, checkpoint_id) the same way the record was stored.
        return {"X-Freesolo-Internal-Key": "internal-key", "X-Freesolo-Org-Id": "org-1"}
    return {"Authorization": "Bearer user-key"}


def _request_case(route: str, record: AdapterRecord, *, stream: bool = False):
    if route == "generate":
        return "/generate", {"adapter_id": record.adapter_id, "prompt": "hello"}
    if route == "adapter_generate":
        return f"/adapters/{record.adapter_id}/generate", {"prompt": "hello"}
    return (
        "/v1/chat/completions",
        {
            "model": record.adapter_id,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
        },
    )


def _base_record(base_model: str = BASE_MODEL) -> AdapterRecord:
    """A base-model row, which is what org-less openrouter traffic can address.

    ``traffic_org_id`` is None for an openrouter principal, so a checkpoint ref would miss the
    org-keyed registry entry and 404 before price admission ever runs.
    """

    return AdapterRecord(
        adapter_id=base_model,
        repo_id=base_model,
        base_model=base_model,
        serve_base_model=True,
        thinking=False,
        org_id=None,
        status="ready",
    )


def _openrouter_principal(
    snapshot: AcceptedPriceSnapshot | None = None,
) -> OpenRouterTrafficPrincipal:
    return OpenRouterTrafficPrincipal(
        publicModelId="public/model",
        providerCatalogDigest="catalog-digest-1",
        acceptedPriceSnapshot=snapshot
        or AcceptedPriceSnapshot(
            promptTokenUsd="0.000001",
            cachedPromptTokenUsd="0.0000005",
            completionTokenUsd="0.000002",
        ),
    )


@pytest.mark.parametrize(
    "route", ["generate", "adapter_generate", "openai_nonstream", "openai_stream"]
)
def test_price_admission_precedes_prepare_and_dispatch(monkeypatch, route: str) -> None:
    record = _revision()
    events: list[str] = []
    pool = _Pool(events)
    price_capture_count = 0

    class OrderedStore(RecordingUsageStore):
        async def capture(self, event) -> None:
            events.append("price_admission")
            await super().capture(event)

    async def authorize(_token: str, _adapter_id: str, _scope: Any = None) -> str:
        events.append("authorize")
        return "org-1"

    original_resolve = context_module.AdapterLookup.resolve

    async def resolve(self, adapter_id: str, **kwargs):
        events.append("resolve")
        return await original_resolve(self, adapter_id, **kwargs)

    original_capture = context_module.ServingContext.capture_price

    def capture(self, traffic, target):
        nonlocal price_capture_count
        price_capture_count += 1
        events.append("price_snapshot")
        return original_capture(self, traffic, target)

    original_prepare = inference_routes._prepare_generate_request

    async def prepare(payload, target):
        events.append("prepare")
        return await original_prepare(payload, target)

    monkeypatch.setattr(context_module.AdapterLookup, "resolve", resolve)
    monkeypatch.setattr(context_module.ServingContext, "capture_price", capture)
    monkeypatch.setattr(inference_routes, "_prepare_generate_request", prepare)
    store = OrderedStore()
    app = _app(record, pool, store=store, authorizer=authorize)
    request_route = "openai" if route.startswith("openai") else route
    path, body = _request_case(request_route, record, stream=route == "openai_stream")
    with TestClient(app, headers=_headers()) as client:
        response = client.post(path, json=body)

    assert response.status_code == 200
    assert price_capture_count == 1
    assert len(store.captured) == 1
    assert len(store.finalized) == 1
    assert store.finalized[0].identity == store.captured[0].identity
    assert store.finalized[0].target == store.captured[0].target
    assert store.finalized[0].price is store.captured[0].price
    assert events == [
        "authorize",
        "resolve",
        "price_snapshot",
        "price_admission",
        "prepare",
        "dispatch",
    ]


@pytest.mark.parametrize("caller", ["freesolo", "trusted_internal"])
@pytest.mark.parametrize("route", ["generate", "adapter_generate", "openai"])
def test_missing_freesolo_price_rejects_before_prepare_or_dispatch(
    monkeypatch, caller: str, route: str
) -> None:
    record = _revision()
    pool = _Pool()

    async def unreachable_prepare(*_args, **_kwargs):
        pytest.fail("request preparation must be unreachable")

    monkeypatch.delitem(_FREESOLO_USD_PER_MTOK, record.base_model)
    monkeypatch.setattr(inference_routes, "_prepare_generate_request", unreachable_prepare)
    app = _app(record, pool)
    path, body = _request_case(route, record, stream=route == "openai")
    with TestClient(app, headers=_headers(caller)) as client:
        response = client.post(path, json=body)

    assert response.status_code == 503
    assert response.json() == {"detail": "durable serving accounting unavailable"}
    assert pool.generate_calls == 0
    assert pool.stream_constructions == 0
    assert pool.stream_advances == 0


@pytest.mark.parametrize("caller", ["freesolo", "trusted_internal"])
@pytest.mark.parametrize(
    "rates",
    [
        ("malformed", "0.19", "0.023"),
        ("-0.01", "0.19", "0.023"),
        ("NaN", "0.19", "0.023"),
        ("Infinity", "0.19", "0.023"),
    ],
    ids=["malformed", "negative", "nan", "infinite"],
)
def test_invalid_freesolo_price_rejects_before_dispatch(
    monkeypatch, rates: tuple[str, str, str], caller: str
) -> None:
    record = _revision()
    pool = _Pool()
    monkeypatch.setitem(_FREESOLO_USD_PER_MTOK, record.base_model, rates)
    app = _app(record, pool)
    with TestClient(app, headers=_headers(caller)) as client:
        response = client.post(
            "/generate", json={"adapter_id": record.adapter_id, "prompt": "hello"}
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "durable serving accounting unavailable"}
    assert pool.generate_calls == 0


@pytest.mark.parametrize(
    "route", ["generate", "adapter_generate", "openai_nonstream", "openai_stream"]
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("promptTokenUsd", "malformed"),
        ("promptTokenUsd", "-0.1"),
        ("cachedPromptTokenUsd", "NaN"),
        ("completionTokenUsd", "Infinity"),
        ("requestUsd", "1e-6"),
    ],
    ids=[
        "prompt-malformed",
        "prompt-negative",
        "cached-nan",
        "completion-infinite",
        "request-exponent",
    ],
)
def test_openrouter_malformed_price_http_matrix_rejects_before_dispatch(
    field: str, value: Any, route: str
) -> None:
    record = _base_record()
    pool = _Pool()
    raw = {
        "promptTokenUsd": "0.000001",
        "cachedPromptTokenUsd": "0.0000005",
        "completionTokenUsd": "0.000002",
        "requestUsd": None,
    }
    raw[field] = value
    snapshot = AcceptedPriceSnapshot.model_construct(**raw)
    principal = _openrouter_principal(snapshot)

    async def authorize(_token: str, _adapter_id: str, _scope: Any = None) -> AuthorizedTraffic:
        return AuthorizedTraffic(principal=principal)

    app = _app(record, pool, authorizer=authorize)
    request_route = "openai" if route.startswith("openai") else route
    path, body = _request_case(request_route, record, stream=route == "openai_stream")
    with TestClient(app, headers=_headers()) as client:
        response = client.post(path, json=body)

    assert response.status_code == 503
    assert response.json() == {"detail": "durable serving accounting unavailable"}
    assert pool.generate_calls == 0
    assert pool.stream_constructions == 0
    assert pool.stream_advances == 0


@pytest.mark.parametrize("missing", ["promptTokenUsd", "completionTokenUsd"])
def test_openrouter_missing_required_price_rejects_before_dispatch(missing: str) -> None:
    record = _base_record()
    pool = _Pool()
    raw = {
        "promptTokenUsd": "0.000001",
        "cachedPromptTokenUsd": None,
        "completionTokenUsd": "0.000002",
        "requestUsd": None,
    }
    raw.pop(missing)
    principal = _openrouter_principal(AcceptedPriceSnapshot.model_construct(**raw))

    async def authorize(_token: str, _adapter_id: str, _scope: Any = None) -> AuthorizedTraffic:
        return AuthorizedTraffic(principal=principal)

    app = _app(record, pool, authorizer=authorize)
    with TestClient(app, headers=_headers()) as client:
        response = client.post(
            "/generate", json={"adapter_id": record.adapter_id, "prompt": "hello"}
        )

    assert response.status_code == 503
    assert pool.generate_calls == 0


def test_openrouter_optional_none_prices_are_preserved() -> None:
    record = _revision()
    principal = _openrouter_principal(
        AcceptedPriceSnapshot(
            promptTokenUsd="0.000001",
            cachedPromptTokenUsd=None,
            completionTokenUsd="0.000002",
            requestUsd=None,
        )
    )

    captured = capture_authoritative_price(principal, record)

    assert captured.snapshot["cachedPromptTokenUsd"] is None
    assert captured.snapshot["requestUsd"] is None


def test_openrouter_stream_succeeds_without_local_freesolo_price(monkeypatch) -> None:
    record = _base_record()
    pool = _Pool()
    store = RecordingUsageStore()
    principal = _openrouter_principal()

    async def authorize(_token: str, _adapter_id: str, _scope: Any = None) -> AuthorizedTraffic:
        return AuthorizedTraffic(principal=principal)

    monkeypatch.delitem(_FREESOLO_USD_PER_MTOK, record.base_model)
    app = _app(record, pool, store=store, authorizer=authorize)
    path, body = _request_case("openai", record, stream=True)
    with (
        TestClient(app, headers=_headers()) as client,
        client.stream("POST", path, json=body) as response,
    ):
        assert response.status_code == 200
        response.read()

    assert pool.stream_constructions == 1
    assert pool.stream_advances == 1
    assert len(store.captured) == 1
    assert len(store.finalized) == 1
    assert store.captured[0].price.source == "openrouter_admission"
    assert store.finalized[0].price.snapshot == principal.acceptedPriceSnapshot.model_dump(
        mode="json"
    )


@pytest.mark.parametrize("route", ["adapter_generate", "openai_nonstream", "openai_stream"])
def test_unsupported_model_remains_400_before_price_or_dispatch(monkeypatch, route: str) -> None:
    record = _revision(base_model="unsupported/model")
    pool = _Pool()

    def unreachable_capture(*_args, **_kwargs):
        pytest.fail("price capture must be unreachable")

    monkeypatch.setattr(context_module, "capture_authoritative_price", unreachable_capture)
    app = _app(record, pool)
    request_route = "openai" if route.startswith("openai") else route
    path, body = _request_case(request_route, record, stream=route == "openai_stream")
    with TestClient(app, headers=_headers()) as client:
        response = client.post(path, json=body)

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Unsupported base model: unsupported/model")
    assert pool.generate_calls == 0
    assert pool.stream_constructions == 0
    assert pool.stream_advances == 0


@pytest.mark.parametrize("route", ["generate", "adapter_generate", "openai"])
def test_nonstream_finalization_failure_terminalizes_identical_attested_event_once(
    route: str,
) -> None:
    record = _revision()

    class FinalizeResponseLostStore(RecordingUsageStore):
        def __init__(self) -> None:
            super().__init__()
            self.finalize_attempts = []

        async def finalize(self, event) -> None:
            self.finalize_attempts.append(event)
            raise UsageOutboxError("finalize response lost after retries")

    store = FinalizeResponseLostStore()
    app = _app(record, _Pool(), store=store)
    path, body = _request_case(route, record, stream=False)

    with TestClient(app, headers=_headers()) as client:
        response = client.post(path, json=body)

    assert response.status_code == 503
    assert response.json() == {"detail": "durable serving accounting unavailable"}
    assert len(store.captured) == 1
    assert len(store.finalize_attempts) == 1
    assert len(store.failed) == 1
    admission = store.captured[0]
    final = store.finalize_attempts[0]
    failed, code = store.failed[0]
    assert final.identity == admission.identity
    assert failed.identity == admission.identity
    assert failed.price is admission.price
    assert failed.rpc_payload() == final.rpc_payload()
    assert failed.attestation_evidence == {"checkpoint_id": record.adapter_id}
    assert code == "finalization_failed"


def test_captured_price_identity_survives_nonstream_finalization(monkeypatch) -> None:
    record = _revision()
    pool = _Pool()
    store = RecordingUsageStore()
    captured = CapturedPrice(
        source="identity-test",
        version="v1",
        snapshot={
            "prompt_token_usd": "0.1",
            "cached_prompt_token_usd": "0.01",
            "completion_token_usd": "0.2",
        },
    )

    monkeypatch.setattr(
        context_module.ServingContext,
        "capture_price",
        lambda *_args, **_kwargs: captured,
    )
    app = _app(record, pool, store=store)
    with TestClient(app, headers=_headers()) as client:
        response = client.post(
            "/generate", json={"adapter_id": record.adapter_id, "prompt": "hello"}
        )

    assert response.status_code == 200
    assert len(store.finalized) == 1
    assert store.finalized[0].price is captured


def test_captured_price_identity_survives_stream_initial_and_final_events(monkeypatch) -> None:
    record = _revision()
    pool = _Pool()
    store = RecordingUsageStore()
    captured = CapturedPrice(
        source="identity-test",
        version="v1",
        snapshot={
            "prompt_token_usd": "0.1",
            "cached_prompt_token_usd": "0.01",
            "completion_token_usd": "0.2",
        },
    )

    monkeypatch.setattr(
        context_module.ServingContext,
        "capture_price",
        lambda *_args, **_kwargs: captured,
    )
    app = _app(record, pool, store=store)
    path, body = _request_case("openai", record, stream=True)
    with (
        TestClient(app, headers=_headers()) as client,
        client.stream("POST", path, json=body) as response,
    ):
        assert response.status_code == 200
        response.read()

    assert store.captured[0].price is captured
    assert store.finalized[0].price is captured


def test_captured_price_identity_survives_disconnect_failure_event() -> None:
    async def scenario() -> tuple[RecordingUsageStore, CapturedPrice]:
        record = _revision()
        store = RecordingUsageStore()
        captured = CapturedPrice(
            source="identity-test",
            version="v1",
            snapshot={
                "prompt_token_usd": "0.1",
                "cached_prompt_token_usd": "0.01",
                "completion_token_usd": "0.2",
            },
        )
        identity = RequestIdentity(request_id=new_generation_id(), correlation_id="correlation-1")
        first = attest(
            record,
            {
                "type": "ready",
                "prompt_token_ids": [1],
                "completion_token_ids": [],
                "prompt_tokens": 1,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cached_tokens_reported": True,
                "reasoning_tokens": 0,
                "thinking": False,
                "request_id": identity.request_id,
                "checkpoint": record.checkpoint,
            },
        )
        session = build_usage_session(
            store,
            identity,
            principal_for_external_org("org-1"),
            record,
            record,
            price=captured,
            deployment_id="deployment-1",
            serving_release="release-1",
            captured_at=datetime.now(UTC),
        )

        async def events():
            yield first

        await inference_routes._discard_prepared_stream(session, events())
        return store, captured

    store, captured = asyncio.run(scenario())
    assert len(store.failed) == 1
    assert store.failed[0][0].price is captured
    assert store.failed[0][1] == "client_disconnected"


def test_capture_authoritative_price_returns_openrouter_snapshot_object_values() -> None:
    record = _revision()
    principal = _openrouter_principal()

    captured = capture_authoritative_price(principal, record)

    assert captured.source == "openrouter_admission"
    assert captured.version == principal.providerCatalogDigest
    assert captured.snapshot == principal.acceptedPriceSnapshot.model_dump(mode="json")
