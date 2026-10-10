from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from flash.serving.src.accounting.usage import (
    _FREESOLO_USD_PER_MTOK,
    FREESOLO_PRICING_SOURCE,
    build_usage_session,
    freesolo_price,
    new_generation_id,
    principal_for_external_org,
)
from flash.serving.src.accounting.usage_facts import usage_facts
from flash.serving.src.accounting.usage_outbox import (
    AuthoritativeProviderDay,
    DurableUsageOutbox,
    FreesoloOrgTrafficPrincipal,
    OfflineUsageStore,
    OutboxSnapshot,
    ProviderSettlementRecord,
    ReconciliationDayResult,
    ReconciliationResult,
    RequestIdentity,
    UsageEvent,
    UsageOutboxError,
    _settlement_principal,
)
from flash.serving.src.engine.model_config import base_models
from flash.serving.src.http.inference_routes import _discard_prepared_stream
from flash.serving.src.http.router import AdapterRouter, build_serving_app
from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.io.streaming import (
    _produce_openai_chat_stream,
    _sse,
    _StreamOutput,
    openai_chat_stream,
)
from flash.serving.src.store.settings import Settings
from tests.serving.conftest import attest

BASE_MODEL = "Qwen/Qwen3.5-9B"


def _revision() -> AdapterRecord:
    run_id = "accounting"
    artifact_revision = hashlib.sha1(run_id.encode()).hexdigest()
    checkpoint_id = f"{run_id}/final"
    return AdapterRecord.model_validate(
        {
            "adapter_id": checkpoint_id,
            "repo_id": "org/accounting",
            "org_id": "org-1",
            "base_model": BASE_MODEL,
            "checkpoint": checkpoint_id,
            "thinking": False,
            "run_id": run_id,
            "checkpoint_step": None,
            "artifact_revision": artifact_revision,
            "artifact_digest": hashlib.sha256(b"accounting-artifact").hexdigest(),
            "artifact_fingerprint": hashlib.sha256(b"accounting-binding").hexdigest(),
            "lora_rank": 16,
        }
    )


class _Pool:
    def __init__(self) -> None:
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
                "reasoning_tokens": 0,
                "thinking": False,
                "inference_time_seconds": 0.01,
                "request_id": generation_id,
                "engine_replica_id": "replica-1",
                "checkpoint": record.checkpoint,
            },
        )

    async def stream_generate(
        self,
        base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ):
        del base_model, expected_checkpoint
        generation_id = payload.generation_id
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

    async def register(self, base_model: str, record: AdapterRecord) -> None:
        del base_model, record

    async def unregister(
        self,
        base_model: str,
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> None:
        del base_model, adapter_id, expected_generation


class _Store(OfflineUsageStore):
    enabled = True

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.captured: list[UsageEvent] = []
        self.finalized: list[UsageEvent] = []

    async def capture(self, event: UsageEvent) -> None:
        if self.fail:
            raise UsageOutboxError("capture_failed")
        self.captured.append(event)

    async def finalize(self, event: UsageEvent) -> None:
        if self.fail:
            raise UsageOutboxError("capture_failed")
        self.finalized.append(event)

    async def fail(self, event: UsageEvent, code: str) -> None:
        if self.fail:
            raise UsageOutboxError("capture_failed")
        self.captured.append(event)


def test_freesolo_prices_match_committed_catalog_fixed_point_contract() -> None:
    # The charged per-token rate is the launch per-Mtok price divided by 1e6 with NO markup applied,
    # so each value below is exactly its published rate shifted six places. Retired models are absent:
    # the table is exactly the active hosted set.
    expected = {
        "Qwen/Qwen3.5-9B": {
            "prompt_token_usd": "0.000000095",
            "cached_prompt_token_usd": "0.0000000276",
            "completion_token_usd": "0.0000001425",
        },
        "Qwen/Qwen3.8-27B": {
            "prompt_token_usd": "0.0000003325",
            "cached_prompt_token_usd": "0.00000003325",
            "completion_token_usd": "0.0000024225",
        },
        "Qwen/Qwen3.6-35B-A3B": {
            "prompt_token_usd": "0.000000095",
            "cached_prompt_token_usd": "0.0000000475",
            "completion_token_usd": "0.0000009025",
        },
    }
    decimal_string = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")

    assert set(_FREESOLO_USD_PER_MTOK) == set(expected)
    assert set(_FREESOLO_USD_PER_MTOK) == set(base_models())
    for model, snapshot in expected.items():
        actual = freesolo_price(model).snapshot
        assert actual == snapshot
        assert all(decimal_string.fullmatch(value) for value in actual.values())
        assert all("e" not in value.lower() for value in actual.values())


def test_generation_id_format_and_public_identity_separation() -> None:
    generation_id = new_generation_id()

    assert len(generation_id) == 38
    assert generation_id.startswith("fsgen-")
    assert generation_id[6:].isalnum()
    assert generation_id[6:] == generation_id[6:].lower()
    identity = RequestIdentity(
        request_id=generation_id,
        correlation_id="correlation-1",
        openai_completion_id="chatcmpl-1",
    )
    assert identity.request_id not in {
        identity.correlation_id,
        identity.openai_completion_id,
    }


@pytest.mark.parametrize(
    "invalid",
    [
        "4c074d3e-f412-4ead-9df7-ab2fc838b221",
        "fsgen-4C074D3EF4124EAD9DF7AB2FC838B221",
        "4c074d3ef4124ead9df7ab2fc838b221",
        "caller-request-id",
        "or-generation-1",
        "correlation-1",
        "chatcmpl-1",
    ],
)
def test_durable_identity_rejects_non_internal_generation_ids(invalid: str) -> None:
    with pytest.raises(ValueError, match="fsgen generation id format"):
        RequestIdentity(request_id=invalid, correlation_id="correlation-1")


def test_durable_identity_rejects_reused_correlation_or_public_id() -> None:
    generation_id = new_generation_id()
    with pytest.raises(ValueError, match="distinct"):
        RequestIdentity(request_id=generation_id, correlation_id=generation_id)
    with pytest.raises(ValueError, match="distinct"):
        RequestIdentity(
            request_id=generation_id,
            correlation_id="correlation-1",
            openai_completion_id=generation_id,
        )


@pytest.mark.parametrize("token_ids", [{}, {"prompt_token_ids": [], "completion_token_ids": []}])
def test_empty_or_omitted_token_ids_preserve_resolved_scalar_counts(
    token_ids: dict[str, list[int]],
) -> None:
    facts = usage_facts(
        {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "cached_tokens": 2,
            "cached_tokens_reported": True,
            "reasoning_tokens": 0,
            **token_ids,
        }
    )

    assert facts.prompt_tokens == 5
    assert facts.completion_tokens == 3
    assert facts.cached_tokens == 2
    assert facts.cached_tokens <= facts.prompt_tokens


def test_capacity_facts_use_top_level_optional_rpc_fields() -> None:
    event = _usage_event()
    event = replace(
        event,
        facts=replace(
            event.facts,
            time_to_first_token_seconds=0.75,
            queue_wait_seconds=0.25,
            replica_in_flight_requests_at_admission=3,
            replica_boot_duration_seconds=91.5,
            replica_freshly_booted=True,
        ),
    )

    payload = event.rpc_payload()

    assert payload["time_to_first_token_seconds"] == 0.75
    assert payload["queue_wait_seconds"] == 0.25
    assert payload["replica_in_flight_requests_at_admission"] == 3
    assert payload["replica_boot_duration_seconds"] == 91.5
    assert payload["replica_freshly_booted"] is True
    assert "capacity" not in payload["attestation_evidence"]


def test_invalid_or_missing_capacity_facts_are_omitted_without_raising() -> None:
    facts = usage_facts(
        {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
            "cached_tokens_reported": False,
            "reasoning_tokens": 0,
            "thinking": False,
            "time_to_first_token_seconds": "not-a-number",
            "queue_wait_seconds": -1,
            "replica_in_flight_requests_at_admission": True,
            "replica_boot_duration_seconds": float("inf"),
            "replica_freshly_booted": "yes",
        }
    )
    payload = replace(_usage_event(), facts=facts).rpc_payload()

    assert "time_to_first_token_seconds" not in payload
    assert "queue_wait_seconds" not in payload
    assert "replica_in_flight_requests_at_admission" not in payload
    assert "replica_boot_duration_seconds" not in payload
    assert "replica_freshly_booted" not in payload


def test_nonstream_capture_is_awaited_before_successful_response() -> None:
    record = _revision()
    pool = _Pool()
    store = _Store()

    async def authorize(_token: str, _adapter_id: str, _scope: dict | None = None) -> str:
        return "org-1"

    app = build_serving_app(
        pool,
        AdapterRouter([record]),
        usage_store=store,
        chat_authorizer=authorize,
        deployment_id="deployment-1",
        deployment_sha="release-1",
    )
    with TestClient(app, headers={"Authorization": "Bearer user-key"}) as client:
        response = client.post(
            "/generate", json={"adapter_id": record.adapter_id, "prompt": "hello"}
        )

    assert response.status_code == 200
    assert len(store.finalized) == 1
    event = store.finalized[0]
    assert event.identity.request_id == pool.generation_id
    assert event.principal == FreesoloOrgTrafficPrincipal(orgId="org-1")
    assert event.target.public_model_id == record.adapter_id
    assert event.target.checkpoint_id == record.checkpoint
    assert event.target.artifact_fingerprint == record.artifact_fingerprint
    assert event.target.artifact_fingerprint != record.artifact_digest
    assert event.facts.prompt_tokens == 2
    assert event.facts.completion_tokens == 1
    assert event.facts.cached_tokens == 1


def test_stream_captures_in_progress_before_response_and_finalizes_same_id() -> None:
    record = _revision()
    pool = _Pool()
    store = _Store()

    async def authorize(_token: str, _adapter_id: str, _scope: dict | None = None) -> str:
        return "org-1"

    app = build_serving_app(
        pool,
        AdapterRouter([record]),
        usage_store=store,
        chat_authorizer=authorize,
    )
    with (
        TestClient(app, headers={"Authorization": "Bearer user-key"}) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": record.adapter_id,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response,
    ):
        assert response.status_code == 200
        assert len(store.captured) == 1
        response.read()

    assert len(store.finalized) == 1
    assert store.captured[0].identity.request_id == pool.generation_id
    assert store.finalized[0].identity.request_id == pool.generation_id
    assert store.captured[0].facts.completion_tokens == 0
    assert store.finalized[0].facts.completion_tokens == 1


def test_stream_finalization_preserves_first_event_attestation() -> None:
    record = _revision()
    store = _Store()
    identity = RequestIdentity(
        request_id=new_generation_id(),
        correlation_id="correlation-1",
    )
    ready = {
        "type": "ready",
        "prompt_token_ids": [1],
        "completion_token_ids": [],
        "cached_tokens_reported": True,
        "reasoning_tokens": 0,
        "thinking": False,
        "request_id": identity.request_id,
        "checkpoint": record.checkpoint,
        "lora_request_adapter": record.adapter_id,
    }
    final = {
        "type": "final",
        "finish_reason": "stop",
        "prompt_token_ids": [1],
        "completion_token_ids": [2],
        "cached_tokens_reported": True,
        "reasoning_tokens": 0,
        "thinking": False,
        "request_id": identity.request_id,
        "checkpoint": record.checkpoint,
    }
    session = build_usage_session(
        store,
        identity,
        FreesoloOrgTrafficPrincipal(orgId="org-1"),
        record,
        record,
        ready,
        deployment_id="deployment-1",
        serving_release="release-1",
        captured_at=datetime.now(UTC),
    )

    asyncio.run(session.finalize(final))

    assert store.finalized[0].attestation_evidence == {"checkpoint_id": record.adapter_id}


def test_stream_capture_failure_before_headers_returns_controlled_503() -> None:
    record = _revision()

    async def authorize(_token: str, _adapter_id: str, _scope: dict | None = None) -> str:
        return "org-1"

    app = build_serving_app(
        _Pool(),
        AdapterRouter([record]),
        usage_store=_Store(fail=True),
        chat_authorizer=authorize,
    )
    with TestClient(app, headers={"Authorization": "Bearer user-key"}) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": record.adapter_id,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "durable serving accounting unavailable"}


def test_thinking_request_is_rejected_before_engine_dispatch() -> None:
    record = _revision().model_copy(update={"thinking": True})
    pool = _Pool()

    async def authorize(_token: str, _adapter_id: str, _scope: dict | None = None) -> str:
        return "org-1"

    app = build_serving_app(
        pool,
        AdapterRouter([record]),
        usage_store=_Store(),
        chat_authorizer=authorize,
    )
    with TestClient(app, headers={"Authorization": "Bearer user-key"}) as client:
        response = client.post(
            "/generate", json={"adapter_id": record.adapter_id, "prompt": "hello"}
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "thinking generation accounting is unavailable"}
    assert pool.generation_id is None


def test_nonstream_capture_failure_returns_controlled_503() -> None:
    record = _revision()

    async def authorize(_token: str, _adapter_id: str, _scope: dict | None = None) -> str:
        return "org-1"

    app = build_serving_app(
        _Pool(),
        AdapterRouter([record]),
        usage_store=_Store(fail=True),
        chat_authorizer=authorize,
    )
    with TestClient(app, headers={"Authorization": "Bearer user-key"}) as client:
        response = client.post(
            "/generate", json={"adapter_id": record.adapter_id, "prompt": "hello"}
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "durable serving accounting unavailable"}


def test_every_settled_event_carries_an_attributed_org() -> None:
    record = _revision()
    identity = RequestIdentity(
        request_id=new_generation_id(),
        correlation_id="correlation-1",
    )
    result = attest(
        record,
        {
            "prompt_token_ids": [1],
            "completion_token_ids": [2],
            "reasoning_tokens": 0,
            "thinking": False,
            "request_id": identity.request_id,
            "checkpoint": record.checkpoint,
        },
    )
    session = build_usage_session(
        OfflineUsageStore(),
        identity,
        principal_for_external_org("org-1"),
        record,
        record,
        result,
        deployment_id="deployment-1",
        serving_release="release-1",
        captured_at=datetime.now(UTC),
    )

    payload = session.event(result).rpc_payload()

    assert payload["traffic_principal_kind"] == "freesolo_org"
    assert payload["org_id"] == "org-1"
    assert payload["checkpoint_id"] == record.adapter_id
    assert payload["public_model_id"] == record.adapter_id
    assert payload["pricing_source"] == FREESOLO_PRICING_SOURCE
    # no marketplace-specific settlement surface survives on the wire.
    for absent in (
        "traffic_source",
        "openrouter_request_id",
        "openrouter_generation_id",
        "upstream_id",
        "provider_catalog_digest",
        "accepted_price_snapshot",
        "quoted_provider_amount_micro_usd",
    ):
        assert absent not in payload


def test_trusted_internal_settlement_requires_explicit_attribution() -> None:
    row = _claimed_row()
    row["traffic_principal_kind"] = "trusted_internal"
    row["billing_attribution_explicit"] = False

    with pytest.raises(UsageOutboxError, match="usage_principal_invalid"):
        _settlement_principal(row)


def test_settlement_rejects_a_row_without_an_org() -> None:
    row = _claimed_row()
    row["org_id"] = None

    with pytest.raises(UsageOutboxError, match="usage_principal_invalid"):
        _settlement_principal(row)


def _usage_event() -> UsageEvent:
    record = _revision()
    identity = RequestIdentity(
        request_id="fsgen-00000000000000000000000000000001",
        correlation_id="correlation-1",
    )
    result = attest(
        record,
        {
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [],
            "prompt_tokens": 2,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cached_tokens_reported": True,
            "reasoning_tokens": 0,
            "thinking": False,
            "request_id": identity.request_id,
            "engine_replica_id": "replica-1",
            "checkpoint": record.checkpoint,
        },
    )
    session = build_usage_session(
        OfflineUsageStore(),
        identity,
        FreesoloOrgTrafficPrincipal(orgId="org-1"),
        record,
        record,
        result,
        deployment_id="deployment-1",
        serving_release="release-1",
        captured_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
    )
    return session.event(result)


def _outbox_settings() -> Settings:
    return Settings(
        _env_file=None,
        SUPABASE_URL="https://project.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="sb_secret_test",
        PLATFORM_BACKEND_URL="https://api.example.com",
        FREESOLO_INTERNAL_KEY="internal-key",
        FREESOLO_DEPLOYMENT_ID="deployment-1",
    )


def _claimed_row(
    *,
    attempt_count: int = 1,
    state: str = "leased",
    outbox_id: str = "10000000-0000-4000-8000-000000000001",
) -> dict[str, Any]:
    return {
        "id": outbox_id,
        "state": state,
        "request_id": new_generation_id(),
        "attempt_count": attempt_count,
        "traffic_principal_kind": "freesolo_org",
        "org_id": "org-1",
        "billing_attribution_explicit": False,
        "public_model_id": "public/model",
    }


class _QueuedClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def post(
        self, url: str, *, headers: dict[str, str] | None = None, json: dict[str, Any]
    ) -> httpx.Response:
        del headers
        self.calls.append((url, json))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = await response()
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(
            response[0],
            json=response[1],
            request=httpx.Request("POST", url),
        )


async def _no_sleep(_seconds: float) -> None:
    return None


def test_capture_uses_deployment_owner_process_epoch_and_server_timing() -> None:
    client = _QueuedClient(
        [
            (
                200,
                [
                    {
                        "outbox_id": "10000000-0000-4000-8000-000000000001",
                        "state": "in_progress",
                        "replay": False,
                        "lease_seconds": 120,
                        "heartbeat_seconds": 20,
                    }
                ],
            )
        ]
    )
    outbox = DurableUsageOutbox(_outbox_settings(), client=client, worker_id="delivery-worker-1")
    event = _usage_event()

    asyncio.run(outbox.capture(event))

    url, payload = client.calls[0]
    assert url.endswith("/rpc/capture_serving_usage")
    assert payload["p_event"]["cached_tokens_reported"] is True
    assert payload["p_event"]["captured_at"] == "2026-08-24T12:00:00+00:00"
    assert payload["p_generation_owner_id"].startswith("fsrouter-")
    assert len(payload["p_generation_owner_id"]) == 41
    assert payload["p_generation_owner_id"] != "delivery-worker-1"
    assert payload["p_generation_owner_epoch"]
    assert outbox._heartbeat_seconds == 20
    assert event.identity.request_id in outbox._active_generations


def test_slow_capture_starts_full_heartbeat_lease_after_rpc_success() -> None:
    event = _usage_event()
    now = [datetime(2026, 8, 24, 12, 0, tzinfo=UTC)]
    sleeps: list[float] = []

    async def delayed_capture() -> tuple[int, Any]:
        now[0] += timedelta(seconds=119)
        return (
            200,
            [
                {
                    "state": "in_progress",
                    "lease_seconds": 120,
                    "heartbeat_seconds": 20,
                }
            ],
        )

    async def advance_clock(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += timedelta(seconds=seconds)

    client = _QueuedClient(
        [
            delayed_capture,
            httpx.ConnectError("transient heartbeat failure"),
            httpx.ConnectError("transient heartbeat failure"),
            (503, {"error": "temporarily unavailable"}),
            (
                200,
                [
                    {
                        "request_id": event.identity.request_id,
                        "generation_lease_expires_at": "2026-08-24T12:04:00Z",
                    }
                ],
            ),
            (200, [{"state": "pending", "replay": False}]),
        ]
    )
    outbox = DurableUsageOutbox(
        _outbox_settings(),
        client=client,
        worker_id="worker-1",
        sleep=advance_clock,
        clock=lambda: now[0],
    )

    async def run() -> None:
        await outbox.capture(event)
        assert outbox._generation_lease_deadlines[event.identity.request_id] == datetime(
            2026, 8, 24, 12, 3, 59, tzinfo=UTC
        )
        await outbox._heartbeat_active_generations_with_retry()
        await outbox.finalize(event)

    asyncio.run(run())

    assert sleeps == [0.05, 20.0, 20.0]
    assert client.calls[-1][0].endswith("/rpc/finalize_serving_usage")
    assert outbox._background_error is None


def test_finalize_uses_exact_owner_epoch_and_stops_heartbeating() -> None:
    client = _QueuedClient(
        [
            (
                200,
                [
                    {
                        "state": "in_progress",
                        "lease_seconds": 120,
                        "heartbeat_seconds": 20,
                    }
                ],
            ),
            (200, [{"state": "pending", "replay": False}]),
        ]
    )
    outbox = DurableUsageOutbox(_outbox_settings(), client=client, worker_id="worker-1")
    event = _usage_event()

    async def run() -> None:
        await outbox.capture(event)
        await outbox.finalize(event)

    asyncio.run(run())

    _, capture = client.calls[0]
    finalize_url, finalize = client.calls[1]
    assert finalize_url.endswith("/rpc/finalize_serving_usage")
    assert finalize["p_generation_owner_id"] == capture["p_generation_owner_id"]
    assert finalize["p_generation_owner_epoch"] == capture["p_generation_owner_epoch"]
    assert event.identity.request_id not in outbox._active_generations


def test_finalize_replays_identical_rpc_after_committed_response_is_lost() -> None:
    event = _usage_event()
    client = _QueuedClient(
        [
            httpx.ConnectError("response lost after commit"),
            (200, [{"state": "pending", "replay": True}]),
        ]
    )
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", sleep=_no_sleep
    )
    outbox._active_generations.add(event.identity.request_id)

    asyncio.run(outbox.finalize(event))

    assert len(client.calls) == 2
    assert client.calls[0] == client.calls[1]
    assert event.identity.request_id not in outbox._active_generations


@pytest.mark.parametrize(
    ("heartbeat_failures", "expected_sleeps"),
    [
        (
            [
                httpx.ConnectError("transient heartbeat failure"),
                httpx.ConnectError("transient heartbeat failure"),
            ],
            [0.05, 20.0],
        ),
        ([(503, {"error": "temporarily unavailable"})], [20.0]),
    ],
)
def test_transient_heartbeat_failure_recovers_without_poisoning_terminal_rpc(
    heartbeat_failures: list[Any], expected_sleeps: list[float]
) -> None:
    event = _usage_event()
    fixed_now = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    client = _QueuedClient(
        [
            *heartbeat_failures,
            (
                200,
                [
                    {
                        "request_id": event.identity.request_id,
                        "generation_lease_expires_at": "2026-08-24T12:02:00Z",
                    }
                ],
            ),
            (200, [{"state": "pending", "replay": False}]),
        ]
    )
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    outbox = DurableUsageOutbox(
        _outbox_settings(),
        client=client,
        worker_id="worker-1",
        sleep=record_sleep,
        clock=lambda: fixed_now,
    )
    outbox._heartbeat_seconds = 20
    outbox._generation_lease_seconds = 120
    outbox._active_generations.add(event.identity.request_id)
    outbox._generation_lease_deadlines[event.identity.request_id] = fixed_now + timedelta(
        seconds=120
    )

    async def run() -> None:
        await outbox._heartbeat_active_generations_with_retry()
        await outbox.finalize(event)

    asyncio.run(run())

    assert sleeps == expected_sleeps
    assert client.calls[-1][0].endswith("/rpc/finalize_serving_usage")
    assert outbox._background_error is None
    assert not outbox._stopping.is_set()
    assert event.identity.request_id not in outbox._active_generations


def test_generation_churn_does_not_postpone_existing_heartbeat_deadline() -> None:
    first = _usage_event()
    second = replace(
        first,
        identity=RequestIdentity(
            request_id="fsgen-00000000000000000000000000000002",
            correlation_id="correlation-2",
        ),
    )
    now = [datetime(2026, 8, 25, 12, 0, tzinfo=UTC)]
    sleeps_started: asyncio.Queue[float] = asyncio.Queue()
    release_sleep = asyncio.Event()

    async def controlled_sleep(seconds: float) -> None:
        await sleeps_started.put(seconds)
        await release_sleep.wait()
        release_sleep.clear()
        now[0] += timedelta(seconds=seconds)

    capture_result = (
        200,
        [
            {
                "state": "in_progress",
                "lease_seconds": 120,
                "heartbeat_seconds": 20,
            }
        ],
    )
    client = _QueuedClient(
        [
            *([capture_result] * 5),
            (
                200,
                [
                    {
                        "request_id": first.identity.request_id,
                        "generation_lease_expires_at": "2026-08-25T12:02:20Z",
                    }
                ],
            ),
        ]
    )
    outbox = DurableUsageOutbox(
        _outbox_settings(),
        client=client,
        worker_id="worker-1",
        sleep=controlled_sleep,
        clock=lambda: now[0],
    )

    async def run() -> None:
        await outbox.capture(first)
        outbox._heartbeat_worker = asyncio.create_task(outbox._run_heartbeat_worker())
        assert await sleeps_started.get() == 20
        for remaining in (16, 12, 8, 4):
            now[0] += timedelta(seconds=4)
            await outbox.capture(second)
            outbox.relinquish(second.identity.request_id)
            assert await sleeps_started.get() == remaining
        release_sleep.set()
        assert await sleeps_started.get() == 20
        await asyncio.sleep(0)
        heartbeat_calls = [
            payload
            for url, payload in client.calls
            if url.endswith("/rpc/heartbeat_serving_generation")
        ]
        assert heartbeat_calls == [
            {
                "p_generation_owner_id": outbox._generation_owner_id,
                "p_generation_owner_epoch": str(outbox._generation_owner_epoch),
                "p_request_ids": [first.identity.request_id],
            }
        ]
        outbox._stopping.set()
        outbox._heartbeat_wake.set()
        await outbox._heartbeat_worker
        outbox._heartbeat_worker = None

    asyncio.run(run())

    assert now[0] == datetime(2026, 8, 25, 12, 0, 20, tzinfo=UTC)
    assert first.identity.request_id in outbox._active_generations
    assert outbox._generation_lease_deadlines[first.identity.request_id] == datetime(
        2026, 8, 25, 12, 2, 20, tzinfo=UTC
    )
    assert outbox._background_error is None


def test_heartbeat_batches_exact_active_generation_authority() -> None:
    event = _usage_event()
    client = _QueuedClient(
        [
            (
                200,
                [
                    {
                        "request_id": event.identity.request_id,
                        "generation_lease_expires_at": "2026-08-24T12:02:00Z",
                    }
                ],
            )
        ]
    )
    outbox = DurableUsageOutbox(_outbox_settings(), client=client, worker_id="worker-1")
    outbox._active_generations.add(event.identity.request_id)

    asyncio.run(outbox._heartbeat_active_generations())

    url, payload = client.calls[0]
    assert url.endswith("/rpc/heartbeat_serving_generation")
    assert payload["p_request_ids"] == [event.identity.request_id]
    assert payload["p_generation_owner_id"].startswith("fsrouter-")
    assert payload["p_generation_owner_epoch"]


@pytest.mark.parametrize("terminal_operation", ["finalize", "fail"])
def test_heartbeat_ignores_generation_terminalized_before_terminal_rpc_returns(
    terminal_operation: str,
) -> None:
    event = _usage_event()
    heartbeat_entered = asyncio.Event()
    terminal_effect_committed = asyncio.Event()
    release_heartbeat = asyncio.Event()
    release_terminal_response = asyncio.Event()

    async def delayed_heartbeat() -> tuple[int, Any]:
        heartbeat_entered.set()
        await release_heartbeat.wait()
        return 200, []

    async def delayed_terminal_response() -> tuple[int, Any]:
        terminal_effect_committed.set()
        await release_terminal_response.wait()
        return 200, [{"state": "pending", "replay": False}]

    client = _QueuedClient([delayed_heartbeat, delayed_terminal_response])
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", sleep=_no_sleep
    )
    outbox._active_generations.add(event.identity.request_id)

    async def run() -> None:
        heartbeat = asyncio.create_task(outbox._heartbeat_active_generations())
        await heartbeat_entered.wait()
        terminal = asyncio.create_task(
            outbox.finalize(event)
            if terminal_operation == "finalize"
            else outbox.fail(event, "client_disconnected")
        )
        await terminal_effect_committed.wait()
        release_heartbeat.set()
        await heartbeat
        assert not terminal.done()
        release_terminal_response.set()
        await terminal

    asyncio.run(run())

    assert event.identity.request_id not in outbox._active_generations
    assert event.identity.request_id not in outbox._terminal_generations
    assert outbox._background_error is None


def test_background_failure_refuses_new_work_but_still_settles_admitted_work() -> None:
    """a dead delivery worker must not strand the charge for a request already served.

    admission is the gate that refuses new chargeable traffic. once a request has passed it and
    generated, the terminal rpcs are idempotent, so refusing to attempt one loses the charge for
    work the customer already received.
    """

    event = _usage_event()
    client = _QueuedClient([(200, [{"state": "completed", "replay": False}])])
    outbox = DurableUsageOutbox(
        _outbox_settings(),
        client=client,
        worker_id="worker-1",
    )
    outbox._background_error = RuntimeError("permanent delivery failure")

    with pytest.raises(UsageOutboxError):
        outbox.assert_healthy()

    asyncio.run(outbox.finalize(event))

    assert client.calls[-1][0].endswith("/rpc/finalize_serving_usage")


def test_shutdown_timeout_is_observable_without_erasing_generation(monkeypatch) -> None:
    import flash.serving.src.accounting.usage_outbox as usage_outbox_module

    event = _usage_event()
    entered = asyncio.Event()

    async def never_returns() -> tuple[int, Any]:
        entered.set()
        await asyncio.Event().wait()
        return 200, []

    monkeypatch.setattr(usage_outbox_module, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    client = _QueuedClient([never_returns])
    outbox = DurableUsageOutbox(_outbox_settings(), client=client, worker_id="worker-1")
    outbox._active_generations.add(event.identity.request_id)

    with pytest.raises(UsageOutboxError, match="usage_outbox_shutdown_failed"):
        asyncio.run(outbox.aclose())

    assert entered.is_set()
    assert event.identity.request_id in outbox._active_generations


def test_shutdown_failure_is_bounded_and_observable_without_erasing_generation() -> None:
    event = _usage_event()
    client = _QueuedClient([(500, {"error": "session failure"})])
    outbox = DurableUsageOutbox(_outbox_settings(), client=client, worker_id="worker-1")
    outbox._active_generations.add(event.identity.request_id)

    with pytest.raises(UsageOutboxError, match="usage_outbox_shutdown_failed"):
        asyncio.run(outbox.aclose())

    assert event.identity.request_id in outbox._active_generations
    assert client.calls[0][0].endswith("/rpc/fail_serving_generation_session")


def test_claimed_row_settlement_acknowledges_exact_identity() -> None:
    row = _claimed_row()
    client = _QueuedClient(
        [
            (
                200,
                {
                    "usageId": "usage-1",
                    "ledgerId": "ledger-1",
                    "priceVersion": "captured-v1",
                    "exactCostMicroUsd": 12,
                    "billedCents": 1,
                    "replay": False,
                },
            ),
            (200, [{"outbox_id": row["id"]}]),
        ]
    )
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", sleep=_no_sleep
    )

    asyncio.run(outbox._deliver(row))

    settlement_url, settlement = client.calls[0]
    assert settlement_url.endswith("/api/billing/serving-usage/durable")
    assert settlement == {
        "outboxId": row["id"],
        "workerId": "worker-1",
        "requestId": row["request_id"],
        "trafficPrincipal": {"kind": "freesolo_org", "orgId": "org-1"},
    }
    ack_url, ack = client.calls[1]
    assert ack_url.endswith("/rpc/acknowledge_serving_usage_delivered")
    assert ack["p_outbox_id"] == row["id"]
    assert ack["p_result"]["usage_id"] == "usage-1"


def test_ack_replays_identical_rpc_after_committed_response_is_lost() -> None:
    row = _claimed_row()
    client = _QueuedClient(
        [
            (
                200,
                {
                    "usageId": "usage-1",
                    "ledgerId": "ledger-1",
                    "priceVersion": "captured-v1",
                    "exactCostMicroUsd": 12,
                    "billedCents": 1,
                    "replay": False,
                },
            ),
            httpx.ConnectError("response lost after commit"),
            (200, [{"outbox_id": row["id"], "replay": True}]),
        ]
    )
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", sleep=_no_sleep
    )

    async def run() -> None:
        await outbox._deliver(row)
        await outbox.aclose()

    asyncio.run(run())

    assert client.calls[1] == client.calls[2]
    assert outbox._active_leases == set()
    assert len(client.calls) == 3


@pytest.mark.parametrize("status_code", [408, 429, 500, 503])
def test_retryable_delivery_is_rescheduled_deterministically(status_code: int) -> None:
    row = _claimed_row(attempt_count=2)
    fixed_now = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    client = _QueuedClient([(status_code, {}), (200, None)])
    outbox = DurableUsageOutbox(
        _outbox_settings(),
        client=client,
        worker_id="worker-1",
        sleep=_no_sleep,
        clock=lambda: fixed_now,
        jitter=lambda _id, _attempt, _base: 0.25,
    )

    asyncio.run(outbox._deliver(row))

    retry_url, retry = client.calls[1]
    assert retry_url.endswith("/rpc/reschedule_serving_usage")
    assert retry["p_error_code"] == f"http_{status_code}"
    assert retry["p_retry_at"] == "2026-08-24T12:00:02.250000+00:00"


def test_transport_failure_is_rescheduled_without_raw_exception() -> None:
    row = _claimed_row()
    client = _QueuedClient([httpx.ConnectError("secret host detail"), (200, None)])
    outbox = DurableUsageOutbox(
        _outbox_settings(),
        client=client,
        worker_id="worker-1",
        sleep=_no_sleep,
        clock=lambda: datetime(2026, 8, 24, tzinfo=UTC),
        jitter=lambda _id, _attempt, _base: 0.0,
    )

    asyncio.run(outbox._deliver(row))

    assert client.calls[1][1]["p_error_code"] == "transport"
    assert "secret" not in str(client.calls[1][1])


@pytest.mark.parametrize(
    ("status_code", "attempt_count", "reason"),
    [
        (400, 1, "permanent_http_400"),
        (404, 1, "permanent_http_404"),
        (500, 8, "delivery_attempts_exhausted"),
    ],
)
def test_permanent_or_exhausted_delivery_is_quarantined(
    status_code: int, attempt_count: int, reason: str
) -> None:
    row = _claimed_row(attempt_count=attempt_count)
    client = _QueuedClient([(status_code, {}), (200, None)])
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", sleep=_no_sleep
    )

    asyncio.run(outbox._deliver(row))

    quarantine_url, quarantine = client.calls[1]
    assert quarantine_url.endswith("/rpc/quarantine_serving_usage")
    assert quarantine["p_reason"] == reason


@pytest.mark.parametrize("settlement", [[], "invalid", None])
def test_non_object_settlement_response_is_quarantined_and_releases_lease(
    settlement: Any,
) -> None:
    row = _claimed_row()
    response_body = b"null" if settlement is None else json.dumps(settlement).encode()
    response = httpx.Response(
        200,
        content=response_body,
        request=httpx.Request("POST", "https://api.example.com/settlement"),
    )
    client = _QueuedClient([response, (200, None)])
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", sleep=_no_sleep
    )
    outbox._active_leases.add(row["id"])

    asyncio.run(outbox._deliver(row))

    quarantine_url, quarantine = client.calls[1]
    assert quarantine_url.endswith("/rpc/quarantine_serving_usage")
    assert quarantine["p_reason"] == "invalid_settlement_response"
    assert outbox._active_leases == set()


def test_startup_claim_recovers_expired_lease_and_delivers() -> None:
    row = _claimed_row(state="leased")
    delivered = asyncio.Event()

    async def acknowledge() -> tuple[int, Any]:
        delivered.set()
        return 200, [{"outbox_id": row["id"]}]

    client = _QueuedClient(
        [
            (200, []),
            (200, []),
            (200, [row]),
            (
                200,
                {
                    "usageId": "usage-1",
                    "ledgerId": None,
                    "priceVersion": "captured-v1",
                    "exactCostMicroUsd": 0,
                    "billedCents": 0,
                    "replay": False,
                },
            ),
            acknowledge,
            # the worker wakes itself after delivering a claimed batch, so it polls again before
            # shutdown. an exhausted queue would surface as a non-transient background failure.
            (200, []),
            (200, []),
            (200, []),
        ]
    )
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", poll_seconds=60
    )

    async def run() -> None:
        await outbox.start()
        await asyncio.wait_for(delivered.wait(), timeout=1)
        await outbox.aclose()

    asyncio.run(run())

    assert client.calls[0][0].endswith("/rpc/recover_stale_serving_generations")
    claim_url, claim = next(
        call for call in client.calls if call[0].endswith("/rpc/claim_serving_usage_batch")
    )
    assert claim_url.endswith("/rpc/claim_serving_usage_batch")
    assert claim["p_worker_id"] == "worker-1"
    assert any(url.endswith("/rpc/acknowledge_serving_usage_delivered") for url, _ in client.calls)


def test_slow_delivery_shutdown_releases_every_single_row_claim() -> None:
    row = _claimed_row()
    entered = asyncio.Event()
    unblock = asyncio.Event()

    async def blocked_settlement() -> tuple[int, Any]:
        entered.set()
        await unblock.wait()
        return 200, {}

    client = _QueuedClient([(200, []), (200, []), (200, [row]), blocked_settlement, (200, None)])
    fixed_now = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    outbox = DurableUsageOutbox(
        _outbox_settings(),
        client=client,
        worker_id="worker-1",
        poll_seconds=60,
        clock=lambda: fixed_now,
    )

    async def run() -> None:
        await outbox.start()
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert outbox._active_leases == {row["id"]}
        await outbox.aclose()
        unblock.set()

    asyncio.run(run())

    claim_calls = [
        payload for url, payload in client.calls if url.endswith("/rpc/claim_serving_usage_batch")
    ]
    assert claim_calls == [{"p_worker_id": "worker-1", "p_limit": 1, "p_lease_seconds": 60}]
    release_calls = [
        payload for url, payload in client.calls if url.endswith("/rpc/reschedule_serving_usage")
    ]
    assert release_calls == [
        {
            "p_outbox_id": row["id"],
            "p_worker_id": "worker-1",
            "p_retry_at": fixed_now.isoformat(),
            "p_error_code": "worker_shutdown",
        }
    ]
    assert not any(url.endswith("/rpc/quarantine_serving_usage") for url, _ in client.calls)
    assert outbox._active_leases == set()


def test_snapshot_is_immutable_and_does_not_mutate_state() -> None:
    client = _QueuedClient(
        [
            (
                200,
                {
                    "captured_at": "2026-08-24T12:00:00Z",
                    "states": {
                        "pending": 3,
                        "leased": 2,
                        "quarantined": 1,
                        "disputed": 4,
                    },
                    "due_pending": 2,
                    "expired_leases": 3,
                    "expired_generation_leases": 5,
                    "oldest_expired_generation_lease_age_seconds": 45,
                    "oldest_undelivered_age_seconds": 90,
                },
            )
        ]
    )
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", sleep=_no_sleep
    )

    snapshot = asyncio.run(outbox.snapshot())

    assert snapshot == OutboxSnapshot(
        captured_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
        pending=3,
        leased=2,
        quarantined=1,
        disputed=4,
        due_pending=2,
        expired_leases=3,
        expired_generation_leases=5,
        oldest_expired_generation_lease_age_seconds=45,
        oldest_undelivered_age_seconds=90,
    )
    with pytest.raises(AttributeError):
        snapshot.pending = 0  # type: ignore[misc]
    assert len(client.calls) == 1
    assert client.calls[0][0].endswith("/rpc/serving_usage_backlog_snapshot")


def _provider_record(index: int) -> ProviderSettlementRecord:
    return ProviderSettlementRecord(
        provider="freesolo",
        usage_date=date(2026, 8, 24),
        source="settlement_export",
        source_version="v1",
        provider_record_id=f"provider-{index}",
        request_id=f"fsgen-{index:032x}",
        provider_amount_micro_usd=index,
    )


@pytest.mark.parametrize("size", [0, 501])
def test_reconcile_batch_rejects_outside_contract_cardinality(size: int) -> None:
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=_QueuedClient([]), worker_id="worker-1", sleep=_no_sleep
    )

    with pytest.raises(ValueError, match="1 to 500"):
        asyncio.run(outbox.reconcile_batch([_provider_record(i) for i in range(size)]))


@pytest.mark.parametrize("size", [1, 500])
def test_reconcile_batch_accepts_contract_boundaries_and_validates_order(size: int) -> None:
    rows = [
        {
            "input_ordinal": index,
            "outbox_id": f"outbox-{index}",
            "reconciliation_status": "matched",
            "dispute_code": None,
            "replay": False,
        }
        for index in range(1, size + 1)
    ]
    client = _QueuedClient([(200, rows)])
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", sleep=_no_sleep
    )
    records = [_provider_record(i) for i in range(1, size + 1)]

    result = asyncio.run(outbox.reconcile_batch(records))

    assert len(result) == size
    assert result[0] == ReconciliationResult(1, "outbox-1", "matched", None, False)
    assert result[-1].input_ordinal == size
    url, payload = client.calls[0]
    assert url.endswith("/rpc/reconcile_serving_usage_batch")
    assert payload == {"p_records": [record.rpc_payload() for record in records]}


def test_reconcile_batch_rejects_unordered_results() -> None:
    client = _QueuedClient(
        [
            (
                200,
                [
                    {
                        "input_ordinal": 2,
                        "outbox_id": "outbox-1",
                        "reconciliation_status": "matched",
                        "dispute_code": None,
                        "replay": False,
                    }
                ],
            )
        ]
    )
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", sleep=_no_sleep
    )

    with pytest.raises(UsageOutboxError, match="usage_reconciliation_batch_invalid"):
        asyncio.run(outbox.reconcile(_provider_record(1)))


def test_finalize_authoritative_empty_day_uses_exact_typed_payload() -> None:
    client = _QueuedClient(
        [
            (
                200,
                [
                    {
                        "reconciliation_day_id": "day-1",
                        "reconciliation_state": "matched",
                        "status_reason": None,
                        "replay": False,
                    }
                ],
            )
        ]
    )
    outbox = DurableUsageOutbox(
        _outbox_settings(), client=client, worker_id="worker-1", sleep=_no_sleep
    )
    day = AuthoritativeProviderDay(
        provider="freesolo",
        usage_date=date(2026, 8, 24),
        source="settlement_export",
        source_version="v1",
        attestation_evidence={"digest": "sha256:empty-day"},
    )

    result = asyncio.run(outbox.finalize_reconciliation_day(day))

    assert result == ReconciliationDayResult("day-1", "matched", None, False)
    url, payload = client.calls[0]
    assert url.endswith("/rpc/finalize_serving_usage_reconciliation_day")
    assert payload == {
        "p_provider": "freesolo",
        "p_usage_date": "2026-08-24",
        "p_source": "settlement_export",
        "p_source_version": "v1",
        "p_attestation_evidence": {"digest": "sha256:empty-day"},
    }


class _StreamSession:
    def __init__(self, *, fail_finalize: bool = False) -> None:
        self.fail_finalize = fail_finalize
        self.captured: list[dict[str, Any]] = []
        self.finalized: list[dict[str, Any]] = []
        self.failed: list[tuple[dict[str, Any], str]] = []
        self.relinquished = 0

    async def capture(self, result: dict[str, Any]) -> None:
        self.captured.append(result.copy())

    async def finalize(self, result: dict[str, Any]) -> None:
        if self.fail_finalize:
            raise UsageOutboxError("finalize_failed")
        self.finalized.append(result.copy())

    async def fail(self, result: dict[str, Any], code: str) -> None:
        self.failed.append((result.copy(), code))

    def relinquish(self) -> None:
        self.relinquished += 1


def _stream_events(generation_id: str):
    async def events():
        yield {
            "type": "ready",
            "prompt_tokens": 2,
            "completion_tokens": 0,
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [],
            "reasoning_tokens": 0,
            "thinking": False,
            "request_id": generation_id,
        }
        yield {
            "type": "delta",
            "text": "answer",
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3],
            "reasoning_tokens": 0,
            "thinking": False,
            "request_id": generation_id,
        }
        yield {
            "type": "final",
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3],
            "reasoning_tokens": 0,
            "thinking": False,
            "request_id": generation_id,
        }

    return events()


def test_stream_finalizes_before_successful_terminal_usage_event() -> None:
    session = _StreamSession()
    generation_id = new_generation_id()
    record = _revision()

    async def run() -> list[bytes]:
        return [
            chunk
            async for chunk in openai_chat_stream(
                AdapterRouter([record]),
                record=record,
                events=_stream_events(generation_id),
                adapter_id=record.adapter_id,
                completion_id="chatcmpl-stream",
                created=123,
                include_usage=True,
                usage_session=session,  # type: ignore[arg-type]
            )
        ]

    chunks = asyncio.run(run())

    assert len(session.finalized) == 1
    assert session.finalized[0]["type"] == "final"
    assert session.captured == []
    terminal_index = next(i for i, chunk in enumerate(chunks) if b'"usage"' in chunk)
    assert chunks[terminal_index + 1] == _sse("[DONE]")


def test_final_event_wins_same_tick_disconnect_and_emits_one_terminal_pair() -> None:
    session = _StreamSession()
    generation_id = new_generation_id()
    record = _revision()
    output: asyncio.Queue[tuple[bytes | None, Exception | None]] = asyncio.Queue()
    disconnected = asyncio.Event()

    async def events():
        yield {
            "type": "ready",
            "prompt_tokens": 2,
            "completion_tokens": 0,
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [],
            "reasoning_tokens": 0,
            "thinking": False,
            "request_id": generation_id,
        }
        disconnected.set()
        yield {
            "type": "final",
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3],
            "reasoning_tokens": 0,
            "thinking": False,
            "request_id": generation_id,
        }

    async def run() -> list[bytes]:
        await _produce_openai_chat_stream(
            AdapterRouter([record]),
            output,
            disconnected,
            record=record,
            events=events(),
            adapter_id=record.adapter_id,
            completion_id="chatcmpl-race",
            created=123,
            include_usage=True,
            usage_session=session,  # type: ignore[arg-type]
            thinking=False,
        )
        chunks: list[bytes] = []
        while not output.empty():
            chunk, error = output.get_nowait()
            assert error is None
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())

    assert len(session.finalized) == 1
    assert session.finalized[0]["type"] == "final"
    assert session.failed == []
    assert sum(b'"usage"' in chunk for chunk in chunks) == 1
    assert chunks.count(_sse("[DONE]")) == 1
    assert chunks[-1] == _sse("[DONE]")


def test_stream_finalization_failure_emits_terminal_sse_error_and_snapshot() -> None:
    session = _StreamSession(fail_finalize=True)
    generation_id = new_generation_id()
    record = _revision()

    async def run() -> list[bytes]:
        return [
            chunk
            async for chunk in openai_chat_stream(
                AdapterRouter([record]),
                record=record,
                events=_stream_events(generation_id),
                adapter_id=record.adapter_id,
                completion_id="chatcmpl-stream",
                created=123,
                include_usage=True,
                usage_session=session,  # type: ignore[arg-type]
            )
        ]

    chunks = asyncio.run(run())

    assert not any(b'"usage"' in chunk for chunk in chunks)
    assert any(b'"type":"accounting_error"' in chunk for chunk in chunks)
    assert chunks[-1] == _sse("[DONE]")
    assert session.captured[-1]["type"] == "final"
    assert session.relinquished == 1


def test_failed_stream_finalization_relinquishes_heartbeat_for_stale_recovery() -> None:
    event = _usage_event()
    record = _revision()
    client = _QueuedClient(
        [
            (500, {"error": "finalize failed"}),
            (
                200,
                [
                    {
                        "state": "in_progress",
                        "lease_seconds": 120,
                        "heartbeat_seconds": 20,
                    }
                ],
            ),
            (200, []),
        ]
    )
    outbox = DurableUsageOutbox(_outbox_settings(), client=client, worker_id="worker-1")
    outbox._active_generations.add(event.identity.request_id)
    session = build_usage_session(
        outbox,
        event.identity,
        event.principal,
        record,
        record,
        {"checkpoint": record.checkpoint, "lora_request_adapter": record.adapter_id},
        deployment_id="deployment-1",
        serving_release="release-1",
        captured_at=event.captured_at,
    )

    async def run() -> list[bytes]:
        chunks = [
            chunk
            async for chunk in openai_chat_stream(
                AdapterRouter([record]),
                record=record,
                events=_stream_events(event.identity.request_id),
                adapter_id=record.adapter_id,
                completion_id="chatcmpl-stream",
                created=123,
                include_usage=True,
                usage_session=session,
            )
        ]
        await outbox._heartbeat_active_generations()
        await outbox.recover_stale_in_progress()
        return chunks

    chunks = asyncio.run(run())

    assert any(b'"type":"accounting_error"' in chunk for chunk in chunks)
    assert event.identity.request_id not in outbox._active_generations
    assert event.identity.request_id not in outbox._generation_lease_deadlines
    assert [call[0].rsplit("/", 1)[-1] for call in client.calls] == [
        "finalize_serving_usage",
        "capture_serving_usage",
        "recover_stale_serving_generations",
    ]


def test_pre_response_disconnect_persists_ready_snapshot_and_closes_iterator() -> None:
    session = _StreamSession()
    generation_id = new_generation_id()
    closed = asyncio.Event()

    async def events():
        try:
            yield {
                "type": "ready",
                "prompt_tokens": 2,
                "completion_tokens": 0,
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [],
                "reasoning_tokens": 0,
                "thinking": False,
                "request_id": generation_id,
            }
            await asyncio.Event().wait()
        finally:
            closed.set()

    asyncio.run(_discard_prepared_stream(session, events()))

    assert closed.is_set()
    assert session.failed[-1][0]["type"] == "ready"
    assert session.failed[-1][1] == "client_disconnected"
    assert session.finalized == []


def test_finish_makes_sentinel_space_after_blocked_terminal_refills_queue() -> None:
    output: asyncio.Queue[tuple[bytes | None, Exception | None]] = asyncio.Queue(maxsize=3)
    disconnected = asyncio.Event()
    stream_output = _StreamOutput(
        output,
        disconnected,
        completion_id="chatcmpl-blocked",
        created=123,
        adapter_id="adapter-1",
    )

    async def run() -> list[bytes | None]:
        for chunk in (b"old-1", b"old-2", b"old-3"):
            output.put_nowait((chunk, None))
        terminal = asyncio.create_task(stream_output.terminal(b"terminal"))
        await asyncio.sleep(0)
        assert not terminal.done()
        disconnected.set()
        output.get_nowait()
        await terminal
        assert output.full()
        await stream_output.finish()
        return [output.get_nowait()[0] for _ in range(output.qsize())]

    queued = asyncio.run(run())

    assert queued == [b"old-3", b"terminal", None]


def test_full_stream_queue_disconnect_does_not_block_cleanup_sentinel() -> None:
    session = _StreamSession()
    generation_id = new_generation_id()
    record = _revision()
    fourth_delta_requested = asyncio.Event()
    closed = asyncio.Event()

    async def events():
        try:
            yield {
                "type": "ready",
                "prompt_tokens": 2,
                "completion_tokens": 0,
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [],
                "reasoning_tokens": 0,
                "thinking": False,
                "request_id": generation_id,
            }
            for index in range(4):
                if index == 3:
                    fourth_delta_requested.set()
                yield {
                    "type": "delta",
                    "text": str(index),
                    "prompt_tokens": 2,
                    "completion_tokens": index + 1,
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": list(range(index + 1)),
                    "reasoning_tokens": 0,
                    "thinking": False,
                    "request_id": generation_id,
                }
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def run() -> bytes:
        stream = openai_chat_stream(
            AdapterRouter([record]),
            record=record,
            events=events(),
            adapter_id=record.adapter_id,
            completion_id="chatcmpl-stream",
            created=123,
            include_usage=True,
            usage_session=session,  # type: ignore[arg-type]
        )
        role = await anext(stream)
        await asyncio.wait_for(fourth_delta_requested.wait(), timeout=1)
        await asyncio.wait_for(stream.aclose(), timeout=1)
        return role

    role = asyncio.run(run())

    assert b'"role":"assistant"' in role
    assert closed.is_set()
    assert session.finalized == []
    assert len(session.failed) == 1
    assert session.failed[0][1] == "client_disconnected"


def test_stream_disconnect_persists_latest_cumulative_snapshot() -> None:
    session = _StreamSession()
    generation_id = new_generation_id()
    record = _revision()
    released = asyncio.Event()

    async def events():
        try:
            yield {
                "type": "ready",
                "prompt_tokens": 2,
                "completion_tokens": 0,
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [],
                "reasoning_tokens": 0,
                "thinking": False,
                "request_id": generation_id,
            }
            yield {
                "type": "delta",
                "text": "partial",
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3],
                "reasoning_tokens": 0,
                "thinking": False,
                "request_id": generation_id,
            }
            await asyncio.Event().wait()
        finally:
            released.set()

    async def run() -> bytes:
        stream = openai_chat_stream(
            AdapterRouter([record]),
            record=record,
            events=events(),
            adapter_id=record.adapter_id,
            completion_id="chatcmpl-stream",
            created=123,
            include_usage=True,
            usage_session=session,  # type: ignore[arg-type]
        )
        await anext(stream)
        partial = await anext(stream)
        await stream.aclose()
        return partial

    partial = asyncio.run(run())

    assert b'"content":"partial"' in partial
    assert released.is_set()
    assert session.finalized == []
    assert session.failed[-1][0]["type"] == "delta"
    assert session.failed[-1][0]["completion_token_ids"] == [3]
    assert session.failed[-1][1] == "client_disconnected"


def test_stream_engine_error_persists_latest_valid_snapshot() -> None:
    session = _StreamSession()
    generation_id = new_generation_id()
    record = _revision()

    async def events():
        yield {
            "type": "delta",
            "text": "partial",
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3],
            "reasoning_tokens": 0,
            "thinking": False,
            "request_id": generation_id,
        }
        raise ValueError("engine failed")

    async def run() -> list[bytes]:
        return [
            chunk
            async for chunk in openai_chat_stream(
                AdapterRouter([record]),
                record=record,
                events=events(),
                adapter_id=record.adapter_id,
                completion_id="chatcmpl-stream",
                created=123,
                include_usage=True,
                usage_session=session,  # type: ignore[arg-type]
            )
        ]

    asyncio.run(run())

    assert session.finalized == []
    assert session.failed[-1][0]["type"] == "delta"
    assert session.failed[-1][0]["completion_tokens"] == 1
    assert session.failed[-1][1] == "engine_failed"
