"""Durable hosted-serving usage capture, delivery, and reconciliation boundaries."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from flash.serving.src.settings import Settings
from flash.serving.src.supabase_rest import supabase_headers

StableId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
DecimalString = Annotated[str, StringConstraints(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")]
_GENERATION_ID_RE = re.compile(r"^fsgen-[0-9a-f]{32}$")
_HEARTBEAT_BATCH_SIZE = 128
_RECOVERY_BATCH_SIZE = 500
_CLEANUP_TIMEOUT_SECONDS = 10.0
_RESPONSE_LOSS_REPLAY_ATTEMPTS = 2
_RESPONSE_LOSS_REPLAY_DELAY_SECONDS = 0.05
_RESPONSE_LOSS_SAFE_RPCS = frozenset(
    {
        "capture_serving_usage",
        "finalize_serving_usage",
        "fail_serving_generation",
        "heartbeat_serving_generation",
        "recover_stale_serving_generations",
        "acknowledge_serving_usage_delivered",
        "reconcile_serving_usage_batch",
        "finalize_serving_usage_reconciliation_day",
    }
)


class AcceptedPriceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    promptTokenUsd: DecimalString
    cachedPromptTokenUsd: DecimalString | None = None
    completionTokenUsd: DecimalString
    requestUsd: DecimalString | None = None


class OpenRouterTrafficPrincipal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["openrouter"] = "openrouter"
    publicModelId: StableId
    providerCatalogDigest: StableId
    acceptedPriceSnapshot: AcceptedPriceSnapshot


class FreesoloOrgTrafficPrincipal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["freesolo_org"] = "freesolo_org"
    orgId: StableId


class TrustedInternalTrafficPrincipal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["trusted_internal"] = "trusted_internal"
    orgId: StableId | None = None
    billingAttributionExplicit: bool = False

    @model_validator(mode="after")
    def validate_attribution(self) -> TrustedInternalTrafficPrincipal:
        if self.orgId is not None and not self.billingAttributionExplicit:
            raise ValueError("trusted_internal org billing requires explicit attribution")
        return self


ServingTrafficPrincipal = Annotated[
    OpenRouterTrafficPrincipal | FreesoloOrgTrafficPrincipal | TrustedInternalTrafficPrincipal,
    Field(discriminator="kind"),
]


@dataclass(frozen=True)
class RequestIdentity:
    request_id: str
    correlation_id: str
    openai_completion_id: str | None = None
    openrouter_request_id: str | None = None
    openrouter_generation_id: str | None = None
    upstream_id: str | None = None

    def __post_init__(self) -> None:
        if _GENERATION_ID_RE.fullmatch(self.request_id) is None:
            raise ValueError("request_id must use the fsgen generation id format")
        public_ids = {
            value
            for value in (
                self.correlation_id,
                self.openai_completion_id,
                self.openrouter_request_id,
                self.openrouter_generation_id,
                self.upstream_id,
            )
            if value is not None
        }
        if self.request_id in public_ids:
            raise ValueError("request_id must be distinct from public and correlation ids")


@dataclass(frozen=True)
class ImmutableTarget:
    public_model_id: str
    base_model: str
    requested_adapter_id: str | None
    resolved_adapter_revision: str | None
    resolved_checkpoint_id: str | None
    resolved_hf_revision: str | None


@dataclass(frozen=True)
class CapturedPrice:
    source: str
    version: str
    snapshot: Mapping[str, Any]
    quoted_provider_amount_micro_usd: int | None = None


@dataclass(frozen=True)
class UsageFacts:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cached_tokens_reported: bool
    reasoning_tokens: int
    generation_duration_seconds: float | None
    engine_replica_id: str | None


@dataclass(frozen=True)
class UsageEvent:
    identity: RequestIdentity
    principal: ServingTrafficPrincipal
    target: ImmutableTarget
    price: CapturedPrice
    captured_at: datetime
    deployment_id: str | None
    serving_release: str | None
    tokenizer_identity: str | None
    tokenizer_version: str | None
    attestation_evidence: Mapping[str, Any]
    facts: UsageFacts

    def rpc_payload(self) -> dict[str, Any]:
        principal = self.principal
        openrouter = principal if principal.kind == "openrouter" else None
        org_id = None if principal.kind == "openrouter" else principal.orgId
        explicit = (
            principal.billingAttributionExplicit if principal.kind == "trusted_internal" else False
        )
        requested_adapter_id = (
            None if principal.kind == "openrouter" else self.target.requested_adapter_id
        )
        accepted = openrouter.acceptedPriceSnapshot.model_dump(mode="json") if openrouter else None
        return {
            "request_id": self.identity.request_id,
            "correlation_id": self.identity.correlation_id,
            "traffic_principal_kind": principal.kind,
            "billing_attribution_explicit": explicit,
            "org_id": org_id,
            "traffic_source": "openrouter" if openrouter else "freesolo",
            "openrouter_request_id": self.identity.openrouter_request_id,
            "openrouter_generation_id": self.identity.openrouter_generation_id,
            "openai_completion_id": self.identity.openai_completion_id,
            "upstream_id": self.identity.upstream_id,
            "public_model_id": (
                openrouter.publicModelId if openrouter else self.target.public_model_id
            ),
            "base_model": self.target.base_model,
            "requested_adapter_id": requested_adapter_id,
            "resolved_adapter_revision": self.target.resolved_adapter_revision,
            "resolved_checkpoint_id": self.target.resolved_checkpoint_id,
            "resolved_hf_revision": self.target.resolved_hf_revision,
            "prompt_tokens": self.facts.prompt_tokens,
            "completion_tokens": self.facts.completion_tokens,
            "reasoning_tokens": self.facts.reasoning_tokens,
            "cached_tokens": self.facts.cached_tokens,
            "cached_tokens_reported": self.facts.cached_tokens_reported,
            "tokenizer_identity": self.tokenizer_identity,
            "tokenizer_version": self.tokenizer_version,
            "generation_duration_seconds": self.facts.generation_duration_seconds,
            "engine_replica_id": self.facts.engine_replica_id,
            "serving_deployment_id": self.deployment_id,
            "serving_release": self.serving_release,
            "captured_at": self.captured_at.astimezone(UTC).isoformat(),
            "attestation_evidence": dict(self.attestation_evidence),
            "pricing_source": self.price.source,
            "price_version": self.price.version,
            "price_snapshot": dict(self.price.snapshot),
            "quoted_provider_amount_micro_usd": self.price.quoted_provider_amount_micro_usd,
            "provider_catalog_digest": (openrouter.providerCatalogDigest if openrouter else None),
            "accepted_price_snapshot": accepted,
        }


@dataclass(frozen=True)
class OutboxSnapshot:
    captured_at: datetime
    pending: int
    leased: int
    quarantined: int
    disputed: int
    due_pending: int
    expired_leases: int
    expired_generation_leases: int
    oldest_expired_generation_lease_age_seconds: int | None
    oldest_undelivered_age_seconds: int | None


class UsageOutboxError(RuntimeError):
    """Sanitized durable accounting failure safe to expose by class only."""


class ReasoningSettlementUnavailable(UsageOutboxError):
    """Exact reasoning-token facts are unavailable for this generation."""


class ReconciliationUnavailable(UsageOutboxError):
    """Authoritative provider settlement is not launch-ready."""


class UsageStore(Protocol):
    enabled: bool

    async def start(self) -> None: ...
    async def capture(self, event: UsageEvent) -> None: ...
    async def finalize(self, event: UsageEvent) -> None: ...
    async def fail(self, event: UsageEvent, code: str) -> None: ...
    async def snapshot(self) -> OutboxSnapshot: ...
    async def recover_stale_in_progress(self) -> None: ...
    async def aclose(self) -> None: ...


class OfflineUsageStore:
    """Explicit offline-only store used when app tests do not inject persistence."""

    enabled = False

    async def start(self) -> None:
        return None

    async def capture(self, event: UsageEvent) -> None:
        del event

    async def finalize(self, event: UsageEvent) -> None:
        del event

    async def fail(self, event: UsageEvent, code: str) -> None:
        del event, code

    async def snapshot(self) -> OutboxSnapshot:
        raise UsageOutboxError("usage_outbox_disabled")

    async def recover_stale_in_progress(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True)
class AuthoritativeProviderDay:
    provider: str
    usage_date: date
    source: str
    source_version: str
    attestation_evidence: Mapping[str, Any]

    def rpc_payload(self) -> dict[str, Any]:
        return {
            "p_provider": self.provider,
            "p_usage_date": self.usage_date.isoformat(),
            "p_source": self.source,
            "p_source_version": self.source_version,
            "p_attestation_evidence": dict(self.attestation_evidence),
        }


@dataclass(frozen=True)
class ReconciliationDayResult:
    reconciliation_day_id: str
    reconciliation_state: str
    status_reason: str | None
    replay: bool


@dataclass(frozen=True)
class ReconciliationResult:
    input_ordinal: int
    outbox_id: str | None
    reconciliation_status: str
    dispute_code: str | None
    replay: bool


@dataclass(frozen=True)
class ProviderSettlementRecord:
    provider: str
    usage_date: date
    source: str
    source_version: str
    provider_record_id: str
    request_id: str | None
    provider_amount_micro_usd: int | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    evidence: Mapping[str, Any] | None = None
    attestation_evidence: Mapping[str, Any] | None = None

    def rpc_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "usage_date": self.usage_date.isoformat(),
            "source": self.source,
            "source_version": self.source_version,
            "provider_record_id": self.provider_record_id,
            "request_id": self.request_id,
            "provider_amount_micro_usd": self.provider_amount_micro_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "evidence": dict(self.evidence or {}),
            "attestation_evidence": dict(self.attestation_evidence or {}),
        }


@dataclass(frozen=True)
class ProviderSettlementPage:
    records: tuple[ProviderSettlementRecord, ...]
    next_cursor: str | None


class ProviderSettlementAdapter(Protocol):
    provider: str
    authoritative: bool
    exact_reasoning_tokens: bool

    async def fetch_day(
        self, usage_date: date, *, cursor: str | None, limit: int
    ) -> ProviderSettlementPage: ...


Jitter = Callable[[str, int, float], float]
Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], datetime]


def _deterministic_jitter(outbox_id: str, attempt: int, base: float) -> float:
    digest = hashlib.sha256(f"{outbox_id}:{attempt}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return fraction * min(base * 0.2, 30.0)


class DurableUsageOutbox:
    """Supabase-backed capture plus one leased backend-settlement worker."""

    enabled = True

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        worker_id: str,
        batch_size: int = 50,
        lease_seconds: int = 60,
        max_attempts: int = 8,
        poll_seconds: float = 2.0,
        jitter: Jitter = _deterministic_jitter,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = lambda: datetime.now(UTC),
        owner_epoch: uuid.UUID | None = None,
    ) -> None:
        if (
            not settings.has_supabase
            or not settings.backend_url
            or not settings.internal_key
            or not settings.deployment_id
        ):
            raise UsageOutboxError("durable_usage_outbox_not_configured")
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._owns_client = client is None
        self._worker_id = worker_id
        self._generation_owner_id = (
            "fsrouter-" + hashlib.sha256(settings.deployment_id.encode()).hexdigest()[:32]
        )
        self._generation_owner_epoch = owner_epoch or uuid.uuid4()
        self._batch_size = batch_size
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._poll_seconds = poll_seconds
        self._jitter = jitter
        self._sleep = sleep
        self._clock = clock
        self._wake = asyncio.Event()
        self._heartbeat_wake = asyncio.Event()
        self._stopping = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._heartbeat_worker: asyncio.Task[None] | None = None
        self._heartbeat_seconds: int | None = None
        self._active_generations: set[str] = set()
        self._active_leases: set[str] = set()
        self._background_error: BaseException | None = None

    async def start(self) -> None:
        if self._worker is not None:
            return
        await self.recover_stale_in_progress()
        self._worker = asyncio.create_task(self._run_worker())
        self._heartbeat_worker = asyncio.create_task(self._run_heartbeat_worker())
        self._wake.set()

    async def capture(self, event: UsageEvent) -> None:
        self._raise_background_error()
        result = await self._generation_rpc("capture_serving_usage", event)
        timing = _generation_capture_result(result)
        self._set_heartbeat_seconds(timing["heartbeat_seconds"])
        if timing["state"] == "in_progress":
            self._active_generations.add(event.identity.request_id)
            self._heartbeat_wake.set()

    async def finalize(self, event: UsageEvent) -> None:
        self._raise_background_error()
        await self._generation_rpc("finalize_serving_usage", event)
        self._active_generations.discard(event.identity.request_id)
        self._wake.set()
        self._heartbeat_wake.set()

    async def fail(self, event: UsageEvent, code: str) -> None:
        self._raise_background_error()
        await self._generation_rpc("fail_serving_generation", event, failure_code=code)
        self._active_generations.discard(event.identity.request_id)
        self._heartbeat_wake.set()

    async def snapshot(self) -> OutboxSnapshot:
        data = await self._rpc("serving_usage_backlog_snapshot", {})
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict):
            raise UsageOutboxError("usage_snapshot_invalid")
        states = data.get("states")
        if not isinstance(states, dict):
            states = {}
        captured_at = _parse_datetime(data.get("captured_at"))
        oldest = data.get("oldest_undelivered_age_seconds")
        expired_age = data.get("oldest_expired_generation_lease_age_seconds")
        return OutboxSnapshot(
            captured_at=captured_at,
            pending=int(states.get("pending") or 0),
            leased=int(states.get("leased") or 0),
            quarantined=int(states.get("quarantined") or 0),
            disputed=int(states.get("disputed") or 0),
            due_pending=int(data.get("due_pending") or 0),
            expired_leases=int(data.get("expired_leases") or 0),
            expired_generation_leases=int(data.get("expired_generation_leases") or 0),
            oldest_expired_generation_lease_age_seconds=(
                None if expired_age is None else int(expired_age)
            ),
            oldest_undelivered_age_seconds=None if oldest is None else int(oldest),
        )

    async def recover_stale_in_progress(self) -> None:
        await self._rpc(
            "recover_stale_serving_generations",
            {"p_limit": _RECOVERY_BATCH_SIZE},
        )

    async def reconcile_batch(
        self, records: Sequence[ProviderSettlementRecord]
    ) -> tuple[ReconciliationResult, ...]:
        if not 1 <= len(records) <= 500:
            raise ValueError("serving usage reconciliation batch must contain 1 to 500 records")
        data = await self._rpc(
            "reconcile_serving_usage_batch",
            {"p_records": [record.rpc_payload() for record in records]},
        )
        if not isinstance(data, list) or len(data) != len(records):
            raise UsageOutboxError("usage_reconciliation_batch_invalid")
        results: list[ReconciliationResult] = []
        for expected_ordinal, row in enumerate(data, 1):
            if not isinstance(row, dict) or row.get("input_ordinal") != expected_ordinal:
                raise UsageOutboxError("usage_reconciliation_batch_invalid")
            status = row.get("reconciliation_status")
            replay = row.get("replay")
            if not isinstance(status, str) or type(replay) is not bool:
                raise UsageOutboxError("usage_reconciliation_batch_invalid")
            outbox_id = row.get("outbox_id")
            dispute_code = row.get("dispute_code")
            if outbox_id is not None and not isinstance(outbox_id, str):
                raise UsageOutboxError("usage_reconciliation_batch_invalid")
            if dispute_code is not None and not isinstance(dispute_code, str):
                raise UsageOutboxError("usage_reconciliation_batch_invalid")
            results.append(
                ReconciliationResult(
                    input_ordinal=expected_ordinal,
                    outbox_id=outbox_id,
                    reconciliation_status=status,
                    dispute_code=dispute_code,
                    replay=replay,
                )
            )
        return tuple(results)

    async def reconcile(self, record: ProviderSettlementRecord) -> ReconciliationResult:
        return (await self.reconcile_batch([record]))[0]

    async def finalize_reconciliation_day(
        self, day: AuthoritativeProviderDay
    ) -> ReconciliationDayResult:
        data = await self._rpc(
            "finalize_serving_usage_reconciliation_day",
            day.rpc_payload(),
        )
        if isinstance(data, list):
            data = data[0] if len(data) == 1 else None
        if not isinstance(data, dict):
            raise UsageOutboxError("usage_reconciliation_day_invalid")
        day_id = data.get("reconciliation_day_id")
        state = data.get("reconciliation_state")
        reason = data.get("status_reason")
        replay = data.get("replay")
        if (
            not isinstance(day_id, str)
            or not isinstance(state, str)
            or (reason is not None and not isinstance(reason, str))
            or type(replay) is not bool
        ):
            raise UsageOutboxError("usage_reconciliation_day_invalid")
        return ReconciliationDayResult(day_id, state, reason, replay)

    async def _generation_rpc(
        self,
        name: str,
        event: UsageEvent,
        *,
        failure_code: str | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "p_event": event.rpc_payload(),
            "p_generation_owner_id": self._generation_owner_id,
            "p_generation_owner_epoch": str(self._generation_owner_epoch),
        }
        if failure_code is not None:
            payload["p_failure_code"] = failure_code
        return await self._rpc(name, payload)

    def _set_heartbeat_seconds(self, heartbeat_seconds: int) -> None:
        if self._heartbeat_seconds is None:
            self._heartbeat_seconds = heartbeat_seconds
            return
        if self._heartbeat_seconds != heartbeat_seconds:
            raise UsageOutboxError("generation_heartbeat_timing_changed")

    def _raise_background_error(self) -> None:
        if self._background_error is not None:
            raise UsageOutboxError("usage_outbox_background_failure") from self._background_error

    async def _rpc(self, name: str, payload: dict[str, Any]) -> Any:
        url = f"{str(self._settings.supabase_url).rstrip('/')}/rest/v1/rpc/{name}"
        headers = supabase_headers(self._settings, "public")
        headers["Authorization"] = f"Bearer {self._settings.supabase_service_role_key}"
        attempts = _RESPONSE_LOSS_REPLAY_ATTEMPTS if name in _RESPONSE_LOSS_SAFE_RPCS else 1
        response: httpx.Response | None = None
        for attempt in range(attempts):
            try:
                response = await self._client.post(url, headers=headers, json=payload)
                break
            except httpx.TransportError as exc:
                if attempt + 1 == attempts:
                    raise UsageOutboxError("supabase_transport_failure") from exc
                await self._sleep(_RESPONSE_LOSS_REPLAY_DELAY_SECONDS)
        if response is None:
            raise UsageOutboxError("supabase_transport_failure")
        if response.is_error:
            raise UsageOutboxError(f"supabase_rpc_{response.status_code}")
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise UsageOutboxError("supabase_rpc_invalid_json") from exc

    async def _run_worker(self) -> None:
        while not self._stopping.is_set():
            self._wake.clear()
            try:
                await self.recover_stale_in_progress()
                claimed = await self._claim()
                for row in claimed:
                    if self._stopping.is_set():
                        return
                    await self._deliver(row)
                if claimed:
                    self._wake.set()
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._sleep(self._poll_seconds)
            if self._wake.is_set():
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_seconds)

    async def _run_heartbeat_worker(self) -> None:
        while not self._stopping.is_set():
            self._heartbeat_wake.clear()
            heartbeat_seconds = self._heartbeat_seconds
            if not self._active_generations or heartbeat_seconds is None:
                await self._heartbeat_wake.wait()
                continue
            try:
                await asyncio.wait_for(self._heartbeat_wake.wait(), timeout=heartbeat_seconds)
                continue
            except TimeoutError:
                pass
            try:
                await self._heartbeat_active_generations()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._background_error = exc
                self._stopping.set()
                self._wake.set()
                return

    async def _heartbeat_active_generations(self) -> None:
        request_ids = sorted(self._active_generations)
        for offset in range(0, len(request_ids), _HEARTBEAT_BATCH_SIZE):
            batch = request_ids[offset : offset + _HEARTBEAT_BATCH_SIZE]
            result = await self._rpc(
                "heartbeat_serving_generation",
                {
                    "p_generation_owner_id": self._generation_owner_id,
                    "p_generation_owner_epoch": str(self._generation_owner_epoch),
                    "p_request_ids": batch,
                },
            )
            renewed = {
                str(row.get("request_id"))
                for row in result or []
                if isinstance(row, dict) and row.get("request_id")
            }
            missing = (set(batch) - renewed) & self._active_generations
            if missing:
                self._active_generations.difference_update(missing)
                raise UsageOutboxError("generation_heartbeat_lease_lost")

    async def _claim(self) -> list[dict[str, Any]]:
        data = await self._rpc(
            "claim_serving_usage_batch",
            {
                "p_worker_id": self._worker_id,
                "p_limit": self._batch_size,
                "p_lease_seconds": self._lease_seconds,
            },
        )
        if not isinstance(data, list):
            raise UsageOutboxError("usage_claim_invalid")
        return [dict(row) for row in data if isinstance(row, dict)]

    async def _deliver(self, row: dict[str, Any]) -> None:
        outbox_id = str(row.get("id") or "")
        request_id = str(row.get("request_id") or "")
        attempt = int(row.get("attempt_count") or 0)
        if not outbox_id or not request_id:
            return
        principal = _settlement_principal(row)
        self._active_leases.add(outbox_id)
        body = {
            "outboxId": outbox_id,
            "workerId": self._worker_id,
            "requestId": request_id,
            "trafficPrincipal": principal,
        }
        url = f"{self._settings.backend_url.rstrip('/')}/api/billing/serving-usage/durable"
        try:
            response = await self._client.post(
                url,
                headers={"Authorization": f"Bearer {self._settings.internal_key}"},
                json=body,
            )
        except httpx.TransportError:
            await self._retry_or_quarantine(outbox_id, attempt, "transport")
            return
        status = response.status_code
        if status in {408, 429} or status >= 500:
            await self._retry_or_quarantine(outbox_id, attempt, f"http_{status}")
            return
        if 400 <= status < 500:
            await self._quarantine(outbox_id, f"permanent_http_{status}")
            return
        try:
            result = response.json()
            ack = {
                "usage_id": result.get("usageId"),
                "ledger_id": result.get("ledgerId"),
                "price_version": result["priceVersion"],
                "exact_cost_micro_usd": int(result["exactCostMicroUsd"]),
                "billed_cents": int(result["billedCents"]),
                "replay": bool(result.get("replay", False)),
            }
        except (KeyError, TypeError, ValueError):
            await self._quarantine(outbox_id, "invalid_settlement_response")
            return
        await self._rpc(
            "acknowledge_serving_usage_delivered",
            {
                "p_outbox_id": outbox_id,
                "p_worker_id": self._worker_id,
                "p_result": ack,
            },
        )
        self._active_leases.discard(outbox_id)

    async def _retry_or_quarantine(self, outbox_id: str, attempt: int, error_code: str) -> None:
        if attempt >= self._max_attempts:
            await self._quarantine(outbox_id, "delivery_attempts_exhausted")
            return
        base = min(2.0 ** max(0, attempt - 1), 300.0)
        delay = base + self._jitter(outbox_id, attempt, base)
        retry_at = self._clock() + timedelta(seconds=delay)
        await self._rpc(
            "reschedule_serving_usage",
            {
                "p_outbox_id": outbox_id,
                "p_worker_id": self._worker_id,
                "p_retry_at": retry_at.isoformat(),
                "p_error_code": error_code[:200],
            },
        )
        self._active_leases.discard(outbox_id)

    async def _quarantine(self, outbox_id: str, reason: str) -> None:
        await self._rpc(
            "quarantine_serving_usage",
            {
                "p_outbox_id": outbox_id,
                "p_worker_id": self._worker_id,
                "p_reason": reason[:500],
            },
        )
        self._active_leases.discard(outbox_id)

    async def aclose(self) -> None:
        self._stopping.set()
        self._wake.set()
        self._heartbeat_wake.set()
        errors: list[BaseException] = []
        workers = [task for task in (self._worker, self._heartbeat_worker) if task is not None]
        for worker in workers:
            worker.cancel()
        if workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*workers, return_exceptions=True),
                    timeout=_CLEANUP_TIMEOUT_SECONDS,
                )
            except BaseException as exc:
                errors.append(exc)
        self._worker = None
        self._heartbeat_worker = None

        if self._active_generations:
            try:
                await asyncio.wait_for(
                    self._rpc(
                        "fail_serving_generation_session",
                        {
                            "p_generation_owner_id": self._generation_owner_id,
                            "p_generation_owner_epoch": str(self._generation_owner_epoch),
                        },
                    ),
                    timeout=_CLEANUP_TIMEOUT_SECONDS,
                )
            except BaseException as exc:
                errors.append(exc)
            else:
                self._active_generations.clear()

        for outbox_id in tuple(self._active_leases):
            try:
                await asyncio.wait_for(
                    self._rpc(
                        "reschedule_serving_usage",
                        {
                            "p_outbox_id": outbox_id,
                            "p_worker_id": self._worker_id,
                            "p_retry_at": self._clock().isoformat(),
                            "p_error_code": "worker_shutdown",
                        },
                    ),
                    timeout=_CLEANUP_TIMEOUT_SECONDS,
                )
            except BaseException as exc:
                errors.append(exc)
            else:
                self._active_leases.discard(outbox_id)

        if self._owns_client:
            try:
                await asyncio.wait_for(self._client.aclose(), timeout=_CLEANUP_TIMEOUT_SECONDS)
            except BaseException as exc:
                errors.append(exc)
        if self._background_error is not None:
            errors.append(self._background_error)
        if errors:
            raise UsageOutboxError("usage_outbox_shutdown_failed") from errors[0]


def _generation_capture_result(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        raise UsageOutboxError("generation_capture_invalid")
    state = data.get("state")
    heartbeat_seconds = data.get("heartbeat_seconds")
    lease_seconds = data.get("lease_seconds")
    if state not in {"in_progress", "pending", "quarantined"}:
        raise UsageOutboxError("generation_capture_invalid")
    if (
        not isinstance(heartbeat_seconds, int)
        or isinstance(heartbeat_seconds, bool)
        or heartbeat_seconds <= 0
        or not isinstance(lease_seconds, int)
        or isinstance(lease_seconds, bool)
        or lease_seconds <= heartbeat_seconds
    ):
        raise UsageOutboxError("generation_lease_timing_invalid")
    return {
        "state": state,
        "heartbeat_seconds": heartbeat_seconds,
        "lease_seconds": lease_seconds,
    }


def _settlement_principal(row: Mapping[str, Any]) -> dict[str, Any]:
    kind = row.get("traffic_principal_kind")
    if kind == "openrouter":
        return {"kind": "openrouter", "publicModelId": row.get("public_model_id")}
    if kind == "freesolo_org":
        return {"kind": "freesolo_org", "orgId": row.get("org_id")}
    if kind == "trusted_internal":
        return {
            "kind": "trusted_internal",
            "orgId": row.get("org_id"),
            "billingAttributionExplicit": bool(row.get("billing_attribution_explicit")),
        }
    raise UsageOutboxError("usage_principal_invalid")


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise UsageOutboxError("usage_snapshot_timestamp_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UsageOutboxError("usage_snapshot_timestamp_invalid") from exc
    if parsed.tzinfo is None:
        raise UsageOutboxError("usage_snapshot_timestamp_invalid")
    return parsed


def exact_reasoning_tokens(result: Mapping[str, Any]) -> int:
    """Return an exact count or refuse to guess from rendered text or delimiters."""
    exact = result.get("reasoning_tokens")
    if isinstance(exact, int) and not isinstance(exact, bool) and exact >= 0:
        return exact
    if result.get("thinking") is False:
        return 0
    raise ReasoningSettlementUnavailable("exact_reasoning_tokens_unavailable")


def usage_facts(result: Mapping[str, Any]) -> UsageFacts:
    prompt_ids = result.get("prompt_token_ids")
    completion_ids = result.get("completion_token_ids", result.get("token_ids"))
    prompt_tokens = result.get("prompt_tokens")
    completion_tokens = result.get("completion_tokens")
    if isinstance(prompt_ids, list):
        prompt_tokens = len(prompt_ids)
    if isinstance(completion_ids, list):
        completion_tokens = len(completion_ids)
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        raise UsageOutboxError("native_token_ids_unavailable")
    cached = result.get("cached_tokens")
    cached_tokens = int(cached) if isinstance(cached, int) and not isinstance(cached, bool) else 0
    cached_reported = result.get("cached_tokens_reported")
    cached_tokens_reported = type(cached_reported) is bool and cached_reported
    if min(prompt_tokens, completion_tokens, cached_tokens) < 0 or cached_tokens > prompt_tokens:
        raise UsageOutboxError("usage_counters_invalid")
    duration = result.get("inference_time_seconds")
    if duration is not None:
        duration = float(duration)
        if duration < 0:
            raise UsageOutboxError("usage_duration_invalid")
    replica = result.get("engine_replica_id")
    return UsageFacts(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cached_tokens_reported=cached_tokens_reported,
        reasoning_tokens=exact_reasoning_tokens(result),
        generation_duration_seconds=duration,
        engine_replica_id=replica if isinstance(replica, str) and replica else None,
    )


def accepted_price_micro_usd(snapshot: AcceptedPriceSnapshot, facts: UsageFacts) -> int:
    uncached = facts.prompt_tokens - facts.cached_tokens
    cost = Decimal(snapshot.promptTokenUsd) * uncached
    cost += Decimal(snapshot.cachedPromptTokenUsd or snapshot.promptTokenUsd) * facts.cached_tokens
    cost += Decimal(snapshot.completionTokenUsd) * facts.completion_tokens
    if snapshot.requestUsd is not None:
        cost += Decimal(snapshot.requestUsd)
    return int((cost * 1_000_000).quantize(Decimal("1")))
