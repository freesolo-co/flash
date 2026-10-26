"""Durable hosted-serving usage capture and delivery boundaries."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import time
import uuid
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from flash.serving.src.accounting.usage_retry import is_transient_rpc_code
from flash.serving.src.store.settings import Settings
from flash.serving.src.store.supabase_rest import supabase_headers

StableId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
_GENERATION_ID_RE = re.compile(r"^fsgen-[0-9a-f]{32}$")
_HEARTBEAT_BATCH_SIZE = 128
_RECOVERY_BATCH_SIZE = 500
_DELIVERY_CONCURRENCY = 8
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
    }
)


class FreesoloOrgTrafficPrincipal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["freesolo_org"] = "freesolo_org"
    orgId: StableId


class TrustedInternalTrafficPrincipal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["trusted_internal"] = "trusted_internal"
    orgId: StableId
    billingAttributionExplicit: Literal[True] = True


ServingTrafficPrincipal = Annotated[
    FreesoloOrgTrafficPrincipal | TrustedInternalTrafficPrincipal,
    Field(discriminator="kind"),
]


@dataclass(frozen=True)
class RequestIdentity:
    request_id: str
    correlation_id: str
    openai_completion_id: str | None = None

    def __post_init__(self) -> None:
        if _GENERATION_ID_RE.fullmatch(self.request_id) is None:
            raise ValueError("request_id must use the fsgen generation id format")
        public_ids = {
            value
            for value in (
                self.correlation_id,
                self.openai_completion_id,
            )
            if value is not None
        }
        if self.request_id in public_ids:
            raise ValueError("request_id must be distinct from public and correlation ids")


@dataclass(frozen=True)
class ImmutableTarget:
    public_model_id: str
    base_model: str
    checkpoint_id: str | None
    artifact_fingerprint: str | None


@dataclass(frozen=True)
class CapturedPrice:
    source: str
    version: str
    snapshot: Mapping[str, Any]


@dataclass(frozen=True)
class UsageFacts:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cached_tokens_reported: bool
    reasoning_tokens: int
    generation_duration_seconds: float | None
    time_to_first_token_seconds: float | None
    queue_wait_seconds: float | None
    replica_in_flight_requests_at_admission: int | None
    replica_boot_duration_seconds: float | None
    replica_freshly_booted: bool | None
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
        # The durable RPC allowlists billing fields. Capacity observations belong in
        # its persisted evidence envelope, so optional telemetry cannot reject usage.
        capacity = {
            key: value
            for key, value in (
                ("time_to_first_token_seconds", self.facts.time_to_first_token_seconds),
                ("queue_wait_seconds", self.facts.queue_wait_seconds),
                (
                    "replica_in_flight_requests_at_admission",
                    self.facts.replica_in_flight_requests_at_admission,
                ),
                ("replica_boot_duration_seconds", self.facts.replica_boot_duration_seconds),
                ("replica_freshly_booted", self.facts.replica_freshly_booted),
            )
            if value is not None
        }
        evidence = dict(self.attestation_evidence)
        if capacity:
            evidence["capacity"] = capacity
        return {
            "request_id": self.identity.request_id,
            "correlation_id": self.identity.correlation_id,
            "traffic_principal_kind": principal.kind,
            "traffic_source": "freesolo",
            "billing_attribution_explicit": principal.kind == "trusted_internal",
            "org_id": principal.orgId,
            "openai_completion_id": self.identity.openai_completion_id,
            "public_model_id": self.target.public_model_id,
            "base_model": self.target.base_model,
            "checkpoint_id": self.target.checkpoint_id,
            "artifact_fingerprint": self.target.artifact_fingerprint,
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
            "attestation_evidence": evidence,
            "pricing_source": self.price.source,
            "price_version": self.price.version,
            "price_snapshot": dict(self.price.snapshot),
        }


class UsageOutboxError(RuntimeError):
    """Sanitized durable accounting failure safe to expose by class only."""


class ReasoningSettlementUnavailable(UsageOutboxError):
    """Exact reasoning-token facts are unavailable for this generation."""


class UsageStore(Protocol):
    enabled: bool

    def assert_healthy(self) -> None: ...
    async def start(self) -> None: ...
    async def capture(self, event: UsageEvent) -> None: ...
    async def finalize(self, event: UsageEvent) -> None: ...
    async def fail(self, event: UsageEvent, code: str) -> None: ...
    def relinquish(self, request_id: str) -> None: ...
    async def recover_stale_in_progress(self) -> None: ...
    async def aclose(self) -> None: ...


class OfflineUsageStore:
    """Explicit offline-only store used when app tests do not inject persistence."""

    enabled = False

    def assert_healthy(self) -> None:
        return None

    async def start(self) -> None:
        return None

    async def capture(self, event: UsageEvent) -> None:
        del event

    async def finalize(self, event: UsageEvent) -> None:
        del event

    async def fail(self, event: UsageEvent, code: str) -> None:
        del event, code

    def relinquish(self, request_id: str) -> None:
        del request_id

    async def recover_stale_in_progress(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


Jitter = Callable[[str, int, float], float]
Sleep = Callable[[float], Coroutine[Any, Any, None]]
WallClock = Callable[[], datetime]
MonotonicClock = Callable[[], float]


def _deterministic_jitter(outbox_id: str, attempt: int, base: float) -> float:
    digest = hashlib.sha256(f"{outbox_id}:{attempt}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return fraction * min(base * 0.2, 30.0)


class DurableUsageOutbox:
    """Supabase-backed capture plus bounded, leased backend settlement."""

    enabled = True

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        worker_id: str,
        lease_seconds: int = 60,
        max_attempts: int = 8,
        poll_seconds: float = 2.0,
        jitter: Jitter = _deterministic_jitter,
        sleep: Sleep = asyncio.sleep,
        clock: WallClock = lambda: datetime.now(UTC),
        monotonic: MonotonicClock = time.monotonic,
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
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._poll_seconds = poll_seconds
        self._jitter = jitter
        self._sleep = sleep
        self._clock = clock
        self._monotonic = monotonic
        self._wake = asyncio.Event()
        self._heartbeat_wake = asyncio.Event()
        self._stopping = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._heartbeat_worker: asyncio.Task[None] | None = None
        self._heartbeat_seconds: int | None = None
        self._generation_lease_seconds: int | None = None
        self._active_generations: set[str] = set()
        self._terminal_generations: set[str] = set()
        self._generation_lease_deadlines: dict[str, float] = {}
        self._active_leases: set[str] = set()
        self._background_error: BaseException | None = None

    def assert_healthy(self) -> None:
        self._raise_background_error()
        if self._stopping.is_set():
            raise UsageOutboxError("usage_outbox_not_accepting_requests")
        if self._worker is not None and self._worker.done():
            raise UsageOutboxError("usage_outbox_worker_stopped")
        if self._heartbeat_worker is not None and self._heartbeat_worker.done():
            raise UsageOutboxError("usage_outbox_heartbeat_stopped")

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
        lease_started_at = self._monotonic()
        timing = _generation_capture_result(result)
        self._set_generation_timing(timing["heartbeat_seconds"], timing["lease_seconds"])
        if timing["state"] == "in_progress":
            request_id = event.identity.request_id
            self._active_generations.add(request_id)
            self._generation_lease_deadlines[request_id] = (
                lease_started_at + timing["lease_seconds"]
            )
            self._heartbeat_wake.set()

    async def finalize(self, event: UsageEvent) -> None:
        await self._terminal_rpc("finalize_serving_usage", event)
        self._wake.set()

    async def fail(self, event: UsageEvent, code: str) -> None:
        await self._terminal_rpc("fail_serving_generation", event, failure_code=code)

    async def _terminal_rpc(
        self, name: str, event: UsageEvent, *, failure_code: str | None = None
    ) -> None:
        # a background delivery failure refuses NEW work at admission; it must not refuse to
        # settle work already performed. the terminal rpcs are idempotent, so attempting one after
        # the worker died is safe, while skipping it drops the charge for a served request.
        request_id = event.identity.request_id
        self._terminal_generations.add(request_id)
        try:
            await self._generation_rpc(name, event, failure_code=failure_code)
        except asyncio.CancelledError:
            self.relinquish(request_id)
            raise
        except Exception:
            self._terminal_generations.discard(request_id)
            raise
        self.relinquish(request_id)

    def relinquish(self, request_id: str) -> None:
        self._active_generations.discard(request_id)
        self._terminal_generations.discard(request_id)
        self._generation_lease_deadlines.pop(request_id, None)
        self._heartbeat_wake.set()

    async def recover_stale_in_progress(self) -> None:
        await self._rpc(
            "recover_stale_serving_generations",
            {"p_limit": _RECOVERY_BATCH_SIZE},
        )

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

    def _set_generation_timing(self, heartbeat_seconds: int, lease_seconds: int) -> None:
        if self._heartbeat_seconds is None:
            self._heartbeat_seconds = heartbeat_seconds
            self._generation_lease_seconds = lease_seconds
            return
        if (
            self._heartbeat_seconds != heartbeat_seconds
            or self._generation_lease_seconds != lease_seconds
        ):
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
                # Every claimed row starts immediately: no lease expires while waiting
                # behind a slow settlement. Await all outcomes before reclaiming work or
                # surfacing a failure, including cancellation during shutdown.
                results = await asyncio.gather(
                    *(self._deliver(row) for row in claimed), return_exceptions=True
                )
                failures = [result for result in results if isinstance(result, BaseException)]
                for failure in failures:
                    if not _is_transient_rpc_error(failure):
                        raise failure
                if failures:
                    raise failures[0]
                if claimed:
                    self._wake.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # a non-transient delivery failure is a settlement defect, not a blip. keep it
                # instead of looping forever so the health gate can refuse new chargeable traffic.
                if not _is_transient_rpc_error(exc):
                    self._background_error = exc
                    self._stopping.set()
                    self._wake.set()
                    self._heartbeat_wake.set()
                    return
                await self._sleep(self._poll_seconds)
            if self._wake.is_set():
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_seconds)

    async def _run_heartbeat_worker(self) -> None:
        next_heartbeat_at: float | None = None
        while not self._stopping.is_set():
            self._heartbeat_wake.clear()
            heartbeat_seconds = self._heartbeat_seconds
            if not self._active_generations or heartbeat_seconds is None:
                next_heartbeat_at = None
                await self._heartbeat_wake.wait()
                continue
            if next_heartbeat_at is None:
                next_heartbeat_at = self._monotonic() + heartbeat_seconds
            remaining = next_heartbeat_at - self._monotonic()
            if remaining > 0:
                async with asyncio.TaskGroup() as group:
                    wake = group.create_task(self._heartbeat_wake.wait())
                    delay = group.create_task(self._sleep(remaining))
                    done, pending = await asyncio.wait(
                        {wake, delay}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                if wake in done and delay not in done:
                    continue
            try:
                await self._heartbeat_active_generations_with_retry()
                next_heartbeat_at = self._monotonic() + heartbeat_seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._background_error = exc
                self._stopping.set()
                self._wake.set()
                return

    async def _heartbeat_active_generations_with_retry(self) -> None:
        while self._active_generations:
            try:
                await self._heartbeat_active_generations()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not _is_transient_rpc_error(exc):
                    raise
                active_deadlines = [
                    deadline
                    for request_id, deadline in self._generation_lease_deadlines.items()
                    if request_id in self._active_generations
                ]
                if not active_deadlines:
                    return
                remaining = min(active_deadlines) - self._monotonic()
                if remaining <= 0:
                    raise UsageOutboxError("generation_heartbeat_lease_expired") from exc
                retry_delay = min(float(self._heartbeat_seconds or 1), remaining)
                await self._sleep(retry_delay)

    async def _heartbeat_active_generations(self) -> None:
        request_ids = sorted(self._active_generations)
        for offset in range(0, len(request_ids), _HEARTBEAT_BATCH_SIZE):
            batch = request_ids[offset : offset + _HEARTBEAT_BATCH_SIZE]
            renewal_started_at = self._monotonic()
            result = await self._rpc(
                "heartbeat_serving_generation",
                {
                    "p_generation_owner_id": self._generation_owner_id,
                    "p_generation_owner_epoch": str(self._generation_owner_epoch),
                    "p_request_ids": batch,
                },
            )
            renewed: set[str] = set()
            for row in result or []:
                if not isinstance(row, dict) or not row.get("request_id"):
                    continue
                request_id = str(row["request_id"])
                renewed.add(request_id)
                expires_at = row.get("generation_lease_expires_at")
                if request_id in self._active_generations and expires_at is not None:
                    # the server's absolute expiry is validated but not stored: comparing it to a
                    # local wall clock is what the skew this fix removes would corrupt. the lease
                    # length is fixed for the outbox, so the renewal is dated from the monotonic
                    # instant the request left here, which is the earliest the server could have
                    # started counting.
                    _parse_datetime(expires_at)
                    if self._generation_lease_seconds is not None:
                        self._generation_lease_deadlines[request_id] = (
                            renewal_started_at + self._generation_lease_seconds
                        )
            missing = (
                (set(batch) - renewed) & self._active_generations
            ) - self._terminal_generations
            if missing:
                self._active_generations.difference_update(missing)
                for request_id in missing:
                    self._generation_lease_deadlines.pop(request_id, None)
                raise UsageOutboxError("generation_heartbeat_lease_lost")

    async def _claim(self) -> list[dict[str, Any]]:
        data = await self._rpc(
            "claim_serving_usage_batch",
            {
                "p_worker_id": self._worker_id,
                "p_limit": _DELIVERY_CONCURRENCY,
                "p_lease_seconds": self._lease_seconds,
            },
        )
        if not isinstance(data, list) or len(data) > _DELIVERY_CONCURRENCY:
            raise UsageOutboxError("usage_claim_invalid")
        rows = [dict(row) for row in data if isinstance(row, dict)]
        if len(rows) != len(data):
            raise UsageOutboxError("usage_claim_invalid")
        for row in rows:
            outbox_id = row.get("id")
            if not isinstance(outbox_id, str) or not outbox_id:
                raise UsageOutboxError("usage_claim_invalid")
            self._active_leases.add(outbox_id)
        return rows

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
        if status == 409 and _settlement_lease_lost(response):
            # Another worker, expiry or shutdown owns this transition. Do not
            # quarantine an immutable event because this delivery lost its lease.
            # The database will reclaim pending/expired work idempotently.
            self._active_leases.discard(outbox_id)
            return
        if status in {408, 429} or status >= 500:
            await self._retry_or_quarantine(outbox_id, attempt, f"http_{status}")
            return
        if 400 <= status < 500:
            await self._quarantine(outbox_id, f"permanent_http_{status}")
            return
        try:
            result = response.json()
            if not isinstance(result, Mapping):
                raise TypeError
            for field in ("exactCostMicroUsd", "billedCents"):
                if type(result[field]) is not int or result[field] < 0:
                    raise ValueError("invalid settlement amount")
            if type(result.get("replay", False)) is not bool:
                raise ValueError("invalid settlement replay flag")
            ack = {
                "usage_id": result.get("usageId"),
                "ledger_id": result.get("ledgerId"),
                "price_version": result["priceVersion"],
                "exact_cost_micro_usd": result["exactCostMicroUsd"],
                "billed_cents": result["billedCents"],
                "replay": result.get("replay", False),
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
                self._generation_lease_deadlines.clear()

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


def _settlement_lease_lost(response: httpx.Response) -> bool:
    try:
        body = response.json()
        return body.get("detail", {}).get("code") == "serving_usage_lease_lost"
    except (ValueError, AttributeError):
        return False


def _is_transient_rpc_error(exc: BaseException) -> bool:
    return isinstance(exc, UsageOutboxError) and is_transient_rpc_code(str(exc))


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
    org_id = row.get("org_id")
    if not isinstance(org_id, str) or not org_id.strip():
        raise UsageOutboxError("usage_principal_invalid")
    if kind == "freesolo_org":
        return {"kind": "freesolo_org", "orgId": org_id}
    if kind == "trusted_internal" and row.get("billing_attribution_explicit") is True:
        return {
            "kind": "trusted_internal",
            "orgId": org_id,
            "billingAttributionExplicit": True,
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
