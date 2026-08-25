"""The per-app collaborators serving request handlers use."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import HTTPException, Request, status

from flash.serving.src.accounting.usage import (
    AuthorizedTraffic,
    UsageSession,
    build_usage_session,
    principal_for_external_org,
    principal_for_trusted_internal,
)
from flash.serving.src.accounting.usage_outbox import RequestIdentity, UsageStore
from flash.serving.src.engine.model_config import hosted_traffic_policy_for
from flash.serving.src.http.headers import _bearer_token, assert_internal, is_trusted_internal
from flash.serving.src.http.routing import AdapterRouter, EnginePool
from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.io.streaming import generate_once, openai_chat_stream, prepare_stream
from flash.serving.src.store.lookup import AdapterLookup
from flash.serving.src.traffic.admission import AdmissionController, AdmissionLease
from flash.serving.src.traffic.capacity import (
    CAPACITY_POLL_INTERVAL_SECONDS,
    CapacityProvider,
    ConfiguredCapacityProvider,
)

APP_STATE_ATTR = "serving_context"


class ServingContext:
    def __init__(
        self,
        pool: EnginePool,
        router: AdapterRouter,
        lookup: AdapterLookup,
        usage: UsageStore,
        capacity: CapacityProvider | None = None,
        *,
        internal_key: str | None,
        deployment_id: str,
        serving_release: str,
        reload_records: Callable[[], list[AdapterRecord]] | None,
        lookup_record: Callable[[str], AdapterRecord | None] | None,
        chat_authorizer: Callable[[str, str], Awaitable[str | AuthorizedTraffic | None]] | None,
    ) -> None:
        self.pool = pool
        self.router = router
        self.lookup = lookup
        self.usage = usage
        self.capacity = capacity or ConfiguredCapacityProvider()
        self.admission = AdmissionController(
            hosted_traffic_policy_for,
            self.capacity.current_dispatch_capacity,
        )
        self.internal_key = internal_key
        self.deployment_id = deployment_id
        self.serving_release = serving_release
        self.reload_records = reload_records
        self.lookup_record = lookup_record
        self.chat_authorizer = chat_authorizer
        self.trusted_internal_keys = (internal_key,) if internal_key else ()
        bind_capacity_changed = getattr(self.capacity, "bind_capacity_changed", None)
        if bind_capacity_changed is not None:
            bind_capacity_changed(self.admission.capacity_changed)

    @staticmethod
    def of(request: Request) -> "ServingContext":
        return getattr(request.app.state, APP_STATE_ATTR)

    def assert_internal(self, request: Request) -> None:
        assert_internal(request, self.internal_key)

    async def authorize_inference(self, request: Request, adapter_id: str) -> AuthorizedTraffic:
        if is_trusted_internal(request, self.trusted_internal_keys):
            return AuthorizedTraffic(principal=principal_for_trusted_internal())
        token = _bearer_token(request)
        if not token:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing Freesolo API key (send 'Authorization: Bearer <key>')",
            )
        if self.chat_authorizer is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "serving auth is not configured"
            )
        authorized = await self.chat_authorizer(token, adapter_id)
        if isinstance(authorized, AuthorizedTraffic):
            if authorized.principal.kind == "trusted_internal":
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "serving auth did not return an attributable principal",
                )
            return authorized
        if isinstance(authorized, str) and authorized:
            return AuthorizedTraffic(principal=principal_for_external_org(authorized))
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "serving auth did not return an attributable principal",
        )

    def reject_unsettleable_thinking(self, payload: Any, target: AdapterRecord) -> None:
        if not self.usage.enabled:
            return
        thinking = target.thinking
        if target.serve_base_model:
            kwargs = getattr(payload, "chat_template_kwargs", None)
            override = kwargs.get("enable_thinking") if isinstance(kwargs, dict) else None
            if isinstance(override, bool):
                thinking = override
        if thinking:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "thinking generation accounting is unavailable",
            )

    async def reload_if_configured(self) -> None:
        if self.reload_records is not None:
            await self.lookup.reload()

    async def refresh_capacity_once(self) -> None:
        await asyncio.gather(
            *(
                self.capacity.capacity_snapshot(model, self.admission.active_count(model))
                for model in self.router.base_models()
            ),
            return_exceptions=True,
        )

    async def refresh_capacity_periodically(self) -> None:
        while True:
            await self.refresh_capacity_once()
            await asyncio.sleep(CAPACITY_POLL_INTERVAL_SECONDS)

    async def acquire_admission(self, model: str) -> AdmissionLease:
        return await self.admission.acquire(model)

    def admission_headers(self, model: str, lease: AdmissionLease) -> dict[str, str]:
        snapshot = self.admission.snapshot(model)
        return {
            "X-Freesolo-Queue-Duration-Seconds": f"{lease.queue_duration_seconds:.6f}",
            "X-Freesolo-Application-Active": str(snapshot.active),
            "X-Freesolo-Application-Pending": str(snapshot.queued),
        }

    async def unregister_safe(
        self, base_model: str, adapter_id: str, expected_generation: str | None
    ) -> None:
        # gpu cleanup may reach a replacement engine. the engine compares this deployment
        # generation under its per-adapter lock so stale cleanup cannot remove a redeployment of the
        # same immutable revision id. durable routing is already disabled, but an exact eviction
        # failure must remain observable rather than making the successful api response imply it ran.
        try:
            await self.pool.unregister(base_model, adapter_id, expected_generation)
        except Exception as error:
            print(
                f"hosted adapter gpu cleanup failed for {adapter_id} on {base_model}: {error!r}",
                flush=True,
            )

    async def generate(
        self,
        payload: Any,
        requested: AdapterRecord,
        target: AdapterRecord,
        *,
        identity: RequestIdentity,
        traffic: AuthorizedTraffic,
        captured_at: Any,
        expected_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        result = await generate_once(
            self.pool,
            self.router,
            payload,
            requested,
            target,
            generation_id=identity.request_id,
            require_generation_id=self.usage.enabled,
            expected_checkpoint=expected_checkpoint,
        )
        session = build_usage_session(
            self.usage,
            identity,
            traffic.principal,
            requested,
            target,
            result,
            deployment_id=self.deployment_id,
            serving_release=self.serving_release,
            captured_at=captured_at,
        )
        finalization = asyncio.create_task(session.finalize(result))
        try:
            await asyncio.shield(finalization)
        except asyncio.CancelledError:
            await finalization
            raise
        return result

    def usage_session(
        self,
        identity: RequestIdentity,
        traffic: AuthorizedTraffic,
        requested: AdapterRecord,
        target: AdapterRecord,
        result: dict[str, Any],
        captured_at: Any,
    ) -> UsageSession:
        return build_usage_session(
            self.usage,
            identity,
            traffic.principal,
            requested,
            target,
            result,
            deployment_id=self.deployment_id,
            serving_release=self.serving_release,
            captured_at=captured_at,
        )

    async def prepare_stream(
        self,
        payload: Any,
        requested: AdapterRecord,
        target: AdapterRecord,
        *,
        generation_id: str,
        expected_checkpoint: str | None,
    ) -> tuple[AsyncIterator[dict[str, Any]], dict[str, str], bool, dict[str, Any]]:
        return await prepare_stream(
            self.pool,
            self.router,
            payload,
            requested,
            target,
            generation_id=generation_id,
            require_generation_id=self.usage.enabled,
            expected_checkpoint=expected_checkpoint,
        )

    def chat_stream(
        self,
        *,
        record: AdapterRecord,
        events: AsyncIterator[dict[str, Any]],
        adapter_id: str,
        completion_id: str,
        created: int,
        include_usage: bool,
        usage_session: UsageSession,
        thinking: bool = False,
    ) -> AsyncIterator[bytes]:
        return openai_chat_stream(
            self.router,
            record=record,
            events=events,
            adapter_id=adapter_id,
            completion_id=completion_id,
            created=created,
            include_usage=include_usage,
            usage_session=usage_session,
            thinking=thinking,
        )
