"""The collaborators one serving app hands to its request handlers.

Split out of router.py's app builder, which held these as a dozen closure variables shared by nine
nested handlers. Bundling them lets the handlers live at module scope on an ``APIRouter``, so each
is measurable and testable on its own instead of only through ``build_serving_app``.

Everything here is per-app, not per-request: one instance is built by ``build_serving_app`` and
reached from a handler through ``ServingContext.of(request)``.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import HTTPException, Request, status

from flash.serving.src.http_headers import _bearer_token, assert_internal, is_trusted_internal
from flash.serving.src.lookup import AdapterLookup
from flash.serving.src.routing import AdapterRouter, EnginePool
from flash.serving.src.schemas import AdapterRecord
from flash.serving.src.streaming import generate_once, openai_chat_stream, prepare_stream
from flash.serving.src.usage import UsageReporter

APP_STATE_ATTR = "serving_context"


class ServingContext:
    def __init__(
        self,
        pool: EnginePool,
        router: AdapterRouter,
        lookup: AdapterLookup,
        usage: UsageReporter,
        *,
        internal_key: str | None,
        reload_records: Callable[[], list[AdapterRecord]] | None,
        lookup_record: Callable[[str], AdapterRecord | None] | None,
        chat_authorizer: Callable[[str, str], Awaitable[str | None]] | None,
    ) -> None:
        self.pool = pool
        self.router = router
        self.lookup = lookup
        self.usage = usage
        self.internal_key = internal_key
        self.reload_records = reload_records
        self.lookup_record = lookup_record
        self.chat_authorizer = chat_authorizer
        # the shared internal key lets a trusted server-to-server caller skip external chat auth: it
        # guards /adapters and is presented by the flash control plane (registration) and the backend
        # /api/sample proxy. compared with hmac.compare_digest to avoid timing leaks.
        self.trusted_internal_keys = (internal_key,) if internal_key else ()

    @staticmethod
    def of(request: Request) -> "ServingContext":
        return getattr(request.app.state, APP_STATE_ATTR)

    def assert_internal(self, request: Request) -> None:
        assert_internal(request, self.internal_key)

    async def authorize_inference(self, request: Request, adapter_id: str) -> str | None:
        """Gate every chat/inference request on a Freesolo API key, and resolve its billing org.

        Always enforced: a valid Freesolo API key is required. For a LoRA adapter the key's org must
        own it; for a base-model serve any valid key is accepted (no owner) -- the backend decides.
        The authorizer returns the caller's org id, which bills a base-model serve to the caller (no
        adapter owner). Trusted internal callers (the backend proxy / control plane) bypass via the
        internal key and return None. With no authorizer wired we fail closed (503).
        """
        if is_trusted_internal(request, self.trusted_internal_keys):
            return None
        token = _bearer_token(request)
        if not token:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing Freesolo API key (send 'Authorization: Bearer <key>')",
            )
        if self.chat_authorizer is None:
            # No authorizer wired -> fail closed rather than serve open (prod always wires it).
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "serving auth is not configured"
            )
        return await self.chat_authorizer(token, adapter_id)

    async def reload_if_configured(self) -> None:
        if self.reload_records is not None:
            await self.lookup.reload()

    def schedule_usage(
        self, record: AdapterRecord, result: dict[str, Any], caller_org: str | None
    ) -> None:
        self.usage.schedule(record, result, caller_org)

    async def unregister_safe(
        self, base_model: str, adapter_id: str, expected_generation: str | None
    ) -> None:
        # gpu cleanup may cold-start a scaled-to-zero engine. the engine compares this deployment
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
        expected_checkpoint: str | None = None,
        caller_org: str | None = None,
    ) -> dict[str, Any]:
        return await generate_once(
            self.pool,
            self.router,
            self.schedule_usage,
            payload,
            requested,
            target,
            expected_checkpoint=expected_checkpoint,
            caller_org=caller_org,
        )

    async def prepare_stream(
        self,
        payload: Any,
        requested: AdapterRecord,
        target: AdapterRecord,
        *,
        expected_checkpoint: str | None,
    ) -> tuple[AsyncIterator[dict[str, Any]], dict[str, str], bool]:
        return await prepare_stream(
            self.pool,
            self.router,
            payload,
            requested,
            target,
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
        caller_org: str | None,
        thinking: bool = False,
    ) -> AsyncIterator[bytes]:
        return openai_chat_stream(
            self.router,
            self.schedule_usage,
            record=record,
            events=events,
            adapter_id=adapter_id,
            completion_id=completion_id,
            created=created,
            include_usage=include_usage,
            caller_org=caller_org,
            thinking=thinking,
        )
