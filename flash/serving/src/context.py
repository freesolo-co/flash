"""The collaborators one serving app hands to its request handlers.

Split out of router.py's app builder, which held these as a dozen closure variables shared by nine
nested handlers. Bundling them lets the handlers live at module scope on an ``APIRouter``, so each
is measurable and testable on its own instead of only through ``build_serving_app``.

Everything here is per-app, not per-request: one instance is built by ``build_serving_app`` and
reached from a handler through ``ServingContext.of(request)``.
"""

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, NoReturn

from fastapi import HTTPException, Request, status

from flash.serving.src.http_headers import assert_internal, bearer_token, is_trusted_internal
from flash.serving.src.lookup import AdapterLookup
from flash.serving.src.model_config import is_supported_base_model
from flash.serving.src.openrouter_auth import OpenRouterAuthorization
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
        openrouter_authorization: OpenRouterAuthorization | None,
    ) -> None:
        self.pool = pool
        self.router = router
        self.lookup = lookup
        self.usage = usage
        self.internal_key = internal_key
        self.reload_records = reload_records
        self.lookup_record = lookup_record
        self.chat_authorizer = chat_authorizer
        self.openrouter_authorization = openrouter_authorization
        # the shared internal key lets a trusted server-to-server caller skip external chat auth: it
        # guards /adapters and is presented by the flash control plane (registration) and the backend
        # /api/sample proxy. compared with hmac.compare_digest to avoid timing leaks.
        self.trusted_internal_keys = (internal_key,) if internal_key else ()

    @staticmethod
    def of(request: Request) -> "ServingContext":
        return getattr(request.app.state, APP_STATE_ATTR)

    def assert_internal(self, request: Request) -> None:
        if is_trusted_internal(request, self.trusted_internal_keys):
            return
        token = bearer_token(request)
        if token is not None and self._matches_openrouter(token):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "OpenRouter is not authorized for this route",
            )
        assert_internal(request, self.internal_key)

    @staticmethod
    def _raise_bearer_unauthorized(detail: str) -> NoReturn:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _external_bearer(self, request: Request) -> str:
        token = bearer_token(request)
        if token is None:
            self._raise_bearer_unauthorized(
                "Missing Freesolo API key (send 'Authorization: Bearer <key>')"
            )
        return token

    async def _authorize_freesolo(self, token: str, adapter_id: str) -> str | None:
        if self.chat_authorizer is None:
            # no authorizer wired -> fail closed rather than serve open (prod always wires it).
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "serving auth is not configured"
            )
        try:
            return await self.chat_authorizer(token, adapter_id)
        except HTTPException as exc:
            if exc.status_code != status.HTTP_401_UNAUTHORIZED:
                raise
            headers = dict(exc.headers or {})
            headers.setdefault("WWW-Authenticate", "Bearer")
            raise HTTPException(exc.status_code, exc.detail, headers=headers) from exc

    def _matches_openrouter(self, token: str) -> bool:
        authorization = self.openrouter_authorization
        return authorization is not None and authorization.matches(token)

    def _authorize_openrouter_base_model(self, model_id: str) -> str:
        authorization = self.openrouter_authorization
        if authorization is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "OpenRouter auth is unavailable"
            )
        if not is_supported_base_model(model_id):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "OpenRouter is authorized only for canonical hosted base models",
            )

        requested = self.router.get(model_id)
        if requested is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "hosted base model routing record is unavailable",
            )
        resolved = self.router.resolve(model_id)
        canonical = (
            requested.serve_base_model
            and requested.status == "ready"
            and requested.adapter_id == requested.base_model == requested.repo_id == model_id
            and requested.org_id is None
            and requested.checkpoint is None
            and not requested.is_alias
            and not requested.is_revision
            and resolved is not None
            and resolved[0] is requested
            and resolved[1] is requested
        )
        if not canonical:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "OpenRouter is authorized only for canonical hosted base models",
            )
        return authorization.settlement_org_id

    async def authorize_inference(self, request: Request, adapter_id: str) -> str | None:
        """Authorize the legacy inference routes without granting the OpenRouter principal."""
        if is_trusted_internal(request, self.trusted_internal_keys):
            return None
        token = self._external_bearer(request)
        if self._matches_openrouter(token):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "OpenRouter is not authorized for this route",
            )
        return await self._authorize_freesolo(token, adapter_id)

    async def authorize_chat_completion(self, request: Request, model_id: str) -> str | None:
        """Authorize chat, including the base-only provisional OpenRouter principal."""
        if is_trusted_internal(request, self.trusted_internal_keys):
            return None
        token = self._external_bearer(request)
        if self._matches_openrouter(token):
            return self._authorize_openrouter_base_model(model_id)
        return await self._authorize_freesolo(token, model_id)

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
        # gpu cleanup is best-effort and may cold-start a scaled-to-zero engine. the engine compares
        # this deployment generation under its per-adapter lock so stale cleanup cannot remove a
        # redeployment of the same immutable revision id.
        with contextlib.suppress(Exception):
            await self.pool.unregister(base_model, adapter_id, expected_generation)

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
