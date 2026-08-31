"""Adapter -> base-model routing for multi-LoRA serving. CPU-side front door: tracks which
base model each adapter belongs to and dispatches to that base model's engine. No modal/vllm
imports, so the routing layer is unit-testable offline against a fake pool.

This module owns app construction only. The routes live in ``adapter_routes`` (control-plane
lifecycle) and ``inference_routes`` (caller-facing generation); both reach the app's collaborators
through ``ServingContext``, which this builder attaches to ``app.state``.
"""

# Do NOT add `from __future__ import annotations`: the FastAPI handlers use closure-local body
# models as annotations, which the future import turns into unresolvable strings -> silent 422.

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from flash.serve.contract.protocol import MAX_CHAT_REQUEST_BYTES
from flash.serving.src.accounting.usage_outbox import OfflineUsageStore, UsageStore
from flash.serving.src.http.adapter_routes import adapter_router
from flash.serving.src.http.body_limit import RequestBodyLimitMiddleware
from flash.serving.src.http.context import APP_STATE_ATTR, ServingContext
from flash.serving.src.http.inference_routes import inference_router
from flash.serving.src.http.routing import AdapterRouter, EnginePool, health_body
from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.store.lookup import AdapterLookup

THINKING_STRUCTURED_OUTPUTS_DEFERRED_CAPABILITY = "thinking_structured_outputs_deferred_v1"
_CLEANUP_TIMEOUT_SECONDS = 10.0

_CAPABILITIES = (
    "permanent_checkpoint_identity",
    THINKING_STRUCTURED_OUTPUTS_DEFERRED_CAPABILITY,
)


def build_offline_serving_app(
    pool: EnginePool,
    router: AdapterRouter,
    **kwargs: Any,
):
    """Build an explicitly unmetered app for hermetic tests and local offline use."""
    return build_serving_app(pool, router, usage_store=OfflineUsageStore(), **kwargs)


def build_serving_app(
    pool: EnginePool,
    router: AdapterRouter,
    *,
    internal_key: str | None = None,
    deployment_sha: str = "",
    deployment_id: str = "",
    reload_records: Callable[[], list[AdapterRecord]] | None = None,
    lookup_record: Callable[[str, str], AdapterRecord | None] | None = None,
    reload_interval_seconds: float = 30.0,
    usage_store: UsageStore,
    chat_authorizer: Callable[[str, str, dict[str, str]], Awaitable["str | None"]] | None = None,
):
    """Front-door FastAPI app. ``reload_records`` re-reads persisted ready adapters so a router
    that missed a (un)registration on another container still resolves it: reload once on a miss
    before 404-ing, and at most once per ``reload_interval_seconds`` on a hit.

    ``lookup_record`` reads one persisted adapter regardless of lifecycle status for control-plane
    status requests. Its result is never inserted into the ready-only routing registry.

    ``usage_store`` is explicit: hosted construction passes the durable store, while offline tests
    must deliberately pass ``OfflineUsageStore``. There is no configuration-based fallback from a
    partially wired hosted deployment to unmetered serving.

    External chat/inference auth is ALWAYS enforced. ``chat_authorizer`` authorizes a user request:
    it is called with ``(freesolo_api_key, adapter_id)`` and must raise an ``HTTPException`` (401/403)
    when the key's org does not own the adapter (a base-model serve is authorized for any valid key).
    It returns the caller's org id, which bills a base-model serve to the caller (no adapter owner).
    Trusted server-to-server callers presenting the shared internal key bypass it. If no
    ``chat_authorizer`` is wired, a non-internal request fails closed (503) — prod always wires it.
    """
    context = ServingContext(
        pool,
        router,
        AdapterLookup(
            router,
            reload_records,
            lookup_record=lookup_record,
            reload_interval_seconds=reload_interval_seconds,
        ),
        usage_store,
        internal_key=internal_key,
        deployment_id=deployment_id,
        serving_release=deployment_sha,
        reload_records=reload_records,
        lookup_record=lookup_record,
        chat_authorizer=chat_authorizer,
    )

    api = FastAPI(
        title="Freesolo LoRA Serving (multi base model)",
        version="0.2.0",
        lifespan=_lifespan_for(context, chat_authorizer),
    )
    setattr(api.state, APP_STATE_ATTR, context)
    # fastapi resolves body parameters before handlers run, so cap the raw receive channel first.
    api.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_CHAT_REQUEST_BYTES)

    @api.get("/healthz", tags=["system"])
    async def healthz() -> Any:
        body = health_body(
            router,
            deployment_sha=deployment_sha,
            deployment_id=deployment_id,
            capabilities=list(_CAPABILITIES),
        )
        # a replica that cannot settle usage must not stay in rotation taking chargeable traffic.
        try:
            usage_store.assert_healthy()
        except Exception:
            body["ok"] = False
            body["accounting_ok"] = False
            return JSONResponse(status_code=503, content=body)
        body["accounting_ok"] = True
        return body

    api.include_router(adapter_router)
    api.include_router(inference_router)
    return api


def _lifespan_for(
    context: ServingContext,
    chat_authorizer: Any,
):
    """Build the app's ordered shutdown lifespan."""

    @contextlib.asynccontextmanager
    async def _lifespan(_app: "FastAPI"):
        try:
            await context.usage.start()
            yield
        finally:
            try:
                await context.usage.aclose()
            finally:
                # close persistent authorization clients even when durable shutdown fails.
                for client_owner in (chat_authorizer,):
                    aclose = getattr(client_owner, "aclose", None)
                    if aclose is not None:
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(aclose(), timeout=_CLEANUP_TIMEOUT_SECONDS)

    return _lifespan
