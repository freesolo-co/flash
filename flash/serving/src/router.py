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

from flash.serving.src.adapter_routes import adapter_router
from flash.serving.src.context import APP_STATE_ATTR, ServingContext
from flash.serving.src.inference_routes import inference_router
from flash.serving.src.lookup import AdapterLookup
from flash.serving.src.routing import AdapterRouter, EnginePool, health_body
from flash.serving.src.schemas import AdapterRecord
from flash.serving.src.usage import UsageReporter

THINKING_STRUCTURED_OUTPUTS_DEFERRED_CAPABILITY = "thinking_structured_outputs_deferred_v1"
_USAGE_REPORT_DRAIN_TIMEOUT_SECONDS = 45.0

_CAPABILITIES = (
    "immutable_adapter_revisions",
    "alias_compare_and_swap",
    "revision_provenance",
    THINKING_STRUCTURED_OUTPUTS_DEFERRED_CAPABILITY,
)


def build_serving_app(
    pool: EnginePool,
    router: AdapterRouter,
    *,
    internal_key: str | None = None,
    deployment_sha: str = "",
    deployment_id: str = "",
    reload_records: Callable[[], list[AdapterRecord]] | None = None,
    reload_interval_seconds: float = 30.0,
    usage_reporter: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    chat_authorizer: Callable[[str, str], Awaitable["str | None"]] | None = None,
    on_startup: Callable[[], Awaitable[None]] | None = None,
):
    """Front-door FastAPI app. ``reload_records`` re-reads persisted ready adapters so a router
    that missed a (un)registration on another container still resolves it: reload once on a miss
    before 404-ing, and at most once per ``reload_interval_seconds`` on a hit.

    ``usage_reporter`` (optional) is called fire-and-forget after each successful generation with
    a usage dict (adapterId, baseModel, promptTokens, completionTokens, cachedTokens, gpuSeconds,
    requestId, engineReplicaId, servingDeploymentId, cachedTokensReported) so the backend can
    meter/bill it. It runs as a managed detached task, and its failures are swallowed so metering
    never affects serving latency or success. Pending reports drain before the shared client closes.

    External chat/inference auth is ALWAYS enforced. ``chat_authorizer`` authorizes a user request:
    it is called with ``(freesolo_api_key, adapter_id)`` and must raise an ``HTTPException`` (401/403)
    when the key's org does not own the adapter (a base-model serve is authorized for any valid key).
    It returns the caller's org id, which bills a base-model serve to the caller (no adapter owner).
    Trusted server-to-server callers presenting the shared internal key bypass it. If no
    ``chat_authorizer`` is wired, a non-internal request fails closed (503) — prod always wires it.

    ``on_startup`` (optional) runs once as a background task without blocking readiness. Serving wires
    the optional warm-floor hook here; at the production zero floor it returns without starting gpu
    engines. Failures are swallowed and the task is cancelled on shutdown if still running.
    """
    context = ServingContext(
        pool,
        router,
        AdapterLookup(router, reload_records, reload_interval_seconds=reload_interval_seconds),
        UsageReporter(
            usage_reporter,
            deployment_id=deployment_id,
            drain_timeout_seconds=_USAGE_REPORT_DRAIN_TIMEOUT_SECONDS,
        ),
        internal_key=internal_key,
        reload_records=reload_records,
        chat_authorizer=chat_authorizer,
    )

    api = FastAPI(
        title="Freesolo LoRA Serving (multi base model)",
        version="0.2.0",
        lifespan=_lifespan_for(context, on_startup, usage_reporter, chat_authorizer),
    )
    setattr(api.state, APP_STATE_ATTR, context)

    @api.get("/healthz", tags=["system"])
    async def healthz() -> dict[str, Any]:
        return health_body(
            router,
            deployment_sha=deployment_sha,
            deployment_id=deployment_id,
            capabilities=list(_CAPABILITIES),
        )

    api.include_router(adapter_router)
    api.include_router(inference_router)
    return api


def _lifespan_for(
    context: ServingContext,
    on_startup: Callable[[], Awaitable[None]] | None,
    usage_reporter: Any,
    chat_authorizer: Any,
):
    """Build the app's lifespan: optional non-blocking startup, then an ordered shutdown."""

    @contextlib.asynccontextmanager
    async def _lifespan(_app: "FastAPI"):
        # run optional startup work without blocking router readiness. the cpu router must accept
        # traffic immediately, and any background startup failure must not crash it.
        startup_task = None
        if on_startup is not None:

            async def _run_startup() -> None:
                # startup work is best-effort and must never crash the router
                with contextlib.suppress(Exception):
                    await on_startup()

            startup_task = asyncio.create_task(_run_startup())
        yield
        if startup_task is not None and not startup_task.done():
            startup_task.cancel()
            try:
                await startup_task
            except asyncio.CancelledError:
                # shutdown raced our own cancel of the startup task — expected when the
                # container is told to stop mid-startup; nothing to clean up, so swallow it.
                pass
            except Exception:  # best-effort cleanup must not fail shutdown
                pass
        # drain detached usage reports before closing their shared client.
        await context.usage.drain()

        # Close persistent httpx clients (usage reporter + chat authorizer) on shutdown so
        # long-lived containers don't leak sockets / emit "Unclosed client" ResourceWarnings.
        for client_owner in (usage_reporter, chat_authorizer):
            aclose = getattr(client_owner, "aclose", None)
            if aclose is not None:
                # best-effort cleanup must not fail shutdown
                with contextlib.suppress(Exception):
                    await aclose()

    return _lifespan
