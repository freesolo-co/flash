"""Adapter lifecycle routes: list, register, read, activate, undeploy.

Split out of router.py's app builder, where these were nested handlers closing over a dozen app
variables. They reach that state through ``ServingContext.of(request)`` instead, so each handler is
a module-scope function that can be read on its own.

Every route here is gated on the shared internal key: these are control-plane operations, not
caller-facing inference.
"""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import JSONResponse

from flash.serving.src.context import ServingContext
from flash.serving.src.registration import activate_revision, persist_revision
from flash.serving.src.schemas import (
    AdapterActivationRequest,
    AdapterRecord,
    ImmutableAdapterRegistration,
    internal_adapter_payload,
)
from flash.serving.src.serving_io import _assert_supported_base_model, _replace_stored_cas
from flash.serving.src.undeploy import (
    apply_teardown,
    disable_matched,
    get_authoritative,
    resolve_undeploy_target,
    undeploy_body,
    undeploy_conflict_detail,
)

adapter_router = APIRouter(tags=["adapters"])


@adapter_router.get("/adapters")
async def list_adapters(request: Request) -> dict[str, Any]:
    # Gate the listing behind the internal key, same as register/teardown. Even with org_id
    # excluded from the schema, repo_id + url still leak the adapter->tenant mapping and HF
    # namespaces to anon callers. Internal consumers (backend/deployment) already send
    # X-Freesolo-Internal-Key, so this preserves their access while blocking enumeration.
    context = ServingContext.of(request)
    context.assert_internal(request)
    await context.reload_if_configured()
    return {"ok": True, "adapters": context.router.ready_adapters()}


@adapter_router.post("/adapters")
async def add_adapter(
    registration: ImmutableAdapterRegistration,
    request: Request,
    background: BackgroundTasks,
) -> AdapterRecord:
    context = ServingContext.of(request)
    context.assert_internal(request)
    _assert_supported_base_model(registration.base_model)
    revision = registration.to_record()

    alias, stored = await persist_revision(context.router, revision)

    context.router.upsert(alias, revive=True)
    context.router.upsert(stored, revive=True)

    background.add_task(_register_revision, context, stored)
    return stored


async def _register_revision(context: ServingContext, stored: AdapterRecord) -> None:
    """Load the revision onto its gpu engine, then promote the durable row to ready.

    Deferred to a background task so registration returns as soon as the row is durable: the
    engine load may cold-start a scaled-to-zero container. A failed load simply leaves the
    revision disabled, which a later registration retries.
    """
    if stored.status == "ready" or stored.updated_at is None:
        return
    # the disabled row's cas timestamp is the lifecycle generation. concurrent registration
    # attempts for this same row share it; a later disable writes a new timestamp.
    registration = stored.model_copy(update={"deployment_generation": stored.updated_at})
    try:
        await context.pool.register(registration.base_model, registration)
    except Exception:  # a failed load leaves the revision disabled
        return
    try:
        committed = await _replace_stored_cas(
            registration.model_copy(update={"status": "ready"}),
            expected_updated_at=registration.updated_at,
        )
    except HTTPException:
        return
    if committed is not None:
        context.router.upsert(committed, revive=True)


@adapter_router.get("/adapters/{adapter_id}")
async def get_adapter(adapter_id: str, request: Request) -> dict[str, Any]:
    context = ServingContext.of(request)
    context.assert_internal(request)
    record = await context.lookup.get_exact(adapter_id)
    return {
        "ok": True,
        "adapter": {
            **internal_adapter_payload(record),
            "lifecycle_state": "ready",
        },
    }


@adapter_router.post("/adapters/{revision_id}/activate")
async def activate_adapter(
    revision_id: str, payload: AdapterActivationRequest, request: Request
) -> dict[str, Any]:
    context = ServingContext.of(request)
    context.assert_internal(request)
    return await activate_revision(context.router, revision_id, payload.expected_adapter_revision)


@adapter_router.delete("/adapters/{adapter_id}")
async def remove_adapter(
    adapter_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    context = ServingContext.of(request)
    context.assert_internal(request)
    await context.reload_if_configured()

    base_record, run_id, matches = await resolve_undeploy_target(context.router, adapter_id)
    if base_record is not None:
        # A base-model serve has no durable row: `_base_model_records()` seeds it in memory and
        # every replica re-adds it on each reload. Removing it here would clear one replica's
        # memory, report success, and be undone by the next reload -- while evicting the shared
        # engine's base weights out from under every other tenant on the way.
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"{adapter_id} is a base model, not a deployed adapter",
        )

    # phase 1: compare-and-swap every matched row to "disabled" in persistence first, collecting
    # the rows that durably converged.
    disabled_aliases, disabled_revisions, stuck_ready, pending_teardown = await disable_matched(
        matches, get_authoritative=get_authoritative
    )

    # phase 2: remove every durably disabled row from routing immediately. gpu eviction is
    # deferred until after either the success or conflict response, so a scaled-to-zero
    # engine's cold start cannot make undeploy callers time out.
    for cleanup_record, expected_generation in apply_teardown(context.router, pending_teardown):
        background_tasks.add_task(
            context.unregister_safe,
            cleanup_record.base_model,
            cleanup_record.adapter_id,
            expected_generation,
        )

    if stuck_ready:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=undeploy_conflict_detail(
                run_id, disabled_aliases, disabled_revisions, stuck_ready
            ),
            background=background_tasks,
        )

    return undeploy_body(
        adapter_id,
        run_id,
        matches[0].base_model,
        disabled_aliases,
        disabled_revisions,
    )
