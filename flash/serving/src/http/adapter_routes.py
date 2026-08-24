"""Adapter lifecycle routes: list, register, read, activate, undeploy.

Split out of router.py's app builder, where these were nested handlers closing over a dozen app
variables. They reach that state through ``ServingContext.of(request)`` instead, so each handler is
a module-scope function that can be read on its own.

Every route here is gated on the shared internal key: these are control-plane operations, not
caller-facing inference.
"""

from typing import Any, TypeGuard

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from flash.serving.src.http.context import ServingContext
from flash.serving.src.io.schemas import (
    AdapterActivationRequest,
    AdapterRecord,
    ImmutableAdapterRegistration,
    internal_adapter_payload,
)
from flash.serving.src.io.serving_io import _assert_supported_base_model, _replace_stored_cas
from flash.serving.src.store.registration import activate_revision, persist_revision
from flash.serving.src.store.undeploy import (
    apply_teardown,
    disable_matched,
    get_authoritative,
    resolve_undeploy_target,
    undeploy_body,
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


async def _claim_loading_generation(stored: AdapterRecord) -> tuple[AdapterRecord, bool]:
    """Persist the exact generation before dispatching a disabled revision load."""

    if stored.status == "ready" or stored.updated_at is None:
        return stored, False
    claimed = await _replace_stored_cas(
        stored.model_copy(update={"deployment_generation": stored.updated_at}),
        expected_updated_at=stored.updated_at,
    )
    if claimed is not None:
        return claimed, True
    return (await get_authoritative(stored.adapter_id) or stored), False


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
    stored, claimed = await _claim_loading_generation(stored)

    context.router.upsert(alias, revive=True)
    context.router.upsert(stored, revive=True)

    if claimed:
        background.add_task(_register_revision, context, stored)
    return stored


def _is_same_generation_winner(
    current: AdapterRecord | None, registration: AdapterRecord
) -> TypeGuard[AdapterRecord]:
    """Return whether another attempt promoted this lifecycle generation."""

    return (
        current is not None
        and current.status == "ready"
        and current.deployment_generation == registration.deployment_generation
    )


def _is_unfenced_original(
    current: AdapterRecord | None, registration: AdapterRecord
) -> TypeGuard[AdapterRecord]:
    """Return whether the original disabled lifecycle generation remains unfenced."""

    return (
        current is not None
        and current.status == "disabled"
        and current.updated_at == registration.updated_at
    )


async def _reconcile_failed_promotion(context: ServingContext, registration: AdapterRecord) -> None:
    """Keep a failed promotion from orphaning its loaded gpu adapter."""

    try:
        current = await get_authoritative(registration.adapter_id)
        if _is_same_generation_winner(current, registration):
            # another attempt from this shared lifecycle generation won the promotion. adopt its
            # durable row instead of unloading the gpu registration that now legitimately backs it.
            context.router.upsert(current, revive=True)
            return

        if _is_unfenced_original(current, registration):
            # advance updated_at before unloading so every in-flight promoter sharing this lifecycle
            # generation loses its old cas. a concurrent winner is adopted after a lost fence race.
            expected_updated_at = current.updated_at
            # unreachable from _register_revision; make the optional field explicit at this boundary.
            if expected_updated_at is None:
                return
            fenced = await _replace_stored_cas(
                current.model_copy(update={"deployment_generation": None}),
                expected_updated_at=expected_updated_at,
            )
            if fenced is None:
                current = await get_authoritative(registration.adapter_id)
                if _is_same_generation_winner(current, registration):
                    context.router.upsert(current, revive=True)
                    return
                if _is_unfenced_original(current, registration):
                    print(
                        f"adapter registration reconciliation skipped for "
                        f"{registration.adapter_id}: lifecycle generation could not be fenced",
                        flush=True,
                    )
                    return
            else:
                context.router.upsert(fenced, revive=True)
    except HTTPException as exc:
        print(
            f"adapter registration reconciliation skipped for {registration.adapter_id}: {exc!r}",
            flush=True,
        )
        return

    await context.unregister_safe(
        registration.base_model,
        registration.adapter_id,
        registration.deployment_generation,
    )


async def _register_revision(context: ServingContext, stored: AdapterRecord) -> None:
    """Load the revision onto its gpu engine, then promote the durable row to ready.

    Deferred to a background task so registration returns as soon as the row is durable: the
    engine load may cold-start a scaled-to-zero container. A failed load simply leaves the
    revision disabled, which a later registration retries.
    """
    if stored.status == "ready" or stored.updated_at is None:
        return
    if stored.deployment_generation is None:
        return
    registration = stored
    try:
        await context.pool.register(registration.base_model, registration)
    except Exception:
        await _reconcile_failed_promotion(context, registration)
        return
    try:
        committed = await _replace_stored_cas(
            registration.model_copy(update={"status": "ready"}),
            expected_updated_at=registration.updated_at,
        )
    except HTTPException:
        await _reconcile_failed_promotion(context, registration)
        return
    if committed is None:
        await _reconcile_failed_promotion(context, registration)
        return
    context.router.upsert(committed, revive=True)


@adapter_router.get("/adapters/{adapter_id}")
async def get_adapter(adapter_id: str, request: Request) -> dict[str, Any]:
    context = ServingContext.of(request)
    context.assert_internal(request)
    record = await context.lookup.get_exact(adapter_id)
    lifecycle_state = record.status
    if record.status == "disabled" and record.deployment_generation is not None:
        lifecycle_state = "loading"
    return {
        "ok": True,
        "adapter": {
            **internal_adapter_payload(record),
            "status": lifecycle_state,
            "lifecycle_state": lifecycle_state,
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
        # a base-model serve has no durable row: `_base_model_records()` seeds it in memory and
        # every replica re-adds it on each reload. removing it here would clear one replica's
        # memory, report success, and be undone by the next reload -- while evicting the shared
        # engine's base weights out from under every other tenant on the way.
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"{adapter_id} is a base model, not a deployed adapter",
        )

    # phase 1: compare-and-swap every matched row to "disabled" in persistence first, collecting
    # the rows that durably converged.
    result = await disable_matched(matches, get_authoritative=get_authoritative)

    # phase 2: remove every durably disabled row from routing immediately. gpu eviction is
    # deferred until after either the success or conflict response, so a scaled-to-zero
    # engine's cold start cannot make undeploy callers time out.
    for cleanup_record, expected_generation in apply_teardown(
        context.router, result.pending_teardown
    ):
        background_tasks.add_task(
            context.unregister_safe,
            cleanup_record.base_model,
            cleanup_record.adapter_id,
            expected_generation,
        )

    if failure_response := result.failure_response(run_id, background_tasks):
        return failure_response

    return undeploy_body(
        adapter_id,
        run_id,
        matches[0].base_model,
        result.disabled_aliases,
        result.disabled_revisions,
    )
