"""Adapter -> base-model routing for multi-LoRA serving. CPU-side front door: tracks which
base model each adapter belongs to and dispatches to that base model's engine. No modal/vllm
imports, so the routing layer is unit-testable offline against a fake pool.
"""

# Do NOT add `from __future__ import annotations`: the FastAPI handlers use closure-local body
# models as annotations, which the future import turns into unresolvable strings -> silent 422.

import contextlib
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from flash.serving.src.engine_errors import (
    engine_error_http,
    raise_if_engine_error,
    terminating_on_engine_error,
    value_error_http,
)
from flash.serving.src.http_headers import _bearer_token, assert_internal, is_trusted_internal
from flash.serving.src.lookup import AdapterLookup
from flash.serving.src.model_config import gpu_for, is_supported_base_model
from flash.serving.src.registration import persist_revision
from flash.serving.src.responses import (
    openai_chat_completion,
    openai_generate_fields,
    openai_include_usage,
)
from flash.serving.src.routing import AdapterRouter, EnginePool
from flash.serving.src.schemas import AdapterRecord, internal_adapter_payload
from flash.serving.src.serving_io import (
    _assert_supported_base_model,
    _expected_checkpoint,
    _get_stored,
    _inference_json_response,
    _parse_generate,
    _prepare_generate_request,
    _provenance_headers,
    _replace_stored_cas,
    _revision_provenance,
    _validate_alias,
    _validate_alias_target,
)
from flash.serving.src.streaming import openai_chat_stream, prepare_stream
from flash.serving.src.structured_outputs import StructuredOutputsError
from flash.serving.src.undeploy import apply_teardown, disable_matched
from flash.serving.src.usage import UsageReporter

_FLASH_CHECKPOINT_MODEL_RE = re.compile(
    r"(?P<run_id>flash-[0-9]{1,20}-[0-9a-f]{8})/step-[0-9]{1,18}"
)
THINKING_STRUCTURED_OUTPUTS_DEFERRED_CAPABILITY = "thinking_structured_outputs_deferred_v1"
_USAGE_REPORT_DRAIN_TIMEOUT_SECONDS = 45.0


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
    import asyncio
    import time
    import uuid
    from contextlib import asynccontextmanager

    from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
    from fastapi.responses import JSONResponse, StreamingResponse

    from flash.serving.src.persistence import PersistenceRecordError
    from flash.serving.src.schemas import (
        AdapterActivationRequest,
        GenerateRequest,
        ImmutableAdapterRegistration,
    )

    # the shared internal key lets a trusted server-to-server caller skip external chat auth: it
    # guards /adapters and is presented by the flash control plane (registration) and the backend
    # /api/sample proxy. compared with hmac.compare_digest to avoid timing leaks.
    _trusted_internal_keys = (internal_key,) if internal_key else ()

    @asynccontextmanager
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
        await _usage.drain()

        # Close persistent httpx clients (usage reporter + chat authorizer) on shutdown so
        # long-lived containers don't leak sockets / emit "Unclosed client" ResourceWarnings.
        for client_owner in (usage_reporter, chat_authorizer):
            aclose = getattr(client_owner, "aclose", None)
            if aclose is not None:
                # best-effort cleanup must not fail shutdown
                with contextlib.suppress(Exception):
                    await aclose()

    api = FastAPI(
        title="Freesolo LoRA Serving (multi base model)", version="0.2.0", lifespan=_lifespan
    )

    _lookup_state = AdapterLookup(
        router, reload_records, reload_interval_seconds=reload_interval_seconds
    )
    _usage = UsageReporter(
        usage_reporter,
        deployment_id=deployment_id,
        drain_timeout_seconds=_USAGE_REPORT_DRAIN_TIMEOUT_SECONDS,
    )

    def _assert_internal(request: Request) -> None:
        assert_internal(request, internal_key)

    def _is_trusted_internal(request: Request) -> bool:
        return is_trusted_internal(request, _trusted_internal_keys)

    async def _authorize_inference(request: Request, adapter_id: str) -> "str | None":
        """Gate every chat/inference request on a Freesolo API key, and resolve its billing org.

        Always enforced: a valid Freesolo API key is required. For a LoRA adapter the key's org must
        own it; for a base-model serve any valid key is accepted (no owner) — the backend decides.
        The authorizer returns the caller's org id, which bills a base-model serve to the caller (no
        adapter owner). Trusted internal callers (the backend proxy / control plane) bypass via the
        internal key and return None. With no authorizer wired we fail closed (503).
        """
        if _is_trusted_internal(request):
            return None
        token = _bearer_token(request)
        if not token:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing Freesolo API key (send 'Authorization: Bearer <key>')",
            )
        if chat_authorizer is None:
            # No authorizer wired -> fail closed rather than serve open (prod always wires it).
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "serving auth is not configured"
            )
        return await chat_authorizer(token, adapter_id)

    def _value_error_http(adapter_id: str, exc: ValueError) -> HTTPException:
        return value_error_http(router, adapter_id, exc)

    def _engine_error_http(adapter_id: str, exc: Exception) -> HTTPException | None:
        return engine_error_http(router, adapter_id, exc)

    def _raise_if_engine_error(adapter_id: str, exc: Exception) -> None:
        raise_if_engine_error(router, adapter_id, exc)

    def _terminating_on_engine_error(
        events: AsyncIterator[dict[str, Any]], adapter_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        return terminating_on_engine_error(router, events, adapter_id)

    async def _reload() -> None:
        await _lookup_state.reload()

    async def _lookup(
        adapter_id: str, *, require_supported_base_model: bool = True
    ) -> tuple[AdapterRecord, AdapterRecord]:
        return await _lookup_state.resolve(
            adapter_id, require_supported_base_model=require_supported_base_model
        )

    async def _lookup_exact(adapter_id: str) -> AdapterRecord:
        return await _lookup_state.get_exact(adapter_id)

    @api.get("/healthz", tags=["system"])
    async def healthz() -> dict[str, Any]:
        models = router.base_models()
        supported_models = [m for m in models if is_supported_base_model(m)]
        unsupported_models = [m for m in models if not is_supported_base_model(m)]
        # report configured per-model gpu tiers rather than live container counts, which modal does
        # not expose here. ``gpus`` is the supported base-model engine count and remains stable when
        # demand-driven containers scale to zero.
        gpu_by_model = {m: gpu_for(m) for m in supported_models}
        body = {
            "ok": True,
            "deployment_sha": deployment_sha,
            "deployment_id": deployment_id,
            "capabilities": [
                "immutable_adapter_revisions",
                "alias_compare_and_swap",
                "revision_provenance",
                THINKING_STRUCTURED_OUTPUTS_DEFERRED_CAPABILITY,
            ],
            "base_models": models,
            "gpus": len(supported_models),
            "gpu_by_model": gpu_by_model,
            "gpu_tiers": sorted(set(gpu_by_model.values())),
            "adapters": len(router.ready_adapters()),
        }
        if unsupported_models:
            body["unsupported_base_models"] = unsupported_models
        return body

    @api.get("/adapters", tags=["adapters"])
    async def list_adapters(request: Request) -> dict[str, Any]:
        # Gate the listing behind the internal key, same as register/teardown. Even with org_id
        # excluded from the schema, repo_id + url still leak the adapter->tenant mapping and HF
        # namespaces to anon callers. Internal consumers (backend/deployment) already send
        # X-Freesolo-Internal-Key, so this preserves their access while blocking enumeration.
        _assert_internal(request)
        if reload_records is not None:
            await _reload()
        return {"ok": True, "adapters": router.ready_adapters()}

    @api.post("/adapters", tags=["adapters"])
    async def add_adapter(
        registration: ImmutableAdapterRegistration,
        request: Request,
        background: BackgroundTasks,
    ) -> AdapterRecord:
        _assert_internal(request)
        _assert_supported_base_model(registration.base_model)
        revision = registration.to_record()

        alias, stored = await persist_revision(router, revision)

        router.upsert(alias, revive=True)
        router.upsert(stored, revive=True)

        async def _register_revision() -> None:
            if stored.status == "ready" or stored.updated_at is None:
                return
            # the disabled row's cas timestamp is the lifecycle generation. concurrent registration
            # attempts for this same row share it; a later disable writes a new timestamp.
            registration = stored.model_copy(update={"deployment_generation": stored.updated_at})
            try:
                await pool.register(registration.base_model, registration)
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
                router.upsert(committed, revive=True)

        background.add_task(_register_revision)
        return stored

    @api.get("/adapters/{adapter_id}", tags=["adapters"])
    async def get_adapter(adapter_id: str, request: Request) -> dict[str, Any]:
        _assert_internal(request)
        record = await _lookup_exact(adapter_id)
        return {
            "ok": True,
            "adapter": {
                **internal_adapter_payload(record),
                "lifecycle_state": "ready",
            },
        }

    @api.post("/adapters/{revision_id}/activate", tags=["adapters"])
    async def activate_adapter(
        revision_id: str, payload: AdapterActivationRequest, request: Request
    ) -> dict[str, Any]:
        _assert_internal(request)
        try:
            revision = await _get_stored(revision_id)
        except PersistenceRecordError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, "invalid revision record") from exc
        if revision is None or not revision.is_revision:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown adapter id")
        if revision.status != "ready":
            raise HTTPException(status.HTTP_409_CONFLICT, "adapter revision is not ready")
        run_id = revision.run_id
        assert run_id is not None
        try:
            alias = await _get_stored(run_id)
        except PersistenceRecordError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, "invalid run alias") from exc
        if alias is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "run alias is missing")
        _validate_alias(alias, revision)
        await _validate_alias_target(alias)
        previous = None if alias.status == "disabled" else alias.alias_of
        if payload.expected_adapter_revision != previous:
            raise HTTPException(status.HTTP_409_CONFLICT, "stale adapter revision expectation")
        if alias.updated_at is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "run alias has no CAS authority")

        replacement = alias.model_copy(
            update={
                "status": "ready",
                "metadata": {
                    "record_type": "alias",
                    "run_id": run_id,
                    "alias_of": revision.adapter_id,
                },
            }
        )
        committed = await _replace_stored_cas(
            replacement,
            expected_updated_at=alias.updated_at,
        )
        if committed is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "run alias changed concurrently")
        router.upsert(committed, revive=True)
        router.upsert(revision, revive=True)
        return {
            "adapter_id": run_id,
            "target_adapter_revision": revision.adapter_id,
            "previous_adapter_revision": previous,
            "checkpoint": revision.checkpoint,
            "updated_at": internal_adapter_payload(committed)["updated_at"],
        }

    async def _unregister_safe(
        base_model: str,
        adapter_id: str,
        expected_generation: str | None,
    ) -> None:
        # gpu cleanup is best-effort and may cold-start a scaled-to-zero engine. the engine compares
        # this deployment generation under its per-adapter lock so stale cleanup cannot remove a
        # redeployment of the same immutable revision id.
        with contextlib.suppress(Exception):
            await pool.unregister(base_model, adapter_id, expected_generation)

    @api.delete("/adapters/{adapter_id}", tags=["adapters"])
    async def remove_adapter(
        adapter_id: str,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        _assert_internal(request)
        if reload_records is not None:
            await _reload()

        async def _get_authoritative(adapter_id: str) -> AdapterRecord | None:
            try:
                return await _get_stored(adapter_id)
            except PersistenceRecordError as exc:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "adapter storage is unavailable",
                ) from exc

        record = router.get(adapter_id)
        if record is None:
            record = await _get_authoritative(adapter_id)
        if record is not None and record.status == "ready" and record.serve_base_model:
            base_model = record.base_model
            router.remove(adapter_id)
            # return after durable routing cleanup; gpu eviction continues after the response.
            background_tasks.add_task(
                _unregister_safe,
                base_model,
                adapter_id,
                record.deployment_generation,
            )
            return {
                "ok": True,
                "removed": adapter_id,
                "base_model": base_model,
                "run_id": adapter_id,
                "disabled_aliases": [],
                "disabled_revisions": [],
            }

        if (
            record is not None
            and (record.is_alias or record.is_revision)
            and record.run_id is not None
        ):
            run_id = record.run_id
        elif record is None and "@" not in adapter_id:
            run_id = adapter_id
        else:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")
        matches = [
            candidate
            for candidate in router.ready_records()
            if (candidate.is_alias and candidate.adapter_id == run_id)
            or (candidate.is_revision and candidate.run_id == run_id)
        ]
        if not matches:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")

        # phase 1: compare-and-swap every matched row to "disabled" in persistence first, collecting
        # the rows that durably converged.
        disabled_aliases, disabled_revisions, stuck_ready, pending_teardown = await disable_matched(
            matches, get_authoritative=_get_authoritative
        )

        # phase 2: remove every durably disabled row from routing immediately. gpu eviction is
        # deferred until after either the success or conflict response, so a scaled-to-zero
        # engine's cold start cannot make undeploy callers time out.
        for cleanup_record, expected_generation in apply_teardown(router, pending_teardown):
            background_tasks.add_task(
                _unregister_safe,
                cleanup_record.base_model,
                cleanup_record.adapter_id,
                expected_generation,
            )

        if stuck_ready:
            # some rows are durably disabled while others could not converge. return the same detail
            # shape as an httpexception while allowing cleanup to run after the conflict response.
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "detail": {
                        "error": "adapter changed concurrently",
                        "run_id": run_id,
                        "disabled_aliases": sorted(disabled_aliases),
                        "disabled_revisions": sorted(disabled_revisions),
                        "stuck": sorted(stuck_ready),
                    }
                },
                background=background_tasks,
            )

        return {
            "ok": True,
            "removed": adapter_id,
            "base_model": matches[0].base_model,
            "run_id": run_id,
            "disabled_aliases": sorted(disabled_aliases),
            "disabled_revisions": sorted(disabled_revisions),
        }

    def _schedule_usage(
        record: AdapterRecord,
        result: dict[str, Any],
        caller_org: str | None,
    ) -> None:
        _usage.schedule(record, result, caller_org)

    async def _generate(
        payload: Any,
        requested: AdapterRecord,
        target: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
        caller_org: str | None = None,
    ) -> dict[str, Any]:
        engine_payload = payload.model_copy(update={"adapter_id": target.adapter_id})
        try:
            result = await pool.generate(
                target.base_model,
                engine_payload,
                target,
                expected_checkpoint=expected_checkpoint,
            )
        except Exception as exc:
            _raise_if_engine_error(requested.adapter_id, exc)
        if "adapter_id" in result:
            result = {**result, "adapter_id": requested.adapter_id}
        _schedule_usage(requested, result, caller_org)
        return result

    def _openai_chat_stream(
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
            router,
            _schedule_usage,
            record=record,
            events=events,
            adapter_id=adapter_id,
            completion_id=completion_id,
            created=created,
            include_usage=include_usage,
            caller_org=caller_org,
            thinking=thinking,
        )

    async def _prepare_stream(
        payload: Any,
        requested: AdapterRecord,
        target: AdapterRecord,
        *,
        expected_checkpoint: str | None,
    ) -> tuple[AsyncIterator[dict[str, Any]], dict[str, str], bool]:
        return await prepare_stream(
            pool, router, payload, requested, target, expected_checkpoint=expected_checkpoint
        )

    @api.post("/generate", tags=["inference"])
    async def generate(payload: GenerateRequest, request: Request) -> JSONResponse:
        caller_org = await _authorize_inference(request, payload.adapter_id)
        requested, target = await _lookup(payload.adapter_id)
        await _prepare_generate_request(payload, target)
        result = await _generate(
            payload,
            requested,
            target,
            expected_checkpoint=_expected_checkpoint(request),
            caller_org=caller_org,
        )
        return _inference_json_response(result, target)

    @api.post("/adapters/{adapter_id}/generate", tags=["inference"])
    async def generate_for_adapter(
        adapter_id: str, payload: dict[str, Any], request: Request
    ) -> JSONResponse:
        # Parse first so the GenerateRequest validator normalizes (strips) the adapter id, then
        # authorize and route against that same normalized value (not the raw path parameter).
        req = _parse_generate({**payload, "adapter_id": adapter_id})
        caller_org = await _authorize_inference(request, req.adapter_id)
        requested, target = await _lookup(req.adapter_id)
        await _prepare_generate_request(req, target)
        result = await _generate(
            req,
            requested,
            target,
            expected_checkpoint=_expected_checkpoint(request),
            caller_org=caller_org,
        )
        return _inference_json_response(result, target)

    @api.post("/v1/chat/completions", tags=["openai"])
    async def chat_completions(payload: dict[str, Any], request: Request) -> Any:
        # time/uuid come from the build_serving_app closure (imported once at app construction);
        # no per-request re-import.
        adapter_id = payload.get("model")
        if not isinstance(adapter_id, str) or not adapter_id.strip():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "model must be the adapter id")
        # use the stripped id consistently for validation, auth, routing, and the echoed response
        # model, so a caller that sends "  qa  " is authorized against and routed to "qa".
        adapter_id = adapter_id.strip()
        match = _FLASH_CHECKPOINT_MODEL_RE.fullmatch(adapter_id)
        if match is not None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "This is a checkpoint identifier, not a serving model identifier. "
                f"Deploy it first or use model {match.group('run_id')}.",
            )
        caller_org = await _authorize_inference(request, adapter_id)
        requested, target = await _lookup(adapter_id)
        try:
            fields = openai_generate_fields(payload, adapter_id)
        except StructuredOutputsError as exc:
            # Malformed OpenAI response_format (json_schema with no schema, or an unknown type) ->
            # 422, not 500.
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        req = _parse_generate(fields)
        await _prepare_generate_request(req, target)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_usage = openai_include_usage(payload)
        if payload.get("stream") is True:
            events, checkpoint_headers, thinking = await _prepare_stream(
                req,
                requested,
                target,
                expected_checkpoint=_expected_checkpoint(request),
            )
            return StreamingResponse(
                _openai_chat_stream(
                    record=requested,
                    events=events,
                    adapter_id=adapter_id,
                    completion_id=completion_id,
                    created=created,
                    include_usage=include_usage,
                    caller_org=caller_org,
                    thinking=thinking,
                ),
                media_type="text/event-stream",
                # Disable proxy and CDN buffering so each SSE chunk reaches the client
                # immediately. Without X-Accel-Buffering, Nginx accumulates tokens until its
                # output buffer fills, adding 100+ ms of hidden TTFT for small completions.
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    **checkpoint_headers,
                },
            )

        generation = await _generate(
            req,
            requested,
            target,
            expected_checkpoint=_expected_checkpoint(request),
            caller_org=caller_org,
        )
        active_checkpoint = generation.get("checkpoint")
        provenance = _revision_provenance(target, active_checkpoint)
        response = openai_chat_completion(
            completion_id=completion_id,
            created=created,
            adapter_id=adapter_id,
            generation=generation,
            provenance=provenance,
        )
        return JSONResponse(response, headers=_provenance_headers(provenance, active_checkpoint))

    return api
