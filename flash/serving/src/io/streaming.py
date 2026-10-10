"""Low-level hosted engine dispatch and OpenAI stream compatibility exports."""

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from flash.serving.src.engine.errors import raise_if_engine_error
from flash.serving.src.http.routing import AdapterRouter, EnginePool
from flash.serving.src.io.openai_stream import (  # noqa: F401
    _assistant_role_chunk,
    _await_producer_shutdown,
    _close_async_iterator,
    _close_stream_sources,
    _next_event_or_disconnect,
    _produce_openai_chat_stream,
    _replay_first_event,
    _sse,
    _StreamOutput,
    openai_chat_stream,
)
from flash.serving.src.io.provenance import (
    _active_checkpoint_ref,
    _checkpoint_provenance,
    _provenance_headers,
    require_attested_checkpoint,
)
from flash.serving.src.io.schemas import AdapterRecord


async def prepare_stream(
    pool: EnginePool,
    router: AdapterRouter,
    payload: Any,
    requested: AdapterRecord,
    target: AdapterRecord,
    *,
    generation_id: str,
    require_generation_id: bool,
    expected_checkpoint: str | None,
) -> tuple[AsyncIterator[dict[str, Any]], dict[str, str], bool, dict[str, Any]]:
    engine_payload = payload.model_copy(
        update={"adapter_id": target.adapter_id, "generation_id": generation_id}
    )
    events: AsyncIterator[dict[str, Any]] | None = None
    try:
        # construction is inside the try with the first advance: ``EnginePool.stream_generate``
        # is declared as an ordinary method returning an AsyncIterator, so a conforming pool may
        # raise while building the iterator rather than on first advance. The current Modal pool
        # is an async generator (whose body is deferred to ``anext``), but the protocol does not
        # require that, and a dispatch failure must map identically either way.
        events = pool.stream_generate(
            target.base_model,
            engine_payload,
            target,
            expected_checkpoint=expected_checkpoint,
        )
        first = await anext(events)
        require_attested_checkpoint(first, target)
    except BaseException as exc:
        # cancellation while waiting for the first engine event must still enter the pool iterator's
        # finally block, which aborts the remote generation. ordinary dispatch failures need the same
        # release before their existing http error mapping runs.
        if events is not None:
            with contextlib.suppress(Exception):
                await _close_async_iterator(events)
        if isinstance(exc, Exception):
            raise_if_engine_error(router, requested.adapter_id, exc)
        raise
    try:
        if require_generation_id and first.get("request_id") != generation_id:
            raise RuntimeError("serving engine returned a mismatched generation id")
        if first.get("type") == "ready":
            active_checkpoint = first.get("checkpoint")
            provenance = _checkpoint_provenance(target, active_checkpoint)
            headers = _provenance_headers(provenance, active_checkpoint)
            if target.is_checkpoint:
                headers["X-Freesolo-LoRA-Request-Adapter"] = first["lora_request_adapter"]
            return (
                _replay_first_event(first, events),
                headers,
                bool(first.get("thinking")),
                first,
            )

        active_checkpoint = _active_checkpoint_ref(target)
        provenance = _checkpoint_provenance(target, active_checkpoint)
        headers = _provenance_headers(provenance, active_checkpoint)
        if target.is_checkpoint:
            headers["X-Freesolo-LoRA-Request-Adapter"] = first["lora_request_adapter"]
        return (
            _replay_first_event(first, events),
            headers,
            False,
            first,
        )
    except BaseException:
        await _close_async_iterator(events)
        raise


async def generate_once(
    pool: EnginePool,
    router: AdapterRouter,
    payload: Any,
    requested: AdapterRecord,
    target: AdapterRecord,
    *,
    generation_id: str,
    require_generation_id: bool,
    expected_checkpoint: str | None = None,
) -> dict[str, Any]:
    """dispatch one non-streaming generation and echo the authorized checkpoint id."""
    engine_payload = payload.model_copy(
        update={"adapter_id": target.adapter_id, "generation_id": generation_id}
    )
    try:
        result = await pool.generate(
            target.base_model,
            engine_payload,
            target,
            expected_checkpoint=expected_checkpoint,
        )
    except Exception as exc:
        raise_if_engine_error(router, requested.adapter_id, exc)
    require_attested_checkpoint(result, target)
    if require_generation_id and result.get("request_id") != generation_id:
        raise RuntimeError("serving engine returned a mismatched generation id")
    if "adapter_id" in result:
        result = {**result, "adapter_id": requested.adapter_id}
    return result
