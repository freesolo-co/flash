"""OpenAI-shaped SSE rendering for streamed chat completions.

Split out of router.py's app builder. The engine pool, the adapter router and the usage reporter
are passed in rather than captured, so the stream can be rendered against a fake pool without
building the app.
"""

from collections.abc import AsyncIterator, Callable
from typing import Any

from flash.serving.src.engine_errors import raise_if_engine_error, terminating_on_engine_error
from flash.serving.src.responses import _ReasoningStreamSplitter, _usage_block
from flash.serving.src.routing import AdapterRouter, EnginePool
from flash.serving.src.schemas import AdapterRecord
from flash.serving.src.serving_io import (
    _active_checkpoint_ref,
    _provenance_headers,
    _revision_provenance,
    _sse,
    require_attested_revision,
)


async def openai_chat_stream(
    router: AdapterRouter,
    schedule_usage: Callable[[AdapterRecord, dict[str, Any], str | None], None],
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
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": adapter_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
    )

    def _delta_chunk(delta: dict[str, Any]) -> bytes:
        return _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": adapter_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )

    splitter = _ReasoningStreamSplitter(thinking)
    final: dict[str, Any] = {}
    async for event in terminating_on_engine_error(router, events, adapter_id):
        kind = event.get("type")
        if kind == "delta":
            text = event.get("text") or ""
            if not text:
                continue
            reasoning_delta, content_delta = splitter.feed(text)
            if reasoning_delta:
                yield _delta_chunk({"reasoning_content": reasoning_delta})
            if content_delta:
                yield _delta_chunk({"content": content_delta})
        elif kind == "final":
            final = event
        elif kind == "error":
            # The 200 and its headers went out with the first chunk, so the status can no
            # longer carry the failure. Without this the failure would propagate and Starlette
            # would drop the connection: the caller receives a well-formed but SILENTLY
            # TRUNCATED stream with no error and no [DONE], indistinguishable from a short
            # completion. Emit the error into the stream and then close the protocol normally
            # -- the same shape vLLM's own OpenAI server uses -- so the failure is detectable
            # by an unmodified OpenAI client.
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": adapter_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                    "error": {
                        "message": event["message"],
                        "type": "engine_error",
                        "code": event["code"],
                    },
                }
            )
            yield _sse("[DONE]")
            return
    trailing = splitter.flush()
    if trailing:
        yield _delta_chunk({"reasoning_content": trailing})

    schedule_usage(record, final, caller_org)
    done_chunk: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": adapter_id,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": final.get("finish_reason"),
            }
        ],
    }
    if include_usage:
        prompt_tokens = final.get("prompt_tokens")
        completion_tokens = final.get("completion_tokens")
        if prompt_tokens is not None and completion_tokens is not None:
            done_chunk["usage"] = _usage_block(
                int(prompt_tokens), int(completion_tokens), final.get("cached_tokens")
            )
    yield _sse(done_chunk)
    yield _sse("[DONE]")


async def prepare_stream(
    pool: EnginePool,
    router: AdapterRouter,
    payload: Any,
    requested: AdapterRecord,
    target: AdapterRecord,
    *,
    expected_checkpoint: str | None,
) -> tuple[AsyncIterator[dict[str, Any]], dict[str, str], bool]:
    engine_payload = payload.model_copy(update={"adapter_id": target.adapter_id})
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
    except Exception as exc:
        raise_if_engine_error(router, requested.adapter_id, exc)
    if first.get("type") == "ready":
        active_checkpoint = first.get("checkpoint")
        provenance = _revision_provenance(target, active_checkpoint)
        # the ready event carries the rendered thinking mode; it precedes every delta, so the
        # openai layer can route the first chunk correctly.
        return (
            events,
            _provenance_headers(provenance, active_checkpoint),
            bool(first.get("thinking")),
        )

    async def replay() -> AsyncIterator[dict[str, Any]]:
        yield first
        async for event in events:
            yield event

    active_checkpoint = _active_checkpoint_ref(target)
    provenance = _revision_provenance(target, active_checkpoint)
    # no ready event, so the rendered mode is unknown. report it as non-thinking rather than
    # guessing from ``target.thinking``: a base-model serve honors a caller enable_thinking
    # override, so the record can disagree with what was actually rendered, and splitting a
    # non-thinking completion that merely quotes </think> would tear the answer in half.
    return replay(), _provenance_headers(provenance, active_checkpoint), False


async def generate_once(
    pool: EnginePool,
    router: AdapterRouter,
    schedule_usage: Callable[[AdapterRecord, dict[str, Any], str | None], None],
    payload: Any,
    requested: AdapterRecord,
    target: AdapterRecord,
    *,
    expected_checkpoint: str | None = None,
    caller_org: str | None = None,
) -> dict[str, Any]:
    """Dispatch one non-streaming generation and meter it.

    The result echoes the REQUESTED adapter id rather than the resolved target's, so an alias
    caller sees the id it asked for instead of the revision behind it.
    """
    engine_payload = payload.model_copy(update={"adapter_id": target.adapter_id})
    try:
        result = await pool.generate(
            target.base_model,
            engine_payload,
            target,
            expected_checkpoint=expected_checkpoint,
        )
    except Exception as exc:
        raise_if_engine_error(router, requested.adapter_id, exc)
    # attest before metering, not after: `schedule_usage` is what bills the caller, so a
    # generation the engine never attested to the resolved immutable adapter must not reach it.
    # this also covers every non-streaming route at once -- the plain `/generate` paths would
    # otherwise serve an unattested adapter with no check at all.
    require_attested_revision(result, target)
    if "adapter_id" in result:
        result = {**result, "adapter_id": requested.adapter_id}
    schedule_usage(requested, result, caller_org)
    return result
