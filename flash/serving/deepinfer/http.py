"""Private Modal transport for the hosted router and direct qualification probes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from flash.serve.contract.protocol import MAX_CHAT_REQUEST_BYTES
from flash.serving.deepinfer.config import MODEL_ID, MODEL_REVISION, RUNTIME_REVISION
from flash.serving.deepinfer.protocol import parse_request
from flash.serving.deepinfer.render import Decoder
from flash.serving.src.http.body_limit import RequestBodyLimitMiddleware
from flash.serving.src.http.stream_response import ClosingStreamingResponse


def _sse(value):
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


async def _watch_disconnect(request, operation):
    task = asyncio.create_task(operation)
    try:
        while not task.done():
            if await request.is_disconnected():
                raise asyncio.CancelledError
            await asyncio.wait({task}, timeout=0.1)
        return await task
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _collect(events, decoder, first):
    current = first
    while True:
        decoder.consume(current)
        if reason := current.get("finish_reason"):
            return reason
        current = await anext(events)


async def _stream(events, decoder, first, envelope, include_usage):
    def chunk(delta, reason=None):
        return {
            **envelope,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
        }

    try:
        yield _sse(chunk({"role": "assistant", "content": ""}))
        current = first
        while True:
            for delta in decoder.consume(current):
                yield _sse(chunk(delta))
            if reason := current.get("finish_reason"):
                yield _sse(chunk({}, reason))
                break
            current = await anext(events)
        if include_usage:
            yield _sse(
                {
                    **envelope,
                    "object": "chat.completion.chunk",
                    "choices": [],
                    "usage": decoder.usage(),
                }
            )
        yield "data: [DONE]\n\n"
    except (RuntimeError, TimeoutError, StopAsyncIteration):
        # An interrupted sequence must never look like a successful stop/length result.
        yield _sse({"error": {"message": "DeepInfer generation failed", "type": "server_error"}})
    finally:
        await events.aclose()


async def _chat(runtime, body, request):
    try:
        parsed = parse_request(body)
        tokens = await asyncio.to_thread(runtime.tokenize, parsed)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    decoder = Decoder(runtime.tokenizer, thinking=parsed.thinking, prompt_tokens=len(tokens))
    events = runtime.generate(parsed, tokens)
    envelope = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "created": int(time.time()),
        "model": MODEL_ID,
    }
    handed_off = False
    try:
        first = await _watch_disconnect(request, anext(events))
        if parsed.stream:
            response = ClosingStreamingResponse(
                _stream(events, decoder, first, envelope, parsed.include_usage),
                on_unstarted=events.aclose,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
            handed_off = True
            return response
        reason = await _watch_disconnect(request, _collect(events, decoder, first))
        message = {"role": "assistant", "content": decoder.text["content"]}
        if parsed.thinking:
            message["reasoning_content"] = decoder.text["reasoning_content"]
        return {
            **envelope,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            "usage": decoder.usage(),
        }
    except OverflowError as exc:
        raise HTTPException(
            429, "DeepInfer replica is at capacity", headers={"Retry-After": "1"}
        ) from exc
    except (RuntimeError, TimeoutError, StopAsyncIteration) as exc:
        raise HTTPException(503, "DeepInfer replica is unavailable") from exc
    finally:
        if not handed_off:
            await events.aclose()


async def _monitor(runtime):
    while True:
        await asyncio.sleep(1)
        try:
            runtime.assert_healthy()
        except RuntimeError:
            # The Modal launcher observes server exit and retires this entire failed EP group.
            os.kill(os.getpid(), signal.SIGTERM)
            return


async def _native_stream(events, first):
    try:
        yield json.dumps(first) + "\n"
        async for event in events:
            yield json.dumps(event) + "\n"
    except (RuntimeError, TimeoutError, StopAsyncIteration):
        yield json.dumps({"type": "error"}) + "\n"
    finally:
        await events.aclose()


async def _native_chat(runtime, body, request):
    from flash.serving.deepinfer.hosted import prepare

    events = None
    handed_off = False
    try:
        events = await prepare(runtime, body)
        first = await _watch_disconnect(request, anext(events))
        response = ClosingStreamingResponse(
            _native_stream(events, first),
            on_unstarted=events.aclose,
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        handed_off = True
        return response
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(429, "DeepInfer replica is at capacity") from exc
    except (RuntimeError, TimeoutError, StopAsyncIteration) as exc:
        raise HTTPException(503, "DeepInfer replica is unavailable") from exc
    finally:
        if events is not None and not handed_off:
            await events.aclose()


def build_app(runtime, *, manage_lifecycle=True):
    @asynccontextmanager
    async def lifespan(app):
        monitor = None
        try:
            if manage_lifecycle:
                await runtime.start()
                monitor = asyncio.create_task(_monitor(runtime))
            yield
        finally:
            if monitor is not None:
                monitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor
            if manage_lifecycle:
                await runtime.close()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_CHAT_REQUEST_BYTES)

    @app.get("/health")
    async def health():
        try:
            runtime.assert_healthy()
        except RuntimeError as exc:
            raise HTTPException(503, "DeepInfer replica is unavailable") from exc
        return {
            "status": "ok",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_revision": RUNTIME_REVISION,
            "gpu": "B200",
            "gpu_count": 4,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "deepseek-ai"}],
        }

    @app.post("/v1/chat/completions")
    async def chat(body: dict, request: Request):
        return await _chat(runtime, body, request)

    @app.post("/internal/generate")
    async def native_chat(body: dict, request: Request):
        return await _native_chat(runtime, body, request)

    return app


def main():
    import uvicorn

    from flash.serving.deepinfer.config import PORT
    from flash.serving.deepinfer.runtime import Runtime

    uvicorn.run(build_app(Runtime()), host="0.0.0.0", port=PORT, timeout_graceful_shutdown=10)


if __name__ == "__main__":
    main()
