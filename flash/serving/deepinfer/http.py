"""Private native-event transport for the existing hosted serving router."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from flash.serve.contract.protocol import MAX_CHAT_REQUEST_BYTES
from flash.serving.deepinfer.config import MODEL_ID, MODEL_REVISION, RUNTIME_REVISION
from flash.serving.src.http.body_limit import RequestBodyLimitMiddleware
from flash.serving.src.http.stream_response import ClosingStreamingResponse, await_until_disconnect


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
        first = await await_until_disconnect(request, anext(events))
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
            "loaded_ranks": getattr(runtime, "provenance", {}),
            "active_requests": len(getattr(runtime, "pending", {})),
        }

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
