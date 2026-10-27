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
from flash.serving.deepinfer.contract.config import (
    GPU_COUNT,
    MAX_INPUTS,
    MODEL_ID,
    MODEL_REVISION,
    REQUEST_SECONDS,
    RUNTIME_REVISION,
)
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


async def _native_stream(events, first, *, sse=False):
    def frame(event):
        data = json.dumps(event)
        return f"data: {data}\n\n" if sse else data + "\n"

    try:
        yield frame(first)
        async for event in events:
            yield frame(event)
    except (RuntimeError, TimeoutError, StopAsyncIteration):
        yield frame({"type": "error"})
    finally:
        await events.aclose()


async def _native_chat(runtime, body, request):
    from flash.serving.deepinfer.app.events import prepare

    events = None
    handed_off = False
    try:
        events = await prepare(runtime, body)
        first = await await_until_disconnect(request, anext(events))
        # Modal flushes SSE immediately; NDJSON can be buffered by its proxy.
        # Negotiate so an older router can still consume this backend during rollout.
        sse = "text/event-stream" in request.headers.get("accept", "")
        response = ClosingStreamingResponse(
            _native_stream(events, first, sse=sse),
            on_unstarted=events.aclose,
            media_type="text/event-stream" if sse else "application/x-ndjson",
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


class _GenerationCount:
    """Count the complete HTTP lifetime, including tokenization and stream draining."""

    def __init__(self, app, counts):
        self.app = app
        self.counts = counts

    async def __call__(self, scope, receive, send):
        generation = (
            scope["type"] == "http"
            and scope.get("path") == "/internal/generate"
            and scope.get("method") == "POST"
        )
        if generation:
            if self.counts["active"] >= MAX_INPUTS:
                from starlette.responses import JSONResponse

                await JSONResponse(
                    {"detail": "DeepInfer replica is not accepting new requests"},
                    status_code=429,
                    headers={"X-DeepInfer-Retryable": "not-started"},
                )(scope, receive, send)
                return
            self.counts["active"] += 1
        try:
            if generation:
                await _bounded_request(self.app, scope, receive, send)
            else:
                await self.app(scope, receive, send)
        finally:
            if generation:
                self.counts["active"] -= 1


async def _bounded_request(app, scope, receive, send):
    """One deadline covers upload, preparation, generation, and stream delivery."""
    started = completed = sse = False

    async def track(message):
        nonlocal started, completed, sse
        if message["type"] == "http.response.start":
            started = True
            sse = b"text/event-stream" in dict(message.get("headers", [])).get(b"content-type", b"")
        await send(message)
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            completed = True

    deadline = asyncio.timeout(REQUEST_SECONDS)
    try:
        async with deadline:
            await app(scope, receive, track)
    except TimeoutError:
        if not deadline.expired():
            raise
        # Cancellation unwinds the stream and its bounded native abort handshake
        # before responding. Never mark a timeout as safe to replay.
        with contextlib.suppress(TimeoutError, OSError):
            async with asyncio.timeout(5):
                if not started:
                    from starlette.responses import JSONResponse

                    await JSONResponse(
                        {"detail": "DeepInfer request deadline exceeded"}, status_code=504
                    )(scope, receive, send)
                elif not completed:
                    error = b'data: {"type":"error"}\n\n' if sse else b'{"type":"error"}\n'
                    await send({"type": "http.response.body", "body": error, "more_body": False})


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
    counts = {"active": 0}
    app.add_middleware(_GenerationCount, counts=counts)
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_CHAT_REQUEST_BYTES)

    @app.get("/health")
    async def health():
        try:
            runtime.assert_healthy()
        except RuntimeError as exc:
            raise HTTPException(503, "DeepInfer replica is unavailable") from exc
        return {
            "status": "ok",
            "deployment_id": os.environ.get("DEEPINFER_ROLLOUT_ID", ""),
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_revision": RUNTIME_REVISION,
            "gpu": "B200",
            "gpu_count": GPU_COUNT,
            "loaded_ranks": getattr(runtime, "provenance", {}),
            "active_requests": len(getattr(runtime, "pending", {})),
            "active_http_requests": counts["active"],
            "request_timeout_seconds": REQUEST_SECONDS,
            "compiled_kernel_cache_verified": getattr(
                runtime, "compiled_kernel_cache_verified", False
            ),
        }

    @app.post("/internal/generate")
    async def native_chat(body: dict, request: Request):
        return await _native_chat(runtime, body, request)

    return app


def main():
    import uvicorn

    from flash.serving.deepinfer.contract.config import PORT
    from flash.serving.deepinfer.runtime.bridge import Runtime

    # Uvicorn binds the port only AFTER lifespan startup (including every warmup)
    # completes. Modal uses that listening port as its rollout readiness gate.
    uvicorn.run(build_app(Runtime()), host="0.0.0.0", port=PORT, timeout_graceful_shutdown=10)


if __name__ == "__main__":
    main()
