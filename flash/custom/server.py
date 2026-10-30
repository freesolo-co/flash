"""Authenticated, single-session worker. Run one process per dedicated GPU."""

import asyncio
import hmac
import json
import time

from flash._internal.diagnostics import sanitize_diagnostic
from flash.custom.session import SessionConflict
from flash.custom.types import MAX_BODY_BYTES, checkpoint_name


def create_app(session, token):
    from fastapi import FastAPI, HTTPException, Request
    from starlette.concurrency import run_in_threadpool

    if not isinstance(token, str) or len(token) < 32:
        raise ValueError("custom session token must contain at least 32 characters")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        from fastapi.responses import JSONResponse

        expected = f"Bearer {token}".encode()
        supplied = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(supplied, expected):
            return JSONResponse({"detail": "invalid session token"}, status_code=401)
        return await call_next(request)

    @app.get("/v1/custom/status")
    def status():
        with session.lock:
            return session.status()

    @app.get("/v1/custom/logs")
    def logs(after: int = 0):
        with session.lock:
            events = [event for event in session.events if event["cursor"] > after]
            earliest = session.events[0]["cursor"] if session.events else 1
            return {
                "events": events,
                "cursor": session.event_cursor,
                "truncated": after < earliest - 1,
            }

    @app.post("/v1/custom/operation")
    async def operation(request: Request):
        raw = bytearray()
        try:
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    raw.extend(chunk)
                    if len(raw) > MAX_BODY_BYTES:
                        raise HTTPException(
                            413,
                            {
                                "code": "custom_rejected",
                                "message": "custom operation body exceeds 4 MiB",
                            },
                        )
            payload = json.loads(raw)
        except (ValueError, TimeoutError) as error:
            raise HTTPException(
                400, {"code": "custom_rejected", "message": "invalid JSON body"}
            ) from error
        try:
            return await run_in_threadpool(session.execute, payload)
        except (SessionConflict, ValueError) as error:
            if session.failed:
                raise HTTPException(
                    500,
                    {
                        "code": "custom_session_failed",
                        "message": sanitize_diagnostic(error, limit=2000),
                        "session_id": session.id,
                    },
                ) from error
            raise HTTPException(
                409 if isinstance(error, SessionConflict) else 400,
                {"code": "custom_rejected", "message": str(error)},
            ) from error
        except Exception as error:
            raise HTTPException(
                500,
                {
                    "code": "custom_session_failed",
                    "message": sanitize_diagnostic(error, limit=2000),
                    "session_id": session.id,
                },
            ) from error

    @app.get("/v1/custom/checkpoints/{name}")
    def download(name: str):
        from flash.custom.exports import checkpoint_response

        try:
            checkpoint_name(name)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return checkpoint_response(session, name)

    return app


def serve_session(session, token, max_wall_seconds, *, tick=None):
    import uvicorn

    app = create_app(session, token)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            access_log=False,
            timeout_graceful_shutdown=5,
        )
    )

    async def serve():
        started = time.monotonic()

        async def watchdog():
            while not server.should_exit:
                await asyncio.sleep(5)
                if tick is not None:
                    await asyncio.to_thread(tick)
                if session.expired() or time.monotonic() - started >= max_wall_seconds:
                    server.should_exit = True

        task = asyncio.create_task(watchdog())
        try:
            await server.serve()
        finally:
            task.cancel()

    asyncio.run(serve())
