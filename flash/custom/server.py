"""Authenticated, single-session worker. Run one process per dedicated GPU."""

import asyncio
import hmac
import json
import os
import time

from flash._internal.diagnostics import sanitize_diagnostic
from flash.custom.session import SessionConflict, TrainingSession
from flash.custom.types import checkpoint_name

MAX_BODY_BYTES = 4 * 1024 * 1024


def create_app(session, token):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
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
                        raise HTTPException(413, "custom operation body exceeds 4 MiB")
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
        try:
            checkpoint_name(name)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        path = session.backend.checkpoint_dir / f"{name}.zip"
        if not path.is_file():
            raise HTTPException(404, "unknown checkpoint")
        session.last_activity = time.monotonic()

        async def content():
            session.active_exports += 1
            try:
                with path.open("rb") as handle:
                    while chunk := await run_in_threadpool(handle.read, 1024 * 1024):
                        yield chunk
            finally:
                session.active_exports -= 1
                session.last_activity = time.monotonic()

        return StreamingResponse(
            content(),
            media_type="application/zip",
            headers={
                "Content-Length": str(path.stat().st_size),
                "Content-Disposition": f'attachment; filename="{name}.zip"',
            },
        )

    return app


def main():
    import uvicorn

    from flash.custom.backend import load_backend

    token = os.environ["FLASH_CUSTOM_TOKEN"]
    config = json.loads(os.environ["FLASH_CUSTOM_CONFIG"])
    backend = load_backend(
        config["model"],
        revision=config.get("revision"),
        lora_rank=config["lora_rank"],
        checkpoint_dir="/tmp/flash-custom-checkpoints",
        max_tokens=config["max_context_tokens"],
    )
    session = TrainingSession(
        backend, max_steps=config["max_steps"], idle_seconds=config["idle_seconds"]
    )
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
                if session.expired() or time.monotonic() - started >= config["max_wall_seconds"]:
                    server.should_exit = True

        task = asyncio.create_task(watchdog())
        try:
            await server.serve()
        finally:
            task.cancel()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
