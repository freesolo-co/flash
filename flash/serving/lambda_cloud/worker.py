"""Private TLS worker reusing the hosted LoRA engine without changing model arguments."""

import asyncio
import contextlib
import hmac
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from flash.serving.app.timeouts import REQUEST_TIMEOUT_SECONDS
from flash.serving.src.engine.lora_engine import _LoraEngineImpl
from flash.serving.src.engine.model_config import engine_overrides_for

from .config import PORT, placement
from .leases import GenerationLeases


class LambdaWorker(_LoraEngineImpl):
    async def _load(self):
        import torch

        _, expected = placement(self.base_model)
        if torch.cuda.device_count() != 1 or expected not in torch.cuda.get_device_name(0):
            raise RuntimeError("Lambda worker received an unexpected GPU shape")
        if (
            torch.cuda.get_device_properties(0).total_memory
            < (75 if expected == "H100" else 170) * 2**30
        ):
            raise RuntimeError("Lambda worker received a partitioned or undersized GPU")
        await super()._load()

    def _self_heal_if_dead(self, reason):
        if self._engine_dead():
            # Docker restarts the same worker; other models and replicas are isolated.
            os._exit(1)


def create_app(engine, token):
    if not token:
        raise ValueError("worker bearer token is required")
    slots = asyncio.Semaphore(int(engine_overrides_for(engine.base_model).get("max_num_seqs", 64)))
    active = set()
    leases = GenerationLeases(
        2 * int(engine_overrides_for(engine.base_model).get("max_num_seqs", 64))
    )

    @asynccontextmanager
    async def lifespan(app):
        await engine._load()
        try:
            yield
        finally:
            await engine._shutdown()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    app.state.generation_leases = leases

    @app.middleware("http")
    async def authenticate(request, call_next):
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def validation_error(request, exc):
        return JSONResponse({"kind": "value_error", "message": str(exc)}, status_code=400)

    @app.exception_handler(TimeoutError)
    async def timeout_error(request, exc):
        return JSONResponse({"kind": "timeout"}, status_code=504)

    @app.get("/health")
    async def health():
        return {**engine._health(), "deployment_sha": os.environ.get("FREESOLO_DEPLOYMENT_SHA", "")}

    @app.post("/register")
    async def register(request: Request):
        body = await request.json()
        return await engine._register(body["record"], body.get("deployment_generation"))

    @app.post("/unregister")
    async def unregister(request: Request):
        body = await request.json()
        return await engine._unregister(
            body["org_id"], body["model_id"], body.get("expected_generation")
        )

    @app.post("/lease")
    async def renew_lease(request: Request):
        body = await request.json()
        leases.renew(body.get("generation_id"))
        return {"ok": True}

    @app.post("/abort")
    async def abort(request: Request):
        body = await request.json()
        if body["generation_id"] in active:
            await engine.engine.abort(body["generation_id"])
        return {"ok": True}

    @asynccontextmanager
    async def admission(body):
        generation_id = body.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id:
            raise ValueError("generation id is required")
        async with slots:
            if generation_id in active:
                raise ValueError("generation id is already running")
            await leases.check(generation_id)
            active.add(generation_id)
            watcher = asyncio.create_task(
                leases.watch(generation_id, asyncio.current_task(), engine.engine.abort)
            )
            try:
                async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                    yield (
                        body["payload"],
                        body.get("record"),
                        body.get("expected_checkpoint"),
                        generation_id,
                    )
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                active.discard(generation_id)
                leases.finish(generation_id)

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        async with admission(body) as args:
            task = asyncio.create_task(engine._generate(*args))
            try:
                while not task.done():
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    await asyncio.wait({task}, timeout=0.25)
                return await task
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    @app.post("/stream")
    async def stream(request: Request):
        body = await request.json()

        async def events():
            try:
                async with admission(body) as args:

                    async def check_lease():
                        await leases.check(args[-1])

                    generator = engine._stream_generate(*args, pre_generate_check=check_lease)
                    try:
                        async for event in generator:
                            yield json.dumps(event, allow_nan=False) + "\n"
                    finally:
                        await generator.aclose()
            except ValueError as exc:
                yield json.dumps({"lambda_error": "value_error", "message": str(exc)}) + "\n"
            except TimeoutError:
                yield json.dumps({"lambda_error": "timeout"}) + "\n"
            except Exception:
                yield json.dumps({"lambda_error": "engine_error"}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    return app


def main():
    import uvicorn

    engine = LambdaWorker()
    engine.base_model = os.environ["FLASH_LAMBDA_MODEL"]
    _, engine.pinned_gpu = placement(engine.base_model)
    app = create_app(engine, os.environ["FLASH_LAMBDA_TOKEN"])
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        ssl_keyfile="/run/flash/key.pem",
        ssl_certfile="/run/flash/cert.pem",
        access_log=False,
        timeout_graceful_shutdown=REQUEST_TIMEOUT_SECONDS,
    )


if __name__ == "__main__":
    main()
