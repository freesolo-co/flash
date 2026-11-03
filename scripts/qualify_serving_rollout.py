"""CPU-only Modal handoff qualification; never targets the serving application.

Run with ``python -m scripts.qualify_serving_rollout`` in a Modal-authenticated
environment. It migrates a legacy engine under traffic, then changes concurrency
under the stable model identity. Each replacement takes ten seconds to start.
The temporary app is stopped in finally; request results contain no user data.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import modal

from flash.serving.app.rollout import warm_active_models

APP_NAME = os.environ.get("ROLLOUT_QUALIFICATION_APP", "flash-rollout-qualification-local")
if not APP_NAME.startswith("flash-rollout-qualification-"):
    raise ValueError("Qualification must use an isolated app name")
PHASE = os.environ.get("ROLLOUT_QUALIFICATION_PHASE", "legacy")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi==0.135.2")
    .env({"ROLLOUT_QUALIFICATION_APP": APP_NAME, "ROLLOUT_QUALIFICATION_PHASE": PHASE})
    .add_local_python_source("flash")
)
app = modal.App(APP_NAME, image=image)


class _Engine:
    @modal.enter()
    async def load(self):
        await asyncio.sleep(10)

    @modal.method()
    def health(self):
        return {"ok": True, "base_model": "base"}

    @modal.method()
    async def generate(self):
        await asyncio.sleep(2)
        return {"engine_phase": PHASE}


# Separate method objects, like production's class factory. Keep the legacy name
# registered in every phase so old routers can dispatch until they finish draining.
def _build(name, concurrency):
    class Engine:
        load = _Engine.__dict__["load"]
        health = _Engine.__dict__["health"]
        generate = _Engine.__dict__["generate"]

    Engine.__name__ = Engine.__qualname__ = name
    globals()[name] = Engine
    return app.cls(max_containers=2, scaledown_window=60, timeout=120)(
        modal.concurrent(max_inputs=concurrency)(Engine)
    )


LegacyEngine = _build("LegacyEngine", 4)
if PHASE != "legacy":
    StableModelEngine = _build("StableModelEngine", 4 if PHASE == "model" else 8)


@app.function(min_containers=1, max_containers=2, startup_timeout=120, timeout=120)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def router():
    from fastapi import FastAPI
    from starlette.responses import StreamingResponse

    engine = LegacyEngine() if PHASE == "legacy" else StableModelEngine()

    @contextlib.asynccontextmanager
    async def lifespan(_api):
        if PHASE != "legacy":
            await warm_active_models({"base": engine}, {"base": LegacyEngine()})
        yield

    api = FastAPI(lifespan=lifespan)

    @api.get("/")
    async def generate():
        return {"router_phase": PHASE, **await engine.generate.remote.aio()}

    @api.get("/stream")
    async def stream():
        async def events():
            for _ in range(5):
                result = await engine.generate.remote.aio()
                yield f"data: {json.dumps(result)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return api


async def _qualify():
    import httpx

    name = f"flash-rollout-qualification-{uuid.uuid4().hex[:10]}"
    results = []
    directory = Path(tempfile.mkdtemp(prefix="flash-rollout-qualification-"))
    env = {**os.environ, "ROLLOUT_QUALIFICATION_APP": name}
    stop = asyncio.Event()
    observed = {phase: asyncio.Event() for phase in ("legacy", "model", "updated")}

    async def command(*args, phase="legacy"):
        with (directory / f"{phase}.log").open("ab") as output:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "modal",
                *args,
                env={**env, "ROLLOUT_QUALIFICATION_PHASE": phase},
                stdout=output,
                stderr=asyncio.subprocess.STDOUT,
            )
            if await process.wait():
                raise RuntimeError(f"Modal command failed; see {directory}")

    async def request(client, url):
        started = time.monotonic()
        try:
            response = await client.get(url)
            row = {"status": response.status_code, **response.json()}
        except Exception as exc:
            row = {"error": type(exc).__name__}
        results.append({**row, "seconds": time.monotonic() - started})
        if row.get("router_phase") in observed:
            observed[row["router_phase"]].set()

    async def traffic(client, url):
        while not stop.is_set():
            await request(client, url)

    async def streaming_traffic(client, url):
        while not stop.is_set():
            try:
                async with client.stream("GET", url + "/stream") as response:
                    events = [line async for line in response.aiter_lines() if line]
                    complete = len(events) == 6 and events[-1] == "data: [DONE]"
                    results.append({"status": response.status_code, "stream_complete": complete})
            except Exception as exc:
                results.append({"error": type(exc).__name__, "stream_complete": False})

    tasks = []
    try:
        await command("deploy", "--strategy", "rolling", __file__)
        url = await modal.Function.from_name(name, "router").get_web_url.aio()
        async with httpx.AsyncClient(timeout=120) as client:
            await request(client, url)  # establish an active legacy pool before migration
            tasks = [asyncio.create_task(traffic(client, url)) for _ in range(4)]
            tasks.append(asyncio.create_task(streaming_traffic(client, url.rstrip("/"))))
            for phase in ("model", "updated"):
                await command("deploy", "--strategy", "rolling", __file__, phase=phase)
                async with asyncio.timeout(180):
                    await observed[phase].wait()
                print(f"Serving phase {phase}; {len(results)} requests observed", flush=True)
            stop.set()
            await asyncio.gather(*tasks)
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        (directory / "requests.json").write_text(json.dumps(results, indent=2))
        await command("app", "stop", "-y", name)
        print(f"Stopped {name}; evidence: {directory}", flush=True)
    failures = [
        row for row in results if row.get("status") != 200 or row.get("stream_complete") is False
    ]
    print(json.dumps({"requests": len(results), "failures": len(failures)}))
    if failures:
        raise RuntimeError("Rollout interrupted requests")


if __name__ == "__main__":
    asyncio.run(_qualify())
