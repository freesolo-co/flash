"""CPU-only Modal handoff qualification; never targets the serving application.

Run with ``python -m scripts.qualify_serving_rollout`` in a Modal-authenticated
environment. It migrates a legacy engine under traffic, then changes concurrency
under the stable model identity, then restores the legacy implementation through
the production rollback artifact builder. Each replacement takes ten seconds to start.
Use --unsafe-rollback to reproduce deletion failures in the disposable app.
The temporary app is stopped in finally; request results contain no user data.
"""

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import modal

from flash.serving.app.rollback import prepare_artifact
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


def _build_engine(gpu, name, concurrency, target_inputs):
    class Engine:
        base_model: str = modal.parameter(default="base")

        @modal.enter()
        async def load(self):
            await asyncio.sleep(10)

        @modal.method()
        def health(self):
            return {"ok": True, "base_model": self.base_model}

        @modal.method()
        async def generate(self):
            await asyncio.sleep(2)
            return {"engine_phase": PHASE}

        @modal.method()
        async def stream_generate_call(self):
            return {"engine_phase": PHASE}

        @modal.method()
        def register(self):
            pass

        @modal.method()
        def unregister(self):
            pass

    Engine.__name__ = Engine.__qualname__ = name
    globals()[name] = Engine
    return app.cls(max_containers=2, scaledown_window=60, timeout=120)(
        modal.concurrent(max_inputs=concurrency, target_inputs=target_inputs)(Engine)
    )


LegacyEngine = _build_engine("CPU", "LegacyEngine", 4, 4)
if PHASE not in {"legacy", "rollback"}:
    cap = 4 if PHASE == "model" else 8
    StableModelEngine = _build_engine("CPU", "StableModelEngine", cap, cap)


def gpu_for(model):
    if model != "base":
        raise ValueError("Unsupported base model")
    return "CPU"


def _engine_concurrency(model):
    gpu_for(model)
    return (8, 8) if PHASE == "updated" else (4, 4)


def _engine_cls_for(model):
    gpu_for(model)
    return LegacyEngine if PHASE in {"legacy", "rollback"} else StableModelEngine


class _ModalEnginePool:
    async def start(self):
        if PHASE not in {"legacy", "rollback"}:
            await warm_active_models(
                {"base": StableModelEngine(base_model="base")},
                {"base": LegacyEngine(base_model="base")},
            )


@app.function(min_containers=1, max_containers=2, startup_timeout=120, timeout=120)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def router():
    from fastapi import FastAPI
    from starlette.responses import StreamingResponse

    engine = _engine_cls_for("base")(base_model="base")
    pool = _ModalEnginePool()

    @contextlib.asynccontextmanager
    async def lifespan(_api):
        await pool.start()
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


def _prepare_predecessor(directory, unsafe_rollback):
    deploy_cwd = directory / "predecessor"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "flash",
        deploy_cwd / "flash",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    entrypoint = deploy_cwd / "flash/serving/app/qualification_app.py"
    shutil.copyfile(__file__, entrypoint)
    if not unsafe_rollback:
        prepare_artifact(
            entrypoint,
            {
                "version": 1,
                "source_sha": "a" * 40,
                "startup_seconds": 120,
                "classes": {
                    "LegacyEngine": ["base"],
                    "StableModelEngine": ["base"],
                },
            },
        )
    return entrypoint, deploy_cwd


async def _qualify(*, unsafe_rollback=False):
    import httpx

    name = f"flash-rollout-qualification-{uuid.uuid4().hex[:10]}"
    results = []
    directory = Path(tempfile.mkdtemp(prefix="flash-rollout-qualification-"))
    env = {**os.environ, "ROLLOUT_QUALIFICATION_APP": name}
    stop = asyncio.Event()
    observed = {phase: asyncio.Event() for phase in ("legacy", "model", "updated", "rollback")}

    async def command(*args, phase="legacy", cwd=None):
        with (directory / f"{phase}.log").open("ab") as output:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "modal",
                *args,
                env={**env, "ROLLOUT_QUALIFICATION_PHASE": phase},
                cwd=cwd,
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
        phase = row.get("router_phase")
        if phase in observed and (phase != "rollback" or row.get("engine_phase") == "rollback"):
            observed[phase].set()

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
            for phase in ("model", "updated", "rollback"):
                entrypoint, deploy_cwd = __file__, None
                if phase == "rollback":
                    entrypoint, deploy_cwd = await asyncio.to_thread(
                        _prepare_predecessor, directory, unsafe_rollback
                    )
                await command(
                    "deploy", "--strategy", "rolling", str(entrypoint), phase=phase, cwd=deploy_cwd
                )
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unsafe-rollback",
        action="store_true",
        help="Reproduce handle deletion in this disposable CPU app",
    )
    args = parser.parse_args()
    asyncio.run(_qualify(unsafe_rollback=args.unsafe_rollback))
