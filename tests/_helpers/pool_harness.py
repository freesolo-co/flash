"""Test harness for the rollout pool: a faithful fake vLLM backend + real-localhost servers.

The fake backend implements vLLM's actual dynamic-LoRA + OpenAI surface (load/unload, chat with
``model`` = lora name, 400 when a lora isn't loaded) so the router/client exercise the SAME code
path they would against real vLLM. Servers run on real loopback ports (no ASGI-transport quirks,
real concurrency) which is what makes the overlap/balancing tests meaningful.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from dataclasses import dataclass, field

import httpx
import uvicorn


# --------------------------------------------------------------------------------------
# Fake vLLM backend
# --------------------------------------------------------------------------------------
@dataclass
class BackendRecord:
    base_model: str
    loaded: set[str] = field(default_factory=set)
    load_calls: list[str] = field(default_factory=list)
    unload_calls: list[str] = field(default_factory=list)
    chat_calls: list[str] = field(default_factory=list)  # model per chat request
    max_inflight: int = 0
    _inflight: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)


def make_fake_vllm(
    base_model: str,
    *,
    latency: float = 0.0,
    fail: bool = False,
    completion: str | None = None,
):
    """A fake vLLM OpenAI server. ``latency`` simulates decode time; ``fail`` makes chat 500."""
    from fastapi import FastAPI, HTTPException

    rec = BackendRecord(base_model=base_model)
    app = FastAPI()
    app.state.record = rec

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict:
        data = [{"id": base_model, "object": "model"}]
        data += [{"id": n, "object": "model"} for n in sorted(rec.loaded)]
        return {"object": "list", "data": data}

    @app.post("/v1/load_lora_adapter")
    async def load_lora(body: dict) -> dict:
        name = body["lora_name"]
        rec.load_calls.append(name)
        if name in rec.loaded:  # vLLM 400s on a duplicate load
            raise HTTPException(status_code=400, detail=f"lora {name} has already been loaded")
        rec.loaded.add(name)
        return {"ok": True}

    @app.post("/v1/unload_lora_adapter")
    async def unload_lora(body: dict) -> dict:
        name = body["lora_name"]
        rec.unload_calls.append(name)
        if name not in rec.loaded:
            raise HTTPException(status_code=400, detail=f"lora {name} not found")
        rec.loaded.discard(name)
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(body: dict) -> dict:
        import anyio

        model = body.get("model", base_model)
        rec.chat_calls.append(model)
        # vLLM rejects a request for a lora model that isn't loaded.
        if model != base_model and model not in rec.loaded:
            raise HTTPException(status_code=400, detail=f"model {model} not found / lora not loaded")
        if fail:
            raise HTTPException(status_code=500, detail="injected failure")
        with rec._lock:
            rec._inflight += 1
            rec.max_inflight = max(rec.max_inflight, rec._inflight)
        try:
            if latency:
                await anyio.sleep(latency)
        finally:
            with rec._lock:
                rec._inflight -= 1
        n = int(body.get("n", 1))
        text = completion if completion is not None else f"[{model}] ok"
        choices = [
            {"index": i, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            for i in range(n)
        ]
        return {"object": "chat.completion", "model": model, "choices": choices}

    return app, rec


def make_fake_reward(scorer=None, *, reward_id: str = "default", latency: float = 0.0):
    """A reward worker; default scorer = completion length / 100."""
    from flash.pool.rewards import create_reward_app

    def _default(prompts, completions, info):
        if latency:
            time.sleep(latency)
        return [min(1.0, len(c) / 100.0) for c in completions]

    return create_reward_app(scorer or _default, reward_id=reward_id)


# --------------------------------------------------------------------------------------
# Real loopback server runner
# --------------------------------------------------------------------------------------
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class RunningServer:
    def __init__(self, app, port: int):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self, timeout: float = 10.0) -> RunningServer:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError(f"server on :{self.port} did not start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)


def start_server(app) -> RunningServer:
    return RunningServer(app, _free_port()).start()


@dataclass
class PoolHarness:
    """A running router + its backends/reward-workers, with handles for assertions."""

    router_url: str
    router: RunningServer
    backends: dict[str, tuple[RunningServer, BackendRecord]] = field(default_factory=dict)
    rewards: dict[str, RunningServer] = field(default_factory=dict)
    _servers: list[RunningServer] = field(default_factory=list)

    def client(self) -> httpx.Client:
        return httpx.Client(base_url=self.router_url, timeout=30.0)

    def record(self, backend_id: str) -> BackendRecord:
        return self.backends[backend_id][1]

    def status(self) -> dict:
        with self.client() as c:
            return c.get("/pool/status").json()

    def stop(self) -> None:
        for s in self._servers:
            with contextlib.suppress(Exception):
                s.stop()


def build_harness(
    backends: list[dict],
    *,
    reward_workers: int = 0,
    reward_latency: float = 0.0,
    max_retries: int = 2,
) -> PoolHarness:
    """Start fake backends + reward workers + the router, all on loopback, and register them.

    ``backends`` = list of ``{id, base_model, latency?, fail?, max_loras?, completion?}``.
    """
    from flash.pool.config import RouterConfig
    from flash.pool.server import build_app

    servers: list[RunningServer] = []
    be_handles: dict[str, tuple[RunningServer, BackendRecord]] = {}
    for b in backends:
        app, rec = make_fake_vllm(
            b["base_model"], latency=b.get("latency", 0.0), fail=b.get("fail", False),
            completion=b.get("completion"),
        )
        srv = start_server(app)
        servers.append(srv)
        be_handles[b["id"]] = (srv, rec)

    reward_handles: dict[str, RunningServer] = {}
    for i in range(reward_workers):
        srv = start_server(make_fake_reward(reward_id="default", latency=reward_latency))
        servers.append(srv)
        reward_handles[f"rw{i}"] = srv

    # Router with the health loop OFF (deterministic; backends are healthy on registration).
    router_app = build_app(RouterConfig(health_interval=0.0, max_retries=max_retries))
    router_srv = start_server(router_app)
    servers.append(router_srv)

    # Register backends + reward workers.
    with httpx.Client(base_url=router_srv.url, timeout=30.0) as c:
        for b in backends:
            srv, _ = be_handles[b["id"]]
            c.post(
                "/pool/backends",
                json={
                    "id": b["id"],
                    "url": srv.url,
                    "base_model": b["base_model"],
                    "max_loras": b.get("max_loras", 8),
                    "max_concurrency": b.get("max_concurrency", 256),
                    "cost_per_hour": b.get("cost_per_hour", 1.0),
                },
            ).raise_for_status()
        for wid, srv in reward_handles.items():
            c.post("/rewards/workers", json={"id": wid, "url": srv.url}).raise_for_status()

    return PoolHarness(
        router_url=router_srv.url,
        router=router_srv,
        backends=be_handles,
        rewards=reward_handles,
        _servers=servers,
    )
