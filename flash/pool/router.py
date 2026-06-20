"""The rollout router — an "nginx for GRPO rollouts" (FastAPI ASGI app).

It accepts OpenAI-compatible generation requests from trainers, resolves the requested adapter/base
model to a healthy GPU backend (least outstanding requests, LoRA-aware), lazily loads/hot-swaps the
adapter there, forwards the request, and fails over to another backend on error. It also fans
reward scoring out to off-GPU reward workers, and exposes registration + status endpoints.

State mutations go through a single :class:`asyncio.Lock`; the (potentially slow) network IO
(loading a LoRA, forwarding a generation) happens OUTSIDE the lock, with in-flight slots reserved
under the lock first so concurrent picks balance correctly.

Build it with :func:`create_pool_app`. The :class:`Router` holds the wiring so tests can drive the
balancing logic directly and inject a fake gateway.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from flash.pool.config import RouterConfig
from flash.pool.gateway import BackendGateway, GatewayError
from flash.pool.protocol import join
from flash.pool.rewards import NoRewardCapacityError, RewardRegistry, RewardWorker
from flash.pool.state import Adapter, Backend, NoCapacityError, PlacementDecision, PoolState


class Router:
    def __init__(
        self,
        state: PoolState | None = None,
        *,
        gateway: BackendGateway | None = None,
        rewards: RewardRegistry | None = None,
        config: RouterConfig | None = None,
    ):
        self.state = state or PoolState()
        self.gateway = gateway or BackendGateway()
        self.rewards = rewards or RewardRegistry()
        self.config = config or RouterConfig()
        self._lock = asyncio.Lock()

    # ---------- adapter resolution ----------
    async def _resolve(self, model: str) -> tuple[str | None, str]:
        """Map an OpenAI ``model`` field to (adapter_name | None, base_model). An adapter name is
        returned when ``model`` is a registered run; otherwise ``model`` is treated as a base id
        that some backend serves directly."""
        async with self._lock:
            if model in self.state.adapters:
                ad = self.state.adapters[model]
                return ad.name, ad.base_model
            serves = any(b.base_model == model for b in self.state.backends.values())
        if serves:
            return None, model
        raise NoCapacityError(f"unknown model {model!r}: not a registered adapter or served base model")

    # ---------- the core generation proxy ----------
    async def generate(self, body: dict, *, kind: str = "chat") -> tuple[dict, str]:
        """Forward an OpenAI chat/completions ``body`` to a backend. Returns (response, backend_id).
        Retries on a different backend up to ``config.max_retries`` times."""
        model = body.get("model")
        if not model:
            raise ValueError("request body missing 'model'")
        adapter_name, base_model = await self._resolve(model)

        tried: set[str] = set()
        last_err: Exception | None = None
        for _ in range(self.config.max_retries + 1):
            # --- pick + reserve a backend atomically ---
            async with self._lock:
                try:
                    if adapter_name is not None:
                        decision = self.state.pick_backend(adapter_name, exclude=tried)
                    else:
                        decision = PlacementDecision(self.state.pick_for_base(base_model, exclude=tried))
                except NoCapacityError as e:
                    last_err = e
                    break
                be = decision.backend
                ad = self.state.adapters.get(adapter_name) if adapter_name else None
                # Decide whether the adapter must be (re)loaded: not present, or a stale version
                # (a weight-sync bumped it). ``reload_existing`` means hot-swap (unload then load).
                present = adapter_name in be.adapters if adapter_name else True
                stale = bool(ad is not None and present and ad.loaded_version.get(be.id, -1) != ad.version)
                must_load = decision.needs_load or (adapter_name is not None and (not present or stale))
                reload_existing = adapter_name is not None and present and must_load
                evict = decision.evict
                self.state.acquire(be.id, adapter_name)  # reserve the slot before slow IO

            # --- slow IO outside the lock ---
            try:
                if must_load and ad is not None:
                    if reload_existing:  # hot-swap: vLLM needs an unload before re-loading a name
                        await self.gateway.unload_lora(be, adapter_name)
                        async with self._lock:
                            self.state.mark_unloaded(be.id, adapter_name)
                    if evict:
                        await self.gateway.unload_lora(be, evict)
                        async with self._lock:
                            self.state.mark_unloaded(be.id, evict)
                    await self.gateway.load_lora(be, adapter_name, ad.uri)
                    async with self._lock:
                        self.state.mark_loaded(be.id, adapter_name)

                payload = dict(body)
                payload["model"] = adapter_name or base_model
                resp = await (
                    self.gateway.chat(be, payload) if kind == "chat" else self.gateway.completions(be, payload)
                )
                async with self._lock:
                    self.state.release(be.id)
                return resp, be.id
            except GatewayError as e:
                last_err = e
                async with self._lock:
                    self.state.release(be.id, failed=True)
                    be_now = self.state.backends.get(be.id)
                    if be_now is not None and be_now.total_failures % max(1, self.config.fail_threshold) == 0:
                        self.state.set_health(be.id, False, str(e))
                tried.add(be.id)
                continue

        raise NoCapacityError(f"all backends failed for model {model!r}: {last_err}")

    # ---------- reward fan-out ----------
    async def score(self, body: dict) -> tuple[dict, str]:
        tried: set[str] = set()
        last_err: Exception | None = None
        for _ in range(self.config.max_retries + 1):
            async with self._lock:
                try:
                    w = self.rewards.pick(exclude=tried)
                except NoRewardCapacityError as e:
                    last_err = e
                    break
                self.rewards.acquire(w.id)
            try:
                resp = await self.gateway.post(w.id, join(w.url, "/score"), body)
                async with self._lock:
                    self.rewards.release(w.id)
                return resp, w.id
            except GatewayError as e:
                last_err = e
                async with self._lock:
                    self.rewards.release(w.id, failed=True)
                    self.rewards.set_health(w.id, False)
                tried.add(w.id)
                continue
        raise NoRewardCapacityError(f"all reward workers failed: {last_err}")

    # ---------- weight sync (per GRPO step) ----------
    async def sync_adapter(self, name: str, uri: str | None = None) -> dict:
        """Record new weights for ``name`` and hot-swap them onto every backend hosting it.

        This is the per-step GRPO weight transfer: the trainer pushes its updated LoRA, the router
        unloads+reloads it on each placement so the pool serves the fresh policy. Reloads run
        concurrently; a backend that fails its reload is marked stale (lazy reload on next serve)."""
        async with self._lock:
            if name not in self.state.adapters:
                raise NoCapacityError(f"unknown adapter {name!r}")
            ad = self.state.bump_adapter_version(name, uri)
            placements = [self.state.backends[b] for b in ad.placements if b in self.state.backends]
            version = ad.version

        async def _reload(be: Backend) -> bool:
            try:
                await self.gateway.unload_lora(be, name)
                await self.gateway.load_lora(be, name, ad.uri)
            except GatewayError:
                return False
            async with self._lock:
                cur = self.state.adapters.get(name)
                if cur is not None and be.id in cur.placements:
                    cur.loaded_version[be.id] = version
            return True

        results = await asyncio.gather(*[_reload(be) for be in placements]) if placements else []
        return {"name": name, "version": version, "reloaded": sum(results), "placements": len(placements)}

    async def place_adapter(self, name: str) -> dict:
        """Eagerly load ``name`` onto enough backends to reach its desired replica count."""
        async with self._lock:
            decisions = self.state.plan_placements(name)
            ad = self.state.adapters[name]
        loaded = 0
        for d in decisions:
            be = d.backend
            try:
                if d.evict:
                    await self.gateway.unload_lora(be, d.evict)
                    async with self._lock:
                        self.state.mark_unloaded(be.id, d.evict)
                await self.gateway.load_lora(be, name, ad.uri)
                async with self._lock:
                    self.state.mark_loaded(be.id, name)
                loaded += 1
            except GatewayError:
                continue
        async with self._lock:
            return {"name": name, "placements": sorted(self.state.adapters[name].placements), "loaded": loaded}

    # ---------- health ----------
    async def health_sweep(self) -> None:
        backends = list(self.state.backends.values())
        workers = list(self.rewards.workers.values())
        be_results = await asyncio.gather(*[self.gateway.health(b) for b in backends]) if backends else []
        for be, ok in zip(backends, be_results, strict=True):
            async with self._lock:
                self.state.set_health(be.id, ok, "" if ok else "health probe failed")
        wk_results = (
            await asyncio.gather(*[self._reward_health(w) for w in workers]) if workers else []
        )
        for w, ok in zip(workers, wk_results, strict=True):
            async with self._lock:
                self.rewards.set_health(w.id, ok)

    async def _reward_health(self, w: RewardWorker) -> bool:
        return await self.gateway.health(Backend(id=w.id, url=w.url, base_model=""))


def create_pool_app(
    state: PoolState | None = None,
    *,
    gateway: BackendGateway | None = None,
    rewards: RewardRegistry | None = None,
    config: RouterConfig | None = None,
    health_loop: bool | None = None,
):
    """Build the rollout-router FastAPI app. Inject ``gateway`` (tests) to dispatch to in-process
    fake backends; ``health_loop`` defaults to on when ``config.health_interval > 0``."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    router = Router(state, gateway=gateway, rewards=rewards, config=config)
    run_health = router.config.health_interval > 0 if health_loop is None else health_loop

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task: asyncio.Task | None = None
        if run_health:

            async def _loop() -> None:
                while True:
                    await asyncio.sleep(router.config.health_interval)
                    with contextlib.suppress(Exception):
                        await router.health_sweep()

            task = asyncio.create_task(_loop())
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await router.gateway.aclose()

    app = FastAPI(title="flash rollout pool router", lifespan=lifespan)
    app.state.router = router  # expose for tests / introspection

    @app.get("/health")
    async def health() -> dict:
        snap = router.state.snapshot()
        return {"status": "ok", "backends": snap["summary"]["healthy_backends"], "adapters": snap["summary"]["adapters"]}

    @app.get("/pool/status")
    async def status() -> dict:
        return {"pool": router.state.snapshot(), "rewards": router.rewards.snapshot()}

    # ---- backend registration ----
    @app.post("/pool/backends")
    async def add_backend(body: dict) -> dict:
        try:
            be = Backend(
                id=body["id"],
                url=body["url"],
                base_model=body["base_model"],
                gpu_label=body.get("gpu_label", ""),
                max_loras=int(body.get("max_loras", 8)),
                max_concurrency=int(body.get("max_concurrency", 256)),
                cost_per_hour=float(body.get("cost_per_hour", 0.0)),
            )
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"missing field {e}") from e
        async with router._lock:
            router.state.add_backend(be)
        return be.snapshot()

    @app.delete("/pool/backends/{backend_id}")
    async def remove_backend(backend_id: str) -> dict:
        async with router._lock:
            removed = router.state.remove_backend(backend_id)
        if removed is None:
            raise HTTPException(status_code=404, detail=f"no backend {backend_id!r}")
        return {"removed": backend_id}

    # ---- adapter (run) registration ----
    @app.post("/adapters")
    async def register_adapter(body: dict) -> dict:
        try:
            ad = Adapter(
                name=body["name"],
                base_model=body["base_model"],
                uri=body["uri"],
                replicas=int(body.get("replicas", router.config.default_replicas)),
            )
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"missing field {e}") from e
        async with router._lock:
            router.state.register_adapter(ad)
        result = ad.snapshot()
        if body.get("place"):  # optional eager placement to reach the replica count
            result["placement"] = await router.place_adapter(ad.name)
        return result

    @app.post("/adapters/{name}/sync")
    async def sync_adapter(name: str, body: dict | None = None) -> dict:
        try:
            return await router.sync_adapter(name, (body or {}).get("uri"))
        except NoCapacityError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.post("/adapters/{name}/place")
    async def place_adapter(name: str) -> dict:
        async with router._lock:
            known = name in router.state.adapters
        if not known:
            raise HTTPException(status_code=404, detail=f"unknown adapter {name!r}")
        return await router.place_adapter(name)

    @app.delete("/adapters/{name}")
    async def drop_adapter(name: str) -> dict:
        async with router._lock:
            ad = router.state.adapters.get(name)
            placements = list(ad.placements) if ad else []
            router.state.drop_adapter(name)
        # best-effort unload from the backends it lived on
        for bid in placements:
            be = router.state.backends.get(bid)
            if be is not None:
                with contextlib.suppress(GatewayError):
                    await router.gateway.unload_lora(be, name)
        return {"dropped": name}

    # ---- OpenAI-compatible generation ----
    @app.get("/v1/models")
    async def list_models() -> dict:
        snap = router.state.snapshot()
        data = [{"id": m, "object": "model"} for m in snap["summary"]["base_models"]]
        data += [{"id": a["name"], "object": "model", "base_model": a["base_model"]} for a in snap["adapters"]]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict) -> JSONResponse:
        return await _gen(body, "chat")

    @app.post("/v1/completions")
    async def completions(body: dict) -> JSONResponse:
        return await _gen(body, "completions")

    async def _gen(body: dict, kind: str) -> JSONResponse:
        try:
            resp, backend_id = await router.generate(body, kind=kind)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except NoCapacityError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return JSONResponse(resp, headers={"X-Flash-Backend": backend_id})

    # ---- reward fan-out ----
    @app.post("/rewards/workers")
    async def add_reward_worker(body: dict) -> dict:
        try:
            w = RewardWorker(id=body["id"], url=body["url"])
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"missing field {e}") from e
        async with router._lock:
            router.rewards.add(w)
        return w.snapshot()

    @app.delete("/rewards/workers/{worker_id}")
    async def remove_reward_worker(worker_id: str) -> dict:
        async with router._lock:
            removed = router.rewards.remove(worker_id)
        if removed is None:
            raise HTTPException(status_code=404, detail=f"no reward worker {worker_id!r}")
        return {"removed": worker_id}

    @app.post("/rewards/score")
    async def reward_score(body: dict) -> JSONResponse:
        try:
            resp, worker_id = await router.score(body)
        except NoRewardCapacityError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return JSONResponse(resp, headers={"X-Flash-Reward-Worker": worker_id})

    return app
