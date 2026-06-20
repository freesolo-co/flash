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


class _BackendSwapGuard:
    """A per-backend single-writer / many-reader guard that serializes adapter (un)loads against
    in-flight generations on the SAME backend, while keeping reads concurrent and DIFFERENT backends
    fully parallel.

    An adapter (un)load is a *writer* (hot-swap / lazy load / eviction): exclusive on that backend so
    a generate never forwards a request while that backend is mid-swap. A generate is a *reader*:
    many run at once, but none start (or are in flight) while a swap holds the backend. This is the
    fix for the "parallel LoRA reload races generate" race — the swap window is no longer observable
    by a forwarded request on the same backend, yet other backends keep serving.
    """

    def __init__(self) -> None:
        self._readers = 0
        self._writing = False
        self._writers_waiting = 0  # writer-preference: don't let a stream of reads starve a swap
        self._cond = asyncio.Condition()

    @contextlib.asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        async with self._cond:
            # Yield to a writing OR waiting writer so a per-step weight sync can't be starved by a
            # continuous flow of generations on a hot backend.
            await self._cond.wait_for(lambda: not self._writing and self._writers_waiting == 0)
            self._readers += 1
        try:
            yield
        finally:
            async with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextlib.asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        async with self._cond:
            self._writers_waiting += 1
            try:
                await self._cond.wait_for(lambda: not self._writing and self._readers == 0)
            finally:
                self._writers_waiting -= 1
            self._writing = True
        try:
            yield
        finally:
            async with self._cond:
                self._writing = False
                self._cond.notify_all()


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
        # Per-backend single-writer/many-reader guards: an adapter (un)load is a writer (exclusive
        # on that backend), a forwarded generation is a reader. This serializes hot-swaps against
        # in-flight generations on the SAME backend without serializing across DIFFERENT backends.
        self._swap_guards: dict[str, _BackendSwapGuard] = {}

    def _guard(self, backend_id: str) -> _BackendSwapGuard:
        g = self._swap_guards.get(backend_id)
        if g is None:
            g = self._swap_guards[backend_id] = _BackendSwapGuard()
        return g

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
        failovers = 0
        reloads = 0
        # An adapter can vanish from a backend between our pick and the request: a concurrent
        # weight-sync hot-swap (unload+load) or vLLM's own LRU eviction under max_loras pressure.
        # That surfaces as a 400 "lora not loaded" — it is NOT a backend failure, so we reload the
        # adapter and retry the SAME backend rather than condemning it. Bound the reloads so a truly
        # broken adapter can't loop forever.
        max_reloads = len(self.state.backends) + 2
        while True:
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

            # --- slow IO outside the global lock; per-backend swap-guarded ---
            guard = self._guard(be.id)
            # True only once we're past (re)load and into the generation call, so a "missing adapter"
            # error is attributed to a chat miss (concurrent swap / LRU eviction) — NOT to a failed
            # load_lora (e.g. a bad adapter URI surfacing "not found"/"does not exist"), which must
            # fail over to a different backend instead of looping reloads on this one.
            in_generation = False
            try:
                if must_load and ad is not None:
                    # The (un)load is a WRITER on this backend: exclusive, so no generate forwards a
                    # request to it while it's mid-swap (and no two swaps race each other here).
                    async with guard.write():
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
                # Forwarding is a READER: concurrent with other generates, but never overlapping an
                # adapter swap on this backend (so a request can't hit a half-loaded adapter).
                async with guard.read():
                    in_generation = True
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
                # Adapter swapped/evicted out from under us AT GENERATION TIME -> drop its placement
                # here and retry the SAME backend (the next pick reloads it). Only for a chat miss,
                # not a load/unload failure. Don't fail over, don't mark the backend unhealthy.
                if (
                    in_generation
                    and adapter_name is not None
                    and _is_missing_adapter(e)
                    and reloads < max_reloads
                ):
                    reloads += 1
                    async with self._lock:
                        self.state.mark_unloaded(be.id, adapter_name)
                    continue
                # Genuine backend trouble: fail over to a different backend. Only a TRANSPORT error
                # (no HTTP status = connection refused/timeout) condemns the backend as unhealthy;
                # an HTTP 5xx might be transient, so we just exclude it for this request.
                async with self._lock:
                    if e.status is None:
                        self.state.set_health(be.id, False, str(e))
                tried.add(be.id)
                failovers += 1
                if failovers > self.config.max_retries:
                    break
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
        concurrently across DIFFERENT backends but are serialized against in-flight generations on
        the SAME backend (the per-backend swap guard), so no request hits a half-swapped adapter. A
        backend whose reload FAILS is recorded as no longer hosting the adapter (its placement is
        dropped) so the tracked state matches reality — the next serve loads it fresh rather than
        forwarding to a backend the router wrongly believes is warm."""
        async with self._lock:
            if name not in self.state.adapters:
                raise NoCapacityError(f"unknown adapter {name!r}")
            ad = self.state.bump_adapter_version(name, uri)
            placements = [self.state.backends[b] for b in ad.placements if b in self.state.backends]
            version = ad.version

        async def _reload(be: Backend) -> bool:
            # Writer on this backend: exclusive vs generates + other swaps on the same backend.
            async with self._guard(be.id).write():
                try:
                    await self.gateway.unload_lora(be, name)
                    await self.gateway.load_lora(be, name, ad.uri)
                except GatewayError:
                    # The unload likely already removed it (or the load failed): vLLM no longer has
                    # this adapter on this backend, so drop the placement instead of leaving it
                    # marked synced/stale-but-present. State now reflects reality.
                    async with self._lock:
                        self.state.mark_unloaded(be.id, name)
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
                # Writer on this backend: don't race a concurrent generate's load/hot-swap.
                async with self._guard(be.id).write():
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


def _is_missing_adapter(e: GatewayError) -> bool:
    """True when a generation 400 means 'this LoRA isn't loaded here' (a concurrent hot-swap or
    vLLM LRU eviction) rather than a real backend fault — so the router reloads + retries."""
    if e.status != 400:
        return False
    msg = str(e).lower()
    return any(s in msg for s in ("lora not loaded", "not found", "does not exist", "not loaded", "no adapter"))


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
