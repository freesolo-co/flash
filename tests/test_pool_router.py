"""End-to-end router tests: real loopback router + fake vLLM backends over real HTTP.

These exercise the exact request path the router uses against real vLLM (load_lora / chat with
model=lora / failover), proving the user's asks: many adapters share one GPU, requests spread
across GPUs, rollout + reward run off the trainer, and per-step weight sync hot-swaps the adapter.
"""

from __future__ import annotations

import asyncio

import pytest

from flash.pool.gateway import GatewayError
from flash.pool.router import Router
from flash.pool.state import Adapter, Backend, PoolState
from tests._helpers.pool_harness import build_harness


@pytest.fixture
def harness():
    h = build_harness(
        [
            {"id": "gpu0", "base_model": "Q"},
            {"id": "gpu1", "base_model": "Q"},
        ],
        reward_workers=2,
    )
    yield h
    h.stop()


def _register(c, name, base="Q", uri="/lora/x", replicas=1, place=True):
    return c.post(
        "/adapters",
        json={"name": name, "base_model": base, "uri": uri, "replicas": replicas, "place": place},
    )


def test_health_and_status(harness):
    with harness.client() as c:
        assert c.get("/health").json()["status"] == "ok"
        snap = c.get("/pool/status").json()
        assert snap["pool"]["summary"]["healthy_backends"] == 2
        assert snap["rewards"]["healthy"] == 2


def test_chat_lazily_loads_lora_then_routes(harness):
    with harness.client() as c:
        _register(c, "run1", place=False).raise_for_status()
        r = c.post("/v1/chat/completions", json={"model": "run1", "messages": [{"role": "user", "content": "hi"}]})
        r.raise_for_status()
        served_by = r.headers["X-Flash-Backend"]
        # the lora was loaded on whichever backend served it
        assert "run1" in harness.record(served_by).loaded
        assert harness.record(served_by).load_calls == ["run1"]
        assert r.json()["choices"][0]["message"]["content"] == "[run1] ok"


def test_multiple_runs_share_one_gpu():
    # Single GPU, three runs -> all three LoRAs co-resident on the one backend ("multiple models
    # on one GPU"), each routed correctly.
    h = build_harness([{"id": "gpu0", "base_model": "Q", "max_loras": 8}])
    try:
        with h.client() as c:
            for r in ("a", "b", "cc"):
                _register(c, r).raise_for_status()
                resp = c.post("/v1/chat/completions", json={"model": r, "messages": [{"role": "user", "content": "x"}]})
                resp.raise_for_status()
                assert resp.json()["choices"][0]["message"]["content"] == f"[{r}] ok"
        assert h.record("gpu0").loaded == {"a", "b", "cc"}
    finally:
        h.stop()


def test_requests_spread_across_gpus(harness):
    # An adapter replicated on both GPUs; many requests distribute across them (load balancing).
    with harness.client() as c:
        _register(c, "run", replicas=2, place=True).raise_for_status()
        snap = c.get("/pool/status").json()
        placements = next(a for a in snap["pool"]["adapters"] if a["name"] == "run")["placements"]
        assert set(placements) == {"gpu0", "gpu1"}
        for i in range(20):
            c.post("/v1/chat/completions", json={"model": "run", "messages": [{"role": "user", "content": str(i)}]}).raise_for_status()
    c0 = len(harness.record("gpu0").chat_calls)
    c1 = len(harness.record("gpu1").chat_calls)
    assert c0 > 0  # both GPUs got traffic
    assert c1 > 0
    assert c0 + c1 >= 20  # all 20 served (a transient failover retry could add a few)


def test_failover_to_healthy_backend():
    # gpu0 always 500s; the router must retry and succeed on gpu1.
    h = build_harness(
        [
            {"id": "gpu0", "base_model": "Q", "fail": True},
            {"id": "gpu1", "base_model": "Q"},
        ],
        max_retries=2,
    )
    try:
        with h.client() as c:
            _register(c, "run", replicas=2, place=True).raise_for_status()
            r = c.post("/v1/chat/completions", json={"model": "run", "messages": [{"role": "user", "content": "x"}]})
            r.raise_for_status()
            assert r.headers["X-Flash-Backend"] == "gpu1"
    finally:
        h.stop()


def test_weight_sync_hot_swaps_adapter(harness):
    with harness.client() as c:
        _register(c, "run", replicas=2, place=True).raise_for_status()
        # initial placement loaded run on both backends
        assert harness.record("gpu0").load_calls.count("run") == 1
        # a weight sync (per GRPO step) -> unload+reload on every placement
        out = c.post("/adapters/run/sync", json={"uri": "/lora/run/v1"}).json()
        assert out["version"] == 1
        assert out["reloaded"] == 2
        for bid in ("gpu0", "gpu1"):
            rec = harness.record(bid)
            assert rec.unload_calls.count("run") == 1
            assert rec.load_calls.count("run") == 2  # initial + reload
            assert "run" in rec.loaded


def test_reward_fan_out(harness):
    with harness.client() as c:
        body = {"prompts": ["p"], "completions": ["abcdefghij" * 5], "info": [{}]}
        r = c.post("/rewards/score", json=body)
        r.raise_for_status()
        assert r.headers["X-Flash-Reward-Worker"] in ("rw0", "rw1")
        assert r.json()["scores"][0] == pytest.approx(0.5)  # len 50 / 100


def test_adapter_evicted_from_backend_is_reloaded_not_failed(harness):
    # Simulate vLLM LRU-evicting (or a hot-swap gap) the adapter out from under the router: the
    # router's state still thinks it's loaded, but the backend 400s "not loaded". The router must
    # reload + retry (self-heal) rather than condemn the backend and 503.
    with harness.client() as c:
        _register(c, "run", replicas=1, place=True).raise_for_status()
        served = next(
            a for a in c.get("/pool/status").json()["pool"]["adapters"] if a["name"] == "run"
        )["placements"][0]
        rec = harness.record(served)
        loads_before = rec.load_calls.count("run")
        rec.loaded.discard("run")  # vLLM evicted it; router's registry is now stale
        r = c.post("/v1/chat/completions", json={"model": "run", "messages": [{"role": "user", "content": "x"}]})
        r.raise_for_status()  # must succeed, not 503
        assert rec.load_calls.count("run") == loads_before + 1  # reloaded
        # backend stayed healthy (a missing adapter is not a backend fault)
        snap = c.get("/pool/status").json()
        assert all(b["healthy"] for b in snap["pool"]["backends"])


def test_concurrent_generate_and_weight_sync_no_503(harness):
    # The prefetch race: generate the next batch WHILE syncing weights for the current one. The
    # hot-swap unload/reload window must not 503 any request (the bug the demo surfaced).
    import concurrent.futures as cf

    with harness.client() as c:
        _register(c, "run", replicas=2, place=True).raise_for_status()

        def gen(i):
            return c.post(
                "/v1/chat/completions",
                json={"model": "run", "messages": [{"role": "user", "content": str(i)}]},
            ).status_code

        def sync(_i):
            return c.post("/adapters/run/sync", json={"uri": "/lora/run/vX"}).status_code

        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            jobs = [pool.submit(gen if i % 3 else sync, i) for i in range(30)]
            codes = [j.result() for j in jobs]
        assert all(code == 200 for code in codes), codes


def test_unknown_model_is_503(harness):
    with harness.client() as c:
        r = c.post("/v1/chat/completions", json={"model": "nope", "messages": []})
        assert r.status_code == 503


def test_deregister_backend(harness):
    with harness.client() as c:
        c.delete("/pool/backends/gpu1").raise_for_status()
        snap = c.get("/pool/status").json()
        assert snap["pool"]["summary"]["backends"] == 1


# --------------------------------------------------------------------------------------
# In-process Router tests (fake gateway) for the swap-vs-generate serialization + failed-reload
# state. These drive Router directly (no loopback) so the concurrency is deterministic.
# --------------------------------------------------------------------------------------
class _RecordingGateway:
    """A fake gateway that records a per-backend event timeline so a test can assert no chat is
    forwarded to a backend while that backend is mid-(un)load. ``load_lora`` can be made to fail."""

    def __init__(self, *, load_delay: float = 0.0, fail_load_on: set[str] | None = None):
        self.load_delay = load_delay
        self.fail_load_on = fail_load_on or set()
        self.events: list[tuple[str, str, str]] = []  # (backend_id, op, name)
        self.active_swap: dict[str, int] = {}  # backend_id -> in-progress (un)load count
        self.chat_during_swap: list[str] = []  # backend_ids where a chat overlapped a swap

    async def unload_lora(self, be, name):
        self.events.append((be.id, "unload", name))
        self.active_swap[be.id] = self.active_swap.get(be.id, 0) + 1
        try:
            await asyncio.sleep(self.load_delay)
        finally:
            self.active_swap[be.id] -= 1

    async def load_lora(self, be, name, uri):
        self.events.append((be.id, "load", name))
        if name in self.fail_load_on:
            raise GatewayError(f"load failed for {name}", status=500)
        self.active_swap[be.id] = self.active_swap.get(be.id, 0) + 1
        try:
            await asyncio.sleep(self.load_delay)
        finally:
            self.active_swap[be.id] -= 1

    async def chat(self, be, body):
        if self.active_swap.get(be.id, 0) > 0:
            self.chat_during_swap.append(be.id)  # a generate hit a backend mid-swap -> bug
        self.events.append((be.id, "chat", body.get("model", "")))
        return {"object": "chat.completion", "choices": [{"message": {"content": "ok"}}]}

    async def completions(self, be, body):
        return await self.chat(be, body)

    async def aclose(self):
        pass


def _router_with(gw, *, backends, adapter="run", base="Q", replicas=1):
    state = PoolState()
    for bid in backends:
        state.add_backend(Backend(id=bid, url=f"http://{bid}", base_model=base, max_loras=8))
    r = Router(state, gateway=gw)
    state.register_adapter(Adapter(name=adapter, base_model=base, uri="/lora/run/v0", replicas=replicas))
    return r


def test_generate_never_overlaps_a_backend_swap():
    # HIGH: a per-step weight sync (hot-swap) must not race in-flight generates on the SAME backend.
    # With the per-backend swap guard, a chat is never forwarded to a backend while it is mid
    # unload/load, even while many generates and a sync run concurrently.
    gw = _RecordingGateway(load_delay=0.01)
    r = _router_with(gw, backends=["gpu0", "gpu1"], replicas=2)

    async def scenario():
        await r.place_adapter("run")  # warm both backends
        gens = [r.generate({"model": "run", "messages": []}) for _ in range(20)]
        syncs = [r.sync_adapter("run", f"/lora/run/v{i + 1}") for i in range(4)]
        await asyncio.gather(*gens, *syncs)

    asyncio.run(scenario())
    assert gw.chat_during_swap == [], f"a generate overlapped a swap on {gw.chat_during_swap}"


def test_sync_failed_reload_marks_backend_unsynced():
    # MED: if a backend's reload fails (unload ok, load raises), the router must NOT keep treating it
    # as hosting the adapter — the placement is dropped so tracked state matches vLLM reality.
    gw = _RecordingGateway(fail_load_on={"run"})
    r = _router_with(gw, backends=["gpu0"], replicas=1)

    async def scenario():
        # place it first WITHOUT the failing gateway so it's genuinely loaded...
        ok_gw = _RecordingGateway()
        r.gateway = ok_gw
        await r.place_adapter("run")
        assert "gpu0" in r.state.adapters["run"].placements
        # ...now a sync whose load fails must drop the placement.
        r.gateway = gw
        return await r.sync_adapter("run", "/lora/run/v1")

    out = asyncio.run(scenario())
    assert out["reloaded"] == 0
    assert out["placements"] == 1  # it WAS placed, but the reload failed
    ad = r.state.adapters["run"]
    assert "gpu0" not in ad.placements  # placement dropped (no longer marked as having the adapter)
    assert "run" not in r.state.backends["gpu0"].adapters
    # and it is NOT reported as a stale-but-present placement (which would imply "still loaded")
    assert ad.stale_placements() == set()
