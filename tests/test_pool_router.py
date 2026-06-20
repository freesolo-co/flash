"""End-to-end router tests: real loopback router + fake vLLM backends over real HTTP.

These exercise the exact request path the router uses against real vLLM (load_lora / chat with
model=lora / failover), proving the user's asks: many adapters share one GPU, requests spread
across GPUs, rollout + reward run off the trainer, and per-step weight sync hot-swaps the adapter.
"""

from __future__ import annotations

import pytest

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
