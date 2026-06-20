"""Trainer-side client tests: generate/score/sync over the pool, and the pipelined overlap that
makes reward latency free (it overlaps the optimizer step instead of costing trainer-GPU time)."""

from __future__ import annotations

import threading
import time

import pytest

from flash.pool.client import RolloutPoolClient, ShortCompletionGroupError
from tests._helpers.pool_harness import build_harness, start_server


@pytest.fixture
def harness():
    h = build_harness(
        [{"id": "gpu0", "base_model": "Q"}, {"id": "gpu1", "base_model": "Q"}],
        reward_workers=1,
    )
    yield h
    h.stop()


def _client(h, **kw) -> RolloutPoolClient:
    return RolloutPoolClient(h.router_url, adapter="run", base_model="Q", **kw)


def test_register_generate_score_sync(harness):
    with _client(harness) as c:
        c.register(uri="/lora/run/v0", replicas=2)
        outs = c.generate("hello", n=4)
        assert outs == ["[run] ok"] * 4
        scores = c.score(["p", "p"], ["abcde" * 10, "x"])  # len 50 -> 0.5, len 1 -> 0.01
        assert scores[0] == pytest.approx(0.5)
        sync = c.sync_weights("/lora/run/v1")
        assert sync["version"] == 1
        assert sync["reloaded"] == 2


def test_experience_stream_shapes_and_rewards(harness):
    with _client(harness) as c:
        c.register(uri="/lora/run/v0", replicas=2)
        batches = [["p1", "p2"], ["p3", "p4", "p5"]]
        exps = list(c.experience_stream(batches, n=3, prefetch=2))
        assert [e.num_prompts for e in exps] == [2, 3]
        for e in exps:
            assert all(len(g) == 3 for g in e.completions)  # n=3 group per prompt
            assert all(len(r) == 3 for r in e.rewards)  # one reward per completion
            assert e.num_completions == e.num_prompts * 3


def test_generation_fans_out_across_gpus(harness):
    # The producer issues per-prompt generate calls concurrently -> both GPUs get traffic.
    with _client(harness) as c:
        c.register(uri="/lora/run/v0", replicas=2)
        big_batch = [f"p{i}" for i in range(24)]
        exps = list(c.experience_stream([big_batch], n=1, prefetch=1, max_concurrency=24))
        assert exps[0].num_prompts == 24
    c0 = len(harness.record("gpu0").chat_calls)
    c1 = len(harness.record("gpu1").chat_calls)
    assert c0 > 0
    assert c1 > 0
    assert c0 + c1 >= 24  # all 24 prompts generated (a transient failover retry could add a few)


def _run_pipelined(h, *, batches, n, train_time, prefetch=2) -> float:
    """Time a trainer that consumes experiences and 'trains' (sleeps) per batch."""
    with _client(h) as c:
        c.register(uri="/lora/run/v0", replicas=1)
        t0 = time.monotonic()
        for _exp in c.experience_stream(batches, n=n, prefetch=prefetch, max_concurrency=8):
            time.sleep(train_time)  # the optimizer step on the trainer GPU
        return time.monotonic() - t0


def _run_serial(h, *, batches, n, train_time) -> float:
    """Baseline: generate, then score, then train — no overlap (reward latency is on the path)."""
    with _client(h) as c:
        c.register(uri="/lora/run/v0", replicas=1)
        t0 = time.monotonic()
        for batch in batches:
            comps = [c.generate(p, n=n) for p in batch]
            for p, group in zip(batch, comps, strict=True):
                c.score([p] * len(group), group)
            time.sleep(train_time)
        return time.monotonic() - t0


def test_pipelining_overlaps_rollout_with_training():
    # Producer (gen+score) and consumer (train) are comparable -> pipelining ~halves wall-clock.
    h = build_harness([{"id": "gpu0", "base_model": "Q", "latency": 0.05}], reward_workers=1, reward_latency=0.05)
    try:
        batches = [[f"b{j}"] for j in range(6)]
        pipelined = _run_pipelined(h, batches=batches, n=1, train_time=0.10)
        serial = _run_serial(h, batches=batches, n=1, train_time=0.10)
        assert pipelined < serial * 0.8, f"pipelined={pipelined:.3f} serial={serial:.3f}"
    finally:
        h.stop()


def test_reward_latency_does_not_cost_trainer_time():
    # THE point: a SLOW reward must barely change the trainer's wall-clock, because it overlaps the
    # optimizer step on the reward workers instead of blocking the trainer GPU. With overlap, going
    # from a fast (0.02s) to a slow (0.30s) reward adds almost nothing per step; serially it would
    # add ~0.28s * num_batches.
    batches = [[f"b{j}"] for j in range(5)]
    fast = build_harness([{"id": "g", "base_model": "Q", "latency": 0.02}], reward_workers=1, reward_latency=0.02)
    slow = build_harness([{"id": "g", "base_model": "Q", "latency": 0.02}], reward_workers=1, reward_latency=0.30)
    try:
        train_time = 0.40  # trainer is the bottleneck, so reward fully hides behind it
        pipe_fast = _run_pipelined(fast, batches=batches, n=1, train_time=train_time)
        pipe_slow = _run_pipelined(slow, batches=batches, n=1, train_time=train_time)
        ser_fast = _run_serial(fast, batches=batches, n=1, train_time=train_time)
        ser_slow = _run_serial(slow, batches=batches, n=1, train_time=train_time)
        pipelined_reward_cost = pipe_slow - pipe_fast
        serial_reward_cost = ser_slow - ser_fast
        # Overlap makes the slow reward almost free vs. paying it on the critical path serially.
        assert pipelined_reward_cost < serial_reward_cost * 0.5, (
            f"pipelined reward cost={pipelined_reward_cost:.3f} "
            f"serial reward cost={serial_reward_cost:.3f}"
        )
    finally:
        fast.stop()
        slow.stop()


def _producer_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "flash-rollout-producer" and t.is_alive()]


def test_experience_stream_producer_stops_on_early_break():
    # The producer must observe the consumer stopping early and wind down — not keep generating
    # ahead forever and then block on a full prefetch queue (leaking the thread / wasting rollouts).
    h = build_harness([{"id": "gpu0", "base_model": "Q", "latency": 0.02}], reward_workers=1)
    try:
        with _client(h) as c:
            c.register(uri="/lora/run/v0", replicas=1)
            many = [[f"p{i}"] for i in range(200)]  # far more than we'll consume
            stream = c.experience_stream(many, n=1, prefetch=2, max_concurrency=4)
            first = next(stream)  # consume exactly one, then abandon
            assert first.step == 0
            stream.close()  # consumer goes away (mirrors a `break` out of the for-loop)
        # the producer thread must be gone (joined) shortly after close
        deadline = time.monotonic() + 5.0
        while _producer_threads() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _producer_threads(), "producer thread still alive after consumer stopped"
        # and it stopped EARLY: it did not generate all 200 batches' worth of rollouts
        total_chat = len(h.record("gpu0").chat_calls)
        assert total_chat < 50, f"producer kept generating after stop: {total_chat} chat calls"
    finally:
        h.stop()


def test_generate_rejects_short_completion_group():
    # GRPO needs exactly n completions per prompt; a backend returning fewer must raise, not silently
    # pass a short group into advantage normalization.
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(body: dict) -> dict:
        # Always return ONE choice regardless of the requested n.
        return {"object": "chat.completion", "choices": [{"message": {"content": "only-one"}}]}

    srv = start_server(app)
    try:
        c = RolloutPoolClient(srv.url, adapter="run", base_model="Q")
        assert c.generate("hi", n=1) == ["only-one"]  # n matches -> fine
        with pytest.raises(ShortCompletionGroupError):
            c.generate("hi", n=4)  # asked for 4, got 1 -> error
        c.close()
    finally:
        srv.stop()
