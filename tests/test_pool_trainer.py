"""The distributed GRPO loop end-to-end: rollout + reward run on the pool servers, the (stubbed)
GPU policy step runs locally, and weights sync to the pool each step."""

from __future__ import annotations

import threading
import time

import pytest

from flash.engine.pool_trainer import (
    GRPOPoolLoop,
    _cycle_prompts_to,
    compute_group_advantages,
)
from flash.pool.client import RolloutPoolClient
from tests._helpers.pool_harness import build_harness


def _producer_threads_alive() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "flash-rollout-producer" and t.is_alive()]


def test_compute_group_advantages_zero_mean_unit_std():
    adv = compute_group_advantages([[0.0, 1.0]])
    # group mean 0.5, std 0.5 -> advantages -1, +1
    assert adv[0][0] == pytest.approx(-1.0, abs=1e-3)
    assert adv[0][1] == pytest.approx(1.0, abs=1e-3)


def test_compute_group_advantages_constant_group_is_zero():
    adv = compute_group_advantages([[0.7, 0.7, 0.7]])
    assert all(abs(a) < 1e-3 for a in adv[0])


def test_compute_group_advantages_handles_empty():
    assert compute_group_advantages([[]]) == [[]]


def test_cycle_prompts_to_repeats_in_order_without_doubling():
    """The prompt cap cycles the base list to EXACTLY `need`, in the base's repeated order —
    not the old `prompts += prompts` power-of-two doubling that over-allocated and could blow
    memory for a large step count over a tiny prompt set."""
    base = ["a", "b", "c"]
    # need not a multiple of len(base): exactly `need` items, cycled in order, no overshoot.
    assert _cycle_prompts_to(base, 7) == ["a", "b", "c", "a", "b", "c", "a"]
    # need < len(base): truncates to need (the doubling loop never even ran here, but lock it).
    assert _cycle_prompts_to(base, 2) == ["a", "b"]
    # need == len(base): the base list unchanged.
    assert _cycle_prompts_to(base, 3) == ["a", "b", "c"]
    # A large need over a tiny base yields exactly `need` (doubling would over-allocate to the
    # next power of two; islice stops precisely at need).
    big = _cycle_prompts_to(["x", "y"], 1000)
    assert len(big) == 1000
    assert big[:4] == ["x", "y", "x", "y"]
    assert big[-1] == "y"  # 1000 even -> last is the 2nd base item


def test_cycle_prompts_to_empty_base_is_safe():
    """An empty base (or need<=0) must NOT make itertools.cycle spin forever — it returns []."""
    assert _cycle_prompts_to([], 5) == []
    assert _cycle_prompts_to(["a"], 0) == []
    assert _cycle_prompts_to(["a"], -3) == []


def test_distributed_grpo_loop_runs_off_gpu():
    h = build_harness(
        [{"id": "gpu0", "base_model": "Q"}, {"id": "gpu1", "base_model": "Q"}],
        reward_workers=2,
    )
    try:
        client = RolloutPoolClient(h.router_url, adapter="run", base_model="Q")
        loop = GRPOPoolLoop(client, group_size=4, prefetch=2)
        loop.register(uri="/lora/run/v0", replicas=2)

        update_calls: list[int] = []

        def fake_policy_update(exp, advantages):
            # the GPU step: here a stub that just versions the adapter dir
            assert exp.num_completions == exp.num_prompts * 4
            assert len(advantages) == exp.num_prompts
            update_calls.append(exp.step)
            return f"/lora/run/v{exp.step + 1}"

        prompt_batches = [["p1", "p2"], ["p3", "p4"], ["p5", "p6"]]
        results = loop.run(prompt_batches, fake_policy_update)

        assert len(results) == 3
        assert update_calls == [0, 1, 2]
        # weights synced each step -> adapter version advanced to 3
        assert [r.adapter_version for r in results] == [1, 2, 3]
        # rollout actually happened on the pool GPUs (off the trainer): one group-call per prompt
        # per step, at minimum (a transient failover retry could add a few — that's fine).
        total_chat = sum(len(h.record(b).chat_calls) for b in ("gpu0", "gpu1"))
        assert total_chat >= 3 * 2  # 3 steps * 2 prompts
        client.close()
    finally:
        h.stop()


def test_pool_policy_glue_helpers_import_without_torch():
    # The worker-entry helpers must import + work without the GPU stack (torch is imported lazily
    # only inside the policy builder), so the control plane stays torch-free.
    from flash.engine.pool_policy import batched, prompts_from_env

    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    class _Env:
        dataset = ({"prompt": "p1"}, {"question": "p2"}, {"text": "p3"})

    assert prompts_from_env(_Env()) == ["p1", "p2", "p3"]
    assert prompts_from_env(_Env(), limit=2) == ["p1", "p2"]
    assert prompts_from_env(object()) == []  # no dataset -> empty, no crash


def test_loop_respects_step_cap():
    h = build_harness([{"id": "gpu0", "base_model": "Q"}], reward_workers=1)
    try:
        client = RolloutPoolClient(h.router_url, adapter="run", base_model="Q")
        loop = GRPOPoolLoop(client, group_size=2)
        loop.register(uri="/lora/run/v0")
        # infinite prompt source; the cap stops it
        def prompts():
            i = 0
            while True:
                yield [f"p{i}"]
                i += 1

        results = loop.run(prompts(), lambda exp, adv: None, steps=2)
        assert len(results) == 2
        client.close()
    finally:
        h.stop()


def test_loop_closes_stream_and_joins_producer_on_early_break():
    # The ``steps`` cap breaks the consumption loop EARLY (before the infinite prompt source is
    # exhausted). ``run()`` must close the experience_stream generator so its background producer
    # thread (and the rollout ThreadPoolExecutor) is torn down — otherwise the producer leaks and
    # keeps generating/scoring rollouts (a live thread + wasted pool spend) after run() returns.
    before = {t.ident for t in _producer_threads_alive()}
    h = build_harness([{"id": "gpu0", "base_model": "Q"}], reward_workers=1)
    try:
        client = RolloutPoolClient(h.router_url, adapter="run", base_model="Q")
        loop = GRPOPoolLoop(client, group_size=2, prefetch=2)
        loop.register(uri="/lora/run/v0")

        def prompts():  # never runs out; only the steps cap stops the loop
            i = 0
            while True:
                yield [f"p{i}"]
                i += 1

        results = loop.run(prompts(), lambda exp, adv: f"/lora/run/v{exp.step + 1}", steps=2)
        assert len(results) == 2

        # The producer thread this run started must be gone (joined) by the time run() returns —
        # contextlib.closing(...) ran the generator's finally on the early break. Give a brief grace
        # window since join() has a (generous) timeout in the generator's finally.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            leaked = [t for t in _producer_threads_alive() if t.ident not in before]
            if not leaked:
                break
            time.sleep(0.05)
        leaked = [t for t in _producer_threads_alive() if t.ident not in before]
        assert leaked == [], f"producer thread leaked after early break: {leaked}"
        client.close()
    finally:
        h.stop()
