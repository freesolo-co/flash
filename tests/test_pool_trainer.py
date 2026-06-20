"""The distributed GRPO loop end-to-end: rollout + reward run on the pool servers, the (stubbed)
GPU policy step runs locally, and weights sync to the pool each step."""

from __future__ import annotations

import pytest

from flash.engine.pool_trainer import GRPOPoolLoop, compute_group_advantages
from flash.pool.client import RolloutPoolClient
from tests._helpers.pool_harness import build_harness


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
