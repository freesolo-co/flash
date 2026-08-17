from __future__ import annotations

import asyncio
import inspect
import sys
import types

import pytest

from flash.engine.worker.train.rl.child import patches
from flash.engine.worker.train.rl.identity import RolloutIdentityLedger
from flash.engine.worker.train.rl.reward_module import render_reward_module


def _identity(step: int, index: int, ordinal: int) -> dict[str, int]:
    return {
        "optimizer_step": step,
        "sample_index": index,
        "rollout_ordinal": ordinal,
    }


def test_identity_ledger_seals_exact_prompt_and_ordinal_set():
    ledger = RolloutIdentityLedger(2, 4)
    for index in (7, 11):
        for ordinal in range(4):
            ledger.record(_identity(3, index, ordinal), index)
    ledger.seal(3)
    ledger.assert_idle()


def test_identity_ledger_rejects_duplicate_one_drop_one_with_constant_count():
    ledger = RolloutIdentityLedger(2, 2)
    arrivals = [
        (_identity(1, 0, 0), 0),
        (_identity(1, 0, 1), 0),
        (_identity(1, 1, 0), 1),
        (_identity(1, 1, 0), 1),
    ]
    for value, index in arrivals[:-1]:
        ledger.record(value, index)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.record(*arrivals[-1])
    with pytest.raises(ValueError, match="expected exactly 4"):
        ledger.seal(1)


@pytest.mark.parametrize(
    ("value", "index", "message"),
    [
        (None, 0, "missing"),
        ({"optimizer_step": 1, "sample_index": 0}, 0, "field set"),
        (_identity(0, 0, 0), 0, "must be positive"),
        (_identity(1, 2, 0), 1, "does not match"),
        (_identity(1, 0, 2), 0, "outside"),
    ],
)
def test_identity_ledger_rejects_malformed_or_out_of_range_identity(value, index, message):
    ledger = RolloutIdentityLedger(1, 2)
    with pytest.raises(ValueError, match=message):
        ledger.record(value, index)


def test_identity_ledger_rejects_cross_step_and_late_identity():
    ledger = RolloutIdentityLedger(1, 2)
    ledger.record(_identity(1, 0, 0), 0)
    with pytest.raises(ValueError, match="cross-step"):
        ledger.record(_identity(2, 0, 0), 0)
    ledger.record(_identity(1, 0, 1), 0)
    ledger.seal(1)
    with pytest.raises(ValueError, match="late"):
        ledger.record(_identity(1, 0, 0), 0)


def _fake_agent_loop_module(original):
    agent_loop = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop.AgentLoopWorker = type("AgentLoopWorker", (), {"_run_agent_loop": original})
    experimental = types.ModuleType("verl.experimental")
    package = types.ModuleType("verl.experimental.agent_loop")
    package.agent_loop = agent_loop
    verl = types.ModuleType("verl")
    verl.experimental = experimental
    experimental.agent_loop = package
    return verl, experimental, package, agent_loop


def test_exact_identity_patch_copies_kwargs_without_mutating_inputs(monkeypatch):
    calls = []

    async def original(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        calls.append((sampling_params, trajectory, agent_name, trace, kwargs))
        return kwargs

    modules = _fake_agent_loop_module(original)
    for name, module in zip(
        (
            "verl",
            "verl.experimental",
            "verl.experimental.agent_loop",
            "verl.experimental.agent_loop.agent_loop",
        ),
        modules,
        strict=True,
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(inspect, "getsource", lambda _fn: "pinned-boundary")
    monkeypatch.setattr(
        patches,
        "_PINNED_RUN_AGENT_LOOP_SHA256",
        __import__("hashlib").sha256(b"pinned-boundary").hexdigest(),
    )

    patches.install_exact_rollout_identity()
    worker = modules[-1].AgentLoopWorker()
    sampling = {"temperature": 1.0}
    trajectory = {"step": 5, "sample_index": 9, "rollout_n": 1}
    extra_info = {"index": 9, "split": "train"}
    kwargs = {"extra_info": extra_info, "raw_prompt": [{"role": "user", "content": "x"}]}
    result = asyncio.run(
        worker._run_agent_loop(
            sampling,
            trajectory,
            agent_name="single_turn_agent",
            **kwargs,
        )
    )

    expected = _identity(5, 9, 1)
    assert result["flash_rollout_identity"] == expected
    assert result["extra_info"]["flash_rollout_identity"] == expected
    assert sampling == {"temperature": 1.0}
    assert trajectory == {"step": 5, "sample_index": 9, "rollout_n": 1}
    assert extra_info == {"index": 9, "split": "train"}
    assert kwargs == {"extra_info": extra_info, "raw_prompt": [{"role": "user", "content": "x"}]}
    assert calls[0][2:4] == ("single_turn_agent", True)


def test_exact_identity_patch_fails_closed_on_boundary_drift(monkeypatch):
    async def original(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        return None

    modules = _fake_agent_loop_module(original)
    for name, module in zip(
        (
            "verl",
            "verl.experimental",
            "verl.experimental.agent_loop",
            "verl.experimental.agent_loop.agent_loop",
        ),
        modules,
        strict=True,
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(inspect, "getsource", lambda _fn: "drifted")
    with pytest.raises(RuntimeError, match="boundary drifted"):
        patches.install_exact_rollout_identity()


def test_single_turn_parent_bridge_records_exact_identities(monkeypatch):
    from flash.engine.worker.train.rl.multi_turn import start_reward_server

    ledger = RolloutIdentityLedger(1, 2)
    server, url = start_reward_server(
        lambda index, solution: float(index),
        example_count=1,
        identity_ledger=ledger,
    )
    module = types.ModuleType("flash_grpo_multiturn")

    def post_json(_url, path, payload, *, error_style):
        import json
        import urllib.request

        request = urllib.request.Request(
            url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode())

    module.post_json = post_json
    monkeypatch.setitem(sys.modules, "flash_grpo_multiturn", module)
    monkeypatch.setenv("FLASH_VERL_REWARD_URL", url)
    namespace: dict = {}
    exec(render_reward_module(), namespace)
    try:
        for ordinal in range(2):
            assert (
                namespace["compute_score"](
                    "flash",
                    "answer",
                    "",
                    {
                        "index": 0,
                        "flash_rollout_identity": _identity(6, 0, ordinal),
                    },
                )
                == 0.0
            )
        ledger.seal(6)
        ledger.assert_idle()
    finally:
        server.shutdown()


def test_multi_turn_bridge_carries_and_records_the_same_identity():
    from flash.engine.worker.train.rl.multi_turn import MultiTurnBridge
    from flash.envs.base import RolloutReward

    class Env:
        def new_rollout_state(self, example, prompt):
            return {"prompt": list(prompt), "messages": list(prompt)}

        def record_model_turn(self, state, text):
            state["messages"].append({"role": "assistant", "content": text})

        def rollout_done(self, state, max_turns=None):
            return True

        def env_reply(self, messages, state):
            return []

        def rollout_rewards_many(self, items):
            return [RolloutReward(episode=1.0) for _item in items]

    ledger = RolloutIdentityLedger(1, 2)
    bridge = MultiTurnBridge(
        Env(),
        [{"id": 0}],
        env_prompts=[[{"role": "user", "content": "x"}]],
        max_turns=1,
        identity_ledger=ledger,
        score_flush_wait_s=0.001,
    )
    try:
        for ordinal in range(2):
            identity = _identity(4, 0, ordinal)
            session_id = f"session-{ordinal}"
            bridge.start({"index": 0, "session_id": session_id, "identity": identity})
            bridge.step({"session_id": session_id, "completion_text": "answer"})
            assert (
                bridge.score(
                    {
                        "session_id": session_id,
                        "turn_count": 1,
                        "identity": identity,
                    }
                )["score"]
                == 1.0
            )
            bridge.close({"session_id": session_id})
        ledger.seal(4)
        ledger.assert_idle()
    finally:
        bridge.shutdown()


def test_single_turn_reward_bridge_carries_identity_without_rewriting_it(monkeypatch):
    posted = []
    module = types.ModuleType("flash_grpo_multiturn")

    def post_json(url, path, payload, *, error_style):
        posted.append((url, path, payload, error_style))
        return {"score": 0.5}

    module.post_json = post_json
    monkeypatch.setitem(sys.modules, "flash_grpo_multiturn", module)
    monkeypatch.setenv("FLASH_VERL_REWARD_URL", "http://127.0.0.1:1")
    namespace: dict = {}
    exec(render_reward_module(), namespace)
    identity = _identity(2, 3, 1)
    score = namespace["compute_score"](
        "flash",
        "answer",
        "",
        {"index": 3, "flash_rollout_identity": identity},
    )
    assert score == 0.5
    assert posted[0][2]["identity"] == identity
    assert identity == _identity(2, 3, 1)
