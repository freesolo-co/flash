from __future__ import annotations

import asyncio
import hashlib
import inspect
import sys
import types
from types import SimpleNamespace

import pytest

from flash.engine.worker.train.rl.child import patches
from flash.engine.worker.train.rl.identity import RolloutIdentityLedger, parse_rollout_identity
from flash.engine.worker.train.rl.reward_module import render_reward_module


def _identity(step: int, index: int, ordinal: int, *, validate: bool = False) -> dict:
    return {
        "optimizer_step": step,
        "sample_index": index,
        "rollout_ordinal": ordinal,
        "validate": validate,
    }


def _expected(step: int, indexes: tuple[int, ...], group_size: int) -> list[dict]:
    return [_identity(step, index, ordinal) for index in indexes for ordinal in range(group_size)]


def test_identity_ledger_accepts_resumed_step_and_exact_out_of_order_set():
    ledger = RolloutIdentityLedger(2, 2)
    expected = _expected(17, (10, 11), 2)
    ledger.register(expected)
    for identity in (expected[3], expected[0], expected[2], expected[1]):
        ledger.record(identity, identity["sample_index"])
    ledger.seal(17)
    ledger.assert_idle()


def test_identity_ledger_rejects_whole_prompt_substitution_with_constant_count():
    ledger = RolloutIdentityLedger(2, 2)
    ledger.register(_expected(3, (10, 11), 2))
    ledger.record(_identity(3, 10, 0), 10)
    ledger.record(_identity(3, 10, 1), 10)
    with pytest.raises(ValueError, match="not registered"):
        ledger.record(_identity(3, 99, 0), 99)
    with pytest.raises(ValueError, match="does not equal registration"):
        ledger.seal(3)


def test_identity_ledger_rejects_validate_true_at_positive_step():
    ledger = RolloutIdentityLedger(1, 2)
    with pytest.raises(ValueError, match="validate=true"):
        ledger.register(
            [
                _identity(8, 0, 0, validate=True),
                _identity(8, 0, 1, validate=True),
            ]
        )


def test_identity_ledger_rejects_duplicate_one_drop_one_with_constant_count():
    ledger = RolloutIdentityLedger(2, 2)
    expected = _expected(1, (0, 1), 2)
    ledger.register(expected)
    for identity in expected[:3]:
        ledger.record(identity, identity["sample_index"])
    with pytest.raises(ValueError, match="duplicate"):
        ledger.record(expected[2], expected[2]["sample_index"])
    with pytest.raises(ValueError, match="does not equal registration"):
        ledger.seal(1)


def test_identity_ledger_rejects_duplicate_and_conflicting_registration():
    ledger = RolloutIdentityLedger(2, 2)
    expected = _expected(5, (10, 11), 2)
    ledger.register(expected)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.register(expected)
    with pytest.raises(ValueError, match="conflicting"):
        ledger.register(_expected(5, (10, 99), 2))


def test_identity_ledger_rejects_reward_before_registration():
    ledger = RolloutIdentityLedger(1, 2)
    with pytest.raises(ValueError, match="before identity registration"):
        ledger.record(_identity(23, 0, 0), 0)


def test_identity_ledger_allows_noncontiguous_steps_and_rejects_late_prior_step():
    ledger = RolloutIdentityLedger(1, 2)
    first = _expected(40, (3,), 2)
    ledger.register(first)
    for identity in reversed(first):
        ledger.record(identity, 3)
    ledger.seal(40)

    resumed = _expected(91, (7,), 2)
    ledger.register(resumed)
    with pytest.raises(ValueError, match="late"):
        ledger.record(first[0], 3)
    for identity in resumed:
        ledger.record(identity, 7)
    ledger.seal(91)
    ledger.assert_idle()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "missing"),
        ({"optimizer_step": 1, "sample_index": 0, "rollout_ordinal": 0}, "field set"),
        (_identity(0, 0, 0), "must be positive"),
        ({**_identity(1, 0, 0), "validate": 0}, "must be a boolean"),
    ],
)
def test_identity_parser_rejects_malformed_identity(value, message):
    with pytest.raises(ValueError, match=message):
        parse_rollout_identity(value)


def _fake_agent_loop_module(original_run, worker_generate, manager_generate, trajectory_info):
    agent_loop = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop.asyncio = asyncio
    agent_loop.auto_await = lambda function: function
    agent_loop.get_trajectory_info = trajectory_info
    agent_loop.AgentLoopWorker = type(
        "AgentLoopWorker",
        (),
        {
            "_run_agent_loop": original_run,
            "generate_sequences": worker_generate,
        },
    )
    agent_loop.AgentLoopManager = type(
        "AgentLoopManager",
        (),
        {"generate_sequences": manager_generate},
    )
    experimental = types.ModuleType("verl.experimental")
    package = types.ModuleType("verl.experimental.agent_loop")
    package.agent_loop = agent_loop
    verl = types.ModuleType("verl")
    verl.experimental = experimental
    experimental.agent_loop = package
    return verl, experimental, package, agent_loop


def _install_fake_modules(monkeypatch, modules):
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


def _pin_fake_sources(monkeypatch, sources):
    monkeypatch.setattr(inspect, "getsource", lambda function: sources[function])
    monkeypatch.setattr(
        patches,
        "_PINNED_RUN_AGENT_LOOP_SHA256",
        hashlib.sha256(sources[next(iter(sources))].encode()).hexdigest(),
    )


def test_manager_registers_exact_set_before_dispatch_and_sidecars_global_ordinals(monkeypatch):
    events = []
    ledger = RolloutIdentityLedger(2, 2)

    async def trajectory_info(step, indexes, validate):
        ordinals: dict[int, int] = {}
        result = []
        for index in indexes:
            ordinal = ordinals.get(index, 0)
            ordinals[index] = ordinal + 1
            result.append(
                {
                    "step": step,
                    "sample_index": index,
                    "rollout_n": ordinal,
                    "validate": validate,
                }
            )
        return result

    async def original_run(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        identity = kwargs["flash_rollout_identity"]
        events.append(("reward", dict(identity)))
        ledger.record(identity, kwargs["extra_info"]["index"])
        return identity

    async def worker_generate(self, batch):
        trajectories = await modules[-1].get_trajectory_info(
            batch.meta_info["global_steps"],
            batch.non_tensor_batch["index"].tolist(),
            batch.meta_info["validate"],
        )
        values = []
        for trajectory, index in zip(
            trajectories, batch.non_tensor_batch["index"].tolist(), strict=True
        ):
            values.append(
                await self._run_agent_loop(
                    {},
                    trajectory,
                    agent_name="single_turn_agent",
                    extra_info={"index": index},
                )
            )
        return SimpleNamespace(
            values=values,
            meta_info={
                "metrics": [
                    {
                        "generate_sequences": 0.0,
                        "tool_calls": 0.0,
                        "compute_score": 0.0,
                        "num_preempted": 0,
                    }
                    for _value in values
                ]
            },
        )

    async def manager_generate(self, prompts):
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        return await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )

    modules = _fake_agent_loop_module(
        original_run,
        worker_generate,
        manager_generate,
        trajectory_info,
    )
    _install_fake_modules(monkeypatch, modules)

    run_source = 'trajectory["rollout_n"]\nself._agent_loop_postprocess'
    worker_source = "trajectory_info = await get_trajectory_info\nself._run_agent_loop("
    manager_source = "chunkes = prompts.chunk\nworker.generate_sequences.remote(chunk)"
    sources = {
        original_run: run_source,
        worker_generate: worker_source,
        manager_generate: manager_source,
    }
    monkeypatch.setattr(inspect, "getsource", lambda function: sources[function])
    monkeypatch.setattr(
        patches,
        "_PINNED_RUN_AGENT_LOOP_SHA256",
        hashlib.sha256(run_source.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        patches,
        "_PINNED_WORKER_GENERATE_SHA256",
        hashlib.sha256(worker_source.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        patches,
        "_PINNED_MANAGER_GENERATE_SHA256",
        hashlib.sha256(manager_source.encode()).hexdigest(),
    )

    bridge = types.ModuleType("flash_grpo_multiturn")

    def post_json(_url, path, payload, *, error_style):
        assert path == "/identity/register"
        assert error_style == "reward"
        events.append(("register", [dict(value) for value in payload["identities"]]))
        step = ledger.register(payload["identities"])
        return {"optimizer_step": step, "registered": len(payload["identities"])}

    bridge.post_json = post_json
    monkeypatch.setitem(sys.modules, "flash_grpo_multiturn", bridge)
    monkeypatch.setenv("FLASH_VERL_REWARD_URL", "http://127.0.0.1:1")

    class Indexes(list):
        def tolist(self):
            return list(self)

    class Batch:
        def __init__(self, indexes):
            self.non_tensor_batch = {"index": Indexes(indexes)}
            self.meta_info = {"global_steps": 17, "validate": False}

        def __len__(self):
            return len(self.non_tensor_batch["index"])

        def chunk(self, count):
            assert count == len(self)
            return [Batch([index]) for index in self.non_tensor_batch["index"]]

    class Combined:
        def __init__(self, outputs):
            self.values = [value for output in outputs for value in output.values]
            self.meta_info = {}

    modules[-1].DataProto = SimpleNamespace(concat=lambda outputs: Combined(outputs))
    patches.install_exact_rollout_identity()

    workers = []
    for _index in range(4):
        worker = modules[-1].AgentLoopWorker()
        remote = SimpleNamespace(
            remote=lambda batch, identities, worker=worker: (
                worker.generate_sequences_with_flash_identities(batch, identities)
            )
        )
        workers.append(SimpleNamespace(generate_sequences_with_flash_identities=remote))
    manager = modules[-1].AgentLoopManager()
    manager.agent_loop_workers = workers
    manager._performance_metrics = lambda metrics, output: {"rows": len(output.values)}

    output = asyncio.run(manager.generate_sequences(Batch([10, 10, 11, 11])))

    assert events[0][0] == "register"
    expected = _expected(17, (10, 11), 2)
    assert events[0][1] == expected
    assert [value for kind, value in events if kind == "reward"] == expected
    assert output.values == expected
    ledger.seal(17)
    ledger.assert_idle()


def test_exact_identity_patch_fails_closed_on_manager_boundary_drift(monkeypatch):
    async def original_run(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        return None

    async def get_trajectory_info(*_args):
        return []

    async def worker_generate(self, batch):
        trajectory_info = await get_trajectory_info(batch)
        return self._run_agent_loop(trajectory_info)

    worker = SimpleNamespace(generate_sequences=SimpleNamespace(remote=lambda _chunk: None))

    async def manager_generate(self, prompts):
        chunkes = prompts.chunk(1)
        return worker.generate_sequences.remote(chunkes[0])

    modules = _fake_agent_loop_module(
        original_run,
        worker_generate,
        manager_generate,
        get_trajectory_info,
    )
    _install_fake_modules(monkeypatch, modules)
    bridge = types.ModuleType("flash_grpo_multiturn")
    bridge.post_json = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "flash_grpo_multiturn", bridge)
    monkeypatch.setattr(inspect, "getsource", lambda _function: "drifted")
    with pytest.raises(RuntimeError, match="boundary drifted"):
        patches.install_exact_rollout_identity()


def test_single_turn_reward_server_registration_and_scoring_use_same_identity(monkeypatch):
    from flash.engine.worker.train.rl.child.multiturn import post_json
    from flash.engine.worker.train.rl.multi_turn import start_reward_server

    ledger = RolloutIdentityLedger(1, 2)
    server, url = start_reward_server(
        lambda index, solution: float(index),
        example_count=1,
        identity_ledger=ledger,
    )
    module = types.ModuleType("flash_grpo_multiturn")
    module.post_json = post_json
    monkeypatch.setitem(sys.modules, "flash_grpo_multiturn", module)
    monkeypatch.setenv("FLASH_VERL_REWARD_URL", url)
    namespace: dict = {}
    exec(render_reward_module(), namespace)
    expected = _expected(6, (0,), 2)
    try:
        registered = post_json(
            url,
            "/identity/register",
            {"identities": expected},
            error_style="reward",
        )
        assert registered == {"optimizer_step": 6, "registered": 2}
        for identity in expected:
            assert (
                namespace["compute_score"](
                    "flash",
                    "answer",
                    "",
                    {"index": 0, "flash_rollout_identity": identity},
                )
                == 0.0
            )
        ledger.seal(6)
        ledger.assert_idle()
    finally:
        server.shutdown()


def test_multi_turn_bridge_carries_and_records_the_registered_identity():
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
    expected = _expected(4, (0,), 2)
    ledger.register(expected)
    bridge = MultiTurnBridge(
        Env(),
        [{"id": 0}],
        env_prompts=[[{"role": "user", "content": "x"}]],
        max_turns=1,
        identity_ledger=ledger,
        score_flush_wait_s=0.001,
    )
    try:
        for identity in expected:
            session_id = f"session-{identity['rollout_ordinal']}"
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
