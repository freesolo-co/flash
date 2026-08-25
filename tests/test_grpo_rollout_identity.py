from __future__ import annotations

import asyncio
import contextvars
import hashlib
import inspect
import sys
import threading
import types
from types import SimpleNamespace

import cloudpickle
import pytest

from flash.engine.worker.train.rl.child import patches
from flash.engine.worker.train.rl.rollout.identity import (
    RolloutIdentityLedger,
    parse_rollout_identity,
)
from flash.engine.worker.train.rl.rollout.reward_module import render_reward_module


def _identity(step: int, index: int, ordinal: int, *, validate: bool = False) -> dict:
    return {
        "optimizer_step": step,
        "sample_index": index,
        "rollout_ordinal": ordinal,
        "validate": validate,
    }


def _expected(step: int, indexes: tuple[int, ...], group_size: int) -> list[dict]:
    return [_identity(step, index, ordinal) for index in indexes for ordinal in range(group_size)]


def _identity_sha256(identities: list[dict]) -> str:
    digest = hashlib.sha256()
    for identity in sorted(
        identities,
        key=lambda value: (
            value["optimizer_step"],
            value["sample_index"],
            value["rollout_ordinal"],
            value["validate"],
        ),
    ):
        digest.update(
            (
                f"{identity['optimizer_step']}:{identity['sample_index']}:"
                f"{identity['rollout_ordinal']}:{int(identity['validate'])}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


async def get_trajectory_info(*_args):
    raise AssertionError("test must replace the pinned Verl trajectory helper")


def test_identity_ledger_accepts_resumed_step_and_exact_out_of_order_set():
    ledger = RolloutIdentityLedger(2, 2)
    expected = _expected(17, (10, 11), 2)
    ledger.register(expected)
    for identity in (expected[3], expected[0], expected[2], expected[1]):
        ledger.record(identity, identity["sample_index"])
    ledger.seal(17)
    ledger.finalize({17})


def test_identity_ledger_summarizes_p64_g2_terminal_evidence_and_returns_deep_copies():
    ledger = RolloutIdentityLedger(64, 2)
    exact_by_step = {}
    for step in (1, 2):
        expected = _expected(step, tuple(range(64)), 2)
        exact_by_step[step] = expected
        ledger.register(reversed(expected))
        for identity in reversed(expected):
            ledger.record(identity, identity["sample_index"])
        ledger.seal(step)

    evidence = ledger.finalize({1, 2})
    # both sides are digested, and each digest is checked against the identities the test built
    # itself rather than against the other side of the payload. a payload that simply copied one
    # digest into the other field would satisfy an internal comparison while proving nothing.
    assert evidence == {
        "steps": [
            {
                "optimizer_step": step,
                "registered": {
                    "count": len(exact_by_step[step]),
                    "sha256": _identity_sha256(exact_by_step[step]),
                },
                "observed": {
                    "count": len(exact_by_step[step]),
                    "sha256": _identity_sha256(exact_by_step[step]),
                },
            }
            for step in (1, 2)
        ],
        "validation": [],
    }
    for step in evidence["steps"]:
        assert step["registered"]["count"] == 128
        assert step["observed"]["count"] == 128
        assert step["registered"]["sha256"] == step["observed"]["sha256"]

    # the returned payload must be a copy: mutating it cannot reach into the ledger.
    evidence["steps"][0]["registered"]["count"] = 999
    evidence["steps"][0]["observed"]["count"] = 0
    evidence["steps"].append({"optimizer_step": 999, "registered": {}, "observed": {}})
    fresh = ledger.evidence()
    assert [step["optimizer_step"] for step in fresh["steps"]] == [1, 2]
    assert fresh["steps"][0]["registered"]["count"] == 128
    assert fresh["steps"][0]["observed"]["count"] == 128


def test_identity_evidence_summaries_agree_across_both_sides():
    ledger = RolloutIdentityLedger(2, 2)
    expected = _expected(7, (4, 9), 2)
    ledger.register(reversed(expected))
    for identity in expected:
        ledger.record(identity, identity["sample_index"])
    ledger.seal(7)

    step = ledger.finalize({7})["steps"][0]
    assert step["registered"]["count"] == len(expected)
    assert step["observed"]["count"] == len(expected)
    assert step["registered"]["sha256"] == _identity_sha256(expected)
    assert step["observed"]["sha256"] == _identity_sha256(expected)


def test_finalize_evidence_is_constant_size_per_step():
    """Evidence must not grow with the completions it summarizes.

    Asserted as a size relation between two ledgers whose per-step work differs by 8x, rather than
    against a hardcoded byte count: a fixed number would pass for any payload that happens to be
    small today, including one that still embeds every identity at a small group size.

    The unbounded form measured 43 KB of json per step at the documented 512-completions-per-step
    envelope, so a 10,000-step run wrote ~434 MB into `train_meta.json` -- doubled to ~867 MB by
    the deepcopy in verl_config -- during `finalize()`, after paid training had already succeeded.
    """
    import json

    def evidence_for(prompts_per_step, group_size, steps):
        ledger = RolloutIdentityLedger(prompts_per_step, group_size)
        for step in range(1, steps + 1):
            expected = _expected(step, tuple(range(prompts_per_step)), group_size)
            ledger.register(expected)
            for identity in expected:
                ledger.record(identity, identity["sample_index"])
            ledger.seal(step)
        return ledger.finalize(set(range(1, steps + 1)))

    small = json.dumps(evidence_for(8, 2, 2))  # 16 completions per step
    large = json.dumps(evidence_for(64, 8, 2))  # 512 completions per step, 32x the work
    # not asserted equal: the counts themselves are rendered into the json, so "512" is three bytes
    # wider than "16". a handful of bytes for 32x the completions is the property under test; the
    # unbounded form differed by ~86 KB at these sizes.
    assert abs(len(large) - len(small)) < 32, (
        f"evidence size tracks completion count: {len(small)} vs {len(large)} bytes for the same "
        "step count with 16 vs 512 completions per step"
    )

    # and it grows only with the number of steps, linearly and by a small constant.
    four_steps = json.dumps(evidence_for(64, 8, 4))
    assert len(four_steps) < 2 * len(large) + 64


def test_failed_identity_seal_does_not_publish_terminal_evidence():
    ledger = RolloutIdentityLedger(1, 2)
    expected = _expected(3, (0,), 2)
    ledger.register(expected)
    ledger.record(expected[0], 0)
    with pytest.raises(ValueError, match="does not equal registration"):
        ledger.seal(3)
    assert ledger.evidence() == {"steps": [], "validation": []}


def test_identity_ledger_finalize_rejects_missing_and_extra_resume_steps():
    missing = RolloutIdentityLedger(1, 2)
    with pytest.raises(ValueError, match=r"missing=\[6, 7, 8, 9\], extra=\[\]"):
        missing.finalize(range(6, 10))

    extra = RolloutIdentityLedger(1, 2)
    for step in (6, 7, 8, 9, 99):
        expected = _expected(step, (0,), 2)
        extra.register(expected)
        for identity in expected:
            extra.record(identity, 0)
        extra.seal(step)
    with pytest.raises(ValueError, match=r"missing=\[\], extra=\[99\]"):
        extra.finalize(range(6, 10))


def test_identity_ledger_empty_finalize_is_atomic_and_permanently_rejects_mutation():
    ledger = RolloutIdentityLedger(1, 2)
    register_started = threading.Event()
    release_register = threading.Event()
    result = []

    def late_register():
        register_started.set()
        release_register.wait(timeout=1)
        try:
            ledger.register(_expected(1, (0,), 2))
        except ValueError as error:
            result.append(str(error))

    thread = threading.Thread(target=late_register)
    thread.start()
    assert register_started.wait(timeout=1)
    assert ledger.finalize(range(5, 5)) == {"steps": [], "validation": []}
    release_register.set()
    thread.join(timeout=1)
    assert result == ["GRPO rollout identity ledger is already finalized"]

    identity = _identity(1, 0, 0)
    for mutation in (
        lambda: ledger.require_registered(identity, 0),
        lambda: ledger.record(identity, 0),
        lambda: ledger.seal(1),
        lambda: ledger.finalize(()),
    ):
        with pytest.raises(ValueError, match="already finalized"):
            mutation()


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
    ledger.finalize({40, 91})


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
        trajectories = await get_trajectory_info(
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

    monkeypatch.setitem(worker_generate.__globals__, "get_trajectory_info", trajectory_info)
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
    ledger.finalize({17})


def test_patched_actor_serializes_and_concurrent_calls_keep_identity_task_local(monkeypatch):
    async def trajectory_info(step, indexes, validate):
        return [
            {
                "step": step,
                "sample_index": index,
                "rollout_n": ordinal,
                "validate": validate,
            }
            for ordinal, index in enumerate(indexes)
        ]

    async def original_run(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        assert sampling_params == {}
        assert agent_name == "single_turn_agent"
        assert trace is True
        assert kwargs["extra_info"]["flash_rollout_identity"] == kwargs["flash_rollout_identity"]
        return kwargs["flash_rollout_identity"]

    async def worker_generate(self, batch):
        await self.enter()
        trajectory_info = await get_trajectory_info(
            batch.meta_info["global_steps"],
            batch.non_tensor_batch["index"].tolist(),
            batch.meta_info["validate"],
        )
        return [
            await self._run_agent_loop(
                {},
                trajectory,
                agent_name="single_turn_agent",
                extra_info={"index": index},
            )
            for trajectory, index in zip(
                trajectory_info, batch.non_tensor_batch["index"].tolist(), strict=True
            )
        ]

    async def manager_generate(self, prompts):
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        return await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )

    monkeypatch.setitem(worker_generate.__globals__, "get_trajectory_info", trajectory_info)
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
    bridge.post_json = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "flash_grpo_multiturn", bridge)

    patches.install_exact_rollout_identity()
    actor_class = modules[-1].AgentLoopWorker
    cloudpickle.dumps(actor_class)
    closure = inspect.getclosurevars(actor_class.generate_sequences_with_flash_identities)
    assert set(closure.nonlocals) == {"original_worker_generate"}
    for value in closure.nonlocals.values():
        cloudpickle.dumps(value)

    bad_sidecar = contextvars.ContextVar("flash_rollout_identity_sidecar")

    async def nonserializable_capture():
        return bad_sidecar.get(None)

    with pytest.raises(TypeError, match="ContextVar"):
        cloudpickle.dumps(nonserializable_capture)

    class Indexes(list):
        def tolist(self):
            return list(self)

    class Batch:
        def __init__(self, step, index):
            self.non_tensor_batch = {"index": Indexes([index])}
            self.meta_info = {"global_steps": step, "validate": False}

    async def exercise_concurrent_calls():
        worker = actor_class()
        both_entered = asyncio.Event()
        release = asyncio.Event()
        entered = 0

        async def enter():
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await both_entered.wait()
            await release.wait()

        worker.enter = enter
        first_identity = _identity(12, 4, 7)
        second_identity = _identity(99, 8, 3)
        first = asyncio.create_task(
            worker.generate_sequences_with_flash_identities(Batch(12, 4), [first_identity])
        )
        second = asyncio.create_task(
            worker.generate_sequences_with_flash_identities(Batch(99, 8), [second_identity])
        )
        await both_entered.wait()
        release.set()
        outputs = await asyncio.gather(first, second)
        return outputs, first_identity, second_identity

    outputs, first_identity, second_identity = asyncio.run(exercise_concurrent_calls())
    assert outputs == [[first_identity], [second_identity]]

    async def proceed():
        return None

    worker = actor_class()
    worker.enter = proceed
    with pytest.raises(RuntimeError, match="length does not match"):
        asyncio.run(worker.generate_sequences_with_flash_identities(Batch(12, 4), []))
    with pytest.raises(RuntimeError, match="index mismatch"):
        asyncio.run(
            worker.generate_sequences_with_flash_identities(Batch(12, 4), [_identity(12, 8, 7)])
        )


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
    from flash.engine.worker.train.rl.rollout.multi_turn import start_reward_server

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
        ledger.finalize({6})
    finally:
        server.shutdown()


def test_multi_turn_bridge_carries_and_records_the_registered_identity():
    from flash.engine.worker.train.rl.rollout.multi_turn import MultiTurnBridge
    from flash.envs.loading.base import RolloutReward

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
        ledger.finalize({4})
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
