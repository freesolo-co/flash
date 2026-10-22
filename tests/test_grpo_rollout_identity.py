from __future__ import annotations

import asyncio
import hashlib
import inspect
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from flash.engine.worker.train.rl.child import reward as reward_module
from flash.engine.worker.train.rl.rollout.identity import (
    RolloutIdentityLedger,
    parse_rollout_identity,
)


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
    from flash.engine.worker.train.rl.child import multiturn

    monkeypatch.setattr(multiturn, "post_json", post_json)
    monkeypatch.setenv("FLASH_VERL_REWARD_URL", url)
    namespace: dict = {}
    exec(inspect.getsource(reward_module), namespace)
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
    from flash.engine.worker.train.rl.child import multiturn

    monkeypatch.setattr(multiturn, "post_json", post_json)
    monkeypatch.setenv("FLASH_VERL_REWARD_URL", "http://127.0.0.1:1")
    namespace: dict = {}
    exec(inspect.getsource(reward_module), namespace)
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


def _agent_recipe(monkeypatch, run, dispatch=None):
    from flash.engine.worker.train.core.child.options import RuntimeOptions
    from tests.test_verl_child_plugins import _import_recipe, _modules

    _modules(
        monkeypatch,
        {
            "verl.experimental.agent_loop.agent_loop": {
                "AgentLoopBase": object,
                "AgentLoopOutput": SimpleNamespace,
                "register": lambda name: lambda cls: cls,
            },
            "verl.experimental.agent_loop.single_turn_agent_loop": {
                "SingleTurnAgentLoop": type("SingleTurnAgentLoop", (), {"run": run}),
            },
            "verl.trainer.ppo.v1.agent_loop_tq": {
                "AgentLoopManagerTQ": type(
                    "AgentLoopManagerTQ", (), {"generate_sequences": dispatch}
                ),
            },
            "verl.utils": {
                "tensordict_utils": SimpleNamespace(
                    get=lambda batch, key, default=None: batch.get(key, default),
                    assign_non_tensor_data=lambda batch, key, value: batch.update({key: value}),
                )
            },
        },
    )
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.rl.child.rollout")
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            model=SimpleNamespace(runtime_options=RuntimeOptions("grpo", 20, "unused").encode()),
            rollout=SimpleNamespace(n=2),
        )
    )
    monkeypatch.setenv("FLASH_VERL_REWARD_URL", "http://unused")
    return recipe, config


def _agent(recipe, config):
    from flash.engine.worker.train.core.child.observations import ObservedRolloutClient

    agent = recipe.FlashGrpoSingleTurnAgentLoop.__new__(recipe.FlashGrpoSingleTurnAgentLoop)
    agent.config = config
    agent.observations = ObservedRolloutClient(None)
    return agent


def test_manager_registers_exact_set_before_dispatch_with_native_session_ordinals(monkeypatch):
    ledger = RolloutIdentityLedger(2, 2)
    events = []

    async def run(self, params, **kwargs):
        identity = kwargs["flash_rollout_identity"]
        ledger.record(identity, kwargs["index"])
        events.append(("reward", identity))
        return SimpleNamespace(reward_score=1.0, extra_fields={})

    def dispatch(self, prompts):
        async def generate():
            return await asyncio.gather(
                *[
                    _agent(recipe, config).run(
                        {}, global_steps=prompts["global_steps"], index=index, session_id=ordinal
                    )
                    for index in prompts["index"]
                    for ordinal in range(2)
                ]
            )

        return asyncio.run(generate())

    recipe, config = _agent_recipe(monkeypatch, run, dispatch)

    def register(url, path, payload, **kwargs):
        assert path == "/identity/register"
        ledger.register(payload["identities"])
        events.append(("registered", payload["identities"]))

    monkeypatch.setattr(recipe, "post_json", register)
    manager = recipe.FlashGrpoManager.__new__(recipe.FlashGrpoManager)
    manager.config = config
    manager.generate_sequences({"index": [10, 11], "global_steps": 17})
    expected = _expected(17, (10, 11), 2)
    assert events[0] == ("registered", expected)
    assert [value for kind, value in events if kind == "reward"] == expected
    ledger.seal(17)
    ledger.finalize({17})


def test_concurrent_agent_calls_keep_identity_task_local(monkeypatch):
    async def run(self, params, **kwargs):
        await asyncio.sleep(0)
        return SimpleNamespace(
            reward_score=1.0, extra_fields={"identity": kwargs["flash_rollout_identity"]}
        )

    recipe, config = _agent_recipe(monkeypatch, run)
    agent = _agent(recipe, config)

    async def generate():
        return await asyncio.gather(
            agent.run({}, global_steps=12, index=4, session_id=7),
            agent.run({}, global_steps=99, index=8, session_id=3),
        )

    outputs = asyncio.run(generate())
    assert [output.extra_fields["identity"] for output in outputs] == [
        _identity(12, 4, 7),
        _identity(99, 8, 3),
    ]


def test_sampling_policy_preserves_identity_and_combines_stops_image_ban_and_constraints(
    monkeypatch,
):
    calls = []

    async def run(self, params, **kwargs):
        calls.append((params, kwargs))
        return SimpleNamespace(reward_score=1.0, extra_fields={})

    recipe, config = _agent_recipe(monkeypatch, run)
    module = types.ModuleType("vllm.sampling_params")
    module.StructuredOutputsParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", module)
    config.actor_rollout_ref.model.runtime_options.update(
        stop_sequences=["END"],
        image_pad_token_id=99,
        structured_outputs={"choice": ["yes"]},
    )
    params = {"temperature": 0.6, "logit_bias": {10: 1.0}}
    extra = {"index": 3}
    asyncio.run(
        _agent(recipe, config).run(params, global_steps=2, index=3, session_id=1, extra_info=extra)
    )
    actual, forwarded = calls[0]
    assert actual == {
        "temperature": 0.6,
        "logit_bias": {10: 1.0, 99: -100.0},
        "stop": ["END"],
        "structured_outputs": {"choice": ["yes"]},
    }
    assert forwarded["extra_info"]["flash_rollout_identity"] == _identity(2, 3, 1)
    assert params == {"temperature": 0.6, "logit_bias": {10: 1.0}}
    assert extra == {"index": 3}
