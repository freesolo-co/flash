"""Generation metadata reaches user scorers without changing text reward semantics."""

import asyncio
import copy
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flash.engine.worker.train.rl.child import reward
from flash.engine.worker.train.rl.child.contract import ContractRolloutClient
from flash.engine.worker.train.rl.child.multiturn import _score_episode
from flash.engine.worker.train.rl.rollout.multi_turn import (
    _BadRequest,
    _reward_context,
    start_reward_server,
)
from flash.engine.worker.train.rl.rollout.single_turn import (
    score_single_turn,
    score_single_turn_batch,
)
from flash.envs.loading.adapter import FreesoloEnvironment
from flash.envs.scoring.context import RolloutContext
from flash.envs.scoring.response import ScoredResponseText
from tests.test_grpo_rollout_identity import _agent_recipe
from tests.test_verifiers import _EnvironmentSingleTurn, _install_fake_freesolo, _RewardResult
from tests.test_versioned_rollout_contract import Server, episode_output, load


def identity(ordinal=0):
    return {"optimizer_step": 1, "sample_index": 0, "rollout_ordinal": ordinal, "validate": False}


def agent_for(server):
    return SimpleNamespace(
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(model=server.model_config, rollout=server.config)
        ),
        tokenizer=server.model_config.tokenizer,
    )


async def observe(client, server, ordinal=0, *, multi_turn=False):
    with client.episode_contract(agent_for(server), identity(ordinal)):
        await client.generate(
            prompt_ids=[1], sampling_params={"logprobs": True}, request_id=f"first-{ordinal}"
        )
        if multi_turn:
            await client.generate(
                prompt_ids=[1, 2, 3, 9],
                sampling_params={"logprobs": True},
                request_id=f"second-{ordinal}",
            )
        return client.reward_context()


@pytest.fixture
def context_wire():
    async def generate():
        server = Server()
        await load(server)
        return await observe(ContractRolloutClient(server), server)

    return asyncio.run(generate())


def test_context_preserves_turn_boundaries_sampling_and_task_local_identity():
    async def generate():
        server = Server()
        await load(server)
        client = ContractRolloutClient(server)
        contexts = await asyncio.gather(
            observe(client, server, 0, multi_turn=True), observe(client, server, 1)
        )
        with pytest.raises(RuntimeError, match="contract scope"):
            client.reward_context()
        return [RolloutContext.decode(json.loads(json.dumps(value))) for value in contexts]

    first, second = asyncio.run(generate())
    assert (first.rollout_ordinal, second.rollout_ordinal) == (0, 1)
    assert len(first.turns) == 2
    assert len(second.turns) == 1
    assert first.turns[1].prompt_token_ids == (1, 2, 3, 9)
    assert first.turns[1].token_ids == (2, 3)  # environment token 9 has no generated logprob
    assert first.turns[1].token_logprobs == (-0.2, -0.3)
    assert first.turns[1].sampling == {"temperature": 0.8, "max_tokens": 2}
    assert first.policy.policy_version == 0
    assert first.logprob_definition == "processed_logprobs"


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_http_bridge_to_sdk_scorer_preserves_context_and_legacy_text(
    monkeypatch, worker_context, context_wire, batched, thinking
):
    _install_fake_freesolo(monkeypatch)
    seen = []

    class Env(_EnvironmentSingleTurn):
        def score_responses(self, example, responses):
            results = []
            for response in responses:
                assert response == "4"
                assert isinstance(response, str)
                context = response.rollout_context
                assert context.turns[0].token_logprobs == (-0.2, -0.3)
                assert response.raw == ("<think>work</think>4" if thinking else "4")
                seen.append(context.rollout_ordinal)
                results.append(_RewardResult(score=0.25 + context.rollout_ordinal))
            return results

    env = FreesoloEnvironment(Env(), "owner/env", source=None)
    example = {"id": "0", "input": "2+2", "output": "4"}
    kwargs = {
        "tok": None,
        "thinking": thinking,
        "prompt_opened_thinking": False,
        "think_penalty": 0,
        "context": worker_context,
    }
    # Use real thinking cleanup; only the worker's run-level setting is changed.
    monkeypatch.setattr(worker_context, "config", replace(worker_context.config, thinking=thinking))

    def score_batch(requests):
        rows = score_single_turn_batch(env, [(text, example) for _, text in requests], **kwargs)
        return [score for score, _ in rows]

    server, url = start_reward_server(
        lambda index, text: score_single_turn(env, text, example, **kwargs),
        example_count=1,
        score_batch=score_batch if batched else None,
    )
    monkeypatch.setenv("FLASH_VERL_REWARD_URL", url)
    try:

        def score(ordinal):
            wire = dict(context_wire, rollout_ordinal=ordinal)
            return reward.compute_score(
                "env",
                "<think>work</think>4" if thinking else "4",
                "",
                {
                    "index": 0,
                    "flash_rollout_identity": identity(ordinal),
                    "flash_rollout_context": wire,
                },
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(score, range(2))) == [0.25, 1.25]
        assert sorted(seen) == [0, 1]
    finally:
        server.shutdown()


def test_single_turn_policy_attaches_context_before_reward_and_loss_mask(monkeypatch):
    async def run(self, params, **kwargs):
        await self.observations.generate(
            prompt_ids=[1], sampling_params=params, request_id="single"
        )
        output = episode_output()
        output.reward_score = None
        output.extra_fields["finish_reason"] = "length"
        return output

    recipe, config = _agent_recipe(monkeypatch, run)
    server = Server()
    server.model_config.runtime_options.update(
        total_steps=20, readiness_path="unused", mask_truncated_completions=True
    )
    config.actor_rollout_ref.model = server.model_config
    config.actor_rollout_ref.rollout = server.config
    agent = recipe.FlashGrpoSingleTurnAgentLoop.__new__(recipe.FlashGrpoSingleTurnAgentLoop)
    agent.config = config
    agent.tokenizer = server.model_config.tokenizer
    agent.tokenizer.decode = lambda ids, **kwargs: "4"
    agent.observations = ContractRolloutClient(server)
    captured = []

    def score(data_source, text, ground_truth, extra):
        captured.append(RolloutContext.decode(extra["flash_rollout_context"]))
        return 0.75

    monkeypatch.setattr(recipe, "compute_score", score)

    async def generate():
        await load(server)
        return await agent.run({"logprobs": True}, global_steps=1, index=0, session_id=0)

    output = asyncio.run(generate())
    assert output.reward_score == 0.75
    assert output.response_mask == [0, 0]
    assert captured[0].turns[0].token_ids == (2, 3)
    assert captured[0].turns[0].token_logprobs == (-0.2, -0.3)


def test_multiturn_child_snapshots_context_before_executor_and_bridge_preserves_it():
    from tests.test_rl_train import _bridge, _BridgeEnv
    from tests.test_verifiers import _EnvironmentEpisode

    async def generate():
        server = Server()
        await load(server)
        client = ContractRolloutClient(server)
        env = _BridgeEnv()
        bridge = _bridge(env)
        bridge.start({"index": 0, "session_id": "episode"})
        bridge.step({"session_id": "episode", "completion_text": "4"})
        try:
            with client.episode_contract(agent_for(server), identity()):
                await client.generate(
                    prompt_ids=[1], sampling_params={"logprobs": True}, request_id="multi"
                )
                agent = SimpleNamespace(server_manager=client, loop=asyncio.get_running_loop())

                def post(url, path, payload):
                    assert path == "/multiturn/score"
                    return bridge.score(payload)

                scored = await _score_episode(
                    agent,
                    post,
                    "unused",
                    "episode",
                    SimpleNamespace(turn_spans=[(0, 2)]),
                    identity(),
                )
            assert scored == (1.0, None)
            state = env.scored[0]
            adapter = FreesoloEnvironment.__new__(FreesoloEnvironment)
            adapter._EnvironmentEpisode = _EnvironmentEpisode
            episode = adapter._episode_from_state(dict(state, step_metadata=[{"keep": True}]))
            assert episode.metadata["rollout_context"].turns[0].token_ids == (2, 3)
            assert episode.metadata["steps"] == [{"keep": True}]
        finally:
            bridge.shutdown()

    asyncio.run(generate())


@pytest.mark.parametrize(
    "change",
    [
        {"token_logprobs": [-0.2]},
        {"token_logprobs": [float("nan"), -0.3]},
        {"token_logprobs": [True, -0.3]},
        {"token_ids": [True, 3]},
        {"finish_reason": "cancelled"},
        {"sampling": {}},
    ],
)
def test_malformed_context_is_rejected_before_scoring(context_wire, change):
    context_wire["turns"][0].update(change)
    with pytest.raises(_BadRequest, match="invalid reward context"):
        _reward_context({"rollout_context": context_wire, "identity": identity()})


def test_context_cannot_be_reassigned_to_a_different_rollout(context_wire):
    with pytest.raises(_BadRequest, match="does not match rollout identity"):
        _reward_context({"rollout_context": context_wire, "identity": identity(1)})


def test_response_context_survives_copy_and_pickle_without_aliasing(context_wire):
    context = RolloutContext.decode(context_wire)
    context_wire["turns"][0]["sampling"]["temperature"] = 99
    assert context.turns[0].sampling["temperature"] == 0.8
    response = ScoredResponseText(
        "4", raw="<think>work</think>4", thinking="work", rollout_context=context
    )
    for restored in (copy.deepcopy(response), pickle.loads(pickle.dumps(response))):
        assert restored == "4"
        assert restored.raw == response.raw
        assert restored.rollout_context == context
        assert restored.rollout_context is not context


def test_batch_retry_keeps_each_native_scorers_context(worker_context, context_wire):
    contexts = [RolloutContext.decode(dict(context_wire, rollout_ordinal=i)) for i in range(2)]
    responses = [
        ScoredResponseText("4", raw="4", thinking=None, rollout_context=context)
        for context in contexts
    ]
    seen = []

    class Env:
        def reward_many(self, items):
            raise RuntimeError("retry individually")

        def reward(self, text, example, state):
            context = state["rollout_context"]
            seen.append(context)
            return context.rollout_ordinal + 0.25

    results = score_single_turn_batch(
        Env(),
        [(response, {}) for response in responses],
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0,
        context=worker_context,
    )
    assert results == [(0.25, []), (1.25, [])]
    assert seen == contexts


def test_legacy_scoring_does_not_invent_context():
    from flash.envs.loading.adapter import _completion_for_scoring

    assert type(_completion_for_scoring("4", None)) is str
    response = _completion_for_scoring("4", {"raw": "<think>work</think>4", "thinking": "work"})
    assert response == "4"
    assert response.rollout_context is None


@pytest.mark.parametrize(
    "changes",
    [
        {"turns": {}},
        {"turns": None},
        {"logprob_definition": "unspecified"},
        {"optimizer_step": True},
        {"sample_index": -1},
    ],
)
def test_context_envelope_rejects_malformed_metadata(context_wire, changes):
    with pytest.raises(_BadRequest, match="invalid reward context"):
        _reward_context({"rollout_context": dict(context_wire, **changes), "identity": identity()})
