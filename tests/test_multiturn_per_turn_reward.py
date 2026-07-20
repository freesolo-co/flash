"""Torch-free coverage for multi-turn per-turn reward collection."""

from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace

import pytest
from freesolo.environments.types import RewardResult

from flash.engine.multiturn_reward_scoring import RolloutScoreRequest, score_rollouts
from flash.engine.multiturn_rollout import rollout_one
from flash.envs.adapter import FreesoloEnvironment
from flash.envs.base import BaseEnvironment, RolloutReward


class SyntheticPerTurnEnv(BaseEnvironment):
    """Two-turn environment with deterministic target-token rewards."""

    multi_turn = True

    def __init__(self):
        super().__init__(id="synthetic-per-turn")
        self.rollout_reward_calls = 0

    def dataset(self):
        return [{"input": "prompt", "targets": ["hit", "hit"]}]

    def prompt_messages(self, example):
        return [{"role": "user", "content": str(example["input"])}]

    def new_rollout_state(self, example):
        return {
            "prompt": self.prompt_messages(example),
            "assistant_turns": [],
            "targets": list(example["targets"]),
        }

    def record_model_turn(self, state, content):
        state["assistant_turns"].append(content)

    def rollout_done(self, state, max_turns):
        return len(state["assistant_turns"]) >= min(2, max_turns)

    def env_reply(self, messages, state):
        return [{"role": "user", "content": "next"}]

    def reward(self, completion, example, state=None):
        state = state or {}
        targets = state.get("targets") or example["targets"]
        return float(
            sum(
                target in text
                for text, target in zip(state.get("assistant_turns", []), targets, strict=True)
            )
        )

    def rollout_rewards_many(self, items):
        self.rollout_reward_calls += 1
        rewards = []
        for example, state in items:
            targets = state.get("targets") or example["targets"]
            turns = tuple(
                1.0 if target in text else 0.0
                for text, target in zip(state.get("assistant_turns", []), targets, strict=True)
            )
            rewards.append(RolloutReward(episode=sum(turns), turns=turns))
        return rewards


_TOKEN_IDS = {"prompt": 3, "miss": 4, "hit": 5, "next": 6}


def _rollout(env, turn_texts):
    generated = iter(turn_texts)

    def generate(prefix_ids, max_tokens):
        _ = prefix_ids, max_tokens
        text = next(generated)
        return [_TOKEN_IDS[text]], [-0.1], text

    return rollout_one(
        example={"input": "prompt", "targets": ["hit", "hit"]},
        active_env=env,
        render=lambda messages, add_generation_prompt: [_TOKEN_IDS["prompt"]],
        generate=generate,
        env_glue=lambda messages: [_TOKEN_IDS["next"]],
        max_turns=2,
        per_turn_max_tokens=4,
    )


def test_rollout_records_turn_spans_and_scores_terminal_episode_once():
    env = SyntheticPerTurnEnv()

    result = _rollout(env, ["miss", "hit"])

    assert result["completion_ids"] == [4, 6, 5]
    assert result["env_mask"] == [1, 0, 1]
    assert result["turn_spans"] == [(0, 1), (2, 3)]
    assert result["turn_rewards"] == [0.0, 1.0]
    assert result["reward"] == 1.0
    assert env.rollout_reward_calls == 1


@pytest.mark.parametrize(
    "invalid_turns",
    [(float("nan"), 1.0), (1.0,), ("bad", 1.0)],
    ids=["non-finite", "wrong-length", "non-number"],
)
def test_invalid_turn_rewards_degrade_to_episode_credit(capsys, invalid_turns):
    class InvalidTurnEnv(SyntheticPerTurnEnv):
        def rollout_rewards_many(self, items):
            self.rollout_reward_calls += 1
            assert len(items) == 1
            return [RolloutReward(episode=1.0, turns=invalid_turns)]

    invalid = _rollout(InvalidTurnEnv(), ["miss", "hit"])

    assert invalid["reward"] == 1.0
    assert invalid["turn_rewards"] is None
    assert capsys.readouterr().out.count("using episode reward") == 1


def test_non_finite_episode_reward_disables_per_turn_credit(capsys):
    class NonFiniteEpisodeEnv(SyntheticPerTurnEnv):
        def rollout_rewards_many(self, items):
            self.rollout_reward_calls += 1
            assert len(items) == 1
            return [RolloutReward(episode=float("nan"), turns=(0.0, 1.0))]

    env = NonFiniteEpisodeEnv()
    result = _rollout(env, ["miss", "hit"])

    assert result["turn_rewards"] is None
    assert env.rollout_reward_calls == 1
    assert capsys.readouterr().out.count("rollout unscorable") == 1


@pytest.mark.parametrize("bad_episode", [float("nan"), float("inf"), float("-inf")])
def test_score_rollouts_canonicalizes_non_finite_episode_to_nan(bad_episode, capsys):
    class _NonFiniteEnv(BaseEnvironment):
        def rollout_rewards_many(self, items):
            return [RolloutReward(episode=bad_episode, turns=(0.0, 1.0)) for _ in items]

    [reward] = score_rollouts(
        _NonFiniteEnv(id="non-finite"),
        [RolloutScoreRequest(example={}, state={}, turn_count=2)],
    )
    # nan and inf both canonicalize to nan, trl's only unscorable marker (isnan(inf) is false)
    assert math.isnan(reward.episode)
    assert reward.turns is None
    assert capsys.readouterr().out.count("rollout unscorable") == 1


def test_score_rollouts_uses_one_typed_scoring_pass():
    class TypedRewardEnv(SyntheticPerTurnEnv):
        def __init__(self):
            super().__init__()
            self.score_calls = 0

        def reward(self, completion, example, state=None):
            raise AssertionError("scalar reward must not run after typed terminal scoring")

        def rollout_rewards_many(self, items):
            self.score_calls += 1
            assert len(items) == 1
            return [RolloutReward(episode=1.0, turns=(0.0, 1.0))]

    env = TypedRewardEnv()
    rewards = score_rollouts(
        env,
        [RolloutScoreRequest(example={"input": "prompt"}, state={}, turn_count=2)],
    )

    assert rewards == [RolloutReward(episode=1.0, turns=(0.0, 1.0))]
    assert env.score_calls == 1


def test_score_rollouts_uses_reward_many_fallback_for_duck_typed_env():
    class BatchedEnvironment:
        def __init__(self):
            self.batch_calls = 0

        def reward_many(self, items):
            self.batch_calls += 1
            return [float(example["reward"]) for example, _state in items]

    env = BatchedEnvironment()
    requests = [
        RolloutScoreRequest(example={"reward": 0.25}, state={}, turn_count=1),
        RolloutScoreRequest(example={"reward": 0.75}, state={}, turn_count=1),
    ]

    rewards = score_rollouts(env, requests)

    assert rewards == [
        RolloutReward(episode=0.25, turns=None),
        RolloutReward(episode=0.75, turns=None),
    ]
    assert env.batch_calls == 1


def test_base_environment_default_uses_reward_many_once():
    class BatchedEnvironment(BaseEnvironment):
        def __init__(self):
            super().__init__(id="batched")
            self.batch_calls = 0

        def reward(self, completion, example, state=None):
            raise AssertionError("scalar reward must not run when reward_many exists")

        def reward_many(self, items):
            self.batch_calls += 1
            return [float(example["reward"]) for example, _state in items]

    env = BatchedEnvironment()
    items = [({"reward": 0.25}, {}), ({"reward": 0.75}, {})]

    rewards = env.rollout_rewards_many(items)

    assert rewards == [
        RolloutReward(episode=0.25, turns=None),
        RolloutReward(episode=0.75, turns=None),
    ]
    assert env.batch_calls == 1


def test_score_rollouts_uses_scalar_fallback_for_duck_typed_env_in_input_order():
    class ScalarEnvironment:
        reward_thread_safe = False

        def __init__(self):
            self.calls = []

        def reward(self, completion, example, state=None):
            self.calls.append((completion, example, state))
            return float(example["reward"])

    env = ScalarEnvironment()
    requests = [
        RolloutScoreRequest(example={"reward": 0.2}, state={"index": 0}, turn_count=1),
        RolloutScoreRequest(example={"reward": 0.8}, state={"index": 1}, turn_count=1),
    ]

    rewards = score_rollouts(env, requests)

    assert rewards == [
        RolloutReward(episode=0.2, turns=None),
        RolloutReward(episode=0.8, turns=None),
    ]
    assert env.calls == [
        ("", requests[0].example, requests[0].state),
        ("", requests[1].example, requests[1].state),
    ]


def _freesolo_env(monkeypatch, result):
    env = object.__new__(FreesoloEnvironment)
    env.multi_turn = True
    state = {
        "turns": ["first", "second"],
        "turn": 2,
        "max_episode_turns": 2,
        "step_metadata": [{"per_turn_rewards": [0.25]}],
    }
    score_calls = []

    def score_episodes(task, episodes):
        score_calls.append((task, episodes))
        return [result]

    env._env = SimpleNamespace(reward_thread_safe=True, score_episodes=score_episodes)
    monkeypatch.setattr(env, "_task_example", lambda example: example)
    monkeypatch.setattr(env, "_episode_from_state", lambda terminal_state: terminal_state)
    return env, state, score_calls


def test_freesolo_rollout_rewards_uses_complete_terminal_metadata_at_cap(monkeypatch):
    result = RewardResult(score=1.0, metadata={"per_turn_rewards": [0.25, 0.75]})
    env, state, score_calls = _freesolo_env(monkeypatch, result)

    rewards = env.rollout_rewards_many([({"input": "prompt"}, state)])

    assert rewards == [RolloutReward(episode=1.0, turns=(0.25, 0.75))]
    assert score_calls == [({"input": "prompt"}, [state])]


@pytest.mark.parametrize(
    "metadata_value",
    ["not-a-list", [0.25, "bad"]],
    ids=["not-list", "non-number"],
)
def test_freesolo_malformed_per_turn_metadata_degrades_to_episode_reward(
    monkeypatch, capsys, metadata_value
):
    result = RewardResult(score=1.5, metadata={"per_turn_rewards": metadata_value})
    env, state, _score_calls = _freesolo_env(monkeypatch, result)

    rewards = env.rollout_rewards_many([({"input": "prompt"}, state)])

    assert rewards == [RolloutReward(episode=1.5, turns=None)]
    assert capsys.readouterr().out.count("[grpo][warn]") == 1


@pytest.mark.parametrize(
    "metadata_value",
    ["bad-string", [0.0, 1.0], ""],
    ids=["str", "list", "empty-str"],
)
def test_freesolo_non_mapping_metadata_degrades_to_episode_reward(
    monkeypatch, capsys, metadata_value
):
    # a non-mapping metadata container (env bug) must degrade to episode credit, not raise.
    result = RewardResult(score=2.0, metadata=metadata_value)
    env, state, _score_calls = _freesolo_env(monkeypatch, result)

    rewards = env.rollout_rewards_many([({"input": "prompt"}, state)])

    assert rewards == [RolloutReward(episode=2.0, turns=None)]
    assert capsys.readouterr().out.count("[grpo][warn]") == 1


def test_build_rollout_func_always_emits_per_turn_fields(monkeypatch):
    from flash.engine import multiturn_rollout

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.RequestOutputKind = SimpleNamespace(FINAL_ONLY="final_only")
    vllm.sampling_params = sampling_params
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    rollout = {
        "prompt_ids": [1],
        "completion_ids": [2],
        "logprobs": [-0.1],
        "env_mask": [1],
        "reward": 0.5,
        "turn_spans": [(0, 1)],
        "turn_rewards": None,
    }
    monkeypatch.setattr(multiturn_rollout, "rollout_async", lambda **kwargs: [rollout])
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(
            model_config=SimpleNamespace(get_vocab_size=lambda: 32),
        )
    )
    trainer = SimpleNamespace(
        vllm_generation=SimpleNamespace(llm=engine),
        args=SimpleNamespace(vllm_enable_sleep_mode=False),
    )
    rollout_func = multiturn_rollout.build_rollout_func(
        active_env=BaseEnvironment(id="episode-only"),
        tok=SimpleNamespace(),
        examples_by_key={},
        max_completion=4,
        max_turns=1,
        temperature=0.7,
        top_p=1.0,
        stop=None,
        thinking=False,
    )

    output = rollout_func([[{"role": "user", "content": "prompt"}]], trainer)

    assert output["turn_spans"] == [[(0, 1)]]
    assert output["turn_rewards"] == [None]
    assert set(output) == set(rollout)
