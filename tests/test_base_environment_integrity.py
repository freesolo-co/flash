"""Fallback environment scoring must use the completion actually produced by the rollout."""

import pytest

from flash.engine.worker.train.rl.rollout.scoring import RolloutScoreRequest, score_rollouts
from flash.envs.loading.base import BaseEnvironment


@pytest.mark.parametrize("thread_safe", [False, True])
def test_scalar_rollout_rewards_score_each_actual_response(thread_safe):
    env = BaseEnvironment(id="audit")
    env.reward_thread_safe = thread_safe
    requests = [
        RolloutScoreRequest({"output": "42"}, {"response_text": "42"}, 1),
        RolloutScoreRequest({"output": "42"}, {"response_text": "incorrect"}, 1),
    ]
    assert [reward.episode for reward in score_rollouts(env, requests)] == [1.0, 0.0]


@pytest.mark.parametrize("value", [0, False])
def test_base_environment_preserves_falsy_example_values(value):
    env = BaseEnvironment(id="audit")
    example = {"input": value, "output": value}
    assert env.prompt_messages(example) == [{"role": "user", "content": str(value)}]
    assert env.sft_completion(example) == [{"role": "assistant", "content": str(value)}]
    assert env.grade(str(value), example) is True


@pytest.mark.parametrize("example", [{}, {"input": None, "output": None}])
def test_base_environment_missing_values_remain_empty(example):
    env = BaseEnvironment(id="audit")
    assert env.prompt_messages(example) == [{"role": "user", "content": ""}]
    assert env.sft_completion(example) == [{"role": "assistant", "content": ""}]
    assert env.grade("anything", example) is False
