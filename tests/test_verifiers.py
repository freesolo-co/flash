"""Tests for the Freesolo SDK environment adapter."""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import sys
import tarfile
import threading
import time
import tracemalloc
import types
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

import flash.engine.worker.runtime.state as worker_state


@dataclass
class _TaskExample:
    record: dict
    input: str
    id: str | None = None
    output: object | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _RewardMetric:
    name: str
    score: float | None = None


@dataclass(frozen=True)
class _RewardResult:
    score: float
    metrics: tuple[_RewardMetric, ...] = ()
    success: bool | None = None
    threshold: float | None = None
    error: str | None = None

    def resolved_success(self) -> bool:
        if self.success is not None:
            return self.success
        if self.error:
            return False
        if self.threshold is not None:
            return self.score >= self.threshold
        return self.score > 0.0


@dataclass(frozen=True)
class _EnvironmentTurn:
    role: str
    content: str | list[dict]


@dataclass(frozen=True)
class _EnvironmentEpisode:
    messages: tuple[dict, ...]
    response_text: str
    turns: tuple[_EnvironmentTurn, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _EnvironmentStepResult:
    done: bool = True
    messages: tuple[dict, ...] = ()
    final_response_text: str | None = None
    metadata: dict = field(default_factory=dict)


class _EnvironmentSingleTurn:
    pass


class _EnvironmentMultiTurn:
    pass


class _FakeSingleTurnEnv(_EnvironmentSingleTurn):
    # no class-level dataset: like a real sdk env that does not build one, so file-backed
    # sources exercise the fallback path. tests that need an env-built dataset set it on
    # the instance explicitly.

    def start_episode(self, example, prompt_text):
        return [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": example.input},
        ]

    def score_responses(self, example, response_texts):
        out = []
        for response in response_texts:
            score = 1.0 if str(example.output) in response else 0.0
            out.append(
                _RewardResult(
                    score=score,
                    success=score == 1.0,
                    metrics=(_RewardMetric("match", score),),
                )
            )
        return out


class _FakeMultiTurnEnv(_EnvironmentMultiTurn):
    def __init__(self):
        self.step_tasks = []
        self.score_tasks = []

    def start_episode(self, example, prompt_text):
        return [{"role": "user", "content": f"{prompt_text}:{example.input}"}]

    def step_episode(self, example, messages, assistant_response):
        self.step_tasks.append(example)
        return _EnvironmentStepResult(
            done=True,
            messages=({"role": "user", "content": f"observed {assistant_response}"},),
            final_response_text=f"final {assistant_response}",
            metadata={"input": example.input},
        )

    def score_episodes(self, example, episodes):
        self.score_tasks.append(example)
        return [
            _RewardResult(
                score=0.5,
                success=True,
                metrics=(_RewardMetric("episode", 0.5),),
            )
            for _episode in episodes
        ]


class _BudgetMultiTurnEnv(_EnvironmentMultiTurn):
    """Multi-turn env with a per-example budget that never self-terminates (done=False),
    so the rollout cap (max_episode_turns) is what must stop it."""

    def start_episode(self, example, prompt_text):
        return [{"role": "user", "content": "go"}]

    def max_episode_turns(self, example):
        return 15

    def step_episode(self, example, messages, assistant_response):
        return _EnvironmentStepResult(
            done=False,
            messages=({"role": "user", "content": "more"},),
            final_response_text=None,
            metadata={},
        )

    def score_episodes(self, example, episodes):
        return [_RewardResult(score=0.0, success=False, metrics=()) for _ in episodes]


class _PerEpisodeImageEnv(_EnvironmentSingleTurn):
    """Single-turn env that CHOOSES its image inside start_episode.

    The record names a pool, not one image, so nothing in the raw row identifies what the model
    was shown. The env records its pick on the task it was handed and grades against that pick --
    the documented way an env carries per-episode state, and the only one the SDK offers, since
    score_responses(example, texts) takes no prompt.
    """

    def __init__(self, picks):
        self._picks = list(picks)
        self.prompt_tasks = []
        self.score_tasks = []
        self.graded_against = []

    def start_episode(self, example, prompt_text):
        self.prompt_tasks.append(example)
        chosen = self._picks.pop(0)
        example.metadata["chosen_image"] = chosen
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what color?"},
                    {"type": "image", "image": chosen},
                ],
            }
        ]

    def score_responses(self, example, response_texts):
        self.score_tasks.append(example)
        chosen = example.metadata.get("chosen_image")
        self.graded_against.append(chosen)
        out = []
        for response in response_texts:
            # the grader can only be right if it is told which image was actually rendered.
            score = 1.0 if chosen and chosen.split("/")[-1].split(".")[0] in response else 0.0
            out.append(
                _RewardResult(
                    score=score, success=score == 1.0, metrics=(_RewardMetric("match", score),)
                )
            )
        return out


def test_start_episode_image_choice_reaches_single_turn_scoring(monkeypatch):
    """An image the env chose INSIDE start_episode must reach single-turn scoring.

    Single-turn scoring used to rebuild the TaskExample from the raw dataset record, so a
    per-episode choice the env made while building the prompt never reached the grader: the model
    was shown one image and graded against a record that does not identify it. The reward was
    silently wrong, and GRPO optimizes exactly that number.

    A top-level record image was never the broken case (it is part of the record and survives a
    rebuild); an env that RANDOMIZES or GENERATES per episode is.
    """
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    sdk_env = _PerEpisodeImageEnv(picks=["pool/red.png", "pool/blue.png"])
    # no "metadata" key on the record: task_example_from_record substitutes a FRESH {} for such a
    # row, so this is the shape whose per-episode state a rebuild silently dropped.
    env = FreesoloEnvironment(
        sdk_env,
        "owner/env",
        source=[{"input": "what color?", "image_pool": ["red", "blue"]}],
        contract_text="",
    )
    # the row object GRPO generates from and grades on is the one dataset() built.
    (row,) = env.dataset()

    prompt = env.prompt_messages(row)
    rendered = [
        block["image"]
        for message in prompt
        for block in (message["content"] if isinstance(message["content"], list) else [])
        if block.get("type") == "image"
    ]
    assert rendered == ["pool/red.png"]

    # the model answered with what it was actually shown, so a grader that sees the same episode
    # must score 1.0. scoring 0.0 here means it graded a DIFFERENT episode.
    assert env.reward("it is red", row) == 1.0
    assert sdk_env.prompt_tasks[0] is sdk_env.score_tasks[0]
    assert sdk_env.graded_against == ["pool/red.png"]
    # and the choice must not be re-rolled by the scoring call: a second start_episode would both
    # consume the next pick and grade against an episode that was never generated.
    assert sdk_env._picks == ["pool/blue.png"]


def test_per_episode_state_survives_every_single_turn_scoring_entry_point(monkeypatch):
    """reward, grade, scores_breakdown, reward_with_error and the BATCHED paths all grade the
    episode that was generated -- one rebuilt task on any of them is a silently wrong reward."""
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    sdk_env = _PerEpisodeImageEnv(picks=["pool/red.png"])
    env = FreesoloEnvironment(
        sdk_env, "owner/env", source=[{"input": "what color?"}], contract_text=""
    )
    (row,) = env.dataset()
    env.prompt_messages(row)

    assert env.reward("it is red", row) == 1.0
    assert env.grade("it is red", row) is True
    assert env.scores_breakdown("it is red", row) == {"match": 1.0, "total": 1.0}
    assert env.reward_with_error("it is red", row)[0] == 1.0
    # batched entry points: GRPO scores a whole group through these.
    assert env.reward_many([(row, {"response_text": "it is red"})]) == [1.0]
    assert env.scores_breakdown_many([(row, {"response_text": "it is red"})]) == [
        {"match": 1.0, "total": 1.0}
    ]
    # every one of them graded the generated episode, and none re-ran start_episode.
    assert set(sdk_env.graded_against) == {"pool/red.png"}
    assert sdk_env._picks == []


def test_unprepared_rollout_preserves_nested_prompt_identity(monkeypatch):
    nested_content = [{"type": "text", "text": "go"}]

    class _NestedPromptEnv(_FakeMultiTurnEnv):
        def start_episode(self, example, prompt_text):
            return [{"role": "user", "content": nested_content}]

    sdk_env = _NestedPromptEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        sdk_env,
        "owner/env",
        source=[{"input": "go"}],
        contract_text="",
    )
    state = env.new_rollout_state({"input": "go"})

    assert state["messages"][0] is not state["prompt"][0]
    assert state["messages"][0]["content"] is state["prompt"][0]["content"]


def test_sibling_rollouts_get_isolated_tasks(monkeypatch):
    sdk_env = _FakeMultiTurnEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        sdk_env,
        "owner/env",
        source=[
            {
                "input": "browse",
                "metadata": {"session": "original"},
                "image": {"url": "original.png"},
                "output": {"target": "original"},
            }
        ],
        contract_text="",
    )
    (row,) = env.dataset()

    first = env.new_rollout_state(row)
    second = env.new_rollout_state(row)

    assert first["task"] is not second["task"]
    first["task"].metadata["session"] = "first"
    first["task"].record["image"]["url"] = "first.png"
    first["task"].output["target"] = "first"
    assert second["task"].metadata == {"session": "original"}
    assert second["task"].record["image"] == {"url": "original.png"}
    assert second["task"].output == {"target": "original"}

    env.record_model_turn(first, "click")
    env.env_reply(first["messages"], first)
    env.reward("ignored", row, first)
    assert sdk_env.step_tasks[0] is first["task"]
    assert sdk_env.score_tasks[0] is first["task"]


def test_bridge_clones_the_prepared_task_for_each_sibling_rollout(monkeypatch):
    """the bridge must step and score copies of the task that produced the frozen prompt."""

    class _PreparedTaskNonceEnv(_EnvironmentMultiTurn):
        def __init__(self):
            self.start_calls = 0
            self.scored: list[tuple[str, str, str]] = []

        def start_episode(self, task, prompt_text):
            self.start_calls += 1
            nonce = f"nonce-{self.start_calls}"
            task.metadata["prompt_nonce"] = nonce
            return [{"role": "user", "content": f"prompt:{nonce}"}]

        def max_episode_turns(self, task):
            return 2

        def step_episode(self, task, messages, assistant_response):
            task.metadata["session"] = assistant_response
            return _EnvironmentStepResult(done=True, final_response_text=assistant_response)

        def score_episodes(self, task, episodes):
            nonce = str(task.metadata.get("prompt_nonce") or "")
            session = str(task.metadata.get("session") or "")
            rewards = []
            for episode in episodes:
                prompt = str(episode.messages[0].get("content") or "")
                response = str(episode.response_text or "")
                self.scored.append((nonce, session, prompt))
                correct = prompt == f"prompt:{nonce}" and response == session
                rewards.append(_RewardResult(score=float(correct), success=correct))
            return rewards

    _install_fake_freesolo(monkeypatch)

    from flash.engine.worker.train.rl.rollout.multi_turn import MultiTurnBridge
    from flash.envs.loading.adapter import FreesoloEnvironment

    sdk_env = _PreparedTaskNonceEnv()
    env = FreesoloEnvironment(
        sdk_env,
        "owner/env",
        source=[{"input": "go"}],
        contract_text="",
    )
    (row,) = env.dataset()
    prepared_prompt = env.prompt_messages(row)
    prepared_task = env._row_tasks[id(row)]
    with pytest.raises(RuntimeError, match="dataset row task"):
        env.new_rollout_state(dict(row), prepared_prompt)

    batch_sizes: list[int] = []
    rollout_rewards_many = env.rollout_rewards_many

    def record_batch(items):
        batch_sizes.append(len(items))
        return rollout_rewards_many(items)

    env.rollout_rewards_many = record_batch
    bridge = MultiTurnBridge(
        env,
        [row],
        env_prompts=[prepared_prompt],
        max_turns=2,
        score_batch_size=2,
    )
    try:
        for session_id in ("left", "right"):
            bridge.start({"index": 0, "session_id": session_id})

        left_task = bridge._sessions["left"]["state"]["task"]
        right_task = bridge._sessions["right"]["state"]["task"]
        assert sdk_env.start_calls == 1
        assert left_task.metadata["prompt_nonce"] == "nonce-1"
        assert right_task.metadata["prompt_nonce"] == "nonce-1"
        assert left_task is not right_task
        assert left_task is not prepared_task
        assert right_task is not prepared_task

        bridge.step({"session_id": "left", "completion_text": "left"})
        bridge.step({"session_id": "right", "completion_text": "right"})
        assert left_task.metadata["session"] == "left"
        assert right_task.metadata["session"] == "right"
        assert "session" not in prepared_task.metadata

        scores: dict[str, dict] = {}
        errors: list[BaseException] = []
        ready = threading.Barrier(3)

        def score(session_id):
            try:
                ready.wait()
                scores[session_id] = bridge.score({"session_id": session_id, "turn_count": 1})
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=score, args=(session_id,)) for session_id in ("left", "right")
        ]
        for thread in threads:
            thread.start()
        ready.wait()
        for thread in threads:
            thread.join(timeout=5)

        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        assert batch_sizes == [2]
        assert scores == {"left": {"score": 1.0}, "right": {"score": 1.0}}
        assert sorted(sdk_env.scored) == [
            ("nonce-1", "left", "prompt:nonce-1"),
            ("nonce-1", "right", "prompt:nonce-1"),
        ]
    finally:
        bridge.shutdown()


def test_batched_scoring_uses_each_siblings_own_task(monkeypatch):
    """Batched scoring must not score every sibling against the first sibling's task.

    `_grouped_results` hands the scorer ONE task per group, so grouping siblings of a row on the
    row value alone scores them all against whichever session came first. That is invisible while
    siblings share a task object and becomes a reward-correctness bug once each rollout owns its
    own. Driven through `rollout_rewards_many`, the door that actually reaches the grouped scorer.
    """
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    class _SessionScoringEnv(_EnvironmentMultiTurn):
        """Scores an episode by the session recorded on the task it is handed."""

        def __init__(self):
            self.scored_sessions = []

        def start_episode(self, example, prompt_text):
            return [{"role": "user", "content": str(example.input)}]

        def step_episode(self, example, messages, assistant_response):
            example.metadata["session"] = assistant_response
            return _EnvironmentStepResult(
                done=True,
                messages=({"role": "user", "content": "ok"},),
                final_response_text=assistant_response,
            )

        def score_episodes(self, example, episodes):
            session = str(example.metadata.get("session") or "")
            self.scored_sessions.append(session)
            return [
                _RewardResult(
                    score=1.0 if session == "b" else 0.0,
                    success=session == "b",
                    metrics=(_RewardMetric("episode", 1.0),),
                )
                for _episode in episodes
            ]

    sdk_env = _SessionScoringEnv()
    env = FreesoloEnvironment(
        sdk_env,
        "owner/env",
        source=[{"input": "browse", "metadata": {}}],
        contract_text="",
    )
    (row,) = env.dataset()

    states = []
    for turn in ("a", "b"):
        state = env.new_rollout_state(row)
        env.record_model_turn(state, turn)
        env.env_reply(state["messages"], state)
        states.append(state)

    rewards = env.rollout_rewards_many([(row, state) for state in states])

    # each sibling is scored against the session ITS OWN episode recorded, so the two disagree.
    assert sorted(sdk_env.scored_sessions) == ["a", "b"]
    assert [reward.episode for reward in rewards] == [0.0, 1.0]
    assert env.reward_many([(row, state) for state in states]) == [0.0, 1.0]


def test_each_row_keeps_its_own_episode_and_non_dataset_rows_are_not_retained(monkeypatch):
    """Rows must not share a task with each other, and only dataset rows get a stable one.

    Ids are POSITIONAL (example_000000 ...), so keying on the id rather than the row would let two
    rows collide on one task. And a caller that mints a row per call -- `flash env eval` builds one
    per case -- must not have every one of them pinned for the life of the environment.
    """
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    sdk_env = _PerEpisodeImageEnv(picks=["pool/red.png", "pool/blue.png"])
    env = FreesoloEnvironment(
        sdk_env,
        "owner/env",
        source=[{"input": "what color?"}, {"input": "and this one?"}],
        contract_text="",
    )
    first, second = env.dataset()
    env.prompt_messages(first)
    env.prompt_messages(second)

    # each row grades against the episode IT generated, not the other's.
    assert env.reward("it is red", first) == 1.0
    assert env.reward("it is blue", second) == 1.0

    # one entry per dataset row, and nothing more: scoring rows the dataset never produced adds
    # no entries, so an eval suite cannot grow this map.
    assert len(env._row_tasks) == 2
    for index in range(50):
        env.reward("x", {"id": f"case_{index}", "input": "held out"})
    assert len(env._row_tasks) == 2


def test_single_turn_reward_many_batches_by_example_value_identical(monkeypatch):
    """Single-turn reward_many groups same-example rollouts into ONE score_responses() call
    (env-concurrent, the win for judge/network rewards) while staying byte-identical and in input
    order vs the per-item reward() reference. Without grouping, a GRPO step scored its whole
    completion batch with serial per-rollout reward() calls (one blocking judge round-trip each)."""
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    class _CountingSingleTurnEnv(_FakeSingleTurnEnv):
        def __init__(self):
            self.score_responses_calls = 0
            self.batch_sizes = []

        def score_responses(self, example, response_texts):
            self.score_responses_calls += 1
            self.batch_sizes.append(len(response_texts))
            return super().score_responses(example, response_texts)

    sdk_env = _CountingSingleTurnEnv()
    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    assert env.multi_turn is False

    ex_a = {"id": "a", "input": "2+2?", "output": "4"}
    ex_b = {"id": "b", "input": "3+3?", "output": "6"}
    # ex_a appears twice (a GRPO group) interleaved with ex_b -> grouping must preserve input order.
    items = [
        (ex_a, {"response_text": "the answer is 4"}),  # -> 1.0
        (ex_b, {"response_text": "it is 6"}),  # -> 1.0
        (ex_a, {"response_text": "nope"}),  # -> 0.0
    ]
    reference = [env.reward(st["response_text"], ex, st) for ex, st in items]
    sdk_env.score_responses_calls = 0
    sdk_env.batch_sizes = []

    out = env.reward_many(items)

    assert out == reference == [1.0, 1.0, 0.0]  # byte-identical + input order
    assert sdk_env.score_responses_calls == 2  # grouped: {ex_a:2, ex_b:1}, not 3 serial calls
    assert sorted(sdk_env.batch_sizes) == [1, 2]  # ex_a's two completions scored in ONE call


def test_single_turn_reward_many_serial_when_not_thread_safe(monkeypatch):
    """An env that opts out with reward_thread_safe = False must NOT have a group's
    completions batched into one env-concurrent score_responses call (a scorer with mutable/thread-
    bound state would be raced). reward_many must score each rollout with its OWN single-item call,
    byte-identical and in input order — the pre-batching serial behavior."""
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    class _UnsafeCountingSingleTurnEnv(_FakeSingleTurnEnv):
        reward_thread_safe = False  # scorer keeps mutable/thread-bound state -> never race it

        def __init__(self):
            self.batch_sizes = []

        def score_responses(self, example, response_texts):
            self.batch_sizes.append(len(response_texts))
            return super().score_responses(example, response_texts)

    sdk_env = _UnsafeCountingSingleTurnEnv()
    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    assert env.reward_thread_safe is False

    ex_a = {"id": "a", "input": "2+2?", "output": "4"}
    ex_b = {"id": "b", "input": "3+3?", "output": "6"}
    items = [
        (ex_a, {"response_text": "the answer is 4"}),  # -> 1.0
        (ex_b, {"response_text": "it is 6"}),  # -> 1.0
        (ex_a, {"response_text": "nope"}),  # -> 0.0
    ]
    sdk_env.batch_sizes = []

    out = env.reward_many(items)

    assert out == [1.0, 1.0, 0.0]  # correct + input order
    # ex_a's two rollouts were NOT batched: every score_responses call carried exactly one response.
    assert sdk_env.batch_sizes == [1, 1, 1]


def test_single_turn_scores_breakdown_many_batches_named_metrics_in_order(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    class _CountingSingleTurnEnv(_FakeSingleTurnEnv):
        def __init__(self):
            self.batch_sizes = []

        def score_responses(self, example, response_texts):
            self.batch_sizes.append(len(response_texts))
            return super().score_responses(example, response_texts)

    sdk_env = _CountingSingleTurnEnv()
    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    ex_a = {"id": "a", "input": "2+2?", "output": "4"}
    ex_b = {"id": "b", "input": "3+3?", "output": "6"}

    breakdowns = env.scores_breakdown_many(
        [
            (ex_a, {"response_text": "4"}),
            (ex_b, {"response_text": "nope"}),
            (ex_a, {"response_text": "also 4"}),
        ]
    )

    assert breakdowns == [
        {"match": 1.0, "total": 1.0},
        {"match": 0.0, "total": 0.0},
        {"match": 1.0, "total": 1.0},
    ]
    assert sorted(sdk_env.batch_sizes) == [1, 2]


def test_single_turn_scores_breakdown_many_is_serial_when_not_thread_safe(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    class _UnsafeSingleTurnEnv(_FakeSingleTurnEnv):
        reward_thread_safe = False

        def __init__(self):
            self.batch_sizes = []

        def score_responses(self, example, response_texts):
            self.batch_sizes.append(len(response_texts))
            return super().score_responses(example, response_texts)

    sdk_env = _UnsafeSingleTurnEnv()
    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    example = {"id": "a", "input": "2+2?", "output": "4"}

    assert env.scores_breakdown_many(
        [
            (example, {"response_text": "4"}),
            (example, {"response_text": "nope"}),
        ]
    ) == [
        {"match": 1.0, "total": 1.0},
        {"match": 0.0, "total": 0.0},
    ]
    assert sdk_env.batch_sizes == [1, 1]


def test_single_turn_scores_breakdown_many_rejects_wrong_length(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    class _ShortSingleTurnEnv(_FakeSingleTurnEnv):
        def score_responses(self, example, response_texts):
            return super().score_responses(example, response_texts[:-1])

    env = FreesoloEnvironment(_ShortSingleTurnEnv(), "owner/env", source=None, contract_text="")
    example = {"id": "a", "input": "2+2?", "output": "4"}

    with pytest.raises(RuntimeError, match="score_responses returned the wrong length"):
        env.scores_breakdown_many(
            [
                (example, {"response_text": "4"}),
                (example, {"response_text": "also 4"}),
            ]
        )


def test_single_turn_scoring_gets_completion_thinking_and_raw(monkeypatch):
    """Thinking-mode GRPO passes an answer-only string to existing scorers while exposing
    structured thinking/raw fields for scorers that need them."""
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    class _ThinkingAwareEnv(_EnvironmentSingleTurn):
        def start_episode(self, example, prompt_text):
            return [{"role": "user", "content": example.input}]

        def score_responses(self, example, response_texts):
            response = response_texts[0]
            assert response == " 5"
            assert response.completion == " 5"
            assert response.thinking == "4"
            assert response.raw == "<think>4</think> 5"
            legacy_score = 1.0 if str(example.output) in response else 0.0
            thinking_score = 1.0 if response.thinking == "4" else 0.0
            raw_score = 1.0 if "<think>4</think>" in response.raw else 0.0
            answer_score = 1.0 if response.completion == " 5" else 0.0
            return [
                _RewardResult(
                    score=legacy_score + thinking_score + raw_score + answer_score,
                    metrics=(
                        _RewardMetric("legacy_answer_match", legacy_score),
                        _RewardMetric("thinking", thinking_score),
                        _RewardMetric("raw_reasoning", raw_score),
                        _RewardMetric("answer", answer_score),
                    ),
                )
            ]

    env = FreesoloEnvironment(_ThinkingAwareEnv(), "owner/env", source=None, contract_text="")
    raw = "<think>4</think> 5"
    state = {"completion": " 5", "thinking": "4", "raw": raw}

    breakdown = env.scores_breakdown(" 5", {"id": "q", "input": "q", "output": "4"}, state)

    assert breakdown["legacy_answer_match"] == 0.0
    assert breakdown["thinking"] == 1.0
    assert breakdown["raw_reasoning"] == 1.0
    assert breakdown["answer"] == 1.0
    assert breakdown["total"] == 3.0
    assert env.scores_breakdown_many(
        [
            (
                {"id": "q", "input": "q", "output": "4"},
                {"response_text": " 5", **state},
            )
        ]
    ) == [breakdown]


def test_freesolo_sft_completion_full_gold_trajectory(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(_FakeSingleTurnEnv(), "owner/env", source=None, contract_text="")
    # A record whose `output` is a chat-message list -> the full gold trajectory (multi-turn SFT).
    gold = [
        {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        {"role": "user", "content": "<tool_result>...</tool_result>"},
        {"role": "assistant", "content": "done"},
    ]
    assert env.sft_completion({"input": "x", "output": gold}) == gold  # len>1 -> multi-turn
    assert env.sft_completion({"input": "x", "output": {"messages": gold}}) == gold
    # A scalar `output` is single-turn SFT -> one assistant turn.
    assert env.sft_completion({"input": "x", "output": "4"}) == [
        {"role": "assistant", "content": "4"}
    ]
    assert env.sft_completion({"input": "x"}) == [{"role": "assistant", "content": ""}]
    assert env.sft_completion({"input": "x", "output": None}) == [
        {"role": "assistant", "content": ""}
    ]
    assert env.sft_completion({"input": "x", "output": []}) == [
        {"role": "assistant", "content": "[]"}
    ]
    assert env.sft_completion({"input": "x", "output": ["red", "blue"]}) == [
        {"role": "assistant", "content": "['red', 'blue']"}
    ]


def test_freesolo_sft_completion_reports_raw_output_fallback_provenance(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    class _HookEnv(_FakeSingleTurnEnv):
        def __init__(self):
            self.calls = 0

        def sft_completion(self, example):
            self.calls += 1
            return [{"role": "assistant", "content": "from hook"}]

    sdk_env = _HookEnv()
    hook_env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    messages, coerced_scalar_output = hook_env.sft_completion_with_provenance(
        {"input": "x", "output": "raw"}
    )

    assert messages == [{"role": "assistant", "content": "from hook"}]
    assert coerced_scalar_output is False
    assert sdk_env.calls == 1

    fallback_env = FreesoloEnvironment(
        _FakeSingleTurnEnv(), "owner/env", source=None, contract_text=""
    )
    messages, coerced_scalar_output = fallback_env.sft_completion_with_provenance(
        {"input": "x", "output": "raw"}
    )

    assert messages == [{"role": "assistant", "content": "raw"}]
    assert coerced_scalar_output is True


def test_freesolo_sft_completion_does_not_flag_structured_targets_as_coerced(monkeypatch):
    """An explicitly structured target is NOT a scalar coercion.

    a message list and a {"messages": [...]} container both encode a real trajectory, so counting
    them as coerced would make the collapse warning fire on datasets that already use the
    supported encoding -- telling users to do the thing they are already doing.
    """
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(_FakeSingleTurnEnv(), "owner/env", source=None, contract_text="")
    single = [{"role": "assistant", "content": "structured"}]

    messages, coerced_scalar_output = env.sft_completion_with_provenance(
        {"input": "x", "output": single}
    )
    assert messages == single
    assert coerced_scalar_output is False

    messages, coerced_scalar_output = env.sft_completion_with_provenance(
        {"input": "x", "output": {"messages": single}}
    )
    assert messages == single
    assert coerced_scalar_output is False

    # a scalar gold answer IS a coercion, so the flag still separates the two.
    _messages, coerced_scalar_output = env.sft_completion_with_provenance(
        {"input": "x", "output": "42"}
    )
    assert coerced_scalar_output is True


@pytest.mark.parametrize(
    ("row_id", "output", "actionable_detail"),
    [
        (
            "extra-key",
            {"messages": [{"role": "assistant", "content": "4"}], "meta": 1},
            "sibling keys ['meta']",
        ),
        ("not-list", {"messages": "bad"}, "'messages' value of type str"),
        (
            "bad-nested-entry",
            {"messages": [{"role": "assistant", "content": "4"}, "bad"]},
            "'messages' entries at indexes [1]",
        ),
        (
            "bad-list-entry",
            [{"role": "assistant", "content": "4"}, "bad"],
            "non-object output entries at indexes [1]",
        ),
    ],
)
def test_freesolo_sft_completion_rejects_malformed_message_containers(
    monkeypatch, row_id, output, actionable_detail
):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(_FakeSingleTurnEnv(), "owner/env", source=None, contract_text="")
    embedded_image = "data:image/png;base64," + "A" * 4096
    record = {"id": row_id, "input": "x", "output": output, "image": embedded_image}

    with pytest.raises(ValueError, match="sft output for row id") as exc_info:
        env.sft_completion(record)

    message = str(exc_info.value)
    assert f"row id {row_id!r}" in message
    assert actionable_detail in message
    assert embedded_image not in message
    assert "data:image/png;base64" not in message


def test_freesolo_sft_completion_error_handles_missing_row_id(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(_FakeSingleTurnEnv(), "owner/env", source=None, contract_text="")

    with pytest.raises(ValueError, match=r"row id None.*'messages' value of type str"):
        env.sft_completion({"input": "x", "output": {"messages": "bad"}})


def test_freesolo_multiturn_respects_per_example_budget(monkeypatch):
    _install_fake_freesolo(monkeypatch, sdk_env=_BudgetMultiTurnEnv())

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _BudgetMultiTurnEnv(),
        "owner/env",
        source=[{"id": "go", "input": "go", "output": ""}],
        contract_text="",
    )
    # max_turns is now the dataset-wide max_episode_turns (15), not the old hardcoded 8 —
    # otherwise the rollout loop would truncate this 15-turn scenario at turn 8.
    assert env.max_turns == 15

    state = env.new_rollout_state({"id": "go", "input": "go", "output": ""})
    assert state["max_episode_turns"] == 15
    # rollout_done honors THIS rollout's budget even when the batch cap is larger, and even
    # though the env never sets done=True.
    state["turn"] = 14
    assert env.rollout_done(state, max_turns=999) is False
    state["turn"] = 15
    assert env.rollout_done(state, max_turns=999) is True
    # done=True still short-circuits before the budget.
    state["turn"] = 0
    state["done"] = True
    assert env.rollout_done(state, max_turns=999) is True


def _install_fake_freesolo(monkeypatch, *, sdk_env=None, seen=None):
    sdk_env = sdk_env or _FakeSingleTurnEnv()
    seen = seen if seen is not None else {}

    def task_example_from_record(record):
        return _TaskExample(
            record=dict(record),
            input=str(record["input"]),
            id=record.get("id"),
            output=record.get("output"),
            metadata=dict(record.get("metadata") or {}),
        )

    def load_task_examples(source):
        if isinstance(source, (list, tuple)):
            return [
                item if isinstance(item, _TaskExample) else task_example_from_record(item)
                for item in source
            ]
        path = os.fspath(source)
        rows = []
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            rows = loaded if isinstance(loaded, list) else loaded.get("records", [])
        return [task_example_from_record(row) for row in rows]

    def load_environment(reference, **kwargs):
        seen["reference"] = reference
        seen["kwargs"] = kwargs
        return sdk_env

    freesolo = types.ModuleType("freesolo")
    datasets = types.ModuleType("freesolo.datasets")
    records = types.ModuleType("freesolo.datasets.records")
    records.load_task_examples = load_task_examples
    records.task_example_from_record = task_example_from_record
    envs = types.ModuleType("freesolo.environments")
    envs.EnvironmentEpisode = _EnvironmentEpisode
    envs.EnvironmentMultiTurn = _EnvironmentMultiTurn
    envs.EnvironmentSingleTurn = _EnvironmentSingleTurn
    envs.EnvironmentStepResult = _EnvironmentStepResult
    envs.EnvironmentTurn = _EnvironmentTurn
    envs.RewardMetric = _RewardMetric
    envs.RewardResult = _RewardResult
    envs.load_environment = load_environment
    monkeypatch.setitem(sys.modules, "freesolo", freesolo)
    monkeypatch.setitem(sys.modules, "freesolo.datasets", datasets)
    monkeypatch.setitem(sys.modules, "freesolo.datasets.records", records)
    monkeypatch.setitem(sys.modules, "freesolo.environments", envs)
    return seen


def _github_environment_tarball(
    top_dir: str,
    *,
    env_path: str = "envs/e/environment.py",
    env_text: str = "def load_environment(**kwargs):\n    return None\n",
) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        dir_info = tarfile.TarInfo(f"{top_dir}/")
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        tar.addfile(dir_info)

        env_info = tarfile.TarInfo(f"{top_dir}/{env_path}")
        env_bytes = env_text.encode("utf-8")
        env_info.size = len(env_bytes)
        env_info.mode = 0o644
        tar.addfile(env_info, io.BytesIO(env_bytes))
    return buf.getvalue()


def test_freesolo_adapter_mapping(monkeypatch, tmp_path):
    seen = _install_fake_freesolo(monkeypatch)
    env_file = tmp_path / "freesolo" / "environment.py"
    env_file.parent.mkdir()
    env_file.write_text("def load_environment(**kwargs): pass\n")
    dataset = tmp_path / "freesolo" / "dataset" / "train.jsonl"
    dataset.parent.mkdir()
    dataset.write_text(
        '{"id":"row-1","input":"2+2?","output":"4",'
        '"image":"dataset/red.png","reward_metadata":{"kind":"exact"}}\n'
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file),
        dataset_path="dataset/train.jsonl",
        contract_text="be brief",
        difficulty="hard",
    )

    assert env.id == str(env_file)
    assert env.package_root == env_file.parent.resolve()
    assert seen["reference"] == str(env_file)
    assert seen["kwargs"]["dataset_path"] == str(dataset)
    assert seen["kwargs"]["difficulty"] == "hard"

    train = env.dataset()
    assert train == [
        {
            "id": "row-1",
            "input": "2+2?",
            "output": "4",
            "image": "dataset/red.png",
            "reward_metadata": {"kind": "exact"},
        }
    ]
    assert env.prompt_messages(train[0]) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "2+2?"},
    ]
    image_content = [
        {"type": "text", "text": "what color?"},
        {"type": "image", "image": "dataset/red.png"},
    ]
    assert env._with_system_prompt([{"role": "user", "content": image_content}]) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": image_content},
    ]
    assert env.reward("the answer is 4", train[0]) == 1.0
    assert env.grade("the answer is 4", train[0]) is True
    assert env.reward("nope", train[0]) == 0.0
    assert env.scores_breakdown("the answer is 4", train[0]) == {"match": 1.0, "total": 1.0}
    assert env.sft_completion({"output": "4"}) == [{"role": "assistant", "content": "4"}]


def _split_env(tmp_path, extra_files):
    env_file = tmp_path / "freesolo" / "environment.py"
    env_file.parent.mkdir()
    env_file.write_text("def load_environment(**kwargs): pass\n")
    for rel, text in extra_files.items():
        f = tmp_path / "freesolo" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return env_file


def test_freesolo_adapter_split_param_selects_split_dataset(monkeypatch, tmp_path):
    """[environment.params] split must pick dataset/<split>.jsonl for Flash's own dataset()
    (SFT targets AND GRPO problem selection), not silently train on the default train.jsonl."""
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "dataset/oracle.jsonl": '{"id":"o","input":"2+2?","output":"4"}\n',
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), split="oracle", contract_text="c")
    assert env.dataset() == [{"id": "o", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_forwards_the_normalized_split_to_the_env(monkeypatch, tmp_path):
    """a padded split name selects the right file locally; the env must see the same name."""
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "dataset/oracle.jsonl": '{"id":"o","input":"2+2?","output":"4"}\n',
        },
    )
    env_file.write_text(
        "def load_environment(**kwargs):\n"
        "    assert kwargs.get('split') == 'oracle', repr(kwargs.get('split'))\n"
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), split=" oracle ", contract_text="c")
    assert env.dataset() == [{"id": "o", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_missing_split_file_refuses_silent_train_fallback(monkeypatch, tmp_path):
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path, {"dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n'}
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    with pytest.raises(ValueError, match="split='oracle'"):
        load_freesolo_environment(str(env_file), split="oracle", contract_text="c")


def test_freesolo_adapter_rejects_unsafe_split_names(monkeypatch, tmp_path):
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path, {"dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n'}
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    with pytest.raises(ValueError, match="split must be a simple dataset name"):
        load_freesolo_environment(str(env_file), split="../oracle", contract_text="c")


def test_freesolo_adapter_split_train_uses_default_dataset(monkeypatch, tmp_path):
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path, {"dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n'}
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), split="train", contract_text="c")
    assert env.dataset() == [{"id": "t", "input": "train?", "output": "no"}]


def test_freesolo_adapter_datasets_plural_dir_raises_actionable_error(monkeypatch, tmp_path):
    """a package with datasets/ (plural) but no dataset/ has no dataset flash can probe, so it
    must fail loudly and name the expected directory rather than resolve rows from elsewhere."""
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path, {"datasets/train.jsonl": '{"id":"legacy","input":"old?","output":"old"}\n'}
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    with pytest.raises(ValueError, match="'datasets/' directory"):
        load_freesolo_environment(str(env_file), split="train", contract_text="c")


def test_freesolo_adapter_datasets_plural_dir_allowed_with_explicit_dataset_path(
    monkeypatch, tmp_path
):
    """explicit [environment.params] dataset_path resolves the datasets/ ambiguity, so the
    loader must not raise and must train on the named file."""
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path, {"datasets/train.jsonl": '{"id":"t","input":"x","output":"y"}\n'}
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file), dataset_path="datasets/train.jsonl", contract_text="c"
    )
    assert env.dataset() == [{"id": "t", "input": "x", "output": "y"}]


def test_freesolo_adapter_datasets_plural_dir_allowed_beside_a_probeable_split(
    monkeypatch, tmp_path
):
    """datasets/ next to a supported top-level <split>.jsonl is a package carrying other assets,
    not the ambiguous layout: the probe resolves rows, so the guard must stay quiet."""
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {
            "train.jsonl": '{"id":"t","input":"2+2?","output":"4"}\n',
            "datasets/eval.jsonl": '{"id":"e","input":"held?","output":"out"}\n',
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), contract_text="c")
    assert env.dataset() == [{"id": "t", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_env_built_dataset_wins_over_packaged_file(monkeypatch, tmp_path, capsys):
    """an env that filters/subsamples its dataset in load_environment (the documented
    single-turn.mdx pattern) is what flash trains on; the packaged train.jsonl is only a
    fallback, and the override is logged with both row counts."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "kept", "input": "2+2?", "output": "4"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"id":"kept","input":"2+2?","output":"4"}\n'
                '{"id":"dropped","input":"3+3?","output":"6"}\n'
            )
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), contract_text="c")
    assert env.dataset() == [{"id": "kept", "input": "2+2?", "output": "4"}]
    logged = capsys.readouterr().out
    assert "environment's own dataset (1 rows)" in logged
    assert "(2 rows)" in logged


def test_freesolo_adapter_skips_the_row_count_of_a_large_json_file(monkeypatch, tmp_path, capsys):
    """the override diagnostic has to parse a whole .json file to count it, and the env has
    already replaced it: past the cap it gives the count up instead of materializing the file."""
    from flash.envs.loading import adapter as adapter_module

    monkeypatch.setattr(adapter_module, "_MAX_ROW_COUNT_JSON_BYTES", 8)
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "kept", "input": "2+2?", "output": "4"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.json": (
                '[{"id":"kept","input":"2+2?","output":"4"},'
                '{"id":"dropped","input":"3+3?","output":"6"}]'
            )
        },
    )

    env = adapter_module.load_freesolo_environment(str(env_file), contract_text="c")
    assert env.dataset() == [{"id": "kept", "input": "2+2?", "output": "4"}]
    assert "environment's own dataset" not in capsys.readouterr().out


def test_freesolo_adapter_env_dataset_matching_file_logs_nothing(monkeypatch, tmp_path, capsys):
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "t", "input": "2+2?", "output": "4"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path, {"dataset/train.jsonl": '{"id":"t","input":"2+2?","output":"4"}\n'}
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), contract_text="c")
    assert env.dataset() == [{"id": "t", "input": "2+2?", "output": "4"}]
    assert "environment's own dataset" not in capsys.readouterr().out


def test_freesolo_adapter_requested_split_wins_over_a_hardcoded_env_dataset(monkeypatch, tmp_path):
    """the scaffolded pattern is a class-level dataset pinned to dataset/train.jsonl that ignores
    the dataset_path it is handed, so an explicitly requested split must stay authoritative."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "t", "input": "train?", "output": "no"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "dataset/oracle.jsonl": '{"id":"o","input":"2+2?","output":"4"}\n',
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), split="oracle", contract_text="c")
    assert env.dataset() == [{"id": "o", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_explicit_dataset_path_wins_over_a_hardcoded_env_dataset(
    monkeypatch, tmp_path
):
    """an explicit non-default dataset_path names the rows to train on, and the scaffolded
    class-level dataset pattern ignores the injected path, so the named file must stay
    authoritative over the env's default train rows."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "t", "input": "train?", "output": "no"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "dataset/oracle.jsonl": '{"id":"o","input":"2+2?","output":"4"}\n',
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file), dataset_path="dataset/oracle.jsonl", contract_text="c"
    )
    assert env.dataset() == [{"id": "o", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_default_dataset_path_keeps_env_precedence(monkeypatch, tmp_path):
    """dataset_path naming the default train file is the documented filter-the-injected-file
    pattern, so the env's (possibly filtered) rows keep winning over re-reading the file."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "kept", "input": "2+2?", "output": "4"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"id":"kept","input":"2+2?","output":"4"}\n'
                '{"id":"dropped","input":"3+3?","output":"6"}\n'
            )
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file), dataset_path="dataset/train.jsonl", contract_text="c"
    )
    assert env.dataset() == [{"id": "kept", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_empty_env_dataset_is_a_hard_error(monkeypatch, tmp_path):
    """this used to fall back to the packaged file; it now raises, and the change is the point.

    an env that sets `dataset` to a filter result that matched nothing has SAID which rows are
    trainable: none. re-reading the unfiltered file behind its back trains on exactly the rows it
    rejected, and nothing in the run surfaces that -- a silent wrong-data run is worse than a
    hard failure the operator sees on the first step. `examples` still reads the same way as
    `dataset`, so neither attribute can be the quiet one.
    """
    from flash.envs.loading.adapter import load_freesolo_environment

    rows = '{"id":"t","input":"2+2?","output":"4"}\n'
    for attribute in ("dataset", "examples"):
        sdk_env = _FakeSingleTurnEnv()
        setattr(sdk_env, attribute, [])
        _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
        root = tmp_path / attribute
        root.mkdir()
        env_file = _split_env(root, {"dataset/train.jsonl": rows})

        env = load_freesolo_environment(str(env_file), contract_text="c")
        with pytest.raises(ValueError, match="environment produced 0 rows"):
            env.dataset()


def test_freesolo_adapter_empty_env_dataset_errors_under_default_dataset_path(
    monkeypatch, tmp_path
):
    """dataset_path naming the default train file keeps env precedence, so an env that filtered
    every row away must fail there too rather than re-read the file it was handed."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = []
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"id":"kept","input":"2+2?","output":"4"}\n'
                '{"id":"dropped","input":"3+3?","output":"6"}\n'
            )
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file), dataset_path="dataset/train.jsonl", contract_text="c"
    )
    with pytest.raises(ValueError, match="environment produced 0 rows"):
        env.dataset()


def test_freesolo_adapter_empty_env_dataset_yields_to_an_explicit_dataset_path(
    monkeypatch, tmp_path
):
    """an explicit non-default dataset_path is already authoritative over the env's dataset, so
    an empty one is not a rejection of those rows and must not block the run."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = []
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "dataset/oracle.jsonl": '{"id":"o","input":"2+2?","output":"4"}\n',
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file), dataset_path="dataset/oracle.jsonl", contract_text="c"
    )
    assert env.dataset() == [{"id": "o", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_absent_env_dataset_still_falls_back_to_the_packaged_file(
    monkeypatch, tmp_path
):
    """no dataset attribute (or an explicit None) is "no in-code dataset", not "zero rows": the
    packaged file stays the source, which is the common env and must not start erroring."""
    from flash.envs.loading.adapter import load_freesolo_environment

    rows = '{"id":"t","input":"2+2?","output":"4"}\n'
    for label in ("absent", "none"):
        sdk_env = _FakeSingleTurnEnv()
        if label == "none":
            sdk_env.dataset = None
            sdk_env.examples = None
        _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
        root = tmp_path / label
        root.mkdir()
        env_file = _split_env(root, {"dataset/train.jsonl": rows})

        env = load_freesolo_environment(str(env_file), contract_text="c")
        assert env.dataset() == [{"id": "t", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_datasets_plural_dir_allowed_when_the_env_owns_its_rows(
    monkeypatch, tmp_path
):
    """an env that builds every row in load_environment never reads the packaged file, so a
    datasets/ directory is raw or eval assets there: the guard must not stop it from loading."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "built", "input": "2+2?", "output": "4"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path, {"datasets/raw.jsonl": '{"id":"raw","input":"raw?","output":"raw"}\n'}
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), contract_text="c")
    assert env.dataset() == [{"id": "built", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_datasets_plural_dir_raises_when_the_env_needs_the_file(
    monkeypatch, tmp_path
):
    """the guard is deferred, not dropped: an env with no rows of its own still depends on the
    file the datasets/ layout hid, and an env whose dataset came back empty does too. both get
    the layout message, which names the fix, over the adapter's generic empty-dataset one."""
    from flash.envs.loading.adapter import load_freesolo_environment

    for label, rows in (("absent", None), ("empty", [])):
        sdk_env = _FakeSingleTurnEnv()
        if rows is not None:
            sdk_env.dataset = rows
        _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
        root = tmp_path / label
        root.mkdir()
        env_file = _split_env(
            root, {"datasets/train.jsonl": '{"id":"legacy","input":"old?","output":"old"}\n'}
        )

        with pytest.raises(ValueError, match="'datasets/' directory"):
            load_freesolo_environment(str(env_file), contract_text="c")


def test_freesolo_adapter_datasets_plural_dir_raises_for_a_requested_side_split(
    monkeypatch, tmp_path
):
    """deferring the guard because the env exposes rows must not swallow an explicit side
    split: the layout hid any packaged split file, and rows the env supplies in code cannot
    be verified against the requested split, so loading them would silently train on the
    wrong split -- the exact failure the split rule exists to stop."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "built", "input": "2+2?", "output": "4"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path, {"datasets/oracle.jsonl": '{"id":"o","input":"o?","output":"o"}\n'}
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    with pytest.raises(ValueError, match="'datasets/' directory"):
        load_freesolo_environment(str(env_file), split="oracle", contract_text="c")


def test_freesolo_adapter_datasets_plural_split_raises_beside_a_singular_dataset_dir(
    monkeypatch, tmp_path
):
    """a package can hold BOTH directories. when the requested split resolves under neither
    dataset/ nor a top-level file but is sitting unread in datasets/, the singular directory
    must not silence the guard: precedence would otherwise hand back the env's own train rows
    and train on the wrong split. the message names both paths."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "built", "input": "2+2?", "output": "4"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/dev.jsonl": '{"id":"d","input":"dev?","output":"dev"}\n',
            "datasets/oracle.jsonl": '{"id":"o","input":"o?","output":"o"}\n',
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    with pytest.raises(ValueError, match="'datasets/' directory") as excinfo:
        load_freesolo_environment(str(env_file), split="oracle", contract_text="c")
    message = str(excinfo.value)
    assert "datasets/oracle.jsonl" in message
    assert "dataset/oracle.jsonl" in message


def test_freesolo_adapter_datasets_plural_split_raises_beside_a_packaged_train_file(
    monkeypatch, tmp_path
):
    """same layout with a default dataset/train.jsonl present: the split is still unread under
    datasets/, so the layout error (which names the file that exists) replaces the generic
    missing-split message rather than leaving the operator to guess."""
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "datasets/oracle.jsonl": '{"id":"o","input":"o?","output":"o"}\n',
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    with pytest.raises(ValueError, match="'datasets/' directory") as excinfo:
        load_freesolo_environment(str(env_file), split="oracle", contract_text="c")
    message = str(excinfo.value)
    assert "datasets/oracle.jsonl" in message


def test_freesolo_adapter_datasets_plural_dir_allowed_beside_a_probeable_split_dir(
    monkeypatch, tmp_path
):
    """the raw-assets case must stay unaffected: a split that resolves normally under dataset/
    trains on that file even when a datasets/ directory carries a file of the same name."""
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "dataset/oracle.jsonl": '{"id":"o","input":"2+2?","output":"4"}\n',
            "datasets/oracle.jsonl": '{"id":"raw","input":"raw?","output":"raw"}\n',
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), split="oracle", contract_text="c")
    assert env.dataset() == [{"id": "o", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_records_param_wins_over_env_dataset(monkeypatch, tmp_path):
    """explicit [environment.params] records never reach the sdk env, so they keep
    precedence over a hardcoded env dataset."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "envrow", "input": "e?", "output": "e"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(tmp_path, {})

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file),
        records=[{"id": "r", "input": "r?", "output": "r"}],
        contract_text="c",
    )
    assert env.dataset() == [{"id": "r", "input": "r?", "output": "r"}]


def test_freesolo_adapter_explicit_dataset_path_wins_over_split(monkeypatch, tmp_path):
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "dataset/oracle.jsonl": '{"id":"o","input":"2+2?","output":"4"}\n',
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file),
        dataset_path="dataset/train.jsonl",
        split="oracle",
        contract_text="c",
    )
    assert env.dataset() == [{"id": "t", "input": "train?", "output": "no"}]


def test_freesolo_adapter_explicit_default_train_path_beats_env_rows_under_side_split(
    monkeypatch, tmp_path
):
    """dataset_path naming the default train file normally keeps env precedence (the documented
    filtering pattern), but combined with an explicit side split that precedence would let an
    env honoring split= deliver the side rows against the file the operator named. the codified
    rule -- an explicit dataset_path wins over split -- has to hold even when the env exposes
    rows, so the explicitly authored path stays authoritative."""
    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "o", "input": "2+2?", "output": "4"}]
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "dataset/oracle.jsonl": '{"id":"o","input":"2+2?","output":"4"}\n',
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file),
        dataset_path="dataset/train.jsonl",
        split="oracle",
        contract_text="c",
    )
    assert env.dataset() == [{"id": "t", "input": "train?", "output": "no"}]


def test_freesolo_adapter_preserves_top_level_record_keys(monkeypatch, tmp_path):
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"id":"t","input":"x","output":"y","initial_state":[1,2],'
                '"metadata":{"kept":true}}\n'
            )
        },
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), contract_text="c")
    rows = env.dataset()
    assert rows == [
        {
            "id": "t",
            "input": "x",
            "output": "y",
            "initial_state": [1, 2],
            "metadata": {"kept": True},
        }
    ]


def test_freesolo_adapter_no_longer_requires_record_id(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(_FakeSingleTurnEnv(), "owner/env", source=None, contract_text="")

    # Ids are auto-generated, so a record with no id (or a blank one) is accepted
    # rather than rejected; only a missing input is an error.
    assert env._canonical_record({"input": "x"}) == {"input": "x"}
    assert env._canonical_record({"id": "   ", "input": "x"}) == {"id": "   ", "input": "x"}
    with pytest.raises(ValueError, match="input field"):
        env._canonical_record({"id": "1"})


def test_freesolo_adapter_allows_missing_output(monkeypatch, tmp_path):
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {"dataset/train.jsonl": '{"id":"t","input":"x","difficulty":"easy"}\n'},
    )

    from flash.envs.loading.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), contract_text="c")
    assert env.dataset() == [{"id": "t", "input": "x", "difficulty": "easy"}]


def test_freesolo_adapter_prepends_missing_contract_system_prompt(monkeypatch):
    class NoSystemEnv(_EnvironmentSingleTurn):
        dataset: ClassVar[list[dict]] = [{"id": "math", "input": "2+2?", "output": "4"}]

        def start_episode(self, example, prompt_text):
            return [{"role": "user", "content": example.input}]

        def score_responses(self, example, response_texts):
            return [_RewardResult(score=0.0) for _ in response_texts]

    _install_fake_freesolo(monkeypatch, sdk_env=NoSystemEnv())

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        NoSystemEnv(),
        "owner/env",
        source=None,
        contract_text="follow the contract",
    )

    assert env.prompt_messages({"id": "math", "input": "2+2?", "output": "4"}) == [
        {"role": "system", "content": "follow the contract"},
        {"role": "user", "content": "2+2?"},
    ]


def test_freesolo_adapter_fills_blank_contract_system_prompt(monkeypatch):
    class BlankSystemEnv(_EnvironmentSingleTurn):
        dataset: ClassVar[list[dict]] = [{"id": "math", "input": "2+2?", "output": "4"}]

        def start_episode(self, example, prompt_text):
            return [
                {"role": "system", "content": "   "},
                {"role": "user", "content": example.input},
            ]

        def score_responses(self, example, response_texts):
            return [_RewardResult(score=0.0) for _ in response_texts]

    _install_fake_freesolo(monkeypatch, sdk_env=BlankSystemEnv())

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        BlankSystemEnv(),
        "owner/env",
        source=None,
        contract_text="follow the contract",
    )

    expected = [
        {"role": "system", "content": "follow the contract"},
        {"role": "user", "content": "2+2?"},
    ]
    assert env.prompt_messages({"id": "math", "input": "2+2?", "output": "4"}) == expected
    state = env.new_rollout_state({"id": "math", "input": "2+2?", "output": "4"})
    assert state["prompt"] == expected
    assert state["messages"] == expected
    assert state["messages"] is not state["prompt"]
    state["messages"][0]["content"] = "changed"
    assert state["prompt"][0]["content"] == "follow the contract"


def test_freesolo_adapter_uses_env_dataset_when_no_source(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    sdk_env = _FakeSingleTurnEnv()
    sdk_env.dataset = [{"id": "ex-1", "input": "2+2?", "output": "4"}]
    env = FreesoloEnvironment(
        sdk_env,
        "github:owner/repo@main:env/environment.py",
        source=None,
        contract_text="",
    )
    assert env.dataset()[0]["output"] == "4"


def test_freesolo_adapter_exports_sdk_examples_as_input_output(monkeypatch):
    class SdkExampleEnv(_EnvironmentSingleTurn):
        dataset: ClassVar[list[_TaskExample]] = [
            _TaskExample(
                record={"id": "ex-1", "input": "2+2?"},
                input="2+2?",
                id="ex-1",
                output="4",
                metadata={"split": "train"},
            )
        ]

    _install_fake_freesolo(monkeypatch, sdk_env=SdkExampleEnv())

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        SdkExampleEnv(),
        "owner/env",
        source=None,
        contract_text="",
    )
    assert env.dataset() == [
        {
            "input": "2+2?",
            "output": "4",
            "id": "ex-1",
        }
    ]


def test_freesolo_adapter_stamps_example_id_onto_record(monkeypatch):
    class SdkExampleEnv(_EnvironmentSingleTurn):
        dataset: ClassVar[list[_TaskExample]] = [
            _TaskExample(record={"input": "2+2?"}, input="2+2?", id="ex-1", output="4")
        ]

    _install_fake_freesolo(monkeypatch, sdk_env=SdkExampleEnv())

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(SdkExampleEnv(), "owner/env", source=None, contract_text="")
    # The record carries no id; the SDK-assigned (auto-generated) example id is
    # stamped onto it instead of raising.
    assert env.dataset() == [{"input": "2+2?", "output": "4", "id": "ex-1"}]


def test_freesolo_adapter_does_not_accept_record_aliases(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _FakeSingleTurnEnv(),
        "owner/env",
        source=None,
        contract_text="",
    )
    # `expected_output` is NOT an alias for `output`, so there is no gold completion (empty turn).
    assert env.sft_completion({"expected_output": "4"}) == [{"role": "assistant", "content": ""}]
    with pytest.raises(ValueError, match="input field"):
        env.prompt_messages({"task": "2+2?", "output": "4"})


def test_freesolo_multiturn_hooks(monkeypatch):
    _install_fake_freesolo(monkeypatch, sdk_env=_FakeMultiTurnEnv())

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _FakeMultiTurnEnv(),
        "github:owner/repo@main:env/environment.py",
        source=[{"id": "browse", "input": "browse", "output": "done"}],
        contract_text="contract",
    )
    state = env.new_rollout_state({"id": "browse", "input": "browse", "output": "done"})
    assert state["prompt"] == [
        {"role": "system", "content": "contract"},
        {"role": "user", "content": "contract:browse"},
    ]
    assert state["messages"] == [
        {"role": "system", "content": "contract"},
        {"role": "user", "content": "contract:browse"},
    ]
    env.record_model_turn(state, "click")
    replies = env.env_reply(state["messages"], state)
    assert replies == [{"role": "user", "content": "observed click"}]
    assert state["done"] is True
    assert env.rollout_done(state) is True
    assert (
        env.reward("ignored", {"id": "browse", "input": "browse", "output": "done"}, state) == 0.5
    )
    assert (
        env.grade("ignored", {"id": "browse", "input": "browse", "output": "done"}, state) is True
    )


def test_github_environment_ref_parsing():
    from flash.envs.loading.adapter import (
        is_freesolo_environment_id,
        is_github_environment_ref,
        is_managed_environment_slug,
        managed_slug_to_github_ref,
    )

    assert is_github_environment_ref("github:owner/repo@dev:envs/e/environment.py")
    assert is_github_environment_ref("github:owner/repo")
    assert not is_github_environment_ref("github:owner/repo@main:/etc/passwd")
    assert not is_github_environment_ref("github:owner/repo/extra@main:envs/e/environment.py")
    assert not is_github_environment_ref("github:owner/repo@:envs/e/environment.py")
    assert is_github_environment_ref("https://github.com/owner/repo/blob/dev/envs/e/environment.py")
    assert is_github_environment_ref("https://github.com/owner/repo")
    assert not is_github_environment_ref("owner/env")
    assert not is_github_environment_ref("gsm8k")
    assert not is_github_environment_ref("github:owner/repo@main:../../etc/passwd")
    assert not is_github_environment_ref("https://github.com/owner/repo/blob/dev/../../etc/passwd")
    assert not is_github_environment_ref("https://github.com/owner/repo/blob/main:/etc/passwd")
    assert not is_github_environment_ref("https://github.com/owner/repo/issues/1")
    assert not is_github_environment_ref("github:owner /repo@main:envs/e/environment.py")
    assert not is_github_environment_ref("github:owner/repo@bad/ref:envs/e/environment.py")
    assert not is_github_environment_ref(
        "https://github.com/owner/repo/blob/bad ref/envs/e/environment.py"
    )
    assert is_managed_environment_slug("owner/project/env")
    assert is_freesolo_environment_id("owner/project/env")
    assert managed_slug_to_github_ref("owner/project/env") == (
        "github:freesolo-co/environment-hub@main:owner/project/env/environment.py"
    )
    assert not is_managed_environment_slug("owner/project/env/extra")
    assert not is_freesolo_environment_id("gsm8k")


def test_github_environment_resolves_by_commit_sha(tmp_path, monkeypatch):
    import flash.envs.loading.loader as adapter

    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(
        adapter,
        "_resolve_ref_sha",
        lambda parsed, **_: "b" * 40,
    )

    downloads: list[str] = []

    def fake_download(ref):
        downloads.append(ref.ref)
        return _github_environment_tarball("repo-root")

    monkeypatch.setattr(adapter, "_download_github_tarball", fake_download)

    resolved = adapter._resolve_environment_reference(
        "github:owner/repo@main:envs/e/environment.py"
    )
    assert resolved.endswith("envs/e/environment.py")
    assert downloads == ["b" * 40]


def test_github_environment_directory_ref_uses_environment_entrypoint(tmp_path, monkeypatch):
    import flash.envs.loading.loader as adapter

    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **_: "b" * 40)

    downloads: list[str] = []

    def fake_download(ref):
        downloads.append(ref.ref)
        return _github_environment_tarball("repo-root")

    monkeypatch.setattr(adapter, "_download_github_tarball", fake_download)

    resolved = adapter._resolve_environment_reference("github:owner/repo@main:envs/e")
    assert resolved.endswith("envs/e/environment.py")
    assert adapter._resolve_environment_reference("github:owner/repo@main:envs/e") == resolved
    assert downloads == ["b" * 40]


def test_github_tree_url_ending_at_freesolo_dir_uses_single_entrypoint(tmp_path, monkeypatch):
    import flash.envs.loading.loader as adapter

    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **_: "b" * 40)

    downloads: list[str] = []

    def fake_download(ref):
        downloads.append(ref.path)
        return _github_environment_tarball(
            "repo-root",
            env_path="envs/e/freesolo/environment.py",
        )

    monkeypatch.setattr(adapter, "_download_github_tarball", fake_download)

    resolved = adapter._resolve_environment_reference(
        "https://github.com/owner/repo/tree/main/envs/e/freesolo"
    )
    assert resolved.endswith("envs/e/freesolo/environment.py")
    assert "freesolo/freesolo" not in resolved
    assert downloads == ["envs/e/freesolo/environment.py"]


def test_github_environment_directory_ref_missing_entrypoint_error(tmp_path, monkeypatch):
    import flash.envs.loading.loader as adapter

    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **_: "b" * 40)
    monkeypatch.setattr(
        adapter,
        "_download_github_tarball",
        lambda ref: _github_environment_tarball("repo-root", env_path="envs/e/helper.py"),
    )

    with pytest.raises(FileNotFoundError, match=r"envs/e/environment\.py"):
        adapter._resolve_environment_reference("github:owner/repo@main:envs/e")


def test_safe_extract_archive_rejects_unbounded_members_and_size(monkeypatch, tmp_path):
    from flash.envs.loading.loader import _safe_extract_archive

    def make_members_tar(members: list[tuple[str, bytes | None]]):
        tar = io.BytesIO()
        with tarfile.open(fileobj=tar, mode="w:gz") as handle:
            for name, payload in members:
                info = tarfile.TarInfo(name)
                if payload is None:
                    info.type = tarfile.DIRTYPE
                    info.size = 0
                    handle.addfile(info)
                else:
                    info.size = len(payload)
                    handle.addfile(info, io.BytesIO(payload))
        return tar.getvalue()

    dest = tmp_path / "extract_many"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.loading.loader._MAX_ARCHIVE_MEMBERS", 0)
    with pytest.raises(RuntimeError, match="too many members"):
        _safe_extract_archive(make_members_tar([("a", b"")]), dest)

    dest = tmp_path / "extract_big"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.loading.loader._MAX_ARCHIVE_MEMBERS", 5)
    monkeypatch.setattr("flash.envs.loading.loader._MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(RuntimeError, match="too large"):
        _safe_extract_archive(
            make_members_tar([("repo-root/", None), ("repo-root/keep.txt", b"xx")]), dest
        )

    dest = tmp_path / "extract_pax"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.loading.loader._MAX_ARCHIVE_BYTES", 100)
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w:gz") as handle:
        pax = tarfile.TarInfo("pax_global_header")
        pax.type = tarfile.XGLTYPE
        handle.addfile(pax)
        root = tarfile.TarInfo("repo-root/")
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        handle.addfile(root)
        file_info = tarfile.TarInfo("repo-root/environment.py")
        payload = b"def load_environment(**k):\n    return None\n"
        file_info.size = len(payload)
        handle.addfile(file_info, io.BytesIO(payload))
    assert _safe_extract_archive(tar.getvalue(), dest) == dest / "repo-root"


def test_safe_extract_archive_rejects_longname_decompression_bomb(tmp_path):
    # A GNU LONGNAME header payload is consumed inside tarfile.next() and never yielded, so per-member
    # accounting can't see it. A tiny gzip declaring a 400MB name must be rejected with memory bounded
    # near the limit, not OOM the worker.
    from flash.envs.loading.loader import _safe_extract_archive

    def header(name: str, size: int, typeflag: str) -> bytes:
        h = bytearray(512)
        nb = name.encode()[:100]
        h[0 : len(nb)] = nb
        h[100:108] = b"0000644\0"
        h[124 : 124 + 12] = f"{size:011o}\0".encode()
        h[136:148] = b"00000000000\0"
        h[156] = ord(typeflag)
        h[257:263] = b"ustar\0"
        h[263:265] = b"00"
        chk = sum(h[:148]) + sum(h[156:]) + 32 * 8
        h[148 : 148 + 8] = f"{chk:06o}\0 ".encode()
        return bytes(h)

    name_len = 400 * 1024 * 1024
    longlink = header("././@LongLink", name_len, "L")
    pad = b"\0" * ((512 - name_len % 512) % 512)
    tail = header("repo/environment.py", 1, "0") + b"x" + b"\0" * 511 + b"\0" * 1024
    buf = io.BytesIO()
    # Stream the 400 MB LONGNAME payload into gzip in fixed-size chunks instead of building it in
    # RAM twice (once as b"A"*name_len, once via bytes(raw)) — that peaks ~1 GB and can OOM CI. The
    # construction here also runs BEFORE tracemalloc.start() below, so it's pure setup overhead the
    # peak-memory assertion never covers. Chunked writes feed one zlib stream, so output is identical.
    block = b"A" * min(name_len, 1 << 20)
    with gzip.GzipFile(fileobj=buf, mode="wb") as g:
        g.write(longlink)
        remaining = name_len
        while remaining > 0:
            n = min(remaining, len(block))
            g.write(block if n == len(block) else block[:n])
            remaining -= n
        g.write(pad)
        g.write(tail)
    bomb = buf.getvalue()
    assert len(bomb) < 2 * 1024 * 1024

    dest = tmp_path / "extract_bomb"
    dest.mkdir()
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="too large"):
            _safe_extract_archive(bomb, dest)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 600 * 1024 * 1024, f"peak memory {peak} not bounded by the limit"


def test_worker_deps():
    import flash.envs.loading.base as registry

    env_id = "github:owner/repo@main:env/environment.py"
    assert registry.worker_pip_for_env(env_id) == ["freesolo>=0.4.2"]


# ============================================================================================
# MP-005 — single-turn and multi-turn grading must receive the same thinking-text shape
# ============================================================================================
class _ThinkingRecordingMultiTurnEnv(_EnvironmentMultiTurn):
    """Multi-turn env that records exactly what text its scorer was handed."""

    def __init__(self):
        self.scored: list[object] = []

    def start_episode(self, example, prompt_text):
        return [{"role": "user", "content": "go"}]

    def step_episode(self, example, messages, assistant_response):
        return _EnvironmentStepResult(done=True, messages=(), final_response_text=None)

    def score_episodes(self, example, episodes):
        for episode in episodes:
            self.scored.append(episode.response_text)
        return [_RewardResult(score=1.0, success=True) for _ in episodes]


class _SteppingMultiTurnEnv(_EnvironmentMultiTurn):
    """Multi-turn env that records exactly what step_episode was handed, and keeps going."""

    def __init__(self):
        self.stepped: list[tuple[str, str]] = []
        self.scored: list[object] = []

    def start_episode(self, example, prompt_text):
        return [{"role": "user", "content": "go"}]

    def step_episode(self, example, messages, assistant_response):
        # an env that parses the model's action records BOTH what it was handed and the transcript
        # message it is supposed to match, so a divergence between them is visible to the test.
        last = str(messages[-1].get("content", "")) if messages else ""
        self.stepped.append((assistant_response, last))
        return _EnvironmentStepResult(
            done=False, messages=({"role": "user", "content": "next"},), final_response_text=None
        )

    def score_episodes(self, example, episodes):
        for episode in episodes:
            self.scored.append(episode.response_text)
        return [_RewardResult(score=1.0, success=True) for _ in episodes]


_THINK_COMPLETION = "<think>let me work it out</think>the answer is 4"
# a turn that ran out of budget mid-reasoning: no </think>, and no <think> either because the chat
# template already opened the block in the generation prompt. tagless reasoning, not an answer.
_TRUNCATED_THINK_COMPLETION = "let me work it out, first I take 2 and then"


def test_multi_turn_scoring_strips_thinking_like_the_single_turn_path(monkeypatch):
    """MP-005: the multi-turn path used to hand score_episodes the RAW turn including <think>.

    The single-turn path grades ``graded_text`` (answer only), so an env whose scorer does
    ``response.strip() == expected`` scored correctly single-turn and silently mis-scored the same
    completion multi-turn. Both paths must see the same shape.
    """
    sdk_env = _ThinkingRecordingMultiTurnEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    env.thinking = True

    state = env.new_rollout_state({"id": "a", "input": "2+2?", "output": "4"})
    env.record_model_turn(state, _THINK_COMPLETION)
    env._score_episode({"id": "a", "input": "2+2?", "output": "4"}, state)

    assert sdk_env.scored == ["the answer is 4"], (
        f"multi-turn scorer saw raw thinking text: {sdk_env.scored!r}"
    )


def test_multi_turn_scored_text_keeps_the_reasoning_available(monkeypatch):
    """Stripping must not destroy the reasoning: thinking-aware scorers still opt into .thinking."""
    sdk_env = _ThinkingRecordingMultiTurnEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    env.thinking = True

    state = env.new_rollout_state({"id": "a", "input": "2+2?", "output": "4"})
    env.record_model_turn(state, _THINK_COMPLETION)
    env._score_episode({"id": "a", "input": "2+2?", "output": "4"}, state)

    scored = sdk_env.scored[0]
    assert scored.raw == _THINK_COMPLETION
    assert scored.thinking == "let me work it out"


def test_multi_turn_transcript_keeps_the_raw_turn(monkeypatch):
    """The message the next turn conditions on stays raw; only the SCORED text is answer-only."""
    sdk_env = _ThinkingRecordingMultiTurnEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    env.thinking = True

    state = env.new_rollout_state({"id": "a", "input": "2+2?", "output": "4"})
    msg = env.record_model_turn(state, _THINK_COMPLETION)
    assert msg["content"] == _THINK_COMPLETION
    assert state["messages"][-1]["content"] == _THINK_COMPLETION
    assert state["turns"][-1].content == _THINK_COMPLETION


def test_non_thinking_run_scores_the_completion_unchanged(monkeypatch):
    """With thinking off, nothing is stripped — a literal <think> is just text the model wrote."""
    sdk_env = _ThinkingRecordingMultiTurnEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    assert env.thinking is False  # default: a CLI-side load grades raw text

    state = env.new_rollout_state({"id": "a", "input": "2+2?", "output": "4"})
    env.record_model_turn(state, _THINK_COMPLETION)
    env._score_episode({"id": "a", "input": "2+2?", "output": "4"}, state)

    assert sdk_env.scored == [_THINK_COMPLETION]


def test_worker_marks_the_env_thinking_from_the_job_spec(monkeypatch):
    """The worker is what knows whether the run samples <think>; it must tell the env."""
    from flash.core.spec import JobSpec

    class _Env:
        thinking = False

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "thinking": True,
            "environment": {"id": "org/env"},
        }
    )
    monkeypatch.setattr(worker_state, "JOB_SPEC", spec)
    monkeypatch.setattr(
        worker_state, "load_staged_freesolo_environment", lambda *a, **k: (_Env(), None)
    )

    assert worker_state._load_active_env().thinking is True


def _thinking_env(monkeypatch, sdk_env, *, prompt_opens_thinking: bool):
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    env.thinking = True
    env.prompt_opens_thinking = prompt_opens_thinking
    return env


def test_opd_prepared_thinking_completion_steps_raw_and_grades_answer_only(monkeypatch):
    from flash.engine.worker.train.opd.orchestration import prompt_preparation
    from flash.engine.worker.train.opd.orchestration.state import _OpdRequest

    class _Tokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            enable_thinking,
            preserve_thinking,
        ):
            assert messages == [{"role": "user", "content": "go"}]
            assert add_generation_prompt is True
            assert enable_thinking is True
            assert preserve_thinking is False
            if tokenize:
                return [10, 11]
            return "<|im_start|>assistant\n<think>\n"

    class _TeacherClient:
        def __init__(self, *_args):
            pass

    sdk_env = _SteppingMultiTurnEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)
    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    tokenizer = _Tokenizer()
    monkeypatch.setattr(prompt_preparation._worker_state, "THINKING", True)
    monkeypatch.setattr(
        prompt_preparation._worker_hf,
        "load_tokenizer",
        lambda model_id, revision: tokenizer,
    )
    monkeypatch.setattr(
        prompt_preparation,
        "_thinking_prefill_text",
        lambda _tokenizer: "<think>\n",
    )
    monkeypatch.setattr(
        prompt_preparation._backend,
        "clamp_engine_len",
        lambda requested, _limit: requested,
    )
    monkeypatch.setattr(
        prompt_preparation._backend,
        "model_max_position_embeddings",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        prompt_preparation,
        "validate_glue_template",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        prompt_preparation,
        "liveness_heartbeat",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    import flash.engine.worker.teacher.client as teacher_client

    monkeypatch.setattr(teacher_client, "TeacherClient", _TeacherClient)
    request = _OpdRequest(
        spec=None,
        env=env,
        multi_turn=True,
        max_turns=2,
        knobs=types.SimpleNamespace(
            teacher_model="teacher",
            max_length=128,
            max_completion=8,
        ),
        model_id="model",
        model_revision="revision",
    )
    prompt_preparation.prepare_prompts(
        request,
        [
            (
                {"id": "a", "input": "2+2?", "output": "4"},
                [{"role": "user", "content": "go"}],
            )
        ],
        False,
        "capability",
        "https://control.invalid",
    )

    example = {"id": "a", "input": "2+2?", "output": "4"}
    state = env.new_rollout_state(example)
    env.record_model_turn(state, _THINK_COMPLETION)
    env.env_reply(state["messages"], state)
    env._score_episode(example, state)

    assert env.thinking is True
    assert env.prompt_opens_thinking is True
    handed, last_message = sdk_env.stepped[0]
    assert handed == _THINK_COMPLETION
    assert last_message == _THINK_COMPLETION
    scored = sdk_env.scored[0]
    assert str(scored) == "the answer is 4"
    assert scored.raw == _THINK_COMPLETION
    assert scored.thinking == "let me work it out"


def test_multi_turn_grading_honors_a_prompt_opened_think_block(monkeypatch):
    """A turn truncated mid-reasoning must grade empty here too, exactly as it does single-turn.

    When the chat template pre-opens <think> in the generation prompt (Qwen with enable_thinking),
    a completion cut off before </think> is tagless REASONING. strip_think without the flag sees no
    tags at all and returns the whole ramble as the answer, so the rollout can be rewarded for
    unfinished thinking that the single-turn path correctly scores 0.
    """
    sdk_env = _ThinkingRecordingMultiTurnEnv()
    env = _thinking_env(monkeypatch, sdk_env, prompt_opens_thinking=True)

    example = {"id": "a", "input": "2+2?", "output": "4"}
    state = env.new_rollout_state(example)
    env.record_model_turn(state, _TRUNCATED_THINK_COMPLETION)
    env._score_episode(example, state)

    assert sdk_env.scored == [""], (
        f"unterminated reasoning was graded as the answer: {sdk_env.scored!r}"
    )
    # the reasoning is still reachable -- stripping decides the GRADE, it does not destroy the text.
    assert sdk_env.scored[0].thinking == _TRUNCATED_THINK_COMPLETION
    assert sdk_env.scored[0].raw == _TRUNCATED_THINK_COMPLETION


def test_multi_turn_grading_matches_the_single_turn_path_exactly(monkeypatch):
    """Same completion, same flag, same parsers -- the two modes must not diverge.

    This is the whole point of the shared flash.content.thinking parsers: pin multi-turn grading against
    the single-turn helper rather than against a hand-written expectation, so a future change to
    one path that doesn't reach the other fails here.
    """
    from flash.content.thinking import strip_think

    for opened in (True, False):
        for completion in (_THINK_COMPLETION, _TRUNCATED_THINK_COMPLETION):
            sdk_env = _ThinkingRecordingMultiTurnEnv()
            env = _thinking_env(monkeypatch, sdk_env, prompt_opens_thinking=opened)

            example = {"id": "a", "input": "2+2?", "output": "4"}
            state = env.new_rollout_state(example)
            env.record_model_turn(state, completion)
            env._score_episode(example, state)

            expected = strip_think(completion, prompt_opened_thinking=opened)
            assert sdk_env.scored == [expected], (
                f"multi-turn diverged from single-turn for opened={opened} on {completion!r}: "
                f"{sdk_env.scored!r} != {[expected]!r}"
            )


def test_a_tagless_answer_is_not_stripped_when_the_template_ignores_thinking(monkeypatch):
    """The flag is derived, not assumed: with it False a tagless answer grades as the answer.

    A template that ignores enable_thinking never opens a block, so its plain answers must not be
    read as unterminated reasoning and zeroed. This is why the worker derives the flag from a real
    rendered prompt instead of from the thinking flag.
    """
    sdk_env = _ThinkingRecordingMultiTurnEnv()
    env = _thinking_env(monkeypatch, sdk_env, prompt_opens_thinking=False)

    example = {"id": "a", "input": "2+2?", "output": "4"}
    state = env.new_rollout_state(example)
    env.record_model_turn(state, _TRUNCATED_THINK_COMPLETION)
    env._score_episode(example, state)

    assert sdk_env.scored == [_TRUNCATED_THINK_COMPLETION]


def test_the_env_defaults_to_no_prompt_opened_thinking(monkeypatch):
    """Nothing may assume the opener: an env nobody configured must grade text as written."""
    sdk_env = _ThinkingRecordingMultiTurnEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    assert env.prompt_opens_thinking is False


def test_step_episode_receives_the_raw_turn_not_the_scored_one(monkeypatch):
    """step_episode drives the episode; it must see what the model actually emitted.

    An env that parses the model's action commonly requires assistant_response to equal
    messages[-1]["content"] -- and that message is the RAW turn. Handing it the answer-only text
    steps the env on something the model never emitted, so it can advance or terminate wrongly.
    """
    sdk_env = _SteppingMultiTurnEnv()
    env = _thinking_env(monkeypatch, sdk_env, prompt_opens_thinking=False)

    example = {"id": "a", "input": "2+2?", "output": "4"}
    state = env.new_rollout_state(example)
    env.record_model_turn(state, _THINK_COMPLETION)
    env.env_reply(state["messages"], state)

    handed, last_message = sdk_env.stepped[0]
    assert handed == _THINK_COMPLETION, f"step_episode was handed the stripped text: {handed!r}"
    assert handed == last_message, (
        f"assistant_response disagrees with messages[-1]: {handed!r} != {last_message!r}"
    )


class _BlockReplyMultiTurnEnv(_EnvironmentMultiTurn):
    """multi-turn env whose terminal reply contains raw text and image blocks."""

    def __init__(self):
        self.reply_content = [
            {"type": "text", "text": "first "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "second"},
        ]
        self.scored: list[object] = []

    def start_episode(self, example, prompt_text):
        return [{"role": "user", "content": "go"}]

    def step_episode(self, example, messages, assistant_response):
        return _EnvironmentStepResult(
            done=True,
            messages=({"role": "user", "content": self.reply_content},),
            final_response_text=None,
        )

    def score_episodes(self, example, episodes):
        self.scored.extend(episodes)
        return [_RewardResult(score=1.0, success=True) for _ in episodes]


def test_terminal_block_reply_reaches_score_episode_without_flattening(monkeypatch):
    sdk_env = _BlockReplyMultiTurnEnv()
    env = _thinking_env(monkeypatch, sdk_env, prompt_opens_thinking=False)
    expected = deepcopy(sdk_env.reply_content)

    example = {"id": "a", "input": "2+2?", "output": "4"}
    state = env.new_rollout_state(example)
    env.record_model_turn(state, "a turn")
    env.env_reply(state["messages"], state)
    sdk_env.reply_content[0]["text"] = "mutated by environment"
    sdk_env.reply_content.append({"type": "text", "text": "late mutation"})
    env._score_episode(example, state)

    episode = sdk_env.scored[0]
    assert episode.messages[-1]["content"] == expected
    assert episode.turns[-1].content == expected
    episode.messages[-1]["content"][0]["text"] = "mutated in messages"
    assert episode.turns[-1].content == expected


def test_stepping_the_env_leaves_the_scored_text_stripped(monkeypatch):
    """Fixing step_episode must not un-fix grading: the scorer still sees answer-only text."""
    sdk_env = _SteppingMultiTurnEnv()
    env = _thinking_env(monkeypatch, sdk_env, prompt_opens_thinking=False)

    example = {"id": "a", "input": "2+2?", "output": "4"}
    state = env.new_rollout_state(example)
    env.record_model_turn(state, _THINK_COMPLETION)
    env.env_reply(state["messages"], state)
    env._score_episode(example, state)

    assert sdk_env.scored == ["the answer is 4"]


def test_a_final_response_override_reaches_both_views(monkeypatch):
    """When the env supplies the episode's answer it is already final: graded AND stepped as-is."""

    class _OverridingEnv(_SteppingMultiTurnEnv):
        def step_episode(self, example, messages, assistant_response):
            last = str(messages[-1].get("content", "")) if messages else ""
            self.stepped.append((assistant_response, last))
            return _EnvironmentStepResult(
                done=False,
                messages=({"role": "user", "content": "next"},),
                final_response_text="env says 7",
            )

    sdk_env = _OverridingEnv()
    env = _thinking_env(monkeypatch, sdk_env, prompt_opens_thinking=False)

    example = {"id": "a", "input": "2+2?", "output": "4"}
    state = env.new_rollout_state(example)
    env.record_model_turn(state, _THINK_COMPLETION)
    env.env_reply(state["messages"], state)

    # the override is the answer, so it is not stripped -- and the STEPPING view tracks it, or a
    # later step_episode would be handed a turn the env already superseded.
    assert state["response_text"] == "env says 7"
    assert state["raw_response_text"] == "env says 7"

    # equality alone would pass on a bare str, and a bare str is the defect: a thinking-aware
    # score_episode reading .thinking would raise on exactly the episodes an env terminates by
    # overriding. the model's reasoning from this turn survives the override that replaced its
    # answer, since the env supplied the answer but did not produce the reasoning.
    assert state["response_text"].thinking == "let me work it out"
    assert env._episode_from_state(state).response_text.thinking == "let me work it out"

    # .raw is the model's emission, NOT the override -- see the dedicated test below. asserting it
    # here as "env says 7" is what made the scorer-facing raw view track the stepping view.
    assert state["response_text"].raw == _THINK_COMPLETION


def test_a_final_response_override_keeps_the_models_raw_output_for_scorers(monkeypatch):
    """`.raw` is the model's original output, so an env override must not overwrite it.

    The documented contract (flash/cli/scaffold/__init__.py) defines `.raw` as the original raw model
    output, and it is the one view a scorer cannot reconstruct once it is gone: `.completion` and
    `str(response_text)` are both the override already. Assigning the override to `.raw` also left
    the object incoherent -- an env-authored `.raw` paired with the model's own `.thinking`, so
    `.raw` did not contain the reasoning `.thinking` was extracted from.

    Distinct from the stepping view: `state["raw_response_text"]` SHOULD carry the override, so a
    later turn steps the env on the answer it substituted. The two are asserted together here
    because keeping them apart is the whole point.
    """

    class _OverridingEnv(_SteppingMultiTurnEnv):
        def step_episode(self, example, messages, assistant_response):
            last = str(messages[-1].get("content", "")) if messages else ""
            self.stepped.append((assistant_response, last))
            return _EnvironmentStepResult(
                done=True,
                messages=(),
                final_response_text="env says 7",
            )

    sdk_env = _OverridingEnv()
    env = _thinking_env(monkeypatch, sdk_env, prompt_opens_thinking=False)

    example = {"id": "a", "input": "2+2?", "output": "4"}
    state = env.new_rollout_state(example)
    env.record_model_turn(state, _THINK_COMPLETION)
    env.env_reply(state["messages"], state)

    scored = env._episode_from_state(state).response_text
    # what the scorer grades is the override...
    assert str(scored) == "env says 7"
    assert scored.completion == "env says 7"
    # ...but a scorer that explicitly inspects what the model emitted still gets it, reasoning and
    # all -- including the reasoning `.thinking` was taken from.
    assert scored.raw == _THINK_COMPLETION
    assert scored.thinking == "let me work it out"
    assert scored.thinking in scored.raw
    # the stepping view is the override, deliberately: it is what a later step_episode receives.
    assert state["raw_response_text"] == "env says 7"


def test_rl_hands_the_derived_opener_flag_to_the_env():
    """The env cannot derive the flag (no tokenizer, no rendered prompt); the rl path must pass it.

    the rl path needs a GPU, so pin the wiring at the source level: the flag the single-turn path
    derives is the same one the multi-turn env is given. _resolve_grpo_inputs is where the rendered
    prompt exists, so it is where the flag is derived and handed to the env.
    """
    import inspect

    from flash.engine.worker.train.rl.launch.inputs import _resolve_grpo_inputs

    src = inspect.getsource(_resolve_grpo_inputs)
    assert 'hasattr(env, "prompt_opens_thinking")' in src
    assert "env.prompt_opens_thinking = prompt_opened_thinking" in src


class _SlowGroupedEnv(_FakeSingleTurnEnv):
    """Records how many score_responses calls are in flight at once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.calls = 0

    def score_responses(self, example, response_texts):
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.calls += 1
        try:
            time.sleep(0.05)
            return [
                _RewardResult(score=float(len(text)), success=True, metrics=())
                for text in response_texts
            ]
        finally:
            with self.lock:
                self.live -= 1


def _grouped_items():
    # four DISTINCT prompts, two rollouts each: four groups, interleaved so a correct scatter
    # cannot be produced by accident from group-ordered output.
    return [
        (
            {"id": f"ex-{prompt}", "input": f"q{prompt}", "output": "4"},
            {"response_text": "x" * (prompt * 10 + rollout + 1)},
        )
        for rollout in range(2)
        for prompt in range(4)
    ]


def test_reward_task_groups_are_scored_concurrently(monkeypatch):
    """Distinct prompts must overlap their scoring instead of waiting on each other.

    Each score_responses call already scores its own completions concurrently, but the adapter used
    to loop over task groups SERIALLY, so a step with N prompts paid N judge latencies end to end
    while the gpu sat idle. For a remote-judge env that is the dominant cost of the step. Asserted
    on observed OVERLAP (peak in-flight calls > 1), not on wall-clock, so it cannot pass on a loaded
    machine by luck.
    """
    sdk_env = _SlowGroupedEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    rewards = env.reward_many(_grouped_items())

    assert sdk_env.calls == 4, sdk_env.calls
    assert sdk_env.peak > 1, f"groups did not overlap (peak={sdk_env.peak})"
    # and the values still land in INPUT order, not group order: each score is the response length.
    assert rewards == [float(len(state["response_text"])) for _, state in _grouped_items()]


class _SlowGroupedMultiTurnEnv(_FakeMultiTurnEnv):
    """Multi-turn env that records concurrent score_episodes calls."""

    def __init__(self):
        self.lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.calls = 0

    def score_episodes(self, example, episodes):
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.calls += 1
        try:
            time.sleep(0.05)
            return [
                _RewardResult(score=0.5, success=True, metrics=(_RewardMetric("episode", 0.5),))
                for _episode in episodes
            ]
        finally:
            with self.lock:
                self.live -= 1


def _multiturn_grouped_items():
    return [
        (
            {"id": f"ex-{prompt}", "input": f"q{prompt}", "output": "4"},
            {"response_text": f"r{rollout}", "episode": {"turns": []}},
        )
        for rollout in range(2)
        for prompt in range(4)
    ]


def test_reward_group_concurrency_is_skipped_for_a_non_thread_safe_env(monkeypatch):
    """`reward_thread_safe = False` means the scorer must never be raced, batching win or not.

    Such an env keeps thread-bound state (sqlite handles, browser sessions). Driven through
    rollout_rewards_many DELIBERATELY: reward_many diverts a non-thread-safe env to a serial path
    before _grouped_results is ever reached, so a reward_many-based test passes with the guard
    deleted and proves nothing. rollout_rewards_many is the door that actually reaches the grouped
    scorer, so it is the one that can fail.
    """
    sdk_env = _SlowGroupedMultiTurnEnv()
    sdk_env.reward_thread_safe = False
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    rewards = env.rollout_rewards_many(_multiturn_grouped_items())

    assert sdk_env.peak == 1, f"a non-thread-safe scorer was raced (peak={sdk_env.peak})"
    assert [reward.episode for reward in rewards] == [0.5] * 8


def test_multi_turn_reward_groups_are_scored_concurrently(monkeypatch):
    """The same overlap win must reach the multi-turn path, which scores whole episodes.

    Pairs with the guard test above: together they show the concurrency is real on this path AND
    that reward_thread_safe is what turns it off, rather than the path being serial anyway.
    """
    sdk_env = _SlowGroupedMultiTurnEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    rewards = env.rollout_rewards_many(_multiturn_grouped_items())

    assert sdk_env.calls == 4, sdk_env.calls
    assert sdk_env.peak > 1, f"multi-turn groups did not overlap (peak={sdk_env.peak})"
    assert [reward.episode for reward in rewards] == [0.5] * 8


class _FailingGroupedEnv(_FakeSingleTurnEnv):
    """Raises on one designated group and counts how many groups actually executed.

    ``slow_head`` makes group ``q0`` an order of magnitude slower than the rest, which is what
    real scorers look like -- group cost tracks completion length and judge latency, so it is
    uneven. A pool that waits on results in INPUT order cannot report the failure until that
    leading group returns, by which time the whole batch has drained.

    ``fast`` removes the per-group sleep entirely, which is the OPPOSITE regime and defeats a
    different implementation: when every call returns immediately, workers drain an eagerly
    submitted queue before the consumer reads its first result, so there is nothing left for
    ``cancel_futures`` to drop. A cheap scorer (a regex or exact-match grader) is exactly this.

    ``slow_error`` makes the FAILING group slow while the rest stay fast, which is what a real
    failure looks like -- an HTTP timeout or a late 5xx takes longer than a success, not less. The
    doomed call sits in flight while successes are consumed around it, and a window that refills on
    each consumed success walks the whole batch before the raise is ever seen. An in-flight call
    that will fail is indistinguishable from a slow one that will succeed, so only refusing to run
    ahead of the oldest unconsumed item bounds this.
    """

    def __init__(
        self,
        fail_on: str,
        slow_head: bool = False,
        fast: bool = False,
        slow_error: bool = False,
    ):
        self.fail_on = fail_on
        self.slow_head = slow_head
        self.fast = fast
        self.slow_error = slow_error
        self.lock = threading.Lock()
        self.executed = 0

    def score_responses(self, example, response_texts):
        with self.lock:
            self.executed += 1
        # hold the worker long enough that the groups queued behind this one are still QUEUED, not
        # started, when the raise surfaces -- that is what cancel_futures can drop.
        head = self.slow_head and str(example.input) == "q0"
        doomed = str(example.input) == self.fail_on
        if head or (doomed and self.slow_error):
            time.sleep(0.5)
        elif not (self.fast or self.slow_error):
            time.sleep(0.05)
        if doomed:
            raise RuntimeError("scorer exploded")
        return [_RewardResult(score=1.0, success=True, metrics=()) for _text in response_texts]


@pytest.mark.parametrize(
    ("fail_on", "ceiling", "slow_head", "fast", "slow_error"),
    [
        ("q1", 16, False, False, False),
        ("q19", 32, False, False, False),
        ("q1", 16, True, False, False),
        ("q19", 32, True, False, False),
        ("q1", 16, False, True, False),
        ("q19", 32, False, True, False),
        ("q1", 16, False, False, True),
        ("q19", 32, False, False, True),
    ],
)
def test_a_failing_scorer_group_wastes_at_most_a_pool_width(
    monkeypatch, fail_on, ceiling, slow_head, fast, slow_error
):
    """A raise must not drag the whole batch through the scorer before it propagates.

    The serial loop stopped AT the failing group; the pool also runs whatever is already in flight,
    and rl_train.py's batch-level retry re-scores everything serially afterwards, so those calls are
    billed TWICE. Bounding that waste is what makes the concurrency acceptable.

    The uniform-cost arms pass under every implementation tried, so the three skewed arms are the
    ones with teeth, and each defeats a DIFFERENT implementation:

    ``slow_head`` breaks consuming results in input order (what `pool.map` yields) -- a slow leading
    group defers the raise until every other group has run: 40/40 under `map`, against 9 here.

    ``fast`` breaks submitting the whole batch up front and relying on `cancel_futures`. With no
    sleep at all the workers drain the entire queue before the consumer reads its first result, so
    the cancel finds nothing pending: measured 36/40, and 10000/10000 on a larger batch, against 6.

    ``slow_error`` breaks a window that refills whenever any success is consumed. The doomed call is
    still in flight, so each consumed success pulls in another item and the window walks the whole
    batch: measured 2000/2000, waste scaling with error-latency over success-latency rather than
    with the cap, against 9 once the window refuses to run ahead of the oldest unconsumed item.

    Asserted as a CEILING (one pool width past the failure point, doubled for scheduling slack), not
    an equality: the exact count depends on how many workers have picked up work when the raise
    lands.
    """
    sdk_env = _FailingGroupedEnv(fail_on, slow_head=slow_head, fast=fast, slow_error=slow_error)
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    items = [
        ({"id": f"ex-{prompt}", "input": f"q{prompt}", "output": "4"}, {"response_text": "x"})
        for prompt in range(40)
    ]

    with pytest.raises(RuntimeError, match="scorer exploded"):
        env.reward_many(items)

    assert sdk_env.executed < 40, "the whole batch ran: queued groups were not cancelled"
    assert sdk_env.executed <= ceiling, sdk_env.executed


def test_a_single_task_group_still_scores_inline(monkeypatch):
    """One group must not spawn a pool: a thread for a single call is pure overhead.

    This is the common single-prompt path, so it stays exactly as it was.
    """
    sdk_env = _SlowGroupedEnv()
    _install_fake_freesolo(monkeypatch, sdk_env=sdk_env)

    from flash.envs.loading.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(sdk_env, "owner/env", source=None, contract_text="")
    example = {"id": "ex-1", "input": "2+2?", "output": "4"}
    rewards = env.reward_many(
        [(example, {"response_text": "ab"}), (example, {"response_text": "c"})]
    )

    assert sdk_env.calls == 1
    assert sdk_env.peak == 1
    assert rewards == [2.0, 1.0]
