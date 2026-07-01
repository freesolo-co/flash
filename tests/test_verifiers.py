"""Tests for the Freesolo SDK environment adapter."""

from __future__ import annotations

import gzip
import io
import json
import os
import sys
import tarfile
import tracemalloc
import types
from dataclasses import dataclass, field
from typing import ClassVar

import pytest


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
    content: str


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
    dataset: ClassVar[list[dict]] = [
        {
            "id": "ex-1",
            "input": "2+2?",
            "output": "4",
            "metadata": {"split": "train"},
        }
    ]

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
    def start_episode(self, example, prompt_text):
        return [{"role": "user", "content": f"{prompt_text}:{example.input}"}]

    def step_episode(self, example, messages, assistant_response):
        return _EnvironmentStepResult(
            done=True,
            messages=({"role": "user", "content": f"observed {assistant_response}"},),
            final_response_text=f"final {assistant_response}",
            metadata={"input": example.input},
        )

    def score_episodes(self, example, episodes):
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


def test_single_turn_reward_many_batches_by_example_value_identical(monkeypatch):
    """Single-turn reward_many groups same-example rollouts into ONE score_responses() call
    (env-concurrent, the win for judge/network rewards) while staying byte-identical and in input
    order vs the per-item reward() reference. Without grouping, a GRPO step scored its whole
    completion batch with serial per-rollout reward() calls (one blocking judge round-trip each)."""
    _install_fake_freesolo(monkeypatch)

    from flash.envs.adapter import FreesoloEnvironment

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
    """Codex MtMlT: an env that opts out with reward_thread_safe = False must NOT have a group's
    completions batched into one env-concurrent score_responses call (a scorer with mutable/thread-
    bound state would be raced). reward_many must score each rollout with its OWN single-item call,
    byte-identical and in input order — the pre-batching serial behavior."""
    _install_fake_freesolo(monkeypatch)

    from flash.envs.adapter import FreesoloEnvironment

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


def test_single_turn_scoring_gets_completion_thinking_and_raw(monkeypatch):
    """Thinking-mode GRPO passes an answer-only string to existing scorers while exposing
    structured thinking/raw fields for scorers that need them."""
    _install_fake_freesolo(monkeypatch)

    from flash.envs.adapter import FreesoloEnvironment

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

    breakdown = env.scores_breakdown(" 5", {"input": "q", "output": "4"}, state)

    assert breakdown["legacy_answer_match"] == 0.0
    assert breakdown["thinking"] == 1.0
    assert breakdown["raw_reasoning"] == 1.0
    assert breakdown["answer"] == 1.0
    assert breakdown["total"] == 3.0


def test_freesolo_sft_completion_full_gold_trajectory(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(_FakeSingleTurnEnv(), "owner/env", source=None, contract_text="")
    # A record whose `output` is a chat-message list -> the full gold trajectory (multi-turn SFT).
    gold = [
        {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        {"role": "user", "content": "<tool_result>...</tool_result>"},
        {"role": "assistant", "content": "done"},
    ]
    assert env.sft_completion({"input": "x", "output": gold}) == gold  # len>1 -> multi-turn
    # A scalar `output` is single-turn SFT -> one assistant turn.
    assert env.sft_completion({"input": "x", "output": "4"}) == [
        {"role": "assistant", "content": "4"}
    ]
    assert env.sft_completion({"input": "x"}) == [{"role": "assistant", "content": ""}]


def test_freesolo_multiturn_respects_per_example_budget(monkeypatch):
    _install_fake_freesolo(monkeypatch, sdk_env=_BudgetMultiTurnEnv())

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _BudgetMultiTurnEnv(),
        "owner/env",
        source=[{"input": "go", "output": ""}],
        contract_text="",
    )
    # max_turns is now the dataset-wide max_episode_turns (15), not the old hardcoded 8 —
    # otherwise the rollout loop would truncate this 15-turn scenario at turn 8.
    assert env.max_turns == 15

    state = env.new_rollout_state({"input": "go", "output": ""})
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
    dataset = tmp_path / "freesolo" / "datasets" / "train.jsonl"
    dataset.parent.mkdir()
    dataset.write_text('{"id":"row-1","input":"2+2?","output":"4"}\n')

    from flash.envs.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file),
        dataset_path="datasets/train.jsonl",
        contract_text="be brief",
        difficulty="hard",
    )

    assert env.id == str(env_file)
    assert seen["reference"] == str(env_file)
    assert seen["kwargs"]["dataset_path"] == str(dataset)
    assert seen["kwargs"]["difficulty"] == "hard"

    train = env.dataset()
    assert train == [{"id": "row-1", "input": "2+2?", "output": "4"}]
    assert env.prompt_messages(train[0]) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "2+2?"},
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
    """[environment.params] split must pick datasets/<split>.jsonl for Flash's own dataset()
    (SFT targets AND GRPO problem selection), not silently train on the default train.jsonl."""
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {
            "datasets/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "datasets/oracle.jsonl": '{"id":"o","input":"2+2?","output":"4"}\n',
        },
    )

    from flash.envs.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), split="oracle", contract_text="c")
    assert env.dataset() == [{"id": "o", "input": "2+2?", "output": "4"}]


def test_freesolo_adapter_missing_split_file_refuses_silent_train_fallback(
    monkeypatch, tmp_path
):
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path, {"datasets/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n'}
    )

    from flash.envs.adapter import load_freesolo_environment

    with pytest.raises(ValueError, match="split='oracle'"):
        load_freesolo_environment(str(env_file), split="oracle", contract_text="c")


def test_freesolo_adapter_split_train_uses_default_dataset(monkeypatch, tmp_path):
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path, {"datasets/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n'}
    )

    from flash.envs.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), split="train", contract_text="c")
    assert env.dataset() == [{"id": "t", "input": "train?", "output": "no"}]


def test_freesolo_adapter_explicit_dataset_path_wins_over_split(monkeypatch, tmp_path):
    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {
            "datasets/train.jsonl": '{"id":"t","input":"train?","output":"no"}\n',
            "datasets/oracle.jsonl": '{"id":"o","input":"2+2?","output":"4"}\n',
        },
    )

    from flash.envs.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file),
        dataset_path="datasets/train.jsonl",
        split="oracle",
        contract_text="c",
    )
    assert env.dataset() == [{"id": "t", "input": "train?", "output": "no"}]


def test_freesolo_adapter_warns_on_dropped_record_keys(monkeypatch, tmp_path):
    """Canonicalization keeps only input/output/id/metadata; extra top-level keys used to be
    dropped silently — now a once-per-key warnings.warn names them (per environment instance,
    not via a mutable module global checked per record)."""
    import warnings

    _install_fake_freesolo(monkeypatch)
    env_file = _split_env(
        tmp_path,
        {
            "datasets/train.jsonl": (
                '{"id":"t","input":"x","output":"y","initial_state":[1,2],'
                '"metadata":{"kept":true}}\n'
            )
        },
    )

    from flash.envs.adapter import load_freesolo_environment

    env = load_freesolo_environment(str(env_file), contract_text="c")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = env.dataset()
        assert rows == [{"id": "t", "input": "x", "output": "y", "metadata": {"kept": True}}]
        texts = [str(w.message) for w in caught]
        assert any("initial_state" in t and "dropped" in t for t in texts)
        # Re-canonicalizing the same records must not warn again (once per key per instance).
        before = len(caught)
        env._canonical_record(rows[0] | {"initial_state": [1, 2]})
        assert len(caught) == before


def test_freesolo_adapter_prepends_missing_contract_system_prompt(monkeypatch):
    class NoSystemEnv(_EnvironmentSingleTurn):
        dataset: ClassVar[list[dict]] = [{"input": "2+2?", "output": "4"}]

        def start_episode(self, example, prompt_text):
            return [{"role": "user", "content": example.input}]

        def score_responses(self, example, response_texts):
            return [_RewardResult(score=0.0) for _ in response_texts]

    _install_fake_freesolo(monkeypatch, sdk_env=NoSystemEnv())

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        NoSystemEnv(),
        "owner/env",
        source=None,
        contract_text="follow the contract",
    )

    assert env.prompt_messages({"input": "2+2?", "output": "4"}) == [
        {"role": "system", "content": "follow the contract"},
        {"role": "user", "content": "2+2?"},
    ]


def test_freesolo_adapter_fills_blank_contract_system_prompt(monkeypatch):
    class BlankSystemEnv(_EnvironmentSingleTurn):
        dataset: ClassVar[list[dict]] = [{"input": "2+2?", "output": "4"}]

        def start_episode(self, example, prompt_text):
            return [
                {"role": "system", "content": "   "},
                {"role": "user", "content": example.input},
            ]

        def score_responses(self, example, response_texts):
            return [_RewardResult(score=0.0) for _ in response_texts]

    _install_fake_freesolo(monkeypatch, sdk_env=BlankSystemEnv())

    from flash.envs.adapter import FreesoloEnvironment

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
    assert env.prompt_messages({"input": "2+2?", "output": "4"}) == expected
    state = env.new_rollout_state({"input": "2+2?", "output": "4"})
    assert state["prompt"] == expected
    assert state["messages"] == expected
    assert state["messages"] is not state["prompt"]
    state["messages"][0]["content"] = "changed"
    assert state["prompt"][0]["content"] == "follow the contract"


def test_freesolo_adapter_uses_env_dataset_when_no_source(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _FakeSingleTurnEnv(),
        "github:owner/repo@main:env/environment.py",
        source=None,
        contract_text="",
    )
    assert env.dataset()[0]["output"] == "4"


def test_freesolo_adapter_exports_sdk_examples_as_input_output(monkeypatch):
    class SdkExampleEnv(_EnvironmentSingleTurn):
        dataset: ClassVar[list[_TaskExample]] = [
            _TaskExample(
                record={},
                input="2+2?",
                id="ex-1",
                output="4",
                metadata={"split": "train"},
            )
        ]

    _install_fake_freesolo(monkeypatch, sdk_env=SdkExampleEnv())

    from flash.envs.adapter import FreesoloEnvironment

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
            "metadata": {"split": "train"},
        }
    ]


def test_freesolo_adapter_does_not_accept_record_aliases(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.adapter import FreesoloEnvironment

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

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _FakeMultiTurnEnv(),
        "github:owner/repo@main:env/environment.py",
        source=[{"input": "browse", "output": "done"}],
        contract_text="contract",
    )
    state = env.new_rollout_state({"input": "browse", "output": "done"})
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
    assert env.reward("ignored", {"input": "browse", "output": "done"}, state) == 0.5
    assert env.grade("ignored", {"input": "browse", "output": "done"}, state) is True
    assert (
        env.reward_from_messages(
            [{"role": "assistant", "content": "final"}],
            {"input": "browse", "output": "done"},
            [{"role": "user", "content": "contract:browse"}],
        )
        == 0.5
    )


def test_github_environment_ref_parsing():
    from flash.envs.adapter import (
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
    assert is_managed_environment_slug("owner/env")
    assert is_freesolo_environment_id("owner/env")
    assert managed_slug_to_github_ref("owner/env") == (
        "github:freesolo-co/environment-hub@main:owner/env/environment.py"
    )
    assert not is_managed_environment_slug("owner/env/extra")
    assert not is_freesolo_environment_id("gsm8k")


def test_github_environment_resolves_by_commit_sha(tmp_path, monkeypatch):
    import flash.envs.loader as adapter

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
    import flash.envs.loader as adapter

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
    import flash.envs.loader as adapter

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
    import flash.envs.loader as adapter

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
    from flash.envs.loader import _safe_extract_archive

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
    monkeypatch.setattr("flash.envs.loader._MAX_ARCHIVE_MEMBERS", 0)
    with pytest.raises(RuntimeError, match="too many members"):
        _safe_extract_archive(make_members_tar([("a", b"")]), dest)

    dest = tmp_path / "extract_big"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.loader._MAX_ARCHIVE_MEMBERS", 5)
    monkeypatch.setattr("flash.envs.loader._MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(RuntimeError, match="too large"):
        _safe_extract_archive(
            make_members_tar([("repo-root/", None), ("repo-root/keep.txt", b"xx")]), dest
        )

    dest = tmp_path / "extract_pax"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.loader._MAX_ARCHIVE_BYTES", 100)
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
    from flash.envs.loader import _safe_extract_archive

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
    import flash.envs.registry as registry

    env_id = "github:owner/repo@main:env/environment.py"
    assert registry.worker_pip_for_env(env_id) == ["freesolo>=0.2.52"]
