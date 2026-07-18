"""CPU coverage for multi-turn GRPO per-turn credit assignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flash.engine.multiturn_rollout import rollout_one
from flash.engine.worker.grpo_perturn_trainer import (
    GRPOPerTurnTrainer,
    build_per_turn_advantages,
)
from flash.envs.base import BaseEnvironment


class SyntheticPerTurnEnv(BaseEnvironment):
    """Two-turn environment with deterministic target-token rewards."""

    multi_turn = True

    def __init__(self):
        super().__init__(id="synthetic-per-turn")
        self.turn_reward_calls = 0

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

    def turn_rewards(self, example, state):
        self.turn_reward_calls += 1
        targets = state.get("targets") or example["targets"]
        return [
            1.0 if target in text else 0.0
            for text, target in zip(state.get("assistant_turns", []), targets, strict=True)
        ]


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


def test_build_per_turn_advantages_centers_by_turn_and_assigns_spans():
    spans = [
        [(0, 2), (4, 5)],
        [(0, 1)],
        [(0, 1), (3, 5)],
        [(0, 2), (4, 6)],
    ]
    rewards = [[1.0, 10.0], [5.0], [2.0, 7.0], [6.0, 11.0]]

    actual = build_per_turn_advantages(spans, rewards, num_generations=2, completion_len=8)

    expected = torch.tensor(
        [
            [-2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(actual, expected)
    assert actual[0, 0].item() == pytest.approx(1.0 - 3.0)


def test_build_per_turn_advantages_handles_variable_turns_and_mixed_fallback():
    spans = [
        [(0, 1), (3, 4)],
        [(0, 2)],
        [(0, 1), (2, 3)],
        [(0, 1), (3, 5)],
        [(0, 2)],
        [(0, 1), (2, 4)],
    ]
    rewards = [[1.0, 10.0], [3.0], [5.0, 14.0], None, [9.0], [5.0, 1.0]]
    scalar = torch.tensor([0.0, 0.0, 0.0, 2.5, -1.0, 1.0], dtype=torch.float64)

    actual = build_per_turn_advantages(
        spans,
        rewards,
        num_generations=3,
        completion_len=7,
        episode_advantages=scalar,
    )

    expected = torch.tensor(
        [
            [-2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [2.5, 2.5, 2.5, 2.5, 2.5, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == scalar.dtype


def test_rollout_records_turn_spans_and_calls_turn_rewards():
    env = SyntheticPerTurnEnv()

    result = _rollout(env, ["miss", "hit"])

    assert result["completion_ids"] == [4, 6, 5]
    assert result["env_mask"] == [1, 0, 1]
    assert result["turn_spans"] == [(0, 1), (2, 3)]
    assert result["turn_rewards"] == [0.0, 1.0]
    assert result["reward"] == 1.0
    assert env.turn_reward_calls == 1


def test_non_finite_turn_reward_degrades_group_to_scalar(capsys):
    class NonFiniteTurnEnv(SyntheticPerTurnEnv):
        def turn_rewards(self, example, state):
            _ = example, state
            return [float("nan"), 1.0]

    invalid = _rollout(NonFiniteTurnEnv(), ["miss", "hit"])
    assert invalid["turn_rewards"] is None
    assert capsys.readouterr().out.count("using episode reward") == 1

    spans = [invalid["turn_spans"], [(0, 1), (2, 3)]]
    scalar = torch.tensor([1.0, -1.0])
    actual = build_per_turn_advantages(
        spans,
        [invalid["turn_rewards"], [1.0, 0.0]],
        num_generations=2,
        completion_len=3,
        episode_advantages=scalar,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
    )
    assert bool(torch.isfinite(actual).all())


def test_non_finite_episode_reward_disables_per_turn_credit(capsys):
    class NonFiniteEpisodeEnv(SyntheticPerTurnEnv):
        def reward(self, completion, example, state=None):
            _ = completion, example, state
            return float("nan")

    env = NonFiniteEpisodeEnv()
    result = _rollout(env, ["miss", "hit"])

    assert result["turn_rewards"] is None
    assert env.turn_reward_calls == 0
    assert capsys.readouterr().out.count("using episode reward") == 1


def test_rollout_uses_one_full_result_scoring_pass():
    class FullResultEnv(SyntheticPerTurnEnv):
        def __init__(self):
            super().__init__()
            self.score_calls = 0

        def reward(self, completion, example, state=None):
            raise AssertionError("scalar reward must come from the full result scoring pass")

        def turn_rewards(self, example, state):
            raise AssertionError("per-turn rewards must come from the same full result")

        def reward_results_many(self, items):
            self.score_calls += 1
            assert len(items) == 1
            return [SimpleNamespace(score=1.0, metadata={"per_turn_rewards": [0.0, 1.0]})]

    env = FullResultEnv()
    result = _rollout(env, ["miss", "hit"])

    assert result["reward"] == 1.0
    assert result["turn_rewards"] == [0.0, 1.0]
    assert env.score_calls == 1


def test_base_environment_keeps_episode_only_fallback():
    env = BaseEnvironment(id="episode-only")
    assert env.turn_rewards({}, {}) is None


def test_freesolo_adapter_uses_complete_terminal_per_turn_metadata(monkeypatch):
    from freesolo.environments.types import RewardResult

    from flash.envs.adapter import FreesoloEnvironment

    env = object.__new__(FreesoloEnvironment)
    state = {
        "turns": ["first", "second"],
        "step_metadata": [{"per_turn_rewards": [0.25]}],
    }
    score_calls = []

    def score_episode(example, terminal_state):
        score_calls.append((example, terminal_state))
        return RewardResult(score=1.0, metadata={"per_turn_rewards": [0.25, 0.75]})

    monkeypatch.setattr(env, "_score_episode", score_episode)

    assert env.turn_rewards({"input": "prompt"}, state) == [0.25, 0.75]
    assert score_calls == [({"input": "prompt"}, state)]


def test_trainer_replaces_scalar_advantages_in_output_row_order(monkeypatch):
    from trl import GRPOTrainer

    inputs = [
        {"turn_spans": [(0, 1), (2, 3)], "turn_rewards": [0.0, 1.0]},
        {"turn_spans": [(0, 1), (2, 3)], "turn_rewards": [1.0, 0.0]},
    ]
    scalar = torch.tensor([-3.0, 3.0])

    def fake_super(self, rows):
        assert rows is inputs
        return {
            "completion_ids": torch.zeros((2, 4), dtype=torch.long),
            "advantages": scalar,
        }

    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", fake_super)
    trainer = object.__new__(GRPOPerTurnTrainer)
    trainer.accelerator = SimpleNamespace(num_processes=1)
    trainer.model = SimpleNamespace(training=True)
    trainer.num_generations = 2
    trainer.num_generations_eval = 2

    output = trainer._generate_and_score_completions(inputs)

    assert output["advantages"].shape == (2, 4)
    torch.testing.assert_close(
        output["advantages"],
        torch.tensor([[-0.5, 0.0, 0.5, 0.0], [0.5, 0.0, -0.5, 0.0]]),
    )


def test_trainer_rejects_distributed_per_turn_credit(monkeypatch):
    from trl import GRPOTrainer

    monkeypatch.setattr(
        GRPOTrainer,
        "_generate_and_score_completions",
        lambda self, inputs: {
            "completion_ids": torch.zeros((2, 1), dtype=torch.long),
            "advantages": torch.zeros(2),
        },
    )
    trainer = object.__new__(GRPOPerTurnTrainer)
    trainer.accelerator = SimpleNamespace(num_processes=2)
    trainer.model = SimpleNamespace(training=True)
    trainer.num_generations = 2
    trainer.num_generations_eval = 2
    inputs = [
        {"turn_spans": [(0, 1)], "turn_rewards": [0.0]},
        {"turn_spans": [(0, 1)], "turn_rewards": [1.0]},
    ]

    with pytest.raises(NotImplementedError, match="single-process"):
        trainer._generate_and_score_completions(inputs)


def test_cpu_trainer_step_uses_two_dimensional_advantages(tmp_path):
    datasets = pytest.importorskip("datasets")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from trl import GRPOConfig

    tokenizer_backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            vocab={"<pad>": 0, "<eos>": 1, "<unk>": 2, **_TOKEN_IDS},
            unk_token="<unk>",
        )
    )
    tokenizer_backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )
    observed_shapes = []

    class RecordingTrainer(GRPOPerTurnTrainer):
        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            observed_shapes.append(tuple(output["advantages"].shape))
            return output

    def rollout_func(prompts, trainer):
        _ = trainer
        rows = []
        choices = [("miss", "hit"), ("hit", "miss")]
        for index, prompt in enumerate(prompts):
            _ = prompt
            rows.append(_rollout(SyntheticPerTurnEnv(), choices[index % len(choices)]))
        return {
            key: [row[key] for row in rows]
            for key in (
                "prompt_ids",
                "completion_ids",
                "logprobs",
                "env_mask",
                "reward",
                "turn_spans",
                "turn_rewards",
            )
        }

    config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=8,
        max_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_vllm=False,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        scale_rewards="none",
        loss_type="dr_grpo",
    )
    trainer = RecordingTrainer(
        model=model,
        args=config,
        train_dataset=datasets.Dataset.from_list([{"prompt": "prompt"}]),
        reward_funcs=lambda completions, **kwargs: kwargs["reward"],
        processing_class=tokenizer,
        rollout_func=rollout_func,
    )

    trainer.train()

    assert observed_shapes
    assert all(len(shape) == 2 for shape in observed_shapes)
    assert observed_shapes[0][0] == 2
