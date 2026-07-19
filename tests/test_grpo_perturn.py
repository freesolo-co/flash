"""Torch-dependent coverage for multi-turn GRPO per-turn credit assignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flash.engine.worker.grpo_perturn_trainer import (
    GRPOPerTurnTrainer,
    build_per_turn_advantages,
)
from tests.test_multiturn_per_turn_reward import (
    _TOKEN_IDS,
    SyntheticPerTurnEnv,
    _rollout,
)


def test_build_per_turn_advantages_centers_by_turn_and_assigns_spans():
    spans = [
        [(0, 2), (4, 5)],
        [(0, 1)],
        [(0, 1), (3, 5)],
        [(0, 2), (4, 6)],
    ]
    rewards = [[1.0, 10.0], [5.0], [2.0, 7.0], [6.0, 11.0]]

    actual = build_per_turn_advantages(
        spans,
        rewards,
        num_generations=2,
        completion_len=8,
        episode_advantages=torch.zeros(4),
    )

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


def test_invalid_turn_rewards_degrade_group_to_scalar_advantages():
    spans = [[(0, 1), (2, 3)], [(0, 1), (2, 3)]]
    scalar = torch.tensor([1.0, -1.0])

    actual = build_per_turn_advantages(
        spans,
        [None, [1.0, 0.0]],
        num_generations=2,
        completion_len=3,
        episode_advantages=scalar,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
    )
    assert bool(torch.isfinite(actual).all())


def test_trainer_noops_when_all_turn_rewards_are_none(monkeypatch):
    from trl import GRPOTrainer

    inputs = [
        {"turn_spans": [(0, 1)], "turn_rewards": None},
        {"turn_spans": [(0, 1)], "turn_rewards": None},
    ]
    scalar = torch.tensor([-1.0, 1.0])
    stock_output = {
        "completion_ids": torch.zeros((2, 1), dtype=torch.long),
        "advantages": scalar,
    }
    monkeypatch.setattr(
        GRPOTrainer,
        "_generate_and_score_completions",
        lambda self, rows: stock_output,
    )
    trainer = object.__new__(GRPOPerTurnTrainer)

    output = trainer._generate_and_score_completions(inputs)

    assert output is stock_output
    assert output["advantages"] is scalar
    assert output["advantages"].shape == (2,)


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
