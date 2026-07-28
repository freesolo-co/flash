"""Torch-dependent coverage for multi-turn GRPO per-turn credit assignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flash.engine.worker.grpo_perturn_trainer import (
    GRPOPerTurnTrainer,
    build_per_turn_advantages,
)
from flash.spec import DEFAULT_CREDIT_ASSIGNMENT, PER_TURN_CREDIT_ASSIGNMENT
from tests.test_multiturn_per_turn_reward import (
    _TOKEN_IDS,
    SyntheticPerTurnEnv,
    _rollout,
)


def _tiny_gpt2_and_tokenizer():
    """a tiny word-level gpt2 + tokenizer for driving a real trl train() step on cpu."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            vocab={"<pad>": 0, "<eos>": 1, "<unk>": 2, **_TOKEN_IDS}, unk_token="<unk>"
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
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
    return model, tokenizer


@pytest.mark.parametrize(
    ("credit_assignment", "is_multi_turn", "use_rollout_func", "expects_per_turn_trainer"),
    [
        (PER_TURN_CREDIT_ASSIGNMENT, True, True, True),
        (PER_TURN_CREDIT_ASSIGNMENT, False, False, False),
        (DEFAULT_CREDIT_ASSIGNMENT, False, False, False),
        (DEFAULT_CREDIT_ASSIGNMENT, True, False, False),
        (DEFAULT_CREDIT_ASSIGNMENT, True, True, False),
    ],
)
def test_credit_assignment_selects_trainer_class(
    credit_assignment, is_multi_turn, use_rollout_func, expects_per_turn_trainer
):
    from trl import GRPOTrainer

    from flash.engine.worker.rl import select_grpo_trainer

    expected = GRPOPerTurnTrainer if expects_per_turn_trainer else GRPOTrainer

    assert (
        select_grpo_trainer(
            GRPOTrainer,
            credit_assignment=credit_assignment,
            is_multi_turn=is_multi_turn,
            use_rollout_func=use_rollout_func,
        )
        is expected
    )


def test_per_turn_credit_rejects_tool_calling_multi_turn_environment():
    from trl import GRPOTrainer

    from flash.engine.worker.rl import select_grpo_trainer

    with pytest.raises(
        RuntimeError,
        match=(
            "credit_assignment='per_turn' is not supported for tool-calling multi-turn "
            "environments; use 'per_episode'"
        ),
    ):
        select_grpo_trainer(
            GRPOTrainer,
            credit_assignment=PER_TURN_CREDIT_ASSIGNMENT,
            is_multi_turn=True,
            use_rollout_func=False,
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
    from trl import GRPOConfig

    model, tokenizer = _tiny_gpt2_and_tokenizer()
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


def test_per_turn_extracts_signal_that_per_episode_discards(tmp_path):
    """decisive end-to-end proof through the REAL trainer: two rollouts with EQUAL episode reward but
    OPPOSITE per-turn rewards. per-turn assigns exact turn-differentiated [B,T] advantages (each turn
    = reward - group-mean over that turn index), while stock per-episode gives a uniform ZERO advantage
    because both episodes are equal -- so per-turn extracts turn-level signal that per-episode discards.
    """
    datasets = pytest.importorskip("datasets")
    from trl import GRPOConfig, GRPOTrainer

    # completion tokens [miss, next(glue), hit]: turn0 span [0,1), glue [1,2), turn1 span [2,3)
    completion = [_TOKEN_IDS["miss"], _TOKEN_IDS["next"], _TOKEN_IDS["hit"]]
    rollouts = [
        {
            "prompt_ids": [_TOKEN_IDS["prompt"]],
            "completion_ids": completion,
            "logprobs": [-0.1, -0.1, -0.1],
            "env_mask": [1, 0, 1],
            "reward": 1.0,
            "turn_spans": [(0, 1), (2, 3)],
            "turn_rewards": [1.0, 0.0],
        },
        {
            "prompt_ids": [_TOKEN_IDS["prompt"]],
            "completion_ids": completion,
            "logprobs": [-0.1, -0.1, -0.1],
            "env_mask": [1, 0, 1],
            "reward": 1.0,
            "turn_spans": [(0, 1), (2, 3)],
            "turn_rewards": [0.0, 1.0],
        },
    ]
    fields = (
        "prompt_ids",
        "completion_ids",
        "logprobs",
        "env_mask",
        "reward",
        "turn_spans",
        "turn_rewards",
    )

    def _captured_advantages(trainer_cls, emitted_fields):
        captured = {}

        class _Recording(trainer_cls):
            def _generate_and_score_completions(self, inputs):
                output = super()._generate_and_score_completions(inputs)
                captured["advantages"] = output["advantages"].detach().float().cpu().clone()
                return output

        model, tokenizer = _tiny_gpt2_and_tokenizer()
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
            scale_rewards="none",
            loss_type="dr_grpo",
        )
        trainer = _Recording(
            model=model,
            args=config,
            train_dataset=datasets.Dataset.from_list([{"prompt": "prompt"}]),
            reward_funcs=lambda completions, **kwargs: kwargs["reward"],
            processing_class=tokenizer,
            rollout_func=lambda prompts, trainer: {
                k: [r[k] for r in rollouts] for k in emitted_fields
            },
        )
        trainer.train()
        return captured["advantages"]

    per_turn = _captured_advantages(GRPOPerTurnTrainer, fields)[:, :3]
    # stock GRPOTrainer (per-episode): emit only the base fields, no per-turn metadata
    per_episode = _captured_advantages(GRPOTrainer, fields[:5])
    if per_episode.dim() == 1:
        per_episode = per_episode.unsqueeze(1)

    # turn0 rewards [1,0] mean 0.5 -> +0.5/-0.5 ; turn1 rewards [0,1] mean 0.5 -> -0.5/+0.5
    torch.testing.assert_close(
        per_turn, torch.tensor([[0.5, 0.0, -0.5], [-0.5, 0.0, 0.5]]), atol=1e-4, rtol=0
    )
    # within a rollout, the two turns get different advantages (turn-differentiated credit)
    assert per_turn[0, 0] != per_turn[0, 2]
    # equal episodes -> per-episode advantage is uniformly zero: no learning signal
    assert float(per_episode.abs().max()) < 1e-4
    # the +/-0.5 per-turn magnitude is exactly the turn-level signal per-episode discards
    assert float(per_turn.abs().max()) > 0.4


# --------------------------------------------------------------------------- AS-028


@pytest.fixture
def reset_fallback_warning(monkeypatch):
    """The warn-once latch is module state; isolate it so tests do not mask each other."""
    monkeypatch.setattr(
        "flash.engine.worker.grpo_perturn_trainer._WARNED_EPISODE_FALLBACK",
        False,
        raising=False,
    )


def test_missing_per_turn_rewards_warns_instead_of_silently_using_episode_credit(
    reset_fallback_warning, capsys
):
    """per_turn is accepted and echoed in status; falling back to episode credit must not be silent."""
    spans = [[(0, 1)], [(0, 1)]]
    scalar = torch.tensor([1.0, -1.0], dtype=torch.float64)

    build_per_turn_advantages(
        spans,
        [None, None],  # environment emitted no per_turn_rewards
        num_generations=2,
        completion_len=2,
        episode_advantages=scalar,
    )

    out = capsys.readouterr().out
    assert "per_turn" in out
    assert "episode credit" in out
    assert "per_turn_rewards" in out


def test_episode_fallback_warning_is_emitted_once_per_run(reset_fallback_warning, capsys):
    # the condition is a property of the environment, so a per-group warning would flood the log.
    spans = [[(0, 1)], [(0, 1)]]
    scalar = torch.tensor([1.0, -1.0], dtype=torch.float64)
    for _ in range(3):
        build_per_turn_advantages(
            spans,
            [None, None],
            num_generations=2,
            completion_len=2,
            episode_advantages=scalar,
        )
    assert capsys.readouterr().out.count("[grpo][warn]") == 1


def test_complete_per_turn_rewards_do_not_warn(reset_fallback_warning, capsys):
    spans = [[(0, 1)], [(0, 1)]]
    scalar = torch.tensor([1.0, -1.0], dtype=torch.float64)
    build_per_turn_advantages(
        spans,
        [[1.0], [3.0]],
        num_generations=2,
        completion_len=2,
        episode_advantages=scalar,
    )
    assert "[grpo][warn]" not in capsys.readouterr().out
