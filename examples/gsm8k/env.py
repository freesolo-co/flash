"""Built-in GSM8K verifiable math environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from autoslm.envs.base import BaseEnvironment

from . import data as d
from . import grading as g


@dataclass
class GSM8KEnvironment(BaseEnvironment):
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    correct_reward: float = 1.0
    format_reward: float = 0.1
    eval_examples: int = 300
    eval_seed: int = 12345

    def __init__(self, **kwargs):
        super().__init__(id="gsm8k")
        self.dataset_name = kwargs.get("dataset", kwargs.get("dataset_name", self.dataset_name))
        self.dataset_config = kwargs.get("dataset_config", self.dataset_config)
        self.correct_reward = float(kwargs.get("correct_reward", self.correct_reward))
        self.format_reward = float(kwargs.get("format_reward", self.format_reward))
        # Default to the run's [train] eval_examples (the worker exports it as EVAL_NUM)
        # so a config setting only train.eval_examples builds the seeded N-row subset
        # directly, instead of slicing the worker's [:N] off a fixed 300-row sample
        # (values > 300 were silently capped; smaller values used the wrong rows). An
        # explicit environment.params.eval_examples still wins.
        default_eval = int(os.environ.get("EVAL_NUM") or self.eval_examples)
        self.eval_examples = int(kwargs.get("eval_examples", default_eval))
        self.eval_seed = int(kwargs.get("eval_seed", self.eval_seed))

    def dataset(self, split: str) -> list[dict]:
        if split in {"eval", "validation", "test"}:
            return d.fixed_eval_subset(
                self.eval_examples,
                self.eval_seed,
                self.dataset_name,
                self.dataset_config,
                "test",
            )
        return d.load_gsm8k("train", self.dataset_name, self.dataset_config)

    def prompt_messages(self, example: dict) -> list[dict]:
        return d.build_prompt_messages(example["question"])

    def sft_target(self, example: dict) -> str:
        return d.build_target_text(example["solution"])

    def reward(self, completion: str, example: dict) -> float:
        return g.reward(completion, example["gold"], self.correct_reward, self.format_reward)

    def grade(self, completion: str, example: dict) -> bool:
        return g.is_correct(completion, example["gold"])

    def extract_pred(self, completion: str) -> str | None:
        return g.extract_pred_answer(completion)


def load_environment(**kwargs) -> GSM8KEnvironment:
    return GSM8KEnvironment(**kwargs)
