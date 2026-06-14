"""Built-in GSM8K verifiable-math environment (the simplest worked example).

Task wiring lives in three sibling modules so each concern is independently
readable and reusable:

  * ``data.py``    -> dataset loading, prompt formatting, and SFT target.
  * ``grading.py`` -> the dependency-free grader (answer extraction + numeric
                      equivalence) backing the GRPO reward.
  * ``env.py``     -> this file: the ``Environment`` that the registry path-loads
                      as the ``gsm8k`` built-in, delegating to the two above.

Both algorithms (SFT / GRPO) run against this one environment; see the
``gsm8k_*.toml`` configs in this folder and ``docs/algorithms.md``.
"""

from __future__ import annotations

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

    def __init__(self, **kwargs):
        super().__init__(id="gsm8k")
        self.dataset_name = kwargs.get("dataset", kwargs.get("dataset_name", self.dataset_name))
        self.dataset_config = kwargs.get("dataset_config", self.dataset_config)
        self.correct_reward = float(kwargs.get("correct_reward", self.correct_reward))
        self.format_reward = float(kwargs.get("format_reward", self.format_reward))

    # -- data -------------------------------------------------------------
    def dataset(self, split: str) -> list[dict]:
        return d.load_gsm8k("train", self.dataset_name, self.dataset_config)

    # -- task interface ---------------------------------------------------
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
