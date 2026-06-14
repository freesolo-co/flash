"""Built-in competition-math environment ("real, extended" verifiable RL task).

This is the harder sibling of the GSM8K environment — a non-toy competition-math
post-train task:

  * train split  -> ``agentica-org/DeepScaleR-Preview-Dataset`` (~40k AIME/AMC/
    Omni-MATH problems with LaTeX answers) by default

The dataset lives on the Hugging Face Hub and is also mirrored on the Prime
Intellect Environments Hub, so the *same* task can interoperate with prime-rl.

Scoring uses this example's dependency-free math grader (``grading.py``):
``\\boxed{...}`` extraction + LaTeX/numeric equivalence, backing both the SFT target
check and the GRPO reward.

Dataset loading is lazy (``datasets`` is only imported inside ``dataset()``), and
the ``train_path`` JSONL override lets the environment run fully offline (used by
the unit tests and by air-gapped smoke runs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from autoslm.envs.base import BaseEnvironment

from . import grading as mg

SYSTEM_PROMPT = (
    "You are an expert competition mathematician. Solve the problem step by "
    "step. Put ONLY the final answer inside \\boxed{} on the last line, e.g. "
    "\\boxed{42} or \\boxed{\\frac{2}{3}}."
)


@dataclass
class MathEnvironment(BaseEnvironment):
    train_dataset: str = "agentica-org/DeepScaleR-Preview-Dataset"
    train_config: str | None = None
    train_configs: list | None = None
    train_split: str = "train"
    question_key: str = "problem"
    answer_key: str = "answer"
    solution_key: str = "solution"
    correct_reward: float = 1.0
    format_reward: float = 0.1
    train_path: str | None = None

    def __init__(self, **kwargs):
        super().__init__(id="math")
        for field in (
            "train_dataset",
            "train_config",
            "train_configs",
            "train_split",
            "question_key",
            "answer_key",
            "solution_key",
            "train_path",
        ):
            if field in kwargs and kwargs[field] is not None:
                setattr(self, field, kwargs[field])
        self.correct_reward = float(kwargs.get("correct_reward", 1.0))
        self.format_reward = float(kwargs.get("format_reward", 0.1))

    # -- data -------------------------------------------------------------
    def _normalize_rows(self, rows) -> list[dict]:
        out = []
        for ex in rows:
            question = ex.get(self.question_key) or ex.get("question") or ex.get("prompt")
            solution = ex.get(self.solution_key) or ""
            answer = ex.get(self.answer_key)
            # MATH-style sets ship only a worked solution; derive gold from its \boxed{}.
            if answer is None or str(answer).strip() == "":
                answer = mg.extract_boxed(str(solution))
            if question is None or answer is None or str(answer).strip() == "":
                continue
            out.append(
                {
                    "question": question,
                    "gold": str(answer),
                    "solution": solution,
                }
            )
        return out

    def _load_jsonl(self, path: str) -> list[dict]:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _load_hf(self, name: str, config: str | None, split: str) -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
        return [dict(ex) for ex in ds]

    def dataset(self, split: str) -> list[dict]:
        if self.train_path:
            return self._normalize_rows(self._load_jsonl(self.train_path))
        if self.train_configs:
            raw = []
            for cfg in self.train_configs:
                raw.extend(self._load_hf(self.train_dataset, cfg, self.train_split))
            return self._normalize_rows(raw)
        return self._normalize_rows(
            self._load_hf(self.train_dataset, self.train_config, self.train_split)
        )

    # -- task interface ---------------------------------------------------
    def prompt_messages(self, example: dict) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(example["question"]).strip()},
        ]

    def sft_target(self, example: dict) -> str:
        gold = str(example["gold"]).strip()
        solution = str(example.get("solution") or "").strip()
        if solution and r"\boxed" in solution:
            return solution
        if solution:
            return f"{solution}\nThe final answer is \\boxed{{{gold}}}."
        return f"The final answer is \\boxed{{{gold}}}."

    def reward(self, completion: str, example: dict) -> float:
        return mg.reward(completion, str(example["gold"]), self.correct_reward, self.format_reward)

    def grade(self, completion: str, example: dict) -> bool:
        return mg.grade(completion, str(example["gold"]))

    def extract_pred(self, completion: str) -> str | None:
        return mg.extract_answer(completion)


def load_environment(**kwargs) -> MathEnvironment:
    return MathEnvironment(**kwargs)
