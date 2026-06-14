"""Built-in competition-math environment ("real, extended" verifiable RL task).

This is the harder sibling of the GSM8K environment — a non-toy competition-math
post-train task:

  * train split  -> ``agentica-org/DeepScaleR-Preview-Dataset`` (~40k AIME/AMC/
    Omni-MATH problems with LaTeX answers) by default
  * eval split   -> ``HuggingFaceH4/MATH-500`` (the standard 500-problem MATH eval)

Both datasets live on the Hugging Face Hub and are also mirrored on the Prime
Intellect Environments Hub, so the *same* task can interoperate with prime-rl.

Scoring uses this example's dependency-free math grader (``grading.py``):
``\\boxed{...}`` extraction + LaTeX/numeric equivalence. The same grader backs the
SFT target check, the RL reward, and the eval so every arm is scored identically.

Dataset loading is lazy (``datasets`` is only imported inside ``dataset()``),
and ``train_path`` / ``eval_path`` JSONL overrides let the environment run fully
offline (used by the unit tests and by air-gapped smoke runs).
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
    eval_dataset: str = "HuggingFaceH4/MATH-500"
    eval_config: str | None = None
    eval_split: str = "test"
    question_key: str = "problem"
    answer_key: str = "answer"
    solution_key: str = "solution"
    correct_reward: float = 1.0
    format_reward: float = 0.1
    eval_examples: int = 300
    eval_seed: int = 12345
    train_path: str | None = None
    eval_path: str | None = None

    def __init__(self, **kwargs):
        super().__init__(id="math")
        for field in (
            "train_dataset",
            "train_config",
            "train_configs",
            "train_split",
            "eval_dataset",
            "eval_config",
            "eval_split",
            "question_key",
            "answer_key",
            "solution_key",
            "train_path",
            "eval_path",
        ):
            if field in kwargs and kwargs[field] is not None:
                setattr(self, field, kwargs[field])
        self.correct_reward = float(kwargs.get("correct_reward", 1.0))
        self.format_reward = float(kwargs.get("format_reward", 0.1))
        # Default to the run's [train] eval_examples (the worker exports it as
        # EVAL_NUM) so MATH-500 configs that set only train.eval_examples=500 are
        # not pre-truncated to 300 before the worker slices. An explicit
        # environment.params.eval_examples still wins.
        import os

        default_eval = int(os.environ.get("EVAL_NUM") or 300)
        self.eval_examples = int(kwargs.get("eval_examples", default_eval))
        self.eval_seed = int(kwargs.get("eval_seed", 12345))

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
        is_eval = split in {"eval", "validation", "test"}
        path = self.eval_path if is_eval else self.train_path
        if path:
            rows = self._normalize_rows(self._load_jsonl(path))
        elif is_eval:
            rows = self._normalize_rows(
                self._load_hf(self.eval_dataset, self.eval_config, self.eval_split)
            )
        elif self.train_configs:
            raw = []
            for cfg in self.train_configs:
                raw.extend(self._load_hf(self.train_dataset, cfg, self.train_split))
            rows = self._normalize_rows(raw)
        else:
            rows = self._normalize_rows(
                self._load_hf(self.train_dataset, self.train_config, self.train_split)
            )
        if is_eval:
            rows = self._fixed_subset(rows)
        return rows

    def _fixed_subset(self, rows: list[dict]) -> list[dict]:
        import random

        n = min(self.eval_examples, len(rows))
        if n <= 0 or n >= len(rows):
            return rows
        idx = sorted(random.Random(self.eval_seed).sample(range(len(rows)), n))
        return [rows[i] for i in idx]

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
