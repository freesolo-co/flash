"""One-row image color training environment."""

from __future__ import annotations

import json
from pathlib import Path

from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult

DATASET_PATH = Path(__file__).parent / "dataset" / "train.jsonl"


def _load_rows(path: str | Path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ImageColorEnvironment(EnvironmentSingleTurn):
    dataset = _load_rows(DATASET_PATH)

    def build_prompt_messages(self, example: TaskExample, prompt_text: str):
        return [{"role": "user", "content": example.input}]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        score = 1.0 if response_text.strip() == "red" else 0.0
        return RewardResult(score=score, threshold=1.0)


def load_environment(dataset_path: str | None = None, **kwargs) -> ImageColorEnvironment:
    env = ImageColorEnvironment()
    if dataset_path:
        env.dataset = _load_rows(dataset_path)
    return env
