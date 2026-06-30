"""Minimal image-question Freesolo environment for Flash multimodal training."""

from __future__ import annotations

import json
from pathlib import Path

from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult


DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "train.jsonl"


def load_jsonl(path: str | Path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ImageColorEnv(EnvironmentSingleTurn):
    dataset = load_jsonl(DEFAULT_DATASET_PATH)

    def build_prompt_messages(self, example: TaskExample, prompt_text: str):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{example.input} Answer with one word."},
                ],
            }
        ]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        expected = str(example.output or "").strip().lower()
        got = (response_text or "").strip().lower()
        score = 1.0 if expected and expected in got else 0.0
        return RewardResult(score=score, threshold=1.0)


def load_environment(dataset_path: str | None = None, **kwargs) -> ImageColorEnv:
    env = ImageColorEnv()
    if dataset_path:
        env.dataset = load_jsonl(dataset_path)
    return env
