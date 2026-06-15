"""Small, serializable environment interface for SFT/RL jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Environment(Protocol):
    id: str

    def dataset(self, split: str) -> list[dict]:
        """Return the rows for ``split`` (e.g. ``"train"``)."""

    def prompt_messages(self, example: dict) -> list[dict]:
        """Chat messages fed to the model for one example."""

    def sft_target(self, example: dict) -> str:
        """Assistant target text for an SFT example."""

    def reward(self, completion: str, example: dict) -> float:
        """Scalar RL reward for a completion."""

    def grade(self, completion: str, example: dict) -> bool:
        """Boolean correctness scorer the reward can build on."""


@dataclass
class BaseEnvironment:
    id: str

    def dataset(self, split: str) -> list[dict]:
        raise NotImplementedError

    def prompt_messages(self, example: dict) -> list[dict]:
        question = example.get("question") or example.get("prompt") or ""
        return [{"role": "user", "content": question}]

    def sft_target(self, example: dict) -> str:
        return str(example.get("target") or example.get("answer") or "")

    def reward(self, completion: str, example: dict) -> float:
        return 1.0 if self.grade(completion, example) else 0.0

    def grade(self, completion: str, example: dict) -> bool:
        gold = str(example.get("gold") or example.get("answer") or "").strip()
        # A missing/empty gold must NOT grade every completion correct (`"" in x` is
        # always True) — treat it as unscorable -> incorrect.
        return bool(gold) and gold in (completion or "")

    def extract_pred(self, completion: str) -> str | None:
        """Best-effort extracted final answer (optional logging hook).

        Environments with a notion of a canonical predicted answer (e.g. a boxed
        number) override this. Defaults to None.
        """
        return None
