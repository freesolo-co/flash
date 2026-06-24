"""Small, serializable environment interface for SFT/RL jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Environment(Protocol):
    id: str

    def dataset(self) -> list[dict]:
        """Return the training rows (the only split used; eval is on the serving side)."""

    def prompt_messages(self, example: dict) -> list[dict]:
        """Chat messages fed to the model for one example."""

    def sft_completion(self, example: dict) -> list[dict]:
        """Gold completion messages appended after the prompt for one SFT example — a multi-turn
        trajectory or a single assistant turn."""

    def reward(self, completion: str, example: dict, state: dict | None = None) -> float:
        """Scalar RL reward for a completion."""

    def grade(self, completion: str, example: dict, state: dict | None = None) -> bool:
        """Boolean correctness scorer the reward can build on."""


@dataclass
class BaseEnvironment:
    id: str

    def dataset(self) -> list[dict]:
        raise NotImplementedError

    def prompt_messages(self, example: dict) -> list[dict]:
        return [{"role": "user", "content": str(example.get("input") or "")}]

    def sft_completion(self, example: dict) -> list[dict]:
        # Single-turn default: one target assistant turn from the record's scalar ``output``.
        # FreesoloEnvironment overrides this to support multi-turn target trajectories via the
        # freesolo-sdk (``Environment.sft_completion`` -> ``datasets.target_messages``).
        return [{"role": "assistant", "content": str(example.get("output") or "")}]

    def reward(self, completion: str, example: dict, state: dict | None = None) -> float:
        return 1.0 if self.grade(completion, example, state) else 0.0

    def grade(self, completion: str, example: dict, state: dict | None = None) -> bool:
        gold = str(example.get("output") or "").strip()
        # A missing/empty output must NOT grade every completion correct (`"" in x` is
        # always True) — treat it as unscorable -> incorrect.
        return bool(gold) and gold in (completion or "")
