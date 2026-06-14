"""Small, serializable environment interface for SFT/RL/eval jobs."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Protocol


class Environment(Protocol):
    id: str

    def dataset(self, split: str) -> list[dict]: ...

    def prompt_messages(self, example: dict) -> list[dict]: ...

    def sft_target(self, example: dict) -> str: ...

    def reward(self, completion: str, example: dict) -> float: ...

    def grade(self, completion: str, example: dict) -> bool: ...


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
        gold = str(example.get("gold") or example.get("answer") or "")
        return gold.strip() in (completion or "")

    def extract_pred(self, completion: str) -> str | None:
        """Best-effort extracted final answer, for eval logging only.

        Optional; environments that have a notion of a canonical predicted
        answer (e.g. a boxed number) override this. Defaults to None.
        """
        return None


def load_environment_from_path(path: str, **params) -> Environment:
    """Load a custom environment module exposing load_environment(**params)."""
    module_path = path
    if os.path.isdir(path):
        stem = os.path.basename(path.rstrip(os.sep)).replace("-", "_")
        module_path = os.path.join(path, f"{stem}.py")
        # Put the env dir on sys.path so the entry module's sibling imports (the
        # documented `from config import ...`) resolve — spec_from_file_location alone
        # does not add the directory to the import path.
        abs = os.path.abspath(path)
        if abs not in sys.path:
            sys.path.insert(0, abs)
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"environment module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("autoslm_custom_environment", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import environment module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load_environment"):
        raise AttributeError(f"{module_path} must define load_environment(**kwargs)")
    return mod.load_environment(**params)
