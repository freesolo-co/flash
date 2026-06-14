"""Small, serializable environment interface for SFT/RL jobs."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
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


def load_environment_from_path(path: str, **params) -> Environment:
    """Load a custom environment module exposing load_environment(**params)."""
    module_path = path
    abs_dir: str | None = None
    added_to_path = False
    if os.path.isdir(path):
        stem = os.path.basename(path.rstrip(os.sep)).replace("-", "_")
        module_path = os.path.join(path, f"{stem}.py")
        # Prepend the env dir so the entry module's sibling imports (the documented
        # `from config import ...`) resolve and take precedence; we remove it again (and
        # evict the siblings it loaded) after the load so two directory envs that each
        # ship a `config.py` don't shadow one another.
        abs_dir = os.path.abspath(path)
        if abs_dir not in sys.path:
            sys.path.insert(0, abs_dir)
            added_to_path = True
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"environment module not found: {module_path}")
    # Unique entry-module name per load so a second directory env doesn't overwrite the
    # first in sys.modules.
    mod_name = f"autoslm_custom_environment_{abs(hash(os.path.abspath(module_path)))}"
    before = set(sys.modules)
    # A sibling import (the documented `from config import ...`) resolves through
    # sys.modules BEFORE sys.path, so an already-cached top-level module of the same name
    # (e.g. the control plane's own `cli/src/config.py`) would shadow the env's sibling
    # even with abs_dir at the front of sys.path. Temporarily evict any cached module that
    # collides with a .py/package sibling in this env dir, and restore it after the load.
    shadowed: dict = {}
    if abs_dir is not None:
        with contextlib.suppress(OSError):
            for entry in os.listdir(abs_dir):
                if entry.endswith(".py") and entry != "__init__.py":
                    sib = entry[:-3]
                elif os.path.exists(os.path.join(abs_dir, entry, "__init__.py")):
                    sib = entry
                else:
                    continue
                if sib in sys.modules:
                    shadowed[sib] = sys.modules.pop(sib)
    try:
        spec = importlib.util.spec_from_file_location(mod_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import environment module: {module_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        if not hasattr(mod, "load_environment"):
            raise AttributeError(f"{module_path} must define load_environment(**kwargs)")
        return mod.load_environment(**params)
    finally:
        if abs_dir is not None:
            # Evict modules imported from this env dir during the load (its config.py and
            # the entry module) so the next directory env re-resolves its OWN siblings
            # instead of reusing these. The documented pattern imports siblings at module
            # top (resolved during exec above); a lazy in-method `import config` is the one
            # case this does not cover, but the managed worker loads a single env per run.
            for name in set(sys.modules) - before:
                cached = sys.modules.get(name)
                cached_file = getattr(cached, "__file__", None) or ""
                if cached_file and os.path.abspath(cached_file).startswith(abs_dir + os.sep):
                    sys.modules.pop(name, None)
            if added_to_path and abs_dir in sys.path:
                sys.path.remove(abs_dir)
        # Restore the app-level modules we evicted so the env's siblings could win.
        for name, cached in shadowed.items():
            sys.modules[name] = cached
