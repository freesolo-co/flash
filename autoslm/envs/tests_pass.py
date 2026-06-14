"""Coding-agent environment: reward completions by running a test command.

SECURITY: when ``workspace_path`` is set, ``grade`` applies a model-generated diff to a
*temporary copy* of the repo and runs ``test_command`` in a subprocess. This is **not** a
security sandbox — the model's diff and the test command run with the worker's privileges
(and can read worker secrets such as the HF token). It is therefore an explicit operator
opt-in: the code-execution path is refused unless ``AUTOSLM_ALLOW_CODE_EXEC=1`` is set on
the worker. Only enable it for trusted tasks until container sandboxing lands.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass

from .base import BaseEnvironment

_CODE_EXEC_ENV = "AUTOSLM_ALLOW_CODE_EXEC"


def _code_exec_opt_in() -> bool:
    """Whether the operator opted into the workspace_path code-execution path.

    Truthy allowlist (not a falsey denylist): only an explicit truthy value enables it, so
    "false"/"False"/"no"/"off"/"0"/"" all correctly keep the code-exec path disabled.
    """
    return os.environ.get(_CODE_EXEC_ENV, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class TestsPassEnvironment(BaseEnvironment):
    examples_path: str = "examples.jsonl"
    test_command: str = "pytest -q"
    workspace_path: str | None = None
    timeout_seconds: int = 120

    def __init__(self, **kwargs):
        super().__init__(id="tests_pass")
        self.examples_path = kwargs.get("examples_path", self.examples_path)
        self.test_command = kwargs.get("test_command", self.test_command)
        self.workspace_path = kwargs.get("workspace_path")
        # Fail before any paid run: the workspace_path path executes a model-generated
        # diff + test_command on the worker (NOT a sandbox), so it requires an explicit
        # operator opt-in rather than running silently with worker privileges/secrets.
        if self.workspace_path and not _code_exec_opt_in():
            raise ValueError(
                "tests_pass with workspace_path runs a model-generated diff and "
                f"test_command on the worker (not a sandbox). Set {_CODE_EXEC_ENV}=1 on "
                "the worker to enable this code-execution path."
            )
        self.timeout_seconds = int(kwargs.get("timeout_seconds", self.timeout_seconds))

    def dataset(self, split: str) -> list[dict]:
        # An empty/missing dataset would silently train+eval a paid run on nothing.
        # The managed service rejects client-local examples_path/workspace_path, so a
        # managed tests_pass run must bundle examples in the uploaded code tree; fail
        # loudly when none are found rather than returning [].
        if not os.path.exists(self.examples_path):
            raise FileNotFoundError(
                f"tests_pass: no examples at {self.examples_path!r}. Bundle an "
                "examples.jsonl in the environment (or set examples_path) — a managed "
                "run cannot reference a client-local path."
            )
        rows = []
        with open(self.examples_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"tests_pass: {self.examples_path!r} contains no examples")
        return rows

    def prompt_messages(self, example: dict) -> list[dict]:
        prompt = example.get("prompt") or example.get("question") or ""
        return [
            {
                "role": "system",
                "content": "Produce a patch or answer that makes the repository tests pass.",
            },
            {"role": "user", "content": prompt},
        ]

    def sft_target(self, example: dict) -> str:
        return str(example.get("target") or example.get("patch") or "")

    def reward(self, completion: str, example: dict) -> float:
        return 1.0 if self.grade(completion, example) else 0.0

    def grade(self, completion: str, example: dict) -> bool:
        if not self.workspace_path:
            expected = str(example.get("expected") or "")
            return bool(expected and expected in (completion or ""))
        warnings.warn(
            "tests_pass executes a model-generated diff + test command in a subprocess; "
            "this is NOT a sandbox — only use it on trusted tasks.",
            stacklevel=2,
        )
        with tempfile.TemporaryDirectory(prefix="autoslm-tests-pass-") as tmp:
            workdir = os.path.join(tmp, "repo")
            shutil.copytree(self.workspace_path, workdir, dirs_exist_ok=True)
            patch = _extract_patch(completion or "")
            if patch:
                proc = subprocess.run(
                    ["git", "apply", "-"],
                    input=patch,
                    cwd=workdir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout_seconds,
                )
                if proc.returncode != 0:
                    return False
            # shell=False + shlex.split: run the test command as an argv list so a
            # configurable test_command can't inject extra shell commands. (Still not a
            # sandbox — see module docstring — but this removes the shell-parsing footgun.)
            proc = subprocess.run(
                shlex.split(self.test_command),
                cwd=workdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_seconds,
            )
            return proc.returncode == 0


def load_environment(**kwargs) -> TestsPassEnvironment:
    return TestsPassEnvironment(**kwargs)


def _extract_patch(text: str) -> str:
    if "```diff" in text:
        return text.split("```diff", 1)[1].split("```", 1)[0].strip() + "\n"
    if "diff --git " in text:
        return "diff --git " + text.split("diff --git ", 1)[1]
    return text if text.startswith(("diff --git ", "--- ")) else ""
