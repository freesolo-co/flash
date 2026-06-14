"""Bridge environment: run freesolo SDK contracts/datasets on AutoSLM workers.

The freesolo SDK submits training jobs with this environment instead of running
Tinker loops client-side. The job carries the freesolo task inline as params
(contract text + dataset records + optional freesolo environment name); on the
GPU worker the ``freesolo`` package is pip-installed (see
``registry.worker_pip_for_env``) and this adapter reconstructs the freesolo
environment to build prompts and score completions.

Single-turn only: AutoSLM's environment protocol has no episode loop, so
multi-turn freesolo environments are rejected at load time.

Params:
- ``contract_text`` (required): the TRAINING_CONTRACT.md content.
- ``records`` (required): for ``mode="grpo"``, serialized freesolo TaskExample
  dicts (``{"record", "task", "task_id", "metadata", "expected_output"}``);
  for ``mode="sft"``, ``{"messages": [...]}`` conversation dicts whose final
  message is the assistant turn to train on.
- ``mode``: ``"sft"`` or ``"grpo"`` (default ``"grpo"``).
- ``environment``: optional freesolo environment name/reference.
- ``environment_bundle``: optional ``{filename: source}`` of every ``*.py`` in
  the repo's environment directory (``freesolo/environment.py`` plus siblings
  such as ``config.py``). The whole directory is materialized on the worker and
  put on ``sys.path`` so the entry module's sibling imports (the documented
  ``from config import GRPO_CONFIG``) resolve, reproducing the exact scorer
  validated client-side.
- ``reward_command``: optional shell reward command (freesolo ``tests_pass``-style envs).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile


class FreesoloEnvironment:
    id = "freesolo"

    def __init__(
        self,
        *,
        contract_text: str,
        records: list[dict],
        mode: str = "grpo",
        environment: str | None = None,
        environment_bundle: dict[str, str] | None = None,
        reward_command: str | None = None,
        **_: object,
    ) -> None:
        if not contract_text:
            raise ValueError("freesolo environment requires non-empty contract_text")
        if not records:
            raise ValueError("freesolo environment requires non-empty records")
        if mode not in ("sft", "grpo"):
            raise ValueError(f'mode must be "sft" or "grpo", got {mode!r}')
        self.contract_text = contract_text
        self.records = list(records)
        self.mode = mode
        self.environment_name = environment
        self.environment_bundle = environment_bundle
        self.reward_command = reward_command
        self._env = None
        # Holds the materialized environment dir alive for the env's lifetime; its
        # finalizer removes the temp tree on GC / interpreter exit (no mkdtemp leak).
        self._workdir = None

    # -- freesolo plumbing ---------------------------------------------------

    def _freesolo_env(self):
        """Lazily reconstruct the freesolo environment on the worker."""
        if self._env is not None:
            return self._env
        from freesolo.environments.base import (
            EnvironmentMultiTurn,
            load_environment,
        )

        # TemporaryDirectory (kept on self), not mkdtemp: the reconstructed env reads
        # these files (contract/dataset/bundle) for its whole life, so the dir must
        # outlive this call — but be cleaned up when the env is GC'd instead of leaked.
        self._workdir = tempfile.TemporaryDirectory(prefix="freesolo-env-")
        workdir = self._workdir.name
        contract_path = os.path.join(workdir, "TRAINING_CONTRACT.md")
        with open(contract_path, "w", encoding="utf-8") as f:
            f.write(self.contract_text)
        dataset_path = os.path.join(workdir, "dataset.jsonl")
        with open(dataset_path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record) + "\n")
        reference = self.environment_name
        if self.environment_bundle and (
            not reference or reference.split(":", 1)[0].endswith(".py")
        ):
            # Materialize the whole shipped environment directory so the entry
            # module's sibling imports (the documented `from config import
            # GRPO_CONFIG`) resolve; the original repo path does not exist here.
            for filename, source in self.environment_bundle.items():
                with open(
                    os.path.join(workdir, os.path.basename(filename)), "w", encoding="utf-8"
                ) as f:
                    f.write(source)
            # load_environment_from_path uses spec_from_file_location, which does
            # not put the dir on the import path, so add it explicitly.
            if workdir not in sys.path:
                sys.path.insert(0, workdir)
            ref_path = reference.split(":", 1)[0] if reference else ""
            entry = os.path.basename(ref_path) if ref_path else "environment.py"
            factory = (
                reference.split(":", 1)[1] if reference and ":" in reference else "load_environment"
            )
            reference = f"{os.path.join(workdir, entry)}:{factory}"
        if not reference:
            # GRPO needs the environment to compute the reward. With neither an
            # `environment` reference nor an `environment_bundle`, load_environment(None)
            # would fall back to a non-existent freesolo/environment.py and fail deep in
            # the loader — surface the real cause here instead.
            raise ValueError(
                "freesolo GRPO bridge requires an environment: set [environment] "
                "environment to a 'freesolo/environment.py:load_environment' reference, "
                "or ship the environment via environment_bundle. Without one there is no "
                "reward function to train against."
            )
        env = load_environment(
            reference,
            reward_command=self.reward_command,
            contract_path=contract_path,
            dataset_path=dataset_path,
            mode=self.mode,
        )
        if isinstance(env, EnvironmentMultiTurn):
            raise ValueError(
                "freesolo multi-turn environments are not supported on AutoSLM "
                "(single-turn protocol); use a single-turn environment"
            )
        self._env = env
        return env

    def _task_example(self, example: dict):
        from freesolo.datasets.types import TaskExample

        return TaskExample(
            record=example.get("record") or {},
            task=str(example.get("task") or ""),
            task_id=example.get("task_id"),
            metadata=example.get("metadata") or {},
            expected_output=example.get("expected_output"),
        )

    # -- AutoSLM environment protocol -----------------------------------------

    def dataset(self, split: str) -> list[dict]:
        return self.records

    def prompt_messages(self, example: dict) -> list[dict]:
        if self.mode == "sft":
            messages = list(example.get("messages") or [])
            return [dict(m) for m in messages[:-1]]
        env = self._freesolo_env()
        messages = env.start_episode(self._task_example(example), self.contract_text)
        return [dict(m) for m in messages]

    def sft_target(self, example: dict) -> str:
        messages = list(example.get("messages") or [])
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("freesolo SFT records must end with an assistant message")
        return str(messages[-1].get("content") or "")

    def reward(self, completion: str, example: dict) -> float:
        if self.mode == "sft":
            return 1.0 if self.grade(completion, example) else 0.0
        env = self._freesolo_env()
        task = self._task_example(example)
        result = env.score_responses(task, [completion or ""])[0]
        return float(result.score)

    def grade(self, completion: str, example: dict) -> bool:
        if self.mode == "sft":
            # SFT records carry no freesolo environment context on the worker
            # (and need none for training); eval grades against the target
            # turn so SFT jobs are fully self-contained.
            target = self.sft_target(example).strip()
            # Word-boundary match, not raw substring: a short target like "4" must not
            # count as correct inside "14"/"answer42". re.escape keeps targets with
            # regex metacharacters literal.
            return bool(target) and (
                re.search(rf"(?<!\w){re.escape(target)}(?!\w)", completion or "") is not None
            )
        env = self._freesolo_env()
        task = self._task_example(example)
        result = env.score_responses(task, [completion or ""])[0]
        return bool(result.resolved_success())


def load_environment(**params) -> FreesoloEnvironment:
    return FreesoloEnvironment(**params)
