"""Submit-time environment preflight and the load failures it moves off the GPU.

Each failing package below is a minimal copy of a 2026-11-19 sweep environment that passed
`bench-flash preview` and then failed on a rented, cold-started GPU.
"""

from __future__ import annotations

import textwrap
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flash.cli.commands.env.testing.test import cmd_env_test
from flash.core.spec import JobSpec
from flash.envs.loading.loader import _load_resolved_freesolo_environment
from flash.envs.meta.static_preflight import (
    EnvironmentPreflightError,
    StaticEnvironmentFacts,
    check_environment_package,
)
from flash.runner.lifecycle import preparation

_DATASET = '{"input": "q", "output": "a"}\n'


def _package(tmp_path, source: str, **files: str):
    root = tmp_path / "env"
    (root / "dataset").mkdir(parents=True)
    (root / "dataset" / "train.jsonl").write_text(_DATASET)
    (root / "environment.py").write_text(textwrap.dedent(source))
    for name, text in files.items():
        (root / name).write_text(textwrap.dedent(text))
    return root / "environment.py"


_GOOD = textwrap.dedent(
    """
    from pathlib import Path
    from freesolo.datasets.records import load_task_examples
    from freesolo.environments import EnvironmentSingleTurn, RewardResult

    class Env(EnvironmentSingleTurn):
        def __init__(self, dataset_path=None):
            self.dataset = load_task_examples(dataset_path or Path(__file__).parent / "dataset/train.jsonl")

        def build_prompt_messages(self, example, prompt_text):
            return [{"role": "user", "content": example.input}]

        def score_response(self, example, response_text):
            return RewardResult(score=0.0, threshold=1.0)

    def load_environment(**kwargs):
        return Env(**kwargs)
    """
)


def test_a_loadable_package_passes(tmp_path):
    assert check_environment_package(_package(tmp_path, _GOOD)) == StaticEnvironmentFacts()


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "from freesolo.environments.base import Environment, ScoreResult",
            "cannot import name 'ScoreResult' from 'freesolo.environments.base'",
        ),
        (
            "from freesolo.datasets import load_task_examples",
            "cannot import name 'load_task_examples' from 'freesolo.datasets'",
        ),
        ("import freesolo.missing_module", "No module named 'freesolo.missing_module'"),
    ],
)
def test_sdk_imports_the_worker_cannot_resolve_are_rejected(tmp_path, line, expected):
    entrypoint = _package(tmp_path, line + "\n" + _GOOD)

    with pytest.raises(EnvironmentPreflightError, match=expected) as caught:
        check_environment_package(entrypoint)
    assert "environment.py line 1" in str(caught.value)
    assert "before any GPU is allocated" in str(caught.value)


def test_a_sibling_module_import_is_checked_too(tmp_path):
    entrypoint = _package(
        tmp_path,
        "from helpers import rows\n" + _GOOD,
        **{"helpers.py": "from freesolo.datasets import load_task_examples\nrows = []\n"},
    )

    with pytest.raises(EnvironmentPreflightError, match=r"helpers\.py line 1"):
        check_environment_package(entrypoint)


def test_guarded_and_non_sdk_imports_are_not_judged(tmp_path):
    source = """
        try:
            from freesolo.datasets import load_task_examples
        except ImportError:
            from freesolo.datasets.records import load_task_examples
        import some_dependency_installed_by_pip
    """
    check_environment_package(_package(tmp_path, textwrap.dedent(source) + _GOOD))


def test_a_package_pinning_its_own_sdk_skips_the_sdk_comparison(tmp_path):
    entrypoint = _package(tmp_path, "from freesolo.datasets import load_task_examples\n" + _GOOD)

    check_environment_package(entrypoint, check_sdk_imports=False)


def test_an_instantiated_class_missing_an_abstract_hook_is_rejected(tmp_path):
    source = _GOOD.replace("def score_response(", "def _unused(")

    with pytest.raises(EnvironmentPreflightError, match=r"abstract class Env .*'score_response'"):
        check_environment_package(_package(tmp_path, source))


def test_an_abstract_intermediate_base_that_is_never_built_passes(tmp_path):
    source = """
        from freesolo.environments import EnvironmentSingleTurn, RewardResult

        class Base(EnvironmentSingleTurn):
            def build_prompt_messages(self, example, prompt_text):
                return []

        class Env(Base):
            dataset = []

            def score_response(self, example, response_text):
                return RewardResult(score=0.0)

        def load_environment(**kwargs):
            return Env()
    """
    check_environment_package(_package(tmp_path, source))


def test_a_bare_environment_base_is_rejected(tmp_path):
    source = """
        from freesolo.environments.base import Environment

        class Env(Environment):
            def build_prompt_messages(self, example, prompt_text):
                return []

            def score_response(self, example, response_text):
                return None

        def load_environment(**kwargs):
            return Env()
    """
    with pytest.raises(EnvironmentPreflightError, match="subclasses Environment"):
        check_environment_package(_package(tmp_path, source))


def test_dataset_written_as_a_method_is_rejected(tmp_path):
    source = _GOOD.replace("self.dataset = ", "self._rows = ").replace(
        "    def build_prompt_messages",
        "    def dataset(self):\n        return list(self._rows)\n\n    def build_prompt_messages",
    )

    with pytest.raises(EnvironmentPreflightError, match="'method' object is not iterable"):
        check_environment_package(_package(tmp_path, source))


def test_dataset_as_a_property_passes(tmp_path):
    as_property = _GOOD.replace("self.dataset = ", "self._rows = ").replace(
        "    def build_prompt_messages",
        "    @property\n    def dataset(self):\n        return list(self._rows)\n\n"
        "    def build_prompt_messages",
    )
    check_environment_package(_package(tmp_path, as_property))


_ALTERNATIVE = textwrap.dedent(
    """
    class Legacy(EnvironmentSingleTurn):
        image_observations = True

    def load_environment(use_legacy=False, **kwargs):
        if use_legacy:
            return Legacy()
        return Env(**kwargs)
"""
)


def test_an_unselected_factory_branch_is_not_judged(tmp_path):
    # Legacy is incomplete and multimodal, but only a branch that may not run builds it.
    source = _GOOD.replace("def load_environment(**kwargs):\n    return Env(**kwargs)\n", "")
    facts = check_environment_package(_package(tmp_path, source + _ALTERNATIVE))

    assert facts == StaticEnvironmentFacts()


def test_the_unconditionally_built_class_is_still_judged(tmp_path):
    source = _GOOD.replace("def score_response(", "def _unused(").replace(
        "def load_environment(**kwargs):\n    return Env(**kwargs)\n",
        "def load_environment(**kwargs):\n    env = Env(**kwargs)\n    return env\n",
    )

    with pytest.raises(EnvironmentPreflightError, match=r"abstract class Env .*'score_response'"):
        check_environment_package(_package(tmp_path, source))


_INHERITED_DATASET = """
    from freesolo.environments import EnvironmentSingleTurn, RewardResult

    class Base(EnvironmentSingleTurn):
        def dataset(self):
            return []

        def build_prompt_messages(self, example, prompt_text):
            return []

        def score_response(self, example, response_text):
            return RewardResult(score=0.0)

    class Env(Base):
        OVERRIDE

    def load_environment(**kwargs):
        return Env()
"""


def test_a_subclass_property_overriding_a_base_dataset_method_passes(tmp_path):
    source = _INHERITED_DATASET.replace(
        "OVERRIDE", "@property\n        def dataset(self):\n            return [1]"
    )
    check_environment_package(_package(tmp_path, source))


def test_an_inherited_dataset_method_is_still_rejected(tmp_path):
    source = _INHERITED_DATASET.replace("OVERRIDE", "pass")

    with pytest.raises(EnvironmentPreflightError, match=r"Base\.dataset is a method"):
        check_environment_package(_package(tmp_path, source))


def test_image_observations_is_reported_for_the_built_class(tmp_path):
    source = _GOOD.replace(
        "class Env(EnvironmentSingleTurn):\n",
        "class Env(EnvironmentSingleTurn):\n    image_observations = True\n",
    )

    assert check_environment_package(_package(tmp_path, source)).image_observations is True


def test_the_loader_no_longer_forwards_the_contract_path(tmp_path):
    # tau-kimi / html-glm: load_environment(**kwargs) -> Env(**kwargs) with no contract_path.
    entrypoint = _package(tmp_path, _GOOD)
    (entrypoint.parent / "TRAINING_CONTRACT.md").write_text("follow the contract")

    env = _load_resolved_freesolo_environment("org/env", str(entrypoint), {})
    assert env.dataset()
    assert env._contract_text == "follow the contract"

    explicit = entrypoint.parent / "OTHER.md"
    explicit.write_text("explicit contract")
    env = _load_resolved_freesolo_environment(
        "org/env", str(entrypoint), {"contract_path": "OTHER.md"}
    )
    assert env._contract_text == "explicit contract"


def _grpo_spec(**train):
    spec = JobSpec.from_dict(
        {
            "algorithm": "grpo",
            "model": "Qwen/Qwen3.5-9B-Base",
            "environment": {"id": "org/project/env", "resolved_sha": "a" * 40},
        }
    )
    return replace(spec, train=replace(spec.train, **train))


def _resolve_to(monkeypatch, entrypoint):
    monkeypatch.setattr(
        "flash.envs.loading.loader._resolve_environment_reference",
        lambda *_args: str(entrypoint),
    )


def test_submit_preparation_rejects_a_broken_package(monkeypatch, tmp_path):
    entrypoint = _package(tmp_path, "from freesolo.datasets import load_task_examples\n" + _GOOD)
    _resolve_to(monkeypatch, entrypoint)

    with pytest.raises(EnvironmentPreflightError, match="load_task_examples"):
        preparation._preflight_environment_source(_grpo_spec())
    # the package's own SDK pin replaces the plane's SDK as the reference.
    spec = _grpo_spec()
    spec = replace(spec, environment=replace(spec.environment, pip=["freesolo==9.9.9"]))
    preparation._preflight_environment_source(spec)


def test_submit_preparation_skips_an_unresolvable_package(monkeypatch):
    def unreachable(*_args):
        raise OSError("registry down")

    monkeypatch.setattr("flash.envs.loading.loader._resolve_environment_reference", unreachable)
    preparation._preflight_environment_source(_grpo_spec())


@pytest.mark.parametrize(
    ("exclude_modules", "rejected"),
    [(r"^(?!model\.language_model(?:\.|$)).*$", True), (None, False)],
)
def test_image_env_warm_start_from_a_text_only_adapter_is_rejected(
    monkeypatch, tmp_path, exclude_modules, rejected
):
    # miniwob-astra: image_observations = True continued from a text-only SFT adapter.
    source = _GOOD.replace(
        "class Env(EnvironmentSingleTurn):\n",
        "class Env(EnvironmentSingleTurn):\n    image_observations = True\n",
    )
    _resolve_to(monkeypatch, _package(tmp_path, source))
    monkeypatch.setattr(
        "flash.adapters.lora_rank.load_artifact_adapter_config",
        lambda *_args: {"exclude_modules": exclude_modules},
    )
    spec = _grpo_spec(init_from_adapter="org/repo:sft/run/checkpoints/step-1")

    if rejected:
        with pytest.raises(ValueError, match="warm-start modality mismatch"):
            preparation._preflight_environment_source(spec)
    else:
        preparation._preflight_environment_source(spec)


def test_prepare_job_runs_the_environment_preflight():
    import inspect

    from flash.runner.lifecycle import submit

    assert "preparation._preflight_environment_source(worker_spec)" in inspect.getsource(
        submit.prepare_job
    )


_TOOL_HISTORY = [
    {"role": "user", "content": "cancel my order"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_order", "arguments": '{"order_id": "1"}'},
            }
        ],
    },
]


def test_grpo_prompt_with_tool_call_history_names_the_field():
    # tau-astra / spreadsheet-astra: start_episode replayed native tool calls with JSON-string
    # arguments, which crashed in jinja as "Can only get item pairs from a mapping."
    from flash.engine.worker.train.rl.launch.inputs import _load_training_records

    env = SimpleNamespace(
        dataset=lambda: [{"id": "row-7", "input": "x"}],
        prompt_messages=lambda _ex: _TOOL_HISTORY,
    )
    context = SimpleNamespace(config=SimpleNamespace(seed=0))

    with pytest.raises(ValueError, match=r"example 'row-7': message 1 .*\['tool_calls'\]"):
        _load_training_records(env, None, context=context)


def test_env_test_reports_grpo_tool_call_history(monkeypatch, tmp_path, capsys):
    import argparse

    env_dir = tmp_path / "local-env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**kwargs): pass\n")

    class Env:
        multi_turn = False
        max_turns = 1

        def dataset(self):
            return [{"id": "row-7", "input": "x", "output": "y"}]

        def prompt_messages(self, example):
            return _TOOL_HISTORY

        def sft_completion(self, example):
            return [{"role": "assistant", "content": "y"}]

        def reward(self, completion, example, state=None):
            return 1.0

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda *_a, **_k: Env()
    )
    args = argparse.Namespace(path=str(env_dir), split=None, param=[], algorithm="grpo")
    assert cmd_env_test(args) != 0
    assert "carries ['tool_calls']" in capsys.readouterr().err
    args.algorithm = "sft"
    assert cmd_env_test(args) == 0
