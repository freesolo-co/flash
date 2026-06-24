"""Environment behaviors not covered elsewhere: the base env protocol and the registry's
id-based loader (Freesolo env ids; no built-in envs and no local-path mode)."""

from __future__ import annotations

import pytest

from flash.envs.base import BaseEnvironment

# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


def test_base_environment_defaults() -> None:
    env = BaseEnvironment(id="x")
    example = {"input": "Q?", "output": "42"}
    assert env.prompt_messages(example) == [{"role": "user", "content": "Q?"}]
    assert env.sft_completion(example) == [{"role": "assistant", "content": "42"}]
    assert env.grade("the answer is 42", example) is True
    assert env.reward("nope", example) == 0.0


# ---------------------------------------------------------------------------
# registry loader (id-based, Freesolo)
# ---------------------------------------------------------------------------


def test_load_environment_requires_env_id() -> None:
    """An empty env id is a hard error (no default env, no local path)."""
    from flash.envs.registry import load_environment

    with pytest.raises(ValueError, match="no environment specified"):
        load_environment("")


def test_env_setup_scaffolds_a_loadable_freesolo_env(tmp_path, monkeypatch) -> None:
    """`flash env setup` must scaffold a Freesolo SDK env, not a BaseEnvironment subclass."""
    from argparse import Namespace

    from flash.cli import cmd_env_setup

    monkeypatch.chdir(tmp_path)
    assert cmd_env_setup(Namespace()) == 0

    env_module = tmp_path / "environment.py"
    dataset = tmp_path / "datasets/train.jsonl"
    assert env_module.is_file()
    assert dataset.is_file()
    source = env_module.read_text(encoding="utf-8")
    # The scaffold must be a Freesolo SDK env, not the old BaseEnvironment subclass.
    assert "from freesolo.environments import EnvironmentSingleTurn" in source
    assert "class StarterEnv(EnvironmentSingleTurn)" in source
    assert 'Path(__file__).parent / "datasets" / "train.jsonl"' in source
    assert "load_jsonl(DEFAULT_DATASET_PATH)" in source
    assert "BaseEnvironment" not in source
    # The scaffold must read TaskExample's REAL attributes (.input/.output), not the .task/
    # .expected_output names that freesolo>=0.2.49 dropped — those raise AttributeError at scoring
    # time for anyone who edits the reward and submits a run (issue: scaffold vs pinned SDK).
    assert "example.input" in source
    assert "example.output" in source
    assert "example.task" not in source
    assert "expected_output" not in source


def test_scaffolded_env_scores_through_real_task_example(tmp_path, monkeypatch) -> None:
    """Load the unedited scaffold and exercise it with a real freesolo TaskExample — the exact
    path that used to raise ``AttributeError: 'TaskExample' object has no attribute
    'expected_output'``. Skips where the freesolo SDK isn't installed."""
    import importlib.util
    from argparse import Namespace

    pytest.importorskip("freesolo")
    from flash.cli.main import cmd_env_setup
    from freesolo.datasets.records import load_task_examples

    monkeypatch.chdir(tmp_path)
    assert cmd_env_setup(Namespace()) == 0

    spec = importlib.util.spec_from_file_location("scaffold_env", tmp_path / "environment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = mod.load_environment()

    example = load_task_examples([{"input": "What is 2 + 2?", "output": "4"}])[0]
    # build_prompt_messages used example.task; exact_match_reward used example.expected_output —
    # both must now resolve against the real .input/.output fields without raising.
    messages = env.build_prompt_messages(example, "")
    assert messages == [{"role": "user", "content": "What is 2 + 2?"}]
    assert mod.exact_match_reward(example, "the answer is 4").score == 1.0
    assert mod.exact_match_reward(example, "nope").score == 0.0
