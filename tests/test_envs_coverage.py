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

    from flash.cli.main import cmd_env_setup

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
