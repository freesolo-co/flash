"""Environment behaviors not covered elsewhere: the base env protocol and the registry's
id-based loader (verifiers-only — there are no built-in envs and no local-path mode)."""

from __future__ import annotations

import pytest

from flash.envs.base import BaseEnvironment

# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


def test_base_environment_defaults() -> None:
    env = BaseEnvironment(id="x")
    example = {"question": "Q?", "answer": "42"}
    assert env.prompt_messages(example) == [{"role": "user", "content": "Q?"}]
    assert env.sft_target(example) == "42"
    assert env.grade("the answer is 42", example) is True
    assert env.reward("nope", example) == 0.0


# ---------------------------------------------------------------------------
# registry loader (id-based, verifiers-only)
# ---------------------------------------------------------------------------


def test_load_environment_requires_env_id() -> None:
    """Verifiers-only: an empty env id is a hard error (no default env, no local path)."""
    from flash.envs.registry import load_environment

    with pytest.raises(ValueError, match="no environment specified"):
        load_environment("")


def test_env_init_scaffolds_a_loadable_verifiers_env(tmp_path, monkeypatch) -> None:
    """`slm env init` must scaffold a real verifiers env (a vf.Environment), not a
    BaseEnvironment subclass — so a publish to the Hub actually loads."""
    from argparse import Namespace

    from flash.cli.main import cmd_env_init

    monkeypatch.chdir(tmp_path)
    assert cmd_env_init(Namespace(name="my-task")) == 0

    env_module = tmp_path / "environments" / "my_task" / "my_task.py"
    assert env_module.is_file()
    source = env_module.read_text(encoding="utf-8")
    # The scaffold must be a verifiers env, not the old BaseEnvironment subclass.
    assert "import verifiers as vf" in source
    assert "vf.SingleTurnEnv" in source
    assert "BaseEnvironment" not in source
