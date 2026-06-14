"""Environment behaviors not covered elsewhere: the base loader/errors and loading a
LOCAL verifiers env module by path (verifiers-only — there are no built-in envs)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoslm.envs.base import BaseEnvironment, load_environment_from_path

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


def test_load_environment_from_path_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_environment_from_path(str(tmp_path / "missing.py"))

    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(AttributeError, match="load_environment"):
        load_environment_from_path(str(bad))

    good = tmp_path / "good.py"
    good.write_text(
        "from autoslm.envs.base import BaseEnvironment\n"
        "def load_environment(**kw):\n"
        "    return BaseEnvironment(id='custom')\n",
        encoding="utf-8",
    )
    assert load_environment_from_path(str(good)).id == "custom"


def test_load_environment_from_directory_uses_stem_module(tmp_path) -> None:
    pkg = tmp_path / "my-env"
    pkg.mkdir()
    (pkg / "my_env.py").write_text(
        "from autoslm.envs.base import BaseEnvironment\n"
        "def load_environment(**kw):\n"
        "    return BaseEnvironment(id='dir-env')\n",
        encoding="utf-8",
    )
    assert load_environment_from_path(str(pkg)).id == "dir-env"


# ---------------------------------------------------------------------------
# local verifiers env via [environment] path (the registry's path branch)
# ---------------------------------------------------------------------------

_LOCAL_VF_ENV = """\
import verifiers as vf
from datasets import Dataset


def load_environment(**kwargs):
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "2+2?"}], "answer": "4"}]
    )

    def correct(completion, answer, **_):
        text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
        return 1.0 if str(answer) in text else 0.0

    rubric = vf.Rubric(funcs=[correct], weights=[1.0])
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
"""


def test_load_local_verifiers_env_via_path(tmp_path) -> None:
    """A local verifiers env module is wrapped by the adapter, not treated as a builtin."""
    from autoslm.envs.registry import load_environment
    from autoslm.envs.verifiers_adapter import VerifiersEnvironment

    mod = tmp_path / "local_env.py"
    mod.write_text(_LOCAL_VF_ENV, encoding="utf-8")

    env = load_environment("local-env", path=str(mod))
    assert isinstance(env, VerifiersEnvironment)
    assert env.id == "local-env"
    assert env.multi_turn is False
    assert env.reward("the answer is 4", {"answer": "4"}) == 1.0
    assert env.reward("nope", {"answer": "4"}) == 0.0


def test_load_environment_requires_env_or_path() -> None:
    """Verifiers-only: an empty env id with no path is a hard error (no default env)."""
    from autoslm.envs.registry import load_environment

    with pytest.raises(ValueError, match="no environment specified"):
        load_environment("")


def test_env_init_scaffolds_a_loadable_verifiers_env(tmp_path, monkeypatch) -> None:
    """`slm env init` must scaffold a real verifiers env that load_environment can load
    (it returns a vf.Environment wrapped by the adapter), not a BaseEnvironment subclass."""
    from argparse import Namespace

    from autoslm.cli.main import cmd_env_init
    from autoslm.envs.registry import load_environment
    from autoslm.envs.verifiers_adapter import VerifiersEnvironment

    monkeypatch.chdir(tmp_path)
    assert cmd_env_init(Namespace(name="my-task")) == 0

    env_module = tmp_path / "environments" / "my_task" / "my_task.py"
    assert env_module.is_file()
    source = env_module.read_text(encoding="utf-8")
    # The scaffold must be a verifiers env, not the old BaseEnvironment subclass.
    assert "import verifiers as vf" in source
    assert "vf.SingleTurnEnv" in source
    assert "BaseEnvironment" not in source

    env = load_environment("my-task", path=str(env_module))
    assert isinstance(env, VerifiersEnvironment)
    assert env.multi_turn is False
    assert env.reward("the answer is 4", {"answer": "4"}) == 1.0
