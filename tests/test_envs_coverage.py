"""Environment behaviors not covered elsewhere: base loader errors, gsm8k
prompt/target/reward wiring, tests_pass end-to-end grading in a temp repo, and
the freesolo bridge's GRPO paths against a faked freesolo package."""

from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoslm.envs.base import BaseEnvironment, load_environment_from_path
from autoslm.envs.freesolo import FreesoloEnvironment
from autoslm.envs.tests_pass import TestsPassEnvironment
from tests._examples import load_example_module

GSM8KEnvironment = load_example_module("gsm8k", "env").GSM8KEnvironment

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


def test_directory_envs_do_not_shadow_each_others_siblings(tmp_path) -> None:
    # Two directory envs that each ship a `config.py` must each see their OWN sibling —
    # the first-loaded config must not be cached and reused for the second.
    def _make(name: str, env_id: str) -> str:
        d = tmp_path / name
        d.mkdir()
        (d / "config.py").write_text(f"ENV_ID = {env_id!r}\n", encoding="utf-8")
        (d / f"{name}.py").write_text(
            "from autoslm.envs.base import BaseEnvironment\n"
            "from config import ENV_ID\n"
            "def load_environment(**kw):\n"
            "    return BaseEnvironment(id=ENV_ID)\n",
            encoding="utf-8",
        )
        return str(d)

    a = _make("env_a", "from-a")
    b = _make("env_b", "from-b")
    assert load_environment_from_path(a).id == "from-a"
    assert load_environment_from_path(b).id == "from-b"  # not shadowed by env_a's config
    assert load_environment_from_path(a).id == "from-a"  # and reloadable


def test_directory_env_sibling_wins_over_app_module(tmp_path, monkeypatch) -> None:
    # A top-level `config` already cached (e.g. the control plane's own cli/src/config.py)
    # must NOT shadow the env's sibling `config.py`: sys.modules wins over sys.path, so
    # the loader has to evict the app module for the load and restore it afterwards.
    import sys
    import types

    app_config = types.ModuleType("config")
    app_config.ENV_ID = "from-app"
    monkeypatch.setitem(sys.modules, "config", app_config)

    d = tmp_path / "env_x"
    d.mkdir()
    (d / "config.py").write_text("ENV_ID = 'from-env-x'\n", encoding="utf-8")
    (d / "env_x.py").write_text(
        "from autoslm.envs.base import BaseEnvironment\n"
        "from config import ENV_ID\n"
        "def load_environment(**kw):\n"
        "    return BaseEnvironment(id=ENV_ID)\n",
        encoding="utf-8",
    )
    assert load_environment_from_path(str(d)).id == "from-env-x"  # env sibling, not app
    # the app's own `config` module is restored untouched after the load
    assert sys.modules["config"] is app_config
    assert sys.modules["config"].ENV_ID == "from-app"


# ---------------------------------------------------------------------------
# gsm8k (no dataset download: prompt/target/reward wiring only)
# ---------------------------------------------------------------------------


def test_gsm8k_prompt_target_and_grading() -> None:
    env = GSM8KEnvironment(correct_reward=2.0, format_reward=0.5)
    example = {
        "question": "2+2?",
        "solution": "2 + 2 = 4\n#### 4",
        "gold": "4",
    }
    messages = env.prompt_messages(example)
    assert messages[-1]["role"] == "user"
    assert "2+2?" in messages[-1]["content"]
    # gsm8k targets are rewritten to the boxed-answer format the grader expects
    assert "\\boxed{4}" in env.sft_target(example)
    assert env.grade("the total is #### 4", example) is True
    assert env.grade("#### 5", example) is False
    # reward is additive: correctness + a format bonus for a parseable answer
    assert env.reward("#### 4", example) == 2.5
    assert env.reward("#### 5", example) == 0.5
    assert env.reward("no numbers here at all", example) == 0.0


# ---------------------------------------------------------------------------
# tests_pass (real subprocess grading in a temp repo)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git unavailable",
)
def test_tests_pass_requires_code_exec_opt_in(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOSLM_ALLOW_CODE_EXEC", raising=False)
    with pytest.raises(ValueError, match="AUTOSLM_ALLOW_CODE_EXEC"):
        TestsPassEnvironment(workspace_path=str(tmp_path))


def test_tests_pass_applies_diff_and_runs_test_command(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOSLM_ALLOW_CODE_EXEC", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "value.txt").write_text("wrong\n", encoding="utf-8")

    env = TestsPassEnvironment(
        workspace_path=str(repo),
        test_command="grep -q right value.txt",
        timeout_seconds=30,
    )
    diff = "--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-wrong\n+right\n"
    assert env.grade(diff, {"prompt": "fix value.txt"}) is True
    assert env.grade("not a diff at all", {"prompt": "fix value.txt"}) is False
    # the original workspace is never mutated
    assert (repo / "value.txt").read_text() == "wrong\n"


def test_tests_pass_dataset_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n', encoding="utf-8")
    env = TestsPassEnvironment(examples_path=str(path))
    assert [e["prompt"] for e in env.dataset("train")] == ["a", "b"]
    # Missing/empty examples must fail loudly, not silently train a paid run on nothing.
    env_missing = TestsPassEnvironment(examples_path=str(tmp_path / "nope.jsonl"))
    with pytest.raises(FileNotFoundError, match="no examples"):
        env_missing.dataset("train")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no examples"):
        TestsPassEnvironment(examples_path=str(empty)).dataset("train")


# ---------------------------------------------------------------------------
# freesolo bridge: GRPO paths against a faked freesolo package
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_freesolo(monkeypatch):
    """Install a minimal in-memory freesolo package good enough for the bridge."""

    class _RewardResult:
        def __init__(self, score: float):
            self.score = score

        def resolved_success(self) -> bool:
            return self.score > 0.0

    class _Env:
        def start_episode(self, example, contract_text):
            return [
                {"role": "system", "content": contract_text},
                {"role": "user", "content": example.task},
            ]

        def score_responses(self, example, texts):
            return [_RewardResult(1.0 if str(example.expected_output) in t else 0.0) for t in texts]

    captured: dict = {}

    def load_environment(reference, **kwargs):
        captured["reference"] = reference
        captured["kwargs"] = kwargs
        return _Env()

    class _TaskExample:
        def __init__(self, record, task, task_id=None, metadata=None, expected_output=None):
            self.record, self.task = record, task
            self.task_id, self.metadata = task_id, metadata
            self.expected_output = expected_output

    base = types.ModuleType("freesolo.environments.base")
    base.load_environment = load_environment
    base.EnvironmentMultiTurn = type("EnvironmentMultiTurn", (), {})
    dtypes = types.ModuleType("freesolo.datasets.types")
    dtypes.TaskExample = _TaskExample
    for name, mod in {
        "freesolo": types.ModuleType("freesolo"),
        "freesolo.environments": types.ModuleType("freesolo.environments"),
        "freesolo.environments.base": base,
        "freesolo.datasets": types.ModuleType("freesolo.datasets"),
        "freesolo.datasets.types": dtypes,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return captured


GRPO_RECORDS = [{"record": {"q": "2+2"}, "task": "What is 2+2?", "expected_output": "4"}]


def test_grpo_prompts_and_scoring_via_freesolo(fake_freesolo) -> None:
    env = FreesoloEnvironment(
        contract_text="# contract",
        records=GRPO_RECORDS,
        mode="grpo",
        environment="freesolo/environment.py:load_environment",
    )
    messages = env.prompt_messages(GRPO_RECORDS[0])
    assert messages[0] == {"role": "system", "content": "# contract"}
    assert "2+2" in messages[1]["content"]
    assert env.reward("the answer is 4", GRPO_RECORDS[0]) == 1.0
    assert env.grade("no idea", GRPO_RECORDS[0]) is False
    # contract + dataset were materialized for the reconstructed environment
    kwargs = fake_freesolo["kwargs"]
    assert kwargs["mode"] == "grpo"
    with open(kwargs["contract_path"]) as f:
        assert f.read() == "# contract"


def test_grpo_materializes_shipped_environment_bundle(fake_freesolo) -> None:
    env = FreesoloEnvironment(
        contract_text="# contract",
        records=GRPO_RECORDS,
        mode="grpo",
        environment="freesolo/environment.py:make_env",
        environment_bundle={
            "environment.py": "# shipped module\n",
            "config.py": "GRPO_CONFIG = 1\n",
        },
    )
    env.reward("4", GRPO_RECORDS[0])
    ref = fake_freesolo["reference"]
    path, factory = ref.rsplit(":", 1)
    assert factory == "make_env"
    assert os.path.basename(path) == "environment.py"
    with open(path) as f:
        assert f.read() == "# shipped module\n"
    # The sibling config.py lands beside the entry module and its directory is
    # on sys.path, so `from config import GRPO_CONFIG` would resolve (the P1 fix).
    env_dir = os.path.dirname(path)
    with open(os.path.join(env_dir, "config.py")) as f:
        assert f.read() == "GRPO_CONFIG = 1\n"
    assert env_dir in sys.path


def test_grpo_named_reference_passes_through(fake_freesolo) -> None:
    env = FreesoloEnvironment(
        contract_text="# c",
        records=GRPO_RECORDS,
        mode="grpo",
        environment="hub-env",
        environment_bundle={"environment.py": "# ignored for named refs\n"},
    )
    env.reward("4", GRPO_RECORDS[0])
    assert fake_freesolo["reference"] == "hub-env"


def test_grpo_bundle_resolves_sibling_imports_end_to_end(monkeypatch) -> None:
    """The P1 regression: a shipped environment.py that does `from config import
    GRPO_CONFIG` must import on the worker. The fake loader genuinely executes
    the materialized entry module, so a missing config.py or sys.path entry
    raises ImportError instead of silently passing."""
    import importlib.util

    loaded: dict = {}

    def load_environment(reference, **kwargs):
        path, _, factory = reference.partition(":")
        spec = importlib.util.spec_from_file_location("shipped_env", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # runs `from config import GRPO_CONFIG`
        loaded["env"] = getattr(module, factory or "load_environment")()
        return loaded["env"]

    base = types.ModuleType("freesolo.environments.base")
    base.load_environment = load_environment
    base.EnvironmentMultiTurn = type("EnvironmentMultiTurn", (), {})
    dtypes = types.ModuleType("freesolo.datasets.types")
    dtypes.TaskExample = type(
        "TaskExample",
        (),
        {"__init__": lambda self, **kw: self.__dict__.update(kw)},
    )
    for name, mod in {
        "freesolo": types.ModuleType("freesolo"),
        "freesolo.environments": types.ModuleType("freesolo.environments"),
        "freesolo.environments.base": base,
        "freesolo.datasets": types.ModuleType("freesolo.datasets"),
        "freesolo.datasets.types": dtypes,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    env = FreesoloEnvironment(
        contract_text="# contract",
        records=GRPO_RECORDS,
        mode="grpo",
        environment="freesolo/environment.py:load_environment",
        environment_bundle={
            "config.py": "MAX_TOKENS = 256\n",
            "environment.py": (
                "from config import MAX_TOKENS\n"
                "class _E:\n"
                "    def score_responses(self, ex, texts):\n"
                "        return [type('R', (), {'score': float(MAX_TOKENS)})() for _ in texts]\n"
                "def load_environment(**kw):\n"
                "    return _E()\n"
            ),
        },
    )
    # Reconstructs the env (executing the sibling import) and scores with it.
    assert env.reward("anything", GRPO_RECORDS[0]) == 256.0


def test_multi_turn_freesolo_environment_rejected(fake_freesolo, monkeypatch) -> None:
    base = sys.modules["freesolo.environments.base"]

    class _MultiEnv(base.EnvironmentMultiTurn):
        pass

    monkeypatch.setattr(base, "load_environment", lambda *a, **k: _MultiEnv())
    env = FreesoloEnvironment(
        contract_text="# c",
        records=GRPO_RECORDS,
        mode="grpo",
        environment="freesolo/environment.py:load_environment",
    )
    with pytest.raises(ValueError, match="multi-turn"):
        env.reward("x", GRPO_RECORDS[0])
