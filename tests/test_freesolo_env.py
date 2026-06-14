"""The freesolo bridge environment: registry wiring, SFT path, and param validation.

GRPO prompting/scoring requires the ``freesolo`` package on the worker and is
exercised by the SDK's live tests; here we cover everything that runs without it.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoslm.envs.freesolo import FreesoloEnvironment
from autoslm.envs.registry import list_environments, load_environment, worker_pip_for_env

SFT_RECORDS = [
    {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
]


def test_registry_exposes_freesolo_builtin():
    assert "freesolo" in list_environments()
    env = load_environment(
        "freesolo",
        params={"contract_text": "# contract", "records": SFT_RECORDS, "mode": "sft"},
    )
    assert isinstance(env, FreesoloEnvironment)


def test_worker_pip_installs_freesolo_only_when_scoring_needs_it():
    # GRPO reconstructs the freesolo environment on the worker.
    assert worker_pip_for_env("freesolo") == ["freesolo[full]"]
    assert worker_pip_for_env("freesolo", {"mode": "grpo"}) == ["freesolo[full]"]
    # SFT jobs are self-contained (records + target-based grading).
    assert worker_pip_for_env("freesolo", {"mode": "sft"}) == []
    # Other built-ins still need nothing extra.
    assert worker_pip_for_env("gsm8k") == []


def test_sft_grading_is_self_contained():
    env = FreesoloEnvironment(contract_text="# contract", records=SFT_RECORDS, mode="sft")
    # grade/reward never touch the freesolo package in SFT mode.
    assert env.grade("the answer is 4", SFT_RECORDS[0]) is True
    assert env.grade("no idea", SFT_RECORDS[0]) is False
    assert env.reward("4", SFT_RECORDS[0]) == 1.0
    assert env.reward("5", SFT_RECORDS[0]) == 0.0


def test_sft_prompt_and_target_split_conversation():
    env = FreesoloEnvironment(contract_text="# contract", records=SFT_RECORDS, mode="sft")
    assert env.dataset("train") == SFT_RECORDS
    assert env.prompt_messages(SFT_RECORDS[0]) == [{"role": "user", "content": "What is 2 + 2?"}]
    assert env.sft_target(SFT_RECORDS[0]) == "4"


def test_sft_target_requires_trailing_assistant_message():
    env = FreesoloEnvironment(contract_text="# contract", records=SFT_RECORDS, mode="sft")
    with pytest.raises(ValueError, match="assistant"):
        env.sft_target({"messages": [{"role": "user", "content": "hi"}]})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"contract_text": "", "records": SFT_RECORDS}, "contract_text"),
        ({"contract_text": "# c", "records": []}, "records"),
        ({"contract_text": "# c", "records": SFT_RECORDS, "mode": "dpo"}, "mode"),
    ],
)
def test_param_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FreesoloEnvironment(**kwargs)


def test_environment_bundle_param_is_stored_for_worker_materialization():
    env = FreesoloEnvironment(
        contract_text="# contract",
        records=[{"task": "t"}],
        mode="grpo",
        environment_bundle={
            "environment.py": "from config import GRPO_CONFIG\n",
            "config.py": "GRPO_CONFIG = None\n",
        },
    )
    assert env.environment_name is None
    assert set(env.environment_bundle) == {"environment.py", "config.py"}


def test_eval_split_does_not_leak_training_records():
    from autoslm.envs.freesolo import FreesoloEnvironment

    train = [{"task": "a"}, {"task": "b"}]
    # No held-out set: eval is empty, NOT the training records (no label leak).
    env = FreesoloEnvironment(contract_text="# c", records=train, mode="grpo")
    assert env.dataset("train") == train
    assert env.dataset("eval") == []
    # With a held-out set, eval reads it.
    held = [{"task": "z"}]
    env2 = FreesoloEnvironment(contract_text="# c", records=train, eval_records=held, mode="grpo")
    assert env2.dataset("eval") == held
    assert env2.dataset("train") == train


def test_upload_code_includes_examples_tree(monkeypatch):
    # The real autoslm package has a sibling examples/ dir in this repo, so just
    # capture the upload targets. Inject a clean fake huggingface_hub via setitem
    # (auto-restored) so a prior test's stub of that module can't interfere.
    import sys
    import types

    import autoslm.flash.train as train

    calls = []

    class _FakeApi:
        def __init__(self, *a, **k):
            pass

        def create_repo(self, *a, **k):
            pass

        def upload_folder(self, *, folder_path, path_in_repo, **k):
            calls.append(path_in_repo)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("HF_REPO", "org/runs")

    train.upload_code()
    assert "code/autoslm" in calls
    assert "code/examples" in calls  # the P1: examples tree must travel with code
