"""Tests for the Freesolo SDK environment adapter + install manifest."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sys
import tarfile
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pytest


@dataclass
class _TaskExample:
    record: dict
    task: str
    task_id: str | None = None
    expected_output: object | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _RewardMetric:
    name: str
    score: float | None = None


@dataclass(frozen=True)
class _RewardResult:
    score: float
    metrics: tuple[_RewardMetric, ...] = ()
    success: bool | None = None
    threshold: float | None = None
    error: str | None = None

    def resolved_success(self) -> bool:
        if self.success is not None:
            return self.success
        if self.error:
            return False
        if self.threshold is not None:
            return self.score >= self.threshold
        return self.score > 0.0


@dataclass(frozen=True)
class _EnvironmentTurn:
    role: str
    content: str


@dataclass(frozen=True)
class _EnvironmentEpisode:
    messages: tuple[dict, ...]
    response_text: str
    turns: tuple[_EnvironmentTurn, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _EnvironmentStepResult:
    done: bool = True
    messages: tuple[dict, ...] = ()
    final_response_text: str | None = None
    metadata: dict = field(default_factory=dict)


class _EnvironmentSingleTurn:
    pass


class _EnvironmentMultiTurn:
    pass


class _FakeSingleTurnEnv(_EnvironmentSingleTurn):
    dataset: ClassVar[list[dict]] = [
        {
            "id": "ex-1",
            "input": "2+2?",
            "output": "4",
            "metadata": {"split": "train"},
        }
    ]

    def start_episode(self, example, prompt_text):
        return [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": example.task},
        ]

    def score_responses(self, example, response_texts):
        out = []
        for response in response_texts:
            score = 1.0 if str(example.expected_output) in response else 0.0
            out.append(
                _RewardResult(
                    score=score,
                    success=score == 1.0,
                    metrics=(_RewardMetric("match", score),),
                )
            )
        return out


class _FakeMultiTurnEnv(_EnvironmentMultiTurn):
    def start_episode(self, example, prompt_text):
        return [{"role": "user", "content": f"{prompt_text}:{example.task}"}]

    def step_episode(self, example, messages, assistant_response):
        return _EnvironmentStepResult(
            done=True,
            messages=({"role": "user", "content": f"observed {assistant_response}"},),
            final_response_text=f"final {assistant_response}",
            metadata={"input": example.task},
        )

    def score_episodes(self, example, episodes):
        return [
            _RewardResult(
                score=0.5,
                success=True,
                metrics=(_RewardMetric("episode", 0.5),),
            )
            for _episode in episodes
        ]


class _BudgetMultiTurnEnv(_EnvironmentMultiTurn):
    """Multi-turn env with a per-example budget that never self-terminates (done=False),
    so the rollout cap (max_episode_turns) is what must stop it."""

    def start_episode(self, example, prompt_text):
        return [{"role": "user", "content": "go"}]

    def max_episode_turns(self, example):
        return 15

    def step_episode(self, example, messages, assistant_response):
        return _EnvironmentStepResult(
            done=False,
            messages=({"role": "user", "content": "more"},),
            final_response_text=None,
            metadata={},
        )

    def score_episodes(self, example, episodes):
        return [_RewardResult(score=0.0, success=False, metrics=()) for _ in episodes]


def test_freesolo_sft_completion_full_gold_trajectory(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(_FakeSingleTurnEnv(), "owner/env", source=None, contract_text="")
    # A record whose `output` is a chat-message list -> the full gold trajectory (multi-turn SFT).
    gold = [
        {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        {"role": "user", "content": "<tool_result>...</tool_result>"},
        {"role": "assistant", "content": "done"},
    ]
    assert env.sft_completion({"input": "x", "output": gold}) == gold  # len>1 -> multi-turn
    # A scalar `output` is single-turn SFT -> one assistant turn.
    assert env.sft_completion({"input": "x", "output": "4"}) == [
        {"role": "assistant", "content": "4"}
    ]
    assert env.sft_completion({"input": "x"}) == [{"role": "assistant", "content": ""}]


def test_freesolo_multiturn_respects_per_example_budget(monkeypatch):
    _install_fake_freesolo(monkeypatch, sdk_env=_BudgetMultiTurnEnv())

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _BudgetMultiTurnEnv(),
        "owner/env",
        source=[{"input": "go", "output": ""}],
        contract_text="",
    )
    # max_turns is now the dataset-wide max_episode_turns (15), not the old hardcoded 8 —
    # otherwise the rollout loop would truncate this 15-turn scenario at turn 8.
    assert env.max_turns == 15

    state = env.new_rollout_state({"input": "go", "output": ""})
    assert state["max_episode_turns"] == 15
    # rollout_done honors THIS rollout's budget even when the batch cap is larger, and even
    # though the env never sets done=True.
    state["turn"] = 14
    assert env.rollout_done(state, max_turns=999) is False
    state["turn"] = 15
    assert env.rollout_done(state, max_turns=999) is True
    # done=True still short-circuits before the budget.
    state["turn"] = 0
    state["done"] = True
    assert env.rollout_done(state, max_turns=999) is True


def _install_fake_freesolo(monkeypatch, *, sdk_env=None, seen=None):
    sdk_env = sdk_env or _FakeSingleTurnEnv()
    seen = seen if seen is not None else {}

    def task_example_from_record(record):
        return _TaskExample(
            record=dict(record),
            task=str(record["input"]),
            task_id=record.get("id"),
            expected_output=record.get("output"),
            metadata=dict(record.get("metadata") or {}),
        )

    def load_task_examples(source):
        if isinstance(source, (list, tuple)):
            return [
                item if isinstance(item, _TaskExample) else task_example_from_record(item)
                for item in source
            ]
        path = os.fspath(source)
        rows = []
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            rows = loaded if isinstance(loaded, list) else loaded.get("records", [])
        return [task_example_from_record(row) for row in rows]

    def load_environment(reference, **kwargs):
        seen["reference"] = reference
        seen["kwargs"] = kwargs
        return sdk_env

    freesolo = types.ModuleType("freesolo")
    datasets = types.ModuleType("freesolo.datasets")
    records = types.ModuleType("freesolo.datasets.records")
    records.load_task_examples = load_task_examples
    records.task_example_from_record = task_example_from_record
    envs = types.ModuleType("freesolo.environments")
    envs.EnvironmentEpisode = _EnvironmentEpisode
    envs.EnvironmentMultiTurn = _EnvironmentMultiTurn
    envs.EnvironmentSingleTurn = _EnvironmentSingleTurn
    envs.EnvironmentStepResult = _EnvironmentStepResult
    envs.EnvironmentTurn = _EnvironmentTurn
    envs.RewardMetric = _RewardMetric
    envs.RewardResult = _RewardResult
    envs.load_environment = load_environment
    monkeypatch.setitem(sys.modules, "freesolo", freesolo)
    monkeypatch.setitem(sys.modules, "freesolo.datasets", datasets)
    monkeypatch.setitem(sys.modules, "freesolo.datasets.records", records)
    monkeypatch.setitem(sys.modules, "freesolo.environments", envs)
    return seen


def _github_environment_tarball(
    top_dir: str,
    *,
    env_path: str = "envs/e/environment.py",
    env_text: str = "def load_environment(**kwargs):\n    return None\n",
) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        dir_info = tarfile.TarInfo(f"{top_dir}/")
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        tar.addfile(dir_info)

        env_info = tarfile.TarInfo(f"{top_dir}/{env_path}")
        env_bytes = env_text.encode("utf-8")
        env_info.size = len(env_bytes)
        env_info.mode = 0o644
        tar.addfile(env_info, io.BytesIO(env_bytes))
    return buf.getvalue()


def test_freesolo_adapter_mapping(monkeypatch, tmp_path):
    seen = _install_fake_freesolo(monkeypatch)
    env_file = tmp_path / "freesolo" / "environment.py"
    env_file.parent.mkdir()
    env_file.write_text("def load_environment(**kwargs): pass\n")
    dataset = tmp_path / "freesolo" / "datasets" / "train.jsonl"
    dataset.parent.mkdir()
    dataset.write_text('{"id":"row-1","input":"2+2?","output":"4"}\n')

    from flash.envs.adapter import load_freesolo_environment

    env = load_freesolo_environment(
        str(env_file),
        dataset_path="datasets/train.jsonl",
        contract_text="be brief",
        difficulty="hard",
    )

    assert env.id == str(env_file)
    assert seen["reference"] == str(env_file)
    assert seen["kwargs"]["dataset_path"] == str(dataset)
    assert seen["kwargs"]["difficulty"] == "hard"

    train = env.dataset()
    assert train == [{"id": "row-1", "input": "2+2?", "output": "4"}]
    assert env.prompt_messages(train[0]) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "2+2?"},
    ]
    assert env.reward("the answer is 4", train[0]) == 1.0
    assert env.grade("the answer is 4", train[0]) is True
    assert env.reward("nope", train[0]) == 0.0
    assert env.scores_breakdown("the answer is 4", train[0]) == {"match": 1.0, "total": 1.0}
    assert env.sft_completion({"output": "4"}) == [{"role": "assistant", "content": "4"}]


def test_freesolo_adapter_uses_env_dataset_when_no_source(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _FakeSingleTurnEnv(),
        "owner/env",
        source=None,
        contract_text="",
    )
    assert env.dataset()[0]["output"] == "4"


def test_freesolo_adapter_exports_sdk_examples_as_input_output(monkeypatch):
    class SdkExampleEnv(_EnvironmentSingleTurn):
        dataset: ClassVar[list[_TaskExample]] = [
            _TaskExample(
                record={},
                task="2+2?",
                task_id="ex-1",
                expected_output="4",
                metadata={"split": "train"},
            )
        ]

    _install_fake_freesolo(monkeypatch, sdk_env=SdkExampleEnv())

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        SdkExampleEnv(),
        "owner/env",
        source=None,
        contract_text="",
    )
    assert env.dataset() == [
        {
            "input": "2+2?",
            "output": "4",
            "id": "ex-1",
            "metadata": {"split": "train"},
        }
    ]


def test_freesolo_adapter_does_not_accept_record_aliases(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _FakeSingleTurnEnv(),
        "owner/env",
        source=None,
        contract_text="",
    )
    # `expected_output` is NOT an alias for `output`, so there is no gold completion (empty turn).
    assert env.sft_completion({"expected_output": "4"}) == [{"role": "assistant", "content": ""}]
    with pytest.raises(ValueError, match="input field"):
        env.prompt_messages({"task": "2+2?", "output": "4"})


def test_freesolo_multiturn_hooks(monkeypatch):
    _install_fake_freesolo(monkeypatch, sdk_env=_FakeMultiTurnEnv())

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _FakeMultiTurnEnv(),
        "owner/env",
        source=[{"input": "browse", "output": "done"}],
        contract_text="contract",
    )
    state = env.new_rollout_state({"input": "browse", "output": "done"})
    assert state["prompt"] == [{"role": "user", "content": "contract:browse"}]
    assert state["messages"] == [{"role": "user", "content": "contract:browse"}]
    env.record_model_turn(state, "click")
    replies = env.env_reply(state["messages"], state)
    assert replies == [{"role": "user", "content": "observed click"}]
    assert state["done"] is True
    assert env.rollout_done(state) is True
    assert env.reward("ignored", {"input": "browse", "output": "done"}, state) == 0.5
    assert env.grade("ignored", {"input": "browse", "output": "done"}, state) is True
    assert (
        env.reward_from_messages(
            [{"role": "assistant", "content": "final"}],
            {"input": "browse", "output": "done"},
            [{"role": "user", "content": "contract:browse"}],
        )
        == 0.5
    )


def _env_package_tar(files: dict[str, str]) -> bytes:
    """A flat env package tarball (environment.py at the root), like `flash env push` uploads."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_environment_id_helpers():
    from flash.envs.adapter import is_freesolo_environment_id, is_managed_environment_slug

    assert is_managed_environment_slug("owner/env")
    assert is_freesolo_environment_id("owner/env")
    assert not is_managed_environment_slug("owner/env/extra")
    assert not is_managed_environment_slug("gsm8k")
    assert not is_managed_environment_slug("owner/env:tag")
    # github: / github.com forms are no longer environment ids (storage is Azure Blob now).
    assert not is_freesolo_environment_id("github:owner/repo@main:env/environment.py")
    assert not is_freesolo_environment_id("https://github.com/owner/repo")


def test_resolve_environment_reference_downloads_and_verifies(tmp_path, monkeypatch):
    import flash.envs.adapter as adapter

    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    data = _env_package_tar(
        {
            "environment.py": "def load_environment(**k):\n    return None\n",
            "datasets/train.jsonl": '{"x": 1}\n',
        }
    )
    sha = hashlib.sha256(data).hexdigest()
    downloads: list[str] = []

    def fake_download(url):
        downloads.append(url)
        return data

    monkeypatch.setattr(adapter, "_download_bytes", fake_download)

    resolved = adapter._resolve_environment_reference("owner/env", "https://blob/sas", sha)
    assert resolved.endswith("environment.py")
    assert Path(resolved).is_file()
    # A sidecar dataset is extracted alongside the entrypoint.
    assert (Path(resolved).parent / "datasets" / "train.jsonl").is_file()
    # Content-addressed cache: a second resolve does not re-download.
    again = adapter._resolve_environment_reference("owner/env", "https://blob/sas", sha)
    assert again == resolved
    assert downloads == ["https://blob/sas"]


def test_resolve_environment_reference_handles_single_wrapper_dir(tmp_path, monkeypatch):
    import flash.envs.adapter as adapter

    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    data = _github_environment_tarball("repo-root", env_path="environment.py")
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(adapter, "_download_bytes", lambda url: data)

    resolved = adapter._resolve_environment_reference("owner/env", "https://blob/sas", sha)
    assert resolved.endswith("environment.py")
    assert Path(resolved).is_file()


def test_resolve_environment_reference_sha_mismatch(tmp_path, monkeypatch):
    import flash.envs.adapter as adapter

    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    data = _env_package_tar({"environment.py": "x = 1\n"})
    monkeypatch.setattr(adapter, "_download_bytes", lambda url: data)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        adapter._resolve_environment_reference("owner/env", "https://blob/sas", "a" * 64)


def test_resolve_environment_reference_missing_entrypoint(tmp_path, monkeypatch):
    import flash.envs.adapter as adapter

    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    data = _env_package_tar({"helper.py": "x = 1\n"})
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(adapter, "_download_bytes", lambda url: data)

    with pytest.raises(FileNotFoundError, match=r"environment\.py"):
        adapter._resolve_environment_reference("owner/env", "https://blob/sas", sha)


def test_resolve_environment_reference_unresolved_slug_errors():
    import flash.envs.adapter as adapter

    # A managed slug with no SAS URL (not published / control-plane resolve failed) and no local
    # checkout -> a clear error. There is no GitHub fallback.
    with pytest.raises(RuntimeError, match="could not be resolved"):
        adapter._resolve_environment_reference("owner/env", None, None)


def test_extract_env_package_rejects_unbounded_and_traversal(monkeypatch, tmp_path):
    from flash.envs.adapter import _extract_env_package

    dest = tmp_path / "many"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.adapter._MAX_ARCHIVE_MEMBERS", 0)
    with pytest.raises(RuntimeError, match="too many members"):
        _extract_env_package(_env_package_tar({"environment.py": "x"}), dest)

    dest = tmp_path / "big"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.adapter._MAX_ARCHIVE_MEMBERS", 50)
    monkeypatch.setattr("flash.envs.adapter._MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(RuntimeError, match="too large"):
        _extract_env_package(_env_package_tar({"environment.py": "xxxxx"}), dest)

    dest = tmp_path / "trav"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.adapter._MAX_ARCHIVE_BYTES", 1000)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        payload = b"pwn"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="unsafe path"):
        _extract_env_package(buf.getvalue(), dest)


def test_install_manifest_and_worker_deps():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["FLASH_ENVS_MANIFEST"] = os.path.join(tmp, "envs.json")
        import flash.envs.registry as registry

        importlib.reload(registry)
        env_id = "owner/env"
        registry.record_installed_env(env_id, package="freesolo")
        assert registry.worker_pip_for_env(env_id) == ["freesolo"]
        assert registry.list_installed_environments() == [env_id]

        os.environ.pop("FLASH_ENVS_MANIFEST", None)
        importlib.reload(registry)
