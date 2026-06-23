"""Tests for the Freesolo SDK environment adapter + install manifest."""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
import tarfile
import tempfile
import types
from dataclasses import dataclass, field
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
            "task": "2+2?",
            "expected_output": "4",
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
            metadata={"task": example.task},
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


def _install_fake_freesolo(monkeypatch, *, sdk_env=None, seen=None):
    sdk_env = sdk_env or _FakeSingleTurnEnv()
    seen = seen if seen is not None else {}

    def task_example_from_record(record):
        return _TaskExample(
            record=dict(record),
            task=str(record.get("task") or record.get("prompt") or ""),
            task_id=record.get("id"),
            expected_output=record.get("expected_output"),
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
    dataset.write_text('{"id":"row-1","task":"2+2?","expected_output":"4"}\n')

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
    assert train == [{"id": "row-1", "task": "2+2?", "expected_output": "4"}]
    assert env.prompt_messages(train[0]) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "2+2?"},
    ]
    assert env.reward("the answer is 4", train[0]) == 1.0
    assert env.grade("the answer is 4", train[0]) is True
    assert env.reward("nope", train[0]) == 0.0
    assert env.scores_breakdown("the answer is 4", train[0]) == {"match": 1.0, "total": 1.0}
    assert env.sft_target({"expected_output": "4"}) == "4"


def test_freesolo_adapter_uses_env_dataset_when_no_source(monkeypatch):
    _install_fake_freesolo(monkeypatch)

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _FakeSingleTurnEnv(),
        "github:owner/repo@main:env/environment.py",
        source=None,
        contract_text="",
    )
    assert env.dataset()[0]["expected_output"] == "4"


def test_freesolo_multiturn_hooks(monkeypatch):
    _install_fake_freesolo(monkeypatch, sdk_env=_FakeMultiTurnEnv())

    from flash.envs.adapter import FreesoloEnvironment

    env = FreesoloEnvironment(
        _FakeMultiTurnEnv(),
        "github:owner/repo@main:env/environment.py",
        source=[{"task": "browse", "expected_output": "done"}],
        contract_text="contract",
    )
    state = env.new_rollout_state({"task": "browse", "expected_output": "done"})
    assert state["messages"] == [{"role": "user", "content": "contract:browse"}]
    env.record_model_turn(state, "click")
    replies = env.env_reply(state["messages"], state)
    assert replies == [{"role": "user", "content": "observed click"}]
    assert state["done"] is True
    assert env.rollout_done(state) is True
    assert env.reward("ignored", {"task": "browse", "expected_output": "done"}, state) == 0.5
    assert env.grade("ignored", {"task": "browse", "expected_output": "done"}, state) is True
    assert (
        env.reward_from_messages(
            [{"role": "assistant", "content": "final"}],
            {"task": "browse", "expected_output": "done"},
            [{"role": "user", "content": "contract:browse"}],
        )
        == 0.5
    )


def test_github_environment_ref_parsing():
    from flash.envs.adapter import is_github_environment_ref

    assert is_github_environment_ref("github:owner/repo@dev:envs/e/environment.py")
    assert is_github_environment_ref("github:owner/repo")
    assert not is_github_environment_ref("github:owner/repo@main:/etc/passwd")
    assert not is_github_environment_ref("github:owner/repo/extra@main:envs/e/environment.py")
    assert not is_github_environment_ref("github:owner/repo@:envs/e/environment.py")
    assert is_github_environment_ref("https://github.com/owner/repo/blob/dev/envs/e/environment.py")
    assert is_github_environment_ref("https://github.com/owner/repo")
    assert not is_github_environment_ref("owner/env")
    assert not is_github_environment_ref("gsm8k")
    assert not is_github_environment_ref("github:owner/repo@main:../../etc/passwd")
    assert not is_github_environment_ref("https://github.com/owner/repo/blob/dev/../../etc/passwd")
    assert not is_github_environment_ref("https://github.com/owner/repo/blob/main:/etc/passwd")
    assert not is_github_environment_ref("https://github.com/owner/repo/issues/1")


def test_github_environment_resolves_by_commit_sha(tmp_path, monkeypatch):
    import flash.envs.adapter as adapter

    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(
        adapter,
        "_resolve_ref_sha",
        lambda parsed: "b" * 40,
    )

    downloads: list[str] = []

    def fake_download(ref):
        downloads.append(ref.ref)
        return _github_environment_tarball("repo-root")

    monkeypatch.setattr(adapter, "_download_github_tarball", fake_download)

    resolved = adapter._resolve_environment_reference(
        "github:owner/repo@main:envs/e/environment.py"
    )
    assert resolved.endswith("envs/e/environment.py")
    assert downloads == ["b" * 40]


def test_safe_extract_archive_rejects_unbounded_members_and_size(monkeypatch, tmp_path):
    from flash.envs.adapter import _safe_extract_archive

    def make_members_tar(members: list[tuple[str, bytes | None]]):
        tar = io.BytesIO()
        with tarfile.open(fileobj=tar, mode="w:gz") as handle:
            for name, payload in members:
                info = tarfile.TarInfo(name)
                if payload is None:
                    info.type = tarfile.DIRTYPE
                    info.size = 0
                    handle.addfile(info)
                else:
                    info.size = len(payload)
                    handle.addfile(info, io.BytesIO(payload))
        return tar.getvalue()

    dest = tmp_path / "extract_many"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.adapter._MAX_ARCHIVE_MEMBERS", 0)
    with pytest.raises(RuntimeError, match="too many members"):
        _safe_extract_archive(make_members_tar([("a", b"")]), dest)

    dest = tmp_path / "extract_big"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.adapter._MAX_ARCHIVE_MEMBERS", 5)
    monkeypatch.setattr("flash.envs.adapter._MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(RuntimeError, match="too large"):
        _safe_extract_archive(
            make_members_tar([("repo-root/", None), ("repo-root/keep.txt", b"xx")]), dest
        )

    dest = tmp_path / "extract_pax"
    dest.mkdir()
    monkeypatch.setattr("flash.envs.adapter._MAX_ARCHIVE_BYTES", 100)
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w:gz") as handle:
        pax = tarfile.TarInfo("pax_global_header")
        pax.type = tarfile.XGLTYPE
        handle.addfile(pax)
        root = tarfile.TarInfo("repo-root/")
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        handle.addfile(root)
        file_info = tarfile.TarInfo("repo-root/environment.py")
        payload = b"def load_environment(**k):\n    return None\n"
        file_info.size = len(payload)
        handle.addfile(file_info, io.BytesIO(payload))
    assert _safe_extract_archive(tar.getvalue(), dest) == dest / "repo-root"


def test_install_manifest_and_worker_deps():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["FLASH_ENVS_MANIFEST"] = os.path.join(tmp, "envs.json")
        import flash.envs.registry as registry

        importlib.reload(registry)
        env_id = "github:owner/repo@main:env/environment.py"
        registry.record_installed_env(env_id, package="freesolo")
        assert registry.worker_pip_for_env(env_id) == ["freesolo"]
        assert registry.worker_hub_env_ids(env_id) == []
        assert registry.list_installed_environments() == [env_id]

        os.environ.pop("FLASH_ENVS_MANIFEST", None)
        importlib.reload(registry)
