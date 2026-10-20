"""Preparation overlaps independent work but starts no reward service on partial failure."""

import threading

import pytest

from flash.engine.worker.train.entry import rl_train_runner as runner


def test_file_staging_and_probe_overlap(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)
    monkeypatch.setattr(runner, "rl_workdir", lambda: str(tmp_path))

    def files(*args):
        barrier.wait(timeout=5)
        return {"files": True}

    def probe():
        barrier.wait(timeout=5)
        return "capabilities"

    monkeypatch.setattr(runner, "_prepare_rl_files", files)
    monkeypatch.setattr(runner, "_write_rl_shim", lambda *a: None)
    monkeypatch.setattr(runner, "_start_reward_runtime", lambda *a: "reward")
    assert runner._prepare_rl_runtime({}, None, None, [], capability_probe=probe) == (
        {"files": True},
        "reward",
        "capabilities",
    )


@pytest.mark.parametrize("failing", ["files", "probe"])
def test_partial_failure_does_not_start_reward_service(monkeypatch, tmp_path, failing):
    monkeypatch.setattr(runner, "rl_workdir", lambda: str(tmp_path))

    def prepare(name):
        if name == failing:
            raise RuntimeError(name)
        return {}

    monkeypatch.setattr(runner, "_prepare_rl_files", lambda *a: prepare("files"))
    monkeypatch.setattr(runner, "_write_rl_shim", lambda *a: None)
    monkeypatch.setattr(runner, "_start_reward_runtime", lambda *a: pytest.fail("premature server"))
    with pytest.raises(RuntimeError, match=failing):
        runner._prepare_rl_runtime({}, None, None, [], capability_probe=lambda: prepare("probe"))
