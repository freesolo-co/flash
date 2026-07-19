from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest


def _deadline_fields() -> dict[str, float]:
    now = time.time()
    return {
        "run_created_at": now,
        "run_max_wall_seconds": 3600.0,
        "deadline_at": now + 3600.0,
    }


def _input_data(*, code_prefix: str, extra_pip: list[str] | None = None) -> dict:
    return {
        "phase": "sft",
        "seed": 0,
        "hf_repo": "owner/runs",
        "job_spec_json": json.dumps({"algorithm": "sft", "run_id": "flash-test-run"}),
        "env": {"ATTEMPT": "7", "HF_TOKEN": "tok", "PYTHONPATH": "base-pythonpath"},
        "extra_pip": list(extra_pip or []),
        "code_prefix": code_prefix,
        **_deadline_fields(),
    }


def _remove_worker_artifacts() -> None:
    for path in (
        "/tmp/console_sft.txt",
        "/tmp/console_sft.txt.tail",
        "/tmp/job_spec.json",
        "/tmp/metrics.json",
        "/tmp/train_meta.json",
    ):
        Path(path).unlink(missing_ok=True)


def test_train_body_creates_unique_venvs_and_cleans_stale_state(monkeypatch, tmp_path):
    from flash.providers.runpod.train import endpoints

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    venv_root = tmp_path / "flash-runpod-job-venvs"
    stale_venv = venv_root / "previous-job"
    stale_venv.mkdir(parents=True)
    (stale_venv / "stale-package.txt").write_text("old", encoding="utf-8")

    created: list[Path] = []

    def fake_run(cmd, *, check, env=None):
        assert check is True
        assert env is None
        assert cmd[:4] == [sys.executable, "-m", "venv", "--system-site-packages"]
        path = Path(cmd[4])
        created.append(path)
        (path / "bin").mkdir(parents=True)

    monkeypatch.setattr(subprocess, "run", fake_run)

    for _ in range(2):
        with pytest.raises(ValueError, match="invalid code_prefix"):
            endpoints._train_body(_input_data(code_prefix="../code/flash"))

    assert len(created) == 2
    assert created[0] != created[1]
    assert all(path.parent == venv_root for path in created)
    assert all(path.name.startswith("flash-test-run-a7-") for path in created)
    assert not stale_venv.exists()
    assert all(not path.exists() for path in created)


def _run_successful_job(monkeypatch, tmp_path, *, extra_pip: list[str]):
    import huggingface_hub

    from flash.providers.runpod.train import endpoints

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    code_prefix = "code/0123456789abcdef0123456789abcdef/flash"
    run_calls: list[dict] = []
    launch_calls: list[dict] = []

    def fake_run(cmd, *, check, env=None):
        assert check is True
        call = {"cmd": list(cmd), "env": env}
        run_calls.append(call)
        if cmd[1:4] == ["-m", "venv", "--system-site-packages"]:
            venv_path = Path(cmd[4])
            (venv_path / "bin").mkdir(parents=True)
            (venv_path / "bin" / "python").touch()
            return
        assert cmd[1:4] == ["-m", "pip", "install"]

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_tree(self, **kwargs):
            return [types.SimpleNamespace(path=f"{code_prefix}/__init__.py", size=1)]

        def upload_file(self, **kwargs):
            return None

    def fake_download(*, filename, local_dir, **kwargs):
        return os.path.join(local_dir, filename)

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            launch_calls.append({"cmd": list(cmd), **kwargs})
            self.stdout = iter(["worker completed\n"])
            self.returncode = 0

        def wait(self):
            Path("/tmp/metrics.json").write_text('{"score": 1.0}', encoding="utf-8")
            return 0

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    _remove_worker_artifacts()
    try:
        result = endpoints._train_body(
            _input_data(code_prefix=code_prefix, extra_pip=extra_pip)
        )
    finally:
        _remove_worker_artifacts()

    return result, run_calls, launch_calls, code_prefix


def test_extra_pip_and_worker_use_per_job_venv_python(monkeypatch, tmp_path):
    packages = ["env-package==1.2.3", "other-package==4.5.6"]
    result, run_calls, launch_calls, code_prefix = _run_successful_job(
        monkeypatch,
        tmp_path,
        extra_pip=packages,
    )

    assert result == {"score": 1.0}
    assert len(run_calls) == 2
    venv_call, pip_call = run_calls
    venv_path = Path(venv_call["cmd"][4])
    venv_python = str(venv_path / "bin" / "python")
    assert venv_call["cmd"] == [
        sys.executable,
        "-m",
        "venv",
        "--system-site-packages",
        str(venv_path),
    ]
    assert pip_call["cmd"] == [venv_python, "-m", "pip", "install", *packages]
    assert pip_call["cmd"][0] != sys.executable

    assert len(launch_calls) == 1
    launch = launch_calls[0]
    assert launch["cmd"] == [venv_python, "-m", "flash.engine.worker_entrypoint"]
    assert launch["cwd"] == f"/runcode/{os.path.dirname(code_prefix)}"
    assert launch["env"]["RUN_MODE"] == "sft"
    assert launch["env"]["HF_TOKEN"] == "tok"
    assert launch["env"]["PYTHONPATH"] == (
        f"/runcode/{os.path.dirname(code_prefix)}{os.pathsep}base-pythonpath"
    )
    assert not venv_path.exists()


def test_empty_extra_pip_keeps_cold_job_behavior_in_isolated_venv(monkeypatch, tmp_path):
    result, run_calls, launch_calls, _ = _run_successful_job(
        monkeypatch,
        tmp_path,
        extra_pip=[],
    )

    assert result == {"score": 1.0}
    assert len(run_calls) == 1
    venv_path = Path(run_calls[0]["cmd"][4])
    assert run_calls[0]["cmd"][1:4] == ["-m", "venv", "--system-site-packages"]
    assert launch_calls[0]["cmd"][0] == str(venv_path / "bin" / "python")
    assert not venv_path.exists()


def test_deadline_exit_removes_job_venv_before_hard_exit(monkeypatch, tmp_path):
    from flash.providers.runpod.train import endpoints

    class DeadlineExit(Exception):
        pass

    timers = []
    created: list[Path] = []

    class FakeTimer:
        def __init__(self, interval, callback):
            self.interval = interval
            self.callback = callback
            self.daemon = False
            self.cancelled = False
            timers.append(self)

        def start(self):
            return None

        def cancel(self):
            self.cancelled = True

    def fake_exit(code):
        assert code == 124
        raise DeadlineExit

    def fake_run(cmd, *, check, env=None):
        assert cmd[1:4] == ["-m", "venv", "--system-site-packages"]
        path = Path(cmd[4])
        created.append(path)
        (path / "bin").mkdir(parents=True)
        timers[0].callback()

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(threading, "Timer", FakeTimer)
    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DeadlineExit):
        endpoints._train_body(_input_data(code_prefix="../code/flash"))

    assert len(created) == 1
    assert not created[0].exists()
    assert timers[0].cancelled is True
