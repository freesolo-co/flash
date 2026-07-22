from __future__ import annotations

import json
import subprocess
import sys

import pytest

from flash.engine.worker import openrlhf_common


class _FakeProcess:
    def __init__(self, lines: list[str], returncode: int = 0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.running = True
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float | None] = []

    def poll(self):
        return None if self.running else self.returncode

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        self.running = False
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.running = False


def _write_adapter(directory, *, config=None, weight_name="adapter_model.safetensors"):
    directory.mkdir(parents=True)
    directory.joinpath("adapter_config.json").write_text(json.dumps(config or {}))
    directory.joinpath(weight_name).write_bytes(b"weights")


def test_resolve_openrlhf_python_prefers_preset(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_OPENRLHF_PYTHON", "  /opt/openrlhf/bin/python  ")

    assert openrlhf_common.resolve_openrlhf_python(str(tmp_path)) == "/opt/openrlhf/bin/python"


def test_resolve_openrlhf_python_falls_back_to_current_interpreter(monkeypatch, tmp_path):
    monkeypatch.delenv("FLASH_OPENRLHF_PYTHON", raising=False)

    assert openrlhf_common.resolve_openrlhf_python(str(tmp_path)) == sys.executable


def test_run_openrlhf_training_builds_entrypoint_and_streams_callbacks(monkeypatch, capsys):
    process = _FakeProcess(["global_step: 3 loss=1.0\n", "done\n"], returncode=7)
    popen_calls = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return process

    monkeypatch.setattr(openrlhf_common.subprocess, "Popen", fake_popen)
    lines = []
    steps = []
    heartbeats = []
    env = {"FLASH_OPENRLHF_REWARD_URL": "http://127.0.0.1:1234/score"}

    returncode = openrlhf_common.run_openrlhf_training(
        "/opt/openrlhf/bin/python",
        ["--actor.model_name_or_path", "Qwen/Qwen3.5-0.8B"],
        env=env,
        cwd="/workspace/run",
        on_line=lines.append,
        on_step=steps.append,
        heartbeat=lambda: heartbeats.append(True),
        heartbeat_interval_s=0,
    )

    assert returncode == 7
    assert popen_calls == [
        (
            [
                "/opt/openrlhf/bin/python",
                "-m",
                "openrlhf.cli.train_ppo_ray",
                "--actor.model_name_or_path",
                "Qwen/Qwen3.5-0.8B",
            ],
            {
                "cwd": "/workspace/run",
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
            },
        )
    ]
    assert lines == ["global_step: 3 loss=1.0\n", "done\n"]
    assert steps == [3]
    assert heartbeats == [True, True]
    assert capsys.readouterr().out == "global_step: 3 loss=1.0\ndone\n"


def test_run_openrlhf_training_terminates_on_callback_failure(monkeypatch):
    process = _FakeProcess(["step: 1\n"])
    monkeypatch.setattr(openrlhf_common.subprocess, "Popen", lambda *args, **kwargs: process)

    def fail(_line):
        raise RuntimeError("upload failed")

    with pytest.raises(RuntimeError, match="upload failed"):
        openrlhf_common.run_openrlhf_training("python", [], env={}, on_line=fail)

    assert process.terminated
    assert process.wait_timeouts == [10]


def test_export_openrlhf_adapter_copies_safetensors_and_stamps_base(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "adapter"
    _write_adapter(
        checkpoint,
        config={"r": 16, "base_model_name_or_path": None, "revision": "abc123"},
    )

    openrlhf_common.export_openrlhf_adapter(
        str(checkpoint),
        str(output),
        "Qwen/Qwen3.5-0.8B",
        "/opt/openrlhf/bin/python",
    )

    assert sorted(path.name for path in output.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert output.joinpath("adapter_model.safetensors").read_bytes() == b"weights"
    config = json.loads(output.joinpath("adapter_config.json").read_text())
    assert config == {
        "r": 16,
        "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "revision": "abc123",
    }


def test_export_openrlhf_adapter_converts_zero3_peft_bin(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "adapter"
    _write_adapter(checkpoint, weight_name="adapter_model.bin")
    run_calls = []

    def fake_run(cmd, **kwargs):
        run_calls.append((cmd, kwargs))
        target = cmd[-1]
        with open(target, "wb") as output_file:
            output_file.write(b"safe-weights")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(openrlhf_common.subprocess, "run", fake_run)

    openrlhf_common.export_openrlhf_adapter(
        str(checkpoint), str(output), "Qwen/Qwen3.5-0.8B", "/openrlhf/python"
    )

    assert len(run_calls) == 1
    command, kwargs = run_calls[0]
    assert command[:3] == ["/openrlhf/python", "-c", openrlhf_common._BIN_TO_SAFETENSORS]
    assert command[-2:] == [
        str(checkpoint / "adapter_model.bin"),
        str(output / "adapter_model.safetensors"),
    ]
    assert kwargs == {"check": True}
    assert output.joinpath("adapter_model.safetensors").read_bytes() == b"safe-weights"
    assert not output.joinpath("adapter_model.bin").exists()


def test_export_openrlhf_adapter_resolves_matching_deepspeed_hf_export(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    actor_tag = checkpoint_root / "_actor" / "global_step2"
    actor_tag.mkdir(parents=True)
    hf_export = checkpoint_root / "global_step2_hf"
    output = tmp_path / "adapter"
    _write_adapter(hf_export)

    openrlhf_common.export_openrlhf_adapter(
        str(actor_tag), str(output), "Qwen/Qwen3.5-0.8B", "/openrlhf/python"
    )

    assert output.joinpath("adapter_model.safetensors").read_bytes() == b"weights"


def test_export_openrlhf_adapter_rejects_raw_deepspeed_checkpoint(tmp_path):
    actor_tag = tmp_path / "checkpoints" / "_actor" / "global_step2"
    actor_tag.mkdir(parents=True)
    actor_tag.joinpath("mp_rank_00_model_states.pt").write_bytes(b"state")

    with pytest.raises(RuntimeError, match=r"enable --ckpt\.save_hf"):
        openrlhf_common.export_openrlhf_adapter(
            str(actor_tag), str(tmp_path / "adapter"), "Qwen/Qwen3.5-0.8B", "/openrlhf/python"
        )


def test_stamp_adapter_dir_provenance_validates_and_stamps_revision(tmp_path):
    adapter = tmp_path / "adapter"
    _write_adapter(
        adapter,
        config={
            "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
            "revision": "abc123",
            "r": 8,
        },
    )

    openrlhf_common.stamp_adapter_dir_provenance(
        str(adapter), "Qwen/Qwen3.5-0.8B", "abc123"
    )

    config = json.loads(adapter.joinpath("adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "Qwen/Qwen3.5-0.8B"
    assert config["revision"] == "abc123"
    assert config["r"] == 8


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"base_model_name_or_path": "other/model"}, "does not match validated target"),
        (
            {"base_model_name_or_path": "Qwen/Qwen3.5-0.8B", "revision": "old"},
            "revision does not match",
        ),
    ],
)
def test_stamp_adapter_dir_provenance_rejects_mismatch(tmp_path, config, message):
    adapter = tmp_path / "adapter"
    _write_adapter(adapter, config=config)

    with pytest.raises(RuntimeError, match=message):
        openrlhf_common.stamp_adapter_dir_provenance(
            str(adapter), "Qwen/Qwen3.5-0.8B", "new"
        )
