from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from flash.engine.worker import openrlhf_common


def _resolve_tensor_python() -> str | None:
    candidates = (sys.executable, "/usr/bin/python3", shutil.which("python3"))
    seen = set()
    for candidate in candidates:
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "-c", "import torch, safetensors"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            continue
        if result.returncode == 0:
            return candidate
    return None


_TENSOR_PYTHON = _resolve_tensor_python()
requires_torch = pytest.mark.skipif(
    _TENSOR_PYTHON is None,
    reason="no interpreter with torch+safetensors (offline CI)",
)


class _FakeProcess:
    def __init__(self, lines: list[str], returncode: int = 0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.running = True
        self.wait_timeouts: list[float | None] = []

    def poll(self):
        return None if self.running else self.returncode

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        self.running = False
        return self.returncode


def _write_adapter(directory, *, config=None, weight_name="adapter_model.safetensors"):
    directory.mkdir(parents=True)
    directory.joinpath("adapter_config.json").write_text(json.dumps(config or {}))
    directory.joinpath(weight_name).write_bytes(b"weights")


def _module_env(module_dir):
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(module_dir), existing]))
    return env


def _write_process_tree_module(module_dir, *, ignore_term: bool = False):
    module_dir.joinpath("process_tree_entry.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import signal",
                "import subprocess",
                "import sys",
                "import time",
                "",
                f"ignore_term = {ignore_term!r}",
                "if ignore_term:",
                "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "child_code = 'import signal, time'",
                "if ignore_term:",
                "    child_code += '; signal.signal(signal.SIGTERM, signal.SIG_IGN)'",
                "child_code += '; time.sleep(60)'",
                "child = subprocess.Popen([sys.executable, '-c', child_code])",
                "with open(sys.argv[1], 'w') as pid_file:",
                "    json.dump([os.getpid(), child.pid], pid_file)",
                "print('ready', flush=True)",
                "while True:",
                "    time.sleep(1)",
            ]
        )
    )


def _pid_is_running(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as stat_file:
            stat = stat_file.read().split()
    except FileNotFoundError:
        return False
    return len(stat) > 2 and stat[2] != "Z"


def _assert_processes_stopped(pids: list[int]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(_pid_is_running(pid) for pid in pids):
        time.sleep(0.05)
    assert not any(_pid_is_running(pid) for pid in pids)


def _create_tensor_weights(path, values: list[float], *, safetensors: bool = False):
    if safetensors:
        body = "from safetensors.torch import save_file; save_file({'weight': tensor}, sys.argv[1])"
    else:
        body = "torch.save({'weight': tensor}, sys.argv[1])"
    subprocess.run(
        [
            _TENSOR_PYTHON,
            "-c",
            f"import sys, torch; tensor = torch.tensor({values!r}); {body}",
            str(path),
        ],
        check=True,
    )


def _assert_safetensors_values(path, values: list[float]):
    subprocess.run(
        [
            _TENSOR_PYTHON,
            "-c",
            (
                "import sys, torch; from safetensors.torch import load_file; "
                f"expected = torch.tensor({values!r}); "
                "actual = load_file(sys.argv[1])['weight']; "
                "assert torch.equal(actual, expected), (actual, expected)"
            ),
            str(path),
        ],
        check=True,
    )


def test_resolve_openrlhf_python_prefers_preset(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_OPENRLHF_PYTHON", "  /opt/openrlhf/bin/python  ")

    assert openrlhf_common.resolve_openrlhf_python(str(tmp_path)) == "/opt/openrlhf/bin/python"


def test_resolve_openrlhf_python_falls_back_to_current_interpreter(monkeypatch, tmp_path):
    monkeypatch.delenv("FLASH_OPENRLHF_PYTHON", raising=False)

    assert openrlhf_common.resolve_openrlhf_python(str(tmp_path)) == sys.executable


def test_run_openrlhf_training_builds_explicit_entrypoint_and_streams_callbacks(
    monkeypatch, capsys
):
    process = _FakeProcess(["global_step: 3 loss=1.0\n", "done\n"], returncode=7)
    popen_calls = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return process

    monkeypatch.setattr(openrlhf_common.subprocess, "Popen", fake_popen)
    lines = []
    steps = []
    env = {"FLASH_OPENRLHF_REWARD_URL": "http://127.0.0.1:1234/score"}

    returncode = openrlhf_common.run_openrlhf_training(
        "/opt/openrlhf/bin/python",
        ["--actor.model_name_or_path", "Qwen/Qwen3.5-0.8B"],
        env=env,
        entrypoint="openrlhf.cli.train_sft",
        cwd="/workspace/run",
        on_line=lines.append,
        on_step=steps.append,
    )

    assert returncode == 7
    assert popen_calls == [
        (
            [
                "/opt/openrlhf/bin/python",
                "-u",
                "-m",
                "openrlhf.cli.train_sft",
                "--actor.model_name_or_path",
                "Qwen/Qwen3.5-0.8B",
            ],
            {
                "cwd": "/workspace/run",
                "env": {**env, "PYTHONUNBUFFERED": "1"},
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
                "start_new_session": True,
            },
        )
    ]
    assert lines == ["global_step: 3 loss=1.0\n", "done\n"]
    assert steps == [3]
    assert capsys.readouterr().out == "global_step: 3 loss=1.0\ndone\n"


def test_run_openrlhf_training_defaults_to_ppo_entrypoint(monkeypatch):
    process = _FakeProcess([])
    commands = []

    def fake_popen(cmd, **_kwargs):
        commands.append(cmd)
        return process

    monkeypatch.setattr(openrlhf_common.subprocess, "Popen", fake_popen)

    assert openrlhf_common.run_openrlhf_training("python", [], env={}) == 0
    assert commands == [["python", "-u", "-m", "openrlhf.cli.train_ppo_ray"]]


def test_run_openrlhf_training_propagates_real_entrypoint_exit_code(tmp_path):
    tmp_path.joinpath("exit_entry.py").write_text(
        "import sys\nraise SystemExit(int(sys.argv[1]))\n"
    )

    returncode = openrlhf_common.run_openrlhf_training(
        sys.executable,
        ["17"],
        env=_module_env(tmp_path),
        entrypoint="exit_entry",
    )

    assert returncode == 17


def test_run_openrlhf_training_callback_failure_kills_real_process_group(tmp_path):
    pid_file = tmp_path / "pids.json"
    _write_process_tree_module(tmp_path)

    def fail(_line):
        raise RuntimeError("upload failed")

    with pytest.raises(RuntimeError, match="upload failed"):
        openrlhf_common.run_openrlhf_training(
            sys.executable,
            [str(pid_file)],
            env=_module_env(tmp_path),
            entrypoint="process_tree_entry",
            on_line=fail,
        )

    pids = json.loads(pid_file.read_text())
    assert len(pids) == 2
    _assert_processes_stopped(pids)


def test_run_openrlhf_training_escalates_process_group_kill_after_timeout(
    monkeypatch, tmp_path
):
    pid_file = tmp_path / "pids.json"
    _write_process_tree_module(tmp_path, ignore_term=True)
    monkeypatch.setattr(openrlhf_common, "_PROCESS_GROUP_TERM_TIMEOUT_S", 0.15)
    sent_signals = []
    real_killpg = os.killpg

    def recording_killpg(process_group_id, sent_signal):
        sent_signals.append(sent_signal)
        real_killpg(process_group_id, sent_signal)

    monkeypatch.setattr(openrlhf_common.os, "killpg", recording_killpg)

    with pytest.raises(RuntimeError, match="upload failed"):
        openrlhf_common.run_openrlhf_training(
            sys.executable,
            [str(pid_file)],
            env=_module_env(tmp_path),
            entrypoint="process_tree_entry",
            on_line=lambda _line: (_ for _ in ()).throw(RuntimeError("upload failed")),
        )

    assert signal.SIGTERM in sent_signals
    assert signal.SIGKILL in sent_signals
    _assert_processes_stopped(json.loads(pid_file.read_text()))


def test_run_openrlhf_training_heartbeat_aborts_silent_child(tmp_path):
    pid_file = tmp_path / "pid.txt"
    tmp_path.joinpath("silent_entry.py").write_text(
        "import os, sys, time\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        openrlhf_common.run_openrlhf_training(
            sys.executable,
            [str(pid_file)],
            env=_module_env(tmp_path),
            entrypoint="silent_entry",
            heartbeat=lambda: (_ for _ in ()).throw(RuntimeError("heartbeat failed")),
            heartbeat_interval_s=0.05,
        )

    assert time.monotonic() - started < 2
    _assert_processes_stopped([int(pid_file.read_text())])


def test_export_openrlhf_adapter_matching_revision_copies_and_stamps_base(tmp_path):
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
        "abc123",
        _TENSOR_PYTHON,
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


def test_export_openrlhf_adapter_requires_validated_expected_revision(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_adapter(checkpoint, config={"revision": None})

    with pytest.raises(ValueError, match="requires a validated base model revision"):
        openrlhf_common.export_openrlhf_adapter(
            str(checkpoint),
            str(tmp_path / "adapter"),
            "Qwen/Qwen3.5-0.8B",
            "",
            _TENSOR_PYTHON,
        )


def test_export_openrlhf_adapter_rejects_mismatched_revision(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_adapter(
        checkpoint,
        config={"base_model_name_or_path": "Qwen/Qwen3.5-0.8B", "revision": "old"},
    )

    with pytest.raises(RuntimeError, match="revision does not match"):
        openrlhf_common.export_openrlhf_adapter(
            str(checkpoint),
            str(tmp_path / "adapter"),
            "Qwen/Qwen3.5-0.8B",
            "new",
            _TENSOR_PYTHON,
        )


def test_export_openrlhf_adapter_stamps_expected_revision_when_export_has_null(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "adapter"
    _write_adapter(
        checkpoint,
        config={"base_model_name_or_path": "Qwen/Qwen3.5-0.8B", "revision": None},
    )

    openrlhf_common.export_openrlhf_adapter(
        str(checkpoint),
        str(output),
        "Qwen/Qwen3.5-0.8B",
        "abc123",
        _TENSOR_PYTHON,
    )

    config = json.loads(output.joinpath("adapter_config.json").read_text())
    assert config["revision"] == "abc123"


def test_export_openrlhf_adapter_accepts_matching_immutable_snapshot_path(tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "adapter"
    snapshot = tmp_path / "hub" / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    _write_adapter(
        checkpoint,
        config={"base_model_name_or_path": str(snapshot), "revision": None},
    )

    openrlhf_common.export_openrlhf_adapter(
        str(checkpoint),
        str(output),
        "Qwen/Qwen3.5-0.8B",
        revision,
        _TENSOR_PYTHON,
    )

    config = json.loads(output.joinpath("adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "Qwen/Qwen3.5-0.8B"
    assert config["revision"] == revision


@requires_torch
def test_export_openrlhf_adapter_converts_real_zero3_peft_bin(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "adapter"
    checkpoint.mkdir()
    checkpoint.joinpath("adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": None, "revision": "abc123"})
    )
    _create_tensor_weights(checkpoint / "adapter_model.bin", [1.5, -2.0])

    openrlhf_common.export_openrlhf_adapter(
        str(checkpoint),
        str(output),
        "Qwen/Qwen3.5-0.8B",
        "abc123",
        _TENSOR_PYTHON,
    )

    _assert_safetensors_values(output / "adapter_model.safetensors", [1.5, -2.0])
    assert not output.joinpath("adapter_model.bin").exists()


@requires_torch
def test_export_openrlhf_adapter_prefers_authoritative_bin_when_both_weights_exist(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "adapter"
    checkpoint.mkdir()
    checkpoint.joinpath("adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": None, "revision": "abc123"})
    )
    _create_tensor_weights(checkpoint / "adapter_model.bin", [1.0])
    _create_tensor_weights(checkpoint / "adapter_model.safetensors", [99.0], safetensors=True)

    openrlhf_common.export_openrlhf_adapter(
        str(checkpoint),
        str(output),
        "Qwen/Qwen3.5-0.8B",
        "abc123",
        _TENSOR_PYTHON,
    )

    _assert_safetensors_values(output / "adapter_model.safetensors", [1.0])


def test_export_openrlhf_adapter_resolves_matching_deepspeed_hf_export(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    actor_tag = checkpoint_root / "_actor" / "global_step2"
    actor_tag.mkdir(parents=True)
    hf_export = checkpoint_root / "global_step2_hf"
    output = tmp_path / "adapter"
    _write_adapter(hf_export, config={"revision": "abc123"})

    openrlhf_common.export_openrlhf_adapter(
        str(actor_tag),
        str(output),
        "Qwen/Qwen3.5-0.8B",
        "abc123",
        _TENSOR_PYTHON,
    )

    assert output.joinpath("adapter_model.safetensors").read_bytes() == b"weights"


def test_export_openrlhf_adapter_resolves_checkpoint_root_latest(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    actor_dir = checkpoint_root / "_actor"
    actor_dir.mkdir(parents=True)
    actor_dir.joinpath("latest").write_text("global_step8\n")
    hf_export = checkpoint_root / "global_step8_hf"
    output = tmp_path / "adapter"
    _write_adapter(hf_export, config={"revision": "abc123"})

    openrlhf_common.export_openrlhf_adapter(
        str(checkpoint_root),
        str(output),
        "Qwen/Qwen3.5-0.8B",
        "abc123",
        _TENSOR_PYTHON,
    )

    assert output.joinpath("adapter_model.safetensors").read_bytes() == b"weights"


def test_export_openrlhf_adapter_rejects_raw_deepspeed_checkpoint(tmp_path):
    actor_tag = tmp_path / "checkpoints" / "_actor" / "global_step2"
    actor_tag.mkdir(parents=True)
    actor_tag.joinpath("mp_rank_00_model_states.pt").write_bytes(b"state")

    with pytest.raises(RuntimeError, match=r"enable --ckpt\.save_hf"):
        openrlhf_common.export_openrlhf_adapter(
            str(actor_tag),
            str(tmp_path / "adapter"),
            "Qwen/Qwen3.5-0.8B",
            "abc123",
            _TENSOR_PYTHON,
        )


def test_export_openrlhf_adapter_rejects_full_model_hf_export(tmp_path):
    checkpoint = tmp_path / "full-model"
    checkpoint.mkdir()
    checkpoint.joinpath("config.json").write_text("{}")
    checkpoint.joinpath("model.safetensors").write_bytes(b"full-model")

    with pytest.raises(RuntimeError, match="full-model HF export, not a PEFT adapter"):
        openrlhf_common.export_openrlhf_adapter(
            str(checkpoint),
            str(tmp_path / "adapter"),
            "Qwen/Qwen3.5-0.8B",
            "abc123",
            _TENSOR_PYTHON,
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
        (
            {
                "base_model_name_or_path": "/cache/models--Qwen--Qwen3.5-0.8B/snapshots/old",
                "revision": None,
            },
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
