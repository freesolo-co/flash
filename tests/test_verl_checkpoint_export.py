"""Tests for verl checkpoint export and merger supervision."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def actor_dir(tmp_path):
    actor = tmp_path / "global_step_7"
    (actor / "huggingface").mkdir(parents=True)
    (actor / "model_world_size_2_rank_0.pt").write_bytes(b"a" * 4096)
    (actor / "model_world_size_2_rank_1.pt").write_bytes(b"b" * 4096)
    return actor


def _disk_usage(free: int):
    return SimpleNamespace(total=1 << 40, used=0, free=free)


def test_model_shard_accounting_uses_only_top_level_model_shards(actor_dir):
    from flash.engine.worker.verl import checkpoints

    (actor_dir / "optim_world_size_2_rank_0.pt").write_bytes(b"o" * 65536)
    (actor_dir / "extra_state_world_size_2_rank_0.pt").write_bytes(b"e" * 2048)
    nested = actor_dir / "nested"
    nested.mkdir()
    (nested / "model_world_size_2_rank_0.pt").write_bytes(b"n" * 16384)

    assert checkpoints._model_shard_bytes(str(actor_dir)) == 8192


@pytest.mark.parametrize(("free", "raises"), [(8191, True), (8192, False)])
def test_merge_headroom_matches_shard_bytes(monkeypatch, actor_dir, tmp_path, free, raises):
    from flash.engine.worker.verl import checkpoints

    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(free))

    def call():
        checkpoints.require_merge_headroom(str(actor_dir), str(tmp_path / "adapter_merge"))

    if raises:
        with pytest.raises(checkpoints.MergeDiskHeadroomError) as caught:
            call()
        message = str(caught.value)
        assert "save_every" in message
        assert "save_at_steps" in message
        assert "disk_gb" not in message
    else:
        call()


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.ENOSPC, "no space left on device"),
        OSError(errno.EDQUOT, "disk quota exceeded"),
        RuntimeError("I/O error: Disk quota exceeded (os error 122)"),
        subprocess.CalledProcessError(
            1,
            ["merge"],
            output=b"I/O error: No space left on device (os error 28)",
        ),
    ],
)
def test_direct_disk_evidence_wins_even_when_the_volume_looks_roomy(
    monkeypatch, actor_dir, tmp_path, error
):
    from flash.engine.worker.verl import checkpoints

    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(1 << 40))
    with pytest.raises(checkpoints.MergeDiskExhaustedError) as caught:
        checkpoints.raise_for_merge_disk_exhaustion(
            error, str(actor_dir), str(tmp_path / "adapter_merge")
        )
    message = str(caught.value)
    assert "save_every" in message
    assert "save_at_steps" in message
    assert "disk_gb" not in message
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(("free", "disk_error"), [(0, True), (1 << 40, False)])
def test_free_space_is_only_a_fallback_for_non_disk_errors(
    monkeypatch, actor_dir, tmp_path, free, disk_error
):
    from flash.engine.worker.verl import checkpoints

    error = RuntimeError("unexpected pos 100 vs 88")
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(free))
    if disk_error:
        with pytest.raises(checkpoints.MergeDiskExhaustedError):
            checkpoints.raise_for_merge_disk_exhaustion(
                error, str(actor_dir), str(tmp_path / "adapter_merge")
            )
    else:
        assert (
            checkpoints.raise_for_merge_disk_exhaustion(
                error, str(actor_dir), str(tmp_path / "adapter_merge")
            )
            is None
        )


def test_layout_error_after_success_is_not_reclassified(monkeypatch, actor_dir, tmp_path):
    from flash.engine.worker.verl import checkpoints

    free = {"value": 1 << 40}

    def merge(cmd, env):
        wrong = tmp_path / "adapter_merge" / "wrong_adapter"
        wrong.mkdir(parents=True)
        (wrong / "adapter_config.json").write_text("{}")
        free["value"] = 0

    monkeypatch.setattr(checkpoints, "_run_merger", merge)
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(free["value"]))
    with pytest.raises(RuntimeError, match="did not produce a peft adapter") as caught:
        checkpoints.export_peft_adapter(
            str(actor_dir),
            str(tmp_path / "adapter"),
            base_model_id="org/model",
            python_bin="/verl/python",
        )
    assert not isinstance(caught.value, checkpoints.MergeDiskExhaustedError)


def test_placement_error_after_success_is_not_reclassified(monkeypatch, actor_dir, tmp_path):
    from flash.engine.worker.verl import checkpoints

    free = {"value": 1 << 40}
    denied = PermissionError(errno.EACCES, "permission denied")

    def merge(cmd, env):
        lora = tmp_path / "adapter_merge" / "lora_adapter"
        lora.mkdir(parents=True)
        (lora / "adapter_config.json").write_text("{}")
        free["value"] = 0

    monkeypatch.setattr(checkpoints, "_run_merger", merge)
    monkeypatch.setattr(checkpoints.os, "replace", lambda src, dst: (_ for _ in ()).throw(denied))
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(free["value"]))
    with pytest.raises(PermissionError) as caught:
        checkpoints.export_peft_adapter(
            str(actor_dir),
            str(tmp_path / "adapter"),
            base_model_id="org/model",
            python_bin="/verl/python",
        )
    assert caught.value is denied


def test_disk_classification_happens_before_merge_cleanup(monkeypatch, actor_dir, tmp_path):
    from flash.engine.worker.verl import checkpoints

    merge_out = tmp_path / "adapter_merge"
    sampled_with_evidence: list[bool] = []
    free = {"value": 1 << 40}

    def merge(cmd, env):
        merge_out.mkdir()
        (merge_out / "partial.safetensors").write_bytes(b"partial")
        free["value"] = 0
        raise RuntimeError("unexpected pos 100 vs 88")

    def disk_usage(path):
        sampled_with_evidence.append((merge_out / "partial.safetensors").exists())
        return _disk_usage(free["value"])

    monkeypatch.setattr(checkpoints, "_run_merger", merge)
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", disk_usage)
    with pytest.raises(checkpoints.MergeDiskExhaustedError):
        checkpoints.export_peft_adapter(
            str(actor_dir),
            str(tmp_path / "adapter"),
            base_model_id="org/model",
            python_bin="/verl/python",
        )
    assert sampled_with_evidence[-1] is True
    assert not merge_out.exists()


def test_export_moves_adapter_files_and_cleans_merge_tree(monkeypatch, actor_dir, tmp_path):
    from flash.engine.worker.verl import checkpoints

    merge_out = tmp_path / "adapter_merge"
    lora = merge_out / "lora_adapter"
    duplicated_at_cleanup: list[str] = []
    real_rmtree = checkpoints.shutil.rmtree

    def merge(cmd, env):
        lora.mkdir(parents=True)
        (lora / "adapter_config.json").write_text("{}")
        (lora / "adapter_model.safetensors").write_bytes(b"weights")

    def rmtree(path, *args, **kwargs):
        if str(path) == str(merge_out) and lora.is_dir():
            duplicated_at_cleanup.extend(os.listdir(lora))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(checkpoints, "_run_merger", merge)
    monkeypatch.setattr(checkpoints.shutil, "rmtree", rmtree)
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(1 << 40))
    adapter = tmp_path / "adapter"
    checkpoints.export_peft_adapter(
        str(actor_dir),
        str(adapter),
        base_model_id="org/model",
        python_bin="/verl/python",
    )
    assert (adapter / "adapter_model.safetensors").read_bytes() == b"weights"
    assert duplicated_at_cleanup == []
    assert not merge_out.exists()


def test_merger_preserves_invalid_output_first_disk_marker_and_exit_code(monkeypatch, capsys):
    from flash.engine.worker.verl import checkpoints

    child = (
        "import sys; "
        "sys.stdout.buffer.write(b'start\\n\\xffbad\\n'); "
        "sys.stdout.buffer.write(b'No space left on device first\\n'); "
        "sys.stdout.buffer.write(b'Disk quota exceeded second\\n'); "
        "sys.stdout.flush(); raise SystemExit(7)"
    )
    with pytest.raises(subprocess.CalledProcessError) as caught:
        checkpoints._run_merger([sys.executable, "-c", child], dict(os.environ))
    assert caught.value.returncode == 7
    assert caught.value.output == "No space left on device first"
    assert "�bad" in capsys.readouterr().out


@pytest.mark.wallclock
def test_merger_output_is_live(monkeypatch, tmp_path):
    from flash.engine.worker.verl import checkpoints

    gate = tmp_path / "continue"
    child = (
        "import pathlib,time\n"
        "print('merging shard 1', flush=True)\n"
        f"p = pathlib.Path({str(gate)!r})\n"
        "while not p.exists():\n"
        "    time.sleep(0.01)\n"
    )
    process: subprocess.Popen | None = None
    real_popen = checkpoints.subprocess.Popen
    seen_alive: list[bool] = []

    def popen(*args, **kwargs):
        nonlocal process
        process = real_popen(*args, **kwargs)
        return process

    def live_print(line, **kwargs):
        assert process is not None
        seen_alive.append(process.poll() is None)
        gate.write_text("go")

    monkeypatch.setattr(checkpoints.subprocess, "Popen", popen)
    monkeypatch.setattr(checkpoints, "print", live_print, raising=False)
    try:
        checkpoints._run_merger([sys.executable, "-c", child], dict(os.environ))
        assert seen_alive == [True]
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=1)


@pytest.mark.wallclock
def test_cancellation_kills_and_reaps_the_merger(monkeypatch):
    from flash.engine.worker.verl import checkpoints

    child = "import signal; print('started', flush=True); signal.pause()"
    process: subprocess.Popen | None = None
    real_popen = checkpoints.subprocess.Popen

    def popen(*args, **kwargs):
        nonlocal process
        process = real_popen(*args, **kwargs)
        return process

    def cancel(line, **kwargs):
        raise SystemExit(1)

    monkeypatch.setattr(checkpoints.subprocess, "Popen", popen)
    monkeypatch.setattr(checkpoints, "print", cancel, raising=False)
    try:
        with pytest.raises(SystemExit):
            checkpoints._run_merger([sys.executable, "-c", child], dict(os.environ))
        assert process is not None
        assert process.poll() is not None
        assert process.returncode is not None
        assert process.returncode < 0
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=1)


@pytest.mark.wallclock
def test_stdout_eof_before_child_exit_is_not_success(monkeypatch):
    from flash.engine.worker.verl import checkpoints

    child = "import os,signal; os.close(1); os.close(2); signal.pause()"
    process: subprocess.Popen | None = None
    real_popen = checkpoints.subprocess.Popen

    def tracking_popen(*args, **kwargs):
        nonlocal process
        process = real_popen(*args, **kwargs)
        return process

    monkeypatch.setattr(checkpoints.subprocess, "Popen", tracking_popen)
    monkeypatch.setattr(checkpoints, "_MERGER_KILL_REAP_SECONDS", 0.05)
    try:
        with pytest.raises(RuntimeError, match="did not exit"):
            checkpoints._run_merger([sys.executable, "-c", child], dict(os.environ))
        assert process is not None
        assert process.poll() is not None
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=1)
