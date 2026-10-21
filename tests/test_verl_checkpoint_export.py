"""Tests for verl checkpoint export and merger supervision."""

from __future__ import annotations

import ctypes
import errno
import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_checkpoint_export_saves_complete_processor_sidecars(monkeypatch, tmp_path):
    from flash.engine.worker.train.sft.setup import checkpoints

    actor_dir = tmp_path / "actor"
    actor_dir.mkdir()
    adapter_dir = tmp_path / "adapter"

    def export(_actor, target, **_kwargs):
        os.makedirs(target)
        (adapter_dir / "adapter_config.json").write_text("{}")

    class Processor:
        def save_pretrained(self, target):
            (adapter_dir / "preprocessor_config.json").write_text(
                '{"image_processor_type":"QwenVLImageProcessor"}'
            )
            (adapter_dir / "processor_config.json").write_text(
                '{"processor_class":"QwenVLProcessor"}'
            )

    monkeypatch.setattr(checkpoints, "export_peft_adapter", export)
    monkeypatch.setattr(checkpoints, "stamp_adapter_dir_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(checkpoints._worker_hf, "write_base_model_provenance", lambda *args: None)

    checkpoints._export_checkpoint_adapter(
        str(actor_dir),
        str(adapter_dir),
        model_id="org/model",
        model_revision="commit",
        python_bin="/verl/python",
        preprocessor=Processor(),
    )

    assert (adapter_dir / "preprocessor_config.json").read_text() == (
        '{"image_processor_type":"QwenVLImageProcessor"}'
    )
    assert (adapter_dir / "processor_config.json").read_text() == (
        '{"processor_class":"QwenVLProcessor"}'
    )


def test_multimodal_checkpoint_export_passes_explicit_modality_marker(monkeypatch, tmp_path):
    from flash.engine.worker.train.sft.setup import checkpoints
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    actor_dir = tmp_path / "actor"
    actor_dir.mkdir()
    adapter_dir = tmp_path / "adapter"

    def export(_actor, target, **_kwargs):
        os.makedirs(target)
        (adapter_dir / "adapter_config.json").write_text("{}")

    captured = {}

    def stamp(*args, **kwargs):
        inspect.signature(stamp_adapter_dir_provenance).bind(*args, **kwargs)
        captured.update(kwargs)

    monkeypatch.setattr(checkpoints, "export_peft_adapter", export)
    monkeypatch.setattr(checkpoints, "stamp_adapter_dir_provenance", stamp)
    monkeypatch.setattr(checkpoints._worker_hf, "write_base_model_provenance", lambda *args: None)

    checkpoints._export_checkpoint_adapter(
        str(actor_dir),
        str(adapter_dir),
        model_id="org/model",
        model_revision="commit",
        python_bin="/verl/python",
        exclude_modules=None,
    )

    assert "exclude_modules" in captured
    assert captured["exclude_modules"] is None


def test_text_checkpoint_export_does_not_invent_processor_sidecars(monkeypatch, tmp_path):
    from flash.engine.worker.train.sft.setup import checkpoints

    actor_dir = tmp_path / "actor"
    actor_dir.mkdir()
    adapter_dir = tmp_path / "adapter"

    def export(_actor, target, **_kwargs):
        os.makedirs(target)
        (adapter_dir / "adapter_config.json").write_text("{}")

    monkeypatch.setattr(checkpoints, "export_peft_adapter", export)
    monkeypatch.setattr(checkpoints, "stamp_adapter_dir_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(checkpoints._worker_hf, "write_base_model_provenance", lambda *args: None)

    checkpoints._export_checkpoint_adapter(
        str(actor_dir),
        str(adapter_dir),
        model_id="org/model",
        model_revision="commit",
        python_bin="/verl/python",
    )

    assert sorted(path.name for path in adapter_dir.iterdir()) == ["adapter_config.json"]


def _has_libc() -> bool:
    try:
        ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return False
    return True


_needs_process_teardown = pytest.mark.skipif(
    not hasattr(os, "fork") or not os.path.isdir("/proc") or not _has_libc(),
    reason="teardown tests drive real process groups: needs os.fork, /proc and libc",
)


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


def test_chained_disk_evidence_wins_over_an_outer_nondisk_oserror(monkeypatch, actor_dir, tmp_path):
    from flash.engine.worker.verl import checkpoints

    cause = OSError(errno.ENOSPC, "no space left on device")
    error = PermissionError(errno.EACCES, "permission denied")
    error.__cause__ = cause
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(1 << 40))

    with pytest.raises(checkpoints.MergeDiskExhaustedError) as caught:
        checkpoints.raise_for_merge_disk_exhaustion(
            error, str(actor_dir), str(tmp_path / "adapter_merge")
        )
    assert caught.value.__cause__ is error


def _watcher_boundary(kind: str):
    if kind == "sft":
        from flash.engine.worker.train.sft.setup.checkpoints import _VerlCheckpointWatcher

        return (
            object.__new__(_VerlCheckpointWatcher),
            "raise_if_failed",
            "verl checkpoint watcher failed",
        )
    from flash.engine.worker.train.rl.launch.checkpoints import _VerlResumeUploader

    return object.__new__(_VerlResumeUploader), "raise_if_incomplete", "verl resume uploader failed"


@pytest.mark.parametrize("kind", ["sft", "rl"])
@pytest.mark.parametrize("error_name", ["MergeDiskHeadroomError", "MergeDiskExhaustedError"])
def test_watcher_boundaries_preserve_merge_disk_errors(kind, error_name):
    from flash.engine.worker.verl import checkpoints

    watcher, method_name, _ = _watcher_boundary(kind)
    error_type = getattr(checkpoints, error_name)
    error = error_type("specific disk diagnosis")
    watcher._error = error

    with pytest.raises(error_type, match="specific disk diagnosis") as caught:
        getattr(watcher, method_name)()
    assert caught.value is error


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_watcher_boundaries_still_wrap_unrelated_errors(kind):
    watcher, method_name, wrapper = _watcher_boundary(kind)
    error = ValueError("unrelated failure")
    watcher._error = error

    with pytest.raises(RuntimeError, match=wrapper) as caught:
        getattr(watcher, method_name)()
    assert caught.value.__cause__ is error


def test_stderr_disk_marker_is_not_hidden_by_benign_stdout(monkeypatch, actor_dir, tmp_path):
    from flash.engine.worker.verl import checkpoints

    error = subprocess.CalledProcessError(
        1,
        ["merge"],
        output="saving tokenizer config\n",
        stderr="Disk quota exceeded (os error 122)\n",
    )
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(1 << 40))

    with pytest.raises(checkpoints.MergeDiskExhaustedError):
        checkpoints.raise_for_merge_disk_exhaustion(
            error, str(actor_dir), str(tmp_path / "adapter_merge")
        )


@pytest.mark.parametrize(("free", "disk_error"), [(0, True), (1 << 40, False)])
def test_free_space_only_confirms_a_recognizable_short_write(
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


def test_low_space_does_not_relabel_an_unrelated_child_failure(monkeypatch, actor_dir, tmp_path):
    from flash.engine.worker.verl import checkpoints

    error = subprocess.CalledProcessError(
        1, ["merge"], output="full model saved\n", stderr="tokenizer config save failed\n"
    )
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(0))

    assert (
        checkpoints.raise_for_merge_disk_exhaustion(
            error, str(actor_dir), str(tmp_path / "adapter_merge")
        )
        is None
    )


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(errno.ENOENT, "missing"),
        PermissionError(errno.EACCES, "unexpected pos 100 vs 88"),
    ],
)
def test_low_space_preserves_non_disk_merger_launch_errors(monkeypatch, actor_dir, tmp_path, error):
    from flash.engine.worker.verl import checkpoints

    free = {"value": 1 << 40}

    def launch(cmd, env):
        free["value"] = 0
        raise error

    monkeypatch.setattr(checkpoints, "_run_merger", launch)
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(free["value"]))

    with pytest.raises(type(error)) as caught:
        checkpoints.export_peft_adapter(
            str(actor_dir),
            str(tmp_path / "adapter"),
            base_model_id="org/model",
            python_bin="/verl/python",
        )
    assert caught.value is error


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


def test_merger_uses_the_shared_streaming_supervisor(monkeypatch):
    from flash.engine.worker.train.entry import backend_common
    from flash.engine.worker.verl import checkpoints

    seen = {}

    def supervise(cmd, *, env, on_line, errors):
        seen.update(cmd=cmd, env=env, errors=errors)
        on_line("merged\n")
        return 0

    monkeypatch.setattr(backend_common, "_run_streaming_verl_subprocess", supervise)
    checkpoints._run_merger(["merge", "checkpoint"], {"KEEP": "yes"})

    assert seen == {
        "cmd": ["merge", "checkpoint"],
        "env": {"KEEP": "yes", "PYTHONUNBUFFERED": "1"},
        "errors": "replace",
    }


def test_short_write_marker_survives_the_merger_subprocess_boundary(
    monkeypatch, actor_dir, tmp_path
):
    from flash.engine.worker.train.entry import backend_common
    from flash.engine.worker.verl import checkpoints

    free = {"value": 1 << 40}

    def supervise(cmd, *, env, on_line, errors):
        on_line("unexpected pos 221967808 vs 221967696\n")
        free["value"] = 0
        return 7

    monkeypatch.setattr(backend_common, "_run_streaming_verl_subprocess", supervise)
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(free["value"]))

    with pytest.raises(checkpoints.MergeDiskExhaustedError) as caught:
        checkpoints.export_peft_adapter(
            str(actor_dir),
            str(tmp_path / "adapter"),
            base_model_id="org/model",
            python_bin="/verl/python",
        )
    cause = caught.value.__cause__
    assert isinstance(cause, subprocess.CalledProcessError)
    assert cause.output == "unexpected pos 221967808 vs 221967696"


def test_direct_disk_marker_replaces_an_earlier_short_write_marker(
    monkeypatch, actor_dir, tmp_path
):
    from flash.engine.worker.train.entry import backend_common
    from flash.engine.worker.verl import checkpoints

    def supervise(cmd, *, env, on_line, errors):
        on_line("unexpected pos 221967808 vs 221967696\n")
        on_line("Disk quota exceeded (os error 122)\n")
        return 7

    monkeypatch.setattr(backend_common, "_run_streaming_verl_subprocess", supervise)
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda path: _disk_usage(1 << 40))

    with pytest.raises(checkpoints.MergeDiskExhaustedError) as caught:
        checkpoints.export_peft_adapter(
            str(actor_dir),
            str(tmp_path / "adapter"),
            base_model_id="org/model",
            python_bin="/verl/python",
        )
    cause = caught.value.__cause__
    assert isinstance(cause, subprocess.CalledProcessError)
    assert cause.output == "Disk quota exceeded (os error 122)"


def test_merger_preserves_cancellation_identity(monkeypatch):
    from flash.engine.worker.train.entry import backend_common
    from flash.engine.worker.verl import checkpoints

    cancellation = SystemExit("cancelled")

    def cancel(*args, **kwargs):
        raise cancellation

    monkeypatch.setattr(backend_common, "_run_streaming_verl_subprocess", cancel)
    with pytest.raises(SystemExit) as caught:
        checkpoints._run_merger(["merge"], {})
    assert caught.value is cancellation


@_needs_process_teardown
def test_merger_preserves_invalid_output_first_disk_marker_and_exit_code(capsys):
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
@_needs_process_teardown
def test_merger_output_is_live_without_explicit_child_flush(monkeypatch, tmp_path):
    from flash.engine.worker.train.entry import backend_common
    from flash.engine.worker.verl import checkpoints

    gate = tmp_path / "continue"
    child = (
        "import pathlib,time\n"
        "print('merging shard 1')\n"
        f"p = pathlib.Path({str(gate)!r})\n"
        "deadline = time.monotonic() + 1.0\n"
        "while not p.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "raise SystemExit(0 if p.exists() else 9)\n"
    )
    process: subprocess.Popen | None = None
    real_popen = backend_common.subprocess.Popen
    seen_alive: list[bool] = []

    def popen(*args, **kwargs):
        nonlocal process
        process = real_popen(*args, **kwargs)
        return process

    def live_print(line, **kwargs):
        assert process is not None
        seen_alive.append(process.poll() is None)
        gate.write_text("go")

    monkeypatch.setattr(backend_common.subprocess, "Popen", popen)
    monkeypatch.setattr(checkpoints, "print", live_print, raising=False)
    checkpoints._run_merger([sys.executable, "-c", child], dict(os.environ))
    assert seen_alive == [True]
