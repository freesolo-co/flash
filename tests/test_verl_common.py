"""CPU unit tests for the shared verl subprocess harness (verl_common)."""

from __future__ import annotations

import json
import os

import pytest

from flash.engine.worker import verl_common as vc


def test_latest_global_step_dir_picks_highest(tmp_path):
    for step in (1, 5, 20, 3):
        os.makedirs(tmp_path / f"global_step_{step}" / "actor", exist_ok=True)
    # a non-checkpoint dir must be ignored, not crash the scan.
    os.makedirs(tmp_path / "not_a_step", exist_ok=True)
    actor, step = vc.latest_global_step_dir(str(tmp_path))
    assert step == 20
    assert actor == os.path.join(str(tmp_path), "global_step_20", "actor")


def test_latest_global_step_dir_raises_when_empty(tmp_path):
    with pytest.raises(RuntimeError, match="no global_step_N checkpoint"):
        vc.latest_global_step_dir(str(tmp_path))


def test_stamp_adapter_dir_provenance_sets_base_and_revision(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": None, "r": 16}))
    vc.stamp_adapter_dir_provenance(str(tmp_path), "org/model", "deadbeef")
    out = json.loads(cfg.read_text())
    assert out["base_model_name_or_path"] == "org/model"
    assert out["revision"] == "deadbeef"
    # untouched fields survive the stamp.
    assert out["r"] == 16


def test_stamp_adapter_dir_provenance_rejects_base_mismatch(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": "org/other"}))
    with pytest.raises(RuntimeError, match="does not match validated target"):
        vc.stamp_adapter_dir_provenance(str(tmp_path), "org/model")


def test_resolve_verl_python_prefers_preset(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_VERL_PYTHON", "/opt/verl/bin/python")
    assert vc.resolve_verl_python(str(tmp_path)) == "/opt/verl/bin/python"


def test_resolve_verl_python_installs_pinned_gpu_dependencies(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", lambda command, check: calls.append(command))

    python_bin = vc.resolve_verl_python(str(tmp_path))

    assert python_bin.endswith("/verl-venv/bin/python")
    assert calls[0][:2] == ["uv", "venv"]
    install = calls[1]
    assert "verl==0.8.0" in install
    assert "liger-kernel" in install
    assert "bitsandbytes>=0.49" in install


def test_run_verl_training_streams_steps_and_returns_code():
    seen: list[int] = []
    lines: list[str] = []
    beats: list[int] = []
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'foo step: 1 bar'; echo 'step: 2'; echo done"],
        env=dict(os.environ),
        on_step=seen.append,
        on_line=lines.append,
        heartbeat=lambda: beats.append(1),
        heartbeat_interval_s=0.0,
    )
    assert code == 0
    assert seen == [1, 2]
    assert lines[-1] == "done\n"
    # heartbeat_interval_s=0 => fires on every scanned line (3 lines here).
    assert len(beats) >= 1


def test_run_verl_training_propagates_nonzero_exit():
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'step: 1'; exit 7"],
        env=dict(os.environ),
        on_step=lambda _s: None,
    )
    assert code == 7


def test_run_verl_training_terminates_child_when_callback_fails():
    def fail(_line):
        raise RuntimeError("checkpoint upload failed")

    with pytest.raises(RuntimeError, match="checkpoint upload failed"):
        vc.run_verl_training(
            ["bash", "-c", "echo ready; sleep 30"],
            env=dict(os.environ),
            on_line=fail,
        )
