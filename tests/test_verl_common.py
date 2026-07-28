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


def _fake_verl_venv(tmp_path, *, stamp: str | None):
    """materialize a verl-venv as an earlier attempt would have left it on the pod."""
    venv = tmp_path / "verl-venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python").write_text("")
    if stamp is not None:
        (venv / "flash-verl-requirement").write_text(stamp)
    return venv


def _record_run(calls, *, keep_check: bool = False):
    """stand in for subprocess.run, creating the venv dir `uv venv` would have created."""

    def fake_run(command, check):
        calls.append((command, check) if keep_check else command)
        if command[:2] == ["uv", "venv"]:
            os.makedirs(os.path.join(command[2], "bin"), exist_ok=True)

    return fake_run


def test_resolve_verl_python_installs_pinned_gpu_dependencies(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    python_bin = vc.resolve_verl_python(str(tmp_path))

    assert python_bin.endswith("/verl-venv/bin/python")
    assert calls[0][:2] == ["uv", "venv"]
    install = calls[1]
    assert vc.VERL_REQUIREMENT == (
        "verl @ git+https://github.com/freesolo-co/verl@0f821c22325a1a51384431d57b899cc5dcf3d837"
    )
    assert vc.VERL_REQUIREMENT in install
    assert "liger-kernel" in install
    assert "bitsandbytes>=0.49" in install
    assert "qwen-vl-utils" in install
    assert "torchvision" in install
    assert "xgrammar==0.1.25" in install
    assert "tqdm" in install
    assert "pyarrow" in install
    assert len(calls) == 2
    # the stamp is written only after a successful install, so a crashed install is never reused.
    stamp = tmp_path / "verl-venv" / "flash-verl-requirement"
    assert stamp.read_text() == vc.VERL_REQUIREMENT


def test_resolve_verl_python_reuses_a_venv_built_from_the_current_pin(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    _fake_verl_venv(tmp_path, stamp=vc.VERL_REQUIREMENT)

    vc.resolve_verl_python(str(tmp_path))

    # reinstalling verl's torch/vllm on every retry would cost many minutes of paid gpu time.
    assert calls == []


@pytest.mark.parametrize("stale", ["verl @ git+https://github.com/freesolo-co/verl@" + "0" * 40, None])
def test_resolve_verl_python_rebuilds_a_venv_that_is_not_the_current_pin(monkeypatch, tmp_path, stale):
    # a retry reuses the pod workdir. a venv from an earlier pin (or a partial install, stamp=None)
    # must be rebuilt rather than silently training on the wrong verl.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    venv = _fake_verl_venv(tmp_path, stamp=stale)
    (venv / "marker").write_text("from the previous attempt")

    vc.resolve_verl_python(str(tmp_path))

    assert [c[:2] for c in calls] == [["uv", "venv"], ["uv", "pip"]]
    assert not (venv / "marker").exists()


def test_verl_pin_is_an_immutable_commit_on_the_freesolo_fork():
    # a branch or tag would let the runtime move under a pinned flash release.
    _, _, ref = vc.VERL_REQUIREMENT.partition("git+")
    url, _, commit = ref.rpartition("@")
    assert url == "https://github.com/freesolo-co/verl"
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_verl_pin_matches_the_version_opd_requires_exactly():
    # the pin MUST stay on the verl 0.8.0 base. opd_verl_plugin patches 0.8.0 internals and imports
    # verl.trainer.main_ppo_sync, which verl deleted after 0.8.0, so a pin built on a newer base
    # installs a verl that fails opd's exact-version gate and cannot import its own entrypoint.
    from flash.engine.worker import opd_verl_plugin as plugin

    assert plugin._STRUCTURED_RUNTIME_EXACT_VERSIONS["verl"] == "0.8.0"
    # asserting the constant alone would let a newer-base commit land silently, so bind the pinned
    # commit itself to that version. this is the sha of the truncation-mask commit cherry-picked onto
    # the v0.8.0 tag; moving the pin must be a deliberate edit here, with the base re-verified.
    _, _, ref = vc.VERL_REQUIREMENT.partition("git+")
    _, _, commit = ref.rpartition("@")
    assert commit == "0f821c22325a1a51384431d57b899cc5dcf3d837"


def test_resolve_verl_python_installs_wandb_best_effort_when_requested(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls, keep_check=True))

    vc.resolve_verl_python(str(tmp_path), install_wandb=True)

    assert vc.VERL_REQUIREMENT in calls[1][0]
    assert calls[2] == (
        ["uv", "pip", "install", "--python", str(tmp_path / "verl-venv/bin/python"), "wandb"],
        False,
    )


def _probe_interpreter(tmp_path, name, body):
    """write a stub interpreter that answers the RolloutConfig probe like a real python would."""
    stub = tmp_path / name
    stub.write_text("#!/bin/sh\n" + body + "\n")
    stub.chmod(0o755)
    return str(stub)


def test_verl_supports_rollout_field_true_when_field_declared(tmp_path):
    fork = _probe_interpreter(tmp_path, "fork-python", "echo 1")
    assert vc.verl_supports_rollout_field(fork, "mask_truncated_completions") is True


def test_verl_supports_rollout_field_false_when_field_absent(tmp_path):
    # stock verl 0.8.0: RolloutConfig has no such field, so the probe prints 0.
    stock = _probe_interpreter(tmp_path, "stock-python", "echo 0")
    assert vc.verl_supports_rollout_field(stock, "mask_truncated_completions") is False


def test_verl_supports_rollout_field_false_when_verl_missing(tmp_path):
    # import error inside the probe: nonzero exit must read as unsupported, not crash the caller.
    broken = _probe_interpreter(tmp_path, "broken-python", "exit 1")
    assert vc.verl_supports_rollout_field(broken, "mask_truncated_completions") is False


def test_verl_supports_rollout_field_false_when_interpreter_missing(tmp_path):
    # a bogus FLASH_VERL_PYTHON must not raise OSError out of the capability check.
    missing = str(tmp_path / "does-not-exist")
    assert vc.verl_supports_rollout_field(missing, "mask_truncated_completions") is False


def test_resolve_verl_python_returns_preset_unmodified(monkeypatch, tmp_path):
    # flash does not own a preset interpreter and must never mutate it; capability is checked
    # separately by verl_supports_rollout_field.
    calls = []
    monkeypatch.setenv("FLASH_VERL_PYTHON", "/opt/verl/bin/python")
    monkeypatch.setattr(vc.subprocess, "run", lambda *a, **k: calls.append(a))

    assert vc.resolve_verl_python(str(tmp_path)) == "/opt/verl/bin/python"
    assert calls == []


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
