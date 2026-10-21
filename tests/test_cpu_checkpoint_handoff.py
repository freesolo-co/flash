"""A staged checkpoint cannot become successful before CPU publication."""

import hashlib
import json
from dataclasses import replace

import pytest

from flash.engine.worker.io import cpu_handoff, hf
from flash.engine.worker.runtime import state
from flash.providers.modal import checkpoint_upload
from flash.providers.modal.handles import ModalJobHandle
from tests._helpers.modal import modal_remote
from tests.test_modal_training_lifecycle import spec


@pytest.fixture
def staged(tmp_path, monkeypatch):
    root = tmp_path / "volume"
    root.mkdir()
    (root / "metrics.json").write_text('{"step":2,"notes":{}}')
    job = spec()
    job = replace(job, train=replace(job.train, max_steps=2))
    monkeypatch.setattr(cpu_handoff, "MOUNT", str(root))
    monkeypatch.setattr(state, "JOB_SPEC", job)
    monkeypatch.setenv(
        "FLASH_EXPECTED_SPEC_SHA256", hashlib.sha256(job.to_json().encode()).hexdigest()
    )
    monkeypatch.setattr(state, "RUN_ID", job.run_id)
    monkeypatch.setattr(state, "ATTEMPT", 3)
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "model.pt").write_bytes(b"native model data")
    return root, source


def test_only_final_step_is_staged_and_copy_survives_source_removal(staged):
    root, source = staged
    assert not cpu_handoff.stage_final_checkpoint(1, str(source), final_step=2)
    assert not (root / "handoff.json").exists()
    assert cpu_handoff.stage_final_checkpoint(2, str(source), final_step=2)
    (source / "model.pt").unlink()
    assert (root / "checkpoint/model.pt").read_bytes() == b"native model data"
    assert (root / "source/flash/engine/worker/io/cpu_handoff.py").is_file()


def test_handoff_rejects_links_before_copy(staged):
    root, source = staged
    (source / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlinks"):
        cpu_handoff.stage_final_checkpoint(2, str(source), final_step=2)
    assert not (root / "handoff.json").exists()


def test_cpu_failure_cannot_publish_done_or_success(staged, monkeypatch):
    _root, source = staged
    cpu_handoff.stage_final_checkpoint(2, str(source), final_step=2)
    monkeypatch.setattr(hf, "upload_resume_checkpoint", lambda *a, **k: False)
    monkeypatch.setattr(hf, "hf_upload_file", lambda *a, **k: pytest.fail("premature completion"))
    with pytest.raises(RuntimeError, match="publication failed"):
        cpu_handoff.main()


def test_cpu_commit_precedes_metrics_done_and_attested_success(staged, monkeypatch):
    root, source = staged
    cpu_handoff.stage_final_checkpoint(2, str(source), final_step=2)
    (root / "metrics.json").write_text('{"step":2,"notes":{}}')
    (root / "DONE").write_text("123")
    (root / "terminal.json").write_text('{"ok":true}')
    events = []
    monkeypatch.setattr(
        hf, "upload_resume_checkpoint", lambda *a, **k: events.append("native") or True
    )
    monkeypatch.setattr(hf, "hf_upload_file", lambda path, name, **k: events.append(name))
    cpu_handoff.main()
    assert events == [
        "native",
        "cpu_upload_attempt3.json",
        "metrics.json",
        "DONE",
        "modal_attempt3.json",
    ]


def test_wrong_attempt_never_uploads(staged, monkeypatch):
    root, source = staged
    cpu_handoff.stage_final_checkpoint(2, str(source), final_step=2)
    manifest = json.loads((root / "handoff.json").read_text())
    manifest["attempt"] = 4
    (root / "handoff.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        hf, "upload_resume_checkpoint", lambda *a, **k: pytest.fail("foreign attempt")
    )
    with pytest.raises(ValueError, match="different attempt"):
        cpu_handoff.main()


def test_old_handles_default_off_and_new_handles_round_trip():
    old = modal_remote()
    old.pop("cpu_uploads", None)
    assert not ModalJobHandle.from_dict(old).cpu_uploads
    assert ModalJobHandle.from_dict(old).to_dict() == old
    new = {**old, "cpu_uploads": True}
    assert ModalJobHandle.from_dict(new).to_dict()["cpu_uploads"] is True
    with pytest.raises(ValueError, match="cpu_uploads"):
        ModalJobHandle.from_dict({**old, "cpu_uploads": "true"})


def test_volume_identity_includes_owner_and_attempt():
    assert checkpoint_upload.volume_name("owner", "run-a0") != checkpoint_upload.volume_name(
        "owner", "run-a1"
    )
    assert checkpoint_upload.volume_name("owner", "run-a0") != checkpoint_upload.volume_name(
        "other", "run-a0"
    )


def test_epoch_based_training_uses_resolved_final_step(staged, monkeypatch):
    root, source = staged
    job = replace(state.JOB_SPEC, train=replace(state.JOB_SPEC.train, max_steps=None))
    monkeypatch.setattr(state, "JOB_SPEC", job)
    assert cpu_handoff.stage_final_checkpoint(2, str(source), final_step=2)
    assert json.loads((root / "handoff.json").read_text())["step"] == 2


def test_cpu_upload_does_not_restage_the_mounted_checkpoint(staged):
    _root, source = staged
    assert not cpu_handoff.stage_final_checkpoint(2, str(source), final_step=None)


def test_cpu_rejects_a_checkpoint_from_before_the_completed_final_step(staged, monkeypatch):
    root, source = staged
    cpu_handoff.stage_final_checkpoint(2, str(source), final_step=2)
    (root / "metrics.json").write_text('{"step":3,"notes":{}}')
    monkeypatch.setattr(
        hf, "upload_resume_checkpoint", lambda *a, **k: pytest.fail("wrong final step")
    )
    with pytest.raises(ValueError, match="different attempt"):
        cpu_handoff.main()


def test_bootstrap_withholds_attested_marker_for_handoff(tmp_path, monkeypatch):
    from flash.providers._lifecycle.bootstrapping import bootstrap

    root = tmp_path / "volume"
    root.mkdir()
    (root / "handoff.json").write_text("{}")
    marker = tmp_path / "marker.json"
    marker.write_text('{"ok":true}')
    published = []
    monkeypatch.setattr(bootstrap, "CHECKPOINT_HANDOFF_ROOT", str(root))
    monkeypatch.setattr(
        bootstrap, "hf_upload", lambda payload, path, name, **kw: published.append(name)
    )
    payload = {"attempt": 2, "flash_arm": "modal", "handoff_volume": "owned-volume"}
    # Initial publication and replay both go through this gate.
    bootstrap._publish_terminal_marker(payload, str(marker), True)
    bootstrap._publish_terminal_marker(payload, str(marker), True)
    assert published == ["modal_gpu_attempt2.json"] * 2
    assert (root / "terminal.json").read_bytes() == marker.read_bytes()
    bootstrap._publish_terminal_marker(payload, str(marker), False)
    assert published[-1] == "modal_attempt2.json"
