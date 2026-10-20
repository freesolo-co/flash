"""A staged checkpoint cannot become successful before CPU publication."""

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
    job = spec()
    job = replace(job, train=replace(job.train, max_steps=2))
    monkeypatch.setenv("FLASH_MODAL_CPU_UPLOADS", "1")
    monkeypatch.setattr(cpu_handoff, "MOUNT", str(root))
    monkeypatch.setattr(state, "JOB_SPEC", job)
    monkeypatch.setattr(state, "RUN_ID", job.run_id)
    monkeypatch.setattr(state, "ATTEMPT", 3)
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "model.pt").write_bytes(b"native model data")
    return root, source


def test_only_final_step_is_staged_and_copy_survives_source_removal(staged):
    root, source = staged
    assert not cpu_handoff.stage_final_checkpoint(1, str(source))
    assert not (root / "handoff.json").exists()
    assert cpu_handoff.stage_final_checkpoint(2, str(source))
    (source / "model.pt").unlink()
    assert (root / "checkpoint/model.pt").read_bytes() == b"native model data"
    assert (root / "source/flash/engine/worker/io/cpu_handoff.py").is_file()


def test_handoff_rejects_links_before_copy(staged):
    root, source = staged
    (source / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlinks"):
        cpu_handoff.stage_final_checkpoint(2, str(source))
    assert not (root / "handoff.json").exists()


def test_cpu_failure_cannot_publish_done_or_success(staged, monkeypatch):
    _root, source = staged
    cpu_handoff.stage_final_checkpoint(2, str(source))
    monkeypatch.setattr(hf, "upload_resume_checkpoint", lambda *a, **k: False)
    monkeypatch.setattr(hf, "hf_upload_file", lambda *a, **k: pytest.fail("premature completion"))
    with pytest.raises(RuntimeError, match="publication failed"):
        cpu_handoff.main()


def test_cpu_commit_precedes_metrics_done_and_attested_success(staged, monkeypatch):
    root, source = staged
    cpu_handoff.stage_final_checkpoint(2, str(source))
    (root / "metrics.json").write_text('{"notes":{}}')
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
    cpu_handoff.stage_final_checkpoint(2, str(source))
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


def test_cpu_mode_requires_explicit_last_step(monkeypatch):
    monkeypatch.setenv("FLASH_MODAL_CPU_UPLOADS", "1")
    job = spec()
    job = replace(job, train=replace(job.train, max_steps=None))
    with pytest.raises(ValueError, match="explicit"):
        checkpoint_upload.enabled(job)
