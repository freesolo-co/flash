"""Deployable RL checkpoints (CPU-only).

Covers the three control-plane-adjacent pieces of the feature:
  - the worker publishing each save's LoRA adapter to a stable per-step path,
  - the lister that enumerates those snapshots from HF,
  - the backend client that mirrors them to the freesolo run_checkpoints store.

All HF/network boundaries are stubbed; nothing here touches a GPU or the network.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from flash.runner.checkpoints import (
    CheckpointListingError,
    checkpoint_adapter_prefix,
    final_adapter_exists,
    list_checkpoints,
)
from flash.spec import JobSpec

SPEC_DICT = {
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {"steps": 1, "hf_repo": "org/test-runs"},
    "gpu": {"type": "RTX 5090"},
    "run_id": "flash-ckpt-1",
}


def _spec() -> JobSpec:
    return JobSpec.from_dict(SPEC_DICT)


# --------------------------------------------------------------------------------------------
# Worker: publish_deployable_checkpoint
# --------------------------------------------------------------------------------------------
class _RecordingHfApi:
    def __init__(self, files: list[str] | None = None):
        self.uploads: list[dict] = []
        self.deleted: list[str] = []
        self._files = files or []

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)

    def list_repo_files(self, repo_id, repo_type):
        return self._files

    def delete_folder(self, path_in_repo, repo_id, repo_type):
        self.deleted.append(path_in_repo)


def _prime_worker(monkeypatch, recorder, *, repo="org/test-runs", phase="rl", run="flash-ckpt-1"):
    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "HF_REPO", repo)
    monkeypatch.setattr(worker, "PHASE", phase)
    monkeypatch.setattr(worker, "RUN_ID", run)
    monkeypatch.setattr(worker, "SEED", 0)
    monkeypatch.setattr(worker, "hf_api", lambda: recorder)
    # heartbeat would otherwise commit to HF; silence it for the unit test.
    monkeypatch.setattr(worker, "heartbeat", lambda *a, **k: None)
    return worker


def test_publish_deployable_checkpoint_uploads_adapter_only(tmp_path, monkeypatch):
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
    (ckpt / "optimizer.pt").write_bytes(b"opt")

    subfolder = worker.publish_deployable_checkpoint(str(ckpt), 80)

    assert subfolder == "rl/flash-ckpt-1/checkpoints/step-80/adapter"
    assert len(rec.uploads) == 1
    up = rec.uploads[0]
    assert up["path_in_repo"] == "rl/flash-ckpt-1/checkpoints/step-80/adapter"
    assert up["repo_type"] == "dataset"
    # Trainer-state files are excluded so each per-step snapshot is just the small LoRA adapter.
    assert "optimizer.pt" in up["ignore_patterns"]
    # The deployable path must NOT prune older steps (every step stays deployable).
    assert "delete_patterns" not in up


def test_publish_deployable_checkpoint_accepts_legacy_bin_weights(tmp_path, monkeypatch):
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.bin").write_bytes(b"weights")

    subfolder = worker.publish_deployable_checkpoint(str(ckpt), 80)

    assert subfolder == "rl/flash-ckpt-1/checkpoints/step-80/adapter"
    assert len(rec.uploads) == 1


def test_publish_deployable_checkpoint_skips_without_adapter(tmp_path, monkeypatch):
    """A checkpoint that carries no PEFT adapter is never advertised as deployable."""
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "optimizer.pt").write_bytes(b"opt")  # no adapter_config.json

    assert worker.publish_deployable_checkpoint(str(ckpt), 10) is None
    assert rec.uploads == []


def test_publish_deployable_checkpoint_skips_config_without_weights(tmp_path, monkeypatch):
    """A checkpoint with adapter_config.json but no weights file isn't loadable -> not published."""
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-20"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")  # config only, no adapter_model.*

    assert worker.publish_deployable_checkpoint(str(ckpt), 20) is None
    assert rec.uploads == []


def test_publish_deployable_checkpoint_no_repo_is_noop(tmp_path, monkeypatch):
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, repo="")  # local run, no HF repo
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")

    assert worker.publish_deployable_checkpoint(str(ckpt), 5) is None
    assert rec.uploads == []


# --------------------------------------------------------------------------------------------
# Worker: make_checkpoint_upload_callback — durable flush of the final deployable checkpoint
# --------------------------------------------------------------------------------------------
def _make_ckpt_dir(parent, step):
    ckpt = parent / f"checkpoint-{step}"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
    (ckpt / "optimizer.pt").write_bytes(b"opt")
    return ckpt


@pytest.fixture
def fake_trainer_callback(monkeypatch):
    """make_checkpoint_upload_callback does ``from transformers import TrainerCallback`` at call
    time; the offline CI env has no transformers, so stub a minimal base class (we invoke our
    own on_save/on_train_end overrides directly, so the base is just a placeholder)."""
    import sys
    import types

    mod = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    mod.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return mod


def test_latest_checkpoint_dir_picks_highest_step(tmp_path):
    import flash.engine.worker as worker

    _make_ckpt_dir(tmp_path, 4)
    _make_ckpt_dir(tmp_path, 12)
    _make_ckpt_dir(tmp_path, 8)
    (tmp_path / "checkpoint-notanumber").mkdir()  # ignored
    (tmp_path / "checkpoint-99").write_text("a file, not a dir")  # ignored
    step, path = worker._latest_checkpoint_dir(str(tmp_path))
    assert step == 12
    assert path.endswith("checkpoint-12")
    assert worker._latest_checkpoint_dir(str(tmp_path / "missing")) is None


def test_on_train_end_flushes_final_deployable_checkpoint(
    tmp_path, monkeypatch, fake_trainer_callback
):
    """A fast RL run can exit before its last save's async (daemon) upload finishes, so
    on_train_end must SYNCHRONOUSLY publish the latest on-disk checkpoint as a deployable
    snapshot — otherwise `flash checkpoints` is empty even though the run trained fine."""
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    out = tmp_path / "out"
    out.mkdir()
    _make_ckpt_dir(out, 8)  # last save the trainer wrote locally; its async upload was "lost"
    cb = worker.make_checkpoint_upload_callback()

    # No on_save uploads recorded (simulating their daemon threads being killed at exit).
    cb.on_train_end(SimpleNamespace(output_dir=str(out)), None, None)

    deployable = [
        u for u in rec.uploads if u["path_in_repo"].endswith("checkpoints/step-8/adapter")
    ]
    assert len(deployable) == 1, "on_train_end must publish the final deployable checkpoint"
    assert "optimizer.pt" in deployable[0]["ignore_patterns"]


def test_on_train_end_publishes_final_deployable_when_lock_is_held(
    tmp_path, monkeypatch, fake_trainer_callback
):
    """Regression: a slow on_save upload can still hold the upload lock at run end (and the final
    step's own on_save may have been skipped on that busy lock). on_train_end must then publish the
    final deployable WITHOUT the lock instead of timing out and skipping it — otherwise the worker
    exits, kills the daemon mid-upload, and `flash checkpoints` is empty despite a successful run."""
    import threading

    from flash.engine.worker import hf as worker_hf

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    monkeypatch.setattr(worker_hf, "_CKPT_FLUSH_TIMEOUT_S", 0.05)  # give up on the held lock fast

    out = tmp_path / "out"
    out.mkdir()
    _make_ckpt_dir(out, 4)  # earlier step; its on_save upload will block, holding the lock
    _make_ckpt_dir(out, 8)  # the FINAL step (its on_save was skipped on the busy lock)

    holding = threading.Event()
    release = threading.Event()
    base_upload = rec.upload_folder

    def upload(**kwargs):
        base_upload(**kwargs)
        if kwargs["path_in_repo"].endswith("checkpoint/checkpoint-4"):
            holding.set()  # the resume upload now holds the lock...
            release.wait(5)  # ...keep holding it across the on_train_end flush

    rec.upload_folder = upload

    started: list = []
    real_thread = threading.Thread
    monkeypatch.setattr(
        worker.threading,
        "Thread",
        lambda *a, **k: started.append(real_thread(*a, **k)) or started[-1],
    )

    cb = worker.make_checkpoint_upload_callback()
    try:
        cb.on_save(SimpleNamespace(output_dir=str(out)), SimpleNamespace(global_step=4), None)
        assert holding.wait(5), "the step-4 upload should be holding the lock"
        cb.on_train_end(SimpleNamespace(output_dir=str(out)), None, None)
        deployable = [
            u for u in rec.uploads if u["path_in_repo"].endswith("checkpoints/step-8/adapter")
        ]
        assert len(deployable) == 1, "final deployable must publish even when the lock is held"
    finally:
        release.set()
        for t in started:
            t.join(5)


def test_on_train_end_no_checkpoints_is_noop(tmp_path, monkeypatch, fake_trainer_callback):
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    out = tmp_path / "out"
    out.mkdir()
    worker.make_checkpoint_upload_callback().on_train_end(
        SimpleNamespace(output_dir=str(out)), None, None
    )
    assert rec.uploads == []


def test_on_save_publishes_deployable_before_resume(tmp_path, monkeypatch, fake_trainer_callback):
    """The durable, accumulating deployable adapter must be uploaded BEFORE the larger
    latest-only resume checkpoint, so it lands first if the worker is torn down mid-upload."""
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)

    class _SyncThread:  # run the upload body inline so the unit test is deterministic
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(worker.threading, "Thread", _SyncThread)
    out = tmp_path / "out"
    out.mkdir()
    _make_ckpt_dir(out, 4)
    cb = worker.make_checkpoint_upload_callback()
    cb.on_save(
        SimpleNamespace(output_dir=str(out)),
        SimpleNamespace(global_step=4),
        None,
    )
    paths = [u["path_in_repo"] for u in rec.uploads]
    assert paths == [
        "rl/flash-ckpt-1/checkpoints/step-4/adapter",  # deployable first
        "rl/flash-ckpt-1/checkpoint/checkpoint-4",  # resume second
    ]


def test_on_save_queues_busy_step_instead_of_skipping(tmp_path, monkeypatch, fake_trainer_callback):
    """`save_every` must not be advisory: a save that fires while a prior upload is still in
    flight is queued and uploaded once the in-flight one finishes — previously it was dropped
    ("upload busy; skipping step N"), leaving a sparse registered step list."""
    import threading

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)

    out = tmp_path / "out"
    out.mkdir()
    _make_ckpt_dir(out, 4)
    _make_ckpt_dir(out, 8)

    holding = threading.Event()
    release = threading.Event()
    base_upload = rec.upload_folder

    def upload(**kwargs):
        base_upload(**kwargs)
        if kwargs["path_in_repo"].endswith("checkpoint/checkpoint-4"):
            holding.set()  # the step-4 resume upload now holds the lock...
            release.wait(5)  # ...until we let it finish

    rec.upload_folder = upload

    cb = worker.make_checkpoint_upload_callback()
    cb.on_save(SimpleNamespace(output_dir=str(out)), SimpleNamespace(global_step=4), None)
    assert holding.wait(5), "the step-4 upload should be holding the lock"
    # step-8 save fires while step-4 is uploading: it must be queued, not skipped.
    cb.on_save(SimpleNamespace(output_dir=str(out)), SimpleNamespace(global_step=8), None)
    release.set()

    deadline = time.time() + 10
    while time.time() < deadline:
        if any(u["path_in_repo"].endswith("checkpoints/step-8/adapter") for u in rec.uploads):
            break
        time.sleep(0.02)
    deployable = [
        u for u in rec.uploads if u["path_in_repo"].endswith("checkpoints/step-8/adapter")
    ]
    assert len(deployable) == 1, "a busy-lock save must be queued and uploaded, not skipped"


def test_upload_pump_no_lost_wakeup(tmp_path):
    """Regression: a save enqueued in the window where the in-flight upload has passed its own
    drain check but not yet finished must still be started by done() — with the old two-lock
    design it could be silently dropped."""
    from flash.engine.worker.hf import _UploadPump

    ckpt = tmp_path / "checkpoint-8"
    ckpt.mkdir()
    started: list[int] = []
    pump = _UploadPump(lambda step, d: started.append(step))
    # Simulate an in-flight upload that has already drained the queue (the racy window).
    pump.uploading = True
    pump.enqueue(8, str(ckpt))
    assert started == [], "enqueue while busy must queue, not start"
    pump.done()  # in-flight finishes: the queued step MUST be pumped, never lost
    assert started == [8]
    pump.done()
    assert started == [8], "queue is single-slot; nothing left to start"


def test_on_train_end_flushes_queued_deployable_without_resume_race(
    tmp_path, monkeypatch, fake_trainer_callback
):
    """A save queued behind a still-in-flight upload at run end must publish its deployable adapter
    without starting a second resume-checkpoint upload/prune race."""
    import threading

    from flash.engine.worker import hf as worker_hf

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    monkeypatch.setattr(worker_hf, "_CKPT_FLUSH_TIMEOUT_S", 0.05)

    out = tmp_path / "out"
    out.mkdir()
    _make_ckpt_dir(out, 4)
    _make_ckpt_dir(out, 8)

    holding = threading.Event()
    release = threading.Event()
    base_upload = rec.upload_folder

    def upload(**kwargs):
        base_upload(**kwargs)
        if kwargs["path_in_repo"].endswith("checkpoint/checkpoint-4"):
            holding.set()  # the step-4 resume upload is now in flight...
            release.wait(5)  # ...and stays in flight across on_train_end

    rec.upload_folder = upload

    cb = worker.make_checkpoint_upload_callback()
    try:
        cb.on_save(SimpleNamespace(output_dir=str(out)), SimpleNamespace(global_step=4), None)
        assert holding.wait(5), "the step-4 upload should be in flight"
        cb.on_save(SimpleNamespace(output_dir=str(out)), SimpleNamespace(global_step=8), None)
        cb.on_train_end(SimpleNamespace(output_dir=str(out)), None, None)
        paths = [u["path_in_repo"] for u in rec.uploads]
        assert "rl/flash-ckpt-1/checkpoints/step-8/adapter" in paths, (
            "queued step's deployable must be flushed at train end"
        )
        assert "rl/flash-ckpt-1/checkpoint/checkpoint-8" not in paths, (
            "queued step's resume checkpoint must not race the in-flight resume prune"
        )
    finally:
        release.set()


def test_prune_stale_resume_checkpoints_keeps_only_latest(monkeypatch):
    """The streamed resume checkpoint is latest-only, but upload_folder's delete_patterns can't reach
    sibling step dirs (they're matched relative to the per-step path_in_repo), so older checkpoint-N
    dirs are pruned explicitly. The plural deployable tree (checkpoints/) must be left intact."""
    import flash.engine.worker.hf as worker_hf

    prefix = "rl/flash-ckpt-1"
    files = [
        f"{prefix}/checkpoint/checkpoint-20/optimizer.pt",
        f"{prefix}/checkpoint/checkpoint-20/adapter_model.safetensors",
        f"{prefix}/checkpoint/checkpoint-40/optimizer.pt",
        f"{prefix}/checkpoint/checkpoint-60/optimizer.pt",
        f"{prefix}/checkpoints/step-60/adapter/adapter_model.safetensors",  # deployable (plural) -> keep
        f"{prefix}/metrics.json",
    ]
    rec = _RecordingHfApi(files)
    _prime_worker(monkeypatch, rec)

    worker_hf._prune_stale_resume_checkpoints(60)

    assert sorted(rec.deleted) == [
        f"{prefix}/checkpoint/checkpoint-20",
        f"{prefix}/checkpoint/checkpoint-40",
    ]
    assert f"{prefix}/checkpoint/checkpoint-60" not in rec.deleted  # latest kept
    assert all("checkpoints/" not in d for d in rec.deleted)  # deployable tree untouched


def test_prune_stale_resume_checkpoints_no_repo_is_noop(monkeypatch):
    import flash.engine.worker.hf as worker_hf

    rec = _RecordingHfApi(["rl/r/checkpoint/checkpoint-1/optimizer.pt"])
    _prime_worker(monkeypatch, rec, repo="")  # local run, no HF repo
    worker_hf._prune_stale_resume_checkpoints(5)
    assert rec.deleted == []


def test_on_save_skips_deployable_when_vl_recombine_fails(
    tmp_path, monkeypatch, fake_trainer_callback
):
    """A VL warm-start whose recorded SFT dir was evicted makes recombined_warmstart_adapter_dir
    RAISE. The raw checkpoint adapter is GRPO-only / SFT-less, so the deployable publish must be
    SKIPPED (not fall back to the raw adapter and advertise a known-broken step). The resume
    checkpoint is still uploaded so the run can resume and re-merge."""
    import flash.engine.worker as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)

    def _raise(_ckpt):
        raise RuntimeError("recombine: ... SFT-less adapter cannot be recombined")

    monkeypatch.setattr(worker, "recombined_warmstart_adapter_dir", _raise)

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(worker.threading, "Thread", _SyncThread)
    out = tmp_path / "out"
    out.mkdir()
    _make_ckpt_dir(out, 4)
    cb = worker.make_checkpoint_upload_callback()
    cb.on_save(SimpleNamespace(output_dir=str(out)), SimpleNamespace(global_step=4), None)

    paths = [u["path_in_repo"] for u in rec.uploads]
    # NO deployable adapter advertised for this step...
    assert not any(p.endswith("checkpoints/step-4/adapter") for p in paths), paths
    # ...but the resume checkpoint is still uploaded so the run can resume and re-merge.
    assert "rl/flash-ckpt-1/checkpoint/checkpoint-4" in paths


def test_recombined_warmstart_skips_stale_legacy_weights(tmp_path, monkeypatch):
    """The recombined deploy adapter writes fresh safetensors; do not copy stale raw .bin weights."""
    import shutil
    from pathlib import Path

    import flash.engine.worker as worker
    import flash.engine.worker.adapter as worker_adapter

    sft = tmp_path / "sft"
    grpo = tmp_path / "grpo"
    sft.mkdir()
    grpo.mkdir()
    (grpo / "adapter_config.json").write_text("{}")
    (grpo / "adapter_model.bin").write_bytes(b"stale")
    (grpo / "special_tokens_map.json").write_text("{}")
    (grpo / "optimizer.pt").write_text("trainer state")

    def fake_recombine(sft_adir, src_adapter_dir, out_dir, *, model_id=None):
        assert sft_adir == str(sft)
        assert src_adapter_dir == str(grpo)
        out = Path(out_dir)
        (out / "adapter_config.json").write_text("{}")
        (out / "adapter_model.safetensors").write_bytes(b"fresh")
        return 8

    monkeypatch.setattr(worker, "_VL_WARMSTART_SFT_DIR", str(sft), raising=False)
    monkeypatch.setattr(worker, "_VL_WARMSTART_MODEL_ID", "Qwen/Qwen3.5-4B", raising=False)
    monkeypatch.setattr(worker_adapter, "recombine_lora_adapters", fake_recombine)

    out_path = worker_adapter.recombined_warmstart_adapter_dir(str(grpo))
    assert out_path is not None
    out = Path(out_path)

    try:
        assert (out / "adapter_model.safetensors").exists()
        assert not (out / "adapter_model.bin").exists()
        assert (out / "special_tokens_map.json").exists()
        assert not (out / "optimizer.pt").exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)


# --------------------------------------------------------------------------------------------
# Control plane: list_checkpoints
# --------------------------------------------------------------------------------------------
class _FakeHfApiFiles:
    def __init__(self, files):
        self._files = files

    def __call__(self, *a, **k):  # stand in for HfApi(...)
        return self

    def list_repo_files(self, repo, repo_type=None):
        return self._files


def _patch_hf_files(monkeypatch, files):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApiFiles(files))


def test_checkpoint_adapter_prefix():
    assert checkpoint_adapter_prefix(_spec(), 60) == "rl/flash-ckpt-1/checkpoints/step-60"


def test_list_checkpoints_parses_and_sorts(monkeypatch):
    base = "rl/flash-ckpt-1"
    files = [
        f"{base}/checkpoints/step-80/adapter/adapter_config.json",
        f"{base}/checkpoints/step-80/adapter/adapter_model.safetensors",
        f"{base}/checkpoints/step-40/adapter/adapter_config.json",
        f"{base}/checkpoints/step-40/adapter/adapter_model.safetensors",
        # noise that must NOT be picked up:
        f"{base}/checkpoint/checkpoint-90/optimizer.pt",
        f"{base}/adapter/adapter_config.json",
        f"{base}/heartbeat.json",
    ]
    _patch_hf_files(monkeypatch, files)

    out = list_checkpoints(_spec())

    assert [c["step"] for c in out] == [40, 80]
    assert out[0]["adapter_prefix"] == f"{base}/checkpoints/step-40"
    assert out[0]["subfolder"] == f"{base}/checkpoints/step-40/adapter"
    assert out[0]["repo_id"] == "org/test-runs"
    assert out[1]["step"] == 80


def test_list_checkpoints_accepts_legacy_bin_weights(monkeypatch):
    base = "rl/flash-ckpt-1"
    files = [
        f"{base}/checkpoints/step-40/adapter/adapter_config.json",
        f"{base}/checkpoints/step-40/adapter/adapter_model.bin",
    ]
    _patch_hf_files(monkeypatch, files)

    assert [c["step"] for c in list_checkpoints(_spec())] == [40]


def test_list_checkpoints_skips_step_without_weights(monkeypatch):
    """A step with adapter_config.json but no weights file is NOT advertised as deployable."""
    base = "rl/flash-ckpt-1"
    files = [
        # step-40 is complete; step-60 has config only (half-uploaded) and must be excluded.
        f"{base}/checkpoints/step-40/adapter/adapter_config.json",
        f"{base}/checkpoints/step-40/adapter/adapter_model.safetensors",
        f"{base}/checkpoints/step-60/adapter/adapter_config.json",
    ]
    _patch_hf_files(monkeypatch, files)
    assert [c["step"] for c in list_checkpoints(_spec())] == [40]


def test_list_checkpoints_no_repo(monkeypatch):
    spec = JobSpec.from_dict({**SPEC_DICT, "train": {"steps": 1, "hf_repo": ""}})
    assert list_checkpoints(spec) == []


def test_list_checkpoints_swallows_hf_error(monkeypatch):
    import huggingface_hub

    class _Boom:
        def __call__(self, *a, **k):
            return self

        def list_repo_files(self, *a, **k):
            raise RuntimeError("hf down")

    monkeypatch.setattr(huggingface_hub, "HfApi", _Boom())
    assert list_checkpoints(_spec()) == []  # best-effort: never raises into a run/route


def test_final_adapter_exists_requires_config_and_weights(monkeypatch):
    base = "rl/flash-ckpt-1"
    _patch_hf_files(
        monkeypatch,
        [
            f"{base}/adapter/adapter_config.json",
            f"{base}/adapter/adapter_model.safetensors",
        ],
    )

    assert final_adapter_exists(_spec()) is True


def test_final_adapter_exists_accepts_legacy_bin_weights(monkeypatch):
    base = "rl/flash-ckpt-1"
    _patch_hf_files(
        monkeypatch,
        [
            f"{base}/adapter/adapter_config.json",
            f"{base}/adapter/adapter_model.bin",
        ],
    )

    assert final_adapter_exists(_spec()) is True


def test_final_adapter_exists_rejects_incomplete_or_nested_files(monkeypatch):
    base = "rl/flash-ckpt-1"
    _patch_hf_files(
        monkeypatch,
        [
            f"{base}/adapter/adapter_config.json",
            f"{base}/adapter/nested/adapter_model.safetensors",
        ],
    )

    assert final_adapter_exists(_spec()) is False


def test_final_adapter_exists_raises_final_adapter_listing_error(monkeypatch):
    import huggingface_hub

    class _Boom:
        def __call__(self, *a, **k):
            return self

        def list_repo_files(self, *a, **k):
            raise RuntimeError("hf down")

    monkeypatch.setattr(huggingface_hub, "HfApi", _Boom())

    with pytest.raises(CheckpointListingError) as exc_info:
        final_adapter_exists(_spec())

    message = str(exc_info.value)
    assert "could not verify final adapter for flash-ckpt-1: hf down" in message
    assert "deployable checkpoints" not in message


# --------------------------------------------------------------------------------------------
# Backend client: register_run_checkpoints / register_checkpoints_best_effort
# --------------------------------------------------------------------------------------------
def _status(**kw):
    base = {
        "run_id": "flash-ckpt-1",
        "spec": SPEC_DICT,
        "billing_context": {"org_id": "org-xyz"},
    }
    base.update(kw)
    return SimpleNamespace(**base)


_CKPTS = [
    {
        "step": 40,
        "subfolder": "rl/flash-ckpt-1/checkpoints/step-40/adapter",
        "repo_id": "org/test-runs",
        "repo_type": "dataset",
    },
    {
        "step": 80,
        "subfolder": "rl/flash-ckpt-1/checkpoints/step-80/adapter",
        "repo_id": "org/test-runs",
        "repo_type": "dataset",
    },
]


def test_register_run_checkpoints_body_shape(monkeypatch):
    import flash.server.checkpoints as ck

    captured = {}
    monkeypatch.setattr(
        ck,
        "_post_checkpoints",
        lambda *, token, body: captured.update(token=token, body=body) or {},
    )
    ck.register_run_checkpoints(internal_key="int-key", status=_status(), checkpoints=_CKPTS)

    assert captured["token"] == "int-key"
    body = captured["body"]
    assert body["orgId"] == "org-xyz"
    assert body["runId"] == "flash-ckpt-1"
    assert body["baseModel"] == "Qwen/Qwen3.5-4B"
    assert body["repoId"] == "org/test-runs"
    assert body["repoType"] == "dataset"
    assert body["checkpoints"] == [
        {"step": 40, "subfolder": "rl/flash-ckpt-1/checkpoints/step-40/adapter"},
        {"step": 80, "subfolder": "rl/flash-ckpt-1/checkpoints/step-80/adapter"},
    ]


def test_register_run_checkpoints_requires_org():
    import flash.server.checkpoints as ck

    with pytest.raises(ValueError, match="org id"):
        ck.register_run_checkpoints(
            internal_key="k",
            status=_status(billing_context={}, platform_context=None),
            checkpoints=_CKPTS,
        )


def test_register_run_checkpoints_falls_back_to_platform_context(monkeypatch):
    # Internal/operator runs carry org only in platform_context (billing_context is None):
    # registration must still scope rows to that org. _run_org_id falls back to billing-then-platform
    # (same order as routes/serving.py::_run_org; NOT run_registry, which is platform-first).
    import flash.server.checkpoints as ck

    captured = {}
    monkeypatch.setattr(
        ck, "_post_checkpoints", lambda *, token, body: captured.update(body=body) or {}
    )
    ck.register_run_checkpoints(
        internal_key="int-key",
        status=_status(billing_context=None, platform_context={"org_id": "org-plat"}),
        checkpoints=_CKPTS,
    )
    assert captured["body"]["orgId"] == "org-plat"


def test_best_effort_noop_without_internal_key(monkeypatch):
    import flash.server.checkpoints as ck

    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    # Even if HF had checkpoints, no internal key => skip persistence (HF stays source of truth).
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: _CKPTS)
    assert ck.register_checkpoints_best_effort(_status()) == 0


def test_best_effort_registers(monkeypatch):
    import flash.server.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: _CKPTS)
    posted = {}
    monkeypatch.setattr(
        ck, "_post_checkpoints", lambda *, token, body: posted.update(body=body) or {}
    )

    assert ck.register_checkpoints_best_effort(_status()) == 2
    assert posted["body"]["runId"] == "flash-ckpt-1"


def test_best_effort_swallows_backend_failure(monkeypatch):
    import io
    import urllib.error

    import flash.server.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: _CKPTS)

    def boom(*, token, body):
        raise urllib.error.URLError("backend down")

    monkeypatch.setattr(ck, "_post_checkpoints", boom)
    log = io.StringIO()
    assert ck.register_checkpoints_best_effort(_status(), log=log) == 0  # never raises
    # A genuine backend failure MUST stay visible.
    assert "warn" in log.getvalue()


def test_best_effort_skips_silently_when_no_org(monkeypatch):
    # Internal/operator run with no org in either context: skip quietly, do NOT warn, do NOT hit the
    # backend, and (Copilot Msbm-) do NOT perform the HF checkpoint listing — the org check
    # short-circuits BEFORE the network call. Regression guard for the noisy "missing org id" log and
    # the wasted HF listing on an expected-skip run.
    import io

    import flash.server.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    listed = {"called": False}

    def fake_list(spec):
        listed["called"] = True
        return _CKPTS

    monkeypatch.setattr(ck, "list_checkpoints", fake_list)

    def fail(*, token, body):  # pragma: no cover - must never be called
        raise AssertionError("_post_checkpoints must not be called without an org")

    monkeypatch.setattr(ck, "_post_checkpoints", fail)
    log = io.StringIO()
    status = _status(billing_context={}, platform_context=None)
    assert ck.register_checkpoints_best_effort(status, log=log) == 0
    assert "warn" not in log.getvalue()
    assert listed["called"] is False  # org check short-circuits before the HF listing


def test_best_effort_no_checkpoints(monkeypatch):
    import flash.server.checkpoints as ck

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "int-key")
    monkeypatch.setattr(ck, "list_checkpoints", lambda spec: [])
    assert ck.register_checkpoints_best_effort(_status()) == 0


# ---------------------------------------------------------------------------
# Finalize wiring: the FINAL training step is always published as a deployable
# checkpoint (not only when it lands on a save_steps boundary). The per-save
# callback / on_train_end publish at/near save boundaries; an unaligned last
# step would otherwise have no `RUN_ID/step-N` entry even though it IS the served
# default `<prefix>/adapter`. run_rl/run_sft must close that gap after saving
# the final adapter. Source-wiring (a runtime test would need a full trainer).
# ---------------------------------------------------------------------------
def _finalize_src(module_name: str, fn_name: str) -> str:
    import importlib
    import inspect

    mod = importlib.import_module(module_name)
    return inspect.getsource(getattr(mod, fn_name))


def test_run_rl_publishes_final_step_as_deployable_checkpoint():
    src = _finalize_src("flash.engine.worker.rl", "run_rl")
    # Final adapter is saved, recombined for a VL warm-start (SFT⊕GRPO so the deploy isn't SFT-less),
    # then both the default upload and the deployable checkpoint ship that recombined dir — and the
    # recombined temp dir is cleaned up afterwards so finalize doesn't leak /tmp.
    assert "save_pretrained(adapter_dir)" in src
    assert "recombined = _w.recombined_warmstart_adapter_dir(adapter_dir)" in src
    assert "deploy_dir = recombined or adapter_dir" in src
    assert 'hf_upload_folder(deploy_dir, "adapter"' in src
    assert "publish_deployable_checkpoint(deploy_dir, _steps_run)" in src
    assert "shutil.rmtree(recombined" in src


def test_run_sft_publishes_final_step_as_deployable_checkpoint():
    src = _finalize_src("flash.engine.worker.sft", "run_sft")
    assert "save_pretrained(adapter_dir)" in src
    # SFT derives the final step from trainer state (no _steps_run var) and publishes it.
    assert "global_step" in src
    assert "publish_deployable_checkpoint(adapter_dir, _final_step)" in src
