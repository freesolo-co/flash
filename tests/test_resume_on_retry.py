"""Resume-from-latest-checkpoint on crash/preemption retry (CPU-only).

A crash- or preemption-killed run is relaunched on a fresh worker by the control-plane
retry loop, and that fresh worker continues training from the latest streamed checkpoint
rather than restarting from scratch. This file locks in the two live load-bearing pieces:

  1. worker resume: ``hf_resume_checkpoint`` pulls the stream and returns the highest-step
     checkpoint directory used by the training backend.
  2. control plane: ``_run_attempts_supervised`` relaunches on the same run id and seed for every
     infra-shaped failure; a genuine worker error fails fast instead.

All HF/provider/network boundaries are stubbed; nothing here touches a GPU or the network.
"""

from __future__ import annotations

import io
import json
import os
import shutil

import pytest

import flash.engine.worker.io.heartbeat as worker_heartbeat
import flash.engine.worker.io.hf as worker_hf
import flash.engine.worker.runtime.state as worker_state
import flash.engine.worker.train.opd.orchestration.failures as opd_failures
import flash.engine.worker.train.rl.launch.checkpoints as rl_checkpoints
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.supervise.lifecycle as runner_lifecycle
from flash.core.spec import JobSpec
from tests._helpers.profile import attach_sft_profile, stub_revision_geometry
from tests._helpers.source_snapshot import valid_source_snapshot

# Infra-shaped failure categories the retry loop resumes on (see lifecycle._run_attempts_supervised).
# Mirrors the literal tuple in the source; this test is the guard that the set doesn't silently drift.
INFRA_SHAPED = ("stalled", "no_capacity", "poll_error", "job_preempted")
_RUNPOD_FINGERPRINT = "rpk-" + "0" * 64
_SOURCE_SNAPSHOT = valid_source_snapshot()


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


def _prime_worker(monkeypatch, recorder, *, repo="org/test-runs", run="flash-resume-1"):
    monkeypatch.setattr(worker_state, "HF_REPO", repo)
    monkeypatch.setattr(worker_state, "PHASE", "rl")
    monkeypatch.setattr(worker_state, "RUN_ID", run)
    monkeypatch.setattr(worker_state, "SEED", 0)
    monkeypatch.setattr(worker_hf, "hf_api", lambda: recorder)
    monkeypatch.setattr(worker_heartbeat, "heartbeat", lambda *a, **k: None)


# ============================================================================================
# worker resume: hf_resume_checkpoint
# ============================================================================================
def _fake_snapshot(steps, *, with_weights=True):
    """Stand in for huggingface_hub.snapshot_download: materialize the selected checkpoint dirs.



    The real call downloads ``<prefix>/checkpoint/**`` into ``local_dir``; the fake recreates just
    that structure for the steps it's told about, honoring the prefix from ``allow_patterns``.
    """

    def _dl(*, repo_id, repo_type, allow_patterns, local_dir, token=None, **_):
        prefix = allow_patterns[0][: -len("/checkpoint/**")]
        for s in steps:
            d = os.path.join(local_dir, prefix, "checkpoint", f"checkpoint-{s}")
            os.makedirs(d, exist_ok=True)
            if with_weights:
                with open(os.path.join(d, "adapter_model.safetensors"), "wb") as fh:
                    fh.write(b"w")
        return local_dir

    return _dl


@pytest.fixture
def resume_run_id():
    """hf_resume_checkpoint downloads to the worker's hardcoded /tmp/resume; reap our subtree.

    Hermetic: the run id is unique per process and the subtree is cleaned BOTH before and after, so
    stale data left under the same id (a prior aborted run, or a concurrent xdist worker) can't make
    hf_resume_checkpoint resolve an unexpected higher checkpoint and flake the latest-step assertions."""
    run = f"flash-resume-utest-{os.getpid()}"
    path = os.path.join("/tmp/resume", "rl", run)
    shutil.rmtree(path, ignore_errors=True)
    yield run
    shutil.rmtree(path, ignore_errors=True)


def test_hf_resume_checkpoint_returns_latest_step(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([40, 80, 60]))

    path = worker_hf.hf_resume_checkpoint()

    assert path is not None
    assert path.endswith("checkpoint-80"), (
        "must resume from the highest streamed step, not the first"
    )
    assert os.path.isdir(path)


def test_hf_resume_checkpoint_uses_pinned_revision_and_ignores_stray_dir(
    monkeypatch, resume_run_id
):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)
    seen = {}

    def snapshot(**kwargs):
        seen.update(kwargs)
        base = os.path.join(kwargs["local_dir"], "rl", resume_run_id, "checkpoint")
        for name in ("checkpoint-40", "checkpoint-latest", "checkpoint-80"):
            os.makedirs(os.path.join(base, name), exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    path = worker_hf.hf_resume_checkpoint(fail_closed=True, revision="pinned-sha")

    assert seen["revision"] == "pinned-sha"
    assert path is not None
    assert path.endswith("checkpoint-80")


@pytest.mark.parametrize("name", ["checkpoint-0", "checkpoint-040"])
def test_hf_resume_checkpoint_pinned_rejects_noncanonical_only(monkeypatch, resume_run_id, name):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)

    def snapshot(**kwargs):
        base = os.path.join(kwargs["local_dir"], "rl", resume_run_id, "checkpoint", name)
        os.makedirs(base, exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    with pytest.raises(RetriableInfraError, match="required resume checkpoint is missing"):
        worker_hf.hf_resume_checkpoint(fail_closed=True, revision="pinned-sha")


def test_hf_resume_checkpoint_selects_canonical_alongside_noncanonical(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)

    def snapshot(**kwargs):
        base = os.path.join(kwargs["local_dir"], "rl", resume_run_id, "checkpoint")
        for name in ("checkpoint-0", "checkpoint-040", "checkpoint-40"):
            os.makedirs(os.path.join(base, name), exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    path = worker_hf.hf_resume_checkpoint(fail_closed=True, revision="pinned-sha")

    assert path is not None
    assert path.endswith("checkpoint-40")


def test_hf_resume_checkpoint_none_without_repo(monkeypatch):
    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, repo="")
    # no repo is a legitimate fresh start only when the control plane did not require resume.
    assert worker_hf.hf_resume_checkpoint() is None


def test_hf_resume_checkpoint_required_without_repo_raises(monkeypatch):
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, repo="")
    with pytest.raises(RetriableInfraError, match="has no artifact repository"):
        worker_hf.hf_resume_checkpoint(revision="pinned-sha")


def test_hf_resume_checkpoint_none_when_nothing_streamed(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)
    # A run preempted before its first save has no streamed checkpoint -> fresh start, not a crash.
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([]))
    assert worker_hf.hf_resume_checkpoint() is None


def test_hf_resume_checkpoint_swallows_download_error(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)

    def _boom(**_):
        raise RuntimeError("hf down")

    # A transient HF outage must degrade to "start fresh", never abort the paid run.
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)
    assert worker_hf.hf_resume_checkpoint() is None


def test_hf_resume_checkpoint_starts_no_download_at_deadline(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(worker_state, "_remaining_worker_wall_seconds", lambda: 0.0)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: pytest.fail("snapshot download must not start at the deadline"),
    )

    assert worker_hf.hf_resume_checkpoint() is None


def test_hf_resume_checkpoint_fail_closed_on_download_error(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)

    def _boom(**_):
        raise RuntimeError("hf down")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)
    with pytest.raises(RetriableInfraError, match="required resume checkpoint fetch failed"):
        worker_hf.hf_resume_checkpoint(fail_closed=True)


def test_hf_resume_checkpoint_fail_closed_at_deadline(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(worker_state, "_remaining_worker_wall_seconds", lambda: 0.0)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: pytest.fail("snapshot download must not start at the deadline"),
    )

    with pytest.raises(RetriableInfraError, match="required resume checkpoint fetch failed"):
        worker_hf.hf_resume_checkpoint(fail_closed=True)


def test_hf_resume_checkpoint_fail_closed_allows_confirmed_absence(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([]))

    assert worker_hf.hf_resume_checkpoint(fail_closed=True) is None


def test_hf_resume_checkpoint_pinned_missing_base_raises(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([]))

    with pytest.raises(RetriableInfraError, match="required resume checkpoint is missing"):
        worker_hf.hf_resume_checkpoint(fail_closed=True, revision="pinned-sha")


def test_hf_resume_checkpoint_pinned_absence_clears_stale_local_state(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)
    stale = os.path.join("/tmp/resume", "rl", resume_run_id, "checkpoint", "checkpoint-80")
    os.makedirs(stale, exist_ok=True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([]))

    with pytest.raises(RetriableInfraError, match="required resume checkpoint is missing"):
        worker_hf.hf_resume_checkpoint(revision="pinned-sha")

    assert not os.path.exists(stale)


def test_hf_resume_checkpoint_prefer_picks_highest_satisfying_candidate(monkeypatch, resume_run_id):
    """`prefer` steers the pick to the highest candidate it accepts, not the highest step overall,
    when a later, higher, unaccepted checkpoint also streamed -- the case that used to starve a
    compatible lower checkpoint forever once a higher incompatible one became the remote max."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([40, 60, 80]))

    path = worker_hf.hf_resume_checkpoint(prefer=lambda p: not p.endswith("checkpoint-80"))

    assert path is not None
    assert path.endswith("checkpoint-60"), "must pick the highest candidate prefer accepts"


def test_hf_resume_checkpoint_prefer_falls_back_to_highest_overall(monkeypatch, resume_run_id):
    """when no candidate satisfies `prefer`, the selection falls back to the plain highest step --
    the same answer this returned before `prefer` existed, leaving the caller's own discard log as
    the thing that explains a restart from zero."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([40, 60, 80]))

    path = worker_hf.hf_resume_checkpoint(prefer=lambda _p: False)

    assert path is not None
    assert path.endswith("checkpoint-80")


def test_hf_resume_checkpoint_pinned_without_numeric_dir_raises(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, run=resume_run_id)

    def snapshot(**kwargs):
        base = os.path.join(
            kwargs["local_dir"], "rl", resume_run_id, "checkpoint", "checkpoint-latest"
        )
        os.makedirs(base, exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    with pytest.raises(RetriableInfraError, match="required resume checkpoint is missing"):
        worker_hf.hf_resume_checkpoint(fail_closed=True, revision="pinned-sha")


# ============================================================================================
# worker resume: fsdp world-size guard when a retry lands on a different card count
# ============================================================================================
def _staged_checkpoint(root, step, *, world_size=None, fsdp_version=2, shards=0, nested=True):
    """build a downloaded ``checkpoint-N`` shaped like the one verl uploaded.

    ``nested`` mirrors verl's two trainers: rl/opd put the shards under ``actor/``, while sft writes
    them straight into the step dir. a complete native checkpoint has one model, optimizer, and
    extra-state shard for every rank.
    """
    src = root / f"checkpoint-{step}"
    inner = src / "actor" if nested else src
    (inner / "huggingface").mkdir(parents=True)
    (inner / "model.safetensors").write_text("weights")
    if world_size is not None:
        (inner / "fsdp_config.json").write_text(
            json.dumps({"FSDP_version": fsdp_version, "world_size": world_size})
        )
    for kind in ("model", "optim", "extra_state"):
        for rank in range(shards):
            (inner / f"{kind}_world_size_{shards}_rank_{rank}.pt").write_bytes(b"shard")
    (src / "_flash_resume_manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "checkpoint_step": step,
                "checkpoint_attempt": 0,
                "required_adapters": [],
                "first_positive_grad_step": 1,
            }
        )
    )
    return src


def _rl_resume(monkeypatch, tmp_path, src, *, world_size):

    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(worker_hf, "hf_resume_checkpoint", lambda *a, **k: str(src))
    return (
        rl_checkpoints._restore_verl_resume(
            str(local_dir),
            world_size=world_size,
            expected_fsdp_generation=2,
        ),
        local_dir,
    )


def test_matching_world_size_resumes(monkeypatch, tmp_path):
    src = _staged_checkpoint(tmp_path, 7, world_size=4, shards=4)

    step, local_dir = _rl_resume(monkeypatch, tmp_path, src, world_size=4)

    assert step == 7
    assert (local_dir / "latest_checkpointed_iteration.txt").read_text().strip() == "7"
    assert (local_dir / "global_step_7" / "actor" / "model.safetensors").exists()


def test_mismatched_world_size_discards_instead_of_crashing(monkeypatch, tmp_path, capsys):
    # the reachable case: recovery re-allocated a 4x run onto a 2x shape, and verl would look for
    # model_world_size_2_rank_*.pt in a directory that only has the 4-rank shards.
    src = _staged_checkpoint(tmp_path, 7, world_size=4, shards=4)

    step, local_dir = _rl_resume(monkeypatch, tmp_path, src, world_size=2)

    assert step == 0, "a shard set this attempt cannot load must not be staged"
    assert not (local_dir / "latest_checkpointed_iteration.txt").exists()
    assert not (local_dir / "global_step_7").exists()
    line = capsys.readouterr().out
    assert "discarding resume checkpoint checkpoint-7" in line
    assert "stamped 4" in line
    assert "expected 2" in line


def test_fsdp1_same_width_is_discarded(monkeypatch, tmp_path, capsys):
    src = _staged_checkpoint(tmp_path, 3, world_size=4, fsdp_version=1, shards=4)

    step, local_dir = _rl_resume(monkeypatch, tmp_path, src, world_size=4)

    assert step == 0
    assert not (local_dir / "global_step_3").exists()
    assert "restarting from step 0" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("expected_fsdp_generation", "checkpoint_fsdp_generation", "expected_loadable"),
    [(1, 1, True), (1, 2, False), (2, 1, False), (2, 2, True)],
)
def test_native_resume_requires_the_exact_fsdp_generation(
    tmp_path,
    expected_fsdp_generation,
    checkpoint_fsdp_generation,
    expected_loadable,
):
    from flash.engine.worker.verl.checkpoints import inspect_resume_checkpoint

    src = _staged_checkpoint(
        tmp_path,
        5,
        world_size=2,
        fsdp_version=checkpoint_fsdp_generation,
        shards=2,
    )

    assert (
        inspect_resume_checkpoint(
            str(src),
            world_size=2,
            expected_fsdp_generation=expected_fsdp_generation,
        ).loadable
        is expected_loadable
    )


@pytest.mark.parametrize(
    "stamped_config",
    [
        None,
        "not-json",
        "null",
        "[1, 2, 3]",
        json.dumps({"world_size": 2}),
        json.dumps({"FSDP_version": [1, 2], "world_size": 2}),
        json.dumps({"FSDP_version": 2, "world_size": "2"}),
        '{"FSDP_version": 2, "FSDP_version": 2, "world_size": 2}',
        '{"FSDP_version": 2, "world_size": 2, "world_size": 2}',
    ],
)
def test_missing_mixed_or_unreadable_fsdp_stamp_is_discarded(monkeypatch, tmp_path, stamped_config):
    src = _staged_checkpoint(tmp_path, 11, world_size=2, shards=2)
    stamp = src / "actor" / "fsdp_config.json"
    if stamped_config is None:
        stamp.unlink()
    else:
        stamp.write_text(stamped_config)

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=2)[0] == 0


@pytest.mark.parametrize("missing_kind", ["model", "optim", "extra_state"])
def test_incomplete_shard_classes_are_discarded(monkeypatch, tmp_path, missing_kind):
    src = _staged_checkpoint(tmp_path, 13, world_size=2, shards=2)
    for path in (src / "actor").glob(f"{missing_kind}_world_size_*.pt"):
        path.unlink()

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=2)[0] == 0


def test_missing_rank_shard_is_discarded(monkeypatch, tmp_path):
    src = _staged_checkpoint(tmp_path, 15, world_size=2, shards=2)
    (src / "actor" / "optim_world_size_2_rank_1.pt").unlink()

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=2)[0] == 0


@pytest.mark.parametrize("shard_kind", ["model", "optim", "extra_state"])
def test_zero_byte_shards_are_discarded(monkeypatch, tmp_path, shard_kind):
    src = _staged_checkpoint(tmp_path, 16, world_size=1, shards=1)
    (src / "actor" / f"{shard_kind}_world_size_1_rank_0.pt").write_bytes(b"")

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=1)[0] == 0


def test_mixed_shard_widths_are_discarded(monkeypatch, tmp_path):
    src = _staged_checkpoint(tmp_path, 17, world_size=2, shards=2)
    actor = src / "actor"
    (actor / "extra_state_world_size_2_rank_1.pt").rename(
        actor / "extra_state_world_size_4_rank_1.pt"
    )

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=2)[0] == 0


def test_lone_noncanonical_rank_is_discarded(monkeypatch, tmp_path):
    src = _staged_checkpoint(tmp_path, 19, world_size=2, shards=2)
    actor = src / "actor"
    (actor / "model_world_size_2_rank_0.pt").rename(actor / "model_world_size_2_rank_00.pt")

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=2)[0] == 0


def test_lone_noncanonical_world_size_is_discarded(monkeypatch, tmp_path):
    src = _staged_checkpoint(tmp_path, 21, world_size=1, shards=1)
    actor = src / "actor"
    (actor / "model_world_size_1_rank_0.pt").rename(actor / "model_world_size_01_rank_0.pt")

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=1)[0] == 0


def test_duplicate_rank_representation_is_discarded(monkeypatch, tmp_path):
    src = _staged_checkpoint(tmp_path, 22, world_size=1, shards=1)
    actor = src / "actor"
    (actor / "model_world_size_1_rank_00.pt").write_bytes(b"duplicate")

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=1)[0] == 0


def test_inconsistent_numeric_forms_across_shard_classes_are_discarded(monkeypatch, tmp_path):
    src = _staged_checkpoint(tmp_path, 23, world_size=1, shards=1)
    actor = src / "actor"
    (actor / "model_world_size_1_rank_0.pt").rename(actor / "model_world_size_01_rank_0.pt")
    (actor / "optim_world_size_1_rank_0.pt").rename(actor / "optim_world_size_1_rank_00.pt")

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=1)[0] == 0


def test_shard_named_directory_is_discarded(monkeypatch, tmp_path):
    src = _staged_checkpoint(tmp_path, 25, world_size=1, shards=1)
    shard = src / "actor" / "optim_world_size_1_rank_0.pt"
    shard.unlink()
    shard.mkdir()

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=1)[0] == 0


def test_broken_shard_symlink_is_discarded(monkeypatch, tmp_path):
    src = _staged_checkpoint(tmp_path, 27, world_size=1, shards=1)
    shard = src / "actor" / "extra_state_world_size_1_rank_0.pt"
    shard.unlink()
    shard.symlink_to(src / "missing-shard")

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=1)[0] == 0


@pytest.mark.parametrize("world_size", [1, 2])
def test_unknown_topology_is_always_discarded(monkeypatch, tmp_path, world_size, capsys):
    """A checkpoint with no readable topology evidence is discarded at every world size, including
    world_size=1.

    Recovery can retry any spec at any rentable card count (``rentable_gpu_counts`` walks every
    power of two up to the spec's cap, and the allocator rewrites the attempt to whichever one it
    lands on), so a single-gpu attempt is no proof its checkpoint was ever written at one rank -- the
    single-gpu exemption this test used to assert was wrong and is now removed.
    """
    src = _staged_checkpoint(tmp_path, 5)

    step, local_dir = _rl_resume(monkeypatch, tmp_path, src, world_size=world_size)

    assert step == 0, "unreadable topology must never be staged, at any world size"
    assert not (local_dir / "global_step_5").exists()
    assert "invalid or missing fsdp stamp" in capsys.readouterr().out


def test_sft_flat_layout_is_guarded_too(monkeypatch, tmp_path, capsys):
    # sft writes its shards into global_step_n itself rather than a nested actor/ dir.
    from flash.engine.worker.train.entry import sft_train

    src = _staged_checkpoint(tmp_path, 9, world_size=2, shards=2, nested=False)
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(worker_hf, "hf_resume_checkpoint", lambda *a, **k: str(src))

    assert (
        sft_train._restore_verl_resume(str(local_dir), world_size=2, expected_fsdp_generation=2)
        == 9
    )
    shutil.rmtree(local_dir)
    local_dir.mkdir()
    assert (
        sft_train._restore_verl_resume(str(local_dir), world_size=4, expected_fsdp_generation=2)
        == 0
    )
    assert not (local_dir / "global_step_9").exists()
    assert "[SFT] discarding resume checkpoint checkpoint-9" in capsys.readouterr().out

    shutil.rmtree(local_dir)
    local_dir.mkdir()
    fsdp1 = _staged_checkpoint(
        tmp_path,
        10,
        world_size=2,
        fsdp_version=1,
        shards=2,
        nested=False,
    )
    monkeypatch.setattr(worker_hf, "hf_resume_checkpoint", lambda *a, **k: str(fsdp1))
    assert (
        sft_train._restore_verl_resume(str(local_dir), world_size=2, expected_fsdp_generation=2)
        == 0
    )
    assert not (local_dir / "global_step_10").exists()


def test_sft_and_grpo_restore_use_the_canonical_worker_helper(monkeypatch, tmp_path):
    from flash.engine.support.verl_checkpoint import FsdpCheckpointInspection
    from flash.engine.worker.train.entry import sft_train
    from flash.engine.worker.verl import checkpoints as verl_checkpoints

    assert sft_train._restore_verl_resume.func is verl_checkpoints.restore_verl_resume
    assert sft_train._restore_verl_resume.keywords == {"job_label": "SFT"}

    checkpoint = _staged_checkpoint(tmp_path, 3, world_size=1, shards=1)
    inspection = FsdpCheckpointInspection(loadable=True, width=1)
    seen = {}

    def select(**kwargs):
        seen["accept"] = kwargs["accept"]
        assert kwargs["accept"](str(checkpoint), inspection)
        return str(checkpoint), inspection

    def stage(resume, local_dir, **kwargs):
        seen["staged"] = (resume, kwargs["inspection"])
        return 3

    monkeypatch.setattr(verl_checkpoints, "select_verl_resume_checkpoint", select)
    monkeypatch.setattr(verl_checkpoints, "stage_verl_resume", stage)

    assert (
        rl_checkpoints._restore_verl_resume(
            str(tmp_path / "local"), world_size=1, expected_fsdp_generation=2
        )
        == 3
    )
    assert seen["staged"] == (str(checkpoint), inspection)


def test_canonical_resume_selection_inspects_once_and_stages_the_same_verdict(
    monkeypatch, tmp_path
):
    from flash.engine.support.verl_checkpoint import FsdpCheckpointInspection
    from flash.engine.worker.verl import checkpoints as verl_checkpoints

    checkpoint = str(tmp_path / "checkpoint-5")
    inspection = FsdpCheckpointInspection(loadable=True, width=2)
    inspections = []
    staged = []

    def inspect(path, **kwargs):
        inspections.append(path)
        return inspection

    def select(*, prefer=None, **kwargs):
        assert prefer(checkpoint)
        assert prefer(checkpoint)
        return checkpoint

    def stage(resume, local_dir, **kwargs):
        staged.append((resume, kwargs["inspection"]))
        return 5

    monkeypatch.setattr(verl_checkpoints, "inspect_resume_checkpoint", inspect)
    monkeypatch.setattr(worker_hf, "hf_resume_checkpoint", select)
    monkeypatch.setattr(verl_checkpoints, "stage_verl_resume", stage)

    assert (
        verl_checkpoints.restore_verl_resume(
            str(tmp_path / "local"),
            world_size=2,
            expected_fsdp_generation=2,
            job_label="SFT",
        )
        == 5
    )
    assert inspections == [checkpoint]
    assert staged == [(checkpoint, inspection)]


@pytest.mark.parametrize("caller", ["generic", "grpo"])
def test_resume_restore_fails_closed_when_selection_omits_its_inspection(
    monkeypatch, tmp_path, caller
):
    from flash.engine.worker.verl import checkpoints as verl_checkpoints

    checkpoint = str(tmp_path / "checkpoint-5")
    staged = []
    monkeypatch.setattr(
        verl_checkpoints,
        "select_verl_resume_checkpoint",
        lambda **kwargs: (checkpoint, None),
    )
    monkeypatch.setattr(
        verl_checkpoints,
        "stage_verl_resume",
        lambda *args, **kwargs: staged.append((args, kwargs)),
    )

    def restore():
        if caller == "generic":
            return verl_checkpoints.restore_verl_resume(
                str(tmp_path / "local"),
                world_size=2,
                expected_fsdp_generation=2,
                job_label="SFT",
            )
        return rl_checkpoints._restore_verl_resume(
            str(tmp_path / "local"), world_size=2, expected_fsdp_generation=2
        )

    with pytest.raises(RuntimeError, match="missing its inspection"):
        restore()
    assert staged == []


def _fake_hf_resume_checkpoint_over(*candidates):
    """Stand in for hf_resume_checkpoint's own selection, over an already-downloaded set of dirs.

    Mirrors ``_highest_resume_candidate``: highest step ``prefer`` accepts, else highest overall.
    Used to test that the GRPO/SFT resume paths wire a ``prefer`` that actually reaches a compatible
    checkpoint, not just that some checkpoint gets staged.
    """

    def _resume(*, prefer=None, **_kwargs):
        ordered = sorted(candidates, key=lambda p: -int(p.name.rsplit("-", 1)[1]))
        if prefer is not None:
            for path in ordered:
                if prefer(str(path)):
                    return str(path)
        return str(ordered[0]) if ordered else None

    return _resume


def _count_resume_inspections(monkeypatch):
    from flash.engine.worker.verl import checkpoints as verl_checkpoints

    original = verl_checkpoints.inspect_resume_checkpoint
    counts = {}

    def inspect(path, **kwargs):
        counts[path] = counts.get(path, 0) + 1
        return original(path, **kwargs)

    monkeypatch.setattr(verl_checkpoints, "inspect_resume_checkpoint", inspect)
    return counts


def test_grpo_resume_prefers_compatible_checkpoint_over_higher_incompatible_one(
    monkeypatch, tmp_path
):
    """A restart after checkpoint-7 was rejected must resume from a compatible checkpoint-3 streamed
    afterward, instead of re-fetching and re-discarding checkpoint-7 forever.

    Before the ``prefer`` wiring, ``hf_resume_checkpoint`` always picked the highest step regardless
    of whether this attempt could load it, so checkpoint-3 could never be reached while checkpoint-7
    stayed the remote max -- exactly the repeat-discard loop this fix closes.
    """

    counts = _count_resume_inspections(monkeypatch)
    incompatible = _staged_checkpoint(tmp_path, 7, world_size=4, shards=4)
    compatible = _staged_checkpoint(tmp_path, 3, world_size=2, shards=2)
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(
        worker_hf,
        "hf_resume_checkpoint",
        _fake_hf_resume_checkpoint_over(incompatible, compatible),
    )

    step = rl_checkpoints._restore_verl_resume(
        str(local_dir), world_size=2, expected_fsdp_generation=2
    )

    assert step == 3
    assert counts == {str(incompatible): 1, str(compatible): 1}
    assert (local_dir / "global_step_3" / "actor" / "model.safetensors").exists()
    assert not (local_dir / "global_step_7").exists()


def test_grpo_resume_skips_a_higher_loadable_checkpoint_with_a_malformed_manifest(
    monkeypatch, tmp_path
):
    counts = _count_resume_inspections(monkeypatch)
    malformed = _staged_checkpoint(tmp_path, 7, world_size=2, shards=2)
    valid = _staged_checkpoint(tmp_path, 3, world_size=2, shards=2)
    (malformed / "_flash_resume_manifest.json").write_text("{not-json")
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(
        worker_hf,
        "hf_resume_checkpoint",
        _fake_hf_resume_checkpoint_over(malformed, valid),
    )

    assert (
        rl_checkpoints._restore_verl_resume(
            str(local_dir), world_size=2, expected_fsdp_generation=2
        )
        == 3
    )
    assert counts == {str(malformed): 1, str(valid): 1}
    assert (local_dir / "global_step_3" / "actor" / "model.safetensors").exists()
    assert not (local_dir / "global_step_7").exists()


def test_sft_resume_prefers_compatible_checkpoint_over_higher_incompatible_one(
    monkeypatch, tmp_path
):
    """Same repeat-discard-loop fix as the GRPO case, for SFT's flat (non-nested) checkpoint layout."""
    from flash.engine.worker.train.entry import sft_train

    counts = _count_resume_inspections(monkeypatch)
    incompatible = _staged_checkpoint(tmp_path, 9, world_size=4, shards=4, nested=False)
    compatible = _staged_checkpoint(tmp_path, 5, world_size=2, shards=2, nested=False)
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(
        worker_hf,
        "hf_resume_checkpoint",
        _fake_hf_resume_checkpoint_over(incompatible, compatible),
    )

    assert (
        sft_train._restore_verl_resume(str(local_dir), world_size=2, expected_fsdp_generation=2)
        == 5
    )
    assert counts == {str(incompatible): 1, str(compatible): 1}
    assert (local_dir / "global_step_5" / "model.safetensors").exists()
    assert not (local_dir / "global_step_9").exists()


def test_opd_discards_a_mismatched_checkpoint_with_its_accounting(monkeypatch, tmp_path):
    """OPD returns the fresh-run answer rather than half-restoring the loop state.

    The accounting in opd_state.json describes steps a discarded resume simply redoes, so the pair
    has to move together; reading it back beside shards this attempt cannot load would be worse.
    """

    src = _staged_checkpoint(tmp_path, 2, world_size=4, shards=4)
    (src / "opd_state.json").write_text(json.dumps({"unreadable": True}))
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(worker_state, "OPD_RESUME_REVISION", "")
    monkeypatch.setattr(worker_hf, "hf_resume_checkpoint", lambda **_k: str(src))

    step, state = opd_failures._restore_verl_resume(
        str(local_dir),
        prompt_pool_fingerprint="a" * 64,
        update_horizon=8,
        world_size=1,
        expected_fsdp_generation=2,
    )

    assert (step, state) == (0, None)
    assert not (local_dir / "global_step_2").exists()


def test_opd_pinned_revision_topology_mismatch_fails_closed(monkeypatch, tmp_path):
    """A pinned OPD_RESUME_REVISION means an optimizer step already crossed and the replacement
    gate approved continuation from exactly that checkpoint; discarding it and restarting from
    step 0 would repeat billed teacher work, so a world-size mismatch must raise, not restart."""

    src = _staged_checkpoint(tmp_path, 2, world_size=4, shards=4)
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(worker_state, "OPD_RESUME_REVISION", "pinned-sha")
    monkeypatch.setattr(worker_hf, "hf_resume_checkpoint", lambda **_k: str(src))

    with pytest.raises(RuntimeError, match=r"permanent OPD resume failure.*'pinned-sha'"):
        opd_failures._restore_verl_resume(
            str(local_dir),
            prompt_pool_fingerprint="a" * 64,
            update_horizon=8,
            world_size=1,
            expected_fsdp_generation=2,
        )
    assert not (local_dir / "global_step_2").exists()


# ============================================================================================
# 3. control plane - _run_attempts_supervised relaunches the same run on infra-shaped failure
# ============================================================================================
def _spec(run_id="flash-1700000001-rt01", **gpu_kw) -> JobSpec:
    gpu = {"type": "RTX 4090", "max_retries": 2}
    gpu.update(gpu_kw)
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            # authoritative seed 0 matches the literal seed threaded into _run_attempts_supervised below.
            "seed": 0,
            "run_id": run_id,
            "train": {"epochs": 1, "max_examples": 1, "hf_repo": "owner/runs"},
            "gpu": gpu,
        }
    )
    # the relaunch path re-quotes from the spec it holds, and an sft quote reads the workload
    # profile. these tests start *after* preparation, from the spec a prepared run already carries,
    # so the profile is attached here rather than driven through submission: from_dict is the public
    # shape and deliberately cannot carry one.
    return attach_sft_profile(spec)


def _alloc():
    from flash.providers.core.base import Allocation, Candidate

    candidates = (
        Candidate("runpod", "RTX 4090", 0.69, 24),
        Candidate("runpod", "H100", 3.29, 80),
        Candidate("runpod", "B200", 5.89, 180),
    )
    return Allocation(
        provider="runpod",
        gpu="RTX 4090",
        hourly_usd=0.69,
        min_vram_gb=12,
        candidates=candidates,
    )


def _runpod_handle(endpoint_id="ep", job_id="j", attempt=0):
    return {
        "provider": "runpod",
        "endpoint_id": endpoint_id,
        "endpoint_name": f"{endpoint_id}-name",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "job_id": job_id,
        "attempt": attempt,
        "started_ts": float(attempt + 1),
    }


def _vast_handle(attempt=0):
    return {
        "provider": "vast",
        "instance_id": attempt + 1,
        "offer_id": 100 + attempt,
        "machine_id": 200 + attempt,
        "label": f"flash-retry-{attempt}",
        "gpu": "RTX 4090",
        "hourly_usd": 0.5,
        "attempt": attempt,
        "started_ts": float(attempt + 1),
    }


def _lambda_handle(attempt=0):
    return {
        "provider": "lambda",
        "instance_id": f"i-{attempt + 1}",
        "instance_type": "gpu_1x_a10",
        "region": "us-east-1",
        "name": f"flash-retry-{attempt}",
        "gpu": "A10",
        "hourly_usd": 1.29,
        "attempt": attempt,
        "started_ts": float(attempt + 1),
    }


@pytest.fixture
def orch(monkeypatch, tmp_path):
    """The runner package wired to a tmp run-store, with the inter-attempt RunPod teardown stubbed."""
    from flash.providers.core import allocator
    from flash.providers.runpod.client import api as runpod_api

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    # the spec carries a pinned model revision (the profile is keyed on one), which makes the
    # post-allocation quote refresh resolve revision-specific geometry from the hub. these tests are
    # about the retry loop, so read the catalog's numbers instead of the network.
    stub_revision_geometry(monkeypatch)
    # The retry loop tears the prior attempt's endpoint down before relaunching; keep it off-network.
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *a, **k: True)


def _seed_status(orch, spec):
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="queued",
            spec=spec.to_dict(),
            source_snapshot=_SOURCE_SNAPSHOT,
        )
    )


@pytest.mark.parametrize("failure", INFRA_SHAPED)
def test_infra_failure_relaunches_same_run_and_seed(orch, monkeypatch, failure):
    """Every infra-shaped failure (incl. preemption) retries on the SAME run_id + seed.

    The run_id + seed are exactly the key hf_prefix() resumes on, so relaunching them unchanged
    is what lets the replacement worker pick up the prior checkpoint. A changed run_id here would
    silently turn every "retry from last checkpoint" into a restart from scratch.
    """
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    calls = []

    def fake_submit(run_spec, log=None, on_handle=None, attempt=0, **_):
        calls.append((run_spec.run_id, run_spec.seed))
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if attempt == 0:
            return PollResult(False, failure=failure, detail="infra")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    metrics = runner_lifecycle._run_attempts_supervised(spec, log)

    assert metrics["train_tokens"] == 4096
    assert calls == [(spec.run_id, 0), (spec.run_id, 0)], "retry must reuse the same run_id + seed"
    assert "resume from last checkpoint" in log.getvalue()


def test_unconfirmed_instance_teardown_fails_terminal_and_reaps(orch, monkeypatch):
    """When an instance-provider destroy() RAISES (Vast's unconfirmed DELETE — the old box
    may STILL be running and writing this seed's HF artifacts), the retry must NOT launch a second
    worker over it (double-bill + corrupt the shared seed-scoped DONE/metrics). Force-reap the run's
    label (provider.gc, run-scoped / not active-shielded) and FAIL the seed terminally; the handle is
    preserved (not cleared) for the run's outer GC."""
    from flash.providers.core import allocator
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate, PollResult

    submits = []
    gc_calls = []
    vast_candidate = Candidate("vast", "RTX 4090", 0.5, 24)
    vast_larger = Candidate("vast", "H100", 1.0, 80)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *a, **k: Allocation("vast", "RTX 4090", 0.5, 12, (vast_candidate, vast_larger)),
    )

    class _RaisingVast:
        def submit_attempt(self, run_spec, log=None, on_handle=None, attempt=0, **_):
            submits.append(attempt)
            on_handle(_vast_handle(attempt))
            return PollResult(False, failure="stalled", detail="infra")

        def destroy(self, handle):
            from flash.providers.vast.client import api as vast_api

            raise vast_api.VastApiError("destroy unconfirmed (success:false)")

        def gc(self, spec):
            gc_calls.append(spec.run_id)

    vast_provider = _RaisingVast()
    real_get = providers.get_provider
    monkeypatch.setattr(
        providers,
        "get_provider",
        lambda name: vast_provider if name == "vast" else real_get(name),
    )

    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="teardown could not be confirmed"):
        runner_lifecycle._run_attempts_supervised(spec, log)

    assert submits == [0], "must NOT launch a second worker over a possibly-live box"
    assert gc_calls == [spec.run_id], "force-reap the run's label before failing terminally"
    assert "teardown unconfirmed" in log.getvalue()
    # handle preserved (not cleared) so the run's outer GC can still reach the box
    remote = runner_status.get_status(spec.run_id).remote
    assert remote is not None
    assert remote.get("instance_id") == 1


def test_unconfirmed_lambda_teardown_blocks_replacement_and_preserves_handle(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Allocation, Candidate, PollResult
    from flash.providers.lambda_ import jobs as lambda_jobs
    from flash.providers.lambda_.client import api as lambda_api

    submits = []
    gc_calls = []
    candidate = Candidate("lambda", "A10", 1.29, 24)
    larger = Candidate("lambda", "H100", 3.29, 80)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *a, **k: Allocation("lambda", "A10", 1.29, 12, (candidate, larger)),
    )

    def fake_submit(run_spec, log=None, on_handle=None, attempt=0, **_kwargs):
        submits.append(attempt)
        on_handle(_lambda_handle(attempt))
        return PollResult(False, failure="stalled", detail="infra")

    def unconfirmed(_instance_id):
        raise lambda_api.LambdaApiError("private provider termination response")

    monkeypatch.setattr(lambda_jobs, "submit_attempt_lambda", fake_submit)
    monkeypatch.setattr(lambda_api, "terminate_instance_confirmed", unconfirmed)
    monkeypatch.setattr(
        lambda_jobs,
        "terminate_run_instances",
        lambda run_id: gc_calls.append(run_id) or [],
    )

    spec = _spec(type="A10")
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="teardown could not be confirmed") as caught:
        runner_lifecycle._run_attempts_supervised(spec, log)

    assert submits == [0]
    assert gc_calls == [spec.run_id]
    assert "teardown unconfirmed" in log.getvalue()
    assert "private" not in log.getvalue()
    assert "private" not in str(caught.value)
    remote = runner_status.get_status(spec.run_id).remote
    assert remote is not None
    assert remote.get("instance_id") == "i-1"


@pytest.mark.parametrize("terminal_status", ["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"])
def test_terminal_runpod_job_allows_retry_and_persists_leaked_endpoint(
    orch, monkeypatch, terminal_status
):
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

    submits = []

    def fake_submit(run_spec, log=None, on_handle=None, attempt=0, **_):
        submits.append(attempt)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if attempt == 0:
            return PollResult(False, failure="stalled", detail="infra")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: {"status": terminal_status},
    )
    # a COMPLETED job whose output never lands makes _await_runpod_completed_metrics poll for the
    # bounded metrics-lag grace window before falling through to a retry. drive a fake clock that
    # advances on sleep so that ~120s window elapses in fake time instead of real seconds (and the
    # loop can never wall-clock hang the suite). the fail/cancel/timeout statuses are not terminal-ok,
    # so they never enter the pending poll and this is inert for them.
    import time as _time

    fake_clock = {"now": 1_000.0}
    monkeypatch.setattr(_time, "time", lambda: fake_clock["now"])
    monkeypatch.setattr(
        _time, "sleep", lambda seconds: fake_clock.__setitem__("now", fake_clock["now"] + seconds)
    )
    spec = _spec(run_id=f"flash-terminal-{terminal_status.lower()}")
    _seed_status(orch, spec)

    metrics = runner_lifecycle._run_attempts_supervised(spec, io.StringIO())

    assert metrics["train_tokens"] == 4096
    assert submits == [0, 1]
    raw = runner_status._load_status_json(spec.run_id)
    assert [item["endpoint_id"] for item in raw[runner_state._CLEANUP_REMOTES_KEY]] == [
        "ep0",
        "ep1",
    ]
    assert runner_status.get_status(spec.run_id).remote["endpoint_id"] == "ep1"


def test_await_runpod_completed_metrics_bounds_pending_poll_to_grace(monkeypatch):
    # a terminal-ok runpod job whose output never becomes readable must not pin the supervisor for
    # the remainder of a (default 24h) run wall deadline: the synchronous metrics-lag poll is bounded
    # to ~the grace window measured from first observation, not the run deadline that callers pass in.
    import time as _time

    import flash.runner.supervise.lifecycle as lifecycle

    fake_clock = {"now": 1_000.0}
    monkeypatch.setattr(_time, "time", lambda: fake_clock["now"])
    monkeypatch.setattr(
        _time, "sleep", lambda seconds: fake_clock.__setitem__("now", fake_clock["now"] + seconds)
    )
    probes = {"n": 0}

    def always_pending(_handle, *, deadline_at):
        # mirror _runpod_completed_metrics' grace decision against the deadline it is handed.
        probes["n"] += 1
        if (
            deadline_at is not None
            and _time.time() >= deadline_at + lifecycle._RECOVERY_MARKER_GRACE_S
        ):
            return  # grace expired: signal "no metrics" like _runpod_completed_metrics
        raise lifecycle._CompletedAttemptPending("output metrics not readable yet")

    monkeypatch.setattr(lifecycle, "_runpod_completed_metrics", always_pending)
    far_deadline = fake_clock["now"] + 24 * 3600.0

    result = lifecycle._await_runpod_completed_metrics(object(), far_deadline)

    assert result is None
    # bounded to ~grace/poll-interval probes and ~grace of elapsed time, not the 24h deadline.
    max_probes = lifecycle._RECOVERY_MARKER_GRACE_S / lifecycle._RECOVERY_METRICS_POLL_S + 2
    assert probes["n"] <= max_probes
    assert (
        fake_clock["now"] - 1_000.0
        <= lifecycle._RECOVERY_MARKER_GRACE_S + lifecycle._RECOVERY_METRICS_POLL_S
    )


def test_await_runpod_completed_metrics_checks_cancellation_before_sleep(monkeypatch):
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.runner.supervise.errors import _RunCancelled

    monkeypatch.setattr(
        lifecycle,
        "_runpod_completed_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            lifecycle._CompletedAttemptPending("output metrics not readable yet")
        ),
    )
    monkeypatch.setattr(
        lifecycle.time,
        "sleep",
        lambda _seconds: pytest.fail("cancelled pending wait must not sleep"),
    )
    checks = []

    def check_cancelled():
        checks.append(True)
        raise _RunCancelled("run was cancelled")

    with pytest.raises(_RunCancelled, match="was cancelled"):
        lifecycle._await_runpod_completed_metrics(
            object(),
            lifecycle.time.time() + 1_000.0,
            check_cancelled=check_cancelled,
        )

    assert checks == [True]


@pytest.mark.parametrize(
    ("teardown_failure", "status_mode"),
    [
        ("cancel", "in_progress"),
        ("delete_false", "in_progress"),
        ("delete_exception", "raises"),
    ],
)
def test_unconfirmed_runpod_teardown_blocks_replacement_and_preserves_handle(
    orch, monkeypatch, teardown_failure, status_mode
):
    import flash.providers.runpod.serverless.endpoints as runpod_train
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

    submits = []
    teardown_events = []

    def fake_submit(run_spec, log=None, on_handle=None, attempt=0, **_):
        submits.append(attempt)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        return PollResult(False, failure="stalled", detail="infra")

    def cancel_job(endpoint_id, job_id, **_kw):
        teardown_events.append(("cancel", endpoint_id, job_id))
        if teardown_failure == "cancel":
            raise RuntimeError("private cancellation response")
        return {"id": job_id, "status": "CANCELLED"}

    def delete_endpoint(endpoint_id, _fingerprint):
        teardown_events.append(("delete", endpoint_id))
        if teardown_failure == "delete_false":
            return False
        if teardown_failure == "delete_exception":
            raise RuntimeError("private deletion response")
        return teardown_failure != "cancel"

    def job_status(*_args, **_kwargs):
        if status_mode == "raises":
            raise RuntimeError("private status response")
        return {"status": "IN_PROGRESS"}

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)
    monkeypatch.setattr(runpod_api, "cancel_job", cancel_job)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", delete_endpoint)
    monkeypatch.setattr(runpod_api, "job_status", job_status)
    monkeypatch.setattr(runpod_train, "terminate_endpoint", lambda *_args, **_kwargs: None)

    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="teardown could not be confirmed") as caught:
        runner_lifecycle._run_attempts_supervised(spec, log)

    assert submits == [0]
    assert teardown_events[:2] == [("cancel", "ep0", "j0"), ("delete", "ep0")]
    assert "teardown unconfirmed" in log.getvalue()
    assert "private" not in log.getvalue()
    assert "private" not in str(caught.value)
    remote = runner_status.get_status(spec.run_id).remote
    assert remote is not None
    assert remote.get("endpoint_id") == "ep0"
    assert remote.get("job_id") == "j0"


def test_confirmed_teardown_clears_handle_so_next_retry_does_not_retear(orch, monkeypatch):
    """Codex: after a CONFIRMED teardown the in-memory last_handle must be cleared, not just the
    persisted remote. Otherwise a retry that fails BEFORE provisioning a new handle (allocation/search
    error -> on_handle never fires) leaves the prior, already-torn-down handle live, so the NEXT retry
    re-tears-down that already-gone resource. For an instance provider a transient destroy/list failure
    on that phantom re-teardown would mis-classify as the unconfirmed-teardown FATAL path though no
    prior worker remains; for RunPod it re-issues a redundant cancel/delete. Sequence: attempt 0
    provisions ep0 then fails infra; attempt 1 confirms ep0's teardown then fails no_capacity (no new
    handle); attempt 2 must NOT touch ep0 again (last_handle cleared) and must succeed."""
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

    # cancel_job records inter-attempt teardown and the final successful attempt's confirmed cleanup.
    cancels: list[str] = []
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda eid, jid, **_kw: cancels.append(eid) or {"id": jid, "status": "CANCELLED"},
    )

    def fake_submit(run_spec, log=None, on_handle=None, attempt=0, **_):
        if attempt == 0:
            on_handle(_runpod_handle("ep0", "j0"))  # provisioned, then lost infra-shaped
            return PollResult(False, failure="stalled", detail="infra")
        if attempt == 1:
            # Allocation/search-shaped failure BEFORE a new handle is recorded (on_handle never fires).
            # last_handle must already be EMPTY here (ep0's teardown was confirmed at the top of this
            # attempt), so the next retry has no stale handle to re-tear-down.
            return PollResult(False, failure="no_capacity", detail="search flaked")
        on_handle(_runpod_handle("ep2", "j2", 2))  # fresh endpoint on the final retry
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)

    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    metrics = runner_lifecycle._run_attempts_supervised(spec, log)

    assert metrics["train_tokens"] == 4096, "the run must reach the successful final retry"
    assert cancels == ["ep0", "ep2"]
    assert cancels.count("ep0") == 1, "confirmed cleanup must not re-tear the prior endpoint"


def test_worker_error_fails_fast_without_relaunch(orch, monkeypatch):
    """A genuine worker error (the run's own code crashed) must NOT consume a retry.

    Only infra-shaped failures resume; a real training crash would just reproduce on a fresh
    host, so it fails immediately instead of burning the budget on a doomed resume.

    Uses the REAL failure label a worker/code crash produces — ``job_failed`` (the providers emit
    ``"job_preempted" if retriable else "job_failed"``, e.g. runpod/jobs.py) — NOT a placeholder, so
    this actually guards the classification: if ``job_failed`` were ever added to the supervisor's
    infra-shaped retry set, this test would FAIL (it'd relaunch a doomed crash and burn GPU budget).
    """
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    calls = []

    def fake_submit(run_spec, log=None, on_handle=None, attempt=0, **_):
        calls.append(attempt)
        on_handle(_runpod_handle())
        return PollResult(False, failure="job_failed", detail="ValueError in reward_fn")

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._run_attempts_supervised(spec, log)

    assert calls == [0], "a non-infra failure must fail fast, not relaunch"
    assert "not retrying" in log.getvalue()


def test_unreconciled_create_fails_fast_without_relaunch(orch, monkeypatch):
    """An ambiguous, unreconciled non-idempotent create (Vast's ``PUT /asks`` 5xx with no
    instance adoptable) raises ``UnreconciledCreateError`` from submit. The supervisor must classify it
    TERMINAL (job_failed), NOT as the generic ``poll_error`` flake — retrying would rent a SECOND box
    while the phantom from this attempt may still surface and bill under the still-active run."""
    from flash.providers.core.base import UnreconciledCreateError
    from flash.providers.runpod.execution import job_execution as rp_jobs

    calls = []

    def fake_submit(run_spec, log=None, on_handle=None, attempt=0, **_):
        calls.append(attempt)
        raise UnreconciledCreateError("ambiguous vast create; aborting the offer walk")

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._run_attempts_supervised(spec, log)

    assert calls == [0], "an unreconciled create must fail fast, not relaunch (no double-provision)"
    assert "not retrying" in log.getvalue()


def test_a_retry_marks_where_the_previous_attempt_ends_in_the_log(orch, monkeypatch):
    """The run log is one append-only file, so a retry's output follows the dead attempt's traceback.

    `flash runs log` tails that file. Without a boundary line, an operator checking a run that is
    currently retrying healthily reads the OOM stack that ended the PREVIOUS attempt and concludes
    the run is failing. Each new attempt must announce itself and disown what is above it.
    """
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    def fake_submit(run_spec, log=None, on_handle=None, attempt=0, **_):
        if attempt == 0:
            print("Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA OOM", file=log)
            return PollResult(False, failure="stalled", detail="infra")
        print("worker: stage=rl_step attempt=1 step=1", file=log)
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    runner_lifecycle._run_attempts_supervised(spec, log)

    text = log.getvalue()
    marker = "---- attempt 1 starts here; everything above it is from earlier attempts ----"
    assert marker in text, "a retry must say which attempt the following bytes belong to"
    assert text.index("CUDA OOM") < text.index(marker), (
        "the marker must sit after the failure it disowns, or it cannot separate the two attempts"
    )
    assert text.index(marker) < text.index("worker: stage=rl_step attempt=1 step=1"), (
        "the replacement attempt's heartbeat must follow the boundary that assigns its provenance"
    )


def test_the_marker_does_not_claim_one_previous_attempt_after_two_failures(orch, monkeypatch):
    """From the second retry on, "everything above is attempt N-1" is simply false.

    Above attempt 2 sit attempts 0 AND 1. The marker's job is to tell the reader where the current
    attempt begins, which stays true however many failed before it -- so it must not name a single
    owner for the bytes above.
    """
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    def fake_submit(run_spec, log=None, on_handle=None, attempt=0, **_):
        if attempt < 2:
            print(f"attempt {attempt} output", file=log)
            return PollResult(False, failure="stalled", detail="infra")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    runner_lifecycle._run_attempts_supervised(spec, log)

    text = log.getvalue()
    assert "---- attempt 2 starts here" in text
    assert "is attempt 1 ----" not in text, (
        "attempt 0's output is also above attempt 2, so crediting attempt 1 alone is wrong"
    )


def test_a_single_attempt_run_gets_no_boundary_marker(orch, monkeypatch):
    """Attempt 0 has nothing above it to disown. A header on the common path is pure noise."""
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    monkeypatch.setattr(
        rp_jobs,
        "submit_attempt",
        lambda *a, **k: PollResult(True, metrics={"train_tokens": 4096}),
    )
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    runner_lifecycle._run_attempts_supervised(spec, log)

    assert "starts here; everything above" not in log.getvalue()
