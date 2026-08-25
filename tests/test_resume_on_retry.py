"""Resume-from-latest-checkpoint on crash/preemption retry (CPU-only).

A crash- or preemption-killed run is relaunched on a fresh worker by the control-plane
retry loop, and that fresh worker continues training from the latest streamed checkpoint
rather than restarting from scratch. This file locks in the two live load-bearing pieces:

  1. worker resume: ``hf_resume_checkpoint`` pulls the stream and returns the highest-step
     checkpoint directory used by the training backend.
  2. control plane: ``_submit_seed_supervised`` relaunches on the same run id and seed for every
     infra-shaped failure; a genuine worker error fails fast instead.

All HF/provider/network boundaries are stubbed; nothing here touches a GPU or the network.
"""

from __future__ import annotations

import io
import json
import os
import shutil

import pytest

from flash.core.spec import JobSpec
from tests._helpers.profile import attach_sft_profile, stub_revision_geometry
from tests._helpers.source_snapshot import valid_source_snapshot

# Infra-shaped failure categories the retry loop resumes on (see lifecycle._submit_seed_supervised).
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
    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "HF_REPO", repo)
    monkeypatch.setattr(worker, "PHASE", "rl")
    monkeypatch.setattr(worker, "RUN_ID", run)
    monkeypatch.setattr(worker, "SEED", 0)
    monkeypatch.setattr(worker, "hf_api", lambda: recorder)
    monkeypatch.setattr(worker, "heartbeat", lambda *a, **k: None)
    return worker


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
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([40, 80, 60]))

    path = worker.hf_resume_checkpoint()

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
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    seen = {}

    def snapshot(**kwargs):
        seen.update(kwargs)
        base = os.path.join(kwargs["local_dir"], "rl", resume_run_id, "checkpoint")
        for name in ("checkpoint-40", "checkpoint-latest", "checkpoint-80"):
            os.makedirs(os.path.join(base, name), exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    path = worker.hf_resume_checkpoint(fail_closed=True, revision="pinned-sha")

    assert seen["revision"] == "pinned-sha"
    assert path is not None
    assert path.endswith("checkpoint-80")


@pytest.mark.parametrize("name", ["checkpoint-0", "checkpoint-040"])
def test_hf_resume_checkpoint_pinned_rejects_noncanonical_only(monkeypatch, resume_run_id, name):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)

    def snapshot(**kwargs):
        base = os.path.join(kwargs["local_dir"], "rl", resume_run_id, "checkpoint", name)
        os.makedirs(base, exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    with pytest.raises(RetriableInfraError, match="required resume checkpoint is missing"):
        worker.hf_resume_checkpoint(fail_closed=True, revision="pinned-sha")


def test_hf_resume_checkpoint_selects_canonical_alongside_noncanonical(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)

    def snapshot(**kwargs):
        base = os.path.join(kwargs["local_dir"], "rl", resume_run_id, "checkpoint")
        for name in ("checkpoint-0", "checkpoint-040", "checkpoint-40"):
            os.makedirs(os.path.join(base, name), exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    path = worker.hf_resume_checkpoint(fail_closed=True, revision="pinned-sha")

    assert path is not None
    assert path.endswith("checkpoint-40")


def test_hf_resume_checkpoint_none_without_repo(monkeypatch):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, repo="")
    # no repo is a legitimate fresh start only when the control plane did not require resume.
    assert worker.hf_resume_checkpoint() is None


def test_hf_resume_checkpoint_required_without_repo_raises(monkeypatch):
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, repo="")
    with pytest.raises(RetriableInfraError, match="has no artifact repository"):
        worker.hf_resume_checkpoint(revision="pinned-sha")


def test_hf_resume_checkpoint_none_when_nothing_streamed(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    # A run preempted before its first save has no streamed checkpoint -> fresh start, not a crash.
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([]))
    assert worker.hf_resume_checkpoint() is None


def test_hf_resume_checkpoint_swallows_download_error(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)

    def _boom(**_):
        raise RuntimeError("hf down")

    # A transient HF outage must degrade to "start fresh", never abort the paid run.
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)
    assert worker.hf_resume_checkpoint() is None


def test_hf_resume_checkpoint_starts_no_download_at_deadline(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(worker, "_remaining_worker_wall_seconds", lambda: 0.0)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: pytest.fail("snapshot download must not start at the deadline"),
    )

    assert worker.hf_resume_checkpoint() is None


def test_hf_resume_checkpoint_fail_closed_on_download_error(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)

    def _boom(**_):
        raise RuntimeError("hf down")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)
    with pytest.raises(RetriableInfraError, match="required resume checkpoint fetch failed"):
        worker.hf_resume_checkpoint(fail_closed=True)


def test_hf_resume_checkpoint_fail_closed_at_deadline(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(worker, "_remaining_worker_wall_seconds", lambda: 0.0)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: pytest.fail("snapshot download must not start at the deadline"),
    )

    with pytest.raises(RetriableInfraError, match="required resume checkpoint fetch failed"):
        worker.hf_resume_checkpoint(fail_closed=True)


def test_hf_resume_checkpoint_fail_closed_allows_confirmed_absence(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([]))

    assert worker.hf_resume_checkpoint(fail_closed=True) is None


def test_hf_resume_checkpoint_pinned_missing_base_raises(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([]))

    with pytest.raises(RetriableInfraError, match="required resume checkpoint is missing"):
        worker.hf_resume_checkpoint(fail_closed=True, revision="pinned-sha")


def test_hf_resume_checkpoint_pinned_absence_clears_stale_local_state(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    stale = os.path.join("/tmp/resume", "rl", resume_run_id, "checkpoint", "checkpoint-80")
    os.makedirs(stale, exist_ok=True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([]))

    with pytest.raises(RetriableInfraError, match="required resume checkpoint is missing"):
        worker.hf_resume_checkpoint(revision="pinned-sha")

    assert not os.path.exists(stale)


def test_hf_resume_checkpoint_prefer_picks_highest_satisfying_candidate(monkeypatch, resume_run_id):
    """`prefer` steers the pick to the highest candidate it accepts, not the highest step overall,
    when a later, higher, unaccepted checkpoint also streamed -- the case that used to starve a
    compatible lower checkpoint forever once a higher incompatible one became the remote max."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([40, 60, 80]))

    path = worker.hf_resume_checkpoint(prefer=lambda p: not p.endswith("checkpoint-80"))

    assert path is not None
    assert path.endswith("checkpoint-60"), "must pick the highest candidate prefer accepts"


def test_hf_resume_checkpoint_prefer_falls_back_to_highest_overall(monkeypatch, resume_run_id):
    """when no candidate satisfies `prefer`, the selection falls back to the plain highest step --
    the same answer this returned before `prefer` existed, leaving the caller's own discard log as
    the thing that explains a restart from zero."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([40, 60, 80]))

    path = worker.hf_resume_checkpoint(prefer=lambda _p: False)

    assert path is not None
    assert path.endswith("checkpoint-80")


def test_hf_resume_checkpoint_pinned_without_numeric_dir_raises(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from flash.engine.worker.perf import RetriableInfraError

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)

    def snapshot(**kwargs):
        base = os.path.join(
            kwargs["local_dir"], "rl", resume_run_id, "checkpoint", "checkpoint-latest"
        )
        os.makedirs(base, exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    with pytest.raises(RetriableInfraError, match="required resume checkpoint is missing"):
        worker.hf_resume_checkpoint(fail_closed=True, revision="pinned-sha")


# ============================================================================================
# worker resume: fsdp world-size guard when a retry lands on a different card count
# ============================================================================================
def _staged_checkpoint(root, step, *, world_size=None, shards=0, nested=True):
    """Build a downloaded ``checkpoint-N`` shaped like the one verl uploaded.

    ``nested`` mirrors verl's two trainers: RL/OPD put the shards under ``actor/``, SFT writes them
    straight into the step dir. ``world_size`` writes verl's own ``fsdp_config.json`` stamp;
    ``shards`` writes per-rank shard files, which encode the same width in their names.
    """
    src = root / f"checkpoint-{step}"
    inner = src / "actor" if nested else src
    (inner / "huggingface").mkdir(parents=True)
    (inner / "model.safetensors").write_text("weights")
    if world_size is not None:
        (inner / "fsdp_config.json").write_text(
            json.dumps({"FSDP_version": 2, "world_size": world_size})
        )
    for rank in range(shards):
        (inner / f"model_world_size_{shards}_rank_{rank}.pt").write_bytes(b"shard")
    return src


def _rl_resume(monkeypatch, tmp_path, src, *, world_size):
    from flash.engine.worker import rl_train

    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(rl_train._w, "hf_resume_checkpoint", lambda *a, **k: str(src))
    return rl_train._restore_verl_resume(str(local_dir), world_size=world_size), local_dir


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
    assert "world size 4" in line
    assert "runs at 2" in line


def test_world_size_falls_back_to_the_shard_filenames(monkeypatch, tmp_path):
    # verl has always stamped fsdp_config.json, but the shard names carry the same width, so a
    # checkpoint that lost the stamp is still classified rather than guessed at.
    src = _staged_checkpoint(tmp_path, 3, shards=4)

    assert _rl_resume(monkeypatch, tmp_path, src, world_size=4)[0] == 3


@pytest.mark.parametrize("stamped_config", ["null", "[1, 2, 3]"])
def test_world_size_falls_back_when_fsdp_config_is_not_a_mapping(tmp_path, stamped_config):
    """valid json whose top-level value is null or a list is not verl's world_size stamp (a mapping
    with a "world_size" key). checkpoint_world_size must not raise on it -- it must fall back to the
    shard filenames exactly as it does for a genuinely unreadable file."""
    from flash.engine.worker.verl.checkpoints import checkpoint_world_size

    src = _staged_checkpoint(tmp_path, 11, shards=3)
    (src / "actor" / "fsdp_config.json").write_text(stamped_config)

    assert checkpoint_world_size(str(src)) == 3


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
    assert "world size unknown" in capsys.readouterr().out


def test_sft_flat_layout_is_guarded_too(monkeypatch, tmp_path, capsys):
    # SFT writes its shards into global_step_N itself rather than a nested actor/ dir.
    from flash.engine.worker import sft_train

    src = _staged_checkpoint(tmp_path, 9, world_size=2, shards=2, nested=False)
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(sft_train._w, "hf_resume_checkpoint", lambda *a, **k: str(src))

    assert sft_train._restore_verl_resume(str(local_dir), world_size=2) == 9
    shutil.rmtree(local_dir)
    local_dir.mkdir()
    assert sft_train._restore_verl_resume(str(local_dir), world_size=4) == 0
    assert not (local_dir / "global_step_9").exists()
    assert "[SFT] discarding resume checkpoint checkpoint-9" in capsys.readouterr().out


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


def test_grpo_resume_prefers_compatible_checkpoint_over_higher_incompatible_one(
    monkeypatch, tmp_path
):
    """A restart after checkpoint-7 was rejected must resume from a compatible checkpoint-3 streamed
    afterward, instead of re-fetching and re-discarding checkpoint-7 forever.

    Before the ``prefer`` wiring, ``hf_resume_checkpoint`` always picked the highest step regardless
    of whether this attempt could load it, so checkpoint-3 could never be reached while checkpoint-7
    stayed the remote max -- exactly the repeat-discard loop this fix closes.
    """
    from flash.engine.worker import rl_train

    incompatible = _staged_checkpoint(tmp_path, 7, world_size=4, shards=4)
    compatible = _staged_checkpoint(tmp_path, 3, world_size=2, shards=2)
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(
        rl_train._w,
        "hf_resume_checkpoint",
        _fake_hf_resume_checkpoint_over(incompatible, compatible),
    )

    step = rl_train._restore_verl_resume(str(local_dir), world_size=2)

    assert step == 3
    assert (local_dir / "global_step_3" / "actor" / "model.safetensors").exists()
    assert not (local_dir / "global_step_7").exists()


def test_sft_resume_prefers_compatible_checkpoint_over_higher_incompatible_one(
    monkeypatch, tmp_path
):
    """Same repeat-discard-loop fix as the GRPO case, for SFT's flat (non-nested) checkpoint layout."""
    from flash.engine.worker import sft_train

    incompatible = _staged_checkpoint(tmp_path, 9, world_size=4, shards=4, nested=False)
    compatible = _staged_checkpoint(tmp_path, 5, world_size=2, shards=2, nested=False)
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(
        sft_train._w,
        "hf_resume_checkpoint",
        _fake_hf_resume_checkpoint_over(incompatible, compatible),
    )

    assert sft_train._restore_verl_resume(str(local_dir), world_size=2) == 5
    assert (local_dir / "global_step_5" / "model.safetensors").exists()
    assert not (local_dir / "global_step_9").exists()


def test_opd_discards_a_mismatched_checkpoint_with_its_accounting(monkeypatch, tmp_path):
    """OPD returns the fresh-run answer rather than half-restoring the loop state.

    The accounting in opd_state.json describes steps a discarded resume simply redoes, so the pair
    has to move together; reading it back beside shards this attempt cannot load would be worse.
    """
    from flash.engine.worker import opd_train

    src = _staged_checkpoint(tmp_path, 2, world_size=4, shards=4)
    (src / "opd_state.json").write_text(json.dumps({"unreadable": True}))
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(opd_train._w, "OPD_RESUME_REVISION", "")
    monkeypatch.setattr(opd_train._w, "hf_resume_checkpoint", lambda **_k: str(src))

    step, state = opd_train._restore_verl_resume(
        str(local_dir), prompt_pool_fingerprint="a" * 64, update_horizon=8, world_size=1
    )

    assert (step, state) == (0, None)
    assert not (local_dir / "global_step_2").exists()


def test_opd_pinned_revision_topology_mismatch_fails_closed(monkeypatch, tmp_path):
    """A pinned OPD_RESUME_REVISION means an optimizer step already crossed and the replacement
    gate approved continuation from exactly that checkpoint; discarding it and restarting from
    step 0 would repeat billed teacher work, so a world-size mismatch must raise, not restart."""
    from flash.engine.worker import opd_train

    src = _staged_checkpoint(tmp_path, 2, world_size=4, shards=4)
    local_dir = tmp_path / "ckpt"
    local_dir.mkdir()
    monkeypatch.setattr(opd_train._w, "OPD_RESUME_REVISION", "pinned-sha")
    monkeypatch.setattr(opd_train._w, "hf_resume_checkpoint", lambda **_k: str(src))

    with pytest.raises(RuntimeError, match=r"permanent OPD resume failure.*'pinned-sha'"):
        opd_train._restore_verl_resume(
            str(local_dir), prompt_pool_fingerprint="a" * 64, update_horizon=8, world_size=1
        )
    assert not (local_dir / "global_step_2").exists()


# ============================================================================================
# 3. control plane - _submit_seed_supervised relaunches the same run on infra-shaped failure
# ============================================================================================
def _spec(run_id="flash-1700000001-rt01", **gpu_kw) -> JobSpec:
    gpu = {"type": "RTX 4090", "max_retries": 2}
    gpu.update(gpu_kw)
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            # authoritative seed 0 matches the literal seed threaded into _submit_seed_supervised below.
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
    from flash.providers.base import Allocation, Candidate

    candidates = (Candidate("runpod", "RTX 4090", 0.69, 24),)
    return Allocation(
        provider="runpod",
        gpu="RTX 4090",
        hourly_usd=0.69,
        min_vram_gb=12,
        candidates=candidates,
    )


def _runpod_handle(pod_id="pod-1", _unused="", attempt=0):
    return {
        "provider": "runpod",
        "instance_id": pod_id,
        "phase": "exact",
        "label": f"flash-retry-s0-a{attempt}-0123456789abcdef-deadbeef",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "account_id": "account-1",
        "payload_secret_id": f"secret-{attempt}",
        "payload_secret_name": "FLASH_PAYLOAD_0123456789abcdef",
        "data_center_id": "US-KS-2",
        "network_volume_id": None,
        "container_disk_gb": 120,
        "container_registry_auth_id": None,
        "gpu_count": 1,
        "image_name": None,
        "gpu_type_id_override": None,
        "allowed_cuda_versions": None,
        "docker_start_cmd": [],
        "gpu": "RTX 4090",
        "hourly_usd": 0.69,
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
    """Return the runner with a temporary store and offline RunPod teardown."""
    from flash import runner
    from flash.providers import allocator
    from flash.providers.runpod import PROVIDER as runpod_provider
    from flash.runner.supervise import lifecycle

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    # the spec carries a pinned model revision (the profile is keyed on one), which makes the
    # post-allocation quote refresh resolve revision-specific geometry from the hub. these tests are
    # about the retry loop, so read the catalog's numbers instead of the network.
    stub_revision_geometry(monkeypatch)
    # the retry loop tears the prior attempt's pod down and proves absence before relaunching.
    monkeypatch.setattr(runpod_provider, "destroy", lambda _handle: None)
    monkeypatch.setattr(runpod_provider, "run_instances_remaining", lambda _run_id: [])
    monkeypatch.setattr(lifecycle, "_await_runpod_completed_metrics", lambda *args, **kwargs: None)
    return runner


def _seed_status(orch, spec):
    orch._save_status(
        orch.RunStatus(
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
    from flash.providers.base import PollResult
    from flash.providers.runpod import PROVIDER as rp_jobs

    calls = []

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_):
        calls.append((run_spec.run_id, seed))
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if attempt == 0:
            return PollResult(False, failure=failure, detail="infra")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "_submit_run", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    metrics = orch._submit_seed_supervised(spec, 0, log)

    assert metrics["train_tokens"] == 4096
    assert calls == [(spec.run_id, 0), (spec.run_id, 0)], "retry must reuse the same run_id + seed"
    assert "resume from last checkpoint" in log.getvalue()


def test_unconfirmed_instance_teardown_fails_terminal_and_reaps(orch, monkeypatch):
    """When an instance-provider destroy() RAISES (Vast's unconfirmed DELETE — the old box
    may STILL be running and writing this seed's HF artifacts), the retry must NOT launch a second
    worker over it (double-bill + corrupt the shared seed-scoped DONE/metrics). Force-reap the run's
    label (provider.gc, run-scoped / not active-shielded) and FAIL the seed terminally; the handle is
    preserved (not cleared) for the run's outer GC."""
    import flash.providers as providers
    from flash.providers import allocator
    from flash.providers.base import Allocation, Candidate, PollResult

    submits = []
    gc_calls = []
    vast_candidate = Candidate("vast", "RTX 4090", 0.5, 24)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *a, **k: Allocation("vast", "RTX 4090", 0.5, 12, (vast_candidate,)),
    )

    class _RaisingVast:
        def submit_run(self, run_spec, seed, log=None, on_handle=None, attempt=0, **_):
            submits.append(attempt)
            on_handle(_vast_handle(attempt))
            return PollResult(False, failure="stalled", detail="infra")

        def destroy(self, handle):
            from flash.providers.vast import api as vast_api

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
        orch._submit_seed_supervised(spec, 0, log)

    assert submits == [0], "must NOT launch a second worker over a possibly-live box"
    assert gc_calls == [spec.run_id], "force-reap the run's label before failing terminally"
    assert "teardown unconfirmed" in log.getvalue()
    # handle preserved (not cleared) so the run's outer GC can still reach the box
    remote = orch.get_status(spec.run_id).remote
    assert remote is not None
    assert remote.get("instance_id") == 1


def test_unconfirmed_lambda_teardown_blocks_replacement_and_preserves_handle(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import Allocation, Candidate, PollResult
    from flash.providers.lambda_ import api as lambda_api
    from flash.providers.lambda_ import jobs as lambda_jobs

    submits = []
    gc_calls = []
    candidate = Candidate("lambda", "A10", 1.29, 24)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *a, **k: Allocation("lambda", "A10", 1.29, 12, (candidate,)),
    )

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_kwargs):
        submits.append(attempt)
        on_handle(_lambda_handle(attempt))
        return PollResult(False, failure="stalled", detail="infra")

    def unconfirmed(_instance_id):
        raise lambda_api.LambdaApiError("private provider termination response")

    monkeypatch.setattr(lambda_jobs, "submit_run_lambda", fake_submit)
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
        orch._submit_seed_supervised(spec, 0, log)

    assert submits == [0]
    assert gc_calls == [spec.run_id]
    assert "teardown unconfirmed" in log.getvalue()
    assert "private" not in log.getvalue()
    assert "private" not in str(caught.value)
    remote = orch.get_status(spec.run_id).remote
    assert remote is not None
    assert remote.get("instance_id") == "i-1"


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
    from flash.runner import _RunCancelled

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


def test_worker_error_fails_fast_without_relaunch(orch, monkeypatch):
    """a genuine worker error must fail immediately without consuming a retry.

    only infrastructure-shaped failures resume. the active providers emit ``job_failed`` for a
    non-retriable worker crash, so this test guards that label against accidental retry classification.
    """
    from flash.providers.base import PollResult
    from flash.providers.runpod import PROVIDER as rp_jobs

    calls = []

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_):
        calls.append(attempt)
        on_handle(_runpod_handle())
        return PollResult(False, failure="job_failed", detail="ValueError in reward_fn")

    monkeypatch.setattr(rp_jobs, "_submit_run", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="failed after retries"):
        orch._submit_seed_supervised(spec, 0, log)

    assert calls == [0], "a non-infra failure must fail fast, not relaunch"
    assert "not retrying" in log.getvalue()


def test_unreconciled_create_fails_fast_without_relaunch(orch, monkeypatch):
    """An ambiguous, unreconciled non-idempotent create (Vast's ``PUT /asks`` 5xx with no
    instance adoptable) raises ``UnreconciledCreateError`` from submit. The supervisor must classify it
    TERMINAL (job_failed), NOT as the generic ``poll_error`` flake — retrying would rent a SECOND box
    while the phantom from this attempt may still surface and bill under the still-active run."""
    from flash.providers.base import UnreconciledCreateError
    from flash.providers.runpod import PROVIDER as rp_jobs

    calls = []

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_):
        calls.append(attempt)
        raise UnreconciledCreateError("ambiguous vast create; aborting the offer walk")

    monkeypatch.setattr(rp_jobs, "_submit_run", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="failed after retries"):
        orch._submit_seed_supervised(spec, 0, log)

    assert calls == [0], "an unreconciled create must fail fast, not relaunch (no double-provision)"
    assert "not retrying" in log.getvalue()


def test_a_retry_marks_where_the_previous_attempt_ends_in_the_log(orch, monkeypatch):
    """The run log is one append-only file, so a retry's output follows the dead attempt's traceback.

    `flash runs log` tails that file. Without a boundary line, an operator checking a run that is
    currently retrying healthily reads the OOM stack that ended the PREVIOUS attempt and concludes
    the run is failing. Each new attempt must announce itself and disown what is above it.
    """
    from flash.providers.base import PollResult
    from flash.providers.runpod import PROVIDER as rp_jobs

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_):
        if attempt == 0:
            print("Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA OOM", file=log)
            return PollResult(False, failure="stalled", detail="infra")
        print("worker: stage=rl_step attempt=1 step=1", file=log)
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "_submit_run", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    orch._submit_seed_supervised(spec, 0, log)

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
    from flash.providers.base import PollResult
    from flash.providers.runpod import PROVIDER as rp_jobs

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_):
        if attempt < 2:
            print(f"attempt {attempt} output", file=log)
            return PollResult(False, failure="stalled", detail="infra")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "_submit_run", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    orch._submit_seed_supervised(spec, 0, log)

    text = log.getvalue()
    assert "---- attempt 2 starts here" in text
    assert "is attempt 1 ----" not in text, (
        "attempt 0's output is also above attempt 2, so crediting attempt 1 alone is wrong"
    )


def test_a_single_attempt_run_gets_no_boundary_marker(orch, monkeypatch):
    """Attempt 0 has nothing above it to disown. A header on the common path is pure noise."""
    from flash.providers.base import PollResult
    from flash.providers.runpod import PROVIDER as rp_jobs

    monkeypatch.setattr(
        rp_jobs,
        "_submit_run",
        lambda *a, **k: PollResult(True, metrics={"train_tokens": 4096}),
    )
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    orch._submit_seed_supervised(spec, 0, log)

    assert "starts here; everything above" not in log.getvalue()
