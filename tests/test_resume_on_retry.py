"""Resume-from-latest-checkpoint on crash/preemption retry (CPU-only).

A crash- or preemption-killed run is relaunched on a fresh worker by the control-plane
retry loop, and that fresh worker continues training from the latest streamed checkpoint
rather than restarting from scratch. This file locks in the three load-bearing pieces of
that path, none of which were covered by ``test_checkpoints.py`` (which only exercises the
*deployable* per-step adapter snapshots, a separate accumulating path):

  1. WORKER STREAM — ``make_checkpoint_upload_callback`` streams each trainer save's FULL
     state (optimizer/scheduler/RNG kept, not stripped) to ``<prefix>/checkpoint/checkpoint-<N>``,
     pruned latest-only via ``delete_patterns`` so a replacement worker finds exactly one.
  2. WORKER RESUME — ``hf_resume_checkpoint`` pulls that stream and returns the highest-step
     checkpoint dir (what both SFT and GRPO hand to ``trainer.train(resume_from_checkpoint=...)``).
  3. CONTROL PLANE — ``_submit_seed_supervised`` relaunches on the SAME run_id + seed for every
     infra-shaped failure (``stalled``/``no_capacity``/``poll_error``/``job_preempted``), which is
     the key that lets (2) find (1)'s checkpoint; a genuine worker error fails fast instead.

All HF/provider/network boundaries are stubbed; nothing here touches a GPU or the network.
"""

from __future__ import annotations

import io
import os
import shutil
from types import SimpleNamespace

import pytest

from flash.spec import JobSpec

# Infra-shaped failure categories the retry loop resumes on (see lifecycle._submit_seed_supervised).
# Mirrors the literal tuple in the source; this test is the guard that the set doesn't silently drift.
INFRA_SHAPED = ("stalled", "no_capacity", "poll_error", "job_preempted")


# ============================================================================================
# 1. WORKER STREAM — make_checkpoint_upload_callback
# ============================================================================================
class _RecordingHfApi:
    def __init__(self):
        self.uploads: list[dict] = []

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)


class _SyncThread:
    """Stand-in for threading.Thread that runs the target inline on .start().

    The upload callback fires the HF push on a background daemon thread so the train loop never
    blocks on the network; running it synchronously makes the unit test deterministic (no join).
    """

    def __init__(self, target=None, daemon=None, **_):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def _prime_worker(monkeypatch, recorder, *, repo="org/test-runs", run="flash-resume-1"):
    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "HF_REPO", repo)
    monkeypatch.setattr(worker, "PHASE", "rl")
    monkeypatch.setattr(worker, "RUN_ID", run)
    monkeypatch.setattr(worker, "SEED", 0)
    monkeypatch.setattr(worker, "hf_api", lambda: recorder)
    monkeypatch.setattr(worker, "heartbeat", lambda *a, **k: None)
    # Run the daemon upload thread inline so the assert sees the push.
    monkeypatch.setattr(worker.threading, "Thread", _SyncThread)
    return worker


def _full_checkpoint(tmp_path, step):
    """A trainer checkpoint dir carrying the adapter AND optimizer/RNG trainer-state."""
    ckpt = tmp_path / f"checkpoint-{step}"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
    (ckpt / "optimizer.pt").write_bytes(b"opt")
    (ckpt / "rng_state.pth").write_bytes(b"rng")
    return ckpt


def test_upload_callback_streams_full_state_checkpoint_latest_only(tmp_path, monkeypatch):
    """on_save streams the resume checkpoint to <prefix>/checkpoint/checkpoint-<N>, pruned latest-only.

    This is the upload a preempted run resumes FROM, so it must (a) keep the trainer state
    (no ignore_patterns dropping optimizer.pt) and (b) prune older checkpoints in the same commit
    (delete_patterns) so hf_resume_checkpoint never has to disambiguate stale state.
    """
    pytest.importorskip("transformers")  # the callback subclasses transformers.TrainerCallback
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    _full_checkpoint(tmp_path, 60)

    cb = worker.make_checkpoint_upload_callback()
    cb.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=60),
        SimpleNamespace(),
    )

    streams = [u for u in rec.uploads if u["path_in_repo"].endswith("/checkpoint/checkpoint-60")]
    assert len(streams) == 1, "the resumable full-state checkpoint must be streamed exactly once"
    up = streams[0]
    assert up["path_in_repo"] == "rl/flash-resume-1/seed0/checkpoint/checkpoint-60"
    assert up["repo_type"] == "dataset"
    # Latest-only: the prior checkpoint is pruned in the SAME commit so resume sees just one.
    assert up["delete_patterns"] == ["rl/flash-resume-1/seed0/checkpoint/**"]
    # FULL state: this upload must NOT strip optimizer/RNG — that's what makes the resume true
    # (Adam moments + LR schedule + RNG continue) rather than re-initializing the optimizer.
    assert "ignore_patterns" not in up


def test_upload_callback_also_publishes_deployable_snapshot(tmp_path, monkeypatch):
    """Each save additionally mirrors an adapter-only, NON-pruned per-step deployable snapshot.

    The two paths are distinct: the resume stream (above) is full-state + latest-only; the
    deployable snapshot is adapter-only + accumulating. on_save drives both off one checkpoint.
    """
    pytest.importorskip("transformers")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    _full_checkpoint(tmp_path, 60)

    worker.make_checkpoint_upload_callback().on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=60),
        SimpleNamespace(),
    )

    deployable = [u for u in rec.uploads if u["path_in_repo"].endswith("/checkpoints/step-60/adapter")]
    assert len(deployable) == 1
    up = deployable[0]
    # Adapter-only (trainer state stripped) and accumulating (never pruned).
    assert "optimizer.pt" in up["ignore_patterns"]
    assert "delete_patterns" not in up


def test_upload_callback_noop_without_repo(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, repo="")  # local/dev run, no artifact repo
    _full_checkpoint(tmp_path, 10)

    worker.make_checkpoint_upload_callback().on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=10),
        SimpleNamespace(),
    )
    assert rec.uploads == []


def test_upload_callback_skips_when_checkpoint_dir_missing(tmp_path, monkeypatch):
    """A save event whose checkpoint folder isn't on disk yet must not push a phantom commit."""
    pytest.importorskip("transformers")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    # No checkpoint-30 dir created under output_dir.
    worker.make_checkpoint_upload_callback().on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=30),
        SimpleNamespace(),
    )
    assert rec.uploads == []


# ============================================================================================
# 2. WORKER RESUME — hf_resume_checkpoint
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
    """hf_resume_checkpoint downloads to the worker's hardcoded /tmp/resume; reap our subtree."""
    run = "flash-resume-utest"
    yield run
    shutil.rmtree(os.path.join("/tmp/resume", "rl", run), ignore_errors=True)


def test_hf_resume_checkpoint_returns_latest_step(monkeypatch, resume_run_id):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, run=resume_run_id)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot([40, 80, 60]))

    path = worker.hf_resume_checkpoint()

    assert path is not None
    assert path.endswith("checkpoint-80"), "must resume from the highest streamed step, not the first"
    assert os.path.isdir(path)


def test_hf_resume_checkpoint_none_without_repo(monkeypatch):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec, repo="")
    # No repo -> short-circuit to None; snapshot_download must never be reached.
    assert worker.hf_resume_checkpoint() is None


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


# ============================================================================================
# 3. CONTROL PLANE — _submit_seed_supervised relaunches the same run on infra-shaped failure
# ============================================================================================
def _spec(run_id="flash-1700000001-rt01", **gpu_kw) -> JobSpec:
    gpu = {"type": "RTX A6000", "max_retries": 2}
    gpu.update(gpu_kw)
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "run_id": run_id,
            "train": {"epochs": 1, "seeds": [0], "hf_repo": "owner/runs"},
            "gpu": gpu,
        }
    )


def _alloc():
    from flash.providers.base import Allocation, Candidate

    candidates = (Candidate("runpod", "RTX A6000", 0.49, 48),)
    return Allocation(
        provider="runpod",
        gpu="RTX A6000",
        hourly_usd=0.49,
        min_vram_gb=12,
        candidates=candidates,
    )


def _runpod_handle(endpoint_id="ep", job_id="j"):
    return {
        "provider": "runpod",
        "endpoint_id": endpoint_id,
        "endpoint_name": f"{endpoint_id}-name",
        "job_id": job_id,
    }


@pytest.fixture
def orch(monkeypatch, tmp_path):
    """The runner package wired to a tmp run-store, with the inter-attempt RunPod teardown stubbed."""
    from flash import runner
    from flash.providers import allocator
    from flash.providers.runpod import api as runpod_api

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    # The retry loop tears the prior attempt's endpoint down before relaunching; keep it off-network.
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda *a, **k: True)
    return runner


def _seed_status(orch, spec):
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict()))


@pytest.mark.parametrize("failure", INFRA_SHAPED)
def test_infra_failure_relaunches_same_run_and_seed(orch, monkeypatch, failure):
    """Every infra-shaped failure (incl. preemption) retries on the SAME run_id + seed.

    The run_id + seed are exactly the key hf_prefix() resumes on, so relaunching them unchanged
    is what lets the replacement worker pick up the prior checkpoint. A changed run_id here would
    silently turn every "retry from last checkpoint" into a restart from scratch.
    """
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    calls = []

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_):
        calls.append((run_spec.run_id, seed))
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}"))
        if attempt == 0:
            return PollResult(False, failure=failure, detail="infra")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    metrics = orch._submit_seed_supervised(spec, 0, log)

    assert metrics["train_tokens"] == 4096
    assert calls == [(spec.run_id, 0), (spec.run_id, 0)], "retry must reuse the same run_id + seed"
    assert "resume from last checkpoint" in log.getvalue()


def test_worker_error_fails_fast_without_relaunch(orch, monkeypatch):
    """A genuine worker error (the run's own code crashed) must NOT consume a retry.

    Only infra-shaped failures resume; a real training crash would just reproduce on a fresh
    host, so it fails immediately instead of burning the budget on a doomed resume.
    """
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    calls = []

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_):
        calls.append(attempt)
        on_handle(_runpod_handle())
        return PollResult(False, failure="error", detail="ValueError in reward_fn")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="failed after retries"):
        orch._submit_seed_supervised(spec, 0, log)

    assert calls == [0], "a non-infra failure must fail fast, not relaunch"
    assert "not retrying" in log.getvalue()
