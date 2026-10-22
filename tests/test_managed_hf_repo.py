"""The run's HF artifact repo is platform-managed, not a user field.

``submit_job`` assigns it server-side — a private dataset under the operator's namespace,
scoped by environment and finalized run id — after the run_id is finalized, overwriting any inbound value.
This is what prevents the 403: the operator HF_TOKEN (which the control plane uploads/writes
with) cannot create a dataset under a user-chosen namespace like ``freesolo-founders``.
"""

from __future__ import annotations

import os
import tempfile

import flash.runner.accounting.artifacts as runner_artifacts
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.submit as runner_submit
from flash.core.spec import JobSpec
from tests._helpers.profile import satisfy_sft_profile


def _spec(**train) -> JobSpec:
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "max_examples": 8, **train},
            "run_id": "flash-managed-1",
        }
    )


def _submit_worker_spec(monkeypatch, spec: JobSpec) -> tuple[str, dict]:
    # hf_repo and run_id are platform-managed: they live in the internal worker spec + the
    # RunStatus.run_id field, NOT the public status.spec (which omits managed fields). Read the
    # managed assignment from the effective-preparation worker spec, which the worker executes.
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(tmp, "runs"))
        # sft submission is profile-gated; the repo assignment under test is not, so seed the
        # profile rather than exercising the hub round-trips a real submit performs.
        satisfy_sft_profile(monkeypatch, spec)
        status = runner_submit.submit_job(spec, dry_run=True)
        return status.run_id, status.effective_preparation["worker_spec"]


def test_managed_hf_repo_assigned_per_run(monkeypatch):
    _run_id, worker = _submit_worker_spec(monkeypatch, _spec())
    assert worker["train"]["hf_repo"] == runner_artifacts.managed_hf_repo_for_run(
        worker["environment"]["id"], worker["run_id"]
    )


def test_managed_hf_repo_overrides_user_value(monkeypatch):
    # Even if a legacy/old-client spec carries a user namespace, the control plane overrides it.
    _run_id, worker = _submit_worker_spec(monkeypatch, _spec(hf_repo="freesolo-founders/whatever"))
    assert worker["train"]["hf_repo"] == runner_artifacts.managed_hf_repo_for_run(
        worker["environment"]["id"], worker["run_id"]
    )


def test_managed_hf_repo_finalizes_local_run_id(monkeypatch):
    # The JobSpec default run_id "local" is treated as unset: submit_job assigns a real run_id and
    # still assigns the run-scoped managed repo instead of trusting a user value.
    from flash.core.spec import JobSpec

    base = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "max_examples": 8},
        }
    )
    assert base.run_id == "local"
    run_id, worker = _submit_worker_spec(monkeypatch, base)
    assert run_id != "local"
    assert worker["train"]["hf_repo"] == runner_artifacts.managed_hf_repo_for_run(
        worker["environment"]["id"], worker["run_id"]
    )


def test_managed_hf_repo_reuses_repo_for_same_environment():
    env_id = "github:owner/repo@main:env/environment.py"
    assert runner_artifacts.managed_hf_repo_for_environment(
        env_id
    ) == runner_artifacts.managed_hf_repo_for_environment(env_id)
    assert runner_artifacts.managed_hf_repo_for_environment(
        env_id
    ) != runner_artifacts.managed_hf_repo_for_environment("github:owner/repo@main:env/other.py")


def test_source_archive_path_is_content_addressed():
    from flash.snapshot.archive import canonical_archive_path

    digest = "a" * 64
    assert canonical_archive_path(digest) == f"source/{digest}/flash-source.zip"


def test_concurrent_runs_do_not_share_the_repository_commit_quota():
    from huggingface_hub.utils import validate_repo_id

    env = "owner/project/environment"
    one = runner_artifacts.managed_hf_repo_for_run(env, "flash-run-1")
    two = runner_artifacts.managed_hf_repo_for_run(env, "flash-run-2")
    assert one != two
    assert one == runner_artifacts.managed_hf_repo_for_run(env, "flash-run-1")
    assert one != runner_artifacts.managed_hf_repo_for_run("another/environment", "flash-run-1")
    assert one != runner_artifacts.managed_hf_repo_for_environment(env)
    validate_repo_id(one)
    validate_repo_id(runner_artifacts.managed_hf_repo_for_run(env, "long/" * 100))


def test_existing_preparation_keeps_legacy_repository_during_recovery(monkeypatch, tmp_path):
    from flash.runner.lifecycle.status import effective_spec_from_status

    spec = _spec()
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    satisfy_sft_profile(monkeypatch, spec)
    legacy = runner_artifacts.managed_hf_repo_for_environment(spec.environment.id)

    def legacy_assignment(job):
        raw = job.to_internal_dict()
        raw["train"]["hf_repo"] = legacy
        return JobSpec.from_dict(raw)

    monkeypatch.setattr(runner_artifacts, "_assign_managed_hf_repo", legacy_assignment)
    status = runner_submit.submit_job(spec, dry_run=True)

    def unexpected_assignment(_job):
        raise AssertionError("recovery must use its persisted artifact repository")

    monkeypatch.setattr(runner_artifacts, "_assign_managed_hf_repo", unexpected_assignment)
    assert effective_spec_from_status(status).train.hf_repo == legacy
