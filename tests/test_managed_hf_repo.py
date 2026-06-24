"""The run's HF artifact repo is platform-managed, not a user field.

``submit_job`` assigns it server-side — a per-run private dataset under the operator's
namespace — after the run_id is finalized, overwriting any inbound value. This is what
prevents the 403: the operator HF_TOKEN (which the control plane uploads/writes with)
cannot create a dataset under a user-chosen namespace like ``freesolo-founders``.
"""

from __future__ import annotations

import os
import tempfile

from flash.spec import JobSpec


def _spec(**train) -> JobSpec:
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "seeds": [0], **train},
            "run_id": "flash-managed-1",
        }
    )


def _submit(monkeypatch, spec: JobSpec) -> dict:
    from flash import runner

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        return runner.submit_job(spec, dry_run=True).spec


def test_managed_hf_repo_assigned_per_run(monkeypatch):
    spec = _submit(monkeypatch, _spec())
    assert spec["train"]["hf_repo"] == "Freesolo-Co/flashrun-flash-managed-1"


def test_managed_hf_repo_overrides_user_value(monkeypatch):
    # Even if a legacy/old-client spec carries a user namespace, the control plane overrides it.
    spec = _submit(monkeypatch, _spec(hf_repo="freesolo-founders/whatever"))
    assert spec["train"]["hf_repo"] == "Freesolo-Co/flashrun-flash-managed-1"


def test_managed_hf_repo_finalizes_local_run_id(monkeypatch):
    # The JobSpec default run_id "local" is treated as unset: submit_job assigns a real run_id and
    # a matching per-run repo, so default-constructed/programmatic specs never collide on
    # "flashrun-local". (Regression guard for the run_id-finalization review fix.)
    from flash.spec import JobSpec

    base = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "seeds": [0]},
        }
    )
    assert base.run_id == "local"
    spec = _submit(monkeypatch, base)
    assert spec["run_id"] != "local"
    assert spec["train"]["hf_repo"] == f"Freesolo-Co/flashrun-{spec['run_id']}"
