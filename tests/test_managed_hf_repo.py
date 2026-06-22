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
            "environment": {"id": "owner/env"},
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
    monkeypatch.delenv("FLASH_ARTIFACT_NAMESPACE", raising=False)
    spec = _submit(monkeypatch, _spec())
    assert spec["train"]["hf_repo"] == "Freesolo-Co/flashrun-flash-managed-1"


def test_managed_hf_repo_overrides_user_value(monkeypatch):
    # Even if a legacy/old-client spec carries a user namespace, the control plane overrides it.
    monkeypatch.delenv("FLASH_ARTIFACT_NAMESPACE", raising=False)
    spec = _submit(monkeypatch, _spec(hf_repo="freesolo-founders/whatever"))
    assert spec["train"]["hf_repo"] == "Freesolo-Co/flashrun-flash-managed-1"


def test_managed_hf_repo_namespace_env_override(monkeypatch):
    # The operator can point artifact repos at a different owned namespace.
    monkeypatch.setenv("FLASH_ARTIFACT_NAMESPACE", "my-org")
    spec = _submit(monkeypatch, _spec())
    assert spec["train"]["hf_repo"] == "my-org/flashrun-flash-managed-1"
