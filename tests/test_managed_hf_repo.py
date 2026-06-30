"""The run's HF artifact repo is platform-managed, not a user field.

``submit_job`` assigns it server-side — a private dataset under the operator's namespace,
scoped by environment id — after the run_id is finalized, overwriting any inbound value.
This is what prevents the 403: the operator HF_TOKEN (which the control plane uploads/writes
with) cannot create a dataset under a user-chosen namespace like ``freesolo-founders``.
"""

from __future__ import annotations

import os
import re
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


def test_managed_hf_repo_assigned_per_environment(monkeypatch):
    from flash import runner

    spec = _submit(monkeypatch, _spec())
    assert spec["train"]["hf_repo"] == runner.managed_hf_repo_for_environment(spec["environment"]["id"])


def test_managed_hf_repo_overrides_user_value(monkeypatch):
    from flash import runner

    # Even if a legacy/old-client spec carries a user namespace, the control plane overrides it.
    spec = _submit(monkeypatch, _spec(hf_repo="freesolo-founders/whatever"))
    assert spec["train"]["hf_repo"] == runner.managed_hf_repo_for_environment(spec["environment"]["id"])


def test_managed_hf_repo_finalizes_local_run_id(monkeypatch):
    # The JobSpec default run_id "local" is treated as unset: submit_job assigns a real run_id and
    # still assigns the environment-scoped managed repo instead of trusting a user value.
    from flash import runner
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
    assert spec["train"]["hf_repo"] == runner.managed_hf_repo_for_environment(spec["environment"]["id"])


def test_managed_hf_repo_reuses_repo_for_same_environment():
    from flash import runner

    env_id = "github:owner/repo@main:env/environment.py"
    assert runner.managed_hf_repo_for_environment(env_id) == runner.managed_hf_repo_for_environment(env_id)
    assert runner.managed_hf_repo_for_environment(env_id) != runner.managed_hf_repo_for_environment(
        "github:owner/repo@main:env/other.py"
    )


def test_flash_code_prefix_is_content_addressed():
    from flash import runner

    assert re.fullmatch(r"code/[0-9a-f]{32}/flash", runner.flash_code_prefix())
