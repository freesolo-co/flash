"""Checkpoint discovery must name complete artifacts at their exact canonical path."""

import pytest

from flash.core.spec import JobSpec
from flash.runner.results.checkpoints import _checkpoints_from_files
from flash.schema import parse_checkpoint_ref


@pytest.mark.parametrize("prefix", ["backup/sft/audit-run", "other-sft/audit-run", "sft/other-run"])
def test_checkpoint_listing_does_not_adopt_other_artifact_roots(prefix):
    files = [
        f"{prefix}/checkpoints/step-10/adapter/{name}"
        for name in ("adapter_config.json", "adapter_model.safetensors")
    ]
    assert _checkpoints_from_files(JobSpec(run_id="audit-run"), files) == []


@pytest.mark.parametrize("step", ["01", "1\u0661", "9999999999999999999"])
def test_checkpoint_listing_ignores_noncanonical_step_paths(step):
    files = [
        f"sft/audit-run/checkpoints/step-{step}/adapter/{name}"
        for name in ("adapter_config.json", "adapter_model.safetensors")
    ]
    assert _checkpoints_from_files(JobSpec(run_id="audit-run"), files) == []


def test_checkpoint_listing_does_not_merge_files_from_different_step_spellings():
    files = [
        "sft/audit-run/checkpoints/step-01/adapter/adapter_config.json",
        "sft/audit-run/checkpoints/step-1/adapter/adapter_model.safetensors",
    ]
    assert _checkpoints_from_files(JobSpec(run_id="audit-run"), files) == []


@pytest.mark.parametrize("step", [0, 1, 999999999999999999])
def test_checkpoint_listing_retains_complete_canonical_paths(step):
    prefix = f"sft/audit-run/checkpoints/step-{step}"
    files = [
        f"{prefix}/adapter/{name}" for name in ("adapter_config.json", "adapter_model.safetensors")
    ]
    found = _checkpoints_from_files(JobSpec(run_id="audit-run"), files)
    assert len(found) == 1
    assert found[0]["step"] == step
    assert found[0]["adapter_prefix"] == prefix


def test_checkpoint_identity_rejects_non_ascii_step_digits():
    assert parse_checkpoint_ref("audit-run/step-1\u0661") is None
