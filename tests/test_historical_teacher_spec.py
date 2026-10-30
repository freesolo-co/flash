"""Saved SFT/GRPO records from before teacher removal retain checkpoint access."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from flash.core.spec import JobSpec
from flash.runner.lifecycle.preparation import _preparation_digest
from flash.runner.lifecycle.state import RunStatus, _internal_spec_from_status
from flash.runner.lifecycle.status import effective_spec_from_status
from flash.runner.lifecycle.submit import _effective_preparation_snapshot
from flash.schema import ConfigError, spec_from_dict


def historical_status(algorithm):
    # Generated with the pre-removal code at 486dc58f; hashes are not produced by
    # the current implementation, so dropping a legacy key cannot silently pass.
    path = Path(__file__).parent / "fixtures" / f"pre_teacher_removal_{algorithm}.json"
    raw = json.loads(path.read_text())
    raw.pop("generated_by")
    return RunStatus(**raw)


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_old_checkpoint_metadata_loads_without_changing_persisted_record(algorithm):
    status = historical_status(algorithm)
    original = copy.deepcopy(status)
    worker = effective_spec_from_status(status)
    assert worker.algorithm == algorithm
    assert worker.train.hf_repo == "fixtures/historical-checkpoints"
    assert worker.model_revision == "a" * 40
    assert _internal_spec_from_status(status) == worker
    assert "teacher_model" not in worker.to_internal_dict()["train"]
    assert "teacher_model" not in worker.to_dict()["train"]
    assert status == original


@pytest.mark.parametrize("half", ["public", "worker"])
@pytest.mark.parametrize("change", ["omit_teacher", "revision", "unknown_key"])
def test_legacy_digest_and_strict_schema_still_detect_tampering(half, change):
    status = historical_status("sft")
    raw = status.spec if half == "public" else status.effective_preparation["worker_spec"]
    if change == "omit_teacher":
        raw["train"].pop("teacher_model")
    elif change == "revision":
        raw["model_revision"] = "b" * 40
    else:
        raw["train"]["unknown_key"] = True
    with pytest.raises(ValueError, match=r"integrity validation|platform-managed|unknown key"):
        effective_spec_from_status(status)


@pytest.mark.parametrize("value", ["glm-5.3", None, False, 0, {}, []])
def test_only_the_exact_retired_default_is_accepted(value):
    status = historical_status("sft")
    status.spec["train"]["teacher_model"] = value
    with pytest.raises(ValueError, match="teacher_model"):
        effective_spec_from_status(status)


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_snapshot_rewrite_preserves_old_public_metadata_and_remains_verifiable(algorithm):
    status = historical_status(algorithm)
    worker = effective_spec_from_status(status)
    public = JobSpec.from_dict(status.spec)
    status.effective_preparation = _effective_preparation_snapshot(
        public, worker, None, stored_public=status.spec
    )
    assert "teacher_model" not in status.effective_preparation["worker_spec"]["train"]
    assert effective_spec_from_status(status) == worker
    # New runs never write the retired setting or change their digest shape.
    fresh = _effective_preparation_snapshot(public, worker, None)
    assert fresh["preparation_digest"] == _preparation_digest(public, worker, None)
    fresh_status = replace(status, spec=public.to_dict(), effective_preparation=fresh)
    assert effective_spec_from_status(fresh_status) == worker


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_new_authored_specs_cannot_set_even_an_empty_teacher(algorithm):
    with pytest.raises(ConfigError, match="teacher_model"):
        spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": algorithm,
                "environment": {"id": "org/project/env"},
                "train": {"teacher_model": ""},
            }
        )
