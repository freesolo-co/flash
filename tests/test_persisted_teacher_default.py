"""Recover pre-OPD-removal SFT records without restoring managed-teacher support."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flash.core.spec import JobSpec
from flash.runner.lifecycle.state import RunStatus
from flash.runner.lifecycle.status import effective_spec_from_status
from flash.runner.lifecycle.submit import _effective_preparation_snapshot
from flash.schema import ConfigError, spec_from_dict


def legacy_status() -> RunStatus:
    # Written by 486dc58fa0530baefac1d276d92779e9a1cf48b3, before OPD removal.
    # Keep its original digest: regenerating it with today's writer would hide byte drift.
    path = Path(__file__).parent / "fixtures/persisted_sft_empty_teacher.json"
    return RunStatus(**json.loads(path.read_text()))


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_only_persisted_empty_teacher_default_is_readable(algorithm):
    raw = JobSpec.from_dict({"algorithm": algorithm}).to_internal_dict()
    raw["train"]["teacher_model"] = ""
    original = copy.deepcopy(raw)

    restored = JobSpec.from_dict(raw)

    assert restored.algorithm == algorithm
    assert "teacher_model" not in restored.to_dict()["train"]
    assert "teacher_model" not in restored.to_internal_dict()["train"]
    assert raw == original
    with pytest.raises(ConfigError, match="teacher_model"):
        spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": algorithm,
                "environment": {"id": "org/project/env"},
                "train": {"teacher_model": ""},
            }
        )


@pytest.mark.parametrize("value", ["teacher/model", " ", None, False, 0, [], {}])
def test_nonempty_or_malformed_retired_teacher_is_not_silently_discarded(value):
    with pytest.raises(ValueError, match="teacher_model"):
        JobSpec.from_dict({"algorithm": "sft", "train": {"teacher_model": value}})


def test_retired_default_does_not_allow_other_unknown_fields():
    with pytest.raises(ValueError, match="removed_key"):
        JobSpec.from_dict({"train": {"teacher_model": "", "removed_key": True}})


def test_historical_preparation_digest_recovers_without_mutating_stored_record():
    status = legacy_status()
    original = copy.deepcopy(status)

    restored = effective_spec_from_status(status)

    assert restored.algorithm == "sft"
    assert restored.model_revision == "a" * 40
    assert restored.train.hf_repo == "org/legacy-adapters"
    assert restored.train.lora_alpha == 64
    assert status == original


@pytest.mark.parametrize("side", ["public", "worker"])
def test_removing_historical_default_still_breaks_the_original_digest(side):
    status = legacy_status()
    raw = status.spec if side == "public" else status.effective_preparation["worker_spec"]
    del raw["train"]["teacher_model"]
    with pytest.raises(ValueError, match="failed integrity validation"):
        effective_spec_from_status(status)


@pytest.mark.parametrize("field", ["model_revision", "hf_repo", "lora_alpha"])
def test_historical_digest_still_rejects_modified_model_or_adapter_identity(field):
    status = legacy_status()
    raw = status.effective_preparation["worker_spec"]
    if field == "model_revision":
        raw[field] = "c" * 40
    else:
        raw["train"][field] = 128 if field == "lora_alpha" else "other/adapters"
    if field == "lora_alpha":
        status.spec["train"][field] = 128
    with pytest.raises(ValueError, match="failed integrity validation"):
        effective_spec_from_status(status)


def test_rewritten_snapshot_keeps_reading_legacy_public_spec():
    status = legacy_status()
    worker = effective_spec_from_status(status)
    public = copy.deepcopy(status.spec)
    status.effective_preparation = _effective_preparation_snapshot(
        JobSpec.from_dict(status.spec), worker, None, stored_public=status.spec
    )

    assert "teacher_model" not in status.effective_preparation["worker_spec"]["train"]
    assert status.spec == public
    assert effective_spec_from_status(status) == worker
