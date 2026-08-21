"""regressions for strict persisted-spec decoding and current digest validation."""

from __future__ import annotations

from dataclasses import replace

import pytest

import flash.runner as runner
from flash.core.spec import JobSpec
from flash.core.spec_persistence import PREPARATION_ENVELOPE_VERSION


def _current_spec() -> JobSpec:
    return replace(
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": "grpo",
                "run_id": "persisted-envelope",
                "project": "11111111-1111-4111-8111-111111111111",
                "environment": {},
                "gpu": {"type": "H100", "count": 1},
                "train": {"epochs": 1, "group_size": 4, "prompts_per_step": 32},
            }
        ),
        model_revision="a" * 40,
        model_revision_auto=True,
    )


def test_persisted_decoder_rejects_unknown_keys() -> None:
    spec = JobSpec().to_internal_dict()

    with pytest.raises(ValueError, match=r"job spec has unknown key.*removed_key"):
        JobSpec.from_dict({**spec, "removed_key": True})

    spec["train"]["removed_key"] = True
    with pytest.raises(ValueError, match=r"train has unknown key.*removed_key"):
        JobSpec.from_dict(spec)


def test_current_envelope_digest_round_trips_and_detects_tampering() -> None:
    from flash.runner.submit import _effective_preparation_snapshot

    spec = _current_spec()
    snapshot = _effective_preparation_snapshot(spec, spec, None)
    status = runner.RunStatus(
        state="running",
        run_id=spec.run_id,
        spec=spec.to_dict(),
        effective_preparation=snapshot,
    )

    assert runner.effective_spec_from_status(status) == spec

    snapshot["worker_spec"]["workload_profile_producer_version"] = "tampered"
    with pytest.raises(ValueError, match="failed integrity validation"):
        runner.effective_spec_from_status(status)


def test_current_envelope_version_is_required_and_validated() -> None:
    from flash.runner.submit import _effective_preparation_snapshot

    spec = _current_spec()
    snapshot = _effective_preparation_snapshot(spec, spec, None)
    assert snapshot["version"] == PREPARATION_ENVELOPE_VERSION

    status = runner.RunStatus(
        state="running",
        run_id=spec.run_id,
        spec=spec.to_dict(),
        effective_preparation={key: value for key, value in snapshot.items() if key != "version"},
    )
    with pytest.raises(ValueError, match="version must be a positive integer"):
        runner.effective_spec_from_status(status)

    status.effective_preparation = {**snapshot, "version": PREPARATION_ENVELOPE_VERSION + 1}
    with pytest.raises(ValueError, match="unsupported persisted preparation envelope version"):
        runner.effective_spec_from_status(status)
