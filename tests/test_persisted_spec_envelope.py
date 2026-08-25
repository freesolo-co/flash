"""regressions for strict persisted-spec decoding and current digest validation."""

from __future__ import annotations

from dataclasses import replace

import pytest

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
from flash.core.spec import JobSpec
from flash.core.spec_persistence import PREPARATION_ENVELOPE_VERSION


def _current_spec() -> JobSpec:
    return replace(
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
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
    from flash.runner.lifecycle.submit import _effective_preparation_snapshot

    spec = _current_spec()
    snapshot = _effective_preparation_snapshot(spec, spec, None)
    status = runner_state.RunStatus(
        state="running",
        run_id=spec.run_id,
        spec=spec.to_dict(),
        effective_preparation=snapshot,
    )

    assert runner_status.effective_spec_from_status(status) == spec

    snapshot["worker_spec"]["workload_profile_producer_version"] = "tampered"
    with pytest.raises(ValueError, match="failed integrity validation"):
        runner_status.effective_spec_from_status(status)


def test_current_envelope_version_is_validated_but_absence_still_recovers() -> None:
    """An ABSENT version is the pre-stamp shape and reads as version 1; a bad one still raises.

    the stamp landed in 1.2.59 (b144ed68, 2026-08-16), so runs prepared by an older build are
    still in flight. `reallocation_spec_from_status` is what the retry path calls, and
    `server/platform/runtime.py` marks the run `unrecoverable` when it raises -- so rejecting an
    unversioned snapshot retires a live run instead of retrying it. dev draws the line the same
    way (`snapshot.get("version", CURRENT_VERSION)`), and its own regression pins a bool rather
    than an absent key.
    """
    from flash.runner.lifecycle.submit import _effective_preparation_snapshot

    spec = _current_spec()
    snapshot = _effective_preparation_snapshot(spec, spec, None)
    assert snapshot["version"] == PREPARATION_ENVELOPE_VERSION

    status = runner_state.RunStatus(
        state="running",
        run_id=spec.run_id,
        spec=spec.to_dict(),
        effective_preparation={key: value for key, value in snapshot.items() if key != "version"},
    )
    assert runner_status.effective_spec_from_status(status) == spec

    # a PRESENT but malformed value is still rejected: absence is a known shape, a bool is not.
    status.effective_preparation = {**snapshot, "version": True}
    with pytest.raises(ValueError, match="version must be a positive integer"):
        runner_status.effective_spec_from_status(status)

    status.effective_preparation = {**snapshot, "version": PREPARATION_ENVELOPE_VERSION + 1}
    with pytest.raises(ValueError, match="unsupported persisted preparation envelope version"):
        runner_status.effective_spec_from_status(status)
