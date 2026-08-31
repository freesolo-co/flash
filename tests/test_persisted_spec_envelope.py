"""regressions for strict persisted-spec decoding and current digest validation."""

from __future__ import annotations

from copy import deepcopy
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


def test_current_envelope_version_is_required_and_validated() -> None:
    """a present worker envelope must carry the exact supported version."""
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
    with pytest.raises(ValueError, match="preparation envelope version is required"):
        runner_status.effective_spec_from_status(status)

    status.effective_preparation = {**snapshot, "version": True}
    with pytest.raises(ValueError, match="version must be a positive integer"):
        runner_status.effective_spec_from_status(status)

    status.effective_preparation = {**snapshot, "version": PREPARATION_ENVELOPE_VERSION + 1}
    with pytest.raises(ValueError, match="unsupported persisted preparation envelope version"):
        runner_status.effective_spec_from_status(status)


@pytest.mark.parametrize("snapshot", [None, {}])
def test_public_fallback_requires_an_absent_worker_spec_key(snapshot) -> None:
    spec = _current_spec()
    status = runner_state.RunStatus(
        state="running",
        run_id=spec.run_id,
        spec=spec.to_dict(),
        effective_preparation=snapshot,
    )

    assert runner_state._internal_spec_from_status(status) == JobSpec.from_dict(status.spec)
    assert runner_status.effective_spec_from_status(status) == JobSpec.from_dict(status.spec)


@pytest.mark.parametrize(
    "worker_spec",
    [
        pytest.param(None, id="null"),
        pytest.param("worker", id="scalar"),
        pytest.param([], id="sequence"),
        pytest.param({"train": {"epochs": "1"}}, id="malformed-object"),
    ],
)
def test_present_worker_spec_never_falls_back_to_public(worker_spec) -> None:
    spec = _current_spec()
    status = runner_state.RunStatus(
        state="running",
        run_id=spec.run_id,
        spec=spec.to_dict(),
        effective_preparation={"worker_spec": worker_spec},
    )

    with pytest.raises((TypeError, ValueError)):
        runner_state._internal_spec_from_status(status)
    with pytest.raises((TypeError, ValueError)):
        runner_status.effective_spec_from_status(status)


def test_activation_requires_complete_envelope_even_for_plain_unprofiled_runs() -> None:
    spec = replace(
        _current_spec(),
        model_revision="",
        model_revision_auto=False,
        workload_profile={},
        workload_profile_input_digest="",
        workload_profile_producer_version="",
    )
    worker = spec.to_internal_dict()
    worker["train"]["hf_repo"] = "attacker/repo"
    status = runner_state.RunStatus(
        state="running",
        run_id=spec.run_id,
        spec=spec.to_dict(),
        effective_preparation={"worker_spec": worker},
    )

    with pytest.raises(ValueError, match="preparation envelope version is required"):
        runner_status.effective_spec_from_status(status)

    status.effective_preparation = {
        "version": PREPARATION_ENVELOPE_VERSION,
        "worker_spec": worker,
    }
    with pytest.raises(ValueError, match="failed integrity validation"):
        runner_status.effective_spec_from_status(status)


def test_activation_rejects_public_worker_algorithm_disagreement() -> None:
    from flash.runner.lifecycle.submit import _effective_preparation_snapshot

    spec = _current_spec()
    snapshot = _effective_preparation_snapshot(spec, spec, None)
    snapshot["worker_spec"] = {**snapshot["worker_spec"], "algorithm": "opd"}
    status = runner_state.RunStatus(
        state="running",
        run_id=spec.run_id,
        spec=spec.to_dict(),
        effective_preparation=snapshot,
    )

    with pytest.raises(ValueError, match="public and worker model/algorithm"):
        runner_status.effective_spec_from_status(status)


def test_retired_model_structurally_decodes_but_activation_rejects() -> None:
    from flash.core import catalog
    from flash.runner.lifecycle import preparation

    spec = _current_spec()
    retired = deepcopy(spec.to_internal_dict())
    retired["model"] = "retired/model"
    decoded = JobSpec.from_dict(retired)
    assert decoded.model == "retired/model"

    status = runner_state.RunStatus(
        state="cancelled",
        run_id=decoded.run_id,
        spec=decoded.to_dict(),
        effective_preparation={
            "version": PREPARATION_ENVELOPE_VERSION,
            "worker_spec": decoded.to_internal_dict(),
            "preparation_digest": preparation._preparation_digest(decoded, decoded, None),
        },
    )
    assert status.to_dict()["spec"]["model"] == "retired/model"
    assert runner_state._internal_spec_from_status(status).model == "retired/model"
    assert "retired/model" not in catalog.MODELS
    with pytest.raises(ValueError, match="unsupported model"):
        runner_status.effective_spec_from_status(status)
