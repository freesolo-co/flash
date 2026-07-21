"""CPU tests for the multi-gpu submit gate: gpu.count > 1 is rejected until a trainer supports it."""

from __future__ import annotations

import os
import tempfile

import pytest


def _spec(count: int, algorithm: str = "sft"):
    from flash.spec import JobSpec

    return JobSpec.from_dict(
        {
            "model": "test/gate",
            "algorithm": algorithm,
            "gpu": {"type": "RTX 5090", "count": count},
        }
    )


def test_gate_helper_allows_sft_and_rejects_unmigrated_algorithms():
    from flash import runner

    assert runner._require_supported_gpu_count(_spec(2, "sft")) is None
    with pytest.raises(ValueError, match="multi-gpu training"):
        runner._require_supported_gpu_count(_spec(2, "opd"))


def test_gate_helper_allows_single_gpu():
    from flash import runner

    # count == 1 is the default single-gpu path; must not raise.
    assert runner._require_supported_gpu_count(_spec(1)) is None


def test_gate_is_extensible_per_algorithm(monkeypatch):
    from flash import runner

    # opting an algorithm in lifts the gate for it; this is how the sft/opd verl prs enable count > 1.
    monkeypatch.setattr(runner, "_MULTI_GPU_ALGORITHMS", frozenset({"sft"}))
    assert runner._require_supported_gpu_count(_spec(2, "sft")) is None
    with pytest.raises(ValueError, match="multi-gpu training"):
        runner._require_supported_gpu_count(_spec(2, "opd"))


def test_submit_job_rejects_multi_gpu_at_boundary(monkeypatch):
    from flash import runner

    # the gate fires at the top of submit_job, before any provisioning or billing (even in dry-run).
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        with pytest.raises(ValueError, match="multi-gpu training"):
            runner.submit_job(_spec(2, "opd"), dry_run=True)


def test_submit_job_rejects_multi_gpu_prepared_worker_spec(monkeypatch):
    from flash import runner
    from flash.runner import PreparedJob

    # a single-gpu public spec paired with a prepared_job whose EFFECTIVE worker_spec is multi-gpu must
    # still be rejected: allocation and training provision the worker_spec, not the public spec.
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        prepared = PreparedJob(
            public_spec=_spec(1),
            worker_spec=_spec(2, "opd"),
            estimated_cost_usd=0.0,
        )
        with pytest.raises(ValueError, match="multi-gpu training"):
            runner.submit_job(_spec(1), dry_run=True, prepared_job=prepared)


def test_run_training_gates_effective_spec_on_recovery():
    import io

    from flash.runner.lifecycle import _run_training

    # recovery rebuilds the worker spec from a persisted snapshot and calls _run_training directly,
    # bypassing submit_job's gate; the shared submit+recovery path must fail closed on a multi-gpu spec
    # before any provisioning (the gate precedes the first get_status/allocation touch).
    with pytest.raises(ValueError, match="multi-gpu training"):
        _run_training(_spec(2, "opd"), io.StringIO(), prior_cost=0.0)
