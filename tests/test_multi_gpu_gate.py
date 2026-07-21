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


def test_gate_helper_rejects_multi_gpu():
    from flash import runner

    with pytest.raises(ValueError, match="multi-gpu training"):
        runner._require_supported_gpu_count(_spec(2))


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
            runner.submit_job(_spec(2), dry_run=True)
