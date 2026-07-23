"""cpu-only tests for sealed shared-engine bundle admission."""

from __future__ import annotations

from dataclasses import replace

import pytest

from flash.engine.vram import grpo_rollout_seq_len
from flash.runner.openrlhf_shared_bundle import (
    BundleAdmissionOutcome,
    BundleCompatibilityKey,
    LogicalRunStatus,
    SharedEngineBundle,
    SharedEngineBundlePacker,
    estimate_bundle_admission,
)
from flash.spec import GpuSpec, JobSpec, TrainSpec

_REVISION = ""


def _spec(
    run_id: str,
    *,
    model: str = "Qwen/Qwen3.5-9B",
    revision: str = _REVISION,
    rank: int = 64,
    max_context_tokens: int = 8192,
    algorithm: str = "grpo",
    gpu: str = "H100",
) -> JobSpec:
    return JobSpec(
        model=model,
        model_revision=revision,
        algorithm=algorithm,
        run_id=run_id,
        train=TrainSpec(
            lora_rank=rank,
            lora_alpha=rank * 2,
            max_context_tokens=max_context_tokens,
            max_completion_tokens=512,
            group_size=4,
            max_steps=2,
        ),
        gpu=GpuSpec(type=gpu, exact_type=gpu),
    )


def test_compatible_runs_share_one_bundle_and_geometry_mismatches_split():
    packer = SharedEngineBundlePacker(packing_delay_s=10)

    first = packer.offer(_spec("run-a"))
    second = packer.offer(_spec("run-b", algorithm="opd"))
    different_base = packer.offer(_spec("run-c", model="Qwen/Qwen3.5-4B"))
    different_rank = packer.offer(_spec("run-d", rank=32))
    different_context = packer.offer(_spec("run-e", max_context_tokens=4096))

    assert first.outcome is BundleAdmissionOutcome.ADMITTED
    assert second.outcome is BundleAdmissionOutcome.ADMITTED
    assert first.bundle_id == second.bundle_id
    assert different_base.bundle_id != first.bundle_id
    assert different_rank.bundle_id != first.bundle_id
    assert different_context.bundle_id != first.bundle_id
    assert len(packer.bundles) == 4
    assert packer.bundles[0].member_run_ids == ("run-a", "run-b")


def test_direct_bundle_rejects_incompatible_run_with_specific_reason():
    bundle = SharedEngineBundle("bundle-a", _spec("seed"))
    assert bundle.try_admit(_spec("run-a")).outcome is BundleAdmissionOutcome.ADMITTED

    decision = bundle.try_admit(_spec("run-b", rank=32))

    assert decision.outcome is BundleAdmissionOutcome.REJECTED
    assert decision.estimated_n == bundle.estimated_n
    assert decision.reason is not None
    assert "lora_rank" in decision.reason
    assert bundle.member_run_ids == ("run-a",)


def test_admission_estimate_matches_conservative_h100_and_h200_caps():
    h100 = estimate_bundle_admission(_spec("h100", gpu="H100", rank=64))
    h200 = estimate_bundle_admission(_spec("h200", gpu="H200", rank=64))
    h100_rank32 = estimate_bundle_admission(_spec("h100-r32", gpu="H100", rank=32))

    assert h100.estimated_n == 4
    assert h100.safety_cap == 4
    assert h200.estimated_n == 8
    assert h200.safety_cap == 8
    assert h100_rank32.estimated_n == 8
    assert h100_rank32.safety_cap == 8
    assert h100.shared_base_gib > 0
    assert h100.per_run_persistent_gib > 0
    assert h100.usable_per_run_gib > h100.per_run_persistent_gib


def test_bundle_admits_to_estimated_n_then_queues_with_reason():
    bundle = SharedEngineBundle("bundle-capacity", _spec("seed"))
    assert bundle.estimated_n == 4

    decisions = [bundle.try_admit(_spec(f"run-{index}")) for index in range(5)]

    assert [decision.outcome for decision in decisions[:4]] == [BundleAdmissionOutcome.ADMITTED] * 4
    overflow = decisions[4]
    assert overflow.outcome is BundleAdmissionOutcome.QUEUED
    assert overflow.estimated_n == 4
    assert overflow.reason == "bundle capacity reached: estimated_n=4"
    assert bundle.member_run_ids == ("run-0", "run-1", "run-2", "run-3")


def test_run_status_transitions_are_independent_between_siblings():
    bundle = SharedEngineBundle("bundle-status", _spec("seed"))
    for run_id in ("run-a", "run-b", "run-c"):
        assert bundle.try_admit(_spec(run_id)).outcome is BundleAdmissionOutcome.ADMITTED
    bundle.seal()

    bundle.transition_run("run-a", LogicalRunStatus.ACTIVE)
    bundle.transition_run("run-b", LogicalRunStatus.ACTIVE)
    bundle.transition_run("run-c", LogicalRunStatus.ACTIVE)
    bundle.transition_run("run-a", LogicalRunStatus.FINISHING)
    bundle.transition_run("run-a", LogicalRunStatus.DONE)

    assert bundle.run_snapshot("run-a").status is LogicalRunStatus.DONE
    assert bundle.run_snapshot("run-b").status is LogicalRunStatus.ACTIVE
    assert bundle.run_snapshot("run-c").status is LogicalRunStatus.ACTIVE

    bundle.transition_run("run-c", LogicalRunStatus.FAILED, error="teacher failed")

    assert bundle.run_snapshot("run-a").status is LogicalRunStatus.DONE
    assert bundle.run_snapshot("run-b").status is LogicalRunStatus.ACTIVE
    assert bundle.run_snapshot("run-c").status is LogicalRunStatus.FAILED
    assert bundle.run_snapshot("run-c").error == "teacher failed"


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_packing_window_seals_on_deadline_and_rejects_late_membership():
    clock = _Clock()
    packer = SharedEngineBundlePacker(packing_delay_s=3, clock=clock)
    decision = packer.offer(_spec("run-a"))
    bundle = packer.bundles[0]

    assert decision.outcome is BundleAdmissionOutcome.ADMITTED
    assert bundle.sealed is False
    clock.advance(2.9)
    assert packer.seal_ready() == ()
    clock.advance(0.1)
    assert packer.seal_ready() == (bundle,)
    assert bundle.sealed is True
    assert bundle.member_run_ids == ("run-a",)

    late = bundle.try_admit(_spec("run-b"))
    assert late.outcome is BundleAdmissionOutcome.QUEUED
    assert (
        late.reason
        == "bundle shared-"
        + bundle.compatibility_key.digest[:12]
        + "-1 is sealed; create a new compatible bundle"
    )
    assert bundle.member_run_ids == ("run-a",)


def test_compatibility_key_is_deterministic_and_stable_for_equal_specs():
    spec = _spec("run-a", revision="1" * 40)
    equal = replace(spec, run_id="run-b", worker_env={"IGNORED_RUN_LOCAL_VALUE": "1"})

    first = BundleCompatibilityKey.from_job_spec(spec)
    second = BundleCompatibilityKey.from_job_spec(equal)

    assert first == second
    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.max_model_length == grpo_rollout_seq_len(8192, 512, False)
    assert first.lora_target_modules == ("all-linear",)


def test_exact_model_revision_is_part_of_compatibility():
    first = BundleCompatibilityKey.from_job_spec(_spec("run-a", revision="1" * 40))
    second = BundleCompatibilityKey.from_job_spec(_spec("run-b", revision="2" * 40))

    reason = first.incompatibility_reason(second)
    assert first != second
    assert reason is not None
    assert "model_revision" in reason


def test_unsupported_gpu_is_rejected_before_any_bundle_is_retained():
    packer = SharedEngineBundlePacker()

    decision = packer.offer(_spec("run-a", gpu="A100 SXM"))

    assert decision.outcome is BundleAdmissionOutcome.REJECTED
    assert decision.estimated_n == 0
    assert decision.reason == "shared bundle admission is not profiled for GPU class 'A100 SXM'"
    assert packer.bundles == ()


def test_failed_status_requires_a_reason_and_terminal_status_is_sticky():
    bundle = SharedEngineBundle("bundle-transitions", _spec("seed"))
    bundle.try_admit(_spec("run-a"))
    bundle.transition_run("run-a", LogicalRunStatus.ACTIVE)

    with pytest.raises(ValueError, match="requires an error reason"):
        bundle.transition_run("run-a", LogicalRunStatus.FAILED)

    bundle.transition_run("run-a", LogicalRunStatus.CANCELLED)
    with pytest.raises(ValueError, match="invalid bundle run transition"):
        bundle.transition_run("run-a", LogicalRunStatus.ACTIVE)
