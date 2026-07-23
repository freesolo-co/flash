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

_REVISION = "1" * 40


def _spec(
    run_id: str,
    *,
    model: str = "Qwen/Qwen3.5-9B",
    revision: str = _REVISION,
    rank: int = 64,
    max_context_tokens: int = 8192,
    max_completion_tokens: int = 512,
    batch_size: int = 8,
    group_size: int = 4,
    kl_penalty_coef: float | None = None,
    init_from_adapter: str = "",
    init_from_adapter_revision: str = "",
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
            max_completion_tokens=max_completion_tokens,
            batch_size=batch_size,
            group_size=group_size,
            kl_penalty_coef=kl_penalty_coef,
            init_from_adapter=init_from_adapter,
            init_from_adapter_revision=init_from_adapter_revision,
            max_steps=2,
        ),
        gpu=GpuSpec(type=gpu, exact_type=gpu),
    )


def test_compatible_runs_share_one_bundle_and_geometry_mismatches_split():
    packer = SharedEngineBundlePacker(packing_delay_s=10)

    first = packer.offer(_spec("run-a"))
    second = packer.offer(_spec("run-b"))
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


def test_packer_rejects_run_id_already_assigned_to_a_sealed_bundle():
    packer = SharedEngineBundlePacker()
    decisions = [packer.offer(_spec(f"run-{index}")) for index in range(4)]
    bundle = packer.bundles[0]

    assert all(decision.outcome is BundleAdmissionOutcome.ADMITTED for decision in decisions)
    assert bundle.sealed is True

    duplicate = packer.offer(_spec("run-0"))

    assert duplicate.outcome is BundleAdmissionOutcome.REJECTED
    assert duplicate.bundle_id == bundle.bundle_id
    assert duplicate.reason == f"run 'run-0' is already assigned to bundle {bundle.bundle_id}"
    assert len(packer.bundles) == 1


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


def test_offer_self_enforces_packing_deadline_and_opens_a_new_bundle():
    clock = _Clock()
    packer = SharedEngineBundlePacker(packing_delay_s=3, clock=clock)
    first = packer.offer(_spec("run-a"))
    first_bundle = packer.bundles[0]

    assert first.outcome is BundleAdmissionOutcome.ADMITTED
    clock.advance(3)
    late = packer.offer(_spec("run-b"))

    assert first_bundle.sealed is True
    assert first_bundle.member_run_ids == ("run-a",)
    assert late.outcome is BundleAdmissionOutcome.ADMITTED
    assert late.bundle_id != first.bundle_id
    assert len(packer.bundles) == 2


def test_direct_bundle_queues_late_member_without_external_seal_tick():
    clock = _Clock()
    bundle = SharedEngineBundle("bundle-deadline", _spec("seed"), packing_delay_s=3, clock=clock)
    bundle.try_admit(_spec("run-a"))
    clock.advance(3)

    late = bundle.try_admit(_spec("run-b"))

    assert bundle.sealed is True
    assert late.outcome is BundleAdmissionOutcome.QUEUED
    assert late.reason == "bundle bundle-deadline is sealed; create a new compatible bundle"
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


def test_reference_adapter_requirement_is_keyed_and_charged():
    without_reference = _spec("run-a")
    with_reference = _spec("run-b", kl_penalty_coef=0.1)

    plain_key = BundleCompatibilityKey.from_job_spec(without_reference)
    reference_key = BundleCompatibilityKey.from_job_spec(with_reference)
    plain_estimate = estimate_bundle_admission(without_reference)
    reference_estimate = estimate_bundle_admission(with_reference)

    assert plain_key.reference_adapter_required is False
    assert reference_key.reference_adapter_required is True
    assert plain_key != reference_key
    assert reference_estimate.per_run_persistent_gib > plain_estimate.per_run_persistent_gib


def test_warm_start_geometry_is_isolated_by_immutable_source_identity():
    fresh = BundleCompatibilityKey.from_job_spec(_spec("fresh"))
    warm_a = BundleCompatibilityKey.from_job_spec(
        _spec(
            "warm-a",
            init_from_adapter="repo:rl/source-a",
            init_from_adapter_revision="1" * 40,
        )
    )
    warm_a_equal = BundleCompatibilityKey.from_job_spec(
        _spec(
            "warm-a-equal",
            init_from_adapter="repo:rl/source-a",
            init_from_adapter_revision="1" * 40,
        )
    )
    warm_b = BundleCompatibilityKey.from_job_spec(
        _spec(
            "warm-b",
            init_from_adapter="repo:rl/source-b",
            init_from_adapter_revision="1" * 40,
        )
    )

    assert fresh.lora_geometry_identity == "managed:all-linear"
    assert warm_a == warm_a_equal
    assert warm_a != fresh
    assert warm_a != warm_b


def test_rollout_concurrency_is_keyed_and_reduces_capacity():
    baseline = _spec("baseline", batch_size=8, group_size=4)
    high_concurrency = _spec("high", batch_size=64, group_size=8)

    baseline_key = BundleCompatibilityKey.from_job_spec(baseline)
    high_key = BundleCompatibilityKey.from_job_spec(high_concurrency)
    baseline_estimate = estimate_bundle_admission(baseline)
    high_estimate = estimate_bundle_admission(high_concurrency)

    assert baseline_key.rollout_concurrency == 32
    assert high_key.rollout_concurrency == 512
    assert baseline_key != high_key
    assert baseline_estimate.estimated_n == 4
    assert high_estimate.estimated_n == 1


def test_completion_budget_is_part_of_compatibility():
    short = BundleCompatibilityKey.from_job_spec(
        _spec("short", max_context_tokens=8192, max_completion_tokens=256)
    )
    long = BundleCompatibilityKey.from_job_spec(
        _spec("long", max_context_tokens=8192, max_completion_tokens=2048)
    )

    assert short.max_model_length == long.max_model_length
    assert short.max_completion_tokens == 256
    assert long.max_completion_tokens == 2048
    assert short != long


def test_opd_admission_reuses_existing_peak_vram_model():
    estimate = estimate_bundle_admission(
        _spec(
            "opd-too-large",
            model="Qwen/Qwen3.5-4B",
            algorithm="opd",
            max_context_tokens=16384,
            batch_size=8,
            group_size=1,
        )
    )

    assert estimate.estimated_n == 0
    assert estimate.reason is not None
    assert "OpenRLHF OPD requires an estimated" in estimate.reason


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


def test_direct_bundle_rejects_unsupported_algorithm():
    bundle = SharedEngineBundle("bundle-algorithm", _spec("seed"))

    decision = bundle.try_admit(_spec("run-sft", algorithm="sft"))

    assert decision.outcome is BundleAdmissionOutcome.REJECTED
    assert decision.reason == "shared bundles support only GRPO and OPD"
    assert bundle.member_run_ids == ()


@pytest.mark.parametrize(
    ("train_changes", "reason"),
    [
        ({"save_every": 10}, "train.save_every"),
        ({"stop_sequences": ("</answer>",)}, "train.stop_sequences"),
        ({"structured_outputs": '{"type":"json_object"}'}, "train.structured_outputs"),
        ({"credit_assignment": "per_turn"}, "per-turn credit assignment"),
    ],
)
def test_grpo_worker_deferred_options_are_rejected_during_admission(train_changes, reason):
    packer = SharedEngineBundlePacker()
    base = _spec("unsupported-grpo")
    spec = replace(base, train=replace(base.train, **train_changes))

    decision = packer.offer(spec)

    assert decision.outcome is BundleAdmissionOutcome.REJECTED
    assert decision.reason is not None
    assert reason in decision.reason
    assert packer.bundles == ()


def test_opd_structured_outputs_are_rejected_during_admission():
    packer = SharedEngineBundlePacker()
    base = _spec("unsupported-opd", algorithm="opd")
    spec = replace(
        base,
        train=replace(base.train, structured_outputs='{"type":"json_object"}'),
    )

    decision = packer.offer(spec)

    assert decision.outcome is BundleAdmissionOutcome.REJECTED
    assert decision.reason == "shared OpenRLHF OPD does not support train.structured_outputs"
    assert packer.bundles == ()


def test_unpinned_gpu_is_rejected_before_any_bundle_is_retained():
    packer = SharedEngineBundlePacker()
    spec = replace(_spec("run-a"), gpu=GpuSpec(type="H100"))

    decision = packer.offer(spec)

    assert decision.outcome is BundleAdmissionOutcome.REJECTED
    assert decision.estimated_n == 0
    assert decision.reason == "shared bundle admission requires pinned gpu.exact_type"
    assert packer.bundles == ()
    with pytest.raises(ValueError, match=r"requires pinned gpu\.exact_type"):
        BundleCompatibilityKey.from_job_spec(spec)


def test_unpinned_model_revision_is_rejected_before_any_bundle_is_retained():
    packer = SharedEngineBundlePacker()
    spec = _spec("run-a", revision="")

    decision = packer.offer(spec)

    assert decision.outcome is BundleAdmissionOutcome.REJECTED
    assert decision.estimated_n == 0
    assert decision.reason == "shared bundle admission requires an immutable model_revision"
    assert packer.bundles == ()
    with pytest.raises(ValueError, match="requires an immutable model_revision"):
        BundleCompatibilityKey.from_job_spec(spec)


def test_failed_status_requires_a_reason_and_terminal_status_is_sticky():
    bundle = SharedEngineBundle("bundle-transitions", _spec("seed"))
    bundle.try_admit(_spec("run-a"))
    bundle.transition_run("run-a", LogicalRunStatus.ACTIVE)

    with pytest.raises(ValueError, match="requires an error reason"):
        bundle.transition_run("run-a", LogicalRunStatus.FAILED)

    bundle.transition_run("run-a", LogicalRunStatus.CANCELLED)
    with pytest.raises(ValueError, match="invalid bundle run transition"):
        bundle.transition_run("run-a", LogicalRunStatus.ACTIVE)
