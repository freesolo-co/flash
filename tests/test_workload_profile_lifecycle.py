from __future__ import annotations

from dataclasses import replace

import pytest

from flash.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec
from flash.workload_profile import (
    SFT_PROFILE_KIND,
    SftWorkloadProfile,
    sft_profile_input_digest,
    sft_profile_run_id,
)
from tests._helpers.runner import fresh_runner


def _spec() -> JobSpec:
    return JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        algorithm="sft",
        environment=EnvironmentSpec(
            id="team/example",
            resolved_sha="b" * 40,
            params={"dataset_split": "train"},
        ),
        train=TrainSpec(
            epochs=2,
            batch_size=8,
            max_context_tokens=1024,
            max_steps=12,
            max_examples=64,
        ),
        gpu=GpuSpec(count=2),
        seed=42,
    )


def _profile(input_digest: str) -> SftWorkloadProfile:
    return SftWorkloadProfile(
        input_digest=input_digest,
        producer_version="1.2.3",
        tokenizer_revision="a" * 40,
        environment_id="team/example",
        environment_revision="b" * 40,
        source_examples=80,
        selected_examples=64,
        retained_examples=60,
        dropped_examples=4,
        epochs=2,
        max_length=1024,
        packing_mode="packed",
        architecture_mode="pure-attention",
        packed_blocks=20,
        real_tokens_per_epoch=15_000,
        supervised_tokens_per_epoch=6_000,
        padded_compute_tokens_per_epoch=20_000,
        authoritative_real_tokens=45_000,
        authoritative_supervised_tokens=18_000,
        authoritative_compute_tokens=60_000,
        realized_max_length=980,
        examples_per_update=8,
        derived_steps=8,
        authoritative_steps=12,
        packing_efficiency=0.75,
        sample_policy="exact-prefix",
        created_at=1_780_000_000.0,
    )


def _input_digest(spec: JobSpec) -> str:
    return sft_profile_input_digest(
        spec,
        tokenizer_revision=spec.model_revision,
        producer_version="1.2.3",
    )


def test_profile_cache_miss_returns_one_deterministic_profile_job(tmp_path, monkeypatch) -> None:
    runner = fresh_runner(tmp_path, monkeypatch)
    spec = _spec()
    digest = _input_digest(spec)
    prepared = runner.PreparedJob(spec, spec, 0.25)
    calls = []

    monkeypatch.setattr(runner, "_profile_producer_version", lambda: "1.2.3")

    def prepare_profile(*_args, **kwargs):
        calls.append(kwargs["input_digest"])
        return prepared

    monkeypatch.setattr(runner, "_prepared_sft_profile_job", prepare_profile)

    with pytest.raises(runner.WorkloadProfilePending) as raised:
        runner._require_sft_workload_profile(spec)

    assert raised.value.profile_run_id == sft_profile_run_id(digest)
    assert raised.value.state == "required"
    assert raised.value.prepared_job is prepared
    assert calls == [digest]
    with pytest.raises(FileNotFoundError):
        runner.get_status(spec.run_id)


def test_existing_profile_job_is_reused_without_a_second_writer(tmp_path, monkeypatch) -> None:
    runner = fresh_runner(tmp_path, monkeypatch)
    spec = _spec()
    digest = _input_digest(spec)
    profile_run_id = sft_profile_run_id(digest)
    runner._save_status(
        runner.RunStatus(
            run_id=profile_run_id,
            state="running",
            spec=spec.to_dict(),
            workload_profile_kind="sft",
            workload_profile_input_digest=digest,
        )
    )
    monkeypatch.setattr(runner, "_profile_producer_version", lambda: "1.2.3")
    monkeypatch.setattr(
        runner,
        "_prepared_sft_profile_job",
        lambda *_args, **_kwargs: pytest.fail("duplicate profile writer was prepared"),
    )

    with pytest.raises(runner.WorkloadProfilePending) as raised:
        runner._require_sft_workload_profile(spec)

    assert raised.value.profile_run_id == profile_run_id
    assert raised.value.state == "running"
    assert raised.value.prepared_job is None


def test_successful_profile_is_attached_to_training_spec(tmp_path, monkeypatch) -> None:
    runner = fresh_runner(tmp_path, monkeypatch)
    spec = _spec()
    digest = _input_digest(spec)
    profile = _profile(digest)
    profile_run_id = sft_profile_run_id(digest)
    runner._save_status(
        runner.RunStatus(
            run_id=profile_run_id,
            state="done",
            spec=spec.to_dict(),
            workload_profile_kind="sft",
            workload_profile_input_digest=digest,
            workload_profile=profile.to_dict(),
        )
    )
    monkeypatch.setattr(runner, "_profile_producer_version", lambda: "1.2.3")

    attached = runner._require_sft_workload_profile(spec)

    assert attached.workload_profile_input_digest == digest
    assert attached.workload_profile == profile.to_dict()
    assert attached.to_dict().get("workload_profile") is None


@pytest.mark.parametrize("spent_state", ["failed", "cancelled", "dry_run"])
def test_spent_profile_blocks_training_and_offers_a_replacement(
    tmp_path, monkeypatch, spent_state: str
) -> None:
    """A spent profile must not wedge its workload: it blocks the quote but offers a relaunch.

    The profile id is derived from the workload, not from the account, so a preempted pod would
    otherwise make this exact config unquotable for every user forever with nothing in the system
    able to clear it. The replacement carries the spent run's own timestamp, which is what lets the
    server hand the relaunch to exactly one of several submitters waiting on the same dead profile.
    """
    runner = fresh_runner(tmp_path, monkeypatch)
    spec = _spec()
    digest = _input_digest(spec)
    profile_run_id = sft_profile_run_id(digest)
    spent = runner.RunStatus(
        run_id=profile_run_id,
        state=spent_state,
        spec=spec.to_dict(),
        workload_profile_kind="sft",
        workload_profile_input_digest=digest,
        error="profile worker failed",
    )
    runner._save_status(spent)
    monkeypatch.setattr(runner, "_profile_producer_version", lambda: "1.2.3")

    with pytest.raises(runner.WorkloadProfilePending) as excinfo:
        runner._require_sft_workload_profile(spec)

    assert excinfo.value.state == spent_state
    assert excinfo.value.spent_at == spent.created_at
    assert isinstance(excinfo.value.prepared_job, runner.PreparedJob)
    assert excinfo.value.prepared_job.public_spec.run_id == profile_run_id


def test_submit_propagates_a_profile_miss_instead_of_launching_it_unclaimed(
    tmp_path, monkeypatch
) -> None:
    """``submit_job`` must not launch the profile itself. Only a claim holder may launch one.

    The profile run id is derived from the workload, not the account, so two submitters of the same
    config collide on it by design. ``db.claim_profile_run`` / ``db.reclaim_spent_profile_run``
    settle which one launches, and the ordering they compare against requires the claim to be taken
    BEFORE the run it authorizes is created. A launch from here would take no claim at all: the
    same workload would be profiled and billed twice, and the takeover that unwedges a spent
    profile would lose the ordering in both directions. The claim lives in the server db, which
    this module deliberately does not import, so the miss propagates to the caller that owns the
    key and that caller claims first.
    """
    runner = fresh_runner(tmp_path, monkeypatch)
    spec = _spec()
    monkeypatch.setattr(runner, "_profile_producer_version", lambda: "1.2.3")
    monkeypatch.setattr(runner, "_resolve_model_revision", lambda s, **_kw: s)
    monkeypatch.setattr(runner, "_assign_resolved_env_sha", lambda s, **_kw: s)

    real_submit = runner.submit_job
    launched: list[object] = []
    depth = {"n": 0}

    def recording_submit(submitted_spec, **kwargs):
        # only the NESTED call is a runner-internal launch; the outer one is this test's own.
        if depth["n"]:
            launched.append(kwargs.get("prepared_job"))
        depth["n"] += 1
        try:
            return real_submit(submitted_spec, **kwargs)
        finally:
            depth["n"] -= 1

    monkeypatch.setattr(runner, "submit_job", recording_submit)

    with pytest.raises(runner.WorkloadProfilePending) as raised:
        recording_submit(spec, dry_run=True)

    # the caller is handed the prepared profile job to launch itself, AFTER claiming the id.
    assert isinstance(raised.value.prepared_job, runner.PreparedJob)
    assert raised.value.prepared_job.public_spec.run_id == sft_profile_run_id(_input_digest(spec))
    # and nothing was launched from inside the runner.
    assert launched == []
    # no profile run row was written either, so no unclaimed run exists for a later caller to join.
    with pytest.raises(FileNotFoundError):
        runner.get_status(raised.value.profile_run_id)


def test_relaunching_a_spent_profile_replaces_it_but_a_live_one_is_joined(
    tmp_path, monkeypatch
) -> None:
    """Submitting the deterministic profile id overwrites a spent record and joins a live one.

    Both halves matter. Without the overwrite the takeover winner would return the dead record and
    never launch, so the wedge would survive the relaunch path. Without the join a concurrent
    submitter of the same config would restart a profile that is already running and bill the
    identical work twice.
    """
    runner = fresh_runner(tmp_path, monkeypatch)
    spec = _spec()
    digest = _input_digest(spec)
    monkeypatch.setattr(runner, "_profile_producer_version", lambda: "1.2.3")
    prepared = runner._prepared_sft_profile_job(spec, input_digest=digest)
    profile_run_id = prepared.public_spec.run_id
    runner._save_status(
        runner.RunStatus(
            run_id=profile_run_id,
            state="failed",
            spec=spec.to_dict(),
            workload_profile_kind="sft",
            workload_profile_input_digest=digest,
            error="pod preempted",
        )
    )

    relaunched = runner.submit_job(
        prepared.public_spec, dry_run=True, prepared_job=prepared, owner_key_id=1
    )

    # the spent record is gone rather than returned: a fresh run replaced it under the same id.
    assert relaunched.state == "dry_run"
    assert relaunched.error is None

    # a live profile under the same id is returned untouched instead. written directly because
    # _update refuses to move a run back out of a terminal state, which is the point of this half.
    runner._save_status(
        runner.RunStatus(
            run_id=profile_run_id,
            state="running",
            spec=spec.to_dict(),
            workload_profile_kind="sft",
            workload_profile_input_digest=digest,
        )
    )
    joined = runner.submit_job(
        prepared.public_spec, dry_run=True, prepared_job=prepared, owner_key_id=1
    )

    assert joined.state == "running"


def _profile_spec(spec: JobSpec, digest: str) -> JobSpec:
    return replace(
        spec,
        run_id=sft_profile_run_id(digest),
        gpu=replace(spec.gpu, count=1, max_wall_seconds=600),
        workload_profile_kind=SFT_PROFILE_KIND,
        workload_profile_input_digest=digest,
        workload_profile_producer_version="1.2.3",
        workload_profile={},
    )


def test_profile_job_is_priced_without_the_profile_it_produces() -> None:
    """The training estimator needs a completed profile, so the profile job cannot use it."""
    from flash.cost.spec import estimate_for_spec, runconfig_from_spec

    spec = _profile_spec(_spec(), _input_digest(_spec()))

    estimate = estimate_for_spec(spec)

    assert estimate.total_usd == pytest.approx(
        estimate.gpu_hourly_usd * estimate.gpu_count * 600 / 3600.0
    )
    assert estimate.train_seconds == 600
    assert estimate.gpu_count == 1
    with pytest.raises(ValueError, match="cannot be priced as training"):
        runconfig_from_spec(spec)


def test_profile_quote_uses_the_selected_live_candidate_rate() -> None:
    """The lifecycle repricing after allocation must reach the profile charge, not the training one."""
    from flash.cost.spec import estimate_for_spec
    from flash.providers.base import Allocation, Candidate

    spec = _profile_spec(_spec(), _input_digest(_spec()))
    chosen = Candidate(provider="runpod", gpu="RTX 4090", hourly_usd=0.44, vram_gb=24)
    allocation = Allocation(
        provider="runpod",
        gpu="RTX 4090",
        hourly_usd=0.44,
        min_vram_gb=1,
        candidates=(chosen,),
        gpu_count=1,
    )

    estimate = estimate_for_spec(spec, allocation=allocation)

    assert estimate.gpu == "RTX 4090"
    assert estimate.gpu_hourly_usd == pytest.approx(0.44)
    assert estimate.total_usd == pytest.approx(0.44 * 600 / 3600.0)


def test_profile_job_prepares_a_real_quote_through_the_unmocked_path(tmp_path, monkeypatch) -> None:
    """Exercises _prepared_sft_profile_job itself; a mocked preparation hid a quote crash."""
    from flash.providers.base import GPU_INFO

    runner = fresh_runner(tmp_path, monkeypatch)
    spec = _spec()
    digest = _input_digest(spec)
    monkeypatch.setattr(runner, "_profile_producer_version", lambda: "1.2.3")
    # the spec already pins an immutable revision; skip only the hub round-trip that re-verifies it.
    monkeypatch.setattr(runner, "_resolve_model_revision", lambda spec, **_kwargs: spec)

    with pytest.raises(runner.WorkloadProfilePending) as raised:
        runner._require_sft_workload_profile(spec)

    prepared = raised.value.prepared_job
    assert prepared.worker_spec.workload_profile_kind == SFT_PROFILE_KIND
    assert prepared.worker_spec.run_id == sft_profile_run_id(digest)
    assert prepared.worker_spec.gpu.count == 1
    assert prepared.worker_spec.gpu.max_wall_seconds == 600
    # the cpu-only profile is quoted from the cheapest validated card held for its wall cap, not
    # from the training shape it exists to measure.
    cheapest = min(
        info.hourly_usd for info in GPU_INFO.values() if info.enum_member and info.validated
    )
    assert prepared.estimated_cost_usd == pytest.approx(cheapest * 600 / 3600.0)


def test_profile_allocates_the_cheapest_card_not_the_training_shape(monkeypatch) -> None:
    """A cpu-only profile job must not rent the card its training run would need."""
    from flash.providers import allocator
    from flash.providers.base import Candidate

    offered = [
        Candidate(provider="runpod", gpu="RTX 4090", hourly_usd=0.69, vram_gb=24),
        Candidate(provider="runpod", gpu="H100", hourly_usd=3.29, vram_gb=80),
    ]
    # a list, not a single slot: the training allocation below calls the same fake, and a dict entry
    # would leave only the LAST call's need behind -- so the profile assertion would silently read
    # the training shape and pass or fail for the wrong reason.
    seen: list[int] = []

    def fake_live_candidates(need, _constraints):
        seen.append(need)
        return offered

    provider = type(
        "P",
        (),
        {"live_candidates": staticmethod(fake_live_candidates), "live_capacity": False},
    )
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    monkeypatch.setattr(allocator, "get_provider", lambda _name: provider)

    profile_alloc = allocator.allocate(
        "Qwen/Qwen3.5-0.8B", "sft", train=_spec().train, workload_profile=True
    )
    training_alloc = allocator.allocate("Qwen/Qwen3.5-0.8B", "sft", train=_spec().train)

    assert seen[0] == 1  # the profile never sizes for weights it does not load
    assert seen[1] > 1  # the training allocation still sizes for the model
    assert profile_alloc.gpu == "RTX 4090"
    assert profile_alloc.gpu_count == 1
    assert profile_alloc.min_vram_gb < training_alloc.min_vram_gb


def test_profile_provenance_is_covered_by_preparation_digest(tmp_path, monkeypatch) -> None:
    runner = fresh_runner(tmp_path, monkeypatch)
    spec = _spec()
    digest = _input_digest(spec)
    profile = _profile(digest).to_dict()
    worker = replace(
        spec,
        run_id="training-run",
        workload_profile_input_digest=digest,
        workload_profile_producer_version="1.2.3",
        workload_profile=profile,
    )
    public = replace(
        worker,
        workload_profile_input_digest="",
        workload_profile_producer_version="",
        workload_profile={},
    )
    snapshot = {
        "worker_spec": worker.to_internal_dict(),
        "workload_profile": profile,
        "adapter_identity": None,
        "preparation_digest": runner._preparation_digest(public, worker, None),
    }
    status = runner.RunStatus(
        run_id=worker.run_id,
        state="queued",
        spec=public.to_dict(),
        effective_preparation=snapshot,
    )

    assert runner.effective_spec_from_status(status) == worker

    tampered = dict(profile)
    tampered["packed_blocks"] = 21
    status.effective_preparation = {**snapshot, "workload_profile": tampered}
    with pytest.raises(ValueError, match="persisted workload profile"):
        runner.effective_spec_from_status(status)
