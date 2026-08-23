"""Durable capacity experience and its ordering-only allocation contract."""

from __future__ import annotations

import contextlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest


def _record_refusal_after_barrier(shape, observed_at, barrier) -> None:
    from flash.providers.capacity_experience import record_capacity_refusal

    barrier.wait(timeout=10)
    record_capacity_refusal(shape, now=observed_at)


def _ledger_path(runner) -> Path:
    return Path(runner.RUNS_DIR) / "capacity-experience.json"


def _rank(candidates, *, recent=frozenset(), provider_rank=None):
    from flash.providers.allocator import _cheapest_allocation

    return _cheapest_allocation(
        candidates,
        need=1,
        cost_per_step=lambda candidate: candidate.total_hourly_usd,
        provider_rank=provider_rank or {},
        recent_refusals=recent,
    )


def _allocation_candidates():
    from flash.providers.base import Candidate

    return (
        Candidate("runpod", "A100 PCIe", 1.0, 80),
        Candidate("runpod", "H200", 2.0, 141),
    )


def test_capacity_experience_round_trip_and_success_forgiveness(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.providers.capacity_experience import (
        read_capacity_experience,
        recent_capacity_refusals,
        record_capacity_refusal,
        record_capacity_success,
    )

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    shape = ("runpod", "A100 PCIe", 1)

    record_capacity_refusal(shape, now=1000.0)
    record_capacity_refusal(shape, now=1001.0)
    entry = read_capacity_experience()[shape]
    assert entry.last_refusal_at == 1001.0
    assert entry.refusal_count == 2
    assert entry.last_success_at is None
    assert recent_capacity_refusals({shape: entry}, now=1002.0) == {shape}

    record_capacity_success(shape, now=1003.0)
    forgiven = read_capacity_experience()[shape]
    assert forgiven.last_refusal_at == 1001.0
    assert forgiven.refusal_count == 0
    assert forgiven.last_success_at == 1003.0
    assert recent_capacity_refusals({shape: forgiven}, now=1004.0) == frozenset()


def test_concurrent_capacity_writers_preserve_every_shape(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.providers.capacity_experience import read_capacity_experience

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    shapes = [("runpod", f"test-gpu-{index}", 1) for index in range(12)]
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(len(shapes) + 1)
    processes = [
        context.Process(target=_record_refusal_after_barrier, args=(shape, 1000.0 + index, barrier))
        for index, shape in enumerate(shapes)
    ]

    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    experience = read_capacity_experience()
    assert set(experience) == set(shapes)
    assert all(experience[shape].refusal_count == 1 for shape in shapes)


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        "not-json",
        json.dumps({"version": 1, "shapes": []}),
        json.dumps({"version": 1, "shapes": {"bad-key": {}}}),
    ],
)
def test_malformed_capacity_ledger_degrades_to_empty_and_allocation_succeeds(
    monkeypatch, tmp_path, contents
):
    import flash.runner as runner
    from flash.providers import allocator

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    _ledger_path(runner).parent.mkdir(parents=True)
    _ledger_path(runner).write_text(contents)
    candidates = list(_allocation_candidates())
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *args, **kwargs: 1)
    monkeypatch.setattr(allocator, "_resolved_gpu_count", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        allocator,
        "_gather_candidates",
        lambda *args, **kwargs: (candidates, False, {}),
    )
    monkeypatch.setattr(allocator, "_step_cost_ranker", lambda *args, **kwargs: None)

    allocation = allocator.allocate("model", "sft")

    assert allocation.gpu == "A100 PCIe"
    assert [(candidate.provider, candidate.gpu) for candidate in allocation.candidates] == [
        (candidate.provider, candidate.gpu) for candidate in candidates
    ]


def test_recent_refusal_demotes_cheaper_shape_and_logs(caplog):
    from flash.providers import allocator

    candidates = _allocation_candidates()
    refused = {("runpod", "A100 PCIe", 1)}

    with caplog.at_level("INFO", logger=allocator.logger.name):
        allocation = _rank(candidates, recent=refused)

    assert [candidate.gpu for candidate in allocation.candidates] == ["H200", "A100 PCIe"]
    assert "capacity experience deprioritized recently refused shape" in caplog.text
    assert "1x A100 PCIe@runpod" in caplog.text


def test_expired_refusal_restores_identical_cost_order():
    from flash.providers.capacity_experience import (
        CAPACITY_REFUSAL_TTL_S,
        CapacityExperience,
        recent_capacity_refusals,
    )

    candidates = _allocation_candidates()
    shape = ("runpod", "A100 PCIe", 1)
    baseline = _rank(candidates).candidates
    experience = {shape: CapacityExperience(1000.0, 3, None)}
    expired = recent_capacity_refusals(experience, now=1000.0 + CAPACITY_REFUSAL_TTL_S)

    assert expired == frozenset()
    assert _rank(candidates, recent=expired).candidates == baseline


def test_authored_provider_preference_wins_over_experience():
    from flash.providers.base import Candidate

    preferred_but_refused = Candidate("runpod", "H100", 1.0, 80)
    unrefused_fallback = Candidate("lambda", "H100", 0.5, 80)
    allocation = _rank(
        (unrefused_fallback, preferred_but_refused),
        recent={("runpod", "H100", 1)},
        provider_rank={"runpod": 0, "lambda": 1},
    )

    assert allocation.candidates == (preferred_but_refused, unrefused_fallback)


def test_capacity_experience_ranking_never_removes_a_candidate():
    from flash.providers.base import Candidate

    candidates = (
        Candidate("runpod", "A100 PCIe", 1.0, 80),
        Candidate("runpod", "H200", 2.0, 141),
        Candidate("lambda", "H100", 3.0, 80),
        Candidate("vast", "RTX 4090", 0.5, 24, gpu_count=2),
    )
    recent = {
        ("runpod", "A100 PCIe", 1),
        ("lambda", "H100", 1),
        ("vast", "RTX 4090", 2),
    }

    ranked = _rank(candidates, recent=recent).candidates

    assert len(ranked) == len(candidates)
    assert set(ranked) == set(candidates)


def test_allocate_reads_capacity_ledger_once(monkeypatch):
    from flash.providers import allocator

    reads = []
    candidates = list(_allocation_candidates())
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *args, **kwargs: 1)
    monkeypatch.setattr(allocator, "_resolved_gpu_count", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        allocator,
        "_gather_candidates",
        lambda *args, **kwargs: (candidates, False, {}),
    )
    monkeypatch.setattr(allocator, "_step_cost_ranker", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        allocator,
        "read_capacity_experience",
        lambda: reads.append(True) or {},
    )

    allocator.allocate("model", "sft")

    assert reads == [True]


def test_capacity_ledger_is_bounded_to_newest_shapes(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.providers import capacity_experience

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    total = capacity_experience._MAX_TRACKED_SHAPES + 10
    for index in range(total):
        capacity_experience.record_capacity_refusal(
            ("runpod", f"bounded-gpu-{index}", 1),
            now=1000.0 + index,
        )

    experience = capacity_experience.read_capacity_experience()
    assert len(experience) == capacity_experience._MAX_TRACKED_SHAPES
    assert ("runpod", "bounded-gpu-0", 1) not in experience
    assert ("runpod", f"bounded-gpu-{total - 1}", 1) in experience


def test_shape_separator_is_absent_from_catalog_names():
    from flash.providers import PROVIDER_NAMES
    from flash.providers.base import GPU_INFO
    from flash.providers.capacity_experience import _SHAPE_SEPARATOR

    assert all(_SHAPE_SEPARATOR not in provider for provider in PROVIDER_NAMES)
    assert all(_SHAPE_SEPARATOR not in gpu for gpu in GPU_INFO)


def test_capacity_recording_failure_does_not_raise_into_runner(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.providers import capacity_experience
    from flash.runner.supervise import seed_submission

    not_a_directory = tmp_path / "state-file"
    not_a_directory.write_text("occupied")
    monkeypatch.setattr(runner, "RUNS_DIR", str(not_a_directory))

    seed_submission._record_capacity_observation(
        capacity_experience.record_capacity_refusal,
        ("runpod", "H100", 1),
    )
    seed_submission._record_capacity_observation(
        lambda _shape: (_ for _ in ()).throw(PermissionError("read only")),
        ("runpod", "H100", 1),
    )


def test_durable_experience_never_feeds_capacity_exhaustion(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.providers.base import Candidate, PollResult
    from flash.providers.capacity_experience import record_capacity_refusal
    from flash.runner.supervise.retry_decision import _capacity_exhausted

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    shape = ("runpod", "H100", 1)
    record_capacity_refusal(shape, now=1000.0)
    record_capacity_refusal(shape, now=1001.0)
    candidate = Candidate(*shape[:2], hourly_usd=1.0, vram_gb=80, gpu_count=shape[2])
    outcome = SimpleNamespace(
        result=PollResult(False, failure="no_capacity", detail="dry"),
        chosen=candidate,
        candidates=(candidate,),
    )
    ctx = SimpleNamespace(
        spec=SimpleNamespace(gpu=SimpleNamespace(type="H100", provider="runpod", count=1)),
        capacity_refusals={},
        failed_providers=set(),
        tried_classes=set(),
    )

    assert not _capacity_exhausted(ctx, outcome, first_cache_drop=False)
    ctx.capacity_refusals[shape] = 1
    assert _capacity_exhausted(ctx, outcome, first_cache_drop=False)


def test_stale_refusal_after_success_keeps_the_entry_consistent(monkeypatch, tmp_path):
    """A refusal older than the last success must not persist an entry the reader then rejects.

    `_entry_from_json` requires `refusal_count > 0` to agree with "the latest observation is a
    refusal". A sample taken before the exclusive lock can be merged with a success another writer
    committed in between, producing `refusal_count = 1` with `last_refusal_at <= last_success_at`.
    That entry is unreadable, and before the fix one unreadable entry discarded the WHOLE ledger,
    so an unrelated shape's experience was destroyed by a race on a different shape.
    """
    import flash.runner as runner
    from flash.providers import capacity_experience

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    shape = ("runpod", "H100", 1)
    other = ("vast", "B200", 8)

    capacity_experience.record_capacity_refusal(shape, now=100.0)
    capacity_experience.record_capacity_success(shape, now=200.0)
    capacity_experience.record_capacity_refusal(other, now=150.0)
    # the stale pre-lock sample: predates the success it is merged with.
    capacity_experience.record_capacity_refusal(shape, now=120.0)

    experience = capacity_experience.read_capacity_experience()
    # the unrelated shape survives, which is what the whole-ledger discard used to destroy.
    assert other in experience
    entry = experience[shape]
    active = entry.last_refusal_at is not None and (
        entry.last_success_at is None or entry.last_refusal_at > entry.last_success_at
    )
    assert active == (entry.refusal_count > 0)
    # a market that has since admitted this shape is not demoted by an older refusal.
    assert shape not in capacity_experience.recent_capacity_refusals(experience, now=210.0)


def test_stale_success_under_an_active_refusal_keeps_the_entry_consistent(monkeypatch, tmp_path):
    """The mirror case: a success OLDER than the standing refusal must not zero its count.

    The success branch zeroes `refusal_count` while keeping `last_refusal_at`. If the success
    sample predates that refusal, the entry claims an active refusal with a zero count -- the
    exact disagreement `_entry_from_json` rejects, which drops the shape's hint on the next read.
    """
    import flash.runner as runner
    from flash.providers import capacity_experience

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    shape = ("runpod", "H100", 1)
    other = ("vast", "B200", 8)

    capacity_experience.record_capacity_refusal(other, now=150.0)
    capacity_experience.record_capacity_refusal(shape, now=2000.0)
    # the stale success: predates the refusal that is still standing.
    capacity_experience.record_capacity_success(shape, now=1000.0)

    experience = capacity_experience.read_capacity_experience()
    assert other in experience, "an unrelated shape must not be collateral damage"
    assert shape in experience, "the entry must remain readable rather than being dropped"
    entry = experience[shape]
    active = entry.last_refusal_at is not None and (
        entry.last_success_at is None or entry.last_refusal_at > entry.last_success_at
    )
    assert active == (entry.refusal_count > 0)
    # the standing refusal is newer than the stale success, so the demotion must survive.
    assert shape in capacity_experience.recent_capacity_refusals(experience, now=2010.0)


def test_one_unreadable_entry_does_not_discard_the_others(monkeypatch, tmp_path):
    """A single corrupt ENTRY costs that shape's hint, never every other shape's."""
    import flash.runner as runner
    from flash.providers import capacity_experience

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    good = ("runpod", "H100", 1)
    capacity_experience.record_capacity_refusal(good, now=1000.0)

    path = Path(capacity_experience._paths()[1])
    raw = json.loads(path.read_text())
    # a refusal time with a zero count: the exact disagreement _entry_from_json rejects.
    raw["shapes"]["vast\x1fB200\x1f8"] = {
        "last_refusal_at": 10.0,
        "refusal_count": 0,
        "last_success_at": None,
    }
    path.write_text(json.dumps(raw))

    experience = capacity_experience.read_capacity_experience()
    assert good in experience
    assert ("vast", "B200", 8) not in experience


def _forgiven_shapes_after_failure(monkeypatch, tmp_path, failure, *, unreconciled_create=False):
    """Drive the real _handle_failure and return the shapes it forgave on the durable ledger."""
    import flash.runner as runner
    from flash.providers.base import Candidate, PollResult
    from flash.runner.supervise import lifecycle as _lifecycle
    from flash.runner.supervise import seed_submission

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    forgiven: list[tuple] = []
    monkeypatch.setattr(
        seed_submission._capacity_experience, "record_capacity_success", forgiven.append
    )
    monkeypatch.setattr(
        seed_submission._capacity_experience, "record_capacity_refusal", lambda _shape: None
    )
    # the failure branch is the subject; keep every collaborator around it inert.
    monkeypatch.setattr(_lifecycle, "_await_runpod_completed_metrics", lambda *a, **k: None)
    # _handle_failure imports this from flash.runner inside the body, so patch it at the source.
    monkeypatch.setattr(runner, "_load_run_deadline_at", lambda _run_id: None)

    candidate = Candidate("runpod", "H100", hourly_usd=1.0, vram_gb=80, gpu_count=1)
    outcome = SimpleNamespace(
        result=PollResult(False, failure=failure, detail="d"),
        chosen=candidate,
        candidates=(candidate,),
        run_spec=SimpleNamespace(gpu=SimpleNamespace(network_volume=None)),
        quote_refresh_failed=False,
    )
    ctx = SimpleNamespace(
        seed=0,
        spec=SimpleNamespace(
            run_id="cap-exp",
            gpu=SimpleNamespace(type="H100", provider="runpod", count=1),
        ),
        capacity_refusals={_lifecycle._shape_key(candidate): 1},
        failed_providers=set(),
        tried_classes=set(),
        oom_vram_floor=0.0,
        drop_weight_cache=False,
        retry_budget=SimpleNamespace(
            can_retry=lambda *a, **k: False, record_retry=lambda *a, **k: None
        ),
        last_handle=None,
        last_detail="",
        log=None,
        raise_if_cancelled=lambda: None,
        return_completed_runpod_metrics=lambda metrics: metrics,
        unreconciled_create=unreconciled_create,
    )
    seed_submission._handle_failure(ctx, SimpleNamespace(attempt=0), outcome)
    return forgiven, ctx.capacity_refusals


@pytest.mark.parametrize("failure", ["job_failed", "stalled", "job_preempted", "oom"])
def test_admitted_failure_forgives_a_prior_refusal(monkeypatch, tmp_path, failure):
    """A worker that reached the box and died there proves the shape had capacity."""
    forgiven, in_run = _forgiven_shapes_after_failure(monkeypatch, tmp_path, failure)
    assert forgiven == [("runpod", "H100", 1)], f"{failure} must forgive the admitted shape"
    assert in_run == {}, f"{failure} must also clear the in-run tally"


@pytest.mark.parametrize("failure", ["no_capacity", "poll_error"])
def test_unadmitted_failure_does_not_forgive(monkeypatch, tmp_path, failure):
    """no_capacity and poll_error can both precede admission, so neither clears a refusal."""
    forgiven, _in_run = _forgiven_shapes_after_failure(monkeypatch, tmp_path, failure)
    assert forgiven == [], f"{failure} is ambiguous and must not forgive"


def test_unreconciled_create_job_failed_does_not_forgive(monkeypatch, tmp_path):
    """An ambiguous create reports job_failed without any worker reaching a box."""
    # _submit_provider maps UnreconciledCreateError to job_failed at the create boundary. treating
    # that as proof of capacity would clear a real plane-wide refusal for every nearby run.
    forgiven, in_run = _forgiven_shapes_after_failure(
        monkeypatch, tmp_path, "job_failed", unreconciled_create=True
    )
    assert forgiven == [], "an unreconciled create proves nothing about capacity"
    assert in_run == {("runpod", "H100", 1): 1}, "the standing refusal must survive"


def test_unreconciled_create_flag_is_set_only_at_the_create_boundary(monkeypatch, tmp_path):
    """The flag the forgive branch reads is really raised by an UnreconciledCreateError."""
    import flash.runner as runner
    from flash.providers.base import UnreconciledCreateError
    from flash.runner.supervise import seed_submission

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    ctx = SimpleNamespace(
        spec=SimpleNamespace(run_id="cap-exp"),
        log=None,
        on_handle=lambda _handle: None,
        source_snapshot=None,
        seed=0,
        unreconciled_create=False,
    )
    candidate = SimpleNamespace(provider="runpod", gpu="H100", gpu_count=1)
    plan = SimpleNamespace(
        chosen=candidate,
        run_spec=SimpleNamespace(gpu=SimpleNamespace(network_volume=None)),
        on_last_gpu=False,
        candidates=(candidate,),
    )
    prepared = SimpleNamespace(attempt=0, runtime_secrets={})

    provider = SimpleNamespace(
        submit_run=lambda *a, **k: (_ for _ in ()).throw(UnreconciledCreateError("ambiguous"))
    )
    monkeypatch.setattr("flash.providers.get_provider", lambda _name: provider)
    monkeypatch.setattr(runner, "_load_run_deadline_at", lambda _run_id: 1e12)
    monkeypatch.setattr(runner, "_worker_deadline_at", lambda *a, **k: 1e12)
    monkeypatch.setattr(
        "flash.server.domain.teacher_broker.teacher_attempt_transport",
        lambda *a, **k: contextlib.nullcontext({}),
    )

    result, _quote_refresh_failed = seed_submission._submit_provider(ctx, prepared, plan)

    assert result.failure == "job_failed"
    assert ctx.unreconciled_create is True
