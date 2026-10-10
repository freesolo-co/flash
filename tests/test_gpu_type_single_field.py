"""regression tests for the single-field gpu.type merge (PR #670).

collapsing gpu.exact_type into a pinning gpu.type dropped the old non-empty gpu.type
default that used to seed sizing, which regressed recovery re-allocation: it reloaded
the persisted effective worker_spec whose gpu.type had been overwritten with the
concrete allocated class, hard-pinning an originally auto run to the prior attempt's
class after a control-plane restart/attach.

(the other consumer this file covered was prepare_job's open-model fit preflight, which
sized an uncataloged model against the empty public gpu.type. the open-model path is
gone -- only curated models are trainable, and a curated entry states its own
requirements rather than estimating them from a card -- so that half went with it.)
"""

from __future__ import annotations

import tempfile

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
from flash.core.spec import GpuSpec, JobSpec, TrainSpec


def _persist_effective(public: JobSpec, effective_type: str, effective_count: int = 0):
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=public.run_id,
            state="provisioning",
            spec=public.to_dict(),
        )
    )
    selected_dict = public.to_internal_dict()
    selected_dict["gpu"]["type"] = effective_type
    if effective_count:
        selected_dict["gpu"]["count"] = effective_count
    selected = JobSpec.from_dict(selected_dict)
    assert runner_submit._persist_effective_worker_spec(selected)
    return runner_status.get_status(public.run_id)


def test_reallocation_spec_restores_public_gpu_type_for_auto(monkeypatch):
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        public = JobSpec(
            run_id="realloc-auto",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        stored = _persist_effective(public, effective_type="H100")

        # polling/cleanup keep the concrete allocated class...
        assert runner_status.effective_spec_from_status(stored).gpu.type == "H100"
        # ...but re-allocation restores the empty public value so recovery does not
        # hard-pin an originally-auto run to the prior attempt's class.
        assert runner_status.reallocation_spec_from_status(stored).gpu.type == ""


def test_reallocation_spec_keeps_pin_for_pinned(monkeypatch):
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        public = JobSpec(
            run_id="realloc-pinned",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="H100", max_retries=2),
        )
        stored = _persist_effective(public, effective_type="H100")

        # a pinned run's public and effective type match, so re-allocation keeps the pin.
        assert runner_status.reallocation_spec_from_status(stored).gpu.type == "H100"
        assert runner_status.effective_spec_from_status(stored).gpu.type == "H100"


def test_reallocation_spec_restores_the_authored_card_ceiling_for_an_auto_run(monkeypatch):
    """A narrowed auto run recovers with its authored ceiling, not the count it was given.

    gpu.count is a CEILING the allocator may satisfy with fewer cards. _spec_with_gpu writes the
    SELECTED count onto the worker spec, so handing that snapshot back to allocate() would lower the
    ceiling to the single shape that already failed -- the retry could no longer consider any of the
    multi-card shapes the run authorized.
    """
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        public = JobSpec(
            run_id="realloc-auto-count",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", count=4, max_retries=2),
        )
        stored = _persist_effective(public, effective_type="H100", effective_count=1)

        # polling and cleanup keep the concrete allocation: one H100.
        live = runner_status.effective_spec_from_status(stored)
        assert live.gpu.type == "H100"
        assert live.gpu.count == 1

        recovered = runner_status.reallocation_spec_from_status(stored)
        assert recovered.gpu.type == ""
        assert recovered.gpu.count == 4, (
            "recovery re-entered allocation with the narrowed count as its ceiling, so a run "
            "authored for up to 4 cards can never be offered a multi-card shape again"
        )


def test_reallocation_spec_restores_the_authored_card_ceiling_for_a_pinned_run(monkeypatch):
    """The pinned case needs its own guard: matching gpu.type used to short-circuit the restore.

    A pinned run's public and effective types are equal, so an early return on type alone handed
    back the snapshot untouched -- narrowed count included.
    """
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        public = JobSpec(
            run_id="realloc-pinned-count",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="H100", count=4, max_retries=2),
        )
        stored = _persist_effective(public, effective_type="H100", effective_count=2)

        assert runner_status.effective_spec_from_status(stored).gpu.count == 2

        recovered = runner_status.reallocation_spec_from_status(stored)
        assert recovered.gpu.type == "H100"
        assert recovered.gpu.count == 4, (
            "the pinned early return handed back the snapshot with its narrowed count, so a "
            "recovered 4-card run re-allocates against a ceiling of 2"
        )


def test_reallocation_preserves_auto_count_provenance(monkeypatch):
    """An auto-sized run must recover as auto-sized, not as an authored one-card pin.

    The public half stores the digest-stable placeholder `count = 1` for an omitted gpu.count and
    cannot carry the marker (to_dict strips it). Restoring the public ceiling alone therefore reads
    an auto-sized run back as an authored single-card pin: allocation receives `max_gpu_count=1`
    instead of None, and a run that needs two or more cards can never be re-offered its shape after
    a control-plane restart. The snapshot is written at creation -- before allocation resolves a
    shape and clears the marker -- so the stored worker half still knows the count was unauthored.
    """
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        public = JobSpec(
            run_id="realloc-auto-marker",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type=""),
            gpu_count_auto=True,
        )
        assert public.to_dict()["gpu"]["count"] == 1
        assert "gpu_count_auto" not in public.to_dict()
        stored = _persist_effective(public, effective_type="H200", effective_count=2)

        recovered = runner_status.reallocation_spec_from_status(stored)
        assert recovered.gpu_count_auto is True, (
            "the auto marker was dropped on restore, so recovery hard-pins an auto-sized run to "
            "the placeholder count of 1 and can never re-offer it a multi-card shape"
        )


def test_reallocation_keeps_an_authored_single_card_pin_hard(monkeypatch):
    """The converse: an authored `gpu.count = 1` must never recover as auto-sized.

    Widening it would bill cards the user explicitly capped.
    """
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        public = JobSpec(
            run_id="realloc-authored-one",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", count=1),
        )
        stored = _persist_effective(public, effective_type="H100")

        recovered = runner_status.reallocation_spec_from_status(stored)
        assert recovered.gpu_count_auto is False
        assert recovered.gpu.count == 1
