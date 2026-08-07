"""regression tests for the single-field gpu.type merge (PR #670).

collapsing gpu.exact_type into a pinning gpu.type dropped the old non-empty gpu.type
default that used to seed sizing. two consumers regressed on the now-empty auto value:

- prepare_job's open-model preflight fed the empty public gpu.type straight to
  resolve_model, whose _resolve_open_model falls back to DEFAULT_GPU. an uncataloged
  model larger than DEFAULT_GPU but fitting a managed class then passed schema
  validation (which sizes against provisional_gpu) yet was rejected at prepare_job.
- recovery re-allocation reloaded the persisted effective worker_spec whose gpu.type
  had been overwritten with the concrete allocated class, hard-pinning an originally
  auto run to the prior attempt's class after a control-plane restart/attach.
"""

from __future__ import annotations

import tempfile

import pytest

from flash.catalog import DEFAULT_GPU
from flash.providers.base import provisional_gpu
from flash.spec import GpuSpec, JobSpec, TrainSpec


def _auto_open_spec(run_id: str, *, gpu_type: str = "") -> JobSpec:
    return JobSpec(
        run_id=run_id,
        model="acme/mid-14b",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=1),
        gpu=GpuSpec(type=gpu_type, max_retries=2),
        model_policy="allow",
    )


def test_prepare_job_open_model_auto_sizes_against_provisional(monkeypatch):
    # a 14b open model exceeds DEFAULT_GPU (RTX 5090, 32gb) but fits a larger managed class.
    monkeypatch.setattr(
        "flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: 14.0, raising=True
    )
    import flash.runner as runner

    spec = _auto_open_spec("open-auto-preflight")
    expected = provisional_gpu(
        spec.model,
        spec.algorithm,
        train=spec.train,
        thinking=spec.thinking,
        model_revision=spec.model_revision,
    )
    # precondition: the provisional class the schema validated against is a real, larger
    # class -- not the empty public value and not the DEFAULT_GPU fallback -- otherwise the
    # regression would be invisible (empty resolves to DEFAULT_GPU internally anyway).
    assert expected not in ("", DEFAULT_GPU)

    class ReachedModelResolution(Exception):
        pass

    captured: dict[str, object] = {}

    def capture(*_args, **kwargs):
        captured["gpu"] = kwargs.get("gpu")
        raise ReachedModelResolution

    monkeypatch.setattr(runner, "resolve_model", capture)

    with pytest.raises(ReachedModelResolution):
        runner.prepare_job(spec)

    # prepare_job must size the fit preflight against the provisional managed class, not
    # the empty public gpu.type that would fall back to DEFAULT_GPU and wrongly reject.
    assert captured["gpu"] == expected


def test_prepare_job_pinned_open_model_uses_pinned_type(monkeypatch):
    # a pinned run keeps passing its exact class through to resolve_model unchanged.
    monkeypatch.setattr(
        "flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: 14.0, raising=True
    )
    import flash.runner as runner

    spec = _auto_open_spec("open-pinned-preflight", gpu_type="H200")

    class ReachedModelResolution(Exception):
        pass

    captured: dict[str, object] = {}

    def capture(*_args, **kwargs):
        captured["gpu"] = kwargs.get("gpu")
        raise ReachedModelResolution

    monkeypatch.setattr(runner, "resolve_model", capture)

    with pytest.raises(ReachedModelResolution):
        runner.prepare_job(spec)

    assert captured["gpu"] == "H200"


def _persist_effective(runner, public: JobSpec, effective_type: str, effective_count: int = 0):
    runner._save_status(
        runner.RunStatus(
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
    assert runner._persist_effective_worker_spec(selected)
    return runner.get_status(public.run_id)


def test_reallocation_spec_restores_public_gpu_type_for_auto(monkeypatch):
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        runner = fresh_runner(tmp, monkeypatch)
        public = JobSpec(
            run_id="realloc-auto",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        stored = _persist_effective(runner, public, effective_type="H100")

        # polling/cleanup keep the concrete allocated class...
        assert runner.effective_spec_from_status(stored).gpu.type == "H100"
        # ...but re-allocation restores the empty public value so recovery does not
        # hard-pin an originally-auto run to the prior attempt's class.
        assert runner.reallocation_spec_from_status(stored).gpu.type == ""


def test_reallocation_spec_keeps_pin_for_pinned(monkeypatch):
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        runner = fresh_runner(tmp, monkeypatch)
        public = JobSpec(
            run_id="realloc-pinned",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="H100", max_retries=2),
        )
        stored = _persist_effective(runner, public, effective_type="H100")

        # a pinned run's public and effective type match, so re-allocation keeps the pin.
        assert runner.reallocation_spec_from_status(stored).gpu.type == "H100"
        assert runner.effective_spec_from_status(stored).gpu.type == "H100"


def test_reallocation_spec_restores_the_authored_card_ceiling_for_an_auto_run(monkeypatch):
    """A narrowed auto run recovers with its authored ceiling, not the count it was given.

    gpu.count is a CEILING the allocator may satisfy with fewer cards. _spec_with_gpu writes the
    SELECTED count onto the worker spec, so handing that snapshot back to allocate() would lower the
    ceiling to the single shape that already failed -- the retry could no longer consider any of the
    multi-card shapes the run authorized.
    """
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        runner = fresh_runner(tmp, monkeypatch)
        public = JobSpec(
            run_id="realloc-auto-count",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", count=4, max_retries=2),
        )
        stored = _persist_effective(runner, public, effective_type="H100", effective_count=1)

        # polling and cleanup keep the concrete allocation: one H100.
        live = runner.effective_spec_from_status(stored)
        assert live.gpu.type == "H100"
        assert live.gpu.count == 1

        recovered = runner.reallocation_spec_from_status(stored)
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
        runner = fresh_runner(tmp, monkeypatch)
        public = JobSpec(
            run_id="realloc-pinned-count",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="H100", count=4, max_retries=2),
        )
        stored = _persist_effective(runner, public, effective_type="H100", effective_count=2)

        assert runner.effective_spec_from_status(stored).gpu.count == 2

        recovered = runner.reallocation_spec_from_status(stored)
        assert recovered.gpu.type == "H100"
        assert recovered.gpu.count == 4, (
            "the pinned early return handed back the snapshot with its narrowed count, so a "
            "recovered 4-card run re-allocates against a ceiling of 2"
        )
