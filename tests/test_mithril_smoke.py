"""Regression coverage for races found during paid GPU qualification."""

import importlib
from decimal import Decimal
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from flash.core.spec import GpuSpec, JobSpec, TrainSpec
from flash.providers.core.base import PollResult


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("smoke_mithril_sft")
    monkeypatch.setattr(module, "configured", lambda: True)
    return module


@pytest.mark.parametrize("cap", ["2.62", "NaN", "Infinity", "0", "-1"])
def test_paid_entry_override_cannot_raise_hold_cap_or_touch_storage(smoke, monkeypatch, cap):
    monkeypatch.setattr(smoke, "ArtifactStore", lambda: pytest.fail("must validate before storage"))
    with pytest.raises(ValueError, match="hold cap"):
        smoke.run("unused", gpu="H200", entry_price_cap=Decimal(cap))


@pytest.mark.parametrize("failure", [None, "job_preempted", "worker_retriable"])
def test_interruption_race_always_resumes_with_additional_steps(smoke, monkeypatch, failure):
    cancelled = Event()
    prepared_steps = []
    resumed_steps = []
    deleted = []
    updates = []
    hub = SimpleNamespace(
        create_repo=lambda *a, **kw: None,
        repo_info=lambda *a, **kw: SimpleNamespace(private=True),
        list_repo_files=lambda *a, **kw: ["sft/test/DONE"],
        delete_file=lambda **kw: deleted.append(kw["path_in_repo"]),
    )
    monkeypatch.setattr(smoke, "ArtifactStore", lambda: hub)
    monkeypatch.setattr(smoke.uuid, "uuid4", lambda: SimpleNamespace(hex="test"))
    monkeypatch.setattr(smoke, "_save_status", lambda status: None)
    monkeypatch.setattr(smoke, "_update", lambda *a, **kw: updates.append((a, kw)))
    monkeypatch.setattr(smoke, "publish_source_snapshot", lambda _: {})
    monkeypatch.setattr(smoke, "checkpoint_is_durable", lambda *a: True)
    monkeypatch.setattr(smoke.api, "list_instances", list)
    monkeypatch.setattr(smoke.api, "terminate", lambda _: cancelled.set())

    def prepare(hub, repo, run_id, volume, root, gpu, model, *, provider, steps=2):
        prepared_steps.append(steps)
        hub.list_repo_files = lambda *a, **kw: [f"sft/{run_id}/DONE"]
        return JobSpec(
            run_id=run_id,
            gpu=GpuSpec(type=gpu, provider=provider),
            train=TrainSpec(hf_repo=repo, max_steps=steps),
        )

    def submit(spec, *, on_handle, **kwargs):
        on_handle(SimpleNamespace(to_dict=lambda: {"instance_id": "bid_test"}))
        assert cancelled.wait(2), "monitor never cancelled the first bid"
        return PollResult(ok=failure is None, failure=failure)

    monkeypatch.setattr(smoke, "prepare", prepare)
    monkeypatch.setattr(smoke.jobs, "submit_attempt", submit)
    monkeypatch.setattr(
        smoke, "resume_attempt", lambda spec, *a: resumed_steps.append(spec.train.max_steps)
    )
    smoke.run("test/repo", gpu="H200")
    assert prepared_steps == [2, 4]
    assert resumed_steps == [4]
    assert deleted == ["sft/flash-spot-sft-smoke-test/DONE"]
    assert updates[-1][0][1] == "done"


def test_paid_override_changes_admission_without_changing_billing_or_hold(smoke, monkeypatch):
    from flash.providers.mithril import pricing

    monkeypatch.setattr(pricing, "SPOT_GPUS", dict(pricing.SPOT_GPUS))

    def stop_before_storage():
        assert pricing.SPOT_GPUS["H200"].keep_running_usd == Decimal("2.00")
        assert pricing.hourly_rate("H200") == 2.5
        assert pricing.running_limit("H200") == 2.00
        raise RuntimeError("stop before paid resources")

    monkeypatch.setattr(smoke, "ArtifactStore", stop_before_storage)
    with pytest.raises(RuntimeError, match="stop before paid resources"):
        smoke.run("unused", gpu="H200", entry_price_cap=Decimal("2.00"))


def test_paid_cap_bounds_every_class_the_run_could_take(smoke, monkeypatch):
    from flash.providers.mithril import pricing

    monkeypatch.setattr(pricing, "SPOT_GPUS", dict(pricing.SPOT_GPUS))

    def stop_before_storage():
        # an H100 smoke run may launch on an H200, so the $0.10 test cap must bound both.
        assert pricing.SPOT_GPUS["H100"].keep_running_usd == Decimal("0.10")
        assert pricing.SPOT_GPUS["H200"].keep_running_usd == Decimal("0.10")
        raise RuntimeError("stop before paid resources")

    monkeypatch.setattr(smoke, "ArtifactStore", stop_before_storage)
    with pytest.raises(RuntimeError, match="stop before paid resources"):
        smoke.run("unused", gpu="H100", entry_price_cap=Decimal("0.10"))
