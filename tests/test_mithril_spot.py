"""Spot pricing, recovery and ownership contracts; no paid resources."""

import io
import time
from types import SimpleNamespace

import pytest

from flash.providers.core.registry import get_provider, validated_provider_preferences
from flash.providers.mithril import api, prices, queue
from flash.providers.mithril.pricing import accepts_price


@pytest.mark.parametrize(
    ("gpu", "value", "accepted"),
    [
        ("H100", "0.79", True),
        ("H100", "2.00", True),
        ("H100", "2.01", False),
        ("H200", "2.50", True),
        ("H200", "2.51", False),
        ("L40S", "1.00", False),
        ("H100", "NaN", False),
        ("H100", "Infinity", False),
        ("H100", "-1", False),
        ("H100", True, False),
    ],
)
def test_price_caps(gpu, value, accepted):
    assert accepts_price(gpu, value, currency="usd", unit="GPU hour") is accepted


def test_spot_cannot_fall_back_to_modal():
    with pytest.raises(ValueError, match="cannot fall back"):
        validated_provider_preferences(["spot", "modal"])
    assert get_provider("spot").name == "spot"
    assert not get_provider("spot").supports_weight_cache


def test_spot_structural_fit_does_not_invent_four_gpu_preset():
    from flash.providers.core.allocator import _structurally_fits

    assert not _structurally_fits(("spot",), 200, 4, lambda n: n, ("H100",))
    assert _structurally_fits(("spot",), 200, 8, lambda n: n, ("H100",))


def test_queue_is_cancellable(monkeypatch):
    from flash.runner.lifecycle import status
    from flash.runner.supervise.errors import _RunCancelled

    monkeypatch.setattr(status, "get_status", lambda run: SimpleNamespace(state="cancelled"))
    with pytest.raises(_RunCancelled):
        queue.wait_for_price("H100", "run", deadline_at=time.time() + 100)


def test_queue_deadline_does_not_provision():
    with pytest.raises(TimeoutError, match="deadline"):
        queue.wait_for_price("H100", "run", deadline_at=0)


def test_spot_handle_roundtrips_through_cleanup_identity():
    from flash.providers.mithril.handles import SpotJobHandle
    from flash.runner.lifecycle.cleanup import _remote_resource_identity

    handle = SpotJobHandle(
        instance_id="bid_test",
        gpu="H200",
        hourly_usd=1,
        attempt=0,
        started_ts=100,
        name="flash-test-a0",
        run_id="flash-test",
        hf_repo="org/artifacts",
        phase="sft",
        scope="test",
    )
    assert _remote_resource_identity(handle.to_dict()) == (
        "spot",
        0,
        "bid_test",
        "flash-test-a0",
        "flash-test",
    )


def test_spot_estimate_is_offline_and_uses_fixed_cap(monkeypatch):
    from flash.cost import RunConfig, estimate_cost

    def no_network(region):
        pytest.fail("offline quote must not query spot market")

    monkeypatch.setattr(prices, "affordable_offers", no_network)
    config = {"model_id": "Qwen/Qwen3.5-9B", "method": "sft", "steps": 100, "gpu_type": "H100"}
    spot = estimate_cost(RunConfig(**config, provider="spot"))
    modal = estimate_cost(RunConfig(**config, provider="modal"))
    assert spot.gpu_hourly_usd == 2.5  # quoted at the costliest class an H100 run may take
    assert spot.total_usd < modal.total_usd
    assert spot.provider == "spot"


def test_spot_preference_never_quotes_modal_when_unconfigured(monkeypatch):
    from flash.cost import RunConfig, estimate_cost
    from flash.providers.core import registry

    monkeypatch.setattr(registry, "available_providers", lambda: ("modal",))
    result = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "sft", 10, providers=("spot",)))
    assert result.provider == "spot"


def test_allocation_price_wait_does_not_consume_attempts(monkeypatch):
    from flash.providers.core.base import PollResult
    from flash.runner.lifecycle import deadlines, status
    from flash.runner.supervise import execution as supervision

    now = [100.0]
    states = []
    checks = []
    ctx = SimpleNamespace(
        spec=SimpleNamespace(run_id="run"),
        log=io.StringIO(),
        raise_if_cancelled=lambda: checks.append(1),
    )
    prepared = (None, SimpleNamespace(gpu=SimpleNamespace(provider="spot", providers=())))
    answers = iter([(None, PollResult(False, failure="no_capacity")), ("allocation", None)])
    monkeypatch.setattr(supervision, "_allocate_attempt_once", lambda *args: next(answers))
    monkeypatch.setattr(supervision.time, "time", lambda: now[0])
    monkeypatch.setattr(
        supervision.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(deadlines, "_load_run_deadline_at", lambda run: 500)
    monkeypatch.setattr(status, "_update", lambda run, state: states.append(state))
    assert supervision._allocate_attempt(ctx, prepared) == ("allocation", None)
    assert now[0] == 160
    assert states == ["queued"]
    assert len(checks) >= 60


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_spot_automatic_checkpoint_policy_survives_worker_roundtrip(algorithm):
    from flash.core.spec import GpuSpec, JobSpec

    spec = JobSpec(algorithm=algorithm, gpu=GpuSpec(provider="spot"))
    assert spec.train.save_every is None
    assert spec.train.save_at_steps == ()
    assert JobSpec.from_json(spec.to_json()).train.save_every is None


def test_exact_model_exports_cannot_disable_spot_recovery_saves():
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    spec = JobSpec(gpu=GpuSpec(provider="spot"), train=TrainSpec(max_steps=35, save_at_steps=(17,)))
    assert spec.train.save_at_steps == (17,)
    assert JobSpec.from_json(spec.to_json()).train == spec.train


def test_spot_rejects_manual_interval_but_modal_keeps_it():
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    train = TrainSpec(save_every=100)
    with pytest.raises(ValueError, match="time-based and automatic"):
        JobSpec(gpu=GpuSpec(provider="spot"), train=train)
    assert JobSpec(gpu=GpuSpec(provider="modal"), train=train).train.save_every == 100


@pytest.mark.parametrize("enabled", [True, False])
def test_toml_editable_spot_field(enabled):
    import tomllib

    from flash.core.spec import JobSpec
    from flash.schema import spec_from_dict

    raw = tomllib.loads(f"[gpu]\nspot = {str(enabled).lower()}")
    raw.update(model="Qwen/Qwen3.5-9B", algorithm="sft", environment={"id": "test/project/env"})
    spec = spec_from_dict(raw, project_required=False)
    assert spec.gpu.spot is enabled
    assert (spec.gpu.provider == "spot") is enabled
    assert JobSpec.from_json(spec.to_json()).gpu.spot is enabled


def test_spot_toml_rejects_conflicting_provider_and_non_boolean():
    from flash.schema import ConfigError, spec_from_dict

    for gpu in [
        {"spot": "true"},
        {"spot": True, "provider": "modal"},
        {"spot": False, "provider": "spot"},
    ]:
        with pytest.raises(ConfigError, match="spot"):
            spec_from_dict(
                {
                    "model": "Qwen/Qwen3.5-9B",
                    "algorithm": "sft",
                    "environment": {"id": "test/project/env"},
                    "gpu": gpu,
                },
                project_required=False,
            )


@pytest.mark.parametrize("run_id", ["custom-run", "flash-" + "x" * 60])
@pytest.mark.parametrize("callbacks", [False, True])
def test_orphan_sweep_compares_original_run_ids(monkeypatch, run_id, callbacks):
    instances = [
        {"metadata": {"id": name, "labels": {"flash-run": label}}}
        for name, label in (
            ("terminal", run_id),
            ("active", run_id + "-active"),
            ("unknown", "another-run"),
        )
    ]
    monkeypatch.setattr(api, "list_instances", lambda: instances)
    deleted = []
    monkeypatch.setattr(api, "terminate", deleted.append)
    active = {run_id + "-active"}
    known = {run_id, *active}
    provider = get_provider("spot")
    result = provider.sweep_orphans(
        (lambda: active) if callbacks else active,
        (lambda: known) if callbacks else known,
    )
    assert result == deleted == ["terminal"]


def test_capsule_rejects_public_container_before_using_worker_sas(monkeypatch):
    from contextlib import contextmanager

    from flash.providers.mithril import bootstrap

    @contextmanager
    def container(token=None):
        assert token is None, "must reject public storage before using the worker capability"
        yield SimpleNamespace(
            get_container_properties=lambda: SimpleNamespace(public_access="blob")
        )

    monkeypatch.setattr(bootstrap, "open_container", container)
    with pytest.raises(ValueError, match="must be private"):
        bootstrap.publish_capsule(
            {"hf_repo": "test/artifacts", "env": {"FLASH_ARTIFACT_BLOB_SAS_URL": "worker-sas"}},
            b"capsule",
            deadline_at=time.time() + 60,
        )


def test_stopped_spot_worker_requeues_until_price_recovers(monkeypatch):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core.base import Candidate
    from flash.providers.mithril import jobs
    from flash.providers.mithril.handles import SpotJobHandle
    from flash.runner.lifecycle import status
    from flash.runner.supervise.retry_decision import (
        FailureObservation,
        RetryState,
        transition_failure,
    )

    now = [100.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    spec = JobSpec(
        run_id="flash-test",
        gpu=GpuSpec(provider="spot", max_retries=0),
        train=TrainSpec(hf_repo="org/artifacts"),
    )
    handle = SpotJobHandle(
        instance_id="bid_test",
        gpu="H100",
        hourly_usd=1,
        attempt=0,
        started_ts=90,
        name="flash-test-a0",
        run_id=spec.run_id,
        hf_repo=spec.train.hf_repo,
        phase=spec.phase,
        scope="test",
    )
    monkeypatch.setattr(jobs, "require_scope", lambda handle: None)
    monkeypatch.setattr(jobs, "make_artifact_text_reader", lambda *a, **kw: lambda **kw: None)
    monkeypatch.setattr(jobs, "heartbeat_reader_for", lambda *a, **kw: None)
    monkeypatch.setattr(api, "get_instance", lambda *a, **kw: {"status": "stopped"})
    result = jobs.poll(handle, spec, log=io.StringIO(), deadline_at=500)
    assert result.failure == "job_preempted"
    chosen = Candidate("spot", "H100", 1, 80, 1)
    _, plan = transition_failure(
        RetryState.initial_for_spec(spec),
        FailureObservation(result.failure, chosen, (chosen,)),
        attempt=0,
    )
    assert plan.retry

    states = []
    prices_to_check = iter([[], [], [prices.Offer("H100", 1, "it_test", 0.9, 1500, "test")]])
    monkeypatch.setattr(queue, "quota_available", lambda *a, **kw: True)
    monkeypatch.setattr(queue, "affordable_offers", lambda **kw: next(prices_to_check))
    monkeypatch.setattr(status, "get_status", lambda run: SimpleNamespace(state="queued"))
    monkeypatch.setattr(status, "_update", lambda run, state: states.append(state))
    queued_at = now[0]
    queue.wait_for_price("H100", spec.run_id, deadline_at=500, log=io.StringIO())
    assert now[0] == queued_at + 120
    assert states == ["queued", "queued", "provisioning"]


@pytest.mark.parametrize(
    ("spot", "error", "failure"),
    [
        # a gracefully stopped Spot VM: SIGTERM reaches the worker before the host disappears
        (True, "WorkerInterrupted: worker interrupted by signal 15", "job_preempted"),
        # other transient worker faults keep the bounded infrastructure retry
        (True, "RetriableBootstrapError: train phase exited non-zero", "worker_retriable"),
        # Modal (no flag) keeps its existing classification
        (False, "WorkerInterrupted: worker interrupted by signal 15", "worker_retriable"),
    ],
)
def test_spot_signal_interruption_is_preemption(spot, error, failure):
    from flash.providers._lifecycle.instances.poll_instance import _TerminalArtifacts

    adapter = SimpleNamespace(
        signal_interruption_is_preemption=spot,
        current_attempt=2,
        failure_detail=lambda marker: (marker or {}).get("error") or "",
    )
    artifacts = _TerminalArtifacts(
        adapter, say=print, heartbeat_reader=None, launch_ts=0.0, absolute_deadline=None
    )
    result = artifacts._fail_from_marker({"retriable": True, "error": error})
    assert result.failure == failure
    assert result.detail == error


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "custom"])
def test_mithril_signal_preemption_requeues_without_retry_budget(monkeypatch, algorithm):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers._lifecycle.instances.poll_instance import _TerminalArtifacts
    from flash.providers.core.base import Candidate
    from flash.providers.mithril import jobs
    from flash.providers.mithril.handles import SpotJobHandle
    from flash.runner.supervise.retry_decision import (
        FailureObservation,
        RetryState,
        transition_failure,
    )

    spec = JobSpec(
        algorithm=algorithm,
        run_id="run",
        gpu=GpuSpec(provider="spot", max_retries=0),
        train=TrainSpec(hf_repo="org/repo"),
    )
    handle = SpotJobHandle(
        instance_id="bid_test",
        gpu="H100",
        hourly_usd=1,
        attempt=0,
        started_ts=100,
        name="test",
        run_id="run",
        hf_repo="org/repo",
        phase=spec.phase,
        scope="test",
    )
    monkeypatch.setattr(jobs, "require_scope", lambda _: None)
    monkeypatch.setattr(jobs, "make_artifact_text_reader", lambda *a, **kw: lambda **kw: None)
    monkeypatch.setattr(jobs, "heartbeat_reader_for", lambda *a, **kw: None)

    def interrupted(adapter, **kwargs):
        artifacts = _TerminalArtifacts(
            adapter, say=print, heartbeat_reader=None, launch_ts=100, absolute_deadline=None
        )
        return artifacts._fail_from_marker(
            {"retriable": True, "error": "WorkerInterrupted: worker interrupted by signal 15"}
        )

    monkeypatch.setattr(jobs, "poll_instance_job", interrupted)
    result = jobs.poll(handle, spec)
    assert result.failure == "job_preempted"
    chosen = Candidate("spot", "H100", 1, 80, 1)
    state, plan = transition_failure(
        RetryState.initial_for_spec(spec),
        FailureObservation(result.failure, chosen, (chosen,)),
        attempt=0,
    )
    assert plan.retry
    assert state.infra_used == 0


def test_entry_and_running_caps_do_not_change_fixed_customer_rates():
    from flash.providers.mithril.pricing import SPOT_GPUS, hourly_rate

    assert list(SPOT_GPUS) == ["H100", "H200"]  # Mithril offers only these
    assert [hourly_rate(gpu) for gpu in SPOT_GPUS] == [2.0, 2.5]
    assert [str(shape.keep_running_usd) for shape in SPOT_GPUS.values()] == ["2.01", "2.51"]
    assert not any(hasattr(shape, "entry_below_usd") for shape in SPOT_GPUS.values())


def test_runs_on_h100_may_take_h200_but_not_the_reverse():
    from flash.providers.mithril.pricing import interchangeable

    assert interchangeable("H100", 1) == ("H100", "H200")
    assert interchangeable("H100", 8) == ("H100", "H200")
    assert interchangeable("H200", 1) == ("H200",)  # an 80 GB card cannot host a 141 GB plan


@pytest.mark.parametrize(
    ("live", "expected"),
    [
        (["H200"], "H200"),  # whichever comes up first
        (["H100"], "H100"),
        (["H200", "H100"], "H100"),  # both at once: class priority
    ],
)
def test_queue_takes_whichever_acceptable_class_comes_up_first(monkeypatch, live, expected):
    from flash.runner.lifecycle import status

    offers = [prices.Offer(gpu, 1, f"it_{gpu}", 1.5, 1500, "r") for gpu in live]
    answers = iter([[], offers])
    now = [0.0]
    monkeypatch.setattr(queue.time, "time", lambda: now[0])
    monkeypatch.setattr(queue.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    monkeypatch.setattr(queue, "quota_available", lambda *a, **kw: True)
    monkeypatch.setattr(queue, "affordable_offers", lambda **kw: next(answers))
    monkeypatch.setattr(status, "get_status", lambda run: SimpleNamespace(state="queued"))
    monkeypatch.setattr(status, "_update", lambda run, state: None)
    log = io.StringIO()
    offer = queue.wait_for_price(("H100", "H200"), "test", deadline_at=500, log=log)
    assert offer.gpu == expected
    assert "H100 or H200" in log.getvalue()


def test_attempt_spec_names_the_class_that_came_up(monkeypatch):
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.mithril import jobs
    from flash.providers.mithril import queue as spot_queue

    spec = JobSpec(
        run_id="r", model="Qwen/Qwen3.5-9B-Base", gpu=GpuSpec(type="H100", provider="spot")
    )
    seen = {}
    monkeypatch.setattr(jobs, "scope_fingerprint", lambda: "scope")

    def wait(gpus, run_id, **kw):
        seen["waited_for"] = gpus
        return prices.Offer("H200", 1, "it_h200", 1.5, 1500, "r")

    monkeypatch.setattr(spot_queue, "wait_for_price", wait)

    def build_payload(spec, attempt, **kw):
        seen["gpu"] = spec.gpu.type
        raise RuntimeError("stop before launch")

    monkeypatch.setattr(jobs, "build_payload", build_payload)
    with pytest.raises(RuntimeError, match="stop before launch"):
        jobs.submit_attempt(spec, deadline_at=time.time() + 600, log=io.StringIO())
    assert seen == {"waited_for": ("H100", "H200"), "gpu": "H200"}


def test_mithril_boot_window_and_failure(monkeypatch):
    """A filled bid gets 30 minutes to boot; a VM that never boots is a boot_timeout, not a stall."""
    from flash.providers.mithril import jobs as mithril_jobs

    captured = {}

    def fake_poll(adapter, **kwargs):
        captured.update(kwargs, failure=adapter.load_timeout_failure)
        return "polled"

    monkeypatch.setattr(mithril_jobs, "poll_instance_job", fake_poll)
    monkeypatch.setattr(mithril_jobs, "require_scope", lambda _: None)
    handle = SimpleNamespace(
        to_dict=lambda: {
            "provider": "spot",
            "instance_id": "bid_test",
            "name": "run-a0",
            "run_id": "run",
            "phase": "custom",
            "hf_repo": "org/repo",
            "attempt": 0,
            "gpu": "H200",
            "gpu_count": 1,
            "hourly_usd": 2.5,
            "scope": "x",
            "started_ts": 1.0,
        }
    )
    spec = SimpleNamespace(run_id="run", phase="custom", train=SimpleNamespace(hf_repo="org/repo"))
    assert mithril_jobs.poll(handle, spec, deadline_at=10_000) == "polled"
    assert captured["load_timeout_s"] == mithril_jobs.BOOT_TIMEOUT_S == 1800
    assert captured["failure"] == "boot_timeout"
