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
    from flash.providers.core.base import PollResult
    from flash.providers.mithril import jobs as mithril_jobs

    captured = {}
    polled = PollResult(True)

    def fake_poll(adapter, **kwargs):
        captured.update(kwargs, failure=adapter.load_timeout_failure)
        return polled

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
    assert mithril_jobs.poll(handle, spec, deadline_at=10_000) is polled
    assert captured["load_timeout_s"] == mithril_jobs.BOOT_TIMEOUT_S == 1800
    assert captured["failure"] == "boot_timeout"


def test_a_vm_that_never_booted_cools_its_region_for_every_run(monkeypatch):
    from flash.providers.core.base import PollResult
    from flash.providers.mithril import api as mithril_api
    from flash.providers.mithril import jobs as mithril_jobs
    from flash.providers.mithril import region_health

    timed_out = PollResult(False, failure="boot_timeout", detail="remained starting")
    monkeypatch.setattr(mithril_jobs, "poll_instance_job", lambda adapter, **kw: timed_out)
    monkeypatch.setattr(mithril_jobs, "require_scope", lambda _: None)
    bid = {"fid": "bid_test", "region": "af-south1-a", "instances": []}
    monkeypatch.setattr(mithril_api, "_bid", lambda bid_id: bid)
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
    # a bid that never filled says nothing about the region
    assert mithril_jobs.poll(handle, spec, deadline_at=10_000) is timed_out
    assert not region_health.cooling_down("af-south1-a")
    # an allocated VM that never booted does
    bid["instances"] = ["inst_x"]
    log = io.StringIO()
    assert mithril_jobs.poll(handle, spec, log=log, deadline_at=10_000) is timed_out
    assert "skipping af-south1-a" in log.getvalue()
    assert region_health.cooling_down("af-south1-a")
    assert not region_health.cooling_down("us-central2-b")
    later = time.time() + region_health.COOLDOWN_S + 1
    assert not region_health.cooling_down("af-south1-a", now=later)


def test_region_cooldown_is_not_a_run_record_and_keeps_every_mark(monkeypatch, tmp_path):
    from flash.providers.mithril import region_health
    from flash.runner.lifecycle import state

    monkeypatch.setattr(state, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(
        region_health, "_path", lambda: tmp_path / ".spot-region-health" / "cooldown.json"
    )
    region_health.mark_unbootable("af-south1-a")
    region_health.mark_unbootable("eu-x")
    assert region_health.cooling_down("af-south1-a")
    assert region_health.cooling_down("eu-x")
    assert not list(tmp_path.glob("*.json"))  # run scanners read top-level *.json as run records


def test_spot_allocation_prefers_cheaper_1x_h200_over_8x_h100():
    """Regression: a >80 GB plan took 8x H100 ($16/hr, never under cap) over a fitting 1x H200."""
    from flash.providers.core.allocator import _cheapest_allocation
    from flash.providers.core.base import Candidate

    allocation = _cheapest_allocation(
        [Candidate("spot", "H100", 2.0, 80, 8), Candidate("spot", "H200", 2.5, 141, 1)],
        need=85,
        cost_per_step=None,
        provider_rank={},
    )
    assert (allocation.gpu, allocation.gpu_count) == ("H200", 1)


def test_spot_allocation_prefers_cheaper_1x_h100_over_1x_h200():
    from flash.providers.core.allocator import _cheapest_allocation
    from flash.providers.core.base import Candidate

    allocation = _cheapest_allocation(
        [Candidate("spot", "H200", 2.5, 141, 1), Candidate("spot", "H100", 2.0, 80, 1)],
        need=40,
        cost_per_step=None,
        provider_rank={},
    )
    assert (allocation.gpu, allocation.gpu_count) == ("H100", 1)


def test_spot_quote_prefers_one_h200_over_eight_h100s(monkeypatch):
    from flash.cost import RunConfig, analytical

    monkeypatch.setattr(analytical, "required_vram_gb", lambda *a, **k: 85)
    gpu, _need, count, provider, _hourly = analytical._offline_gpu_shape(
        RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, provider="spot")
    )
    assert (provider, gpu, count) == ("spot", "H200", 1)


def test_spot_quote_names_spot_as_unsupported_when_no_shape_fits(monkeypatch):
    from flash.cost import RunConfig, analytical

    monkeypatch.setattr(analytical, "required_vram_gb", lambda *a, **k: 5000)
    with pytest.raises(
        ValueError, match=r"Spot is not supported for Qwen/Qwen3.5-9B with algorithm 'grpo'"
    ):
        analytical._offline_gpu_shape(RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, provider="spot"))


def test_spot_allocator_names_spot_as_unsupported_when_no_shape_fits():
    from flash.providers.core.allocator import _raise_no_candidate_error
    from flash.providers.core.base import UnsupportedGpuError

    with pytest.raises(UnsupportedGpuError, match=r"Spot is not supported .* 5000 GB.*8x H200"):
        _raise_no_candidate_error(
            model_id="Qwen/Qwen3.5-9B",
            algorithm="custom",
            need=5000,
            cap=8,
            exact="",
            acceptable=(),
            supported_available=("spot",),
            structurally_unsupported={},
            lookup_failed=False,
            executed_width=lambda n: n,
        )


def test_spot_shape_price_is_class_rate_times_cards():
    from decimal import Decimal

    from flash.providers.mithril.pricing import shape_price

    assert shape_price("H100", 1) == Decimal("2.00")
    assert shape_price("H200", 1) == Decimal("2.50")
    assert shape_price("H100", 8) == Decimal("16.00")
    assert shape_price("H200", 8) == Decimal("20.00")


def test_spot_quote_keeps_ceiling_error_when_a_wider_spot_shape_fits(monkeypatch):
    """Review: a 1-card ceiling on a run 8 cards could hold is a ceiling problem, not unsupported."""
    from flash.cost import RunConfig, analytical

    monkeypatch.setattr(analytical, "required_vram_gb", lambda *a, **k: 200)
    with pytest.raises(ValueError, match=r"Raise the card ceiling") as error:
        analytical._offline_gpu_shape(
            RunConfig("Qwen/Qwen3.5-9B", "grpo", 10, provider="spot", gpu_count=1)
        )
    assert "Spot is not supported" not in str(error.value)


def test_spot_allocator_keeps_ceiling_error_when_a_wider_spot_shape_fits():
    from flash.providers.core.allocator import _raise_no_candidate_error

    with pytest.raises(ValueError, match=r"no allocatable GPU") as error:
        _raise_no_candidate_error(
            model_id="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            need=200,
            cap=1,
            exact="",
            acceptable=(),
            supported_available=("spot",),
            structurally_unsupported={},
            lookup_failed=False,
            executed_width=lambda n: n,
        )
    assert "Spot is not supported" not in str(error.value)
