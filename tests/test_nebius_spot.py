"""Spot admission, quote, ownership and provisioning contracts (no paid resources)."""

import io
import json
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from flash.providers.core.base import AllocationConstraints, CapacityLookupError
from flash.providers.core.registry import get_provider, validated_provider_preferences
from flash.providers.nebius import api, prices, queue
from flash.providers.nebius.pricing import SPOT_GPUS, accepts_price, hourly_rate


@pytest.mark.parametrize(
    ("gpu", "value", "accepted"),
    [
        ("H100", "0.79", True),
        ("H100", "1.00", False),
        ("H200", "1.001", False),
        ("B200", "3.999", True),
        ("B200", "4", False),
        ("H100", "NaN", False),
        ("H100", "Infinity", False),
        ("H100", "-1", False),
        ("H100", True, False),
    ],
)
def test_price_caps(gpu, value, accepted):
    assert accepts_price(gpu, value, currency="usd", unit="GPU hour") is accepted


def test_fixed_rates_and_supported_shapes():
    assert set(SPOT_GPUS) == {"H100", "H200", "B200"}
    assert hourly_rate("H100") == hourly_rate("H200") == 1
    assert hourly_rate("B200") == 4
    assert SPOT_GPUS["H200"].preset(8) == "8gpu-128vcpu-1600gb"
    with pytest.raises(ValueError, match="count"):
        SPOT_GPUS["H200"].preset(2)
    assert not accepts_price("H100", "0.8", currency="eur", unit="GPU hour")
    assert not accepts_price("H100", "0.8", currency="usd", unit="hour")


def test_spot_cannot_fall_back_to_modal():
    with pytest.raises(ValueError, match="cannot fall back"):
        validated_provider_preferences(["spot", "modal"])
    assert get_provider("spot").name == "spot"
    assert not get_provider("spot").supports_weight_cache


def test_candidates_use_current_prices_and_real_presets(monkeypatch):
    monkeypatch.setattr(prices, "affordable_gpus", lambda region: {"H100": 0.9, "H200": 0.8})
    found = get_provider("spot").live_candidates(80, AllocationConstraints(max_gpu_count=8))
    assert {(c.gpu, c.gpu_count) for c in found} == {
        ("H100", 1),
        ("H100", 8),
        ("H200", 1),
        ("H200", 8),
    }
    from flash.providers.core.allocator import _cheapest_allocation

    result = _cheapest_allocation(found, need=80, cost_per_step=None, provider_rank={})
    assert result.gpu == "H100"
    assert result.hourly_usd == 1  # fixed estimate, not market price


def test_spot_structural_fit_does_not_invent_four_gpu_preset():
    from flash.providers.core.allocator import _structurally_fits

    assert not _structurally_fits(("spot",), 200, 4, lambda n: n, ("H100",))
    assert _structurally_fits(("spot",), 200, 8, lambda n: n, ("H100",))


def _page(value="0.79", region="eu-north1", **extra):
    return {
        "items": [
            {
                "region": region,
                "computeInstance": {"platform": "gpu-h200-sxm"},
                "price": {"value": value, "currency": "usd", "unitTitle": "GPU hour"},
            }
        ],
        "snapshotTime": datetime.now(UTC).isoformat(),
        **extra,
    }


def _mock_prices(monkeypatch, pages):
    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            page = pages.pop(0)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: page)

    import httpx

    monkeypatch.setattr(httpx, "Client", Client)


def test_price_api_pagination_and_region_filter(monkeypatch):
    _mock_prices(monkeypatch, [_page(region="other", nextPageToken="two"), _page()])
    assert prices.affordable_gpus("eu-north1") == {"H200": 0.79}


@pytest.mark.parametrize("page", [_page(snapshotTime="2020-01-01T00:00:00Z"), {}])
def test_unavailable_prices_fail_closed(monkeypatch, page):
    _mock_prices(monkeypatch, [page])
    with pytest.raises(CapacityLookupError):
        prices.affordable_gpus("eu-north1")


def test_expensive_prices_do_not_admit(monkeypatch):
    _mock_prices(monkeypatch, [_page(value="1.1")])
    assert prices.affordable_gpus("eu-north1") == {}


def test_queue_waits_until_affordable_without_launching(monkeypatch):
    from flash.runner.lifecycle import status

    current = [100.0]
    states = []
    monkeypatch.setattr(queue.time, "time", lambda: current[0])
    monkeypatch.setattr(
        queue.time, "sleep", lambda seconds: current.__setitem__(0, current[0] + seconds)
    )
    monkeypatch.setattr(status, "get_status", lambda run: SimpleNamespace(state="queued"))
    monkeypatch.setattr(status, "_update", lambda run, state: states.append(state))
    answers = iter([{}, {"H200": 0.9}])
    monkeypatch.setattr(queue, "affordable_gpus", lambda region: next(answers))
    queue.wait_for_price("H200", "run", deadline_at=300, log=io.StringIO())
    assert current[0] == 160
    assert states == ["queued", "provisioning"]


def test_queue_is_cancellable(monkeypatch):
    from flash.runner.lifecycle import status
    from flash.runner.supervise.errors import _RunCancelled

    monkeypatch.setattr(status, "get_status", lambda run: SimpleNamespace(state="cancelled"))
    with pytest.raises(_RunCancelled):
        queue.wait_for_price("H100", "run", deadline_at=time.time() + 100)


def test_queue_deadline_does_not_provision():
    with pytest.raises(TimeoutError, match="deadline"):
        queue.wait_for_price("H100", "run", deadline_at=0)


@pytest.mark.parametrize(
    ("disk_gb", "expected_gib"), [(60, 279), (279, 279), (280, 372), (500, 558)]
)
def test_create_always_attaches_server_price_cap_and_owned_disk(monkeypatch, disk_gb, expected_gib):
    monkeypatch.setattr(
        api, "settings", lambda: {"project_id": "p", "subnet_id": "s", "image_id": "image"}
    )
    monkeypatch.setattr(api, "list_instances", lambda **kwargs: [])
    monkeypatch.setattr(api, "pricing_policy", lambda *args, **kwargs: "policy")
    from flash.providers.nebius import bootstrap

    monkeypatch.setattr(
        bootstrap,
        "publish_capsule",
        lambda *a, **kw: {"url": "https://example.com/capsule", "sha256": "digest"},
    )
    seen = []

    def call(*args, **kwargs):
        seen.append(kwargs["body"])
        return {"metadata": {"id": "computeinstance-test"}}

    monkeypatch.setattr(api, "call", call)
    result = api.create(
        name="flash-test-a0",
        payload={
            "run_id": "flash-test",
            "attempt": 0,
            "job_spec_json": "{}",
            "source_snapshot": {},
        },
        image_reference="worker",
        gpu="H100",
        gpu_count=1,
        capsule=b"capsule",
        disk_gb=disk_gb,
        deadline_at=time.time() + 1000,
    )
    assert result == "computeinstance-test"
    spec = seen[0]["spec"]
    assert spec["preemptible"] == {"on_preemption": "STOP"}
    assert spec["recovery_policy"] == "FAIL"
    assert spec["spot_pricing_policy"] == {"id": "policy"}
    assert "follows_spot_price" not in spec
    assert "on_demand" not in spec
    assert spec["boot_disk"]["managed_disk"]["spec"]["size_gibibytes"] == expected_gib
    assert spec["boot_disk"]["managed_disk"]["spec"]["type"] == "NETWORK_SSD_NON_REPLICATED"
    init = json.loads(spec["cloud_init_user_data"].split("\n", 1)[1])
    assert all(f["permissions"] == "0600" for f in init["write_files"])
    launch = init["runcmd"][1]
    assert launch[0] == "timeout"
    assert launch[-3:-1] == ["sh", "-c"]
    install, bootstrap_command = launch[-1].split(" && ")
    assert "'azure-storage-blob>=12.24,<13'" in install
    assert "'azure-identity>=1.19,<2'" in install
    assert bootstrap_command == "exec python -u /opt/flash/capsule.pyz bootstrap"


def test_spot_handle_roundtrips_through_cleanup_identity():
    from flash.providers.nebius.handles import SpotJobHandle
    from flash.runner.lifecycle.cleanup import _remote_resource_identity

    handle = SpotJobHandle(
        instance_id="computeinstance-test",
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
        "computeinstance-test",
        "flash-test-a0",
        "flash-test",
    )


def test_allocator_queues_expensive_spot_without_modal_fallback(monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import CapacityUnavailableError

    monkeypatch.setattr(allocator, "available_providers", lambda: ("modal", "spot"))
    monkeypatch.setattr(prices, "affordable_gpus", lambda region: {})
    with pytest.raises(CapacityUnavailableError):
        allocator.allocate("Qwen/Qwen3.5-9B", "sft", provider="spot")


def test_allocator_prefers_h100_even_when_h200_is_cheaper(monkeypatch):
    from flash.providers.core import allocator

    monkeypatch.setattr(allocator, "available_providers", lambda: ("modal", "spot"))
    monkeypatch.setattr(
        prices, "affordable_gpus", lambda region: {"H100": 0.95, "H200": 0.80, "B200": 2.0}
    )
    result = allocator.allocate("Qwen/Qwen3.5-9B", "sft", provider="spot")
    assert result.provider == "spot"
    assert result.gpu == "H100"
    assert result.gpu_count == 1
    assert result.hourly_usd == 1


def test_spot_estimate_is_offline_and_uses_fixed_cap(monkeypatch):
    from flash.cost import RunConfig, estimate_cost

    def no_network(region):
        pytest.fail("offline quote must not query spot market")

    monkeypatch.setattr(prices, "affordable_gpus", no_network)
    config = {"model_id": "Qwen/Qwen3.5-9B", "method": "sft", "steps": 100, "gpu_type": "H100"}
    spot = estimate_cost(RunConfig(**config, provider="spot"))
    modal = estimate_cost(RunConfig(**config, provider="modal"))
    assert spot.gpu_hourly_usd == 1
    assert spot.total_usd < modal.total_usd / 3
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


def test_ambiguous_create_is_not_blindly_retried(monkeypatch):
    from flash.providers.core.base import UnreconciledCreateError

    monkeypatch.setattr(
        api, "settings", lambda: {"project_id": "p", "subnet_id": "s", "image_id": "image"}
    )
    monkeypatch.setattr(api, "list_instances", lambda **kwargs: [])
    monkeypatch.setattr(api, "pricing_policy", lambda *args, **kwargs: "policy")
    from flash.providers.nebius import bootstrap

    monkeypatch.setattr(
        bootstrap,
        "publish_capsule",
        lambda *a, **kw: {"url": "https://example.com/capsule", "sha256": "digest"},
    )
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise api.NebiusApiError("unavailable")

    monkeypatch.setattr(api, "call", fail)
    with pytest.raises(UnreconciledCreateError):
        api.create(
            name="flash-test-a0",
            payload={
                "run_id": "flash-test",
                "attempt": 0,
                "job_spec_json": "{}",
                "source_snapshot": {},
            },
            image_reference="worker",
            gpu="H100",
            gpu_count=1,
            capsule=b"capsule",
            disk_gb=100,
            deadline_at=time.time() + 100,
        )
    assert len(calls) == 1


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


@pytest.mark.parametrize(("gpu", "hold"), [("H100", "1.50"), ("H200", "2.00"), ("B200", "5.00")])
def test_native_policy_uses_hold_threshold_not_entry_threshold(monkeypatch, gpu, hold):
    monkeypatch.setattr(api, "settings", lambda: {"project_id": "p"})
    calls = []

    def call(*args, **kwargs):
        calls.append(args)
        return {"items": []} if "list" in args else {"metadata": {"id": "policy"}}

    monkeypatch.setattr(api, "call", call)
    assert api.pricing_policy(gpu, deadline_at=time.time() + 60) == "policy"
    creation = calls[-1]
    assert creation[creation.index("--pricing-max-price-v1-max-price") + 1] == hold
    assert not accepts_price(gpu, hold, currency="usd", unit="GPU hour")


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


def test_empty_nebius_project_uses_omitted_repeated_field(monkeypatch):
    monkeypatch.setattr(api, "settings", lambda: {"project_id": "p"})
    monkeypatch.setattr(api, "call", lambda *a, **kw: {})
    assert api.list_instances() == []


def test_capsule_publication_pins_revision_and_checksum(monkeypatch):
    import hashlib

    from flash.providers.nebius import bootstrap

    uploads = []
    from contextlib import contextmanager

    @contextmanager
    def container(token=None):
        assert token in (None, "private-capability")
        # A worker SAS cannot read container properties.
        yield (
            SimpleNamespace(get_container_properties=lambda: SimpleNamespace(public_access=None))
            if token is None
            else SimpleNamespace()
        )

    monkeypatch.setattr(bootstrap, "open_container", container)
    monkeypatch.setattr(
        bootstrap,
        "BlobArtifacts",
        lambda c: SimpleNamespace(immutable=lambda name, data: uploads.append((name, data))),
    )
    ref = bootstrap.publish_capsule(
        {"hf_repo": "test/artifacts", "env": {"FLASH_ARTIFACT_BLOB_SAS_URL": "private-capability"}},
        b"capsule",
        deadline_at=time.time() + 60,
    )
    assert ref["sha256"] == hashlib.sha256(b"capsule").hexdigest()
    assert ref["blob"].endswith(ref["sha256"] + ".pyz")
    assert uploads == [(ref["blob"], b"capsule")]
    init = api.cloud_init({"env": {}}, ref, "worker", deadline_at=time.time() + 60)
    assert len(init.encode()) < 32768
    assert "private-capability" not in str(ref)


def test_delete_timeout_reconciles_until_vm_disappears(monkeypatch):
    current = [100.0]
    monkeypatch.setattr(api.time, "time", lambda: current[0])
    monkeypatch.setattr(
        api.time, "sleep", lambda seconds: current.__setitem__(0, current[0] + seconds)
    )
    instances = iter([{"status": "running"}, {"status": "deleting"}, None])
    monkeypatch.setattr(api, "get_instance", lambda *a, **kw: next(instances))

    def timeout(*a, **kw):
        raise api.NebiusApiError("CLI timeout")

    monkeypatch.setattr(api, "call", timeout)
    api.terminate("owned-vm")
    assert current[0] == 102


@pytest.mark.parametrize("valid_digest", [True, False])
def test_bootstrap_download_uses_private_blob_and_checks_content(
    monkeypatch, tmp_path, valid_digest
):
    import hashlib
    import json
    import urllib.request

    from flash.providers.nebius.bootstrap import DOWNLOAD

    capsule = b"immutable executable capsule"
    (tmp_path / "payload.env").write_text(
        "FLASH_BOOTSTRAP_PAYLOAD="
        + json.dumps(
            {
                "env": {
                    "FLASH_ARTIFACT_BLOB_SAS_URL": "https://example.blob.core.windows.net/private?sig=secret"
                }
            }
        )
    )
    (tmp_path / "capsule.json").write_text(
        json.dumps(
            {
                "blob": "repo/capsules/content.pyz",
                "sha256": hashlib.sha256(capsule).hexdigest() if valid_digest else "wrong",
            }
        )
    )
    requests = []

    def download(request, timeout):
        requests.append(request.full_url)
        assert timeout == 60
        return io.BytesIO(capsule)

    monkeypatch.setattr(urllib.request, "urlopen", download)
    code = DOWNLOAD.replace("'/opt/flash'", repr(str(tmp_path)))
    if valid_digest:
        exec(code, {})
        assert (tmp_path / "capsule.pyz").read_bytes() == capsule
        assert (tmp_path / "capsule.pyz").stat().st_mode & 0o777 == 0o600
    else:
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            exec(code, {})
        assert not (tmp_path / "capsule.pyz").exists()
    assert requests == [
        "https://example.blob.core.windows.net/private/repo/capsules/content.pyz?sig=secret"
    ]


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

    from flash.providers.nebius import bootstrap

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
    from flash.providers.nebius import jobs
    from flash.providers.nebius.handles import SpotJobHandle
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
        instance_id="computeinstance-test",
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
    prices_to_check = iter([{}, {}, {"H100": 0.9}])
    monkeypatch.setattr(queue, "settings", lambda: {"region": "test"})
    monkeypatch.setattr(queue, "affordable_gpus", lambda region: next(prices_to_check))
    monkeypatch.setattr(status, "get_status", lambda run: SimpleNamespace(state="queued"))
    monkeypatch.setattr(status, "_update", lambda run, state: states.append(state))
    queued_at = now[0]
    queue.wait_for_price("H100", spec.run_id, deadline_at=500, log=io.StringIO())
    assert now[0] == queued_at + 120
    assert states == ["queued", "queued", "provisioning"]
