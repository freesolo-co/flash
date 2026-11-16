"""Lambda Cloud training: pinning, catalog, ownership, launch and teardown without renting GPUs."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from flash.providers.core.base import (
    AllocationConstraints,
    CapacityLookupError,
    CapacityUnavailableError,
    UnreconciledCreateError,
    UnsupportedGpuError,
)
from flash.providers.core.registry import get_provider
from flash.providers.lambda_ import api, bootstrap, client, config, jobs, pricing
from flash.providers.lambda_.handles import LambdaJobHandle

API_KEY = "lambda-test-secret-key"
SAS = "https://example.blob.core.windows.net/private?sig=sas-secret"


@pytest.fixture(autouse=True)
def lambda_env(monkeypatch):
    monkeypatch.setenv("FLASH_LAMBDA_API_KEY", API_KEY)
    monkeypatch.setenv("FLASH_LAMBDA_SSH_KEY_NAME", "flash-training")
    monkeypatch.setattr(api, "LAUNCH_RECHECK_SECONDS", 0)


def entry(name, cents, description, *, gpus, storage=2816, regions=("us-east-1",), arch=None):
    instance = {
        "name": name,
        "price_cents_per_hour": cents,
        "gpu_description": description,
        "specs": {"gpus": gpus, "storage_gib": storage, "vcpus": 26, "memory_gib": 225},
    }
    if arch is not None:
        instance["architecture"] = arch
    return {
        "instance_type": instance,
        "regions_with_capacity_available": [{"name": region} for region in regions],
    }


def catalog(**capacity):
    """The live catalog shape (2026-11-15), with per-type region overrides."""
    rows = {
        "gpu_1x_h100_sxm5": entry("gpu_1x_h100_sxm5", 489, "H100 (80 GB SXM5)", gpus=1),
        "gpu_1x_h100_pcie": entry(
            "gpu_1x_h100_pcie", 379, "H100 (80 GB PCIe)", gpus=1, storage=1024
        ),
        "gpu_8x_h100_sxm5": entry("gpu_8x_h100_sxm5", 2392, "H100 (80 GB SXM5)", gpus=8),
        "gpu_1x_b200_sxm6": entry("gpu_1x_b200_sxm6", 799, "B200 (180 GB SXM6)", gpus=1),
        "gpu_2x_b200_sxm6": entry("gpu_2x_b200_sxm6", 1598, "B200 (180 GB SXM6)", gpus=2),
        "gpu_8x_b200_sxm6": entry("gpu_8x_b200_sxm6", 6392, "B200 (180 GB SXM6)", gpus=8),
        "gpu_1x_gh200": entry("gpu_1x_gh200", 259, "GH200 (96 GB)", gpus=1, arch="arm64"),
        "gpu_1x_a100_sxm4": entry("gpu_1x_a100_sxm4", 199, "A100 (40 GB SXM4)", gpus=1),
        "gpu_8x_a100_80gb_sxm4": entry("gpu_8x_a100_80gb_sxm4", 1432, "A100 (80 GB SXM4)", gpus=8),
        "gpu_1x_a10": entry("gpu_1x_a10", 129, "A10 (24 GB PCIe)", gpus=1, storage=512),
    }
    for name, regions in capacity.items():
        rows[name]["regions_with_capacity_available"] = [{"name": r} for r in regions]
    return rows


def serve(monkeypatch, rows, instances=()):
    """Route api.call to a fake Lambda account; return the list of mutating requests."""
    posts = []
    listed = list(instances)

    def call(method, path, *, body=None, deadline_at=None, retries=None):
        if (method, path) == ("GET", "/instance-types"):
            return rows
        if (method, path) == ("GET", "/instances"):
            return listed
        posts.append((method, path, body))
        raise AssertionError(f"unexpected Lambda call {method} {path}")

    monkeypatch.setattr(api, "call", call)
    return posts


# --- schema / spec -------------------------------------------------------------------------------


def _raw(gpu):
    return {
        "model": "Qwen/Qwen3.5-9B-Base",
        "algorithm": "sft",
        "environment": {"id": "test/project/env"},
        "gpu": gpu,
    }


@pytest.mark.parametrize("enabled", [True, False])
def test_toml_lambda_pin_parses_and_round_trips(enabled):
    import tomllib

    from flash.core.spec import JobSpec
    from flash.schema import spec_from_dict

    raw = _raw(tomllib.loads(f"lambda = {str(enabled).lower()}\nmax_count = 2"))
    spec = spec_from_dict(raw, project_required=False)
    assert spec.gpu.provider == ("lambda" if enabled else "")
    assert spec.gpu.spot is False
    assert spec.gpu.max_count == 2
    internal = JobSpec.from_json(spec.to_json())
    assert internal == spec
    public = spec.to_dict()
    assert ("lambda" in public["gpu"]) is enabled
    assert JobSpec.from_dict(public).gpu.provider == spec.gpu.provider
    # a public spec copied back into a config still parses to the same pin.
    assert spec_from_dict(public, project_required=False).gpu.provider == spec.gpu.provider


def test_lambda_pin_leaves_unpinned_digests_alone():
    from flash.core.spec import GpuSpec, JobSpec

    assert "lambda" not in JobSpec(gpu=GpuSpec()).to_internal_dict()["gpu"]
    assert "lambda" not in JobSpec(gpu=GpuSpec(provider="spot")).to_internal_dict()["gpu"]
    pinned = JobSpec(gpu=GpuSpec(provider="lambda")).to_internal_dict()["gpu"]
    assert pinned["lambda"] is True
    assert pinned["provider"] == "lambda"


@pytest.mark.parametrize(
    "gpu",
    [
        {"spot": True, "lambda": True},
        {"lambda": "true"},
        {"lambda": 1},
        {"lambda": False, "provider": "lambda"},
        {"lambda": True, "provider": "modal"},
    ],
)
def test_lambda_toml_rejects_conflicts_and_non_booleans(gpu):
    from flash.schema import ConfigError, spec_from_dict

    with pytest.raises(ConfigError):
        spec_from_dict(_raw(gpu), project_required=False)


def test_persisted_lambda_flag_must_match_its_provider():
    from flash.core.spec import GpuSpec, JobSpec

    data = JobSpec(gpu=GpuSpec(provider="modal")).to_internal_dict()
    data["gpu"]["lambda"] = True
    with pytest.raises(ValueError, match=r"gpu\.lambda"):
        JobSpec.from_dict(data)
    data["gpu"]["lambda"] = "yes"
    with pytest.raises(TypeError):
        JobSpec.from_dict(data)


def test_lambda_custom_training_is_refused_at_parse():
    from flash.schema import ConfigError, spec_from_dict

    raw = _raw({"lambda": True})
    raw["algorithm"] = "custom"
    with pytest.raises(ConfigError):
        spec_from_dict(raw, project_required=False)


# --- registry / config ---------------------------------------------------------------------------


def test_registry_lists_lambda_as_an_opt_in_instance_provider():
    from flash.providers.core.handles import parse_handle
    from flash.providers.core.registry import (
        INSTANCE_PROVIDERS,
        PROVIDER_NAMES,
        validated_provider_preferences,
    )

    assert "lambda" in PROVIDER_NAMES
    assert "lambda" in INSTANCE_PROVIDERS
    provider = get_provider("lambda")
    assert provider.name == "lambda"
    assert provider.live_capacity
    assert not provider.supports_weight_cache
    assert provider.is_configured()
    assert validated_provider_preferences(["lambda"]) == ("lambda",)
    for mixed in (["lambda", "modal"], ["modal", "lambda"], ["spot", "lambda"]):
        with pytest.raises(ValueError, match="cannot fall back"):
            validated_provider_preferences(mixed)
    assert isinstance(parse_handle(handle().to_dict()), LambdaJobHandle)


def test_configuration_needs_key_and_ssh_key_and_scope_hides_the_key(monkeypatch):
    before = config.scope_fingerprint()
    assert API_KEY not in before
    assert before != hashlib.sha256(API_KEY.encode()).hexdigest()  # namespaced
    monkeypatch.setenv("FLASH_LAMBDA_API_KEY", "another-key")
    assert config.scope_fingerprint() != before
    monkeypatch.delenv("FLASH_LAMBDA_SSH_KEY_NAME")
    assert not config.configured()
    assert get_provider("lambda").preflight(require_hf=False)
    with pytest.raises(RuntimeError, match="Configure"):
        config.scope_fingerprint()


# --- catalog / pricing ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "instance", "expected"),
    [
        ("gpu_1x_h100_pcie", {"gpu_description": "H100 (80 GB PCIe)"}, ("H100", 1)),
        ("gpu_8x_b200_sxm6", {"specs": {"gpus": 8}}, ("B200", 8)),
        ("gpu_1x_gh200", {"architecture": "arm64"}, None),
        ("gpu_1x_gh200", {}, None),  # ARM64 even when the catalog omits the architecture
        ("gpu_1x_h100_sxm5", {"architecture": "arm64"}, None),
        ("gpu_8x_a100_80gb_sxm4", {}, None),
        ("gpu_1x_a100", {"gpu_description": "A100 (40 GB PCIe)"}, ("A100 SXM 40GB", 1)),
        ("gpu_1x_h200_sxm5", {}, None),  # no H200 on Lambda; never mapped if it ever appears
        ("gpu_2x_h100_sxm5", {"specs": {"gpus": 1}}, None),
        ("gpu_1x_h100_sxm5", {"gpu_description": "H100 (40 GB)"}, None),
        ("gpu_3x_h100_sxm5", {}, None),
    ],
)
def test_instance_types_map_only_clean_x86_classes(name, instance, expected):
    assert pricing.classify(name, instance) == expected


def test_offers_keep_live_x86_types_with_whole_instance_prices(monkeypatch):
    serve(monkeypatch, catalog(gpu_1x_b200_sxm6=()))
    live = {offer.instance_type: offer for offer in api.offers()}
    assert "gpu_1x_gh200" not in live
    assert "gpu_8x_a100_80gb_sxm4" not in live
    assert "gpu_1x_b200_sxm6" not in live  # no region has capacity
    assert live["gpu_1x_h100_pcie"] == api.Offer(
        "H100", 1, "gpu_1x_h100_pcie", 3.79, 1024.0, ("us-east-1",)
    )
    assert live["gpu_8x_h100_sxm5"].count == 8
    every = {offer.instance_type for offer in api.offers(require_capacity=False)}
    assert "gpu_1x_b200_sxm6" in every


def test_catalog_is_cached_briefly_and_failures_are_lookup_errors(monkeypatch):
    calls = []

    def call(method, path, **kwargs):
        calls.append(path)
        return catalog()

    monkeypatch.setattr(api, "call", call)
    api.offers()
    api.offers()
    assert calls == ["/instance-types"]
    api.offers(force=True)
    assert len(calls) == 2

    def broken(*args, **kwargs):
        raise client.LambdaApiError("down", status=503)

    monkeypatch.setattr(api, "call", broken)
    with pytest.raises(CapacityLookupError):
        api.offers(force=True)


def test_hourly_rate_is_live_per_card_with_static_fallback(monkeypatch):
    serve(monkeypatch, catalog())
    assert pricing.hourly_rate("H100") == pytest.approx(3.79)  # cheapest 1x type (PCIe)
    assert pricing.hourly_rate("H100", gpu_count=8) == pytest.approx(23.92 / 8)
    assert pricing.hourly_rate("B200") == pytest.approx(7.99)
    monkeypatch.setattr(api, "_catalog", {"at": 0.0, "data": None})

    def broken(*args, **kwargs):
        raise client.LambdaApiError("down", status=503)

    monkeypatch.setattr(api, "call", broken)
    assert pricing.hourly_rate("B200") == 7.99
    assert pricing.STATIC_INSTANCE_PRICES["gpu_1x_h100_sxm5"] == 4.89
    with pytest.raises(KeyError):
        pricing.hourly_rate("H200")


def test_live_candidates_take_the_cheapest_type_per_shape_and_respect_limits(monkeypatch):
    serve(monkeypatch, catalog())
    provider = get_provider("lambda")
    found = provider.live_candidates(60, AllocationConstraints(max_gpu_count=1))
    assert {(c.gpu, c.gpu_count): c.hourly_usd for c in found} == {
        ("H100", 1): pytest.approx(3.79),
        ("B200", 1): pytest.approx(7.99),
    }
    # a disk floor above the PCIe type's fixed 1024 GiB drops it for the SXM5 type
    found = provider.live_candidates(
        60, AllocationConstraints(max_gpu_count=1, disk_gb=2000, gpu_type="H100")
    )
    assert [(c.gpu, c.hourly_usd) for c in found] == [("H100", pytest.approx(4.89))]
    wide = provider.live_candidates(10, AllocationConstraints(max_gpu_count=8))
    assert ("B200", 2) in {(c.gpu, c.gpu_count) for c in wide}
    assert all(c.provider == "lambda" for c in wide)


# --- allocation ----------------------------------------------------------------------------------


def _allocate(monkeypatch, need, rows=None, **kwargs):
    from flash.providers.core import allocator

    serve(monkeypatch, catalog() if rows is None else rows)
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: need)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("modal", "lambda"))
    return allocator.allocate("Qwen/Qwen3.5-9B-Base", "sft", **kwargs)


def test_unpinned_runs_never_allocate_on_lambda(monkeypatch):
    allocation = _allocate(monkeypatch, 60)
    assert allocation.provider == "modal"
    assert all(c.provider != "lambda" for c in allocation.candidates)
    allocation = _allocate(monkeypatch, 60, provider="modal")
    assert all(c.provider != "lambda" for c in allocation.candidates)


@pytest.mark.parametrize(("need", "gpu"), [(60, "H100"), (120, "B200")])
def test_pinned_lambda_rents_the_fitting_single_card(monkeypatch, need, gpu):
    allocation = _allocate(monkeypatch, need, provider="lambda")
    assert (allocation.provider, allocation.gpu, allocation.gpu_count) == ("lambda", gpu, 1)
    assert {c.provider for c in allocation.candidates} == {"lambda"}
    # the providers-list spelling pins the same way
    assert _allocate(monkeypatch, need, providers=("lambda",)).provider == "lambda"


def test_pinned_lambda_with_no_capacity_is_retryable(monkeypatch):
    rows = {name: {**row, "regions_with_capacity_available": []} for name, row in catalog().items()}
    with pytest.raises(CapacityUnavailableError):
        _allocate(monkeypatch, 60, rows=rows, provider="lambda")


def test_pinned_lambda_structural_misfits_are_terminal(monkeypatch):
    # bigger than one B200 and limited to one card
    with pytest.raises(UnsupportedGpuError) as error:
        _allocate(monkeypatch, 200, provider="lambda", max_gpu_count=1)
    assert not isinstance(error.value, CapacityUnavailableError)
    with pytest.raises(UnsupportedGpuError, match=r"Lambda is not supported.*no H200"):
        _allocate(monkeypatch, 5000, provider="lambda")


def test_unconfigured_lambda_pin_is_refused(monkeypatch):
    from flash.providers.core import allocator

    monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 60)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("modal",))
    with pytest.raises(UnsupportedGpuError, match="not configured"):
        allocator.allocate("Qwen/Qwen3.5-9B-Base", "sft", provider="lambda")


def test_offline_lambda_quote_uses_static_prices_and_lambda_classes():
    from flash.cost import RunConfig, estimate_cost

    # conftest makes any live Lambda call fail, so this proves the quote is offline.
    quote = estimate_cost(RunConfig("Qwen/Qwen3.5-9B-Base", "sft", 10, provider="lambda"))
    assert quote.provider == "lambda"
    assert quote.gpu in pricing.LAMBDA_GPUS
    assert quote.gpu_hourly_usd == pricing.static_hourly_rate(quote.gpu)
    pinned = estimate_cost(RunConfig("Qwen/Qwen3.5-9B-Base", "sft", 10, providers=("lambda",)))
    assert pinned.provider == "lambda"


def test_lambda_capacity_wait_does_not_consume_attempts(monkeypatch):
    from flash.providers.core.base import PollResult
    from flash.runner.lifecycle import deadlines, status
    from flash.runner.supervise import execution as supervision

    now = [100.0]
    states = []
    log = io.StringIO()
    ctx = SimpleNamespace(
        spec=SimpleNamespace(run_id="run"), log=log, raise_if_cancelled=lambda: None
    )
    prepared = (None, SimpleNamespace(gpu=SimpleNamespace(provider="lambda", providers=())))
    answers = iter([(None, PollResult(False, failure="no_capacity")), ("allocation", None)])
    monkeypatch.setattr(supervision, "_allocate_attempt_once", lambda *args: next(answers))
    monkeypatch.setattr(supervision.time, "time", lambda: now[0])
    monkeypatch.setattr(
        supervision.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(deadlines, "_load_run_deadline_at", lambda run: 500)
    monkeypatch.setattr(status, "_update", lambda run, state: states.append(state))
    assert supervision._allocate_attempt(ctx, prepared) == ("allocation", None)
    assert states == ["queued"]
    assert "Lambda queued" in log.getvalue()


def test_lambda_create_refusal_requeues_without_spending_budget():
    from flash.providers.core.base import Candidate
    from flash.runner.supervise.retry_decision import (
        FailureObservation,
        RetryState,
        transition_failure,
    )

    chosen = Candidate("lambda", "H100", 3.79, 80)
    state, plan = transition_failure(
        RetryState(2, 2, 1), FailureObservation("no_capacity", chosen, (chosen,), False), attempt=0
    )
    assert plan.retry
    assert "lambda" in plan.action
    assert state.infra_used == 0


# --- handles -------------------------------------------------------------------------------------


def handle(**changes):
    values = {
        "instance_id": "0920582c7ff041399e34823a0be62549",
        "gpu": "H100",
        "hourly_usd": 3.79,
        "attempt": 0,
        "started_ts": 100.0,
        "instance_type": "gpu_1x_h100_pcie",
        "region": "us-east-1",
        "name": "flash-test-a0",
        "run_id": "flash-test",
        "hf_repo": "org/artifacts",
        "phase": "sft",
        "scope": config.scope_fingerprint(),
        "gpu_count": 1,
    }
    values.update(changes)
    return LambdaJobHandle(**values)


def test_handle_round_trips_through_cleanup_identity():
    from flash.runner.lifecycle.cleanup import _remote_resource_identity

    data = handle().to_dict()
    assert data["provider"] == "lambda"
    assert LambdaJobHandle.from_dict(data) == handle()
    assert _remote_resource_identity(data) == (
        "lambda",
        0,
        "0920582c7ff041399e34823a0be62549",
        "flash-test-a0",
        "flash-test",
    )


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "spot"},
        {"instance_id": "bad id"},
        {"instance_id": 7},
        {"gpu_count": 3},
        {"gpu_count": True},
        {"gpu_count": 2},  # disagrees with gpu_1x_...
        {"instance_type": "it_h100"},
        {"region": "US East"},
        {"name": " flash-test-a0"},
        {"scope": ""},
        {"hf_repo": None},
    ],
)
def test_handle_validation_is_strict(change):
    data = {**handle().to_dict(), **change}
    with pytest.raises(ValueError, match=r"(?i)lambda"):
        LambdaJobHandle.from_dict(data)


def test_handles_from_another_account_are_refused(monkeypatch):
    from flash.providers.core.base import JobHandle

    persisted = JobHandle.from_dict(handle().to_dict())
    monkeypatch.setenv("FLASH_LAMBDA_API_KEY", "another-account")
    with pytest.raises(RuntimeError, match="owner"):
        get_provider("lambda").destroy(persisted)


# --- REST client ---------------------------------------------------------------------------------


@pytest.fixture
def wire(monkeypatch):
    """Restore the real transport (conftest blocks it) with httpx itself stubbed."""
    from flash.providers.lambda_ import client as module

    monkeypatch.setattr(module, "_once", _REAL_ONCE)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    requests = []

    def install(responses):
        answers = iter(responses)

        def request(self, method, url, **kwargs):
            requests.append((method, url, kwargs))
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        monkeypatch.setattr(httpx.Client, "request", request)
        return requests

    return install


_REAL_ONCE = client._once


def test_client_sends_a_real_user_agent_and_unwraps_data(wire):
    requests = wire([httpx.Response(200, json={"data": {"ok": 1}})])
    assert client.call("GET", "/instance-types") == {"ok": 1}
    _method, url, kwargs = requests[0]
    assert url == "https://cloud.lambda.ai/api/v1/instance-types"
    assert kwargs["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert "python" not in kwargs["headers"]["User-Agent"].lower()


def test_client_errors_keep_only_the_code(wire):
    body = {"error": {"code": "global/invalid-parameters", "message": API_KEY, "suggestion": SAS}}
    wire([httpx.Response(400, json=body)])
    with pytest.raises(client.LambdaApiError) as error:
        client.call("POST", "/instance-operations/terminate", body={"user_data": SAS})
    assert (error.value.status, error.value.code) == (400, "global/invalid-parameters")
    assert API_KEY not in str(error.value)
    assert SAS not in str(error.value)


def test_reads_retry_transient_failures(wire):
    requests = wire(
        [
            httpx.ConnectError("boom"),
            httpx.Response(503, json={}),
            httpx.Response(200, json={"data": []}),
        ]
    )
    assert client.call("GET", "/instances") == []
    assert len(requests) == 3


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("t"), httpx.Response(503, json={})])
def test_launch_is_never_retried(wire, failure):
    requests = wire([failure, httpx.Response(200, json={"data": {"instance_ids": ["x"]}})])
    with pytest.raises(client.LambdaApiError):
        client.call("POST", client.LAUNCH_PATH, body={}, retries=5)
    assert len(requests) == 1


# --- launch / reconcile / terminate --------------------------------------------------------------

OFFER = api.Offer("H100", 1, "gpu_1x_h100_pcie", 3.79, 1024.0, ("us-east-1",))
INSTANCE_ID = "0920582c7ff041399e34823a0be62549"


def account(monkeypatch, *, launch, listings):
    """A fake account: ``launch`` answers the POST, ``listings`` answer successive GET /instances."""
    posts = []
    pending = iter(listings)

    def call(method, path, *, body=None, deadline_at=None, retries=None):
        if (method, path) == ("GET", "/instances"):
            answer = next(pending)
            if isinstance(answer, Exception):
                raise answer
            return answer
        assert (method, path) == ("POST", client.LAUNCH_PATH)
        assert retries == 0
        posts.append(body)
        if isinstance(launch, Exception):
            raise launch
        return launch

    monkeypatch.setattr(api, "call", call)
    return posts


def _launch(**changes):
    kwargs = {
        "name": "flash-test-a0",
        "run_id": "flash-test",
        "attempt": 0,
        "offer": OFFER,
        "region": "us-east-1",
        "user_data": "#cloud-config\n{}",
        "deadline_at": time.time() + 600,
    }
    kwargs.update(changes)
    return api.launch(**kwargs)


def row(name="flash-test-a0", status="booting", instance_id=INSTANCE_ID):
    return {
        "id": instance_id,
        "name": name,
        "status": status,
        "instance_type": {"name": "gpu_1x_h100_pcie"},
        "region": {"name": "us-east-1"},
    }


def test_launch_posts_one_named_vm(monkeypatch):
    posts = account(monkeypatch, launch={"instance_ids": [INSTANCE_ID]}, listings=[[]])
    assert _launch() == INSTANCE_ID
    assert len(posts) == 1
    body = posts[0]
    assert body["name"] == "flash-test-a0"
    assert body["instance_type_name"] == "gpu_1x_h100_pcie"
    assert body["region_name"] == "us-east-1"
    assert body["quantity"] == 1
    assert body["ssh_key_names"] == ["flash-training"]


@pytest.mark.parametrize(
    "error",
    [
        client.LambdaApiError("bad gateway", status=502),
        client.LambdaApiError("timeout"),
        TimeoutError("deadline"),
    ],
)
def test_ambiguous_launch_is_reconciled_by_name_without_a_second_post(monkeypatch, error):
    posts = account(monkeypatch, launch=error, listings=[[], [row()]])
    assert _launch() == INSTANCE_ID
    assert len(posts) == 1


def test_ambiguous_launch_with_no_vm_requeues_after_one_recheck(monkeypatch):
    posts = account(
        monkeypatch, launch=client.LambdaApiError("oops", status=500), listings=[[], [], []]
    )
    with pytest.raises(CapacityUnavailableError, match="outcome unknown") as error:
        _launch()
    assert error.value.cause == "http 500"
    assert len(posts) == 1


def test_unlistable_ambiguous_launch_is_unreconciled(monkeypatch):
    account(
        monkeypatch,
        launch=client.LambdaApiError("oops", status=500),
        listings=[[], client.LambdaApiError("down", status=503)],
    )
    with pytest.raises(UnreconciledCreateError):
        _launch()


def test_capacity_refusal_requeues_and_other_refusals_do_not(monkeypatch):
    code = "instance-operations/launch/insufficient-capacity"
    account(
        monkeypatch, launch=client.LambdaApiError("x", status=400, code=code), listings=[[], []]
    )
    with pytest.raises(CapacityUnavailableError):
        _launch()
    account(
        monkeypatch,
        launch=client.LambdaApiError("x", status=400, code="global/invalid-parameters"),
        listings=[[], []],
    )
    with pytest.raises(client.LambdaApiError) as error:
        _launch()
    assert not isinstance(error.value, CapacityLookupError)


def test_existing_vm_under_the_attempt_name_is_adopted_not_relaunched(monkeypatch):
    posts = account(monkeypatch, launch={"instance_ids": ["other"]}, listings=[[row()]])
    assert _launch() == INSTANCE_ID
    assert posts == []
    account(
        monkeypatch,
        launch={"instance_ids": ["other"]},
        listings=[[row(), row(instance_id="1" * 32)]],
    )
    with pytest.raises(UnreconciledCreateError):
        _launch()


def test_later_attempt_first_terminates_earlier_attempts(monkeypatch):
    terminated = []
    posts = account(
        monkeypatch,
        launch={"instance_ids": [INSTANCE_ID]},
        listings=[[row(name="flash-test-a0", instance_id="0" * 32)]] * 2,
    )
    monkeypatch.setattr(api, "terminate_confirmed", lambda iid, name: terminated.append(name))
    assert _launch(name="flash-test-a1", attempt=1) == INSTANCE_ID
    assert terminated == ["flash-test-a0"]
    assert len(posts) == 1


def test_terminate_is_per_id_and_confirmed(monkeypatch):
    states = iter([row(status="active"), row(status="terminating"), None])
    calls = []

    def call(method, path, *, body=None, deadline_at=None, retries=None):
        calls.append((method, path, body))
        if method == "GET":
            state = next(states)
            if state is None:
                raise client.LambdaApiError("gone", status=404)
            return state
        return {"terminated_instances": [{"id": INSTANCE_ID}]}

    monkeypatch.setattr(api, "call", call)
    monkeypatch.setattr(api.time, "sleep", lambda seconds: None)
    api.terminate_confirmed(INSTANCE_ID, name="flash-test-a0")
    posts = [c for c in calls if c[0] == "POST"]
    assert posts == [
        ("POST", "/instance-operations/terminate", {"instance_ids": [INSTANCE_ID]}),
    ]


def test_unconfirmed_terminate_raises(monkeypatch):
    clock = [1000.0]

    def call(method, path, *, body=None, deadline_at=None, retries=None):
        return row(status="active") if method == "GET" else {}

    monkeypatch.setattr(api, "call", call)
    monkeypatch.setattr(api.time, "time", lambda: clock[0])
    monkeypatch.setattr(api.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + max(s, 1)))
    with pytest.raises(client.LambdaApiError, match="not confirmed"):
        api.terminate_confirmed(INSTANCE_ID, name="flash-test-a0")


def test_foreign_instance_is_never_terminated(monkeypatch):
    posts = []

    def call(method, path, *, body=None, deadline_at=None, retries=None):
        if method == "POST":
            posts.append(body)
        return row(name="someone-else")

    monkeypatch.setattr(api, "call", call)
    with pytest.raises(client.LambdaApiError, match="does not belong"):
        api.terminate_confirmed(INSTANCE_ID, name="flash-test-a0")
    assert posts == []


@pytest.mark.parametrize("callbacks", [False, True])
def test_orphan_sweep_reaps_only_known_inactive_flash_vms(monkeypatch, callbacks):
    long_run = "flash-" + "x" * 80
    from flash.providers._lifecycle.instances.instance import instance_label

    rows = [
        row(name="flash-done-a0", instance_id="a" * 32),
        row(name="flash-done-a1", instance_id="b" * 32, status="terminated"),
        row(name="flash-live-a3", instance_id="c" * 32),
        row(name="flash-elsewhere-a0", instance_id="d" * 32),
        row(name="flash-main-" + "e" * 32, instance_id="e" * 32),  # a serving VM
        row(name=instance_label(long_run, 2), instance_id="f" * 32),
    ]
    serve(monkeypatch, catalog(), rows)
    terminated = []
    monkeypatch.setattr(api, "terminate_confirmed", lambda iid, name: terminated.append(iid))
    active = {"flash-live"}
    known = {"flash-done", "flash-live", long_run, "main"}
    result = get_provider("lambda").sweep_orphans(
        (lambda: active) if callbacks else active, (lambda: known) if callbacks else known
    )
    assert result == terminated == ["a" * 32, "f" * 32]


def test_run_teardown_and_remaining_use_the_run_label(monkeypatch):
    rows = [row(name="flash-test-a0"), row(name="flash-test2-a0", instance_id="9" * 32)]
    serve(monkeypatch, catalog(), rows)
    provider = get_provider("lambda")
    assert provider.run_instances_remaining("flash-test") == [INSTANCE_ID]
    terminated = []
    monkeypatch.setattr(api, "terminate_confirmed", lambda iid, name: terminated.append(name))
    provider.terminate_run("flash-test")
    assert terminated == ["flash-test-a0"]


# --- bootstrap -----------------------------------------------------------------------------------


def payload():
    return {
        "run_id": "flash-test",
        "attempt": 0,
        "phase": "sft",
        "hf_repo": "owner/repo",
        "job_spec_json": "{}",
        "source_snapshot": {},
        "env": {"FLASH_ARTIFACT_BLOB_SAS_URL": SAS, "HF_TOKEN": "hf_private_token"},
    }


def test_user_data_is_cloud_config_with_secrets_only_in_private_storage(monkeypatch):
    published = []

    def publish(payload, data, suffix, **kw):
        published.append((data, suffix))
        return {"blob": "private/" + suffix, "sha256": hashlib.sha256(data).hexdigest()}

    monkeypatch.setattr(bootstrap, "_publish", publish)
    raw = payload()
    raw["env"]["LARGE"] = "x" * 100_000
    text = bootstrap.user_data(raw, b"capsule", "worker@sha256:d", deadline_at=time.time() + 100)
    assert text.startswith("#cloud-config\n")
    assert API_KEY not in text
    assert "hf_private_token" not in text
    assert raw["env"]["LARGE"] not in text
    assert len(text) < 16_000
    config_doc = json.loads(text.split("\n", 1)[1])
    # nothing secret on a command line: runcmd only names the 0600 downloader file
    assert config_doc["runcmd"] == [
        ["systemd-run", "--unit=flash-training", "--collect", "python3", bootstrap.BOOTSTRAP_PATH]
    ]
    assert SAS not in json.dumps(config_doc["runcmd"])
    (written,) = config_doc["write_files"]
    assert written["path"] == bootstrap.BOOTSTRAP_PATH
    assert written["permissions"] == "0600"
    downloader = base64.b64decode(written["content"]).decode()
    compile(downloader, "bootstrap.py", "exec")
    assert [suffix for _, suffix in published] == ["pyz", "py"]
    launch_script = published[1][0].decode()
    compile(launch_script, "launch.py", "exec")
    assert "lambda_claim_0" in launch_script
    assert "If-None-Match" in launch_script
    assert API_KEY not in launch_script


def test_worker_keeps_dockers_default_storage_and_never_prints_secrets(
    monkeypatch, tmp_path, capsys
):
    import subprocess
    import urllib.request

    root = tmp_path / "flash"
    root.mkdir()
    capsule = b"test capsule"
    bundle = {
        "payload": payload(),
        "deadline_at": time.time() + 3600,
        "claim_blob": "repo/run/claim",
        "capsule": {"blob": "repo/capsule", "sha256": hashlib.sha256(capsule).hexdigest()},
        "image": "worker@sha256:digest",
    }

    def download(request, timeout):
        return io.BytesIO(b"" if isinstance(request, urllib.request.Request) else capsule)

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(urllib.request, "urlopen", download)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("os.umask", lambda _: None)
    assert "/mnt/local" not in bootstrap.WORKER
    assert "data-root" not in bootstrap.WORKER
    exec(bootstrap.WORKER.replace("'/opt/flash'", repr(str(root))), {"BUNDLE": bundle})
    docker = next(c for c in commands if c[0] == "timeout")
    assert docker[4:8] == ["docker", "run", "--rm", "--gpus"]
    assert "--ipc=host" in docker
    assert "worker@sha256:digest" in docker
    assert not any(SAS in part or "hf_private_token" in part for part in docker)
    assert commands[-1] == ["shutdown", "-h", "now"]
    assert json.loads((root / "payload.env").read_text().split("=", 1)[1]) == bundle["payload"]
    assert "sas-secret" not in capsys.readouterr().out


def test_worker_never_starts_after_its_deadline(monkeypatch, capsys):
    import subprocess

    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: commands.append(command))
    monkeypatch.setattr("os.umask", lambda value: None)
    bundle = {"payload": payload(), "deadline_at": 0, "claim_blob": "repo/run/claim"}
    exec(bootstrap.WORKER, {"BUNDLE": bundle})
    assert commands == [["shutdown", "-h", "now"]]
    assert "sas-secret" not in capsys.readouterr().out


# --- submit / supervise --------------------------------------------------------------------------


def _submit(monkeypatch, poll, on_handle, *, rows=None):
    import flash.runtime_capsule
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    spec = JobSpec(
        run_id="flash-test",
        model="Qwen/Qwen3.5-9B-Base",
        train=TrainSpec(hf_repo="org/artifacts"),
        gpu=GpuSpec(type="H100", provider="lambda", disk_gb=100),
    )
    serve(monkeypatch, catalog() if rows is None else rows)
    launched = []
    monkeypatch.setattr(jobs, "build_payload", lambda *a, **k: {"env": {}})
    monkeypatch.setattr(flash.runtime_capsule, "build_capsule", lambda *a: (b"capsule", None))
    monkeypatch.setattr(jobs, "worker_image_for_gpu", lambda gpu: "image")
    monkeypatch.setattr(bootstrap, "user_data", lambda *a, **k: "#cloud-config\n{}")

    def launch(**kwargs):
        launched.append(kwargs)
        return INSTANCE_ID

    monkeypatch.setattr(api, "launch", launch)
    terminated = []
    monkeypatch.setattr(api, "terminate_confirmed", lambda iid, name: terminated.append(iid))
    monkeypatch.setattr(jobs, "poll", poll)
    try:
        result = jobs.submit_attempt(
            spec, on_handle=on_handle, deadline_at=time.time() + 600, log=io.StringIO()
        )
    except BaseException as exc:  # TaskStopped is a BaseException
        return type(exc), launched, terminated
    return result, launched, terminated


def test_submit_launches_the_cheapest_live_type_and_always_terminates(monkeypatch):
    handles = []
    result, launched, terminated = _submit(monkeypatch, lambda *a, **k: "done", handles.append)
    assert result == "done"
    assert launched[0]["offer"].instance_type == "gpu_1x_h100_pcie"
    assert launched[0]["region"] == "us-east-1"
    assert launched[0]["name"] == "flash-test-a0"
    persisted = LambdaJobHandle.from_dict(handles[0].to_dict())
    assert (persisted.instance_type, persisted.hourly_usd, persisted.gpu) == (
        "gpu_1x_h100_pcie",
        3.79,
        "H100",
    )
    assert terminated == [INSTANCE_ID]


def test_submit_without_capacity_requeues_before_launching(monkeypatch):
    rows = catalog(gpu_1x_h100_pcie=(), gpu_1x_h100_sxm5=())
    raised, launched, terminated = _submit(
        monkeypatch, lambda *a, **k: "done", lambda h: None, rows=rows
    )
    assert raised is CapacityUnavailableError
    assert launched == []
    assert terminated == []


@pytest.mark.parametrize("persistence", ["recorded", "missing"])
def test_shutdown_keeps_only_a_durably_recorded_vm(monkeypatch, persistence):
    from flash._internal.background import TaskStopped

    def stop(*args, **kwargs):
        raise TaskStopped

    callback = [].append if persistence == "recorded" else None
    raised, _launched, terminated = _submit(monkeypatch, stop, callback)
    assert raised is TaskStopped
    assert terminated == ([] if persistence == "recorded" else [INSTANCE_ID])


def test_poll_stamps_lambda_on_demand_cost(monkeypatch):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core.base import JobHandle

    captured = {}

    def fake_poll(adapter, **kwargs):
        captured["adapter"] = adapter
        return "polled"

    monkeypatch.setattr(jobs, "poll_instance_job", fake_poll)
    monkeypatch.setattr(jobs, "heartbeat_reader_for", lambda *a, **k: None)
    spec = JobSpec(
        run_id="flash-test",
        model="Qwen/Qwen3.5-9B-Base",
        train=TrainSpec(hf_repo="org/artifacts"),
        gpu=GpuSpec(type="H100", provider="lambda"),
    )
    persisted = JobHandle.from_dict(handle().to_dict())
    assert jobs.poll(persisted, spec, deadline_at=time.time() + 60) == "polled"
    adapter = captured["adapter"]
    assert adapter.running_status == "active"
    assert {"terminated", "terminating", "unhealthy"} <= adapter.dead_states
    metrics = {}
    adapter.stamp_cost_and_notes(metrics, end_ts=3700.0, launch_ts=100.0)
    assert metrics["cost_usd"] == pytest.approx(3.79)
    assert metrics["notes"]["provider"] == "lambda"
    assert metrics["notes"]["cost_basis"] == "lambda_on_demand"
