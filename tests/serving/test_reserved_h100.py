"""Reserved H100 fleet: 5 x Qwen3.8-27B + 3 x Qwen3.5-0.8B-Base, one model per slot."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from flash.serving.lambda_cloud import reserved
from flash.serving.lambda_cloud import reserved_watchdog as wd
from flash.serving.lambda_cloud.config import NoCapacityError, placements

REV = "a" * 40
TLS = {"cert": "C", "token": "T"}
LARGE, DRAFTER = "Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-0.8B-Base"


def worker(slot, **extra):
    return {
        "slot": slot,
        "model": reserved.model_for(slot),
        "instance_id": f"i{slot}",
        "ip": f"10.0.0.{slot}",
        "revision": REV,
        "tls": TLS,
        **extra,
    }


def health(model, **extra):
    return {
        "ok": True,
        "base_model": model,
        "deployment_sha": REV,
        "configured_gpu": "H100",
        **extra,
    }


class MemoryStore(dict):
    async def get(self, key):
        return super().get(key)

    async def put(self, key, value):
        self[key] = value


def test_slot_layout_is_five_27b_then_three_08b_base_and_no_9b_base():
    assert reserved.SLOT_COUNT == 8
    assert reserved.slots_for(LARGE) == (1, 2, 3, 4, 5)
    assert reserved.slots_for(DRAFTER) == (6, 7, 8)
    assert reserved.slots_for("Qwen/Qwen3.5-9B-Base") == ()
    assert reserved.MODELS == (LARGE, DRAFTER)
    with pytest.raises(ValueError, match="not declared"):
        reserved.model_for(9)


@pytest.mark.parametrize("environment", ["main", "", None])
def test_reserved_models_are_reserved_only_in_main(environment):
    assert reserved.enabled(LARGE, environment)
    assert reserved.enabled(DRAFTER, environment)
    assert not reserved.enabled("Qwen/Qwen3.5-9B-Base", environment)
    assert not reserved.enabled(LARGE, "dev")


def test_both_reserved_models_can_place_on_a_single_h100_on_lambda():
    # Reserved slots are explicitly provisioned H100s, so H100 must stay a qualified shape.
    assert "H100" in placements(LARGE)
    assert "H100" in placements(DRAFTER)


def test_parse_workers_requires_each_entry_to_name_its_slot_model():
    raw = json.dumps([worker(s) for s in range(1, 9)])
    assert [w["slot"] for w in reserved.parse_workers(raw)] == list(range(1, 9))
    for bad in (
        [worker(1, model=DRAFTER)],  # a drafter entry in a 27B slot
        [{k: v for k, v in worker(6).items() if k != "model"}],  # model omitted
        [{**worker(1), "slot": 9}],  # undeclared slot
        [worker(1), {**worker(2), "instance_id": "i1"}],  # duplicate instance
        [worker(1), worker(1)],  # duplicate slot
        [],
    ):
        with pytest.raises(ValueError, match=r"Reserved fleet|Every reserved worker"):
            reserved.parse_workers(json.dumps(bad))


def proxy_for(slot, base_model, reported, monkeypatch):
    from flash.serving.lambda_cloud import reserved as module

    transport = MagicMock(call=AsyncMock(return_value=reported), aclose=AsyncMock())
    monkeypatch.setattr(module, "WorkerTransport", lambda ip, tls: transport)
    monkeypatch.setenv(reserved.WORKERS_ENV, json.dumps([worker(s) for s in range(1, 9)]))
    proxy = reserved.ReservedLambdaEngineProxy()
    proxy.base_model, proxy.reserved_slot = base_model, slot
    return proxy, transport


def test_proxy_binds_its_slot_worker_after_identity_check(monkeypatch):
    proxy, transport = proxy_for(7, DRAFTER, health(DRAFTER), monkeypatch)
    asyncio.run(proxy._load())
    assert proxy.engine is transport
    assert proxy._owner == "i7"


@pytest.mark.parametrize(
    ("slot", "base_model", "reported", "error"),
    [
        (2, DRAFTER, health(DRAFTER), "serves Qwen/Qwen3.8-27B"),  # wrong model for the slot
        (2, LARGE, health("Qwen/Qwen3.5-9B-Base"), "identity mismatch"),  # leftover 9B worker
        (6, DRAFTER, health(DRAFTER, deployment_sha="b" * 40), "identity mismatch"),
        (6, DRAFTER, health(DRAFTER, configured_gpu="A100-80GB"), "identity mismatch"),
    ],
)
def test_proxy_refuses_a_worker_with_the_wrong_identity(
    monkeypatch, slot, base_model, reported, error
):
    proxy, transport = proxy_for(slot, base_model, reported, monkeypatch)
    with pytest.raises((RuntimeError, ValueError), match=error):
        asyncio.run(proxy._load())
    if error == "identity mismatch":
        transport.aclose.assert_awaited_once()


def test_watchdog_probes_each_worker_with_its_slot_model():
    seen = {}

    class Transport:
        def __init__(self, ip, tls):
            self.slot = int(ip.rsplit(".", 1)[1])

        async def call(self, path, body=None):
            model = reserved.model_for(self.slot)
            if path == "/health":
                return health(model)
            seen[self.slot] = (body["payload"]["model_id"], body["record"]["base_model"])
            return {}

        async def aclose(self):
            pass

    store = MemoryStore()
    report = asyncio.run(
        wd.run_once(
            [worker(s) for s in range(1, 9)],
            store,
            AsyncMock(),
            reserved.model_for,
            transport_factory=Transport,
        )
    )
    assert all(r["healthy"] for r in report.values())
    assert seen == {s: (reserved.model_for(s),) * 2 for s in range(1, 9)}
    assert store[wd.CONFIGURED_KEY] == list(range(1, 9))


def test_a_worker_serving_another_model_fails_its_probe_without_a_canary():
    transport = MagicMock(call=AsyncMock(return_value=health("Qwen/Qwen3.5-9B-Base")))
    status, error = asyncio.run(wd.probe_status(transport, LARGE))
    assert status == wd.FAILED
    assert "slot serves Qwen/Qwen3.8-27B" in error
    transport.call.assert_awaited_once_with("/health")


@pytest.mark.parametrize(("lane", "status"), [(True, wd.FAILED), (False, wd.BUSY)])
def test_canary_timeout_is_a_hang_only_on_workers_with_a_canary_lane(monkeypatch, lane, status):
    sent = []

    async def call(path, body=None):
        if path == "/health":
            return health(LARGE, **({"canary_lane": True} if lane else {}))
        sent.append(body)
        await asyncio.sleep(1)
        return None

    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.01)
    assert asyncio.run(wd.probe_status(MagicMock(call=call), LARGE)) == (status, "canary timeout")
    assert sent[0]["canary"] is True
    assert sent[0]["payload"]["max_tokens"] == 1
    assert sent[0]["record"]["serve_base_model"] is True


def test_laneless_canary_timeout_is_busy_until_generations_stop_finishing():
    state, restart = wd.advance(wd.SlotState(), False, 0, "canary timeout", busy=True)
    assert state.healthy
    assert state.failures == 0
    assert not restart
    state, _ = wd.advance(state, False, wd.BUSY_HUNG_AFTER_SECONDS, "canary timeout", busy=True)
    assert state.failures == 1


def test_routing_uses_only_healthy_configured_slots_of_the_requested_model():
    states = {2: wd.SlotState(healthy=False).dump(), 7: wd.SlotState(healthy=False).dump()}
    store = MemoryStore({wd.CONFIGURED_KEY: [1, 2, 3, 6, 7], **states})
    view = wd.HealthView(store, reserved.SLOT_MODELS)

    async def check():
        large = [await view.next_slot(LARGE) for _ in range(4)]
        drafter = [await view.next_slot(DRAFTER) for _ in range(2)]
        return large, drafter, await view.usable(LARGE), await view.usable(DRAFTER)

    large, drafter, usable_large, usable_drafter = asyncio.run(check())
    assert usable_large == [1, 3]
    assert large == [1, 3, 1, 3]
    assert usable_drafter == [6]
    assert drafter == [6, 6]
    with pytest.raises(ValueError, match="No reserved slot"):
        asyncio.run(view.usable("Qwen/Qwen3.5-9B-Base"))


def test_routing_fails_open_within_a_model_when_all_its_slots_are_quarantined():
    down = wd.SlotState(healthy=False).dump()
    store = MemoryStore({wd.CONFIGURED_KEY: list(range(1, 9)), 6: down, 7: down, 8: down})
    view = wd.HealthView(store, reserved.SLOT_MODELS)
    assert asyncio.run(view.usable(DRAFTER)) == [6, 7, 8]
    assert asyncio.run(view.usable(LARGE)) == [1, 2, 3, 4, 5]


def fleet(environment):
    from flash.serving.lambda_cloud.fleet import Fleet

    return Fleet(MagicMock(), MagicMock(), environment)


@pytest.mark.parametrize("model", [LARGE, DRAFTER])
def test_main_never_allocates_on_demand_gpus_for_reserved_models(model):
    with pytest.raises(NoCapacityError, match="reserved H100s"):
        asyncio.run(fleet("main")._check_capacity(model, REV))


def test_shared_capacity_skips_reserved_models_in_main_only(monkeypatch):
    from flash.serving.lambda_cloud import pool_capacity

    monkeypatch.setattr(pool_capacity, "lambda_models", lambda: [LARGE, "Qwen/Qwen3.5-9B-Base"])
    for environment in ("main", "dev"):
        capacity = pool_capacity.SharedCapacity(
            SimpleNamespace(environment=environment), None, None, REV
        )
        reconciled = []

        async def reconcile_model(model, *, revision=None, seen=reconciled):
            seen.append(model)

        capacity.reconcile_model = reconcile_model
        asyncio.run(capacity.reconcile(activating=True))
        assert (LARGE in reconciled) is (environment == "dev")
        assert "Qwen/Qwen3.5-9B-Base" in reconciled
    with pytest.raises(NoCapacityError):
        asyncio.run(
            pool_capacity.SharedCapacity(
                SimpleNamespace(environment="main"), None, None, REV
            ).reconcile_model(DRAFTER)
        )


class _NotFound(Exception):
    pass


def _modal_with_secret(monkeypatch, exists):
    import modal

    lookups = []

    def from_name(name, environment_name=None):
        lookups.append((name, environment_name))

        def hydrate():
            if not exists:
                raise _NotFound(name)

        return SimpleNamespace(hydrate=hydrate)

    monkeypatch.setattr(modal.Secret, "from_name", from_name)
    monkeypatch.setattr(modal.exception, "NotFoundError", _NotFound)
    return lookups


@pytest.mark.parametrize("exists", [True, False])
def test_active_follows_the_workers_secret_in_main_and_is_decided_once(monkeypatch, capsys, exists):
    monkeypatch.delenv(reserved.ACTIVE_ENV)
    lookups = _modal_with_secret(monkeypatch, exists)
    assert reserved.active("main") is exists
    assert reserved.active("") is exists  # Modal's unset environment is main
    assert lookups == [(reserved.SECRET_NAME, "main")]
    assert reserved.baked_env("main") == {reserved.ACTIVE_ENV: "1" if exists else "0"}
    assert reserved.enabled(LARGE, "main") is exists
    assert reserved.enabled(DRAFTER, "main") is exists
    out = capsys.readouterr().out
    assert f"reserved H100 serving {'ON' if exists else 'OFF'} in main" in out


def test_dev_is_never_reserved_and_never_looks_the_secret_up(monkeypatch):
    monkeypatch.delenv(reserved.ACTIVE_ENV)
    lookups = _modal_with_secret(monkeypatch, True)
    assert reserved.active("dev") is False
    assert reserved.baked_env("dev") == {reserved.ACTIVE_ENV: "0"}
    assert lookups == []


def test_baked_decision_wins_in_remote_containers(monkeypatch):
    lookups = _modal_with_secret(monkeypatch, True)
    monkeypatch.setenv(reserved.ACTIVE_ENV, "0")
    assert reserved.active("main") is False
    monkeypatch.setenv(reserved.ACTIVE_ENV, "1")
    assert reserved.active("main") is True
    assert lookups == []


def test_secret_lookup_errors_other_than_not_found_fail_the_deploy(monkeypatch):
    import modal

    def from_name(name, environment_name=None):
        def hydrate():
            raise ConnectionError("modal unreachable")

        return SimpleNamespace(hydrate=hydrate)

    monkeypatch.setattr(modal.Secret, "from_name", from_name)
    with pytest.raises(ConnectionError, match="unreachable"):
        reserved.secret_exists("main")


def test_secret_absent_main_uses_the_on_demand_pool_for_both_models(monkeypatch):
    from flash.serving.lambda_cloud import pool_capacity

    monkeypatch.setenv(reserved.ACTIVE_ENV, "0")
    for model in (LARGE, DRAFTER):
        assert not reserved.enabled(model, "main")
    monkeypatch.setattr(pool_capacity, "lambda_models", lambda: [LARGE, DRAFTER])
    capacity = pool_capacity.SharedCapacity(SimpleNamespace(environment="main"), None, None, REV)
    reconciled = []

    async def reconcile_model(model, *, revision=None):
        reconciled.append(model)

    capacity.reconcile_model = reconcile_model
    asyncio.run(capacity.reconcile(activating=True))
    assert reconciled == [LARGE, DRAFTER]
    assert wd.warm_names(LARGE, "main", "Engine27B") == ["Engine27B"]


@pytest.mark.parametrize("on", [True, False])
def test_health_body_reports_reserved_models_only_while_reserved_is_on(monkeypatch, on):
    from flash.serving.src.http.routing import health_body

    monkeypatch.setenv(reserved.ACTIVE_ENV, "1" if on else "0")
    monkeypatch.delenv("SERVING_DEPLOYMENT_MODE", raising=False)
    router = SimpleNamespace(
        base_models=lambda: [LARGE, DRAFTER, "Qwen/Qwen3.5-9B-Base"], ready_adapters=list
    )
    body = health_body(router, deployment_sha="s", deployment_id="d", capabilities=[])
    # No model, reserved or on-demand, has a Modal GPU fallback to report.
    assert "fallback_provider_by_model" not in body
    assert body["capacity_by_model"][LARGE]["min"] == (5 if on else 0)
    assert body["capacity_by_model"][DRAFTER]["min"] == (3 if on else 0)
    assert body["capacity_by_model"]["Qwen/Qwen3.5-9B-Base"]["min"] == 0
    assert (body["gpu_by_model"][LARGE] == "H100") is on
