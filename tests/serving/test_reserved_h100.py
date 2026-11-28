"""Reserved H100 fleet: 12 x Qwen3.5-9B-Base; every other model uses the on-demand pool."""

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
BASE = "Qwen/Qwen3.5-9B-Base"
LARGE, DRAFTER = "Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-0.8B-Base"  # never reserved


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


def test_slot_layout_is_twelve_9b_base_slots():
    assert reserved.SLOT_COUNT == 12
    assert reserved.slots_for(BASE) == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    assert reserved.slots_for(LARGE) == reserved.slots_for(DRAFTER) == ()
    assert reserved.MODELS == (BASE,)
    with pytest.raises(ValueError, match="not declared"):
        reserved.model_for(13)


@pytest.mark.parametrize("environment", ["main", "", None])
def test_reserved_models_are_reserved_only_in_main(environment):
    assert reserved.enabled(BASE, environment)
    assert not reserved.enabled(LARGE, environment)
    assert not reserved.enabled(DRAFTER, environment)
    assert not reserved.enabled(BASE, "dev")


def test_the_reserved_model_places_on_a_single_h100_on_lambda():
    assert placements(BASE) == ("H100",)


def test_parse_workers_requires_each_entry_to_name_its_slot_model():
    raw = json.dumps([worker(s) for s in range(1, 13)])
    assert [w["slot"] for w in reserved.parse_workers(raw)] == list(range(1, 13))
    for bad in (
        [worker(1, model=LARGE)],  # an entry for another model
        [{k: v for k, v in worker(6).items() if k != "model"}],  # model omitted
        [{**worker(1), "slot": 13}],  # undeclared slot
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
    monkeypatch.setenv(reserved.WORKERS_ENV, json.dumps([worker(s) for s in range(1, 13)]))
    proxy = reserved.ReservedLambdaEngineProxy()
    proxy.base_model, proxy.reserved_slot = base_model, slot
    return proxy, transport


def test_proxy_binds_its_slot_worker_after_identity_check(monkeypatch):
    proxy, transport = proxy_for(5, BASE, health(BASE), monkeypatch)
    asyncio.run(proxy._load())
    assert proxy.engine is transport
    assert proxy._owner == "i5"


@pytest.mark.parametrize(
    ("slot", "base_model", "reported", "error"),
    [
        (2, DRAFTER, health(DRAFTER), "serves Qwen/Qwen3.5-9B-Base"),  # wrong model for the slot
        (2, BASE, health(LARGE), "identity mismatch"),  # leftover 27B worker
        (6, BASE, health(BASE, deployment_sha="b" * 40), "identity mismatch"),
        (6, BASE, health(BASE, configured_gpu="A100-80GB"), "identity mismatch"),
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
            [worker(s) for s in range(1, 13)],
            store,
            AsyncMock(),
            reserved.model_for,
            transport_factory=Transport,
        )
    )
    assert all(r["healthy"] for r in report.values())
    assert seen == dict.fromkeys(range(1, 13), (BASE, BASE))
    assert store[wd.CONFIGURED_KEY] == list(range(1, 13))


def test_a_worker_serving_another_model_fails_its_probe_without_a_canary():
    transport = MagicMock(call=AsyncMock(return_value=health(LARGE)))
    status, error = asyncio.run(wd.probe_status(transport, BASE))
    assert status == wd.FAILED
    assert "slot serves Qwen/Qwen3.5-9B-Base" in error
    transport.call.assert_awaited_once_with("/health")


@pytest.mark.parametrize(("lane", "status"), [(True, wd.FAILED), (False, wd.BUSY)])
def test_canary_timeout_is_a_hang_only_on_workers_with_a_canary_lane(monkeypatch, lane, status):
    sent = []

    async def call(path, body=None):
        if path == "/health":
            return health(BASE, **({"canary_lane": True} if lane else {}))
        sent.append(body)
        await asyncio.sleep(1)
        return None

    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.01)
    assert asyncio.run(wd.probe_status(MagicMock(call=call), BASE)) == (status, "canary timeout")
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


def test_routing_uses_only_healthy_configured_slots():
    states = {2: wd.SlotState(healthy=False).dump(), 5: wd.SlotState(healthy=False).dump()}
    store = MemoryStore({wd.CONFIGURED_KEY: [1, 2, 3, 5], **states})
    view = wd.HealthView(store, reserved.SLOT_MODELS)

    async def check():
        return [await view.next_slot(BASE) for _ in range(4)], await view.usable(BASE)

    picks, usable = asyncio.run(check())
    assert usable == [1, 3]
    assert picks == [1, 3, 1, 3]
    with pytest.raises(ValueError, match="No reserved slot"):
        asyncio.run(view.usable(LARGE))


def test_affinity_gives_each_checkpoint_three_stable_home_slots():
    slots = tuple(range(1, 9))
    keys = [f"org/flash-{i}/step-10" for i in range(400)]
    homes = {k: wd.home_slots(k, slots, slots) for k in keys}
    assert all(len(h) == 3 and set(h) <= set(slots) for h in homes.values())
    assert homes == {k: wd.home_slots(k, slots, slots) for k in keys}  # deterministic
    # spread: every slot is home to a fair share (3/8 of keys, within sampling noise)
    load = {s: sum(s in h for h in homes.values()) for s in slots}
    assert all(100 <= n <= 200 for n in load.values()), load


def test_affinity_moves_only_the_checkpoints_of_an_unhealthy_slot():
    slots = tuple(range(1, 9))
    keys = [f"org/flash-{i}/step-10" for i in range(400)]
    usable = [s for s in slots if s != 4]
    for k in keys:
        before = wd.home_slots(k, slots, slots)
        after = wd.home_slots(k, slots, usable)
        if 4 in before:
            assert set(before) - {4} <= set(after)
            assert 4 not in after
        else:
            assert after == before
    # with every home slot down, the next healthy slots in hash order take over
    k = keys[0]
    order = wd.affinity_order(k, slots)
    assert wd.home_slots(k, slots, order[3:]) == order[3:6]


def test_next_slot_keeps_a_checkpoint_on_its_home_slots(monkeypatch):
    store = MemoryStore({wd.CONFIGURED_KEY: list(range(1, 9))})
    view = wd.HealthView(store, reserved.SLOT_MODELS)

    async def picks(record, n):
        return [await view.next_slot(BASE, record) for _ in range(n)]

    ckpt = SimpleNamespace(org_id="org", model_id="ckpt", serve_base_model=False)
    served_base = SimpleNamespace(org_id="org", model_id=BASE, serve_base_model=True)
    assert wd.affinity_key(ckpt) == "org/ckpt"
    assert wd.affinity_key(served_base) is wd.affinity_key(None) is None
    home = set(wd.home_slots("org/ckpt", tuple(range(1, 9)), range(1, 9)))
    assert set(asyncio.run(picks(ckpt, 24))) == home
    assert set(asyncio.run(view.home(BASE, ckpt))) == home
    assert set(asyncio.run(picks(served_base, 16))) == set(range(1, 9))
    assert asyncio.run(view.home(BASE, None)) == list(range(1, 9))
    monkeypatch.setattr(wd, "AFFINITY_SLOTS", 0)  # off: back to plain round-robin
    assert set(asyncio.run(picks(ckpt, 16))) == set(range(1, 9))
    assert asyncio.run(view.home(BASE, ckpt)) == list(range(1, 9))


def test_interleaved_checkpoints_each_rotate_over_all_their_homes(monkeypatch):
    """Review repro: three checkpoints in a repeating order must not each stick to one home
    slot, as a rotation counter shared across checkpoints made them do."""
    store = MemoryStore({wd.CONFIGURED_KEY: list(range(1, 9))})
    view = wd.HealthView(store, reserved.SLOT_MODELS)
    records = [
        SimpleNamespace(org_id="org", model_id=f"checkpoint-{i}", serve_base_model=False)
        for i in range(3)
    ]

    async def rounds(n):
        picks = {r.model_id: [] for r in records}
        for _ in range(n):
            for r in records:
                picks[r.model_id].append(await view.next_slot(BASE, r))
        return picks

    picks = asyncio.run(rounds(6))
    for r in records:
        home = wd.home_slots(f"org/{r.model_id}", tuple(range(1, 9)), range(1, 9))
        assert picks[r.model_id] == home * 2  # every home, in order, twice

    # The remembered positions stay bounded on a long-lived router.
    monkeypatch.setattr(wd, "ROTATION_KEYS", 2)
    asyncio.run(rounds(1))
    assert len(view._rotation) == 2


def test_routing_fails_open_when_every_slot_is_quarantined():
    down = wd.SlotState(healthy=False).dump()
    store = MemoryStore({wd.CONFIGURED_KEY: [1, 2, 3], 1: down, 2: down, 3: down})
    view = wd.HealthView(store, reserved.SLOT_MODELS)
    assert asyncio.run(view.usable(BASE)) == [1, 2, 3]


def fleet(environment):
    from flash.serving.lambda_cloud.fleet import Fleet

    return Fleet(MagicMock(), MagicMock(), environment)


def test_main_never_allocates_on_demand_gpus_for_the_reserved_model():
    with pytest.raises(NoCapacityError, match="reserved H100s"):
        asyncio.run(fleet("main")._check_capacity(BASE, REV))


def test_shared_capacity_skips_reserved_models_in_main_only(monkeypatch):
    from flash.serving.lambda_cloud import pool_capacity

    monkeypatch.setattr(pool_capacity, "lambda_models", lambda: [BASE, LARGE])
    for environment in ("main", "dev"):
        capacity = pool_capacity.SharedCapacity(
            SimpleNamespace(environment=environment), None, None, REV
        )
        reconciled = []

        async def reconcile_model(model, *, revision=None, seen=reconciled):
            seen.append(model)

        capacity.reconcile_model = reconcile_model
        asyncio.run(capacity.reconcile(activating=True))
        assert (BASE in reconciled) is (environment == "dev")
        assert LARGE in reconciled
    with pytest.raises(NoCapacityError):
        asyncio.run(
            pool_capacity.SharedCapacity(
                SimpleNamespace(environment="main"), None, None, REV
            ).reconcile_model(BASE)
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
    assert reserved.enabled(BASE, "main") is exists
    assert not reserved.enabled(LARGE, "main")
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


def test_secret_absent_main_uses_the_on_demand_pool_for_9b_base(monkeypatch):
    from flash.serving.lambda_cloud import pool_capacity

    monkeypatch.setenv(reserved.ACTIVE_ENV, "0")
    assert not reserved.enabled(BASE, "main")
    monkeypatch.setattr(pool_capacity, "lambda_models", lambda: [BASE, LARGE])
    capacity = pool_capacity.SharedCapacity(SimpleNamespace(environment="main"), None, None, REV)
    reconciled = []

    async def reconcile_model(model, *, revision=None):
        reconciled.append(model)

    capacity.reconcile_model = reconcile_model
    asyncio.run(capacity.reconcile(activating=True))
    assert reconciled == [BASE, LARGE]
    assert wd.warm_names(BASE, "main", "Engine9BBase") == ["Engine9BBase"]


@pytest.mark.parametrize("on", [True, False])
def test_health_body_reports_reserved_models_only_while_reserved_is_on(monkeypatch, on):
    from flash.serving.src.http.routing import health_body

    monkeypatch.setenv(reserved.ACTIVE_ENV, "1" if on else "0")
    monkeypatch.delenv("SERVING_DEPLOYMENT_MODE", raising=False)
    router = SimpleNamespace(base_models=lambda: [LARGE, DRAFTER, BASE], ready_adapters=list)
    body = health_body(router, deployment_sha="s", deployment_id="d", capabilities=[])
    fallback = set(body["fallback_provider_by_model"])
    assert (BASE in fallback) is (not on)
    assert {LARGE, DRAFTER} <= fallback
    assert body["capacity_by_model"][BASE]["min"] == (12 if on else 0)
    assert body["capacity_by_model"][LARGE]["min"] == 0
    assert (body["gpu_by_model"][BASE] == "H100") is on
