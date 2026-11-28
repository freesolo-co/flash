"""Unattended acquisition of the reserved H100s: ten 9B Base slots by default, the two 27B
slots only once targeted."""

import json

import pytest

from flash.serving.lambda_cloud import reserved
from flash.serving.lambda_cloud import reserved_acquire as ex
from flash.serving.lambda_cloud.api import LambdaAPIError

REV = "b" * 40
TLS = {"cert": "C", "token": "T"}
HEALTHY = {
    "ok": True,
    "canary_lane": True,
    "base_model": reserved.MODEL,
    "deployment_sha": REV,
    "configured_gpu": "H100",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _launch_helpers(monkeypatch):
    from flash.serving.lambda_cloud import bootstrap, resources

    async def fake(api, env, region):
        return f"{env}-{region}"

    monkeypatch.setattr(resources, "firewall", fake)
    monkeypatch.setattr(resources, "cache_filesystem", fake)
    monkeypatch.setattr(bootstrap, "identity", lambda: {"key": "K", **TLS})
    monkeypatch.setattr(bootstrap, "user_data", lambda m, rev, c, tls: f"init {rev} {tls['key']}")


class Store(dict):
    async def get(self, key):
        return super().get(key)

    async def put(self, key, value):
        self[key] = value

    async def pop(self, key):
        return super().pop(key, None)


def offer(region, kind="gpu_1x_h100_pcie", cents=379):
    return {
        kind: {
            "instance_type": {
                "gpu_description": "H100 (80 GB PCIe)",
                "specs": {"gpus": 1},
                "architecture": "x86_64",
                "price_cents_per_hour": cents,
            },
            "regions_with_capacity_available": [{"name": region}] if region else [],
        }
    }


class FakeAPI:
    def __init__(self, catalog, reject=()):
        self.instances, self.catalog, self.launches, self.reject = [], catalog, [], reject
        self.error = None

    async def call(self, method, path, body=None):
        if path == "instances":
            return list(self.instances)
        if path == "instance-types":
            return self.catalog
        self.launches.append(body)
        if body["region_name"] in self.reject:
            raise LambdaAPIError(400, "instance-operations/launch/insufficient-capacity")
        if self.error:
            raise self.error
        iid = f"new-{len(self.launches)}"
        self.instances.append({"id": iid, "name": body["name"], "status": "booting", "ip": None})
        return {"instance_ids": [iid]}


class Transport:
    health = HEALTHY

    def __init__(self, ip, tls):
        pass

    async def call(self, path, body=None):
        return self.health if path == "/health" else {}

    async def aclose(self):
        pass


async def tick(store, api, now=1000.0, transport=Transport):
    return await ex.run_once(
        store,
        api,
        set(),
        revision=REV,
        credentials={},
        transport_factory=transport,
        clock=lambda: now,
    )


def names(api):
    return [b["name"].rsplit("-", 1)[1] for b in api.launches]


@pytest.mark.anyio
async def test_no_capacity_retries_without_launching():
    store, api = Store(), FakeAPI(offer(None))
    assert "no H100 capacity" in (await tick(store, api))["launch"]
    assert api.launches == []
    assert store == {}


@pytest.mark.anyio
async def test_launch_one_per_cycle_then_qualify_and_publish():
    store, api = Store(), FakeAPI(offer("us-west-3"))
    await tick(store, api)
    body = api.launches[0]
    assert body["name"] == "flash-h100-reservation-20261127-1"
    assert body["firewall_rulesets"] == [{"id": "main-us-west-3"}]
    assert body["user_data"] == f"init {REV} K"
    assert store["slot-1"]["tls"] == TLS  # never the private key

    await tick(store, api, now=1200)
    assert names(api) == ["1", "2"]

    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1400)
    entry = {
        "slot": 1,
        "model": reserved.MODEL,
        "instance_id": "new-1",
        "ip": "1.2.3.4",
        "revision": REV,
        "tls": TLS,
    }
    assert store[ex.WORKERS_KEY] == [entry]
    # Published entries are valid workers-secret entries as they stand.
    assert reserved.parse_workers(json.dumps(store[ex.WORKERS_KEY])) == [entry]


@pytest.mark.anyio
async def test_worker_with_wrong_identity_is_not_published():
    class Wrong(Transport):
        health = {**HEALTHY, "deployment_sha": "c" * 40}  # noqa: RUF012

    store, api = Store(), FakeAPI(offer("us-west-3"))
    await tick(store, api)
    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1200, transport=Wrong)
    assert ex.WORKERS_KEY not in store
    assert store["slot-1"]["state"] == "qualifying"


@pytest.mark.anyio
async def test_rejected_placement_rotates_then_retries_after_cooldown():
    catalog = {**offer("us-west-3"), **offer("us-south-3", "gpu_1x_h100_sxm5", 489)}
    store, api = Store(), FakeAPI(catalog, reject={"us-west-3"})
    await tick(store, api)
    assert "slot-1" not in store
    await tick(store, api, now=1100)
    await tick(store, api, now=1000 + ex.REJECTION_COOLDOWN_SECONDS + 1)
    assert [b["region_name"] for b in api.launches] == ["us-west-3", "us-south-3", "us-west-3"]


@pytest.mark.anyio
async def test_ambiguous_launch_is_not_resubmitted_until_it_times_out():
    store, api = Store(), FakeAPI(offer("us-west-3"))
    api.error = TimeoutError()
    await tick(store, api)
    api.error = None
    await tick(store, api, now=1100)
    assert names(api) == ["1", "2"]  # slot 1 still pending
    await tick(store, api, now=1000 + ex.AMBIGUOUS_LAUNCH_SECONDS + 1)
    assert "slot-1" not in store


@pytest.mark.anyio
async def test_unrecorded_instance_is_never_adopted():
    store, api = Store({ex.TARGET_KEY: [1]}), FakeAPI(offer("us-west-3"))
    api.instances.append(
        {"id": "x", "name": ex.instance_name(1), "status": "active", "ip": "1.1.1.1"}
    )
    assert "review" in (await tick(store, api))[1]
    assert api.launches == []


@pytest.mark.anyio
async def test_gone_worker_is_unrouted_and_replaced():
    store, api = Store({ex.TARGET_KEY: [1]}), FakeAPI(offer("us-west-3"))
    await tick(store, api)
    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1200)
    api.instances[0]["status"] = "terminated"
    await tick(store, api, now=1400)
    assert store[ex.WORKERS_KEY] == []
    await tick(store, api, now=1600)
    assert names(api) == ["1", "1"]


@pytest.mark.anyio
async def test_stale_publication_is_dropped_and_replacement_published():
    stale = {
        "slot": 1,
        "model": reserved.MODEL,
        "instance_id": "old",
        "ip": "8.8.4.4",
        "revision": REV,
        "tls": TLS,
    }
    store = Store({ex.TARGET_KEY: [1], ex.WORKERS_KEY: [stale]})
    api = FakeAPI(offer("us-west-3"))
    await tick(store, api)
    assert store[ex.WORKERS_KEY] == []
    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1200)
    assert [w["instance_id"] for w in store[ex.WORKERS_KEY]] == ["new-1"]


@pytest.mark.anyio
async def test_slots_held_by_the_secret_and_stopped_targets_are_never_acquired():
    store, api = Store(), FakeAPI(offer("us-west-3"))
    await ex.run_once(
        store,
        api,
        {1, 2, 3},
        revision=REV,
        credentials={},
        transport_factory=Transport,
        clock=lambda: 1000.0,
    )
    assert names(api) == ["4"]
    store, api = Store({ex.TARGET_KEY: []}), FakeAPI(offer("us-west-3"))
    assert "launch" not in await tick(store, api)
    assert api.launches == []


@pytest.mark.anyio
async def test_only_h100_shapes_are_launched():
    catalog = {
        "gpu_1x_a100": {
            "instance_type": {
                "gpu_description": "A100 (40 GB SXM4)",
                "specs": {"gpus": 1},
                "architecture": "x86_64",
                "price_cents_per_hour": 129,
            },
            "regions_with_capacity_available": [{"name": "us-east-1"}],
        }
    }
    store, api = Store(), FakeAPI(catalog)
    await tick(store, api)
    assert api.launches == []


LARGE_HEALTHY = {**HEALTHY, "base_model": reserved.LARGE_MODEL}


@pytest.mark.anyio
async def test_27b_slots_are_opt_in():
    store, api = Store(), FakeAPI(offer("us-west-3"))
    # Every 9B Base slot already in the secret: nothing left by default, 27B never bought.
    report = await ex.run_once(
        store,
        api,
        set(reserved.slots_for(reserved.MODEL)),
        revision=REV,
        credentials={},
        transport_factory=Transport,
        clock=lambda: 1000.0,
    )
    assert "launch" not in report
    assert api.launches == []
    assert tuple(range(1, 11)) == ex.DEFAULT_TARGETS


@pytest.mark.anyio
async def test_targeted_27b_slot_launches_qualifies_and_publishes_as_27b(monkeypatch):
    from flash.serving.lambda_cloud import bootstrap

    booted = []
    monkeypatch.setattr(
        bootstrap, "user_data", lambda m, rev, c, tls: booted.append(m) or f"init {m}"
    )

    class Large(Transport):
        health = LARGE_HEALTHY

    store, api = Store({ex.TARGET_KEY: [11]}), FakeAPI(offer("us-west-3"))
    await tick(store, api)
    body = api.launches[0]
    assert body["name"] == "flash-27b-reservation-20261128-11"
    assert {"key": "purpose", "value": "flash-27b-serving"} in body["tags"]
    assert booted == [reserved.LARGE_MODEL]
    assert store["slot-11"]["model"] == reserved.LARGE_MODEL

    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1200, transport=Transport)  # a 9B worker: refused
    assert ex.WORKERS_KEY not in store
    await tick(store, api, now=1400, transport=Large)
    [entry] = store[ex.WORKERS_KEY]
    assert entry["slot"] == 11
    assert entry["model"] == reserved.LARGE_MODEL
    assert reserved.parse_workers(json.dumps(store[ex.WORKERS_KEY])) == [entry]


@pytest.mark.anyio
async def test_a_9b_record_or_entry_on_a_27b_slot_is_never_carried_over():
    """Slots 11-12 served 9B Base before 27B: their old records and entries are dropped, and
    the old instance name never matches the 27B slot."""
    old_entry = {
        "slot": 11,
        "model": reserved.MODEL,
        "instance_id": "old-9b",
        "ip": "8.8.4.4",
        "revision": REV,
        "tls": TLS,
    }
    old_record = {"state": "serving", "submitted_at": 0, "instance_id": "old-9b", "revision": REV}
    store = Store({ex.TARGET_KEY: [11], ex.WORKERS_KEY: [old_entry], "slot-11": old_record})
    api = FakeAPI(offer("us-west-3"))
    api.instances.append(
        {"id": "old-9b", "name": "flash-h100-reservation-20261127-11", "status": "active"}
    )
    await tick(store, api)
    assert store[ex.WORKERS_KEY] == []
    assert store["slot-11"]["model"] == reserved.LARGE_MODEL  # relaunched as 27B
    assert names(api) == ["11"]
    assert api.launches[0]["name"].startswith(ex.LARGE_CAMPAIGN)


@pytest.mark.anyio
async def test_a_published_entry_for_another_model_is_dropped_even_off_target():
    wrong = {"slot": 12, "model": reserved.MODEL, "instance_id": "x", "ip": "1.1.1.1"}
    store = Store({ex.TARGET_KEY: [], ex.WORKERS_KEY: [wrong]})
    await tick(store, FakeAPI(offer(None)))
    assert store[ex.WORKERS_KEY] == []
