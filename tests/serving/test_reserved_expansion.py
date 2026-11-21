import pytest

from flash.serving.lambda_cloud import reserved
from flash.serving.lambda_cloud import reserved_expansion as ex
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
        set(range(1, 9)),
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
    assert body["name"] == "flash-h100-reservation-20261112-9"
    assert body["firewall_rulesets"] == [{"id": "main-us-west-3"}]
    assert body["user_data"] == f"init {REV} K"
    assert store["slot-9"]["tls"] == TLS  # never the private key

    await tick(store, api, now=1200)
    assert names(api) == ["9", "10"]

    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1400)
    assert store[ex.WORKERS_KEY] == [
        {"slot": 9, "instance_id": "new-1", "ip": "1.2.3.4", "revision": REV, "tls": TLS}
    ]


@pytest.mark.anyio
async def test_worker_with_wrong_identity_is_not_published():
    class Wrong(Transport):
        health = {**HEALTHY, "deployment_sha": "c" * 40}  # noqa: RUF012

    store, api = Store(), FakeAPI(offer("us-west-3"))
    await tick(store, api)
    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1200, transport=Wrong)
    assert ex.WORKERS_KEY not in store
    assert store["slot-9"]["state"] == "qualifying"


@pytest.mark.anyio
async def test_rejected_placement_rotates_then_retries_after_cooldown():
    catalog = {**offer("us-west-3"), **offer("us-south-3", "gpu_1x_h100_sxm5", 489)}
    store, api = Store(), FakeAPI(catalog, reject={"us-west-3"})
    await tick(store, api)
    assert "slot-9" not in store
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
    assert names(api) == ["9", "10"]  # slot 9 still pending
    await tick(store, api, now=1000 + ex.AMBIGUOUS_LAUNCH_SECONDS + 1)
    assert "slot-9" not in store


@pytest.mark.anyio
async def test_unrecorded_instance_is_never_adopted():
    store, api = Store({ex.TARGET_KEY: [9]}), FakeAPI(offer("us-west-3"))
    api.instances.append(
        {"id": "x", "name": ex.instance_name(9), "status": "active", "ip": "1.1.1.1"}
    )
    assert "review" in (await tick(store, api))[9]
    assert api.launches == []


@pytest.mark.anyio
async def test_gone_worker_is_unrouted_and_replaced():
    store, api = Store({ex.TARGET_KEY: [9]}), FakeAPI(offer("us-west-3"))
    await tick(store, api)
    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1200)
    api.instances[0]["status"] = "terminated"
    await tick(store, api, now=1400)
    assert store[ex.WORKERS_KEY] == []
    await tick(store, api, now=1600)
    assert names(api) == ["9", "9"]


@pytest.mark.anyio
async def test_stale_publication_is_dropped_and_replacement_published():
    stale = {"slot": 9, "instance_id": "old", "ip": "8.8.4.4", "revision": REV, "tls": TLS}
    store = Store({ex.TARGET_KEY: [9], ex.WORKERS_KEY: [stale]})
    api = FakeAPI(offer("us-west-3"))
    await tick(store, api)
    assert store[ex.WORKERS_KEY] == []
    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1200)
    assert [w["instance_id"] for w in store[ex.WORKERS_KEY]] == ["new-1"]


def test_merge_keeps_secret_and_falls_back_on_malformed_entries():
    secret = [{"slot": 9, "instance_id": "s9", "ip": "8.8.8.8", "revision": REV, "tls": TLS}]
    good = {"slot": 10, "instance_id": "e10", "ip": "8.8.4.4", "revision": REV, "tls": TLS}
    merged = ex.merge_workers(secret, [dict(good, slot=9), good])
    assert [w["instance_id"] for w in merged] == ["s9", "e10"]
    for bad in (
        {"slot": 13},
        dict(good, slot=11, tls={}),
        dict(good, slot=11, instance_id="s9"),
        "junk",
    ):
        assert ex.merge_workers(secret, [good, bad]) == secret
