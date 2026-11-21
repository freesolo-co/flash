import json

import pytest

from flash.serving.lambda_cloud import reserved
from flash.serving.lambda_cloud import reserved_expansion as ex
from flash.serving.lambda_cloud.api import LambdaAPIError

REV = "b" * 40
HEALTHY = {
    "ok": True,
    "engine_dead": False,
    "canary_lane": True,
    "base_model": reserved.MODEL,
    "deployment_sha": REV,
    "configured_gpu": "H100",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Store:
    def __init__(self, data=None):
        self.data = dict(data or {})

    async def get(self, key):
        return self.data.get(key)

    async def put(self, key, value):
        self.data[key] = value

    async def pop(self, key):
        return self.data.pop(key, None)


def catalog(*regions, kind="gpu_1x_h100_pcie", cents=379):
    return {
        kind: {
            "instance_type": {
                "name": kind,
                "gpu_description": "H100 (80 GB PCIe)",
                "specs": {"gpus": 1},
                "price_cents_per_hour": cents,
            },
            "regions_with_capacity_available": [{"name": r} for r in regions],
        },
        "gpu_8x_h100_sxm5": {
            "instance_type": {"gpu_description": "H100", "specs": {"gpus": 8}},
            "regions_with_capacity_available": [{"name": "us-east-1"}],
        },
    }


class FakeAPI:
    def __init__(self, cat=None, launch_error=None):
        self.instances = []
        self.catalog = cat or {}
        self.launches = []
        self.launch_error = launch_error
        self.next_id = 0

    async def call(self, method, path, body=None):
        if path == "instances":
            return list(self.instances)
        if path == "instance-types":
            return self.catalog
        if path == "instance-operations/launch":
            self.launches.append(body)
            if self.launch_error:
                raise self.launch_error
            self.next_id += 1
            iid = f"new-{self.next_id}"
            self.instances.append(
                {
                    "id": iid,
                    "name": body["name"],
                    "status": "booting",
                    "ip": None,
                    "instance_type": {
                        "name": body["instance_type_name"],
                        "price_cents_per_hour": 379,
                    },
                    "region": {"name": body["region_name"]},
                }
            )
            return {"instance_ids": [iid]}
        raise AssertionError(path)


class Transport:
    health = HEALTHY
    generate_error = None

    def __init__(self, ip, tls):
        self.tls = tls

    async def call(self, path, body=None):
        if path == "/health":
            return self.health
        if self.generate_error:
            raise self.generate_error
        return {"text": "1"}

    async def aclose(self):
        pass


async def tick(store, api, secret_slots=frozenset(range(1, 9)), now=1000.0, transport=Transport):
    async def firewall(api, env, region):
        assert env == "main"
        return f"fw-{region}"

    async def filesystem(api, env, region):
        return f"fs-{region}"

    return await ex.run_once(
        store,
        api,
        set(secret_slots),
        revision=REV,
        credentials={"HF_TOKEN": "hf"},
        transport_factory=transport,
        identity=lambda: {"key": "KEY", "cert": "CERT", "token": "TOKEN"},
        user_data=lambda model, rev, creds, tls: f"cloud-init {model} {rev} {tls['key']}",
        firewall=firewall,
        filesystem=filesystem,
        clock=lambda: now,
    )


def test_target_slots_default_and_validation():
    assert ex.target_slots(None) == [9, 10, 11, 12]
    assert ex.target_slots([10, 9, 9]) == [9, 10]
    assert ex.target_slots([]) == []
    with pytest.raises(ValueError, match="target slots"):
        ex.target_slots([13])


def test_choices_prefers_cheapest_single_h100_in_us():
    cat = {
        **catalog("us-west-3", "eu-central-1"),
        **catalog("us-south-3", kind="gpu_1x_h100_sxm5", cents=489),
    }
    assert ex.choices(cat) == [
        (379, "gpu_1x_h100_pcie", "us-west-3"),
        (489, "gpu_1x_h100_sxm5", "us-south-3"),
    ]
    assert ex.choices(catalog()) == []


@pytest.mark.anyio
async def test_no_capacity_retries_without_launching():
    store, api = Store(), FakeAPI(catalog())
    report = await tick(store, api)
    assert api.launches == []
    assert "no single-H100 capacity" in report["capacity"]
    assert store.data == {}


@pytest.mark.anyio
async def test_launch_one_per_cycle_then_qualify_and_publish():
    store, api = Store(), FakeAPI(catalog("us-west-3"))
    await tick(store, api)
    assert len(api.launches) == 1
    body = api.launches[0]
    assert body["name"] == "flash-h100-reservation-20261112-9"
    assert body["firewall_rulesets"] == [{"id": "fw-us-west-3"}]
    assert body["file_system_mounts"][0]["mount_point"] == "/data/hosting-cache"
    assert body["user_data"] == f"cloud-init {reserved.MODEL} {REV} KEY"
    record = store.data["slot-9"]
    assert record["instance_id"] == "new-1"
    assert "key" not in record["tls"]

    await tick(store, api, now=1200)  # slot 9 booting; slot 10 launched
    assert [b["name"][-2:] for b in api.launches] == ["-9", "10"]
    assert store.data["slot-9"]["state"] == "booting"

    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1400)
    assert store.data[ex.WORKERS_KEY] == [
        {
            "slot": 9,
            "instance_id": "new-1",
            "ip": "1.2.3.4",
            "revision": REV,
            "tls": {"cert": "CERT", "token": "TOKEN"},
        }
    ]
    merged = ex.merge_workers(
        [{"slot": 1, "instance_id": "a", "ip": "8.8.8.8", "revision": REV, "tls": {}}],
        store.data[ex.WORKERS_KEY],
    )
    assert [w["slot"] for w in merged] == [1, 9]


@pytest.mark.anyio
async def test_unqualified_worker_is_not_published():
    store, api = Store(), FakeAPI(catalog("us-west-3"))
    await tick(store, api)

    class Wrong(Transport):
        async def call(self, path, body=None):
            return {**HEALTHY, "deployment_sha": "c" * 40}

    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1200, transport=Wrong)
    assert ex.WORKERS_KEY not in store.data
    assert store.data["slot-9"]["state"] == "qualifying"


@pytest.mark.anyio
async def test_explicit_rejection_frees_the_slot_ambiguous_does_not():
    store = Store()
    api = FakeAPI(
        catalog("us-west-3"),
        LambdaAPIError(400, "instance-operations/launch/insufficient-capacity"),
    )
    await tick(store, api)
    assert "slot-9" not in store.data

    api.launch_error = TimeoutError()
    await tick(store, api)
    assert store.data["slot-9"]["instance_id"] is None
    api.launch_error = None
    await tick(store, api, now=1100)  # still pending: no second launch for slot 9
    assert [b["name"][-2:] for b in api.launches] == ["-9", "-9", "10"]
    await tick(store, api, now=1000 + ex.AMBIGUOUS_LAUNCH_SECONDS + 1)
    assert "slot-9" not in store.data  # never appeared: free to acquire again


@pytest.mark.anyio
async def test_secret_slots_and_unrecorded_instances_are_left_alone():
    store, api = Store({ex.TARGET_KEY: [9, 10]}), FakeAPI(catalog("us-west-3"))
    api.instances.append(
        {"id": "x", "name": "flash-h100-reservation-20261112-10", "status": "active"}
    )
    report = await tick(store, api, secret_slots=set(range(1, 10)))
    assert "review" in report[10]
    assert api.launches == []


@pytest.mark.anyio
async def test_terminated_published_worker_is_unrouted_and_reacquired():
    store, api = Store(), FakeAPI(catalog("us-west-3"))
    await tick(store, api)
    api.instances[0].update(status="active", ip="1.2.3.4")
    await tick(store, api, now=1200, secret_slots=set(range(1, 9)) | {10, 11, 12})
    assert len(store.data[ex.WORKERS_KEY]) == 1
    api.instances[0]["status"] = "terminated"
    await tick(store, api, now=1400, secret_slots=set(range(1, 9)) | {10, 11, 12})
    assert store.data[ex.WORKERS_KEY] == []
    await tick(store, api, now=1600, secret_slots=set(range(1, 9)) | {10, 11, 12})
    assert [b["name"][-2:] for b in api.launches] == ["-9", "-9"]


def test_merge_ignores_published_slots_held_by_the_secret():
    secret = [{"slot": 9, "instance_id": "s9", "ip": "8.8.8.8", "revision": REV, "tls": {}}]
    published = [{"slot": 9, "instance_id": "e9", "ip": "8.8.4.4", "revision": REV, "tls": {}}]
    assert [w["instance_id"] for w in ex.merge_workers(secret, published)] == ["s9"]
    assert json.dumps(ex.merge_workers(secret, None))


def test_merge_drops_malformed_published_entries():
    secret = [{"slot": 1, "instance_id": "s1", "ip": "8.8.8.8", "revision": REV, "tls": {}}]
    good = {"slot": 10, "instance_id": "e10", "ip": "8.8.4.4", "revision": REV, "tls": {}}
    bad = [{"slot": 13, "instance_id": "x"}, {"slot": 11}, "junk", dict(good, slot=12)]
    assert [w["slot"] for w in ex.merge_workers(secret, [good, *bad])] == [1, 10]
