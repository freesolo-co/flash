"""Releasing the reserved 9B Base fleet after reserved serving is switched off."""

import pytest

from flash.serving.app.timeouts import ROUTER_TIMEOUT_SECONDS
from flash.serving.lambda_cloud import reserved
from flash.serving.lambda_cloud import reserved_expansion as ex
from flash.serving.lambda_cloud import reserved_retire as rr
from flash.serving.lambda_cloud.reserved_watchdog import CONFIGURED_KEY

REV = "c" * 40
T0 = 1_000_000.0
CAMPAIGN_TAG = [{"key": "reservation-campaign", "value": reserved.RESERVATION_SECRET_NAME}]


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Store(dict):
    async def get(self, key):
        return super().get(key)

    async def put(self, key, value):
        self[key] = value

    async def put_if_absent(self, key, value):
        if key in self:
            return False
        self[key] = value
        return True

    async def pop(self, key):
        return super().pop(key, None)


class API:
    def __init__(self, instances):
        self.instances = instances
        self.calls = []

    async def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        if (method, path) == ("GET", "instances"):
            return self.instances
        if (method, path) == ("POST", "instance-operations/terminate"):
            for x in self.instances:
                if x["id"] in body["instance_ids"]:
                    x["status"] = "terminating"
            return {}
        raise AssertionError((method, path))

    def terminated(self):
        return [b["instance_ids"] for m, p, b in self.calls if p == "instance-operations/terminate"]


def instance(id_, name, ip, tags=CAMPAIGN_TAG, status="active"):
    return {"id": id_, "name": name, "ip": ip, "status": status, "tags": tags}


def fleet():
    """Scheduler slot 1, secret slot 5, expansion slot 9, an unrecorded slot 10, a lease worker."""
    return [
        instance("i1", ex.instance_name(1), "10.0.0.1"),
        instance("i5", "operator-named-5", "10.0.0.5", tags=[]),
        instance("i9", ex.instance_name(9), "10.0.0.9"),
        instance("i10", ex.instance_name(10), "10.0.0.10"),
        instance("lease", "flash-main-" + "a" * 32, "10.0.0.99", tags=[]),
    ]


SECRET = [
    {"slot": 1, "instance_id": "i1", "ip": "10.0.0.1"},
    {"slot": 5, "instance_id": "i5", "ip": "10.0.0.5"},
]


def stores():
    health = Store({1: {"healthy": True}, 9: {"healthy": True}, CONFIGURED_KEY: [1, 5, 9]})
    expansion = Store(
        {
            ex.TARGET_KEY: [9, 10],
            ex.WORKERS_KEY: [{"slot": 9, "instance_id": "i9", "ip": "10.0.0.9"}],
            ex.slot_key(9): {"instance_id": "i9", "ip": "10.0.0.9", "state": "serving"},
        }
    )
    ledger = Store({"state": {"mode": "serving", "hold_until": T0 + 30 * 86400}})
    return health, expansion, ledger


async def cycle(health, expansion, ledger, api, at, secret=SECRET, revision=REV):
    return await rr.run_once(
        health, expansion, ledger, api, secret, revision=revision, clock=lambda: at
    )


def test_grace_outlasts_requests_routed_by_the_previous_release():
    assert rr.GRACE_SECONDS >= ROUTER_TIMEOUT_SECONDS + 30 * 60


@pytest.mark.anyio
async def test_nothing_is_released_while_routed_requests_can_still_run():
    health, expansion, ledger = stores()
    api = API(fleet())
    before = (dict(expansion), dict(ledger))
    first = await cycle(health, expansion, ledger, api, T0)
    later = await cycle(health, expansion, ledger, api, T0 + rr.GRACE_SECONDS - 1)
    assert first["phase"] == later["phase"] == "draining"
    assert health[rr.RETIRE_KEY] == {"revision": REV, "started_at": T0}
    assert api.calls == []
    assert (dict(expansion), dict(ledger)) == before


@pytest.mark.anyio
async def test_each_deployment_restarts_the_drain():
    health, expansion, ledger = stores()
    api = API(fleet())
    await cycle(health, expansion, ledger, api, T0)
    report = await cycle(health, expansion, ledger, api, T0 + rr.GRACE_SECONDS, revision="d" * 40)
    assert report["phase"] == "draining"
    assert api.calls == []


@pytest.mark.anyio
async def test_release_stops_acquisition_expires_the_hold_and_terminates_the_rest():
    health, expansion, ledger = stores()
    api = API(fleet())
    await cycle(health, expansion, ledger, api, T0)
    now = T0 + rr.GRACE_SECONDS
    report = await cycle(health, expansion, ledger, api, now)

    assert expansion[ex.TARGET_KEY] == []
    assert expansion[ex.WORKERS_KEY] == []
    assert ledger["state"]["hold_until"] == now
    assert "tick_lock" not in ledger
    assert health[rr.RETIRE_KEY]["hold_released_at"] == now
    # One id per request; the scheduler terminates its own instance; lease workers untouched.
    assert api.terminated() == [["i5"], ["i9"], ["i10"]]
    assert report["waiting_for_scheduler"] == ["i1"]
    assert report["phase"] == "releasing"


@pytest.mark.anyio
async def test_scheduler_instances_are_terminated_here_if_the_scheduler_does_not():
    health, expansion, ledger = stores()
    api = API(fleet())
    await cycle(health, expansion, ledger, api, T0)
    released = T0 + rr.GRACE_SECONDS
    await cycle(health, expansion, ledger, api, released)
    await cycle(health, expansion, ledger, api, released + rr.SCHEDULER_FALLBACK_SECONDS - 1)
    assert ["i1"] not in api.terminated()
    await cycle(health, expansion, ledger, api, released + rr.SCHEDULER_FALLBACK_SECONDS)
    assert api.terminated()[-1] == ["i1"]


@pytest.mark.anyio
async def test_busy_tick_lock_defers_only_the_scheduler_hand_back():
    health, expansion, ledger = stores()
    now = T0 + rr.GRACE_SECONDS
    ledger["tick_lock"] = {"token": "scheduler", "at": now - 60}  # a live tick, not stale
    api = API(fleet())
    await cycle(health, expansion, ledger, api, T0)
    report = await cycle(health, expansion, ledger, api, now)
    assert ledger["state"]["hold_until"] == T0 + 30 * 86400
    assert "hold_released_at" not in health[rr.RETIRE_KEY]
    assert api.terminated() == [["i5"], ["i9"], ["i10"]]
    assert report["waiting_for_scheduler"] == ["i1"]


@pytest.mark.anyio
async def test_an_address_mismatch_is_never_terminated():
    health, expansion, ledger = stores()
    api = API(fleet())
    secret = [dict(SECRET[1], ip="10.9.9.9")]
    await cycle(health, expansion, ledger, api, T0, secret)
    report = await cycle(health, expansion, ledger, api, T0 + rr.GRACE_SECONDS, secret)
    assert ["i5"] not in api.terminated()
    assert "i5" in report["errors"]


@pytest.mark.anyio
async def test_released_once_nothing_is_left_then_slot_records_are_cleared():
    health, expansion, ledger = stores()
    api = API(fleet())
    await cycle(health, expansion, ledger, api, T0)
    now = T0 + rr.GRACE_SECONDS
    await cycle(health, expansion, ledger, api, now)
    for x in api.instances:  # the scheduler's tick and Lambda finish the termination
        if x["id"] != "lease":
            x["status"] = "terminated"
    report = await cycle(health, expansion, ledger, api, now + 600)
    assert report == {"phase": "released", "released_at": now + 600}
    assert set(health) == {rr.RETIRE_KEY}
    assert ex.slot_key(9) not in expansion
    assert expansion[ex.TARGET_KEY] == []
    calls = len(api.calls)
    assert (await cycle(health, expansion, ledger, api, now + 900))["phase"] == "released"
    assert api.terminated() == [["i5"], ["i9"], ["i10"]]
    assert len(api.calls) == calls + 1


@pytest.mark.parametrize(
    ("state", "changed"),
    [
        (None, False),
        ({"mode": "closed", "hold_until": T0 + 99}, False),
        ({"mode": "adopted", "hold_until": T0 + 99}, False),
        ({"mode": "serving", "hold_until": T0 - 1}, False),
        ({"mode": "serving", "hold_until": T0 + 99}, True),
        ({"mode": "reserve_only", "hold_until": T0 + 99}, True),
    ],
)
def test_release_hold_only_shortens_an_active_hold(state, changed):
    released, event = rr.release_hold(state, T0)
    assert (released is not state) is changed
    assert bool(event) is changed
    if changed:
        assert released["hold_until"] == T0
        assert released["hold_released_at"] == T0
        assert released["mode"] == state["mode"]
