"""Fleet transitions preserve accepted streams and the shared GPU budget."""

from copy import deepcopy

import pytest

from flash.serving.deepinfer.app.fleet import controller_lock, reconcile_fleet


class Store(dict):
    def get(self, key, default=None):
        return deepcopy(super().get(key, default))

    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = deepcopy(value)
        return True


def entry(name, version="old", phase="active"):
    return {
        "name": name,
        "version": version,
        "phase": phase,
        "image_id": "im-test",
        "url": f"https://{name}.modal.direct",
        "slots": 1,
    }


class Ops:
    def __init__(self, store):
        self.store = store
        self.loads = {}
        self.stops = []
        self.gone = set()
        self.drains = []
        self.fail_launch = False

    def launch(self, item):
        assert item["name"] in {r["name"] for r in self.store["fleet"]["replicas"]}
        if self.fail_launch:
            raise RuntimeError("provisioning interrupted")
        return f"https://{item['name']}.modal.direct"

    def health(self, item):
        count = self.loads.get(item["name"])
        if count is None:
            return None
        return {"active_http_requests": count, "draining": item["name"] in self.drains}

    def drain(self, item):
        assert item["url"] not in self.store["routes"]["endpoints"]
        self.drains.append(item["name"])
        return self.health(item)

    def stop(self, item):
        assert item["name"] in self.drains
        assert self.loads[item["name"]] == 0
        self.stops.append(item["name"])

    def stopped(self, item):
        return item["name"] in self.gone


def setup(*entries, desired="new"):
    store = Store(
        fleet={"replicas": list(entries), "quiet_since": None},
        desired={"version": desired, "image_id": "im-new"},
    )
    return store, Ops(store)


def test_redeployment_never_retires_a_stream_even_after_one_hour():
    store, ops = setup(entry("old"), entry("new", "new"))
    store["fleet"]["replicas"][1]["replaces"] = "old"
    ops.loads = {"old": 1, "new": 0}
    for now in (0, 3601, 7199):
        reconcile_fleet(store, ops, now=now)
        assert ops.stops == []
        assert store["routes"]["endpoints"] == ["https://new.modal.direct"]
    ops.loads["old"] = 0
    reconcile_fleet(store, ops, now=7201)
    assert ops.stops == ["old"]
    assert len(store["fleet"]["replicas"]) == 2  # Shutdown still consumes a slot.
    ops.gone.add("old")
    reconcile_fleet(store, ops, now=7206)
    assert [r["name"] for r in store["fleet"]["replicas"]] == ["new"]


def test_failed_warmup_keeps_previous_version_serving():
    store, ops = setup(entry("old"), entry("new", "new", "warming"))
    ops.loads["old"] = 10
    reconcile_fleet(store, ops, now=5000)
    assert store["routes"]["endpoints"] == ["https://old.modal.direct"]
    assert ops.drains == []
    assert ops.stops == []


def test_missing_drain_telemetry_is_never_permission_to_stop():
    store, ops = setup(entry("old", phase="draining"), entry("new", "new"))
    ops.loads["new"] = 0
    for now in (0, 10000, 100000):
        reconcile_fleet(store, ops, now=now)
    assert ops.stops == []
    assert len(store["fleet"]["replicas"]) == 2


def test_budget_counts_warming_and_draining_replicas_across_versions():
    store, ops = setup(*(entry(f"old{i}") for i in range(4)))
    ops.loads = {f"old{i}": 120 for i in range(4)}
    reconcile_fleet(store, ops, now=0)
    assert len(store["fleet"]["replicas"]) == 5
    new = store["fleet"]["replicas"][-1]["name"]
    ops.loads[new] = 120
    for now in range(5, 50, 5):
        reconcile_fleet(store, ops, now=now)
        assert len(store["fleet"]["replicas"]) == 5
    assert ops.stops == []
    assert len(store["routes"]["endpoints"]) == 4
    assert f"https://{new}.modal.direct" in store["routes"]["endpoints"]
    assert len(set(ops.drains)) == 1


def test_normal_autoscaling_cannot_allocate_a_fifth_replica():
    store, ops = setup(*(entry(f"new{i}", "new") for i in range(4)))
    ops.loads = {f"new{i}": 128 for i in range(4)}
    for now in range(5):
        reconcile_fleet(store, ops, now=now)
    assert len(store["fleet"]["replicas"]) == 4


def test_interrupted_provisioning_preserves_the_reserved_slot():
    store, ops = setup()
    ops.fail_launch = True
    with pytest.raises(RuntimeError, match="provisioning interrupted"):
        reconcile_fleet(store, ops, now=0)
    name = store["fleet"]["replicas"][0]["name"]
    ops.fail_launch = False
    reconcile_fleet(store, ops, now=5)
    assert [r["name"] for r in store["fleet"]["replicas"]] == [name]


def test_scale_down_waits_for_quiet_period_and_then_drains_busy_replica():
    store, ops = setup(entry("one", "new"), entry("two", "new"))
    ops.loads = {"one": 1, "two": 2}
    for now in range(0, 600, 5):
        reconcile_fleet(store, ops, now=now)
    reconcile_fleet(store, ops, now=599)
    assert ops.drains == []
    reconcile_fleet(store, ops, now=600)
    reconcile_fleet(store, ops, now=601)
    assert ops.drains == ["one"]
    assert ops.stops == []
    ops.loads["one"] = 0
    reconcile_fleet(store, ops, now=602)
    assert ops.stops == ["one"]


def test_controllers_cannot_overlap_during_their_own_redeployment():
    store = Store()
    with controller_lock(store) as first:
        assert first
        with controller_lock(store) as second:
            assert not second
        assert "controller_lock" in store
    assert "controller_lock" not in store

    def failed_tick():
        with controller_lock(store):
            raise RuntimeError("tick failed")

    with pytest.raises(RuntimeError):
        failed_tick()
    assert "controller_lock" not in store


def test_missing_capacity_ledger_never_means_an_empty_fleet():
    store = Store(desired={"version": "new", "image_id": "im-new"})
    ops = Ops(store)
    with pytest.raises(RuntimeError, match="Missing fleet capacity ledger"):
        reconcile_fleet(store, ops, now=0)
    assert "fleet" not in store


def test_four_replica_rollout_replaces_one_at_a_time_without_capacity_collapse():
    store, ops = setup(*(entry(f"old{i}") for i in range(4)))
    ops.loads = {f"old{i}": 80 for i in range(4)}
    for now in range(20):
        reconcile_fleet(store, ops, now=now)
        replicas = store["fleet"]["replicas"]
        assert len(replicas) <= 5
        assert len(store["routes"]["endpoints"]) >= 4
        for item in replicas:
            if item["phase"] == "warming":
                ops.loads[item["name"]] = 80
            elif item["phase"] == "draining":
                ops.loads[item["name"]] = 0
            elif item["phase"] == "stopping":
                ops.gone.add(item["name"])
    assert len(store["fleet"]["replicas"]) == 4
    assert all(item["version"] == "new" for item in store["fleet"]["replicas"])
