"""Pool thresholds, telemetry failures, and complete streaming request lifetimes."""

import asyncio
from types import SimpleNamespace

import pytest

from flash.serving.deepinfer.app.autoscaling import pool_load, reconcile, required_containers
from flash.serving.deepinfer.app.http import _GenerationCount


class Store(dict):
    def put(self, key, value):
        self[key] = value


class Server:
    def __init__(self, floor=1, containers=("a",)):
        self.settings = SimpleNamespace(min_containers=floor, max_containers=3)
        self.containers = containers
        self.updates = []
        self.autoscaler = SimpleNamespace(settings=lambda: self.settings, update=self.update)

    def update(self, **kwargs):
        assert set(kwargs) == {"min_containers"}
        self.updates.append(kwargs["min_containers"])
        self.settings.min_containers = kwargs["min_containers"]

    def _experimental_list_containers(self):
        return [SimpleNamespace(container_id=key) for key in self.containers]


@pytest.mark.parametrize(
    ("active", "expected"),
    [(0, 1), (35, 1), (36, 2), (72, 2), (100, 2), (135, 2), (136, 3), (500, 3)],
)
def test_pool_reserve_boundaries(active, expected):
    assert required_containers(active) == expected


def test_pool_counts_both_replicas_not_each_threshold_separately():
    server = Server(floor=2, containers=("a", "b"))
    store = Store(a={"at": 1000, "active": 68}, b={"at": 1000, "active": 68})
    assert reconcile(server, store, now=1000) == {"active": 136, "complete": True, "floor": 3}
    assert server.updates == [3]


def test_no_oscillation_after_second_container_starts_and_delayed_scale_down():
    server = Server()
    store = Store(a={"at": 1000, "active": 36})
    reconcile(server, store, now=1000)
    server.containers = ("a", "b")
    for now in range(1005, 1705, 5):
        store.update(a={"at": now, "active": 18}, b={"at": now, "active": 18})
        reconcile(server, store, now=now)
    assert server.updates == [2]
    for now in range(1705, 2305, 5):
        store.update(a={"at": now, "active": 0}, b={"at": now, "active": 0})
        reconcile(server, store, now=now)
    assert server.updates == [2]
    store.update(a={"at": 2305, "active": 0}, b={"at": 2305, "active": 0})
    reconcile(server, store, now=2305)
    assert server.updates == [2, 1]


@pytest.mark.parametrize(
    "sample",
    [
        None,
        {"at": 979, "active": 0},
        {"at": 1001, "active": 0},
        {"at": float("nan"), "active": 0},
        {"at": 1000, "active": -1},
        {"at": 1000, "active": True},
    ],
)
def test_bad_telemetry_is_not_zero_load(sample):
    active, complete = pool_load({"a", "b"}, {"a": {"at": 1000, "active": 136}, "b": sample}, 1000)
    assert active == 136
    assert not complete


def test_missing_replica_blocks_scale_down_but_known_load_can_scale_up():
    server = Server(floor=2, containers=("a", "b"))
    store = Store(
        a={"at": 1000, "active": 0}, controller={"at": 995, "floor": 2, "below_since": 390}
    )
    reconcile(server, store, now=1000)
    assert server.updates == []
    assert store["controller"]["below_since"] is None
    store["a"] = {"at": 1005, "active": 136}
    reconcile(server, store, now=1005)
    assert server.updates == [3]


def test_controller_outage_resets_quiet_window_and_departed_samples_do_not_count():
    server = Server(floor=3)
    store = Store(
        a={"at": 1000, "active": 0},
        dead={"at": 900, "active": 1000},
        controller={"at": 900, "floor": 3, "below_since": 300},
    )
    reconcile(server, store, now=1000)
    assert server.updates == []
    assert "dead" not in store
    assert store["controller"]["below_since"] == 1000


def test_disabled_pool_is_not_relaunched():
    server = Server(floor=0)
    server.settings.max_containers = 0
    assert reconcile(server, Store(), now=1000) is None
    assert not server.updates


def test_failed_update_does_not_record_success():
    server = Server()

    def fail(**kwargs):
        raise RuntimeError("control plane unavailable")

    server.autoscaler.update = fail
    store = Store(a={"at": 1000, "active": 36})
    with pytest.raises(RuntimeError):
        reconcile(server, store, now=1000)
    assert "controller" not in store


@pytest.mark.parametrize("path", ["/internal/generate", "/health"])
@pytest.mark.parametrize("finish", ["normal", "error", "cancel"])
def test_counter_covers_preparation_and_stream_lifetime(path, finish):
    async def run():
        counts = {"active": 0}
        started, release = asyncio.Event(), asyncio.Event()

        async def app(scope, receive, send):
            assert counts["active"] == (1 if path == "/internal/generate" else 0)
            started.set()
            await release.wait()
            if finish == "error":
                raise RuntimeError("stream failed")

        middleware = _GenerationCount(app, counts)
        task = asyncio.create_task(
            middleware({"type": "http", "method": "POST", "path": path}, None, None)
        )
        await started.wait()
        assert counts["active"] == (1 if path == "/internal/generate" else 0)
        if finish == "cancel":
            task.cancel()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert counts["active"] == 0

    asyncio.run(run())
