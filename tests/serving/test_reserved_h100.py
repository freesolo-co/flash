import json
from unittest.mock import AsyncMock

import pytest

from flash.serving.lambda_cloud import reserved
from flash.serving.lambda_cloud.config import NoCapacityError
from flash.serving.lambda_cloud.fleet import Fleet


@pytest.fixture
def anyio_backend():
    return "asyncio"


def workers():
    return [
        {"slot": i, "instance_id": str(i), "ip": f"8.8.8.{i}", "revision": "a" * 40, "tls": {}}
        for i in range(1, 5)
    ]


def test_worker_config_accepts_any_subset_of_declared_slots():
    data = [dict(w, slot=w["slot"] + 4, instance_id=f"x{w['slot']}") for w in workers()]
    raw = json.dumps(workers() + data)
    assert reserved.SLOT_COUNT == 15
    assert [w["slot"] for w in reserved.parse_workers(raw)] == list(range(1, 9))
    assert reserved.worker_config(8, raw)["instance_id"] == "x4"
    # Four configured slots: the declared slots beyond them are not configured.
    with pytest.raises(ValueError, match="not configured"):
        reserved.worker_config(5, json.dumps(workers()))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__(1, dict(d[1], slot=1)),
        lambda d: d.__setitem__(0, dict(d[0], slot=reserved.SLOT_COUNT + 1)),
        lambda d: d.__setitem__(0, dict(d[0], slot=0)),
        lambda d: d.clear(),
    ],
)
def test_reject_invalid_reserved_slots(mutate):
    data = workers()
    mutate(data)
    with pytest.raises(ValueError, match="slot"):
        reserved.parse_workers(json.dumps(data))


def test_reject_duplicate_reserved_instance():
    data = workers()
    data[1]["instance_id"] = data[0]["instance_id"]
    with pytest.raises(ValueError, match="distinct"):
        reserved.worker_config(1, json.dumps(data))


@pytest.mark.anyio
async def test_reserved_shutdown_does_not_release_gpu():
    proxy = reserved.ReservedLambdaEngineProxy()
    proxy._transport = AsyncMock()
    proxy._fleet = AsyncMock()
    await proxy._shutdown()
    proxy._transport.aclose.assert_awaited_once()
    proxy._fleet.release.remote.aio.assert_not_awaited()


@pytest.mark.anyio
async def test_wrong_worker_identity_fails_closed(monkeypatch):
    monkeypatch.setenv("FLASH_RESERVED_H100_WORKERS", json.dumps(workers()))
    transport = AsyncMock()
    transport.call.return_value = {
        "ok": True,
        "base_model": reserved.MODEL,
        "deployment_sha": "b" * 40,
        "configured_gpu": "H100",
    }
    monkeypatch.setattr(reserved, "WorkerTransport", lambda *a: transport)
    proxy = reserved.ReservedLambdaEngineProxy()
    proxy.base_model = reserved.MODEL
    proxy.reserved_slot = 1
    with pytest.raises(RuntimeError, match="identity"):
        await proxy._load()
    transport.aclose.assert_awaited_once()


def test_reserved_serving_is_off_by_default():
    assert reserved.ENABLED is False
    assert not reserved.enabled(reserved.MODEL, "main")
    assert not reserved.enabled(reserved.MODEL, "dev")


@pytest.mark.anyio
async def test_dynamic_fleet_cannot_allocate_more_9b_base_gpus_while_reserved(monkeypatch):
    monkeypatch.setattr(reserved, "ENABLED", True)
    api = AsyncMock()
    ledger = AsyncMock()
    fleet = Fleet(api, ledger, "main")
    with pytest.raises(NoCapacityError, match="reserved"):
        await fleet._check_capacity(reserved.MODEL, "a" * 40)
    api.call.assert_not_awaited()


@pytest.mark.anyio
async def test_shared_reconciler_skips_reserved_model(monkeypatch):
    from types import SimpleNamespace

    from flash.serving.lambda_cloud.pool_capacity import SharedCapacity

    monkeypatch.setattr(
        "flash.serving.lambda_cloud.pool_capacity.lambda_models", lambda: [reserved.MODEL]
    )
    state = SimpleNamespace(get=AsyncMock(return_value={"revision": "a" * 40, "retired": []}))
    control = SharedCapacity(AsyncMock(), AsyncMock(), state, "a" * 40)
    control.reconcile_model = AsyncMock()
    await control.reconcile()
    control.reconcile_model.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("environment", ["main", "dev"])
async def test_9b_base_allocates_lambda_only_on_demand_without_reserved_fleet(environment):
    from types import SimpleNamespace

    async def empty_items():
        if False:
            yield None

    api = SimpleNamespace(call=AsyncMock(return_value=[]))
    ledger = SimpleNamespace(items=empty_items)
    fleet = Fleet(api, ledger, environment)
    fleet.shared_capacity = AsyncMock()
    await fleet._check_capacity(reserved.MODEL, "a" * 40)
    # No shared Modal GPU pool: 9B Base has no Modal fallback in either environment.
    fleet.shared_capacity.before_launch.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("old_count", "current_count", "allowed"), [(4, 0, True), (4, 1, False), (0, 4, False)]
)
@pytest.mark.parametrize("ownership", ["leased", "draining", "uncertain"])
async def test_dev_lambda_only_budget_counts_all_revisions(
    old_count, current_count, allowed, ownership
):
    from types import SimpleNamespace

    now = 10000
    records = []
    instances = []
    for i in range(old_count + current_count):
        owner = f"{i:032x}"
        records.append(
            (
                owner,
                {
                    "model": reserved.MODEL,
                    "revision": "old" if i < old_count else "new",
                    "instance_id": None if ownership == "uncertain" else str(i),
                    "released": ownership == "draining",
                    "heartbeat": now,
                },
            )
        )
        if ownership == "draining":
            instances.append(
                {
                    "name": f"flash-dev-{owner}",
                    "status": "terminating",
                    "tags": [
                        {"key": "flash-owner", "value": owner},
                        {"key": "flash-environment", "value": "dev"},
                    ],
                }
            )

    async def items():
        for item in records:
            yield item

    fleet = Fleet(
        SimpleNamespace(call=AsyncMock(return_value=instances)),
        SimpleNamespace(items=items),
        "dev",
        clock=lambda: now,
    )
    fleet.shared_capacity = AsyncMock()
    if allowed:
        await fleet._check_capacity(reserved.MODEL, "new")
    else:
        with pytest.raises(NoCapacityError, match="replica limit"):
            await fleet._check_capacity(reserved.MODEL, "new")
    fleet.shared_capacity.before_launch.assert_not_awaited()


@pytest.mark.anyio
async def test_expansion_proxy_follows_a_replaced_worker(monkeypatch):
    from flash.serving.lambda_cloud import reserved_expansion

    monkeypatch.setenv("FLASH_RESERVED_H100_WORKERS", json.dumps(workers()))
    tls = {"cert": "C", "token": "T"}
    published = [
        {"slot": 9, "instance_id": "e9", "ip": "8.8.4.4", "revision": "a" * 40, "tls": tls}
    ]
    monkeypatch.setattr(reserved_expansion, "published_workers", AsyncMock(return_value=published))
    health = {"ok": True, "base_model": reserved.MODEL, "deployment_sha": "a" * 40}
    made = []

    def factory(ip, tls):
        made.append(AsyncMock(ip=ip))
        made[-1].call.return_value = {**health, "configured_gpu": "H100"}
        return made[-1]

    monkeypatch.setattr(reserved, "WorkerTransport", factory)
    clock = [0.0]
    monkeypatch.setattr(reserved.time, "monotonic", lambda: clock[0])
    proxy = reserved.ReservedLambdaEngineProxy()
    proxy.base_model, proxy.reserved_slot = reserved.MODEL, 9
    await proxy._load()

    published[0] = dict(published[0], instance_id="e9b", ip="9.9.9.9")
    clock[0] += reserved.REBIND_CHECK_SECONDS + 1
    await proxy._generate({"x": 1})
    assert (proxy._transport.ip, proxy._owner) == ("9.9.9.9", "e9b")
    assert proxy.engine is proxy._transport
    made[0].aclose.assert_awaited_once()


@pytest.mark.anyio
async def test_closing_reserved_stream_awaits_transport_cleanup(monkeypatch):
    import asyncio

    closed = asyncio.Event()

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return {"event": "token"}

        async def aclose(self):
            await asyncio.sleep(0)
            closed.set()

    proxy = reserved.ReservedLambdaEngineProxy()
    proxy._expansion = False
    proxy._transport = AsyncMock()
    proxy._transport.stream = lambda body: Stream()
    stream = proxy._stream_generate({"x": 1})
    assert await anext(stream) == {"event": "token"}
    await stream.aclose()
    assert closed.is_set()
