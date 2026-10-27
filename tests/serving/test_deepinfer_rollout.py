"""Readiness uses the real HTTP listener, including startup failure and warmup."""

import asyncio
import socket
import sys
from types import SimpleNamespace

import pytest
import uvicorn

from flash.serving.deepinfer.app.http import build_app
from flash.serving.deepinfer.app.rollout import wait_for_rollout


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("fail", [False, True])
async def test_listener_stays_closed_until_runtime_start_and_warmup_finish(fail):
    starting = asyncio.Event()
    warmed = asyncio.Event()
    closed = asyncio.Event()

    async def start():
        starting.set()
        await warmed.wait()
        if fail:
            raise RuntimeError("warmup failed")

    async def close():
        closed.set()

    runtime = SimpleNamespace(start=start, close=close, assert_healthy=lambda: None)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(build_app(runtime), host="127.0.0.1", port=port, log_level="critical")
    )

    async def serve():
        if fail:
            with pytest.raises(SystemExit) as error:
                await server.serve()
            assert error.value.code == 3
        else:
            await server.serve()

    task = asyncio.create_task(serve())
    try:
        await asyncio.wait_for(starting.wait(), 5)
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(("127.0.0.1", port), timeout=0.1)
        warmed.set()
        if fail:
            await asyncio.wait_for(task, 5)
            assert not server.started
            with pytest.raises(ConnectionRefusedError):
                socket.create_connection(("127.0.0.1", port), timeout=0.1)
        else:
            async with asyncio.timeout(5):
                while not server.started:  # noqa: ASYNC110 -- Uvicorn exposes a flag, not an event.
                    await asyncio.sleep(0.01)
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), 5)
            writer.close()
            await writer.wait_closed()
            assert b"200 OK" in response
    finally:
        warmed.set()
        server.should_exit = True
        await asyncio.wait_for(task, 5)
    assert closed.is_set()


def test_verification_cannot_succeed_on_old_version_or_unrestored_cap(monkeypatch):
    from flash.serving.deepinfer.app import rollout

    samples = {key: {"at": 1000, "active": 0, "deployment_id": "old"} for key in ("a", "b")}
    settings = SimpleNamespace(max_containers=5)
    server = SimpleNamespace(
        _experimental_list_containers=lambda: [
            SimpleNamespace(container_id=key) for key in samples
        ],
        autoscaler=SimpleNamespace(settings=lambda: settings),
    )
    waits = []

    def advance(_):
        waits.append(True)
        if len(waits) == 1:
            for sample in samples.values():
                sample["deployment_id"] = "new"
        else:
            settings.max_containers = 4

    monkeypatch.setattr(rollout.time, "sleep", advance)
    monkeypatch.setattr(rollout.time, "time", lambda: 1000)
    wait_for_rollout(server, samples, "new", timeout=5)
    assert len(waits) == 2


def test_verification_timeout_does_not_stop_or_change_the_pool():
    with pytest.raises(TimeoutError, match="readiness was not confirmed"):
        wait_for_rollout(object(), {}, "new", timeout=0)


def test_begin_bootstraps_rollout_state_before_the_first_app_deployment(monkeypatch):
    from flash.serving.deepinfer.app import rollout

    dictionaries = {}

    def lookup(name, *, create_if_missing=False):
        if create_if_missing:
            dictionaries.setdefault(name, {})
        return SimpleNamespace(put=lambda key, value: dictionaries[name].__setitem__(key, value))

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            config=SimpleNamespace(config={"environment": "main"}),
            Dict=SimpleNamespace(from_name=lookup),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["rollout", "--begin"])
    monkeypatch.setenv("DEEPINFER_ROLLOUT_ID", "first-deployment")
    rollout.main()
    assert dictionaries[rollout.POOL_STATE_NAME]["rollout"] == {"deployment_id": "first-deployment"}
