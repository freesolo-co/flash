"""Modal client ownership across synchronous and ASGI callers."""

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest

from flash.providers.modal import api
from flash.providers.modal.api import call


@pytest.mark.parametrize("inside_loop", [False, True])
@pytest.mark.parametrize("outcome", ["success", "error", "timeout"])
def test_call_keeps_client_on_its_loop_and_closes(monkeypatch, inside_loop, outcome):
    events = []
    caller_thread = threading.get_ident()

    def record(name):
        events.append((name, asyncio.get_running_loop(), threading.get_ident()))

    class Client:
        async def __aexit__(self, *_):
            record("close")

    async def connect(*_):
        record("connect")
        return Client()

    async def operation(_modal, _client):
        record("operation")
        if outcome == "error":
            raise ValueError("sensitive transport detail")
        if outcome == "timeout":
            await asyncio.sleep(10)
        return 42

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(Client=SimpleNamespace(from_credentials=SimpleNamespace(aio=connect))),
    )
    monkeypatch.setattr(api, "credentials", lambda: ("test-id", "test-secret"))

    def run():
        return call(operation, timeout=0.1 if outcome == "timeout" else 5)

    async def async_caller():
        return run()

    def invoke():
        return asyncio.run(async_caller()) if inside_loop else run()

    if outcome == "success":
        assert invoke() == 42
    else:
        with pytest.raises(api.ModalApiError, match=r"^Modal operation failed or timed out$"):
            invoke()
    assert [event[0] for event in events] == ["connect", "operation", "close"]
    assert len({event[1] for event in events}) == 1
    assert len({event[2] for event in events}) == 1
    assert (events[0][2] != caller_thread) == inside_loop
