from __future__ import annotations

import asyncio
import time
import types
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from flash.serving.src.engine.dispatch import admission_acknowledgement


def _passthrough_decorator(*_args: Any, **_kwargs: Any):
    def decorator(value: Any) -> Any:
        return value

    return decorator


@pytest.fixture(scope="module")
def modal_app_module(load_modal_app_under_stub):
    modal_stub = MagicMock(name="modal")
    for name in ("concurrent", "method", "enter", "exit", "asgi_app"):
        getattr(modal_stub, name).side_effect = _passthrough_decorator
    modal_stub.parameter.return_value = None
    app_mock = MagicMock(name="app")
    for name in ("cls", "function", "local_entrypoint"):
        getattr(app_mock, name).side_effect = _passthrough_decorator
    modal_stub.App.return_value = app_mock
    modal_stub.Period.return_value = MagicMock()
    return load_modal_app_under_stub(modal_stub)


class _Call:
    def __init__(
        self,
        result: Callable[[], Awaitable[Any]],
        *,
        cancel_wait: asyncio.Event | None = None,
    ) -> None:
        self.object_id = "fc-exact"
        self.cancel_count = 0
        self._result = result
        self._cancel_wait = cancel_wait
        self.get = types.SimpleNamespace(aio=self._get)
        self.cancel = types.SimpleNamespace(aio=self._cancel)

    async def _get(self) -> Any:
        return await self._result()

    async def _cancel(self) -> None:
        self.cancel_count += 1
        if self._cancel_wait is not None:
            await self._cancel_wait.wait()


class _Queue:
    def __init__(self, acknowledgement: Awaitable[Any]) -> None:
        self.get = types.SimpleNamespace(aio=lambda: acknowledgement)


def _ack() -> dict[str, Any]:
    return admission_acknowledgement(
        generation_id="generation-1",
        invocation_nonce="nonce-1",
        function_call_id="fc-exact",
    )


async def _await(modal_app_module: Any, call: _Call, queue: _Queue, deadline: float) -> Any:
    return await modal_app_module._await_modal_call(
        call,
        queue,
        deadline,
        generation_id="generation-1",
        invocation_nonce="nonce-1",
    )


async def _drain_cleanup(modal_app_module: Any) -> None:
    for _ in range(100):
        if not modal_app_module._MODAL_CLEANUP_TASKS:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("retained modal cleanup did not drain")


def test_admission_before_deadline_allows_completion_after_deadline(modal_app_module) -> None:
    async def scenario() -> tuple[dict[str, bool], int]:
        release = asyncio.Event()

        async def result() -> dict[str, bool]:
            await release.wait()
            return {"ok": True}

        call = _Call(result)
        task = asyncio.create_task(
            _await(
                modal_app_module, call, _Queue(asyncio.sleep(0, result=_ack())), time.time() + 0.01
            )
        )
        await asyncio.sleep(0.02)
        release.set()
        value = await task
        return value, call.cancel_count

    assert asyncio.run(scenario()) == ({"ok": True}, 0)


def test_no_admission_expires_and_cancels_exact_call_once(modal_app_module) -> None:
    async def scenario() -> int:
        call = _Call(asyncio.Event().wait)
        with pytest.raises(modal_app_module.PreHeaderDispatchExpired):
            await _await(
                modal_app_module,
                call,
                _Queue(asyncio.Event().wait()),
                time.time() + 0.01,
            )
        await _drain_cleanup(modal_app_module)
        return call.cancel_count

    assert asyncio.run(scenario()) == 1


def test_completion_before_admission_returns_normally(modal_app_module) -> None:
    async def scenario() -> tuple[dict[str, bool], int]:
        call = _Call(lambda: asyncio.sleep(0, result={"ok": True}))
        result = await _await(
            modal_app_module,
            call,
            _Queue(asyncio.Event().wait()),
            time.time() + 60,
        )
        await _drain_cleanup(modal_app_module)
        return result, call.cancel_count

    assert asyncio.run(scenario()) == ({"ok": True}, 0)


@pytest.mark.parametrize("admitted", [False, True])
def test_caller_cancellation_cancels_exact_call_once(modal_app_module, admitted: bool) -> None:
    async def scenario() -> int:
        call = _Call(asyncio.Event().wait)
        acknowledgement = asyncio.sleep(0, result=_ack()) if admitted else asyncio.Event().wait()
        task = asyncio.create_task(
            _await(modal_app_module, call, _Queue(acknowledgement), time.time() + 60)
        )
        await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _drain_cleanup(modal_app_module)
        return call.cancel_count

    assert asyncio.run(scenario()) == 1


@pytest.mark.parametrize(
    "acknowledgement",
    [
        {"kind": "admitted"},
        admission_acknowledgement(
            generation_id="generation-1",
            invocation_nonce="wrong",
            function_call_id="fc-exact",
        ),
        admission_acknowledgement(
            generation_id="generation-1",
            invocation_nonce="nonce-1",
            function_call_id="fc-other",
        ),
    ],
)
def test_malformed_or_mismatched_admission_fails_closed(
    modal_app_module, acknowledgement: dict[str, Any]
) -> None:
    async def scenario() -> int:
        call = _Call(asyncio.Event().wait)
        with pytest.raises(RuntimeError):
            await _await(
                modal_app_module,
                call,
                _Queue(asyncio.sleep(0, result=acknowledgement)),
                time.time() + 60,
            )
        await _drain_cleanup(modal_app_module)
        return call.cancel_count

    assert asyncio.run(scenario()) == 1


def test_late_admission_never_admits_expired_generation(modal_app_module) -> None:
    async def scenario() -> int:
        async def late_ack() -> dict[str, Any]:
            await asyncio.sleep(0.02)
            return _ack()

        call = _Call(asyncio.Event().wait)
        with pytest.raises(modal_app_module.PreHeaderDispatchExpired):
            await _await(modal_app_module, call, _Queue(late_ack()), time.time() + 0.005)
        await _drain_cleanup(modal_app_module)
        return call.cancel_count

    assert asyncio.run(scenario()) == 1


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_late_spawn_handle_is_cancelled_once_and_retention_clears(
    modal_app_module, cancel_caller: bool
) -> None:
    async def scenario() -> tuple[int, float]:
        release = asyncio.Event()
        call = _Call(lambda: asyncio.sleep(0, result={"ok": True}))

        async def spawn() -> _Call:
            await release.wait()
            return call

        method = types.SimpleNamespace(spawn=types.SimpleNamespace(aio=spawn))
        started = time.monotonic()
        task = asyncio.create_task(modal_app_module._spawn_modal_call(method, time.time() + 0.005))
        if cancel_caller:
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(modal_app_module.PreHeaderDispatchExpired):
                await task
        elapsed = time.monotonic() - started
        assert call.cancel_count == 0
        release.set()
        await _drain_cleanup(modal_app_module)
        return call.cancel_count, elapsed

    cancel_count, elapsed = asyncio.run(scenario())
    assert cancel_count == 1
    assert elapsed < 0.05


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_blocked_call_cancellation_does_not_delay_primary_outcome(
    modal_app_module, cancel_caller: bool
) -> None:
    async def scenario() -> tuple[int, float]:
        release_cancel = asyncio.Event()
        call = _Call(asyncio.Event().wait, cancel_wait=release_cancel)
        started = time.monotonic()
        task = asyncio.create_task(
            _await(
                modal_app_module,
                call,
                _Queue(asyncio.Event().wait()),
                time.time() + 0.005,
            )
        )
        if cancel_caller:
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(modal_app_module.PreHeaderDispatchExpired):
                await task
        elapsed = time.monotonic() - started
        assert call.cancel_count == 1
        assert modal_app_module._MODAL_CLEANUP_TASKS
        release_cancel.set()
        await _drain_cleanup(modal_app_module)
        return call.cancel_count, elapsed

    cancel_count, elapsed = asyncio.run(scenario())
    assert cancel_count == 1
    assert elapsed < 0.05
