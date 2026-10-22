"""deadline and timeout classification coverage for the modal sdk bridge."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from flash.serve.provisioning.common.records import InterruptedProvisioning
from flash.serve.provisioning.modal.execution.sdk import (
    ModalSdkFailure,
    _call_mutation,
    _call_read,
    _sync_value,
)

ROOT = Path(__file__).resolve().parents[1]


async def _never_resolves() -> None:
    await asyncio.Event().wait()


async def _fast_value() -> str:
    return "fast-modal-result"


def test_lazy_loader_binds_pinned_experimental_surface_without_provider_calls() -> None:
    program = r"""
import importlib
import importlib.metadata
import socket
import sys

import modal

provider_calls = []
network_calls = []


def forbidden_provider_call(*args, **kwargs):
    provider_calls.append((args, kwargs))
    raise AssertionError("provider operation attempted")


def forbidden_network_call(*args, **kwargs):
    network_calls.append((args, kwargs))
    raise AssertionError("network operation attempted")


assert importlib.metadata.version("modal") == sys.argv[1]
modal.Client.from_credentials = forbidden_provider_call
modal.Client.from_env = forbidden_provider_call
socket.create_connection = forbidden_network_call

from flash.serve.provisioning.modal.execution.sdk import _load_modal_module

loaded = _load_modal_module()
assert loaded is modal
assert loaded.experimental is importlib.import_module("modal.experimental")
for name in ("list_deployed_apps", "get_app_objects", "get_app_lifecycle", "stop_app"):
    operation = getattr(loaded.experimental, name)
    assert callable(operation)
    assert callable(operation.aio)
assert provider_calls == []
assert network_calls == []
"""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    modal_pin = next(
        requirement
        for requirement in project["project"]["optional-dependencies"]["server"]
        if requirement.startswith("modal==")
    )
    result = subprocess.run(
        [sys.executable, "-c", program, modal_pin.removeprefix("modal==")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_exhausted_deadline_rejects_before_the_operation_starts() -> None:
    started = False

    async def operation() -> None:
        nonlocal started
        started = True

    with pytest.raises(TimeoutError):
        _sync_value(operation, deadline_at=10.0, clock=lambda: 10.0)

    assert started is False


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), InterruptedProvisioning("modal")])
def test_async_bridge_preserves_interruptions(interruption: BaseException) -> None:
    async def interrupt() -> None:
        raise interruption

    with pytest.raises(type(interruption)):
        _sync_value(interrupt, deadline_at=10.0, clock=lambda: 0.0)


def test_modal_mutation_timeout_is_ambiguous() -> None:
    with pytest.raises(ModalSdkFailure) as exc_info:
        _call_mutation(_never_resolves, deadline_at=10.0, clock=lambda: 10.0)
    assert exc_info.value.code == "resource_ambiguous", (
        "a timed-out mutation was treated as a definite failure"
    )
    assert exc_info.value.outcome_unknown is True, (
        "a timed-out mutation lost its unknown provider outcome"
    )


def test_modal_read_timeout_is_a_definite_transport_failure() -> None:
    with pytest.raises(ModalSdkFailure) as exc_info:
        _call_read(_never_resolves, deadline_at=10.0, clock=lambda: 10.0)
    assert exc_info.value.code == "transport_failed", (
        "a timed-out read was treated as an ambiguous mutation"
    )
    assert exc_info.value.outcome_unknown is False, (
        "a timed-out read incorrectly claimed provider state may have changed"
    )


def test_fast_modal_awaitable_is_unaffected_by_the_deadline() -> None:
    result = _sync_value(_fast_value, deadline_at=10.0, clock=lambda: 0.0)

    assert result == "fast-modal-result", "a fast modal call was rejected before its deadline"


def test_client_closes_after_its_operation_deadline_has_expired() -> None:
    from types import SimpleNamespace

    from flash.serve.provisioning.modal.execution.sdk import PinnedModalSdk, _AsyncBridge

    closed = []

    async def close() -> None:
        closed.append(asyncio.get_running_loop())

    sdk = object.__new__(PinnedModalSdk)
    sdk._clock = lambda: 20.0
    sdk._deadline_at = 10.0
    sdk._bridge = _AsyncBridge(sdk._clock)
    sdk._client = SimpleNamespace(_close=SimpleNamespace(aio=close))
    sdk.close()
    assert closed == [sdk._bridge._loop]
    assert sdk._bridge._loop.is_closed()
