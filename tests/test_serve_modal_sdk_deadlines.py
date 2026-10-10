"""deadline and timeout classification coverage for the modal sdk bridge."""

from __future__ import annotations

import asyncio
import subprocess
import sys
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

import modal

provider_calls = []
network_calls = []


def forbidden_provider_call(*args, **kwargs):
    provider_calls.append((args, kwargs))
    raise AssertionError("provider operation attempted")


def forbidden_network_call(*args, **kwargs):
    network_calls.append((args, kwargs))
    raise AssertionError("network operation attempted")


assert importlib.metadata.version("modal") == "1.5.4"
assert not hasattr(modal, "experimental")
modal.Client.from_credentials = forbidden_provider_call
modal.Client.from_env = forbidden_provider_call
socket.create_connection = forbidden_network_call

from flash.serve.provisioning.modal.execution.sdk import _load_modal_module

loaded = _load_modal_module()
assert loaded is modal
assert loaded.experimental is importlib.import_module("modal.experimental")
for name in ("list_deployed_apps", "get_app_objects", "get_app_lifecycle", "stop_app"):
    assert callable(getattr(loaded.experimental, name))
assert provider_calls == []
assert network_calls == []
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
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
