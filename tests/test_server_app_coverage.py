"""Hermetic coverage for server background jobs and optional dependency errors."""

from __future__ import annotations

import asyncio
import builtins
import sys
import types

import pytest

import flash.server.asgi.app as app_mod


def test_start_deployment_job_uses_a_daemon_background_thread(monkeypatch) -> None:
    """Asynchronous deployment work must run through the daemon wrapper and clear its registry."""
    calls = []
    current = {}

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            calls.append(("init", target, args, daemon))
            self.target = target
            self.args = args
            current["thread"] = self

        def start(self):
            calls.append(("start",))

    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    monkeypatch.setattr(app_mod.threading, "Thread", FakeThread)
    monkeypatch.setattr(app_mod.threading, "current_thread", lambda: current["thread"])
    app_mod._open_deployment_jobs()
    observed = []

    def record(value, *, final):
        observed.append((value, final))

    assert app_mod.start_deployment_job(record, "done", final=True) is False
    thread = current["thread"]
    with app_mod._DEPLOYMENT_JOBS_LOCK:
        assert thread in app_mod._DEPLOYMENT_JOBS
        assert len(app_mod._DEPLOYMENT_JOBS) == 1

    thread.target(*thread.args)

    assert observed == [("done", True)]
    assert calls == [
        ("init", app_mod._run_deployment_job, (record, ("done",), {"final": True}), True),
        ("start",),
    ]
    with app_mod._DEPLOYMENT_JOBS_LOCK:
        assert not app_mod._DEPLOYMENT_JOBS


def _run_loop_once(monkeypatch, loop, worker):
    sleeps = 0

    async def sleep(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    async def to_thread(function, *args):
        return worker(function, *args)

    monkeypatch.setattr(app_mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(app_mod.asyncio, "to_thread", to_thread)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop())


def test_idle_endpoint_loop_logs_successful_deletion(monkeypatch) -> None:
    """The idle reaper must log nonzero deletions and propagate cancellation on the next cycle."""
    messages = []
    logger = types.SimpleNamespace(
        info=lambda message, *args: messages.append(message % args),
        debug=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(app_mod, "_log", logger)
    monkeypatch.setattr(app_mod, "_reap_idle_endpoints_once", lambda minimum: 2)

    _run_loop_once(
        monkeypatch, app_mod._reap_idle_endpoints_loop, lambda function, *args: function(*args)
    )

    assert any("reaped 2 idle RunPod endpoint" in message for message in messages)


def test_idle_endpoint_loop_retries_after_a_sweep_failure(monkeypatch) -> None:
    """A provider error must be logged and retried rather than terminating the idle reaper."""
    messages = []
    logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda message, *args, **kwargs: messages.append(message % args),
    )
    monkeypatch.setattr(app_mod, "_log", logger)

    def worker(function, *args):
        raise RuntimeError("runpod unavailable")

    _run_loop_once(monkeypatch, app_mod._reap_idle_endpoints_loop, worker)

    assert messages == ["idle-endpoint reaper sweep failed; retrying next cycle"]


def test_idle_endpoint_loop_propagates_cancellation_from_worker(monkeypatch) -> None:
    """Cancellation raised during the offloaded sweep must escape immediately for clean shutdown."""

    async def sleep(_seconds):
        return None

    async def to_thread(function, *args):
        raise asyncio.CancelledError

    monkeypatch.setattr(app_mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(app_mod.asyncio, "to_thread", to_thread)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(app_mod._reap_idle_endpoints_loop())


def test_instance_orphan_loop_logs_successful_sweep(monkeypatch) -> None:
    """The instance reaper must report torn-down workers before honoring shutdown cancellation."""
    messages = []
    logger = types.SimpleNamespace(
        info=lambda message, *args: messages.append(message % args),
        debug=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(app_mod, "_log", logger)
    monkeypatch.setattr(app_mod, "_sweep_orphan_instances_once", lambda: 3)

    _run_loop_once(
        monkeypatch,
        app_mod._sweep_orphan_instances_loop,
        lambda function, *args: function(*args),
    )

    assert any("swept 3 orphaned instance-provider worker" in message for message in messages)


def test_instance_orphan_loop_retries_failures_and_propagates_cancellation(monkeypatch) -> None:
    """Instance sweep failures must be retryable while cancellation remains a shutdown signal."""
    messages = []
    logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda message, *args, **kwargs: messages.append(message % args),
    )
    monkeypatch.setattr(app_mod, "_log", logger)

    def worker(function, *args):
        raise RuntimeError("provider unavailable")

    _run_loop_once(monkeypatch, app_mod._sweep_orphan_instances_loop, worker)
    assert messages == ["instance orphan sweep failed; retrying next cycle"]

    async def sleep(_seconds):
        return None

    async def cancelled(function, *args):
        raise asyncio.CancelledError

    monkeypatch.setattr(app_mod.asyncio, "sleep", sleep)
    monkeypatch.setattr(app_mod.asyncio, "to_thread", cancelled)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(app_mod._sweep_orphan_instances_loop())


def test_create_app_explains_missing_server_extras(monkeypatch) -> None:
    """Missing FastAPI extras must produce the actionable server installation hint."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fastapi":
            raise ImportError("fastapi missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="server"):
        app_mod.create_app()


def test_run_server_explains_missing_uvicorn(monkeypatch) -> None:
    """Missing Uvicorn must fail with the same actionable optional-extra guidance."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn":
            raise ImportError("uvicorn missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="server"):
        app_mod.run_server()


def test_run_server_builds_the_app_and_delegates_to_uvicorn(monkeypatch) -> None:
    """The server entrypoint must pass its constructed app and explicit bind address to Uvicorn."""
    calls = []
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda application, **kwargs: calls.append((application, kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    application = object()
    monkeypatch.setattr(app_mod, "create_app", lambda: application)
    # run_server preflights the operator's configuration before starting uvicorn; this test runs
    # with none set, so stub it out -- the preflight behaviour itself is covered separately.
    monkeypatch.setattr("flash.providers.core.preflight.require_operator_config", lambda: None)

    app_mod.run_server(host="0.0.0.0", port=9000)

    assert calls == [(application, {"host": "0.0.0.0", "port": 9000})]


def test_run_server_preflights_before_starting_uvicorn(monkeypatch) -> None:
    """Order is the whole point: uvicorn calls sys.exit() on a startup failure internally.

    A PreflightError raised once uvicorn owns the process cannot be caught by our caller, so the
    operator gets it rendered as an unhandled ASGI startup exception under ~20 frames of
    starlette/contextlib. Running the check first is what lets __main__ print `error: ...` instead.
    """
    from flash.providers.core.preflight import PreflightError

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *_args, **_kwargs: pytest.fail(
        "uvicorn started despite a failing preflight"
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(app_mod, "create_app", lambda: pytest.fail("app built before preflight"))

    def _fail():
        raise PreflightError("HF_TOKEN is required")

    monkeypatch.setattr("flash.providers.core.preflight.require_operator_config", _fail)

    with pytest.raises(PreflightError, match="HF_TOKEN"):
        app_mod.run_server(host="0.0.0.0", port=9000)


def test_run_server_does_not_repeat_the_advisory_preflight_logging(monkeypatch) -> None:
    """The early call validates only; the lifespan copy still owns the operator-facing warnings.

    Both call sites run on a successful boot, so an early call that also warned would print the
    GITHUB_TOKEN warning and the `GPU provider(s) configured:` summary twice -- doubling exactly
    the lines SELF_HOSTING.md tells an operator to read, and any alert keyed on them.
    """
    seen: list[str] = []
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(app_mod, "create_app", lambda: object())
    monkeypatch.setattr(
        "flash.providers.core.preflight.require_operator_config",
        lambda: seen.append("require"),
    )
    monkeypatch.setattr(
        "flash.providers.core.preflight.check_run_preflight",
        lambda: seen.append("check"),
    )

    app_mod.run_server(host="0.0.0.0", port=9000)

    # the refusing half only: the advisory summary belongs to the lifespan, which this test's
    # stubbed create_app never builds.
    assert seen == ["require"]
