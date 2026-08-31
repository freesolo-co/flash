"""hydrate or serve one externally bound immutable serving manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import threading
from collections.abc import Callable
from contextlib import contextmanager

from .bootstrap import bootstrap_serving
from .http import create_app
from .manifest import ServingManifest
from .materialize import read_artifact_token_fd
from .progress import emit_boot_progress

_INFERENCE_TOKEN_FD_ENV = "FLASH_INFERENCE_TOKEN_FD"
_HARD_SHUTDOWN_DEADLINE_SECONDS = 10.0


def _read_inference_token(fd: int | None = None) -> str:
    if fd is None:
        raw_fd = os.environ.get(_INFERENCE_TOKEN_FD_ENV)
        if raw_fd is None or not raw_fd.isdecimal():
            raise RuntimeError("inference token fd is not configured")
        fd = int(raw_fd)
    return read_artifact_token_fd(fd)


async def _serve(
    args: argparse.Namespace,
    manifest: ServingManifest,
    *,
    inference_token_fd: int | None = None,
    on_signals_installed: Callable[[], dict[int, object]] | None = None,
) -> None:
    import uvicorn

    token = (
        _read_inference_token()
        if inference_token_fd is None
        else _read_inference_token(inference_token_fd)
    )
    try:
        bearer_digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    finally:
        token = ""
    server: uvicorn.Server | None = None
    shutdown_timer: threading.Timer | None = None

    async def _exit_on_engine_death(_health: object) -> None:
        """stop serving after an engine death or fatal cancellation failure."""

        nonlocal shutdown_timer
        print(
            "serving: runtime cannot safely continue; shutting down so the provider replaces "
            "this container",
            flush=True,
        )
        if server is None:
            os._exit(1)
        server.should_exit = True
        if shutdown_timer is None:
            shutdown_timer = threading.Timer(
                _HARD_SHUTDOWN_DEADLINE_SECONDS,
                os._exit,
                args=(1,),
            )
            shutdown_timer.daemon = True
            shutdown_timer.start()

    owner = await bootstrap_serving(
        manifest, args.cache_root, on_engine_death=_exit_on_engine_death
    )
    try:
        app = create_app(owner, bearer_digest=bearer_digest)
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        if on_signals_installed is not None:

            @contextmanager
            def capture_with_handoff():
                startup_handlers = {
                    signum: signal.signal(signum, server.handle_exit)
                    for signum in uvicorn.server.HANDLED_SIGNALS
                }
                try:
                    restore_handlers = on_signals_installed()
                except BaseException:
                    for signum, handler in startup_handlers.items():
                        signal.signal(signum, handler)
                    raise
                try:
                    yield
                finally:
                    for signum, handler in restore_handlers.items():
                        signal.signal(signum, handler)
                for captured_signal in reversed(server._captured_signals):
                    signal.raise_signal(captured_signal)

            server.capture_signals = capture_with_handoff
        emit_boot_progress("port-bind-starting", host=args.host, port=args.port)
        await server.serve()
    finally:
        try:
            await owner.close()
        finally:
            if shutdown_timer is not None:
                shutdown_timer.cancel()
