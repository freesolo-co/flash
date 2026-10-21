"""hydrate or serve one externally bound immutable serving manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
from collections.abc import Callable
from contextlib import contextmanager

from .bootstrap import bootstrap_serving
from .http import create_app
from .manifest import ServingManifest
from .materialize import read_artifact_token_fd
from .progress import emit_boot_progress

_INFERENCE_TOKEN_FD_ENV = "FLASH_INFERENCE_TOKEN_FD"


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

    async def _exit_on_engine_death(_health: object) -> None:
        """stop serving once the vllm engine core dies.

        a dead engine core cannot be repaired in place. without this the http process stays bound
        and answers 503 for every later request, so the provider never replaces the container.
        asking uvicorn to exit drains in-flight requests and ends the process, which lets the
        provider restart the container.
        """

        print(
            "serving: vllm engine core is dead; shutting down so the provider replaces this "
            "container",
            flush=True,
        )
        if server is not None:
            server.should_exit = True
        else:
            os._exit(1)

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
        await owner.close()
