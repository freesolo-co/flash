"""Managed control-plane command-line entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from flash._internal.logging import configure_logging
from flash.providers.core.preflight import PreflightError
from flash.server.asgi.app import run_server

HOST_ENV = "FLASH_SERVER_HOST"
PORT_ENV = "FLASH_SERVER_PORT"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def _port_from_env() -> int:
    """The port from ``FLASH_SERVER_PORT``, else 8080.

    A platform that assigns the port through the environment is the reason this exists, so an
    unusable value is a hard error at startup: binding 8080 instead of the port the platform
    routed to would leave the plane running and unreachable, which is worse than not starting.

    Only called when ``--port`` was NOT given. Validating at parser-construction time instead
    would make a malformed variable outrank the flag meant to override it, and would break
    ``--help`` for the operator trying to find out what the flag is called.
    """
    raw = os.environ.get(PORT_ENV, "").strip()
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        raise SystemExit(
            f"{PORT_ENV} must be an integer between 1 and 65535, got {raw!r}"
        ) from None
    if not 1 <= port <= 65535:
        raise SystemExit(f"{PORT_ENV} must be between 1 and 65535, got {port}")
    return port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flash.server", description="Flash control plane")
    # The flags win over the environment: an operator typing --port meant that port. The port
    # default stays None here and is resolved after parsing, so an unusable FLASH_SERVER_PORT
    # is only fatal when nothing overrode it.
    parser.add_argument("--host", default=os.environ.get(HOST_ENV, "").strip() or DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    if args.port is None:
        args.port = _port_from_env()
    # The server logs at INFO unless FLASH_LOG_LEVEL says otherwise. Provider resolution,
    # capability revocation, reaper startup, and degraded-configuration warnings are only
    # observable here, and without this call the `flash` logger keeps the NullHandler it gets at
    # import and emits none of them.
    configure_logging(default_level=logging.INFO)
    try:
        run_server(host=args.host, port=args.port)
    except PreflightError as exc:
        # The operator is missing configuration, not looking at a bug: print what to set and stop.
        # 3 is what this path already exited with when the error surfaced from inside the ASGI
        # lifespan (uvicorn's STARTUP_FAILURE), so supervision that keys on it keeps working.
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 0
