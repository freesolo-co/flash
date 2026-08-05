"""`python -m flash.server` — run the managed control plane."""

from __future__ import annotations

import argparse
import logging
import os

from .._logging import configure_logging
from .app import run_server

HOST_ENV = "FLASH_SERVER_HOST"
PORT_ENV = "FLASH_SERVER_PORT"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def _default_port() -> int:
    """The port from ``FLASH_SERVER_PORT``, else 8080.

    A platform that assigns the port through the environment is the reason this exists, so an
    unusable value is a hard error at startup: binding 8080 instead of the port the platform
    routed to would leave the plane running and unreachable, which is worse than not starting.
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
    # The flags win over the environment: an operator typing --port meant that port.
    parser.add_argument("--host", default=os.environ.get(HOST_ENV, "").strip() or DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_default_port())
    args = parser.parse_args(argv)
    # The server logs at INFO unless FLASH_LOG_LEVEL says otherwise. Provider resolution,
    # capability revocation, reaper startup, and degraded-configuration warnings are only
    # observable here, and without this call the `flash` logger keeps the NullHandler it gets at
    # import and emits none of them.
    configure_logging(default_level=logging.INFO)
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
