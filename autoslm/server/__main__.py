"""`python -m autoslm.server` — run the managed control plane."""

from __future__ import annotations

import argparse

from .app import run_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoslm.server", description="AutoSLM control plane")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
