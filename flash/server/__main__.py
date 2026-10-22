"""Run the managed control plane with ``python -m flash.server``."""

from flash.server.asgi.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
