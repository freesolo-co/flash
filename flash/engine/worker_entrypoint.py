"""Safe subprocess entrypoint for the managed GPU worker."""

from __future__ import annotations

import sys

WORKER_FAILURE_LINE = "worker failure; detail suppressed"


def main() -> int:
    try:
        from flash.engine import worker
    except BaseException:
        print(WORKER_FAILURE_LINE, file=sys.stderr, flush=True)
        return 1

    try:
        worker.main()
    except BaseException as exc:
        if not isinstance(exc, Exception):
            print(WORKER_FAILURE_LINE, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
