"""Safe subprocess entrypoint for the managed GPU worker."""

from __future__ import annotations

import sys

WORKER_FAILURE_LINE = "managed worker failed; inspect worker artifacts"


def main() -> int:
    try:
        from flash.engine import worker

        worker.main()
    except BaseException:
        print(WORKER_FAILURE_LINE, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
