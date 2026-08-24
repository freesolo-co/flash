"""Safe subprocess entrypoint for the managed GPU worker."""

from __future__ import annotations

import sys
import traceback

WORKER_FAILURE_LINE = "managed worker failed; inspect worker artifacts"


def _safe_traceback() -> str:
    try:
        from flash._internal.diagnostics import neutralize_control_chars, sanitize_diagnostic

        return neutralize_control_chars(sanitize_diagnostic(traceback.format_exc(), limit=16_000))
    except BaseException:
        return "Traceback unavailable; diagnostic sanitization failed"


def main() -> int:
    try:
        from flash.engine import worker

        worker.main()
    except BaseException:
        print(_safe_traceback(), file=sys.stderr, flush=True)
        print(WORKER_FAILURE_LINE, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
