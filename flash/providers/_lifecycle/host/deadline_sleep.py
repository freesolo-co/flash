"""host helper: cap every cloud-init polling and retry delay at the canonical run deadline.

Runs as a standalone program on the rented box (`python3 capsule.pyz deadline_sleep <seconds>`),
shipped as a declared member of the instance-bootstrap capsule.

The work lives under a ``__main__`` guard because this is a program, not a library: the capsule
dispatches into it with ``runpy.run_module(run_name="__main__")``, while merely IMPORTING it (a
module walker, a test collector, a packaging probe) must do nothing at all. Without the guard the
import reads argv, reads the box payload, and exits the importing process.
"""

import json
import math
import sys
import time


def main() -> None:
    """Sleep for at most the requested seconds, never past the run deadline.

    Exit 124 means the deadline is reached (the caller must stop waiting); exit 125 means the
    deadline could not be established, which is treated the same way rather than sleeping blind.
    """
    try:
        requested = float(sys.argv[1])
        with open("/opt/flash/payload.json") as _payload_file:
            p = json.load(_payload_file)
        deadline = p.get("deadline_at")
        created_at = p.get("run_created_at")
        max_wall_seconds = p.get("run_max_wall_seconds")
        clocks = (deadline, created_at, max_wall_seconds)
        if (
            not math.isfinite(requested)
            or requested < 0
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in clocks
            )
        ):
            raise ValueError("invalid deadline sleep")
        deadline, created_at, max_wall_seconds = map(float, clocks)
        now = time.time()
        if (
            not all(math.isfinite(value) and value > 0 for value in clocks)
            or not math.isfinite(now)
            or now <= 0
            or not math.isclose(deadline, created_at + max_wall_seconds, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError("invalid deadline clock")
        remaining = deadline - now
        if remaining <= 0:
            raise SystemExit(124)
        delay = min(requested, remaining)
        if delay > 0:
            time.sleep(delay)
        if requested >= remaining:
            raise SystemExit(124)
    except Exception:
        # SystemExit is a BaseException, so the 124 exits above pass straight through.
        raise SystemExit(125) from None


if __name__ == "__main__":
    main()
