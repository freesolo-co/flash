"""host helper: best-effort boot-log upload to hf, attempt-scoped path.

Runs as a standalone program on the rented box, shipped as a declared member of the
instance-bootstrap capsule.

The work lives under a ``__main__`` guard because this is a program, not a library: the capsule
dispatches into it with ``runpy.run_module(run_name="__main__")``, while merely IMPORTING it (a
module walker, a test collector, a packaging probe) must do nothing at all. Unguarded, an import
would read the box payload and attempt a real HF upload -- and the blanket ``except`` below would
hide that it had.
"""

import json
import math
import time


def main() -> None:
    """Upload the host boot log to the attempt-scoped HF path. Best effort by design.

    Every failure is swallowed: this is a liveness artifact, and a box must never die because its
    diagnostic upload did. The control plane treats the artifact's ABSENCE as the signal.
    """
    try:
        with open("/opt/flash/payload.json") as _payload_file:
            p = json.load(_payload_file)
        arm = p.get("flash_arm", "instance")
        att = p.get("attempt")
        deadline = p.get("deadline_at")
        created_at = p.get("run_created_at")
        max_wall_seconds = p.get("run_max_wall_seconds")
        clocks = (deadline, created_at, max_wall_seconds)
        if isinstance(att, bool) or not isinstance(att, int) or att < 0:
            raise ValueError("invalid attempt")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in clocks):
            raise ValueError("invalid deadline")
        deadline, created_at, max_wall_seconds = map(float, clocks)
        now = time.time()
        if (
            not all(math.isfinite(value) and value > 0 for value in clocks)
            or not math.isfinite(now)
            or now <= 0
            or now >= deadline
            or not math.isclose(deadline, created_at + max_wall_seconds, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError("invalid clock")
        from huggingface_hub import HfApi

        HfApi(token=(p.get("env") or {}).get("HF_TOKEN")).upload_file(
            path_or_fileobj="/opt/flash/host_boot.log",
            path_in_repo=p["hf_prefix"] + "/" + arm + "_attempt" + str(att) + "_boot.log",
            repo_id=p["hf_repo"],
            repo_type="dataset",
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
