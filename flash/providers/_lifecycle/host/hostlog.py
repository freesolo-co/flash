"""host helper: best-effort boot-log upload to hf, attempt-scoped path.

Runs as a standalone program on the rented box, shipped as a declared member of the
instance-bootstrap capsule.
"""

import json
import math
import time

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
