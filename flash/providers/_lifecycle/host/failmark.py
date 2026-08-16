"""host helper: write the terminal failure marker for a boot that died before the worker.

Runs as a standalone program on the rented box, shipped as a declared member of the
instance-bootstrap capsule.

The work lives under a ``__main__`` guard because this is a program, not a library: the capsule
dispatches into it with ``runpy.run_module(run_name="__main__")``, while merely IMPORTING it (a
module walker, a test collector, a packaging probe) must do nothing at all. Unguarded, an import
would read the box payload and write a FAILURE MARKER for a run that is fine -- and the blanket
``except`` below would hide that it had.
"""

import json
import math
import time


def main() -> None:
    """Write the retriable host failure marker, unless the worker already wrote its own.

    Marker-aware in both directions: a container that fast-failed on a real user error has already
    written ok=false at this path, and relabeling it retriable would turn a user error into an
    endless retry. Best effort -- a marker upload must never be what kills the box.
    """
    try:
        with open("/opt/flash/payload.json") as _payload_file:
            p = json.load(_payload_file)
        arm = p.get("flash_arm", "instance")
        att = p.get("attempt")
        run_id = p.get("run_id")
        deadline = p.get("deadline_at")
        created_at = p.get("run_created_at")
        max_wall_seconds = p.get("run_max_wall_seconds")
        if isinstance(att, bool) or not isinstance(att, int) or att < 0:
            raise ValueError("invalid attempt")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("invalid run id")
        clocks = (deadline, created_at, max_wall_seconds)
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
        marker_path = p["hf_prefix"] + "/" + arm + "_attempt" + str(att) + ".json"
        from huggingface_hub import HfApi

        api = HfApi(token=(p.get("env") or {}).get("HF_TOKEN"))

        def worker_marker_present():
            try:
                return api.file_exists(
                    repo_id=p["hf_repo"], filename=marker_path, repo_type="dataset"
                )
            except Exception:
                return True  # conservative: on a read error, never risk clobbering

        if not worker_marker_present():
            marker = {
                "attempt": att,
                "error": "host bootstrap failure",
                "ok": False,
                "retriable": True,
                "run_id": run_id,
                "ts": now,
            }
            with open("/opt/flash/fm.json", "w") as _marker_file:
                _marker_file.write(json.dumps(marker))
            # re-check right before the upload: the worker may have written its own ok=false marker in
            # the gap since the first check; honor it instead of clobbering with the retriable host one.
            if not worker_marker_present():
                api.upload_file(
                    path_or_fileobj="/opt/flash/fm.json",
                    path_in_repo=marker_path,
                    repo_id=p["hf_repo"],
                    repo_type="dataset",
                )
    except Exception:
        pass


if __name__ == "__main__":
    main()
