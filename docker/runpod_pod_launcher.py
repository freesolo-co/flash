"""Static RunPod Pod entrypoint for the shared instance bootstrap."""

from __future__ import annotations

import contextlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

PAYLOAD_ENV = "FLASH_INSTANCE_PAYLOAD"
PAYLOAD_PATH = Path("/root/flash/payload.json")
CAPSULE_PATH = Path("/opt/flash/instance-bootstrap.pyz")
BAKE_ENTRY_PATH = Path("/opt/flash/bake_pod_entry.py")


def main() -> int:
    payload = os.environ.pop(PAYLOAD_ENV, None)
    if not payload:
        print("flash: payload secret reference did not resolve", file=sys.stderr, flush=True)
        return 1
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError):
        print("flash: payload secret is not valid json", file=sys.stderr, flush=True)
        return 1
    if type(parsed) is not dict:
        print("flash: payload secret must contain a json object", file=sys.stderr, flush=True)
        return 1
    PAYLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        PAYLOAD_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR
    )
    try:
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(parsed, stream, sort_keys=True, separators=(",", ":"))
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise
    os.environ.pop(PAYLOAD_ENV, None)
    if parsed.get("mode") == "kernel_bake":
        return subprocess.call([sys.executable, str(BAKE_ENTRY_PATH), str(PAYLOAD_PATH)])
    return subprocess.call([sys.executable, str(CAPSULE_PATH), "bootstrap"])


if __name__ == "__main__":
    raise SystemExit(main())
