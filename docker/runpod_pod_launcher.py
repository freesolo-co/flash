"""Static RunPod Pod entrypoint for the shared instance bootstrap."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import threading
from pathlib import Path

PAYLOAD_ENV = "FLASH_INSTANCE_PAYLOAD"
PAYLOAD_PATH = Path("/root/flash/payload.json")
CAPSULE_PATH = Path("/opt/flash/instance-bootstrap.pyz")
BAKE_ENTRY_PATH = Path("/opt/flash/bake_pod_entry.py")


def _park_until_stopped(stopped: threading.Event) -> None:
    stopped.wait()


def _run_child_once(
    parsed: dict,
    *,
    popen=None,
    park=None,
    install_signal=None,
) -> int:
    stopped = threading.Event()
    child = None
    stop_signum = None

    def stop(signum, _frame) -> None:
        nonlocal stop_signum
        stop_signum = signum
        stopped.set()
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                child.send_signal(signum)

    register = install_signal or signal.signal
    register(signal.SIGTERM, stop)
    register(signal.SIGINT, stop)
    command = (
        [sys.executable, str(BAKE_ENTRY_PATH), str(PAYLOAD_PATH)]
        if parsed.get("mode") == "kernel_bake"
        else [sys.executable, str(CAPSULE_PATH), "bootstrap"]
    )
    child = (popen or subprocess.Popen)(command)
    if stop_signum is not None and child.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            child.send_signal(stop_signum)
    child.wait()
    (park or _park_until_stopped)(stopped)
    return 0


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
    return _run_child_once(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
