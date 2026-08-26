from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from flash.providers._lifecycle.bootstrapping.processes import (
    start_process_group,
    terminate_process_group,
)


@pytest.mark.wallclock
def test_process_group_termination_reaps_exited_leader_without_full_grace() -> None:
    process, group = start_process_group(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    started = time.monotonic()
    try:
        terminate_process_group(
            process,
            process_group_id=group,
            term_grace_s=1.0,
            kill_grace_s=1.0,
        )
        assert time.monotonic() - started < 0.8
        assert process.returncode == -signal.SIGTERM
    finally:
        if process.poll() is None:
            os.killpg(group, signal.SIGKILL)
            process.wait(timeout=1)


def test_process_group_termination_reaps_child_group() -> None:
    process, group = start_process_group(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                "time.sleep(60)"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert group == process.pid
        terminate_process_group(
            process,
            process_group_id=group,
            term_grace_s=0.2,
            kill_grace_s=0.5,
        )
        assert process.poll() is not None
        deadline = time.time() + 1.0
        while time.time() < deadline:
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("terminated process group still exists")
    finally:
        if process.poll() is None:
            os.killpg(group, signal.SIGKILL)
            process.wait(timeout=1)
