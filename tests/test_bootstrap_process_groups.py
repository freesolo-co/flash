"""Process-group teardown for the worker the instance bootstrap launches.

Real subprocesses throughout: the property under test is that a descendant holding the gpu dies,
and a mocked Popen cannot demonstrate that.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time

import pytest

from flash.providers._lifecycle.bootstrapping import processes as bootstrap_processes


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - visible but not ours
        return True
    return True


def _wait_gone(pid: int, timeout_s: float = 30.0) -> bool:
    """Poll rather than assert once: signal delivery and reaping are asynchronous."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


def test_start_process_group_puts_the_child_in_its_own_group():
    process, group_id = bootstrap_processes.start_process_group(
        [sys.executable, "-c", "import time; time.sleep(300)"]
    )
    try:
        assert group_id == process.pid
        assert os.getpgid(process.pid) == group_id
        # its own group, not the bootstrap's, or terminating it would signal the bootstrap too.
        assert group_id != os.getpgid(0)
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(group_id, signal.SIGKILL)
        process.wait(timeout=30)


def test_start_process_group_refuses_a_caller_supplied_session_flag():
    """The group identity is the whole point, so a caller must not be able to opt out of it."""
    with pytest.raises(ValueError, match="start_new_session"):
        bootstrap_processes.start_process_group(
            [sys.executable, "-c", "pass"], start_new_session=False
        )


def test_terminate_process_group_stops_a_cooperative_worker():
    process, group_id = bootstrap_processes.start_process_group(
        [sys.executable, "-c", "import time; time.sleep(300)"]
    )
    bootstrap_processes.terminate_process_group(process, process_group_id=group_id)
    assert process.poll() is not None
    assert not _alive(group_id)


def test_terminate_process_group_escalates_when_sigterm_is_ignored():
    src = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(300)\n"
    )
    process, group_id = bootstrap_processes.start_process_group(
        [sys.executable, "-c", src], stdout=subprocess.PIPE, text=True
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"
    bootstrap_processes.terminate_process_group(
        process, process_group_id=group_id, term_grace_s=1.0, kill_grace_s=10.0
    )
    assert process.poll() is not None


def test_terminate_process_group_reaps_a_child_that_outlives_the_leader():
    """The failure this exists for: the leader exits on SIGTERM, its gpu-holding child does not.

    Escalating off the leader alone returns as soon as it is reaped, leaving the child holding the
    cuda context and stranding a paid gpu for every later run on the box.
    """
    leader_src = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c',\n"
        '    "import signal,time\\n"\n'
        '    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"\n'
        "    \"print('gready', flush=True)\\n\"\n"
        '    "time.sleep(300)\\n"], stdout=subprocess.PIPE, text=True)\n'
        "assert g.stdout.readline().strip() == 'gready'\n"
        "print(g.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    process, group_id = bootstrap_processes.start_process_group(
        [sys.executable, "-c", leader_src], stdout=subprocess.PIPE, text=True
    )
    child_pid = None
    try:
        assert process.stdout is not None
        child_pid = int(process.stdout.readline().strip())
        bootstrap_processes.terminate_process_group(
            process, process_group_id=group_id, term_grace_s=1.0, kill_grace_s=20.0
        )
        assert process.poll() is not None
        # the leader dying is not the property under test; the child being gone is.
        assert _wait_gone(child_pid), "child ignoring SIGTERM survived the group teardown"
    finally:
        if child_pid is not None:  # pragma: no cover - only on an unexpected failure
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(child_pid, signal.SIGKILL)


def test_terminate_process_group_reaps_an_already_exited_leader():
    """An unwaited leader is a zombie, and a zombie group member reads as a live group.

    Without the ``poll()`` before the liveness probe, an empty group whose leader merely has not
    been waited on burns the full term and kill grace before returning.
    """
    process, group_id = bootstrap_processes.start_process_group([sys.executable, "-c", "pass"])
    # let the leader exit WITHOUT observing it: any poll/wait/kill here would reap it and destroy
    # the very zombie the probe has to cope with.
    time.sleep(1.0)
    assert process.returncode is None, "leader was reaped before the case under test could run"

    started = time.monotonic()
    bootstrap_processes.terminate_process_group(
        process, process_group_id=group_id, term_grace_s=10.0, kill_grace_s=10.0
    )
    elapsed = time.monotonic() - started
    assert process.poll() is not None
    # returns promptly instead of burning both grace windows on a group that is already empty.
    assert elapsed < 5.0, f"empty group took {elapsed:.1f}s, so the zombie leader read as alive"
