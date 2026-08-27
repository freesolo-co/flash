"""parent-side subprocess lifecycle for isolated verl training children."""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Self

from flash.engine.worker.verl.diagnostics import (
    ChildOutputTail,
    VerlChildSilenceWatchdog,
    build_verl_line_handler,
    raise_for_classified_verl_exit,
)


def _run_streaming_verl_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str],
    on_line: Callable[[str], None],
    errors: str | None = None,
    silence_watchdog: VerlChildSilenceWatchdog | None = None,
) -> int:
    """stream a verl subprocess under the shared process-group lifecycle supervisor."""
    adopt_orphaned_descendants()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors=errors,
        bufsize=1,
        start_new_session=True,
    )
    process_group_id = proc.pid
    if silence_watchdog is not None:
        silence_watchdog.bind(
            child_alive=lambda: proc.poll() is None,
            teardown=lambda: kill_process_group(proc, process_group_id=process_group_id),
        )
        silence_watchdog.start()
    try:
        with _ChildExitWatchdog(
            proc, process_group_id=process_group_id, grace_s=_ORPHANED_PIPE_GRACE_S
        ) as watchdog:
            assert proc.stdout is not None
            for line in proc.stdout:
                with watchdog.handling_line():
                    on_line(line)
    except BaseException:
        kill_process_group(proc, process_group_id=process_group_id)
        raise
    finally:
        if silence_watchdog is not None:
            silence_watchdog.stop()
        if proc.poll() is None:
            try:
                proc.wait(timeout=_TEARDOWN_GRACE_S)
            except subprocess.TimeoutExpired:
                kill_process_group(proc, process_group_id=process_group_id)
        reap_stragglers()
    if proc.returncode is None:
        raise RuntimeError(
            f"verl subprocess {proc.pid} did not exit after teardown; its process group is still "
            "holding the gpu"
        )
    return_code = int(proc.returncode)
    if watchdog.tore_down and return_code == 0:
        raise RuntimeError(
            f"verl subprocess {proc.pid} exited 0 but a descendant held its output pipe open for "
            f"{_ORPHANED_PIPE_GRACE_S:.0f}s; the process group was torn down to release the gpu"
        )
    if return_code != 0:
        kill_process_group(proc, process_group_id=process_group_id)
    return return_code


def run_verl_training(
    cmd: list[str],
    *,
    env: dict[str, str],
    on_step: Callable[[int], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    heartbeat: Callable[[], None] | None = None,
    step_pattern: str = r"step:\s*(\d+)",
    heartbeat_interval_s: float = 20.0,
    tail: ChildOutputTail | None = None,
    silence_watchdog: VerlChildSilenceWatchdog | None = None,
) -> int:
    """run a verl trainer subprocess, streaming stdout and surfacing step progress.

    returns the process exit code. stdout+stderr are merged and scanned line by line: ``on_line``
    receives every line, ``on_step`` receives each parsed training step, and ``heartbeat`` is called
    at most once per ``heartbeat_interval_s``. callback failures terminate the process group before
    they are re-raised. ``silence_watchdog`` tears the group down when a child that reached the fit
    loop stops producing distinct output while the parent is idle.
    """
    child_tail = tail if tail is not None else ChildOutputTail()
    handle_line = build_verl_line_handler(
        child_tail,
        on_step=on_step,
        on_line=on_line,
        heartbeat=heartbeat,
        step_pattern=step_pattern,
        heartbeat_interval_s=heartbeat_interval_s,
        silence_watchdog=silence_watchdog,
    )
    return_code = _run_streaming_verl_subprocess(
        cmd,
        env=env,
        on_line=handle_line,
        silence_watchdog=silence_watchdog,
    )
    raise_for_classified_verl_exit(return_code, child_tail)
    if silence_watchdog is not None:
        silence_watchdog.raise_if_failed()
    return return_code


_TEARDOWN_GRACE_S = 10.0

# grace for a descendant holding stdout open after the trainer exits. observe child exit separately
# from EOF because EngineCore can retain the pipe; allow final flushing before group teardown.
_ORPHANED_PIPE_GRACE_S = 30.0


class _ChildExitWatchdog:
    """Tears the group down when the direct child exits but a descendant holds the pipe open.

    this prevents an EngineCore-held pipe from blocking teardown forever (PR #730). it arms only after
    child exit, so ordinary trainer silence remains ``VerlChildSilenceWatchdog``'s responsibility.
    """

    def __init__(self, proc: subprocess.Popen, *, process_group_id: int, grace_s: float) -> None:
        self._proc = proc
        self._process_group_id = process_group_id
        self._grace_s = grace_s
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        # bumped by the reader for every line it takes off the pipe. the exit of the child alone is
        # NOT sufficient evidence of a leak: a child can exit having left a full pipe behind, and a
        # reader working through that backlog -- an on_step callback uploading a checkpoint takes
        # minutes -- would otherwise be killed mid-upload and its successful run reported as failed.
        # a stuck reader cannot advance this; a busy one does, which is exactly the distinction.
        self._lines_read = 0
        # how many lines are being HANDLED right now, not merely taken off the pipe. counting only
        # arrivals makes one long callback look identical to a stuck reader -- the counter cannot
        # advance while an upload runs, because the next line is not read until it returns. so
        # progress is "a line arrived OR one is still in hand", and the two together mean the
        # watchdog only ever fires on a reader that is neither receiving nor working.
        self._lines_in_flight = 0
        # read by the caller after the loop ends, to distinguish "the child closed its own pipe" from
        # "we closed it by killing the group out from under a survivor".
        self.tore_down = False

    @contextlib.contextmanager
    def handling_line(self):
        """Wraps the reader's whole per-line body, not just the moment the line arrives.

        Entered once per line and held until that line's callbacks return, so an `on_step` upload
        that outlasts the grace still reads as progress. Plain int stores, so no lock is needed: the
        watchdog only ever compares them, and either order of the two writes below leaves the reader
        looking busy rather than idle.
        """
        self._lines_read += 1
        self._lines_in_flight += 1
        try:
            yield
        finally:
            self._lines_in_flight -= 1

    def __enter__(self) -> Self:
        self._thread = threading.Thread(
            target=self._watch, name="verl-child-exit-watchdog", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._done.set()
        if self._thread is not None:
            # bounded: the thread only ever sleeps on `_done`, so this is a handoff, not a wait on
            # the child. the thread is a daemon regardless, so it can never hold the worker open.
            self._thread.join(timeout=_TEARDOWN_GRACE_S)

    def _watch(self) -> None:
        # `poll` rather than `wait`: this thread must stay responsive to `_done` instead of blocking
        # on the child. both collect the status, and CPython guards that with `_waitpid_lock`, so
        # whichever of the two threads gets there first is the one that sets `returncode`.
        while not self._done.wait(0.5):
            if self._proc.poll() is None:
                continue
            # the child is gone. that is necessary but not sufficient: require the reader to also be
            # making no progress across the grace, so a backlog being worked through is never killed.
            before = self._lines_read
            if self._done.wait(self._grace_s):
                return
            if self._lines_read != before or self._lines_in_flight:
                # the reader is still draining real output, or is inside a callback for a line it
                # already took. either way it is working, so keep watching rather than tearing down.
                continue
            self.tore_down = True
            kill_process_group(self._proc, process_group_id=self._process_group_id)
            return


def _process_is_zombie(pid: int) -> bool:
    """true when `pid` has exited and is only awaiting a reaper.

    parse state after the last ``)`` because ``/proc/<pid>/stat`` comm may contain parentheses. only
    disappearance proves exit; unreadable status counts as alive so EMFILE cannot suppress SIGKILL and
    strand an EngineCore CUDA context.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
            stat = handle.read()
    except (FileNotFoundError, ProcessLookupError):
        return True  # gone, so there is nothing left to wait for
    except OSError:
        return False  # unreadable, which is not evidence of an exit
    _, _, tail = stat.rpartition(")")
    fields = tail.split()
    return bool(fields) and fields[0] == "Z"


def _process_group_addressable(pgid: int) -> bool:
    """true while the kernel still knows `pgid`, whether or not its members are running."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, this process just may not signal it
    except OSError:
        return False
    return True


# pids this process adopted and killed but could not reap before a teardown deadline expired. each
# is confirmed ours -- `waitpid` answered for it rather than raising -- and was still running at
# that point, so it still owes a status. a zombie holds its pid until it is reaped, so nothing
# recorded here can be recycled behind our back and a later sweep can only ever find the same
# process it recorded.
_UNREAPED_STRAGGLERS: set[int] = set()

# PR_SET_CHILD_SUBREAPER, from linux/prctl.h.
_PR_SET_CHILD_SUBREAPER = 36


def adopt_orphaned_descendants() -> bool:
    """become the reaper for descendants orphaned below this process. true if the kernel agreed.

    `waitpid` then raises `ChildProcessError` here and the zombie is recorded as handled while it
    keeps its pid for the worker's whole life -- and the handler only ever waits on the worker
    itself, so nothing else ever collects it either. marking this process a subreaper makes the
    kernel reparent such orphans HERE instead, which is the condition the rest of this module
    already assumes. set once at teardown-path entry rather than at import, so merely importing this
    module cannot change an unrelated process's semantics.
    """
    global _ADOPTS_ORPHANS
    if _ADOPTS_ORPHANS:
        return True
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            return False
    except (OSError, AttributeError, ValueError):
        # no libc, or a kernel without the option (linux < 3.4). the reaps below degrade to what
        # they did before rather than failing the run: a leaked zombie costs a pid, not the job.
        return False
    _ADOPTS_ORPHANS = True
    return True


# whether this process has already claimed orphaned descendants; the prctl is idempotent but the
# call is not free, and teardown runs once per job on a reused worker.
_ADOPTS_ORPHANS = False


def _reap(pid: int) -> bool:
    """wait on `pid` without blocking. true once nothing is owed, false while it still owes a status.

    `ChildProcessError` means the pid is not ours to wait on, so there is no status left for this
    process to take. that answer is only safe because `adopt_orphaned_descendants` runs first: it
    makes an orphaned grandchild reparent to US, so `ChildProcessError` really does mean someone
    else owns the pid. Without it the same error is returned for a zombie nobody will ever reap.
    """
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True  # not ours, so no status is owed to this process
    except OSError:
        # cannot wait on it, and retrying later would not change that. dropped rather than tracked
        # so the straggler set cannot grow without bound on a path that can never clear it.
        return True
    return reaped != 0


def _reap_group_zombies(pgid: int, skip: int) -> None:
    """wait on any group member this process has adopted, clearing its process-table entry.

    SIGKILL cannot be refused but it also cannot be delivered while a process sits in
    uninterruptible sleep, so one can outlast the drain deadline and only then become a zombie --
    after the last wait this teardown performs. without a record no future wait is ever scheduled
    for it and the entry is permanent on a pid-1 worker.
    """
    for pid in _process_group_members(pgid) or ():
        if pid == skip:
            continue
        if _reap(pid):
            _UNREAPED_STRAGGLERS.discard(pid)
        else:
            _UNREAPED_STRAGGLERS.add(pid)


def reap_stragglers() -> None:
    """take the statuses still owed by processes an earlier teardown could not drain.

    this is the future wait that the final in-loop reap cannot schedule for itself: a member still
    running when its own teardown gave up is cleared by the next one instead.
    """
    for pid in tuple(_UNREAPED_STRAGGLERS):
        if _reap(pid):
            _UNREAPED_STRAGGLERS.discard(pid)


# how long the last drain waits for a straggler that was still running when its teardown gave up.
# it has already been SIGKILLed, so this is the delivery and exit latency of a process leaving
# uninterruptible sleep, not a grace period.
_EXIT_DRAIN_S = 5.0


def _drain_stragglers_before_exit() -> None:
    """block briefly for the statuses this process still owes, because no later teardown will.

    runpod starts a fresh worker subprocess per phase (endpoints.py `_train_body.run_mode`), so remembered stragglers
    otherwise reparent to a handler that never waits for them. perform one final bounded, non-fatal
    wait at interpreter exit.
    """
    deadline = time.monotonic() + _EXIT_DRAIN_S
    while _UNREAPED_STRAGGLERS:
        reap_stragglers()
        if not _UNREAPED_STRAGGLERS or time.monotonic() >= deadline:
            return
        time.sleep(0.05)


atexit.register(_drain_stragglers_before_exit)


def _process_group_alive(pgid: int) -> bool:
    """true while the group still has a RUNNING member.

    A drained verdict is only returned when TWO consecutive walks agree on it, because a single walk
    cannot distinguish a drained group from one it read too early. /proc can be listed while the
    leader is still present, and the leader can then fork and exit before its status is inspected:
    the snapshot is nonempty and zombie-only, yet the child that inherited the group is alive,
    unlisted, and never received the earlier signal -- so teardown returns without SIGKILL and it
    keeps its cuda context.
    """
    if not _process_group_addressable(pgid):
        return False
    for scan in range(2):
        members = _process_group_members(pgid)
        if members is None:
            # /proc could not be enumerated, so fall back to what signal 0 already established. it
            # over-reports rather than returning early on a live process still holding the gpu.
            return True
        if any(not _process_is_zombie(pid) for pid in members):
            return True
        if scan == 0:
            # this walk says drained. it is only believed if a second one, taken after any fork it
            # could have raced, still says so.
            continue
        if not members:
            # the kernel says this group exists and two walks found nobody in it. that contradiction
            # outlives the fork window, so it is the /proc walk that is wrong rather than late: fall
            # back to what signal 0 established rather than reporting a drain the kernel denies.
            return _process_group_addressable(pgid)
    return False


def _process_group_members(pgid: int) -> list[int] | None:
    """The pids currently in `pgid`, or None when /proc cannot answer.

    An empty list is a real answer -- the group has no members left -- and is distinct from None.
    """
    try:
        entries = os.listdir("/proc")
    except OSError:  # pragma: no cover - /proc missing is not reachable on linux
        return None
    members = []
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.getpgid(pid) == pgid:
                members.append(pid)
        except OSError:
            # exited between listdir and here, or not ours to inspect. a process this cannot see is
            # one it also could not have signalled.
            continue
    return members


def kill_process_group(proc: subprocess.Popen, *, process_group_id: int | None = None) -> None:
    """signal the child's whole process group, escalating to SIGKILL if anything survives.

    signalling the group rather than the pid is what reaches vllm's EngineCore grandchild; a
    survivor holds its cuda context and strands the gpu for every later run. the escalation is
    driven off the group, not off the direct child: the usual shape of this failure is the trainer
    dying on the term while the EngineCore ignores it, so waiting only on the child returns before
    the survivor is gone. a captured ``process_group_id`` keeps the group addressable after the
    direct child is reaped.
    """
    # grpo drives its own subprocess and calls this directly, never through `run_verl_training`, so
    # the adoption is claimed here too rather than only at the other entry point. idempotent.
    adopt_orphaned_descendants()
    # collect anything a previous teardown killed but could not drain before its deadline. done on
    # entry rather than on exit because that is when such a process has had the longest to die, and
    # it runs before the early returns below so no path through this function skips it.
    reap_stragglers()
    pgid = process_group_id
    if pgid is None:
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            # already reaped, so there is no group id left to address the survivors by.
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=_TEARDOWN_GRACE_S)
            return

    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, signal.SIGTERM)

    deadline = time.monotonic() + _TEARDOWN_GRACE_S
    # reap the direct child before probing: an unwaited zombie leader is still a group member, so the
    # liveness check below cannot otherwise tell an empty group from one that is merely unreaped.
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_TEARDOWN_GRACE_S)
    while _process_group_alive(pgid) and time.monotonic() < deadline:
        _reap_group_zombies(pgid, skip=proc.pid)
        time.sleep(0.1)
    _reap_group_zombies(pgid, skip=proc.pid)
    if not _process_group_alive(pgid):
        return

    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_TEARDOWN_GRACE_S)
    # sigkill cannot be refused, but delivery and reaping are asynchronous and the caller's next job
    # wants the gpu already free. wait for the group to drain -- bounded, since a zombie awaiting a
    # reaper that is not coming stays addressable and there is nothing stronger left to send.
    drain_deadline = time.monotonic() + _TEARDOWN_GRACE_S
    while _process_group_alive(pgid) and time.monotonic() < drain_deadline:
        _reap_group_zombies(pgid, skip=proc.pid)
        time.sleep(0.1)
    # last pass after the loop: a member killed on the final iteration becomes a zombie only once the
    # kernel has posted its status, which can land after the check that ended the loop. without this
    # the group drains but its process-table entry stays behind for the worker's whole lifetime.
    _reap_group_zombies(pgid, skip=proc.pid)
