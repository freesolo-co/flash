"""What the verl child said before it died, and where ray wrote the rest of it.

A wedged or crashed child is diagnosed from two places: the tail of its own stdout, which
`ChildOutputTail` retains in a bounded ring buffer and `ChildTailStaleness` times, and ray's
per-session log directory, which holds the raylet and worker output the child never printed. Both
end up on a heartbeat payload, so both are size-capped and sanitized.

Split out of `flash.engine.worker.train.entry.backend_common` to keep that module under the file-size limit.
"""

from __future__ import annotations

import collections
import os
import re
import threading
import time
from collections.abc import Callable

from flash._internal.diagnostics import sanitize_diagnostic
from flash.engine.worker.verl.parent_work import ParentWorkGauge

# how many of the child's most recent output lines to retain for stall reporting. the child's last
# words before it wedges are the whole diagnostic, and a stall is usually preceded by a short burst
# (a ray warning, a placement-group notice, a partial traceback), so a small window suffices.
CHILD_TAIL_LINES = 60
# per-line cap when rendering the retained tail. verl prints resolved-config blocks thousands of
# characters wide; an unbounded tail would blow the heartbeat payload it has to travel inside.
_CHILD_TAIL_LINE_CHARS = 300
# how many retained lines ride along on a pre-first-step heartbeat. narrower than what is retained:
# this payload is uploaded every tick, so it stays small enough not to bloat the snapshot.
STALL_TAIL_LINES = 15
_RETRIABLE_VERL_CHILD_SIGNATURES = (
    "cudaErrorDevicesUnavailable",
    "CUDA-capable device(s) is/are busy or unavailable",
)


def _backend_common():
    """The parent module, imported lazily because it imports this one.

    Only used to resolve `open`: the bounded-read test replaces `backend_common.open` rather than
    the builtin, because a global replacement leaks into every later test through pytest's own file
    handling. Reading it back through the parent is what keeps that patch reaching this collector,
    and falling through to the builtin is what keeps a test that patches `builtins.open` working.
    """
    from flash.engine.worker.train.entry import backend_common

    return backend_common


class ChildOutputTail:
    """bounded ring buffer of a subprocess's most recent output lines.

    child stdout is absent from collected logs; only heartbeat markers survive. retain the tail so
    setup stalls can report the child's last words (ISSUES VERL-061).
    """

    def __init__(self, limit: int = CHILD_TAIL_LINES) -> None:
        self._lines: collections.deque[str] = collections.deque(maxlen=limit)
        self._written = 0
        self._last_line: str | None = None
        self._retriable_infra_signature: str | None = None
        self._cuda_oom_evidence: str | None = None
        self._cuda_oom_allocation_detail: str | None = None
        self._host_ram_kill_evidence: str | None = None

    def record(self, line: str) -> None:
        if self._retriable_infra_signature is None:
            self._retriable_infra_signature = next(
                (signature for signature in _RETRIABLE_VERL_CHILD_SIGNATURES if signature in line),
                None,
            )
        if self._cuda_oom_evidence is None:
            from flash.engine.worker.perf.lifecycle import cuda_oom_message_evidence

            self._cuda_oom_evidence = cuda_oom_message_evidence(line)
        if self._cuda_oom_allocation_detail is None:
            # latched independently of the classification token: a child can print the class on one
            # line and the allocation figures on the next, and the tail's ring buffer may evict
            # either before the run ends.
            from flash.engine.worker.perf.lifecycle import cuda_oom_allocation_detail

            self._cuda_oom_allocation_detail = cuda_oom_allocation_detail(line)
        if self._host_ram_kill_evidence is None:
            from flash.engine.worker.perf.lifecycle import host_ram_kill_evidence

            self._host_ram_kill_evidence = host_ram_kill_evidence(line)
        text = line.rstrip("\n")
        if text:
            # sanitize before the per-line cap, not after: truncating first could split a credential
            # across the cut and defeat full-value redaction. this is the worker side, where the
            # run's secret values are known, and every consumer of the retained tail (heartbeat
            # payload, streamed run log, persisted status) reads it from here.
            sanitized = sanitize_diagnostic(text, limit=_CHILD_TAIL_LINE_CHARS)
            self._lines.append(sanitized)
            if sanitized != self._last_line:
                self._written += 1
            self._last_line = sanitized

    @property
    def retriable_infra_signature(self) -> str | None:
        """the first stable retriable-infrastructure signature observed in child output."""
        return self._retriable_infra_signature

    @property
    def cuda_oom_evidence(self) -> str | None:
        """the first authoritative cuda oom message evidence observed in child output."""
        return self._cuda_oom_evidence

    @property
    def cuda_oom_allocation_detail(self) -> str | None:
        """how much the first observed cuda oom asked for, and what the card had."""
        return self._cuda_oom_allocation_detail

    @property
    def host_ram_kill_evidence(self) -> str | None:
        """the first authoritative ray host-RAM kill evidence observed in child output."""
        return self._host_ram_kill_evidence

    @property
    def written(self) -> int:
        """how many consecutive-line changes the child has produced, ever.

        monotonic and independent of the retention limit. consecutive duplicate lines remain in the
        retained tail but do not advance this counter, so a frozen warning loop is silence rather
        than progress while a genuinely new line resets the stall window.
        """
        return self._written

    def tail(self, limit: int | None = None) -> list[str]:
        """the retained lines, oldest first, optionally narrowed to the most recent ``limit``."""
        lines = list(self._lines)
        if limit is not None and limit >= 0:
            lines = lines[len(lines) - limit :] if limit < len(lines) else lines
        return lines


def raise_for_classified_verl_exit(return_code: int, tail: ChildOutputTail) -> None:
    """raise a classified failure when a nonzero verl child reported authoritative evidence."""
    if return_code == 0:
        return
    # host RAM is checked FIRST. a node that runs out of system memory kills the workers holding the
    # gpu, and the resulting cascade (thread-pool failures, bridge HTTP errors) prints AFTER the kill
    # line -- so whichever evidence is raised becomes the run's reported cause, and the ray kill is
    # the one that explains the rest. it is also permanent in the way a cuda oom is not: a
    # larger-VRAM class whose nodes ship the same system ram fails identically.
    host_ram_evidence = tail.host_ram_kill_evidence
    oom_evidence = tail.cuda_oom_evidence
    if host_ram_evidence is not None:
        # both signals present: the tail keeps only the FIRST of each, so it cannot say which
        # resource ended the run -- a worker killed for host pressure and a worker that hit a real
        # allocator OOM can both print here. report both rather than asserting the gpu was fine,
        # but still do not escalate VRAM: a bigger card cannot fix the half that is host RAM, and
        # the operator needs to see the ram kill to know a VRAM-only remedy is incomplete.
        #
        # the cuda evidence is DESCRIBED, never quoted. `is_cuda_oom` classifies this very string,
        # so echoing the matched token would make the message re-match and re-enable the VRAM
        # escalation this branch exists to prevent.
        both = (
            " a gpu allocator OOM was ALSO reported by some worker; the child output does not say "
            "which ended the run, so treat system RAM as scarce and re-check VRAM after fixing it."
            if oom_evidence is not None
            else " gpu memory was not the scarce resource, so retrying on a larger-VRAM class of "
            "the same node shape fails identically."
        )
        raise RuntimeError(
            f"verl subprocess exited with status {return_code} after ray killed its workers: "
            f"{host_ram_evidence}. this is a HOST RAM (system memory) kill, not a gpu OOM.{both} "
            "select a gpu class whose nodes carry more system RAM; the footprint is dominated by a "
            "worker-process floor, and the pool size is a divisor of prompts_per_step x group_size "
            "capped at 8, so lowering prompts_per_step alone usually leaves it unchanged."
        )
    if oom_evidence is not None:
        # the allocation figures are what turn "it OOMed" into a sizing decision: a request the
        # card could never have served points somewhere different from one that missed by a
        # gigabyte. the classification token alone left an operator with neither number.
        allocation = tail.cuda_oom_allocation_detail
        detail = f" ({allocation})" if allocation else ""
        raise RuntimeError(
            f"verl subprocess exited with status {return_code} after reporting "
            f"{oom_evidence}{detail}"
        )
    signature = tail.retriable_infra_signature
    if signature is None:
        return
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    raise RetriableInfraError(
        f"verl subprocess exited with status {return_code} after reporting {signature}"
    )


class ChildTailStaleness:
    """tracks how long a child has been silent, across the ticks that sample its tail.

    the tail alone cannot answer the question a stall actually poses. a child still loading shards
    and a child wedged forever both present a fully populated tail whose newest line is plausible,
    so the only thing separating them is whether the tail CHANGED between two dumps -- and a
    stateless report throws that comparison away, leaving it to be reconstructed by hand from
    consecutive heartbeats after the money is already spent (ISSUES VERL-067). holding the previous
    line count here turns that into a number the first dump already carries.
    """

    def __init__(self) -> None:
        self._written = -1
        self._since = 0

    def observe(self, written: int) -> int:
        """record this tick's line count; return consecutive ticks with no new output.

        0 means the child spoke since the last observation. n>0 means it has been silent for n
        ticks, which is the signal that separates a slow start from a wedge.
        """
        if written != self._written:
            self._written = written
            self._since = 0
        else:
            self._since += 1
        return self._since


def stall_tail_fields(
    step: int,
    tail: ChildOutputTail,
    limit: int = STALL_TAIL_LINES,
    staleness: ChildTailStaleness | None = None,
) -> dict[str, object]:
    """heartbeat fields carrying the child's last words, but only while it has made no progress.

    before the first step, the tail is the only collected setup-stall evidence. with ``staleness``,
    include silent ticks to distinguish a slow start from a wedge. return empty after progress or
    before any child output.
    """
    if step > 0:
        return {}
    recent = tail.tail(limit=limit)
    if not recent:
        # observed even with nothing to report, so a child that starts talking later is measured
        # from its first line rather than from whenever the payload happened to become non-empty.
        if staleness is not None:
            staleness.observe(tail.written)
        return {}
    fields: dict[str, object] = {"child_tail": recent}
    if staleness is not None:
        fields["child_tail_silent_ticks"] = staleness.observe(tail.written)
    return fields


def build_verl_line_handler(
    child_tail: ChildOutputTail,
    *,
    on_step: Callable[[int], None] | None,
    on_line: Callable[[str], None] | None,
    heartbeat: Callable[[], None] | None,
    step_pattern: str,
    heartbeat_interval_s: float,
    silence_watchdog: VerlChildSilenceWatchdog | None,
) -> Callable[[str], None]:
    """build the per-line callback the streaming verl subprocess runs on every child line.

    the watchdog observes the line before the caller's own callbacks: a callback that raises tears
    the process group down, and the line it raised on is still evidence the child was speaking.
    """
    step_re = re.compile(step_pattern)
    last_hb = 0.0

    def handle_line(line: str) -> None:
        nonlocal last_hb
        print(line, end="", flush=True)
        child_tail.record(line)
        if silence_watchdog is not None:
            silence_watchdog.observe_line(line)
        if on_line is not None:
            on_line(line)
        match = step_re.search(line)
        if match:
            step = int(match.group(1))
            if silence_watchdog is not None:
                silence_watchdog.observe_step(step)
            if on_step is not None:
                on_step(step)
        if heartbeat is not None:
            now = time.monotonic()
            if now - last_hb >= heartbeat_interval_s:
                heartbeat()
                last_hb = now

    return handle_line


VERL_CHILD_SILENCE_SECONDS = 1200.0
_VERL_CHILD_WATCH_TICK_SECONDS = 5.0


class VerlChildSilenceWatchdog:
    """tear down one live training child after sustained parent-and-child silence."""

    def __init__(
        self,
        tail: ChildOutputTail,
        *,
        baseline_step: int,
        parent_work: ParentWorkGauge | None = None,
        child_alive: Callable[[], bool] | None = None,
        teardown: Callable[[], None] | None = None,
    ) -> None:
        self._tail = tail
        self._baseline_step = int(baseline_step)
        self._parent_work = parent_work
        self._child_alive = child_alive
        self._teardown = teardown
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._armed = False
        self._last_activity = time.monotonic()
        self._tail_progress = tail.written
        self._parent_completed = self._parent_snapshot().completed
        self._failure: str | None = None
        self._tore_down = False

    def _parent_snapshot(self):
        if self._parent_work is None:
            from flash.engine.worker.verl.parent_work import ParentWorkSnapshot

            return ParentWorkSnapshot(0, 0)
        return self._parent_work.snapshot()

    def bind(self, *, child_alive: Callable[[], bool], teardown: Callable[[], None]) -> None:
        self._child_alive = child_alive
        self._teardown = teardown

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._child_alive is None or self._teardown is None:
            raise RuntimeError("verl child silence watchdog is not bound")
        self._thread = threading.Thread(
            target=self._watch,
            name="verl-child-silence-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_VERL_CHILD_WATCH_TICK_SECONDS + 1.0)

    def _arm(self) -> None:
        with self._lock:
            if not self._armed:
                self._armed = True
                self._last_activity = time.monotonic()

    def observe_line(self, line: str) -> None:
        if "Training Progress" in line:
            self._arm()
        self._record_activity()

    def observe_step(self, step: int) -> None:
        if int(step) > self._baseline_step:
            self._arm()
        self._record_activity()

    def _record_activity(self) -> None:
        snapshot = self._parent_snapshot()
        with self._lock:
            changed = self._tail.written != self._tail_progress
            completed = snapshot.completed != self._parent_completed
            if changed or completed or snapshot.depth > 0:
                self._last_activity = time.monotonic()
            self._tail_progress = self._tail.written
            self._parent_completed = snapshot.completed

    def check(self) -> None:
        self._record_activity()
        teardown = None
        with self._lock:
            alive = self._child_alive is not None and self._child_alive()
            expired = time.monotonic() - self._last_activity >= VERL_CHILD_SILENCE_SECONDS
            if self._armed and alive and expired and self._failure is None:
                self._failure = (
                    "verl_child_silence: no distinct child output or parent work completed for "
                    f"{VERL_CHILD_SILENCE_SECONDS:.0f}s while training"
                )
                if not self._tore_down:
                    self._tore_down = True
                    teardown = self._teardown
        if teardown is not None:
            teardown()

    def _watch(self) -> None:
        while not self._stop.wait(_VERL_CHILD_WATCH_TICK_SECONDS):
            self.check()
            if self._failure is not None:
                return

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError(self._failure)


# the ray logs worth keeping when a raylet dies. the driver's own stdout only ever shows the
# downstream symptom ("Failed to register worker to Raylet: ... End of file"); the reason the raylet
# went away is in these. deliberately a small allowlist -- a ray session dir also holds per-worker
# logs that can run to hundreds of files on a 128-core box.
RAY_FAILURE_LOGS = (
    "raylet.out",
    "raylet.err",
    "gcs_server.out",
    "gcs_server.err",
    "dashboard_agent.log",
    "dashboard.log",
)
# per file. enough to carry a stack and the lines before it, without turning an artifact upload into
# the reason a failing run takes even longer to report. this is the ONLY bound on the result: the
# sanitize pass below is given the same number so it can never truncate a tail we chose to keep.
RAY_LOG_TAIL_BYTES = 64 * 1024


def latest_ray_session_dir(
    root: str = "/tmp/ray", *, started_after: float | None = None
) -> str | None:
    """the most recent ray session directory, or None if THIS run never started one.

    ``started_after`` rejects sessions older than the caller's start. a retry reuses the pod workdir
    and /tmp survives it, so a run that fails BEFORE ray starts -- during dependency provisioning or
    model download -- still finds a previous run's session here. uploading that as the current
    attempt's evidence is worse than uploading nothing: it reads as a raylet failure that never
    happened, and sends the next diagnosis after a cause belonging to a different run.
    """
    try:
        names = os.listdir(root)
    except OSError:
        return None
    # stat once, here: a session directory can vanish between listing and stat, and doing it in a
    # `max(key=...)` would let that raise on an already-failing path.
    dated: list[tuple[float, str]] = []
    for name in names:
        if not name.startswith("session_"):
            continue
        path = os.path.join(root, name)
        try:
            if not os.path.isdir(path):
                continue
            # mtime, not the name's timestamp: the directory keeps being written while ray runs, so
            # a session that STARTED before this run but was still live during it is still ours.
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if started_after is not None and mtime < started_after:
            continue
        dated.append((mtime, path))
    if not dated:
        return None
    return max(dated)[1]


def collect_ray_failure_logs(
    *,
    root: str = "/tmp/ray",
    tail_bytes: int = RAY_LOG_TAIL_BYTES,
    started_after: float | None = None,
) -> str:
    """ray's own logs about why a raylet died, as one credential-safe artifact body ("" if none).

    when a raylet dies the driver prints only its own downstream failure, and ray's session dir --
    which holds the actual cause -- lives on the pod and goes away with it. that makes a raylet
    failure undiagnosable from uploaded artifacts and costs a paid gpu run per guess (VERL-115). one
    string rather than a directory of copies: the caller writes it exactly like the traceback
    artifact beside it, so a dying pod does one upload instead of six against the same bounded hf
    deadline allowance, and there is no staging directory whose only purpose is to be uploaded.
    """
    session = latest_ray_session_dir(root, started_after=started_after)
    if session is None:
        return ""
    logs_dir = os.path.join(session, "logs")
    sections: list[str] = []
    for name in RAY_FAILURE_LOGS:
        src = os.path.join(logs_dir, name)
        try:
            opener = getattr(_backend_common(), "open", open)
            with opener(src, "rb") as handle:
                # seek relative to the file's OWN end and cap the read: ray may still be writing
                # while this runs, and a getsize()-then-read() would consume from the old offset
                # through the new EOF -- unbounded, on a dying pod with a bounded upload deadline.
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                # the tail, not the head: a crash reason is at the end of the file.
                handle.seek(max(0, size - tail_bytes))
                payload = handle.read(tail_bytes)
        except OSError:
            continue
        # seeking to a byte offset can land mid-codepoint, and a decode error here would lose the
        # whole file for a cosmetic reason on the one path that exists to preserve evidence.
        text = payload.decode("utf-8", errors="replace")
        if size > tail_bytes:
            # never begin mid-line: a tail cut can split a credential and defeat prefix or full-value
            # redaction. drop the partial line, then sanitize multiline values line by line; both are
            # required for third-party ray logs.
            newline = text.find("\n")
            text = text[newline + 1 :] if newline != -1 else ""
            if not text:
                # a single line longer than the whole tail. dropping it is the only safe option, but
                # say so: an empty section would otherwise read as "ray logged nothing here".
                text = f"<omitted: final {tail_bytes} bytes are one unterminated line>"
        sections.append(
            f"===== {name} (last {tail_bytes} bytes) =====\n"
            f"{sanitize_diagnostic(text, limit=tail_bytes)}"
        )
    return "\n\n".join(sections)
