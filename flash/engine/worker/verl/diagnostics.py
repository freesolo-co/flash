"""What the verl child said before it died, and where ray wrote the rest of it.

A wedged or crashed child is diagnosed from two places: the tail of its own stdout, which
`ChildOutputTail` retains in a bounded ring buffer and `ChildTailStaleness` times, and ray's
per-session log directory, which holds the raylet and worker output the child never printed. Both
end up on a heartbeat payload, so both are size-capped and sanitized.

Split out of `flash.engine.worker.backend_common` to keep that module under the file-size limit.
"""

from __future__ import annotations

import collections
import contextlib
import math
import os
import threading
import time
from collections.abc import Callable
from typing import Self

from flash._internal.diagnostics import sanitize_diagnostic

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
# bounded from ABOVE by the provider, not just from below by healthy silence. once a step is
# reported the poller uses its training limit, and `stall_kwargs` passes 1500s in production --
# NOT the 1200s default in `poll_until_complete`'s signature, which nothing calls with. the
# throttle window does not add to it either: `surface_heartbeat` returns stage None for a liveness
# ping and never advances the stall key, so the clock runs from the last REAL heartbeat and a
# wedge is torn down generically at 1500s. a threshold above that never gets to classify anything,
# which is the whole point of this watchdog.
#
# from below: one teacher request can spend 105s x 4 attempts + (2 + 4 + 8)s backoff = 434s while
# the child waits silently, and opd batches those requests serially. parent activity resets the
# counter after each completed interaction, which caps healthy teacher-bound silence at one batch
# (~14 ticks) rather than letting it accumulate across them.
#
# 20 min sits between: ~2.8x the 434s healthy worst case, and 300s of headroom under the provider.
# ticks are derived from the heartbeat cadence at runtime.
VERL_CHILD_SILENCE_TIMEOUT_S = 20.0 * 60.0
_RETRIABLE_VERL_CHILD_SIGNATURES = (
    "cudaErrorDevicesUnavailable",
    "CUDA-capable device(s) is/are busy or unavailable",
)
_TEARDOWN_GRACE_S = 10.0
# grace for a descendant holding stdout open after the trainer exits. observe child exit separately
# from eof because enginecore can retain the pipe; allow final flushing before group teardown.
_ORPHANED_PIPE_GRACE_S = 30.0


def _backend_common():
    """The parent module, imported lazily because it imports this one.

    Only used to resolve `open`: the bounded-read test replaces `backend_common.open` rather than
    the builtin, because a global replacement leaks into every later test through pytest's own file
    handling. Reading it back through the parent is what keeps that patch reaching this collector,
    and falling through to the builtin is what keeps a test that patches `builtins.open` working.
    """
    from flash.engine.worker import backend_common

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
            # compare against the retained (sanitized, capped) form rather than the raw line: two
            # reprints of one frozen status line can differ only inside a redacted value or past
            # the per-line cap, and treating those as new content would be exactly the repetition
            # this counter exists to see through.
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
    def host_ram_kill_evidence(self) -> str | None:
        """the first authoritative ray host-RAM kill evidence observed in child output."""
        return self._host_ram_kill_evidence

    @property
    def written(self) -> int:
        """how many non-empty lines of NEW content the child has produced, ever.

        monotonic and independent of the retention limit, which is what makes it usable as a
        staleness signal: a child that has gone silent cannot advance it even though its retained
        tail stays fully populated.

        a line identical to the one immediately before it does not advance this, and that exclusion
        is the whole difference between watching bytes and watching work. a wedged run is not
        silent: the observed wedge had a verl TransferQueue timer re-printing a FROZEN
        `Total success requests: 299` every five minutes, forever. counting each of those as output
        reset every staleness measure built on this, which is why the published silent-tick counter
        never climbed past single digits on runs that were already dead. a repeated line is a
        process still breathing and still not working, so only genuinely new output resets the clock.
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
        raise RuntimeError(
            f"verl subprocess exited with status {return_code} after reporting {oom_evidence}"
        )
    signature = tail.retriable_infra_signature
    if signature is None:
        return
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    raise RetriableInfraError(
        f"verl subprocess exited with status {return_code} after reporting {signature}"
    )


class _ChildExitWatchdog:
    """tear the group down when the direct child exits but a descendant holds the pipe open."""

    def __init__(self, proc, *, process_group_id: int, grace_s: float) -> None:
        self._proc = proc
        self._process_group_id = process_group_id
        self._grace_s = grace_s
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        # a child can exit with a full pipe behind it. count both lines read and lines still inside a
        # callback so a long checkpoint upload is working, not mistaken for an idle inherited pipe.
        self._lines_read = 0
        self._lines_in_flight = 0
        self.tore_down = False

    @contextlib.contextmanager
    def handling_line(self):
        """wrap the reader's whole per-line body, including callbacks."""
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
            self._thread.join(timeout=_TEARDOWN_GRACE_S)

    def _watch(self) -> None:
        while not self._done.wait(0.5):
            if self._proc.poll() is None:
                continue
            before = self._lines_read
            if self._done.wait(self._grace_s):
                return
            if self._lines_read != before or self._lines_in_flight:
                continue
            self.tore_down = True
            _backend_common().kill_process_group(
                self._proc, process_group_id=self._process_group_id
            )
            return


class ChildTailStaleness:
    """tracks how long a child has been silent, across the ticks that sample its tail.

    the tail alone cannot answer the question a stall actually poses. a child still loading shards
    and a child wedged forever both present a fully populated tail whose newest line is plausible,
    so the only thing separating them is whether the tail CHANGED between two dumps -- and a
    stateless report throws that comparison away, leaving it to be reconstructed by hand from
    consecutive heartbeats after the money is already spent (ISSUES VERL-067). holding the previous
    line count here turns that into a number the first dump already carries.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._written = -1
        self._since = 0
        self._clock = clock
        self._silent_since: float | None = None

    def observe(self, written: int, *, active: bool = False) -> int:
        """record this tick's line count; return consecutive ticks with no new output.

        0 means the child spoke or its parent completed work on its behalf since the last observation.
        n>0 means neither side advanced for n ticks, which separates slow work from a wedge.
        """
        if active or written != self._written:
            self._written = written
            self._since = 0
            self._silent_since = None
        else:
            if self._silent_since is None:
                # stamped on the FIRST silent observation, not on construction: the interval before
                # a child has ever spoken is not silence this watchdog is counting.
                self._silent_since = self._clock()
            self._since += 1
        return self._since

    @property
    def silent_ticks(self) -> int:
        """the consecutive silent ticks recorded by the most recent observation."""
        return self._since

    @property
    def silent_seconds(self) -> float:
        """wall-clock seconds since the first silent observation in the current run of silence.

        a tick COUNT is not a duration: each tick costs the loop's sleep plus whatever the work in
        between takes, and `gpu_diagnostics` alone permits two 8s `nvidia-smi` subprocess timeouts.
        forty nominal 30s ticks is 1200s but can really be 1886s -- past the provider's 1500s stall
        window, so the provider tears the run down first and the wedge is never classified.
        """
        return 0.0 if self._silent_since is None else self._clock() - self._silent_since


class VerlChildSilenceWatchdog:
    """tear down and classify a verl child that stays silent while a training step is active."""

    def __init__(
        self,
        tail: ChildOutputTail,
        *,
        tick_s: float,
        baseline_step: int = 0,
        parent_activity: Callable[[], int] | None = None,
        parent_busy: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tail = tail
        self._staleness = ChildTailStaleness(clock)
        self._parent_activity = parent_activity
        # "the parent is inside a unit of work right now", as opposed to "it finished another one".
        # a long enough single call makes the count alone useless, so both are consulted.
        self._parent_busy = parent_busy
        self._parent_activity_count: int | None = None
        self._silent_tick_limit = max(1, math.ceil(VERL_CHILD_SILENCE_TIMEOUT_S / tick_s))
        # the tick count is the NOMINAL schedule; the elapsed clock is what the provider measures.
        # whichever trips first fires, because a tick is not a fixed cost: the liveness loop sleeps
        # `tick_s` and THEN runs `gpu_diagnostics`, which permits two 8s `nvidia-smi` timeouts, so
        # 40 nominal 30s ticks can really be 1886s -- past the 1500s stall window, where the
        # provider tears the run down first and this watchdog never gets to classify the wedge.
        self._silence_seconds = min(VERL_CHILD_SILENCE_TIMEOUT_S, self._silent_tick_limit * tick_s)
        self._failure: RuntimeError | None = None
        self._teardown: Callable[[], None] | None = None
        self._is_running: Callable[[], bool] | None = None
        # the step this child STARTS from, supplied by the caller before launch rather than sampled.
        # it cannot be latched from the first observation: the liveness thread samples one tick in,
        # so a child that reports step 1 and then wedges would make 1 its own baseline and never
        # count as training again. a resumed run passes its resume_step, which is what keeps ray and
        # model loading exempt without giving a wedged child the same free pass.
        self._baseline_step = int(baseline_step)
        self._training_started = False
        self._lock = threading.Lock()

    def observe_line(self, line: str) -> None:
        """arm on verl's fit-loop banner, before its first generation or optimizer step."""
        if "Training Progress" not in line:
            return
        with self._lock:
            self._training_started = True

    def bind_process(self, *, teardown: Callable[[], None], is_running: Callable[[], bool]) -> None:
        """bind the child only after it exists, without holding this lock across teardown."""
        with self._lock:
            self._teardown = teardown
            self._is_running = is_running
            failed = self._failure is not None
        if failed:
            teardown()

    def observe(self, step: int) -> int:
        """sample child and optional parent progress, tearing down once active-step silence expires."""
        teardown = None
        with self._lock:
            if self._failure is not None:
                return self._staleness.silent_ticks
            parent_active = False
            if self._parent_activity is not None:
                try:
                    count = int(self._parent_activity())
                except Exception:
                    # losing an optional activity sample must not lose the child-silence sample too:
                    # repeated probe failures would otherwise disarm the watchdog indefinitely.
                    pass
                else:
                    parent_active = (
                        self._parent_activity_count is not None
                        and count != self._parent_activity_count
                    )
                    self._parent_activity_count = count
            if not parent_active and self._parent_busy is not None:
                # a counter only moves BETWEEN units of parent work. one batched scoring call can
                # outlast the whole silence budget on its own -- grpo coalesces up to 64 completions
                # into a single env call -- so progress alone cannot keep a healthy run alive.
                # "currently inside that call" is the missing edge, and it is a predicate, not a count.
                # suppressed like the count probe above: a failing optional probe must not disarm the
                # watchdog, and must not cost the child-silence sample either.
                with contextlib.suppress(Exception):
                    parent_active = bool(self._parent_busy())
            silent_ticks = self._staleness.observe(self._tail.written, active=parent_active)
            # verl creates its fit-loop progress bar before the first generation. that banner arms
            # the first in-flight step; completed-step progress remains the fallback for callers that
            # do not expose the banner and preserves the pre-fit setup exemption on fresh and resumed runs.
            training = self._training_started or step > self._baseline_step
            # bind_process runs immediately after popen. before that there is no paid child to kill;
            # afterwards this check keeps normal exit and teardown from being reclassified as silence.
            running = self._is_running is not None and self._is_running()
            # EITHER limit fires. the count alone runs past the provider's window whenever a tick
            # costs more than its nominal sleep; the clock alone would never fire if the loop's
            # sleep were shortened. the elapsed reading is what the provider is also measuring.
            silent_seconds = self._staleness.silent_seconds
            expired = (
                silent_ticks >= self._silent_tick_limit
                or silent_seconds >= VERL_CHILD_SILENCE_TIMEOUT_S
            )
            if not training or not running or not expired:
                return silent_ticks
            self._failure = RuntimeError(
                f"verl child produced no output for {max(silent_seconds, self._silence_seconds):.0f}s "
                "while training was running; the process group was torn down to release the gpu"
            )
            teardown = self._teardown
        if teardown is not None:
            teardown()
        return silent_ticks

    def heartbeat_fields(self, step: int) -> dict[str, object]:
        """observe this tick and return the field carrying the result.

        callers merge this into a `fields=` payload, so it doubles as the observation seam: a path
        that publishes the count cannot forget to advance it, and one that never publishes has no
        watchdog at all.
        """
        return {"child_tail_silent_ticks": self.observe(step)}

    def raise_if_failed(self) -> None:
        """raise the failure latched by the liveness thread on the owning training path."""
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise failure


def stall_tail_fields(
    step: int,
    tail: ChildOutputTail,
    limit: int = STALL_TAIL_LINES,
    silent_ticks: int | None = None,
) -> dict[str, object]:
    """heartbeat fields carrying silence on every step and child lines only before the first.

    the single tick count is cheap and remains useful during training, but the retained line list is
    a pre-first-step stall aid: uploading it after progress would bloat every training heartbeat.
    omit both claims before any child output.
    """
    recent = tail.tail(limit=limit)
    if not recent:
        return {}
    fields: dict[str, object] = {}
    if step <= 0:
        fields["child_tail"] = recent
    if silent_ticks is not None:
        fields["child_tail_silent_ticks"] = silent_ticks
    return fields


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
