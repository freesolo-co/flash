"""Deadline and liveness bounds for the OPD verl child's unbounded waits.

Copied into the isolated verl child workdir as `flash_opd_rollout_watchdog.py`. Everything here
exists because verl waits forever in two places -- transfer-queue init and replay-buffer sampling --
and neither wait can be observed or interrupted from outside. A run that hits either one holds its
gpu at 0% utilisation until a human notices, so each is given a deadline and, where possible, a
positive liveness signal that distinguishes "slow" from "nobody is coming".

Split out of the child plugin to keep that module under the file-size limit. Like its siblings it
does not import the plugin back: both files land flat in the child workdir, and a cycle there would
fail at import time inside verl rather than here.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

_TQ_INIT_TIMEOUT_S = 600.0
_ROLLOUT_STALL_TIMEOUT_S = 1800.0
_ROLLOUT_LIVENESS_POLL_S = 15.0


def _describe_ray_resources() -> str:
    """summarise ray's cluster and free resources, or say why they could not be read.

    ray drops a resource key from available_resources() once it is fully allocated rather than reporting
    it as zero -- verified against ray 2.56.1, where consuming every cpu removes 'CPU' from the mapping
    entirely. that is exactly the exhaustion this probe exists to name, so a missing key reads as 0.0
    instead of None. cluster_resources() omits a resource the node does not have at all, which 0.0 also
    describes correctly.

    this is only ever called on the timeout path, so it must not raise: a probe failure here would
    replace the diagnosis it exists to produce. that includes the import itself -- ray lives only in
    the isolated verl interpreter, so anywhere else this must degrade to a note, not an ImportError.
    """
    try:
        import ray
    except ImportError:
        return "ray resources unreadable: ray is not importable here"

    try:
        total = ray.cluster_resources()
        free = ray.available_resources()
    except Exception as error:  # pragma: no cover - defensive, ray is up by this point
        return f"ray resources unreadable: {type(error).__name__}: {error}"
    return (
        f"cluster CPU={total.get('CPU', 0.0)} GPU={total.get('GPU', 0.0)}, "
        f"free CPU={free.get('CPU', 0.0)} GPU={free.get('GPU', 0.0)}"
    )


def _describe_stalled_thread(ident: int | None, depth: int = 4) -> str:
    """render the innermost frames of a thread that is still parked, or say why they are unavailable.

    which of tq.init's three waits is stuck is not decidable from ray's resource counts alone: a
    satisfied placement group rules out the first, but the controller ray.get and the get_config spin
    look identical from outside. the wedged thread is still alive at this point, so its own frames name
    the wait directly. like the resource probe, this runs only on the failure path and must not raise.
    """
    import sys
    import traceback

    try:
        frame = sys._current_frames().get(ident) if ident is not None else None
        if frame is None:
            return "stack unavailable"
        frames = traceback.extract_stack(frame)[-depth:]
        return " <- ".join(
            f"{f.name} ({f.filename.rsplit('/', 1)[-1]}:{f.lineno})" for f in reversed(frames)
        )
    except Exception as error:  # pragma: no cover - defensive, the thread is alive by construction
        return f"stack unreadable: {type(error).__name__}: {error}"


def _init_transfer_queue(init, conf: Any, timeout_s: float = _TQ_INIT_TIMEOUT_S) -> None:
    """run verl's transfer-queue init under a deadline, reporting the resource state on timeout.

    verl force-enables TransferQueue on the opd entry point (main_ppo_sync.main sets
    transfer_queue.enable = True with no opt-out), so this runs before a single gpu is touched. tq.init
    has three separate unbounded waits, none of which can be observed from outside:

      - get_placement_group blocks in ray.get(pg.ready()) until every 1-cpu storage bundle is placed,
      - process_zmq_server_info blocks in ray.get on the controller, itself a 1-cpu actor scheduled
        outside any placement group,
      - _init_from_existing spins `while conf is None` on get_config against a controller that may
        never publish one.

    all three are silent: tq's get_logger defaults to WARNING and TQ_LOGGING_LEVEL is unset here, so
    every progress line inside tq.init is suppressed. a wedge therefore presents as a run that reaches
    the ray banner and then stops, burning the full setup grace period at 0% gpu before the plane
    calls it a stall -- with no indication that transfer_queue was even involved.

    the thread is a daemon and is deliberately not joined after the timeout: it is parked inside an
    unbounded ray.get that no signal will interrupt, and the exception below fails the run anyway. that
    is also what makes its stack readable here, which is the only thing that tells the three waits apart.
    """
    failure: list[BaseException] = []

    def _run() -> None:
        try:
            init(conf)
        except BaseException as error:  # surfaced below, on the caller's thread
            failure.append(error)

    thread = threading.Thread(target=_run, name="flash-tq-init", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise RuntimeError(
            f"verl transfer_queue init did not finish within {timeout_s:.0f}s; "
            f"stalled at {_describe_stalled_thread(thread.ident)}; {_describe_ray_resources()}. "
            "tq.init waits without a timeout in three places -- an unplaceable storage placement group, "
            "a controller rpc that never returns, and a spin on a controller that never publishes its "
            "config -- so read the stalled frame: only the first is a capacity problem"
        )
    if failure:
        raise failure[0]


def _rollout_stall_timeout_s() -> float:
    """Seconds of no rollout progress before the step is failed. 0 or less disables the deadline."""
    raw = os.environ.get("FLASH_OPD_ROLLOUT_STALL_S", "")
    if not raw:
        return _ROLLOUT_STALL_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        return _ROLLOUT_STALL_TIMEOUT_S


def _dead_agent_loop_workers(workers) -> list[str]:
    """Names of agent-loop actors that are no longer alive.

    Liveness is asked of ray rather than inferred from elapsed time, because a rollout that is
    merely slow and one whose worker is gone look identical from the replay buffer. Only a dead
    actor proves nobody is left to clear the entry.

    A handle that answers is alive; one that raises RayActorError is not. Any other error means the
    probe itself failed -- ray unreachable, actor busy -- and must NOT be read as death, or a
    transient blip would kill a healthy run.
    """
    import ray
    from ray.exceptions import RayActorError

    dead: list[str] = []
    pending = {}
    for index, worker in enumerate(workers):
        try:
            pending[worker.__ray_ready__.remote()] = index
        except Exception:  # pragma: no cover - handle unusable, treated as unknown not dead
            continue
    if not pending:
        return dead
    ready, _unready = ray.wait(list(pending), num_returns=len(pending), timeout=0)
    for reference in ready:
        try:
            ray.get(reference)
        except RayActorError:
            dead.append(f"agent-loop worker {pending[reference]}")
        except Exception:  # pragma: no cover - probe failure is not proof of death
            continue
    return dead


def _bounded_replay_buffer_sample(
    sample,
    *,
    running_entries,
    dead_workers,
    stall_timeout_s: float,
    poll_interval_s: float = _ROLLOUT_LIVENESS_POLL_S,
    monotonic=None,
    sleep=None,
):
    """Run verl's replay-buffer sample on a worker thread, bounded by liveness and progress.

    verl's ReplayBuffer.sample is `while True: sleep(poll)` and returns only once no entry is still
    "running". Entries are marked running at dispatch and cleared by the agent-loop actor that owns
    them: "finished" on success, "failure" from its exception handler. An actor that dies without
    unwinding -- which os._exit does by design -- writes neither, so the entry stays running forever
    and the trainer polls a status no living process can ever publish. Observed as 8 allocations,
    8 wedges, 0 optimizer steps: GPU resident and 0% utilised until a human cancels.

    Two independent exits, because they catch different failures:

      - a dead agent-loop actor is decisive and immediate. this is the observed wedge, and waiting
        out a timeout to report it would just burn the GPU more slowly.
      - a stall deadline covers a hang where every actor is alive but no entry ever transitions.
        it is measured from the last observed progress, not from entry, so a long-but-advancing
        rollout cannot trip it.

    sample() itself is unbounded and uninterruptible, so it runs on a daemon thread that is
    deliberately not joined: it is parked in a sleep loop no signal will break, and the exception
    raised here fails the run anyway.
    """
    monotonic = monotonic or time.monotonic
    sleep = sleep or time.sleep
    result: list[Any] = []
    failure: list[BaseException] = []

    def _run() -> None:
        try:
            result.append(sample())
        except BaseException as error:  # surfaced below, on the caller's thread
            failure.append(error)

    thread = threading.Thread(target=_run, name="flash-opd-rollout-sample", daemon=True)
    thread.start()
    last_progress = monotonic()
    last_running = running_entries()
    while True:
        thread.join(poll_interval_s)
        if not thread.is_alive():
            break
        dead = dead_workers()
        if dead:
            raise RuntimeError(
                f"OPD rollout cannot complete: {', '.join(dead)} died while "
                f"{last_running} prompt(s) were still marked running. verl clears a prompt entry "
                "only from the worker that owns it, so those entries can never be cleared and the "
                "trainer would poll them forever. the worker's own stderr on this pod is the "
                "definitive record of why it died"
            )
        running = running_entries()
        if running != last_running:
            last_running = running
            last_progress = monotonic()
            continue
        elapsed = monotonic() - last_progress
        if stall_timeout_s > 0 and elapsed >= stall_timeout_s:
            raise RuntimeError(
                f"OPD rollout made no progress for {elapsed:.0f}s with {running} prompt(s) still "
                "marked running and every agent-loop worker alive. verl's replay buffer waits "
                "without a deadline, so this would otherwise hold the gpu indefinitely at 0% "
                f"utilisation; {_describe_ray_resources()}"
            )
        sleep(0)
    if failure:
        raise failure[0]
    return result[0]


def install_bounded_replay_buffer_sample(trainer) -> None:
    """Bound the trainer's replay-buffer wait using its live agent-loop actor handles.

    Wrapping the bound method on the instance rather than subclassing ReplayBuffer keeps verl's own
    sampling semantics untouched: this only decides how long to wait for it and how to fail, never
    which entries are returned.

    Call this after init_workers -- the actor handles it consults do not exist before then.
    """
    import functools

    buffer = trainer.replay_buffer
    if getattr(buffer, "_flash_bounded_sample", False):
        return
    original_sample = buffer.sample

    def running_entries() -> int:
        with buffer.lock:
            return sum(
                1
                for partition in buffer.partitions.values()
                for tag in partition.values()
                if tag.get("status") == "running"
            )

    def dead_workers() -> list[str]:
        manager = getattr(trainer, "async_rollout_manager", None)
        workers = getattr(manager, "agent_loop_workers", None) or []
        if not workers:
            return []
        try:
            return _dead_agent_loop_workers(workers)
        except Exception:
            # a probe that cannot run is not evidence of death; the stall deadline still bounds
            # the wait.
            return []

    @functools.wraps(original_sample)
    def bounded_sample(*args, **kwargs):
        return _bounded_replay_buffer_sample(
            lambda: original_sample(*args, **kwargs),
            running_entries=running_entries,
            dead_workers=dead_workers,
            stall_timeout_s=_rollout_stall_timeout_s(),
        )

    buffer.sample = bounded_sample
    buffer._flash_bounded_sample = True
