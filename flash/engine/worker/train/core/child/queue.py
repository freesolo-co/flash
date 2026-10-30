"""Bound initialization of the GRPO transfer queue."""

from collections.abc import Callable
from typing import Any

_TQ_INIT_TIMEOUT_S = 600.0


def _describe_ray_resources() -> str:
    """summarise ray's cluster and free resources, or say why they could not be read.

    ray drops a resource key from available_resources() once it is fully allocated rather than reporting
    it as zero -- verified against ray 2.56.1, where consuming every cpu removes 'CPU' from the mapping
    entirely. that is exactly the exhaustion this probe exists to name, so a missing key reads as 0.0
    instead of None. cluster_resources() omits a resource the node does not have at all, which 0.0 also
    describes correctly.

    this is only ever called on the timeout path, so it must not raise: a probe failure here would
    replace the diagnosis it exists to produce.
    """
    import ray

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


def _init_transfer_queue(
    init: Callable[[Any], Any], conf: Any, timeout_s: float = _TQ_INIT_TIMEOUT_S
) -> None:
    """run verl's transfer-queue init under a deadline, reporting the resource state on timeout.

    Flash enables TransferQueue for the synchronous GRPO entrypoint, before GPUs are initialized. tq.init
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
    import threading

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
