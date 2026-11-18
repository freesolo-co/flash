"""The startup phase a training attempt is in between submission and its first training heartbeat.

A cold start is long and, from the outside, silent: the GPU has to be allocated, the instance booted,
the worker image pulled and the model loaded before the worker reports a training step. Without a
visible phase that silence reads as a stall, and a run cancelled at minute five never reaches minute
eleven. ``status.startup`` names the stage Flash last observed for the live attempt and is cleared by
the first training heartbeat, so ``flash runs status`` and ``GET /v1/runs/{id}`` say "starting" for
exactly as long as the run really is starting.
"""

from __future__ import annotations

import contextlib
import os
import time

# Observed time from submission to the first worker heartbeat on Spot H100s: across a 50-run research
# study no training job reached its first heartbeat in under ~630 s, and most took 630-760 s (Spot
# allocation, VM boot, worker image pull, model load). Shown to users as an expectation, not a limit:
# the stall/boot timeouts in ``providers._lifecycle.instances.poll`` are what actually end a start.
TYPICAL_COLD_START = "10-12 minutes"

COLD_START_NOTICE = (
    f"starting: cold start typically takes {TYPICAL_COLD_START} (instance boot, image pull, "
    "model load) before the first training heartbeat; this is normal, not a stall"
)

# Every stage here is one Flash observes directly: the allocator's queue, the provider's instance
# status, or a setup-stage worker heartbeat. Image pull and worker bootstrap are not separately
# observable (both happen inside a running instance before the worker's first heartbeat), so they
# share one stage rather than being guessed apart.
STARTUP_STAGES = {
    "waiting_for_gpu": "waiting for GPU capacity",
    "allocating_gpu": "allocating a GPU",
    "booting_instance": "booting instance",
    "starting_worker": "instance running; pulling the worker image and starting the worker",
    "loading_model": "loading model weights",
    "initializing_training": "preparing data and initializing training",
}

# Stages before a GPU is assigned. A cold-start expectation is only true once one is: while queued,
# "10-12 minutes" would read as the wait, which is unbounded and set by the queue instead.
PRE_GPU_STAGES = frozenset({"waiting_for_gpu", "allocating_gpu"})

# The run states in which a startup phase is meaningful; any other state hides a stale record.
_STARTING_STATES = frozenset({"queued", "provisioning", "running"})


def startup_detail(stage: str, extra: str | None = None) -> str:
    """The one-line ``starting: ...`` description of ``stage``."""
    detail = f"starting: {STARTUP_STAGES[stage]}"
    return f"{detail} ({extra})" if extra else detail


def record_startup_stage(
    run_id: str,
    stage: str | None,
    *,
    attempt: int | None = None,
    extra: str | None = None,
    queue: dict | None = None,
) -> None:
    """Persist the live attempt's startup stage; ``None`` clears it. Diagnostic only, never raises.

    ``queue`` is the Spot queue's view while ``waiting_for_gpu`` (reason, position, rough ETA).
    """
    if stage is not None and stage not in STARTUP_STAGES:
        raise ValueError(f"unknown startup stage: {stage!r}")
    # status persistence is diagnostic here; a failed write must not disturb supervision.
    with contextlib.suppress(Exception):
        _record(run_id, stage, attempt=attempt, extra=extra, queue=queue)


def _record(
    run_id: str,
    stage: str | None,
    *,
    attempt: int | None,
    extra: str | None,
    queue: dict | None = None,
) -> None:
    from flash.runner.lifecycle import state
    from flash.runner.lifecycle.status import get_status

    if not run_id or not os.path.exists(state.runs_file_path(run_id, ".json")):
        return
    with state._status_guard(run_id):
        status = get_status(run_id)
        if stage is None:
            if status.startup is None:
                return
            status.startup = None
        else:
            if status.state in state.TERMINAL_STATES:
                return
            previous = status.startup if isinstance(status.startup, dict) else {}
            if attempt is None:
                # a caller below the attempt loop (the spot price queue) does not know the attempt.
                attempt = previous.get("attempt")
            now = time.time()
            same_stage = previous.get("stage") == stage and previous.get("attempt") == attempt
            record = {
                "stage": stage,
                "detail": startup_detail(stage, extra),
                "since": previous.get("since", now) if same_stage else now,
                "attempt": attempt,
            }
            if stage in PRE_GPU_STAGES:
                if stage == "waiting_for_gpu" and isinstance(queue, dict):
                    record["queue"] = queue
            else:
                record["typical_cold_start"] = TYPICAL_COLD_START
            status.startup = record
        status.updated_at = time.time()
        state._save_status_unlocked(status)


def public_startup(run_state: object, startup: object) -> dict | None:
    """The startup record a status reader may see: only while the run can still be starting."""
    if run_state not in _STARTING_STATES or not isinstance(startup, dict):
        return None
    return startup
