"""Narrate an attempt's cold start in the run log and run status until its first training heartbeat.

The poll loop already sees every startup signal Flash has -- the provider's instance status and the
worker's setup-stage heartbeats -- but logged them only as bare transitions, then nothing for minutes
while the image pulled and the model loaded. ``StartupTracker`` turns those same signals into
``starting: ...`` lines with the time since launch, repeats a reminder while a stage is quiet, and
mirrors the stage into ``status.startup`` (``flash.runner.lifecycle.startup``).

Also owns the poll loop's cancellation probe, so a user cancel that tears the instance down is
reported as a cancel rather than as the instance disappearing.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from flash.providers._lifecycle.instances.poll import (
    SETUP_HEARTBEAT_STAGES,
    heartbeat_progress_ts,
    is_training_heartbeat,
)
from flash.runner.lifecycle.startup import (
    COLD_START_NOTICE,
    STARTUP_STAGES,
    TYPICAL_COLD_START,
    record_startup_stage,
)

# Seconds of silence within one startup stage before the log repeats where the start stands.
STARTUP_REMINDER_S = 120.0

_WORKER_STARTED_STAGES = frozenset({"boot", "sft_start", "rl_start"})
_MODEL_LOAD_STAGES = frozenset(
    {
        "model_prefetching",
        "model_prefetched",
        "checkpoint_prefetching",
        "sft_model_load",
        "custom_model_load",
        "rl_adapter_loading",
    }
)


def heartbeat_startup_stage(stage: object) -> str | None:
    """The startup stage a setup heartbeat proves, or None when it proves no particular one."""
    if not isinstance(stage, str) or stage not in SETUP_HEARTBEAT_STAGES:
        return None
    if stage in _MODEL_LOAD_STAGES or stage in _WORKER_STARTED_STAGES:
        # a worker that has just started goes straight to fetching and loading weights.
        return "loading_model"
    return "initializing_training"


def _elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _run_state(run_id: str) -> object:
    from flash.runner.lifecycle.status import get_status

    return get_status(run_id)


def run_cancel_requested(run_id: str) -> bool:
    """True once the run is cancelled; the cancel persists that state before tearing anything down."""
    try:
        return _run_state(run_id).state == "cancelled"
    except Exception:
        return False


def _attempt_already_training(run_id: str, attempt: int) -> bool:
    """A reattached poll must not narrate a cold start the attempt finished long ago."""
    try:
        heartbeat = _run_state(run_id).last_heartbeat
    except Exception:
        return False
    if not isinstance(heartbeat, dict) or heartbeat.get("attempt") != attempt:
        return False
    return is_training_heartbeat(heartbeat.get("stage"), heartbeat.get("step"))


class StartupTracker:
    """Per-poll narrator of one attempt's startup; inert once training has begun."""

    def __init__(
        self,
        *,
        run_id: str,
        instance_id: object,
        attempt: int,
        launch_ts: float,
        say: Callable[[str], None],
    ) -> None:
        self._run_id = run_id
        self._instance_id = instance_id
        self._attempt = attempt
        self._launch_ts = launch_ts
        self._say = say
        self.active = not _attempt_already_training(run_id, attempt)
        self.stage: str | None = None
        self._announced = False
        self._last_line_at = time.time()

    def _since_launch(self) -> str:
        return _elapsed(time.time() - self._launch_ts)

    def _enter(self, stage: str, extra: str | None = None) -> None:
        if stage == self.stage:
            return
        self.stage = stage
        suffix = f"; {extra}" if extra else ""
        self._say(
            f"starting: {STARTUP_STAGES[stage]}{suffix} ({self._since_launch()} since launch)"
        )
        self._last_line_at = time.time()
        record_startup_stage(self._run_id, stage, attempt=self._attempt, extra=extra)

    def observe_instance(self, status: str, *, running: bool, dead: bool) -> None:
        """Advance on a provider instance status: not yet running is boot, running is worker start."""
        if not self.active or dead:
            return
        if not self._announced:
            self._announced = True
            self._say(COLD_START_NOTICE)
        if running:
            if self.stage in (None, "booting_instance"):
                self._enter("starting_worker", f"instance {self._instance_id}")
        elif self.stage is None:
            self._enter("booting_instance", f"instance {self._instance_id} is {status}")

    def observe_heartbeat(self, hb_key: tuple | None) -> None:
        """Advance on this attempt's heartbeat; the first training heartbeat ends the startup."""
        if not self.active or not isinstance(hb_key, tuple) or len(hb_key) < 4:
            return
        _, fresh = heartbeat_progress_ts(hb_key, self._launch_ts, self._attempt)
        if not fresh:
            return
        stage, step = hb_key[0], hb_key[1]
        if is_training_heartbeat(stage, step):
            self.active = False
            step_text = f" step={step}" if step is not None else ""
            self._say(
                f"started: first training heartbeat (stage={stage}{step_text}) "
                f"{self._since_launch()} after launch"
            )
            record_startup_stage(self._run_id, None)
            return
        startup_stage = heartbeat_startup_stage(stage)
        if startup_stage is not None:
            self._enter(startup_stage, f"worker stage={stage}")

    def tick(self) -> None:
        """Keep the log moving through a long quiet stage with an explicit reminder."""
        if not self.active or self.stage is None:
            return
        if time.time() - self._last_line_at < STARTUP_REMINDER_S:
            return
        self._last_line_at = time.time()
        self._say(
            f"starting: still {STARTUP_STAGES[self.stage]} ({self._since_launch()} since launch; "
            f"cold start typically takes {TYPICAL_COLD_START})"
        )
