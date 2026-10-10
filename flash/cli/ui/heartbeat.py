"""Reading a worker heartbeat for the `flash runs status` panel.

Turning one heartbeat into rows a user can act on: how old it is, whether it belongs to the live
attempt, and -- when it is old -- which of the several very different things that can mean. The
panel cannot observe the worker directly, so every hint here is written to state what the age does
and does not prove, rather than to reassure.

The rendering primitives stay in `flash.cli.ui.render`; this module owns heartbeat interpretation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable


def _heartbeat_age_seconds(value: object) -> float | None:
    """Return heartbeat age in seconds, or None for an unusable timestamp."""
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return max(0.0, time.time() - value)


def _humanize_age_seconds(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    return f"{seconds / 3600:.1f}h ago"


# heartbeat age past which the panel reminds that quiet is normal: worker uploads are throttled
# (240s quiet phases, up to 900s mid-training), so a frozen ts is usually not a dead worker.
_HB_QUIET_HINT_AFTER_S = 300.0
_WARMUP_HEARTBEAT_FRESH_FOR_S = 1200.0
_WARMUP_STAGES = frozenset({"rl_train_start", "rl_initializing"})


def live_attempt(obj: dict) -> int | None:
    """The attempt a status payload says is live, or None when it cannot be established.

    `remote.attempt` is the plane's live attempt. It is preferred over `last_heartbeat.attempt`,
    which is whichever worker produced the ping and may already be superseded. The heartbeat is a
    fallback only when `remote` is ABSENT: an explicitly null `remote` is the teardown window, where
    the attached ping belongs to a worker that is already gone, so falling back there would report a
    dead attempt as the live one.

    Shared so the status line, the log-follow spinner, and worker-artifact labelling cannot disagree
    about which attempt is current -- three surfaces that a user reads within one screen of each
    other, where a disagreement reads as a run that is on two attempts at once.
    """
    from flash.providers._lifecycle.instances.poll import _attempt_int

    remote = obj.get("remote")
    if isinstance(remote, dict):
        attempt = _attempt_int(remote.get("attempt"))
        if attempt is not None:
            return attempt
    elif "remote" in obj:
        # explicitly null: the teardown window. the ping is the dead worker's, so there is no answer.
        return None
    heartbeat = obj.get("last_heartbeat")
    return _attempt_int(heartbeat.get("attempt")) if isinstance(heartbeat, dict) else None


def heartbeat_is_current_attempt(obj: dict, heartbeat: dict) -> bool:
    """False only when the heartbeat provably belongs to a superseded retry attempt.

    Every attempt of a run shares one heartbeat path, so a recovered run flips back to ``running`` for the
    replacement worker while ``last_heartbeat`` can still be the previous attempt's setup ping until
    the new worker publishes one. ``remote.attempt`` is the live attempt; ``last_heartbeat.attempt``
    is the one that produced the ping. When the live attempt is known, keep the reassurance only for a
    heartbeat whose attempt matches it. When it is unknown (e.g. a managed status payload that omits
    ``remote``), fall back to ``warmup_message``'s age gating rather than suppress, so warmup still
    reassures on planes that do not surface a live attempt.
    """
    # reuse the poller's attempt-identity contract (a bounded nonnegative int, never a bool, string,
    # or float) so the status display and stall detection agree on what a valid attempt is.
    from flash.providers._lifecycle.instances.poll import _attempt_int

    remote = obj.get("remote")
    current_attempt = _attempt_int(remote.get("attempt")) if isinstance(remote, dict) else None
    if current_attempt is None:
        return True
    return _attempt_int(heartbeat.get("attempt")) == current_attempt


def heartbeat_is_superseded(obj: dict, heartbeat: dict) -> bool:
    """True when this ping's worker is known to be gone -- by attempt identity OR a cleared remote.

    ``heartbeat_is_current_attempt`` answers True for an explicitly null ``remote`` because it
    cannot prove otherwise from identity alone. But a null ``remote`` (as opposed to an absent one)
    IS the relaunch window: the plane cleared it at teardown, so the attached ping belongs to a
    worker that no longer exists. The log-follow spinner already draws that line the same way
    (`_log_follow_progress`), so the status panel uses one predicate rather than a second contract
    that disagrees with it about whether a run is between attempts.
    """
    remote_cleared = "remote" in obj and obj.get("remote") is None
    return remote_cleared or not heartbeat_is_current_attempt(obj, heartbeat)


def warmup_message(
    stage: object,
    heartbeat_age_seconds: float | None,
    from_current_attempt: bool = True,
) -> str | None:
    """Explain healthy RL setup stages only while the heartbeat is fresh and from the live attempt."""
    stage_name = str(stage)
    if stage_name not in _WARMUP_STAGES:
        return None
    if not from_current_attempt:
        return None
    if heartbeat_age_seconds is None:
        return None
    if heartbeat_age_seconds > _WARMUP_HEARTBEAT_FRESH_FOR_S:
        return None
    return (
        f"warming up (stage={stage_name}): initializing model, vLLM, and training kernels - "
        "typically several minutes, sometimes 15-20 min; setup is not billed; do not cancel"
    )


# worker stdout reaches the artifact repo on an HOURLY snapshot (_CONSOLE_UPLOAD_INTERVAL_S) plus a
# final flush at teardown, so `runs log` is not a live progress feed: for any run shorter than that
# it stays near-empty until the run ends, and then arrives all at once. pointing a user at it to
# resolve a silence is the advice that makes a healthy run look hung, so name the surfaces that do
# update -- the age on this panel, and w&b when the run configured it.
_QUIET_HEARTBEAT_HINT = (
    "heartbeat uploads are throttled; quiet is not dead - watch the age above "
    "(and your [wandb] run, if configured); worker stdout only lands hourly"
)
# a throttled training step is never guaranteed current: the worker holds mid-training commits for
# up to _HB_MIN_INTERVAL_S (900s), so from upload until the next commit the displayed step lags by
# an unknown amount. gate on the same age at which the panel already flags the quiet (300s) rather
# than on 900s -- the incident that motivated this reported 559s and 687s, squarely inside that
# window, where a 900s gate would stay silent and leave only the dead-end quiet hint.
_STALE_STEP_AFTER_S = _HB_QUIET_HINT_AFTER_S
# progress_age_s is the age of latest known progress at payload build time. combine it with upload age
# for a conservative current bound, and speak once that bound reaches the throttle without claiming
# either health or a stall: newer progress may be uncommitted, or the worker may be silent.
_UPLOAD_THROTTLE_S = 900.0
# only the stages the worker actually holds on the 900s upload throttle. opd_step emits ordinary
# post-step and liveness heartbeats, so its uploaded step can lag just like rl_step and sft_step.
_TRAINING_STEP_STAGES = frozenset({"opd_step", "rl_step", "sft_step"})

# the setup stages that hold a liveness thread on the 240s tight cadence (_HB_SETUP_LIVENESS_INTERVAL_S).
# these are the ones where age is genuinely informative: the worker pings every 240s while alive, so
# unlike a 900s-throttled training step, an age well past that cadence is NOT ordinary reporting lag.
_LIVENESS_SETUP_STAGES = frozenset(
    {
        "model_prefetching",
        "checkpoint_prefetching",
        "sft_model_load",
        "opd_model_load",
        "sft_data_loading",
        "rl_data_loading",
        "rl_adapter_loading",
        "sft_pretokenizing",
        "sft_configuring",
        "rl_configuring",
        "opd_configuring",
        "opd_filtering_prompts",
        "opd_prompt_scan",
        "opd_image_prep",
        "sft_initializing",
        "rl_initializing",
        "opd_initializing",
        # adapter export/upload holds a keepalive liveness wrap on the same 240s cadence, so a long
        # silence here is no more ordinary than at any other setup stage. these run AFTER training,
        # where the reassuring reading is the expensive one to get wrong: the work is done and only
        # the upload stands between the user and their adapter.
        "sft_finalizing",
        "rl_finalizing",
        "opd_finalizing",
    }
)
# a liveness-backed setup stage pings every 240s, so silence past ~3 missed ticks is the point where
# "still working" stops being the only explanation and a vanished instance becomes as likely.
_SETUP_SILENT_AFTER_S = 900.0

# what a stage BLOCKS on, so the hint can say what the wait plausibly is instead of guessing. a
# stage absent from every set below is left unnamed rather than described wrongly: EVERY stage here
# holds a liveness wrap because it can block for minutes (a venv install, a cold config read), so
# the honest fallback says the operation is unknown, never that no long call is expected.
#
# these stages pull base weights or a checkpoint over the network, so a cold per-datacenter cache
# volume explains a long silent stretch and the datacenter is worth citing.
_WEIGHT_DOWNLOAD_STAGES = frozenset({"model_prefetching", "checkpoint_prefetching"})
# these MAY fetch an adapter from the hub, but only conditionally: sft_model_load calls
# _warmstart_adapter_path, which returns immediately unless train.init_from_adapter is set, so most
# runs never transfer anything there. hedge the wording rather than assert a transfer that usually
# is not happening. this is the hub, not the weight volume, so no datacenter clause.
_ADAPTER_DOWNLOAD_STAGES = frozenset({"sft_model_load", "rl_adapter_loading"})
# these export and upload the trained adapter. a long blocking stretch is EXPECTED here, so calling
# it unusual would be actively wrong -- and worst of all after training has already finished, when
# the only thing between the user and their result is this upload.
_UPLOAD_STAGES = frozenset({"sft_finalizing", "rl_finalizing", "opd_finalizing"})


def _stale_setup_hint(
    heartbeat: dict,
    heartbeat_age_seconds: float | None,
    *,
    running: bool,
    current_attempt: bool = True,
) -> str | None:
    """Say what a long silence at a liveness-backed setup stage can and cannot mean.

    A setup stage holds a liveness thread on a 240s cadence, so a much older heartbeat is not the
    throttle. It can be one long blocking call, a worker whose best-effort heartbeat uploads keep
    failing while it works on (they roll the throttle slot back, so the age grows unbounded), or a
    vanished instance. The panel cannot tell those apart, and the failure mode is asymmetric: a user
    who reads a slow but healthy stage as a hang cancels a paid GPU. Name them and point at the
    surface that distinguishes them, rather than letting the quiet hint imply everything is fine.

    The blocking call is only described as a download for the stages that actually fetch weights,
    and the datacenter is only cited when it is on the panel to be read -- an explanation that
    names the wrong operation, or points at a row that was not rendered, is a fresh wrong turn
    rather than a resolved ambiguity.
    """
    if not running or heartbeat_age_seconds is None:
        return None
    if not current_attempt:
        return None
    if heartbeat_age_seconds <= _SETUP_SILENT_AFTER_S:
        return None
    stage = str(heartbeat.get("stage") or "")
    if stage not in _LIVENESS_SETUP_STAGES:
        return None
    if stage in _WEIGHT_DOWNLOAD_STAGES:
        blocking = "a cold weight cache downloads tens of GB with no ping"
        if heartbeat.get("dc"):
            blocking += ", and the datacenter above is where it landed"
    elif stage in _ADAPTER_DOWNLOAD_STAGES:
        blocking = "this stage fetches an adapter from the hub if the run warm-starts from one"
    elif stage in _UPLOAD_STAGES:
        blocking = "this stage exports and uploads the adapter, which is often slow"
    else:
        blocking = "this stage can block for minutes on a cold mount or a venv install"
    # worker code is content-addressed per submission, so a run submitted before this stage gained
    # its liveness wrap keeps emitting a single one-shot ping. asserting "pings every ~4 min" as
    # fact would then be false, and every reading built on it inherits that. state the cadence as
    # the expectation it is, and say so outright once a liveness ping has actually been observed.
    cadence = (
        "this setup stage pings every ~4 min while the worker is alive, so this gap is longer than "
        "throttling explains"
        if heartbeat.get("liveness")
        else "this setup stage is expected to ping every ~4 min while the worker is alive, so "
        "unless this run predates that, the gap is longer than throttling explains"
    )
    return (
        f"{cadence}: it may be inside one long blocking call ({blocking}), its heartbeat "
        "uploads may be failing while it keeps working, or the instance is gone. a vanished "
        "instance is reported as a retry or failure, so check whether the attempt advances before "
        "cancelling"
    )


def _stale_step_hint(
    heartbeat: dict,
    heartbeat_age_seconds: float | None,
    *,
    running: bool,
    current_attempt: bool = True,
) -> str | None:
    """Say a frozen training step is stale reporting, not a stalled trainer.

    A throttled worker can leave ``step`` pinned at its first training heartbeat for many minutes
    while the trainer is genuinely progressing. Through the CLI alone that is indistinguishable from
    a hung run, and the obvious reaction -- cancel and relaunch -- throws away a healthy paid GPU.
    Only fires for a *training* stage carrying a step, since a setup stage has no step to be stale.

    Supersedes the generic quiet hint at the same age: both explain the same silence, but that one
    sends you to ``runs log``, which reads the very heartbeats that went stale.
    """
    if not running or heartbeat_age_seconds is None:
        return None
    # a heartbeat from a superseded attempt describes a dead worker's step; calling that ordinary
    # throttled progress hides that the replacement has published nothing.
    if not current_attempt:
        return None
    if str(heartbeat.get("stage") or "") not in _TRAINING_STEP_STAGES:
        return None
    # step 0 is the cold, still-running first step: no optimizer update has landed, so there is no
    # later hidden step for the reassurance to point at. reuse the shared step-gated predicate
    # rather than a bare presence check.
    from flash.providers._lifecycle.instances.poll import is_training_heartbeat

    if not is_training_heartbeat(heartbeat.get("stage"), heartbeat.get("step")):
        return None
    progress_age_s = heartbeat.get("progress_age_s")
    progress_age: float | None = None
    if not isinstance(progress_age_s, bool):
        try:
            candidate = float(progress_age_s)
        except (OverflowError, TypeError, ValueError):
            pass
        else:
            if candidate >= 0 and math.isfinite(candidate):
                progress_age = candidate
    if progress_age is not None:
        progress_age_bound_s = heartbeat_age_seconds + progress_age
        if progress_age_bound_s >= _UPLOAD_THROTTLE_S:
            return (
                "the step above is the last one UPLOADED; "
                f"the last known progress can be as old as {progress_age_bound_s:.1f}s. upload "
                "throttling no longer explains the gap, but newer progress may be uncommitted and "
                "a long step may still be running; this signal does not show recent progress"
            )
        # below the 900s hold, throttling still explains the visible lag. fall through to the exact
        # pre-existing reading so new workers never lose guidance in the incident's 300-900s band.
    if heartbeat_age_seconds <= _STALE_STEP_AFTER_S:
        return None
    # old workers do not publish progress_age_s. keep their existing throttle-only reading exactly so
    # content-addressed in-flight runs do not change interpretation when the CLI upgrades. invalid
    # progress ages use the same fallback because they provide no trustworthy worker-side reading.
    return (
        "the step above is the last one UPLOADED, not necessarily the one training is on; "
        "a throttled worker can hold it for many minutes while the trainer advances normally. "
        "uploads are held up to 15 min, so compare the age above against that "
        "(and your [wandb] run, if configured) "
        "before treating this as a stall"
    )


def _superseded_hint(
    heartbeat_age_seconds: float | None,
    *,
    running: bool,
    current_attempt: bool,
) -> str | None:
    """Explain a quiet heartbeat that belongs to an attempt the plane has already torn down.

    Without this the panel contradicts itself: the datacenter row says ``previous attempt`` while the
    generic quiet hint beside it says ``quiet is not dead`` about that same ping. Both readings
    cannot be true, and the reassuring one wins by being longer, so a run whose worker is provably
    gone reads as healthy.

    Gated at the same age as the hint it replaces, because below that age nothing wrong is printed:
    the label alone is accurate, and a hint about silence with no silence to explain is noise.
    """
    if not running or heartbeat_age_seconds is None:
        return None
    if current_attempt:
        return None
    if heartbeat_age_seconds <= _HB_QUIET_HINT_AFTER_S:
        return None
    return (
        "this ping is from an attempt that is already gone; the replacement has not published one "
        "yet, so the age above measures the OLD worker's last ping, not the new one's silence. "
        "watch for the stage or step to change rather than reading this as a stall"
    )


def _finite_positive(value: object) -> float | None:
    """return a finite positive number from an untrusted heartbeat field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if number > 0 and math.isfinite(number) else None


def _humanize_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _humanize_step_duration(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 600:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def _step_timing_pairs(
    heartbeat: dict,
    *,
    running: bool,
    current_attempt: bool,
) -> list[tuple[str, str]]:
    """render measured RL pace only for the live running attempt."""
    if not running or not current_attempt or heartbeat.get("stage") != "rl_step":
        return []
    step_duration_s = _finite_positive(heartbeat.get("step_duration_s"))
    if step_duration_s is None:
        return []

    projected_remaining_s = _finite_positive(heartbeat.get("projected_remaining_s"))
    pairs = [("median pace", f"{_humanize_step_duration(step_duration_s)}/step")]
    if projected_remaining_s is not None:
        pairs.append(("mean ETA", f"~{_humanize_duration(projected_remaining_s)} left"))
    if heartbeat.get("wall_deadline_at_risk") is True and projected_remaining_s is not None:
        pairs.append(
            (
                "warning",
                "mean-based remaining-time projection exceeds the run's wall time left",
            )
        )
    return pairs


def _heartbeat_pairs(
    obj: dict, *, format_hint: Callable[[str], str] = str
) -> list[tuple[str, str]]:
    """Worker heartbeat rows for the status panel: stage, step, age, and a quiet-is-normal hint."""
    hb = obj.get("last_heartbeat")
    if not isinstance(hb, dict) or not hb.get("stage"):
        return []
    worker = str(hb["stage"])
    step = hb.get("step")
    if step is not None:
        worker += f" · step {step}"
    if hb.get("liveness"):
        worker += " · alive ping"
    pairs = [("worker", worker)]
    # the worker stamps the datacenter it landed in on every heartbeat, but nothing showed it. base
    # weights come from a per-datacenter cache volume and the allocator does not pin the region, so
    # an identical config relaunched minutes later can land somewhere cold and spend a long silent
    # stretch downloading. without the region on the panel that is an unexplainable freeze; with it,
    # two runs of the same config are comparable.
    #
    # a retry reuses the heartbeat path, so until the replacement worker publishes its first ping
    # this `dc` belongs to the attempt that already died. the replacement may still be provisioning
    # or may land in a different region with a different cache state -- exactly the comparison this
    # row exists to support -- so say whose region it is rather than implying it is the live one.
    heartbeat_age_seconds = _heartbeat_age_seconds(hb.get("ts"))
    running = str(obj.get("state") or "") == "running"
    # a cleared `remote` is the same relaunch window as a mismatched attempt: the plane nulls it at
    # teardown, so the attached ping is just as superseded even though the identity cannot prove it.
    from_current_attempt = not heartbeat_is_superseded(obj, hb)
    datacenter = hb.get("dc")
    if datacenter:
        label = "datacenter" if from_current_attempt else "datacenter (previous attempt)"
        pairs.append((label, str(datacenter)[:64]))
    pairs.extend(
        _step_timing_pairs(
            hb,
            running=running,
            current_attempt=from_current_attempt,
        )
    )
    stale_step = _stale_step_hint(
        hb,
        heartbeat_age_seconds,
        running=running,
        current_attempt=from_current_attempt,
    )
    # a setup stage carries no step, so the two hints address disjoint stages and cannot both fire.
    stale_setup = _stale_setup_hint(
        hb,
        heartbeat_age_seconds,
        running=running,
        current_attempt=from_current_attempt,
    )
    # last in the chain: both hints above already return None for a superseded ping, and each says
    # something more specific than "the worker is gone" when it does fire.
    explained = (
        stale_step
        or stale_setup
        or _superseded_hint(
            heartbeat_age_seconds,
            running=running,
            current_attempt=from_current_attempt,
        )
    )
    # warmup is computed AFTER the hints so a stale setup stage can suppress it. the two windows
    # overlap (rl_initializing stays "fresh" for 1200s but goes silent-unexplained at 900s), and
    # showing both puts "setup is not billed; do not cancel" directly above "the instance may be
    # gone, check before cancelling". a panel that argues with itself is worse than either row.
    if running and not stale_setup:
        warmup = warmup_message(
            hb.get("stage"),
            heartbeat_age_seconds,
            from_current_attempt,
        )
        if warmup:
            pairs.append(("warmup", warmup))
    age = _humanize_age_seconds(heartbeat_age_seconds)
    if age:
        # the progress row already explains this silence, and does it more precisely. show one or
        # the other, never both, or the panel gives two readings of the same quiet.
        if running and not explained and heartbeat_age_seconds > _HB_QUIET_HINT_AFTER_S:
            age += format_hint(f"  ({_QUIET_HEARTBEAT_HINT})")
        pairs.append(("heartbeat", age))
    if explained:
        pairs.append(("progress", explained))
    return pairs
