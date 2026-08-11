"""Reading a worker heartbeat for the `flash runs status` panel.

Turning one heartbeat into rows a user can act on: how old it is, whether it belongs to the live
attempt, and -- when it is old -- which of the several very different things that can mean. The
panel cannot observe the worker directly, so every hint here is written to state what the age does
and does not prove, rather than to reassure.

The rendering primitives stay in `flash.cli.ui.render`; this module holds the interpretation. Split
out to keep `render.py` under the file-size limit, and imported back into `render` so
`render._heartbeat_pairs(...)` and friends keep resolving -- which is how the status panel and the
CLI tests reach them.
"""

from __future__ import annotations

import time

from flash.cli.ui.render import _dim


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


def heartbeat_is_current_attempt(obj: dict, heartbeat: dict) -> bool:
    """False only when the heartbeat provably belongs to a superseded retry attempt.

    Retries reuse the seed's heartbeat path, so a recovered run flips back to ``running`` for the
    replacement worker while ``last_heartbeat`` can still be the previous attempt's setup ping until
    the new worker publishes one. ``remote.attempt`` is the live attempt; ``last_heartbeat.attempt``
    is the one that produced the ping. When the live attempt is known, keep the reassurance only for a
    heartbeat whose attempt matches it. When it is unknown (e.g. a managed status payload that omits
    ``remote``), fall back to ``warmup_message``'s age gating rather than suppress, so warmup still
    reassures on planes that do not surface a live attempt.
    """
    # reuse the poller's attempt-identity contract (a bounded nonnegative int, never a bool, string,
    # or float) so the status display and stall detection agree on what a valid attempt is.
    from flash.providers._lifecycle.poll import _attempt_int

    remote = obj.get("remote")
    current_attempt = _attempt_int(remote.get("attempt")) if isinstance(remote, dict) else None
    if current_attempt is None:
        return True
    return _attempt_int(heartbeat.get("attempt")) == current_attempt


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
# only the stages the worker actually holds on the 900s upload throttle. opd_step is excluded: its
# post-update ping is force=True, so it re-commits at the 60s forced floor and an opd_step older
# than 900s means a long step, failed uploads, or a real stall -- not reporting lag.
_TRAINING_STEP_STAGES = frozenset({"rl_step", "sft_step"})

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

# the subset of the above that pulls weights or checkpoints over the network. only these can be
# explained by a cold per-datacenter cache, so only these may say so: telling a user that
# sft_configuring is "downloading tens of GB" sends them looking for a transfer that stage never
# performs, which is the same misdiagnosis this hint exists to prevent.
#
# base weights are already down by the time either model_load stage is emitted (both follow
# prefetch_model), so neither is the tens-of-GB transfer -- that is model_prefetching. sft_model_load
# is still here because its span calls _warmstart_adapter_path -> _download_adapter, a real network
# fetch; opd_model_load only reads the config with local_files_only, so it is not.
_DOWNLOADING_SETUP_STAGES = frozenset(
    {
        "model_prefetching",
        "checkpoint_prefetching",
        "sft_model_load",
        "rl_adapter_loading",
    }
)
# stages that fetch an ADAPTER rather than base weights: same cold-cache cause, but far smaller, so
# describing them as "tens of GB" overstates what the user should expect to be waiting on.
_ADAPTER_DOWNLOAD_STAGES = frozenset({"sft_model_load", "rl_adapter_loading"})


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
    if stage in _DOWNLOADING_SETUP_STAGES:
        if stage in _ADAPTER_DOWNLOAD_STAGES:
            blocking = "this stage fetches an adapter, which a cold cache serves slowly"
        else:
            blocking = "a cold weight cache downloads tens of GB with no ping"
        if heartbeat.get("dc"):
            blocking += ", and the datacenter above is where it landed"
    else:
        blocking = "this stage does no download, so a long one is unusual here"
    return (
        "this setup stage pings every ~4 min while the worker is alive, so this gap is longer than "
        f"throttling explains: it may be inside one long blocking call ({blocking}), its heartbeat "
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
    if heartbeat_age_seconds <= _STALE_STEP_AFTER_S:
        return None
    if str(heartbeat.get("stage") or "") not in _TRAINING_STEP_STAGES:
        return None
    # step 0 is the cold, still-running first step: no optimizer update has landed, so there is no
    # later hidden step for the reassurance to point at. reuse the shared step-gated predicate
    # rather than a bare presence check.
    from flash.providers._lifecycle.poll import is_training_heartbeat

    if not is_training_heartbeat(heartbeat.get("stage"), heartbeat.get("step")):
        return None
    # do not suggest `runs log -f`: it shows control-plane logs, and worker output arrives only after
    # termination (flash/cli/commands/__init__.py cmd_log) on a 3600s upload interval. use heartbeat age against
    # the 900s throttle, with w&b as the optional live signal.
    return (
        "the step above is the last one UPLOADED, not necessarily the one training is on; "
        "a throttled worker can hold it for many minutes while the trainer advances normally. "
        "uploads are held up to 15 min, so compare the age above against that "
        "(and your [wandb] run, if configured) "
        "before treating this as a stall"
    )


def _heartbeat_pairs(obj: dict) -> list[tuple[str, str]]:
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
    datacenter = hb.get("dc")
    if datacenter:
        pairs.append(("datacenter", str(datacenter)[:64]))
    heartbeat_age_seconds = _heartbeat_age_seconds(hb.get("ts"))
    running = str(obj.get("state") or "") == "running"
    stale_step = _stale_step_hint(
        hb,
        heartbeat_age_seconds,
        running=running,
        current_attempt=heartbeat_is_current_attempt(obj, hb),
    )
    # a setup stage carries no step, so the two hints address disjoint stages and cannot both fire.
    stale_setup = _stale_setup_hint(
        hb,
        heartbeat_age_seconds,
        running=running,
        current_attempt=heartbeat_is_current_attempt(obj, hb),
    )
    explained = stale_step or stale_setup
    # warmup is computed AFTER the hints so a stale setup stage can suppress it. the two windows
    # overlap (rl_initializing stays "fresh" for 1200s but goes silent-unexplained at 900s), and
    # showing both puts "setup is not billed; do not cancel" directly above "the instance may be
    # gone, check before cancelling". a panel that argues with itself is worse than either row.
    if running and not stale_setup:
        warmup = warmup_message(
            hb.get("stage"),
            heartbeat_age_seconds,
            heartbeat_is_current_attempt(obj, hb),
        )
        if warmup:
            pairs.append(("warmup", warmup))
    age = _humanize_age_seconds(heartbeat_age_seconds)
    if age:
        # the progress row already explains this silence, and does it more precisely. show one or
        # the other, never both, or the panel gives two readings of the same quiet.
        if running and not explained and heartbeat_age_seconds > _HB_QUIET_HINT_AFTER_S:
            age += _dim(f"  ({_QUIET_HEARTBEAT_HINT})")
        pairs.append(("heartbeat", age))
    if explained:
        pairs.append(("progress", explained))
    return pairs
