"""run listing, status, log, cancellation, and checkpoint handlers."""

from __future__ import annotations

import json
import sys
import time
from typing import NamedTuple

from flash._internal.channel import CLI_NAME
from flash.cli.commands.ops.log_follow import (
    FollowInterrupted,
    _follow_transient_reason,
    _FollowRetry,
    _log_follow_metric_rows,
    _LogFollowSpinner,
    _sleep_with_spinner,
    _warn_follow_retry,
)
from flash.cli.commands.ops.worker_output import (
    _print_worker_output,
    _snapshot_live_attempt,
    _worker_sections,
    live_attempt_of,
)
from flash.cli.ui import cost as cost_ui
from flash.cli.ui import heartbeat as heartbeat_ui
from flash.cli.ui import render, tables
from flash.client import ApiClient, client_from_config
from flash.runner.lifecycle.state import TERMINAL_STATES

_CLI_DONE_STATES = TERMINAL_STATES | {"deployed"}
_OK_STATES = {"done", "dry_run", "deployed"}


def _log_follow_progress(status: dict | None, fallback_state: str) -> tuple[str, str]:
    """Return (authoritative state, compact progress) for the log-follow spinner."""
    status = status or {}
    state = str(status.get("state") or fallback_state or "unknown")
    parts = [state]
    heartbeat = status.get("last_heartbeat") if isinstance(status, dict) else None
    # retries rewind steps while state remains running, so surface the 0-based attempt identity.
    # `live_attempt` owns the provenance order (live `remote.attempt` first, heartbeat only when
    # `remote` is absent rather than cleared at teardown) and is shared with the worker-artifact
    # labelling below, so the spinner and the appended sections name the same current attempt.
    attempt = heartbeat_ui.live_attempt(status)
    if isinstance(heartbeat, dict):
        heartbeat_age_seconds = heartbeat_ui._heartbeat_age_seconds(heartbeat.get("ts"))
        # stage and step come from the heartbeat, attempt from `remote` below. during the relaunch
        # window those are two different attempts, so printing them side by side reads as progress
        # the replacement worker has not made: `stage=sft_step step=455 attempt=1` attributes the
        # superseded worker's 455 steps to an attempt that just started from zero -- the exact
        # rewind the attempt counter was added to explain. mark the heartbeat-sourced fields as the
        # previous attempt's instead of dropping them: the run really did reach that step, and
        # suppressing it entirely would read as no progress at all. a cleared `remote` is the same
        # relaunch window, and the ping is just as superseded there: the worker that produced it has
        # already been torn down. `heartbeat_is_current_attempt` answers True because it cannot
        # prove otherwise from the identity alone, so the qualifier has to come from the clear
        # itself or `step=455` reads as the replacement's progress. `heartbeat_is_superseded` is
        # exactly that pair of conditions, shared with the status panel so the two surfaces cannot
        # disagree about whether a run is between attempts.
        stale_heartbeat = heartbeat_ui.heartbeat_is_superseded(status, heartbeat)
        stage = heartbeat.get("stage")
        if stage:
            parts.append(f"stage={stage}")
            if state == "running":
                warmup = heartbeat_ui.warmup_message(
                    stage,
                    heartbeat_age_seconds,
                    not stale_heartbeat,
                )
                if warmup:
                    parts.append(warmup)
        step = heartbeat.get("step")
        if step is not None:
            parts.append(f"step={step}")
        # live heartbeat age so a long quiet phase reads as "alive, throttled" not "frozen".
        # minute granularity: the non-TTY follow path prints a line whenever this string changes,
        # so a seconds-precision age would emit one line per poll.
        if heartbeat_age_seconds is not None:
            mins = int(heartbeat_age_seconds // 60)
            parts.append(f"hb={mins}m" if mins else "hb=<1m")
        if stale_heartbeat and (stage or step is not None or heartbeat_age_seconds is not None):
            # one marker for the whole heartbeat-sourced group rather than a suffix on each field:
            # the staleness is a property of the ping, not of any one value it carried.
            #
            # it TRAILS the group, and `attempt=` is appended after it, so the split is positional:
            # everything before the marker came from the superseded ping, everything after is live.
            # covering `hb=` matters as much as covering the step -- that age is the old worker's
            # ping too, so a fresh `hb=<1m` printed outside the marker read as the replacement
            # worker being alive when nothing had been heard from it at all, the opposite of what
            # the age is there to say.
            parts.append("(prev attempt)")
    if attempt:
        parts.append(f"attempt={attempt}")
    # what this run has committed to spend so far. while it is live that is the submit-time quote,
    # since the settled charge is not written until the terminal transition -- following a run for an
    # hour and never seeing a cost is how a user loses track of what it is costing. it sits next to
    # realized_cost below so the quote and the settled charge read as one pair.
    #
    # a settled zero is a real answer and prints: `runs list` and `runs status` both show $0.0000
    # there, and dropping it only here made the same terminal run read as costed in one surface and
    # uncosted in another. what stays suppressed is the pre-settlement zero -- state carries no cost
    # yet, and `cost=$0.0000` on a queued run states a charge nobody has computed.
    amount, is_estimate = cost_ui.run_cost(status)
    settled = str(status.get("state") or "") in cost_ui.SETTLED_COST_STATES
    if amount or is_estimate or settled:
        parts.append(f"cost={'~' if is_estimate else ''}${amount:.4f}")
    realized = status.get("realized_cost_usd")
    if realized is not None:
        if isinstance(realized, (int, float)):
            parts.append(f"realized_cost=${realized:.4f}")
        else:
            parts.append(f"realized_cost={realized}")
    return state, " ".join(parts)


class _LogPollResult(NamedTuple):
    state: str
    printed_any: bool
    # `str` carries the teardown sentinel (`_NO_LIVE_WORKER`), which is not an attempt number but a
    # proof there is no live worker -- see `live_attempt_of`.
    live_attempt: int | str | None


def _poll_logs(client: ApiClient, run_id: str, interval: float) -> _LogPollResult:
    """Stream logs until terminal and return the final state, output, and attempt snapshot."""
    offset = 0
    printed_any = False
    attempt: int | None = None
    last_progress: str | None = None
    seen_metric_steps: set = set()
    spinner = _LogFollowSpinner(run_id)
    retry = _FollowRetry(run_id)
    try:
        while True:
            # A blip on either endpoint is the plane, not the run: the run keeps training and keeps
            # billing regardless. Retry both under one budget and keep the loop's place, so a 502
            # storm costs the stream a pause rather than the whole follow.
            try:
                page = client.get_logs(run_id, offset=offset)
                # The log file can lag worker heartbeat/status updates, so lifecycle/progress must
                # come from the run status endpoint. The log page's embedded state is only a
                # fallback for older servers or test doubles.
                status = client.get_run(run_id)
            # broad by design: `note_failure` re-raises anything it cannot prove transient, so
            # the classification lives in one place rather than three except clauses.
            except Exception as exc:
                delay, reason = retry.note_failure(exc)
                _warn_follow_retry(spinner, run_id, reason, retry)
                _sleep_with_spinner(delay, spinner, "reconnecting")
                continue
            retry.reset()
            if page["logs"]:
                spinner.clear()
                print(page["logs"], end="", flush=True)
                printed_any = True
            offset = page["offset"]
            attempt = live_attempt_of(status) if isinstance(status, dict) else attempt
            state, progress = _log_follow_progress(status, str(page.get("state") or ""))
            metric_rows = _log_follow_metric_rows(status, seen_metric_steps)
            if metric_rows:
                spinner.clear()
                for row in metric_rows:
                    print(row, file=sys.stderr, flush=True)
            if state in _CLI_DONE_STATES:
                spinner.clear()
                return _LogPollResult(state, printed_any, attempt)
            if not spinner.enabled and progress != last_progress:
                print(f"status: {progress}", file=sys.stderr, flush=True)
                last_progress = progress
            _sleep_with_spinner(interval, spinner, progress)
    finally:
        spinner.clear()


def _render_status(status: dict, *, force_json: bool = False, one_line: bool = False) -> str:
    """One rendering of a run status: themed panel on a TTY, indented JSON on the machine path.

    ``force_json`` is `--json`: the machine rendering even on a TTY, so a script's output does not
    depend on whether it happens to have one. ``one_line`` keeps each object on a single line for
    the `--follow` stream, which emits many of them: indented objects would run together into
    something no line-by-line JSON reader can parse.
    """
    if force_json and one_line:
        return json.dumps(status, separators=(",", ":"))
    return (
        render.run_status(status)
        if render.styled() and not force_json
        else json.dumps(status, indent=2)
    )


def _print_still_running_note(run_id: str, headline: str) -> None:
    """Say the stream ended and the paid run did not, then name both commands that act on it.

    Every way of losing the stream -- ctrl-c, a dead plane -- reports through here, because they
    share the failure that matters: the user reads "the run stopped", re-runs `flash train`, and
    pays for a duplicate. What differs is only the first line.
    """
    resume = f"resume with `{CLI_NAME} runs log {run_id} --follow`"
    stop = f"stop it with `{CLI_NAME} runs cancel {run_id}`"
    if render.styled():
        print(render.warn(headline), file=sys.stderr)
        print(render.arrow(resume), file=sys.stderr)
        print(render.arrow(stop), file=sys.stderr)
    else:
        print(f"warning: {headline}", file=sys.stderr)
        print(f"note: {resume}", file=sys.stderr)
        print(f"note: {stop}", file=sys.stderr)


def _print_detached_note(run_id: str) -> None:
    """Say what ctrl-c actually did: detached the stream, left the paid run running."""
    _print_still_running_note(
        run_id, f"detached from {run_id}; the run is still going and still billing"
    )


def _follow_run(client: ApiClient, run_id: str) -> int:
    """Poll logs until the run reaches a terminal state, then print the final status."""
    try:
        result = _poll_logs(client, run_id, interval=2.0)
    except KeyboardInterrupt:
        _print_detached_note(run_id)
        return 130
    except FollowInterrupted as exc:
        # The submit succeeded -- this run exists and is training on a GPU. Surfacing the transport
        # error alone reads as "the submit failed", and the user's next move is to submit again and
        # pay for both. Name the run instead, the same way ctrl-c does.
        _print_still_running_note(run_id, f"{exc}; {run_id} is still going and still billing")
        return 1
    try:
        print(_render_status(client.get_run(run_id)))
    except Exception as exc:
        # The run already reached a terminal state; `result.state` is that answer. Only the final
        # render was lost, so print what is known instead of turning a finished run into an error.
        if _follow_transient_reason(exc) is None:
            raise
        print(f"warning: could not fetch the final status ({exc})", file=sys.stderr)
        print(f"note: read it with `{CLI_NAME} runs status {run_id}`", file=sys.stderr)
    return 0 if result.state in _OK_STATES else 1


def _follow_status(
    client: ApiClient, run_id: str, interval: float = 2.0, *, force_json: bool = False
) -> int:
    """Poll run status until terminal, without replaying worker logs."""
    last_rendered: str | None = None
    retry = _FollowRetry(run_id)
    try:
        while True:
            try:
                status = client.get_run(run_id)
            # broad by design: see `_poll_logs` -- `note_failure` re-raises a real answer.
            except Exception as exc:
                delay, reason = retry.note_failure(exc)
                _warn_follow_retry(None, run_id, reason, retry)
                time.sleep(delay)
                continue
            retry.reset()
            rendered = _render_status(status, force_json=force_json, one_line=force_json)
            if rendered != last_rendered:
                print(rendered)
                last_rendered = rendered
            state = str(status.get("state") or "")
            if state in _CLI_DONE_STATES:
                return 0 if state in _OK_STATES else 1
            time.sleep(interval)
    except KeyboardInterrupt:
        _print_detached_note(run_id)
        return 130
    except FollowInterrupted as exc:
        _print_still_running_note(run_id, f"{exc}; {run_id} may still be going and still billing")
        return 1


def cmd_log(args) -> int:
    client = client_from_config()
    if getattr(args, "follow", False):
        try:
            result = _poll_logs(client, args.run_id, interval=2.0)
        except KeyboardInterrupt:
            _print_detached_note(args.run_id)
            return 130
        except FollowInterrupted as exc:
            _print_still_running_note(
                args.run_id, f"{exc}; {args.run_id} may still be going and still billing"
            )
            return 1
        sections = _worker_sections(client, args.run_id)
        _print_worker_output(
            sections,
            printed_any=result.printed_any,
            current_attempt=result.live_attempt,
        )
        return 0 if result.state in _OK_STATES else 1
    text = str(client.get_logs(args.run_id, offset=0).get("logs") or "")
    if text:
        print(text, end="" if text.endswith("\n") else "\n")
    sections = _worker_sections(client, args.run_id)
    attempt = _snapshot_live_attempt(client, args.run_id) if sections else None
    _print_worker_output(sections, printed_any=bool(text), current_attempt=attempt)
    return 0


def cmd_status(args) -> int:
    client = client_from_config()
    force_json = bool(getattr(args, "json", False))
    if getattr(args, "follow", False):
        return _follow_status(client, args.run_id, force_json=force_json)
    print(_render_status(client.get_run(args.run_id), force_json=force_json))
    return 0


def cmd_runs(args) -> int:
    runs = client_from_config().list_runs()
    if not runs:
        if render.styled():
            print(
                render.empty("runs", "0 runs", f"no runs yet — submit one with `{CLI_NAME} train`")
            )
        else:
            print("no runs yet")
        return 0
    if render.styled():
        print(tables.runs_table(runs))
        return 0
    print(f"{'RUN_ID':<32}  {'STATE':<11}  {'ALGO':<5}  {'COST($)':>8}  {'GPU':<22}  MODEL")
    for r in sorted(runs, key=lambda r: r.get("updated_at", 0), reverse=True):
        spec = r.get("spec") or {}
        model = spec.get("model", "")
        algorithm = str(spec.get("algorithm") or "-").upper()
        where = render.gpu_label(spec, r.get("remote") or {})
        amount, is_estimate = cost_ui.run_cost(r)
        cost = f"{'~' if is_estimate else ''}{amount:.4f}"
        print(
            f"{r['run_id']:<32}  {r['state']:<11}  {algorithm:<5}  {cost:>8}  {where:<22}  {model}"
        )
    return 0


def cmd_cancel(args) -> int:
    client = client_from_config()
    status = client.cancel_run(args.run_id)
    payload = {"run_id": args.run_id, "state": status["state"]}
    # A cancelled run is not necessarily worthless: every completed save interval already streamed
    # a deployable checkpoint, even though the run shows adapter_ref=null / cost=0. Surface the
    # surviving steps here so the run isn't discarded unseen. Best-effort: cancel never fails on it.
    checkpoints: list[dict] = []
    if payload["state"] == "cancelled":
        try:
            checkpoints = client.checkpoints(args.run_id)
        except Exception:
            checkpoints = []
    if render.styled():
        print(render.cancelled(payload))
    else:
        print(json.dumps(payload, indent=2))
    if checkpoints:
        # Best-effort hint (the cancel already succeeded), so never crash on a malformed checkpoint
        # shape: coerce steps defensively — a dict missing 'step' or carrying a non-int must not raise
        # a traceback here. Only surface the `step-N` deploy example when we recovered a real step.
        steps = []
        for c in checkpoints:
            try:
                steps.append(int(c["step"]))
            except (KeyError, TypeError, ValueError):
                continue
        # stderr in the plain path so the machine-readable stdout JSON stays untouched.
        out = sys.stdout if render.styled() else sys.stderr
        base = (
            f"{len(checkpoints)} deployable checkpoint(s) survive this cancel — list with "
            f"`{CLI_NAME} runs checkpoint {args.run_id}`"
        )
        msg = (
            f"{base}, deploy one with `{CLI_NAME} models deploy {args.run_id}/step-{max(steps)}`."
            if steps
            else f"{base}."
        )
        print(render.note(msg) if render.styled() else msg, file=out)
    return 0


def cmd_checkpoints(args) -> int:
    checkpoints = client_from_config().checkpoints(args.run_id)
    if not checkpoints:
        message = (
            f"no deployable checkpoints for {args.run_id} yet "
            "(RL/opd stream one per save interval; SFT-only runs have none)."
        )
        if render.styled():
            print(render.empty("checkpoints", "0 deployable", message))
        else:
            print(message, file=sys.stderr)
        return 0
    if render.styled():
        print(tables.checkpoints_table(args.run_id, checkpoints))
        return 0
    from flash.schema import format_checkpoint_ref

    for c in checkpoints:
        # single-space, unpadded columns so a plain `grep "step N"` / awk split works; the ref is
        # the canonical short form, paste-able into train.init_from_adapter.
        print(f"step {c['step']} {format_checkpoint_ref(args.run_id, c['step'])}")
    print(
        f"\ndeploy one with `{CLI_NAME} models deploy {args.run_id}/step-<STEP>`.",
        file=sys.stderr,
    )
    return 0
