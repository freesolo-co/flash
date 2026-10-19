"""Printing the worker-uploaded text artifacts that `flash runs log` appends to a run log.

These sections are NOT part of the chronological log. `cmd_log` prints the control-plane log first
and appends these afterwards, so they always land at the tail no matter how old they are -- and the
plane returns the highest UPLOADED attempt, which during a retry is the attempt that just died, not
the one now running. Read without a label, the last thing on screen is a traceback from a worker
that has already been replaced.

So the job here is provenance: say which attempt produced each section, and say when that attempt is
not the live one. The old output is kept rather than hidden because it carries the failure that
explains why a retry exists at all.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from flash.cli.ui import heartbeat as heartbeat_ui
from flash.cli.ui import render
from flash.client import ClientError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flash.client import ApiClient


# stands in for the live attempt during the teardown window, when the plane has PROVEN there is no
# live worker (an explicitly null ``remote``). distinct from ``None``, which means UNKNOWN.
_NO_LIVE_WORKER = "no-live-worker"


def _artifact_attempt(name: str) -> int | None:
    """Return the bounded attempt encoded by a worker artifact name."""
    from flash.providers._lifecycle.instances.poll import _attempt_int

    match = re.search(r"_attempt(\d+)\.txt$", name)
    return _attempt_int(int(match.group(1))) if match else None


def _worker_section_name(name: str, current_attempt: int | str | None) -> str:
    """Label an attempt-scoped artifact and identify superseded output."""
    attempt = _artifact_attempt(name)
    if attempt is None:
        return name
    if current_attempt == _NO_LIVE_WORKER:
        return f"{name} (attempt={attempt}, worker torn down; no live attempt)"
    if current_attempt is None:
        return f"{name} (attempt={attempt})"
    if attempt != current_attempt:
        return f"{name} (attempt={attempt}, previous attempt; current attempt={current_attempt})"
    return f"{name} (attempt={attempt}, current attempt)"


_HEARTBEAT_LINE_RE = re.compile(r"(?m)^(HEARTBEAT )(?!\[superseded)")


def _mark_superseded_heartbeats(text: str, attempt: int | None, current_attempt: int | str) -> str:
    """Tag every HEARTBEAT line in a dead attempt's console dump.

    The section heading already says which attempt produced this text, but a heading is only read
    by a human looking at the whole screen. The monitoring idiom is
    ``runs log | grep HEARTBEAT | tail -1``, which sees ONE line and no heading -- and because the
    plane appends the highest uploaded attempt after the chronological log, that one line is the
    dead attempt's last heartbeat.

    That is not merely stale, it inverts the reading: the reporter's monitor showed
    ``step 0, 0 completions, device H200`` for twenty minutes while the run was in fact live on
    B200, because H200 was the card that had already OOMed and been torn down.

    Tagging the line itself means the provenance survives the pipe. A consumer that filters on the
    tag gets only live heartbeats; one that does not at least sees the attempt in the line it
    printed rather than silently trusting a dead worker's last words.

    ``attempt`` is ``None`` for the canonical ``console_<phase>.txt``, which encodes no attempt in
    its name. That is reported as ``attempt=unknown`` rather than guessed: the file is written at
    teardown, so on a retry whose terminal upload never ran it belongs to an OLDER attempt, and
    naming a number the filename does not carry would be a second wrong answer rather than a fix.
    """
    current = (
        "worker torn down"
        if current_attempt == _NO_LIVE_WORKER
        else f"current attempt={current_attempt}"
    )
    which = "unknown" if attempt is None else attempt
    return _HEARTBEAT_LINE_RE.sub(f"\\1[superseded attempt={which}; {current}] ", text)


def _worker_sections(client: ApiClient, run_id: str) -> dict[str, str]:
    """Fetch only worker artifacts with printable text."""
    return {name: text for name, text in (client.get_worker_output(run_id) or {}).items() if text}


def live_attempt_of(run: Mapping[str, object]) -> int | str | None:
    """The live attempt, ``_NO_LIVE_WORKER`` during teardown, or ``None`` when unknown.

    ``live_attempt`` answers ``None`` for two opposite situations: an explicitly null ``remote``,
    which is PROOF the attached heartbeat belongs to a worker that is already gone, and a run whose
    shape carries no attempt at all. Collapsing both to ``None`` disabled heartbeat tagging exactly
    during the retry window the tagging exists for -- the dead attempt's heartbeat went back to
    reaching ``grep HEARTBEAT | tail -1`` unmarked while replacement capacity was still being
    acquired. Keep the two apart so teardown can be labelled as what it is.

    Takes the status dict rather than fetching it, because ``--follow`` already holds one: the two
    log paths must derive this from the SAME rule, or a follow that ends mid-teardown prints the
    unmarked dead heartbeats the non-follow path tags.
    """
    attempt = heartbeat_ui.live_attempt(run)
    if attempt is None and run.get("remote", False) is None:
        return _NO_LIVE_WORKER
    return attempt


def _snapshot_live_attempt(client: ApiClient, run_id: str) -> int | str | None:
    """``live_attempt_of`` for the non-follow path, which has to fetch the status itself.

    A lookup failure stays ``None``: not knowing the live attempt must not make this claim there
    is no live worker.
    """
    try:
        run = client.get_run(run_id) or {}
    except ClientError:
        return None
    return live_attempt_of(run)


def _print_worker_output(
    sections: Mapping[str, str],
    *,
    printed_any: bool = False,
    current_attempt: int | str | None = None,
) -> bool:
    """Print worker artifacts under headings naming the attempt that produced them."""
    for name, text in sections.items():
        # label provenance from the filename rather than letting the final section, which is merely
        # the last one appended, read as the live attempt's output.
        section_name = _worker_section_name(name, current_attempt)
        # the heading is invisible to `grep HEARTBEAT | tail -1`, so a superseded attempt's
        # heartbeats carry their own provenance into the pipe. `_NO_LIVE_WORKER` tags every
        # attempt: during teardown the plane has PROVEN no worker is live, so every heartbeat on
        # screen is a dead one and none of them may reach a monitor unmarked.
        # an artifact whose name encodes NO attempt (the canonical `console_<phase>.txt`) is tagged
        # too. it is fetched alongside the scoped snapshot and appended last, so it is what a pipe
        # sees -- and being written at teardown it can belong to an older attempt on a retry.
        # unknown provenance is not evidence of liveness, so it does not earn an untagged pass.
        attempt = _artifact_attempt(name)
        supersedes = current_attempt is not None and (
            current_attempt == _NO_LIVE_WORKER or attempt != current_attempt
        )
        if supersedes:
            text = _mark_superseded_heartbeats(text, attempt, current_attempt)
        sep = "\n" if printed_any else ""
        if render.styled():
            print(f"{sep}{render.log_section(section_name)}")
        else:
            print(f"{sep}----- {section_name} -----")
        print(text, end="" if text.endswith("\n") else "\n")
        printed_any = True
    return printed_any
