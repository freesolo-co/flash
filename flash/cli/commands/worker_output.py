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

from flash.cli.ui import render

if TYPE_CHECKING:
    from collections.abc import Callable

    from flash.client import ApiClient


def _artifact_attempt(name: str) -> int | None:
    """Return the bounded attempt encoded by a worker artifact name."""
    from flash.providers._lifecycle.poll import _attempt_int

    match = re.search(r"_attempt(\d+)\.txt$", name)
    return _attempt_int(int(match.group(1))) if match else None


def _worker_section_name(name: str, current_attempt: int | None) -> str:
    """Label an attempt-scoped artifact and identify superseded output."""
    attempt = _artifact_attempt(name)
    if attempt is None:
        return name
    if current_attempt is None:
        return f"{name} (attempt={attempt})"
    if attempt != current_attempt:
        return f"{name} (attempt={attempt}, previous attempt; current attempt={current_attempt})"
    return f"{name} (attempt={attempt}, current attempt)"


def _print_worker_output(
    client: ApiClient,
    run_id: str,
    *,
    printed_any: bool = False,
    current_attempt: Callable[[], int | None] | int | None = None,
) -> bool:
    """Print each worker artifact under a heading naming the attempt that produced it.

    `current_attempt` comes from the caller, as a value or as a callable resolved lazily, because
    the two callers know it at different times. The follow loop already reads a status every tick
    and passes what it saw; fetching another here would duplicate that request and consume a status
    the loop is pacing itself against. A snapshot read has none, so it passes a callable.

    The callable is invoked only after the artifacts are in hand, and only when there is a heading
    to label. That ordering is load-bearing twice over: a run with no artifacts costs no status
    request at all, and a retry starting mid-command is reflected rather than mislabelled, since
    the endpoint returns the highest UPLOADED attempt and resolving the live attempt first would
    pin a number the artifacts then contradict. None means it could not be established, and each
    section is labelled with its own attempt alone.
    """
    worker_output = client.get_worker_output(run_id) or {}
    if not worker_output:
        return printed_any
    if callable(current_attempt):
        current_attempt = current_attempt()
    for name, text in worker_output.items():
        if not text:
            continue
        # label provenance from the filename rather than letting the final section, which is merely
        # the last one appended, read as the live attempt's output.
        section_name = _worker_section_name(name, current_attempt)
        sep = "\n" if printed_any else ""
        if render.styled():
            print(f"{sep}{render.log_section(section_name)}")
        else:
            print(f"{sep}----- {section_name} -----")
        print(text, end="" if text.endswith("\n") else "\n")
        printed_any = True
    return printed_any
